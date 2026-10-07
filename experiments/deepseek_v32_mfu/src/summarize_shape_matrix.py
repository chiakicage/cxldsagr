"""Derive descriptive statistics from the accepted DeepSeek MFU shape matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

METHODS = ("hbm", "echo", "serial_sparse", "dense_prefetch")
HISTORIES = (4096, 16384, 65536)
EXTENSIONS = (128, 256, 512, 1024)
SHAPES = {(h, a) for h in HISTORIES for a in EXTENSIONS}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return int(row["prefix_tokens"]), int(row["extend_tokens"]), row["method"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptive(values):
    require(len(values) >= 2 and all(math.isfinite(x) and x > 0 for x in values), "Invalid samples")
    mean = statistics.mean(values)
    deviation = statistics.stdev(values)
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": mean,
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": deviation,
        "cv_percent": 100 * deviation / mean,
        "range_over_median_percent": 100 * (max(values) - min(values)) / statistics.median(values),
        "first_sample_is_maximum": values[0] == max(values),
    }


def derive(source):
    source = Path(source).resolve(strict=True)
    manifest = read_json(source / "publication_manifest.json")
    inputs = {
        str(source / "publication_manifest.json"): digest(source / "publication_manifest.json")
    }

    def verified(name, reader=read_json):
        path = source / name
        actual = digest(path)
        require(manifest["files_sha256"].get(name) == actual, f"Published input changed: {name}")
        inputs[str(path)] = actual
        return reader(path)

    metadata = verified("summary.json")
    accepted = verified("audit/0_audit.json")
    require(
        metadata["complete_requested_matrix"]
        and accepted["passed"]
        and accepted["complete_matrix"],
        "Matrix acceptance is incomplete",
    )
    timing = verified("timing.csv", read_csv)
    samples = verified("timing_samples.csv", read_csv)
    cache = verified("cache_metrics_samples.csv", read_csv)
    windows = verified("timeline_windows.csv", read_csv)
    require(
        len(timing) == 48 and len(samples) == 384 and len(cache) == 1152 and len(windows) == 144,
        "Unexpected matrix coverage",
    )
    timing_by_key = {key(row): row for row in timing}
    require(
        set(timing_by_key) == {(h, a, m) for h, a in SHAPES for m in METHODS},
        "Missing or duplicate timing keys",
    )
    sample_groups = defaultdict(list)
    for row in samples:
        sample_groups[(*key(row), row["phase"])].append(row)
    points, point_map = [], {}
    for h, a in sorted(SHAPES):
        for method in METHODS:
            for phase, count in (("prefill", 3), ("extend", 5)):
                selected = sorted(
                    sample_groups[(h, a, method, phase)], key=lambda row: int(row["sample"])
                )
                require(
                    [int(row["sample"]) for row in selected] == list(range(count)),
                    "Sample coverage differs",
                )
                reference = timing_by_key[(h, a, method)]
                require(
                    all(row["bench_run_id"] == reference["bench_run_id"] for row in selected),
                    "Sample run binding differs",
                )
                values = [float(row["wall_ms"]) for row in selected]
                stats = descriptive(values)
                require(
                    stats["median_ms"] == float(reference[phase + "_median_ms"]),
                    "Published median differs from samples",
                )
                row = {
                    "prefix_tokens": h,
                    "extend_tokens": a,
                    "method": method,
                    "phase": phase,
                    **stats,
                    "samples_ms": json.dumps(values),
                    "bench_run_id": reference["bench_run_id"],
                }
                points.append(row)
                point_map[(h, a, method, phase)] = row
    comparisons = []
    winners = Counter()
    offload_winners = Counter()
    for h, a in sorted(SHAPES):
        medians = {m: point_map[(h, a, m, "extend")]["median_ms"] for m in METHODS}
        ordered = sorted(METHODS, key=medians.get)
        offload_order = sorted(METHODS[1:], key=medians.get)
        winners[ordered[0]] += 1
        offload_winners[offload_order[0]] += 1
        for method in METHODS:
            comparisons.append(
                {
                    "prefix_tokens": h,
                    "extend_tokens": a,
                    "method": method,
                    "median_ms": medians[method],
                    "relative_hbm": medians[method] / medians["hbm"],
                    "speedup_vs_hbm": medians["hbm"] / medians[method],
                    "relative_serial": medians[method] / medians["serial_sparse"],
                    "rank": ordered.index(method) + 1,
                    "offload_winner": offload_order[0],
                }
            )
    aggregates = []
    for phase in ("prefill", "extend"):
        for method in METHODS:
            selected = [point_map[(h, a, method, phase)] for h, a in sorted(SHAPES)]
            ratios = [
                row["median_ms"]
                / point_map[(row["prefix_tokens"], row["extend_tokens"], "hbm", phase)]["median_ms"]
                for row in selected
            ]
            cvs = [row["cv_percent"] for row in selected]
            aggregates.append(
                {
                    "phase": phase,
                    "method": method,
                    "geomean_relative_hbm": statistics.geometric_mean(ratios),
                    "min_relative_hbm": min(ratios),
                    "max_relative_hbm": max(ratios),
                    "median_case_cv_percent": statistics.median(cvs),
                    "max_case_cv_percent": max(cvs),
                    "first_sample_maximum_cases": sum(
                        row["first_sample_is_maximum"] for row in selected
                    ),
                }
            )
    scaling = []
    for method in METHODS:
        for h in HISTORIES:
            ratio = (
                point_map[(h, 1024, method, "extend")]["median_ms"]
                / point_map[(h, 128, method, "extend")]["median_ms"]
            )
            scaling.append(
                {
                    "axis": "A",
                    "fixed_tokens": h,
                    "method": method,
                    "from_tokens": 128,
                    "to_tokens": 1024,
                    "latency_ratio": ratio,
                    "tokens_per_ms_ratio": 8 / ratio,
                }
            )
        for a in EXTENSIONS:
            ratio = (
                point_map[(65536, a, method, "extend")]["median_ms"]
                / point_map[(4096, a, method, "extend")]["median_ms"]
            )
            scaling.append(
                {
                    "axis": "H",
                    "fixed_tokens": a,
                    "method": method,
                    "from_tokens": 4096,
                    "to_tokens": 65536,
                    "latency_ratio": ratio,
                    "tokens_per_ms_ratio": 1 / ratio,
                }
            )
    cache_groups = defaultdict(list)
    for row in cache:
        group_key = (*key(row), row["phase"])
        sample = int(row["sample"])
        reference = point_map[group_key]
        require(
            row["bench_run_id"] == reference["bench_run_id"]
            and float(row["wall_ms"]) == json.loads(reference["samples_ms"])[sample],
            "Cache sample binding differs",
        )
        cache_groups[(*group_key, sample)].append(row)
    traffic, totals = [], {}
    count_fields = (
        "host_to_device_bytes",
        "device_to_host_bytes",
        "prefetched_records",
        "recalled_records",
        "prefetch_capacity_failures",
        "evicted_records",
        "capacity_splits",
    )
    for h, a in sorted(SHAPES):
        for method in METHODS:
            records = []
            for phase, count in (("prefill", 3), ("extend", 5)):
                for sample in range(count):
                    layers = cache_groups[(h, a, method, phase, sample)]
                    require(
                        sorted(int(row["layer"]) for row in layers) == [0, 1, 2],
                        "Cache layer coverage differs",
                    )
                    for row in layers:
                        require(
                            int(row["cache_host_to_device_bytes"])
                            == int(row["cache_record_bytes"])
                            * (
                                int(row["cache_prefetched_records"])
                                + int(row["cache_recalled_records"])
                            ),
                            "Traffic arithmetic differs",
                        )
                    total = {
                        field: sum(int(row["cache_" + field]) for row in layers)
                        for field in count_fields
                    }
                    totals[(h, a, method, phase, sample)] = total
                    if phase == "extend":
                        records.append(total)
            row = {"prefix_tokens": h, "extend_tokens": a, "method": method}
            for field, label, divisor in (
                ("host_to_device_bytes", "h2d_mib", 2**20),
                ("device_to_host_bytes", "d2h_mib", 2**20),
                ("prefetched_records", "prefetched_records", 1),
                ("recalled_records", "recalled_records", 1),
                ("prefetch_capacity_failures", "prefetch_capacity_failures", 1),
            ):
                values = [item[field] / divisor for item in records]
                row.update(
                    {
                        label + "_median": statistics.median(values),
                        label + "_min": min(values),
                        label + "_max": max(values),
                    }
                )
            traffic.append(row)
    timeline = []
    window_map = {(*key(row), row["phase"], row["view"]): row for row in windows}
    require(len(window_map) == 144, "Duplicate timeline key")
    for h, a in sorted(SHAPES):
        receipt = verified(f"h{h}_a{a}/compact_receipt.json")
        for method in METHODS:
            main = window_map[(h, a, method, "extend", "three-layers")]
            startup = window_map[(h, a, method, "extend", "extend-startup")]
            metrics = receipt["metrics"]["extend/" + method]["window"]
            require(int(main["end_ns"]) == int(startup["end_ns"]), "Timeline endpoints differ")
            for name in ("window_ms", "gap_ms", "gap_no_io_percent", "gpu_idle_ms"):
                require(float(main[name]) == metrics[name], "Timeline CSV differs from receipt")
            extra = (int(main["start_ns"]) - int(startup["start_ns"])) / 1e6
            require(extra > 0, "Startup view does not precede the main window")
            extra_idle = float(startup["gpu_idle_ms"]) - float(main["gpu_idle_ms"])
            timeline.append(
                {
                    "prefix_tokens": h,
                    "extend_tokens": a,
                    "method": method,
                    "main_window_ms": metrics["window_ms"],
                    "main_gap_ms": metrics["gap_ms"],
                    "main_gap_no_io_percent": metrics["gap_no_io_percent"],
                    "main_gpu_idle_ms": metrics["gpu_idle_ms"],
                    "main_gpu_idle_percent": 100 * metrics["gpu_idle_ms"] / metrics["window_ms"],
                    "control_only_ms": metrics["control_only_ms"],
                    "pure_io_only_ms": metrics["pure_io_only_ms"],
                    "pure_io_only_percent": 100 * metrics["pure_io_only_ms"] / metrics["window_ms"],
                    "startup_window_ms": float(startup["window_ms"]),
                    "pre_main_ms": extra,
                    "pre_main_idle_percent": 100 * extra_idle / extra,
                    "profile_run_id": main["profile_run_id"],
                }
            )
    summary = {
        "source_run_id": metadata["run_id"],
        "batch_shape_counts": [
            item["completed_shape_count"] for item in metadata["execution_batches"]
        ],
        "shape_count": 12,
        "methods": METHODS,
        "sample_counts": {"prefill": 144, "extend": 240},
        "aggregates": aggregates,
        "extend_winner_counts": dict(winners),
        "offload_winner_counts": dict(offload_winners),
        "echo_serial_h2d_equal_all_extend_samples": all(
            totals[(h, a, "echo", "extend", i)]["host_to_device_bytes"]
            == totals[(h, a, "serial_sparse", "extend", i)]["host_to_device_bytes"]
            for h, a in SHAPES
            for i in range(5)
        ),
        "all_eviction_and_capacity_splits_zero": all(
            item["evicted_records"] == item["capacity_splits"] == 0 for item in totals.values()
        ),
        "extend_first_sample_is_maximum_cases": sum(
            row["first_sample_is_maximum"] for row in points if row["phase"] == "extend"
        ),
        "equal_shape_weighting": True,
        "inferential_statistics": False,
        "hardware": metadata["hardware"],
        "source_compatibility": metadata["source_compatibility"],
        "observer_boundary": metadata["observer"]["boundary"],
    }
    return points, comparisons, aggregates, scaling, traffic, timeline, summary, inputs


def validate_report_premises(points, comparisons, traffic, summary):
    """Keep this dataset-specific narrative from silently describing another run."""
    require(
        summary["source_run_id"] == "deepseek_shape_matrix_complete_20261007_01",
        "The narrative is scoped to the published October 7 matrix; review it for another run",
    )
    require(summary["batch_shape_counts"] == [5, 1, 2, 4], "Batch narrative differs")
    require(summary["extend_winner_counts"] == {"hbm": 12}, "HBM winner narrative differs")
    require(
        summary["offload_winner_counts"] == {"dense_prefetch": 10, "serial_sparse": 2},
        "Offload winner narrative differs",
    )
    serial_wins = {
        (r["prefix_tokens"], r["extend_tokens"])
        for r in comparisons
        if r["offload_winner"] == "serial_sparse"
    }
    require(serial_wins == {(65536, 128), (65536, 256)}, "Crossover narrative differs")
    require(
        all(r["relative_serial"] > 1 for r in comparisons if r["method"] == "echo"),
        "ECHO comparison narrative differs",
    )
    require(summary["echo_serial_h2d_equal_all_extend_samples"], "H2D equality narrative differs")
    require(summary["all_eviction_and_capacity_splits_zero"], "Eviction narrative differs")
    require(
        all(
            r[f"{field}_min"] == r[f"{field}_max"]
            for r in traffic
            for field in ("h2d_mib", "d2h_mib")
        ),
        "Stable payload narrative differs",
    )
    require(
        all(
            r["cv_percent"] > 10
            for r in points
            if r["phase"] == "prefill"
            and (r["prefix_tokens"], r["extend_tokens"], r["method"])
            in {(65536, 128, "hbm"), (4096, 1024, "serial_sparse")}
        ),
        "Prefill variability examples differ",
    )
    summary["dataset_specific_narrative_premises_checked"] = True


def markdown(points, comparisons, aggregates, scaling, traffic, timeline, summary):
    p = {(r["prefix_tokens"], r["extend_tokens"], r["method"], r["phase"]): r for r in points}
    c = {(r["prefix_tokens"], r["extend_tokens"], r["method"]): r for r in comparisons}
    tr = {(r["prefix_tokens"], r["extend_tokens"], r["method"]): r for r in traffic}
    tl = {(r["prefix_tokens"], r["extend_tokens"], r["method"]): r for r in timeline}
    echo_serial = [c[(h, a, "echo")]["relative_serial"] for h, a in sorted(SHAPES)]
    lines = [
        "# DeepSeek V3.2 H × A 矩阵统计报告",
        "",
        "## 主要结果",
        "",
        f"本报告汇总 `{summary['source_run_id']}` 的 12 组形状、4 种方法及 384 个正式计时样本。以下排名依据每组的已测中位数。",
        "",
        "- HBM 的 extend 中位数在 12/12 组中最低。三个 offload 方法中，dense prefetch 在 10/12 组最低；serial sparse 在 H=64K、A=128/256 两组最低；ECHO 没有最低延迟点。",
        f"- 以每个形状等权汇总，ECHO、serial sparse、dense prefetch 的 extend 延迟相对 HBM 的几何平均比值分别为 {next(r for r in aggregates if r['phase'] == 'extend' and r['method'] == 'echo')['geomean_relative_hbm']:.3f}、{next(r for r in aggregates if r['phase'] == 'extend' and r['method'] == 'serial_sparse')['geomean_relative_hbm']:.3f}、{next(r for r in aggregates if r['phase'] == 'extend' and r['method'] == 'dense_prefetch')['geomean_relative_hbm']:.3f} 倍。",
        f"- ECHO 比 serial sparse 慢 {(min(echo_serial) - 1) * 100:.2f}%–{(max(echo_serial) - 1) * 100:.2f}%，几何平均延迟比为 {statistics.geometric_mean(echo_serial):.3f}。两者在全部 12 点、每点 5 次计时中的 H2D 字节数逐点相同，因此本组延迟差异不能归因于 ECHO 搬运了更多字节。",
        "- Prefill 的方法间中位数差异较小，但部分三次样本中出现较大波动。所有样本均保留；本报告不给出显著性、置信区间或线上流量加权结论。",
        "",
        "## 数据范围与统计口径",
        "",
        "H=[4K,16K,64K]，A=[128,256,512,1024]，K=1,024。实际执行真实 checkpoint 的 L0–L2、embedding、三个 dense MLP、final norm 与末 token LM head。History chunk=1,024，extend 整批执行 A 个 token、一次完整 CUDA Graph，P=NH=H+A，采用普通持久 append 和 cold 主 KV 驻留设置。硬件为 GPU0 的 H200 SXM（SM90，名称字段 NVIDIA M403），CPU 绑定 0–7。H/A 变化时，P/NH 也按 H+A 变化；这是一组配置对照。该范围不代表完整 61 层模型、C10 GR serving 或相同字节预算下的容量比较。",
        "",
        "每方法预热 1 次；每点 prefill 计时 3 次、extend 计时 5 次。同步 wall-time 包括输入准备、事务、必要同步和提交，排除加载、编译、图准备及 prefix 恢复。数值验收、独立计时和侵入式 NSYS profile 分开运行。来源分为四批；六个点的启动硬件查询期限为 20 秒，另六个为 120 秒，已核验除此之外执行源码一致。GPU/进程为离散观测，存在一条退出附近的归属不确定记录，其他 GPU 有并行任务，CPU/DRAM 未隔离。原始依据见[矩阵报告](../shape_matrix/results.md)及[独立验收](../shape_matrix/audit/0_audit.json)。",
        "",
        "延迟比定义为 `T_method / T_HBM`，大于 1 表示更慢；反向加速比为 `T_HBM / T_method`。几何平均为 12 个形状延迟比的等权几何平均，不是将全部毫秒样本合并，也不代表业务访问频率。CV 为每点样本标准差（ddof=1）除以样本均值。图中的带状区域仅表示实测最小值到最大值。",
        "",
        "## Extend 延迟与方法比较",
        "",
        "![Extend 中位数与实测范围](extend_latency.svg)",
        "",
        "| H | A | HBM ms | ECHO ms | Serial sparse ms | Dense prefetch ms | 最低 offload 中位数 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for h, a in sorted(SHAPES):
        lines.append(
            f"| {h // 1024}K | {a} | "
            + " | ".join(f"{p[(h, a, m, 'extend')]['median_ms']:.3f}" for m in METHODS)
            + f" | `{c[(h, a, 'hbm')]['offload_winner']}` |"
        )
    lines += [
        "",
        "![相对 HBM 的 extend 延迟比](relative_latency.svg)",
        "",
        "| 方法 | 延迟比几何平均 | 各形状延迟比范围 | 反向加速比几何平均 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        if row["phase"] == "extend" and row["method"] != "hbm":
            lines.append(
                f"| `{row['method']}` | {row['geomean_relative_hbm']:.3f}× | {row['min_relative_hbm']:.3f}–{row['max_relative_hbm']:.3f}× | {1 / row['geomean_relative_hbm']:.3f}× |"
            )
    lines += [
        "",
        "H=64K 时，dense/serial 的延迟比随 A=128、256、512、1024 依次为 "
        + "、".join(f"{c[(65536, a, 'dense_prefetch')]['relative_serial']:.3f}" for a in EXTENSIONS)
        + "。已测点只把两者排序变化夹在 A=256 与 512 之间，不能据此确定精确阈值，也不能外推到未测 A。",
        "",
        "## H 与 A 增长时的变化",
        "",
        "下表为 A 从 128 增至 1,024（8 倍）时的 extend 延迟倍数。",
        "",
        "| 方法 | H=4K | H=16K | H=64K |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        values = [
            next(
                r["latency_ratio"]
                for r in scaling
                if r["axis"] == "A" and r["fixed_tokens"] == h and r["method"] == method
            )
            for h in HISTORIES
        ]
        lines.append(f"| `{method}` | " + " | ".join(f"{v:.3f}×" for v in values) + " |")
    lines += [
        "",
        f"这组数据中，8 倍 A 对应约 {min(r['latency_ratio'] for r in scaling if r['axis'] == 'A'):.2f}–{max(r['latency_ratio'] for r in scaling if r['axis'] == 'A'):.2f} 倍延迟，批内每 token 的摊销时间降低。`A / 时间` 只是本次 batch 的 token 处理率，不是 serving 请求吞吐量。固定 A，将 H 从 4K 增至 64K（16 倍）时，extend 延迟变化如下。",
        "",
        "| 方法 | H=64K / H=4K 延迟比范围（四个 A） |",
        "| --- | ---: |",
    ]
    for method in METHODS:
        values = [r["latency_ratio"] for r in scaling if r["axis"] == "H" and r["method"] == method]
        lines.append(f"| `{method}` | {min(values):.3f}–{max(values):.3f}× |")
    lines += [
        "",
        "## Prefill 与样本波动",
        "",
        "下表对每个 H，列出四个 A 配置中完整 prefill 中位数的最小值–最大值。每个中位数各来自三次计时，没有跨 A 合并样本。",
        "",
        "| H | HBM ms | ECHO ms | Serial sparse ms | Dense prefetch ms |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for h in HISTORIES:
        cells = []
        for m in METHODS:
            values = [p[(h, a, m, "prefill")]["median_ms"] for a in EXTENSIONS]
            cells.append(f"{min(values):.3f}–{max(values):.3f}")
        lines.append(f"| {h // 1024}K | " + " | ".join(cells) + " |")
    prefill_growth = [
        p[(65536, a, m, "prefill")]["median_ms"] / p[(4096, a, m, "prefill")]["median_ms"]
        for m in METHODS
        for a in EXTENSIONS
    ]
    prefill_ratios = [r for r in aggregates if r["phase"] == "prefill" and r["method"] != "hbm"]
    lines += [
        "",
        f"H 扩大 16 倍后，完整 prefill 中位数增至原来的 {min(prefill_growth):.3f}–{max(prefill_growth):.3f} 倍。三个 offload 方法相对 HBM 的几何平均延迟比为 "
        + "、".join(f"{r['method']} {r['geomean_relative_hbm']:.3f}×" for r in prefill_ratios)
        + "；结合以下波动，不将这些小差异解释为稳定优势。",
        "",
        "| 阶段 | 方法 | 每点 CV 的中位数 % | 最大 CV % |",
        "| --- | --- | ---: | ---: |",
    ]
    for r in aggregates:
        lines.append(
            f"| {r['phase']} | `{r['method']}` | {r['median_case_cv_percent']:.3f} | {r['max_case_cv_percent']:.3f} |"
        )
    lines += ["", "两个 prefill 点尤其需要保留原始样本查看：", ""]
    for h, a, m in ((65536, 128, "hbm"), (4096, 1024, "serial_sparse")):
        row = p[(h, a, m, "prefill")]
        lines.append(
            f"- `{m}`，H={h // 1024}K/A={a}：三次样本为 "
            + "、".join(f"{v:.3f}" for v in json.loads(row["samples_ms"]))
            + f" ms；中位数 {row['median_ms']:.3f} ms，均值 {row['mean_ms']:.3f} ms，CV {row['cv_percent']:.3f}%。"
        )
    lines += [
        "",
        f"48 个方法–形状组合中，{summary['extend_first_sample_is_maximum_cases']} 个组合的第一次正式 extend 样本为该组最大值。这是样本顺序现象，现有记录不足以确定原因。统计保留第一次样本和其他较大值，未通过删除样本改善结果。样本量较小且未随机化方法/形状顺序，当前排名是描述性结果。",
        "",
        "## 搬运量与 timeline",
        "",
        "![三层 extend H2D payload](h2d_payload.svg)",
        "",
        "以下为一次完整 extend、三层合计的 cache 软件账本 payload（MiB=2²⁰ B），不是物理总线流量。本组每点五次计时的 H2D、D2H 总量均相同。ECHO 与 serial sparse 的 H2D 总量在所有点逐次相等；ECHO 的这些 H2D records 分为 prefetch 和 residual recall 两类。相同计数不证明每次执行预取了相同 token 集合。",
        "",
        "| H | A | ECHO / serial H2D MiB | Dense H2D MiB | ECHO prefetch records | ECHO residual recall records |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for h, a in sorted(SHAPES):
        echo, dense = tr[(h, a, "echo")], tr[(h, a, "dense_prefetch")]
        lines.append(
            f"| {h // 1024}K | {a} | {echo['h2d_mib_median']:.3f} | {dense['h2d_mib_median']:.3f} | {int(echo['prefetched_records_median'])} | {int(echo['recalled_records_median'])} |"
        )
    lines += [
        "",
        "Record 为 1,152 B；三个 offload 方法的 extend D2H 均为 `3 × A × 1152 B`，完整 prefill H2D 为零、D2H 为 `3 × H × 1152 B`。HBM 两方向均为零。所有样本均未发生 eviction 或 capacity split；不同重复中 ECHO 的预取额度竞争失败计数可能变化，但本组搬运总量未变。",
        "",
        "下表只统计独立 NSYS 捕获的完整 extend L0–L2 主窗口。GPU idle 比例的分母为完整窗口；gap 比例的分母扣除仅被独立 IO 覆盖的时间，且融合 ECHO 区间算有效工作。Gap 含 idle 和仅控制操作占用的时间，两列不能混为同一指标。",
        "",
        "| 方法 | 主窗 GPU idle % 范围 | 主窗 gap（扣除仅 IO 时间）% 范围 | 主窗口前的 profile 时间 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        rows = [r for r in timeline if r["method"] == method]

        def span(field, items=rows):
            values = [r[field] for r in items]
            return f"{min(values):.3f}–{max(values):.3f}"

        lines.append(
            f"| `{method}` | {span('main_gpu_idle_percent')} | {span('main_gap_no_io_percent')} | {span('pre_main_ms')} |"
        )
    dense128, dense1024 = tl[(65536, 128, "dense_prefetch")], tl[(65536, 1024, "dense_prefetch")]
    lines += [
        "",
        f"H=64K 时，dense 主窗仅 IO 覆盖的时间从 A=128 的 {dense128['pure_io_only_ms']:.3f} ms（窗口的 {dense128['pure_io_only_percent']:.2f}%）降至 A=1024 的 {dense1024['pure_io_only_ms']:.3f} ms（{dense1024['pure_io_only_percent']:.2f}%），两点 H2D 均为 216 MiB。相应独立延迟由比 serial 高 {(c[(65536, 128, 'dense_prefetch')]['relative_serial'] - 1) * 100:.2f}% 变为低 {(1 - c[(65536, 1024, 'dense_prefetch')]['relative_serial']) * 100:.2f}%。这一现象与较长计算覆盖更多搬运时间相符，但尚不是严格的因果分解。",
        "",
        "启动视图与主视图终点相同，差额覆盖 `forward` 入口到主窗口起点，包括启动、embedding 及这段内其他活动。它不等于独立计时的 CPU 开销，不能从 wall-time 中直接扣除。Prefill timeline 只覆盖最后一个 1,024-token chunk，不能拿来替代上文完整 prefill 时间。每形状只有一次侵入式 profile，不用不同运行的字节数与 profile 时间拼出带宽，也不把阶段时间相加解释全部 wall-time。",
        "",
        "## 数据与复现",
        "",
        "- [逐点统计及全部样本](point_statistics.csv)：96 行（两阶段），保留均值、中位数、最小/最大值、标准差与 CV。",
        "- [Extend 比较](comparisons.csv)、[方法汇总](aggregate_statistics.csv)、[H/A 缩放](scaling.csv)。",
        "- [逐点搬运量](traffic_statistics.csv)、[主窗与启动窗统计](timeline_statistics.csv)。",
        "- [统计汇总与边界](summary.json)、[输入及脚本哈希](provenance.json)、[发布清单](publication_manifest.json)。",
        "- [原始 12 组 timeline 导航](../shape_matrix/results.md#timelines)。",
        "",
        "本报告仅对已有测量做 CPU 统计，没有新增 GPU 测量、逐算子 MFU 或 serving 性能结论。复现时从仓库根目录运行，输出目录须使用新名称：",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES= .venv/bin/python -m experiments.deepseek_v32_mfu.src.summarize_shape_matrix \\",
        "  --source-report experiments/deepseek_v32_mfu/report/shape_matrix \\",
        "  --run-id deepseek_shape_statistics_new \\",
        "  --output-dir experiments/deepseek_v32_mfu/output/data/deepseek_shape_statistics_new \\",
        "  --publish-dir experiments/deepseek_v32_mfu/report/shape_matrix_statistics_new",
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path)
    args = parser.parse_args()
    from experiments.deepseek_v32_mfu.src import plot_shape_statistics

    require(not args.output_dir.exists(), "Output already exists")
    require(args.publish_dir is None or not args.publish_dir.exists(), "Publication already exists")
    points, comparisons, aggregates, scaling, traffic, timeline, summary, inputs = derive(
        args.source_report
    )
    validate_report_premises(points, comparisons, traffic, summary)
    for module in (Path(__file__).resolve(), Path(plot_shape_statistics.__file__).resolve()):
        inputs[str(module)] = digest(module)
    args.output_dir.mkdir(parents=True)
    for name, rows in (
        ("point_statistics", points),
        ("comparisons", comparisons),
        ("aggregate_statistics", aggregates),
        ("scaling", scaling),
        ("traffic_statistics", traffic),
        ("timeline_statistics", timeline),
    ):
        write_csv(args.output_dir / (name + ".csv"), rows)
    summary["statistics_run_id"] = args.run_id
    write_json(args.output_dir / "summary.json", summary)
    plot_shape_statistics.render(args.output_dir)
    (args.output_dir / "results.md").write_text(
        markdown(points, comparisons, aggregates, scaling, traffic, timeline, summary)
    )
    require(
        all(digest(path) == value for path, value in inputs.items()), "Statistical inputs changed"
    )
    write_json(
        args.output_dir / "provenance.json",
        {
            "run_id": args.run_id,
            "source_run_id": summary["source_run_id"],
            "input_and_source_sha256": inputs,
            "command": sys.argv,
            "boundary": "CPU descriptive statistics of accepted published matrix data; no new GPU measurements. No sample filtering, significance tests or workload-frequency weighting.",
        },
    )
    write_json(
        args.output_dir / "publication_manifest.json",
        {
            "schema": "deepseek-mfu-shape-statistics-v1",
            "run_id": args.run_id,
            "source_run_id": summary["source_run_id"],
            "raw_report_directory": str(args.output_dir.resolve()),
            "files_sha256": {
                str(path.relative_to(args.output_dir)): digest(path)
                for path in sorted(args.output_dir.rglob("*"))
                if path.is_file()
            },
        },
    )
    if args.publish_dir:
        shutil.copytree(args.output_dir, args.publish_dir)
    print(args.output_dir / "results.md")


if __name__ == "__main__":
    main()
