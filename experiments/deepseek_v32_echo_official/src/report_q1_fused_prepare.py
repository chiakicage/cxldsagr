"""Publish accepted private preparation controls without importing GPU code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "candidate")
POLICIES = ("zero", "small", "partial", "warm", "empty")
ACCEPTED_COMPONENT_SHA256 = "687e6fab030ad35a62497efa616735fc8b0c6b57846718405708b02b8b32c57c"
ACCEPTED_ANALYSIS_SHA256 = "fd31fc52cfd6d6882602de67f9ba9cea9e465837887bcefddf6ab180312ea11f"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def close(actual, expected):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), "Statistic differs")


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def receipt_payload(path, identity, signature):
    """Check the signed record; the prior independent reviews audit its artifacts."""
    receipt = read(path)
    require(receipt["receipt_sha256"] == signature, "Receipt signature differs")
    require(
        identity_digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}) == signature,
        "Invalid receipt signature",
    )
    require(receipt["identity"] == identity, "Receipt execution identity differs")
    require(receipt["checks"]["passed"] is True, "Numerical acceptance failed")
    return receipt


def component_rows(summary, review):
    require(summary["mode"] == "bench" and review["passed"] is True, "Invalid component run")
    require(len(summary["samples"]) == review["samples"] == 3000, "Incomplete component samples")
    require(len(summary["rows"]) == len(review["rows"]) == 15, "Incomplete component groups")
    result, consumed = [], set()
    for layer in range(3):
        for policy in POLICIES:
            case = f"kernel_inputs_layer_{layer}"
            samples = [
                row for row in summary["samples"] if row["case"] == case and row["policy"] == policy
            ]
            require(len(samples) == 200, "Wrong component group length")
            values, deltas = {arm: [] for arm in ARMS}, {"AB": [], "BA": []}
            for pair in range(100):
                order = ARMS if pair % 2 == 0 else ARMS[::-1]
                selected = samples[pair * 2 : pair * 2 + 2]
                require(
                    [row["variant"] for row in selected] == list(order)
                    and all(
                        row["pair"] == pair and row["order"] == list(order) for row in selected
                    ),
                    "Component execution order differs",
                )
                pair_values = {}
                for row in selected:
                    require(math.isfinite(row["gpu_us"]) and row["gpu_us"] > 0, "Invalid latency")
                    require(
                        type(row["attempts"]) is int
                        and row["attempts"] >= 0
                        and row["staged"] == min(64, row["attempts"])
                        and row["h2d_bytes"] == row["staged"] * 1152,
                        "Component traffic differs",
                    )
                    values[row["variant"]].append(row["gpu_us"])
                    pair_values[row["variant"]] = row["gpu_us"]
                deltas["AB" if pair % 2 == 0 else "BA"].append(
                    pair_values["candidate"] - pair_values["baseline"]
                )
            all_deltas = deltas["AB"] + deltas["BA"]
            row = {
                "case": case,
                "policy": policy,
                "pairs": 100,
                "warmups": 20,
                "baseline_us": statistics.median(values["baseline"]),
                "candidate_us": statistics.median(values["candidate"]),
                "paired_delta_us": statistics.median(all_deltas),
                "candidate_wins": sum(value < 0 for value in all_deltas),
                "AB_delta_us": statistics.median(deltas["AB"]),
                "BA_delta_us": statistics.median(deltas["BA"]),
            }
            originals = [r for r in summary["rows"] if (r["case"], r["policy"]) == (case, policy)]
            reviewed = [r for r in review["rows"] if (r["case"], r["policy"]) == (case, policy)]
            require(len(originals) == len(reviewed) == 1, "Duplicate component group")
            for original in (originals[0], reviewed[0]):
                for key, value in original.items():
                    if isinstance(value, (int, float)):
                        close(row[key], value)
                    else:
                        require(row[key] == value, "Component label differs")
            consumed.add((case, policy))
            result.append(row)
    require(
        {(r["case"], r["policy"]) for r in summary["samples"]} == consumed,
        "Unexpected component group",
    )
    return result


def model_pairs(bench, analysis, review):
    require(bench["completed"] is True and bench["mode"] == "bench", "Invalid model run")
    require(
        analysis["passed"] is True
        and review["passed"] is True
        and review["measurement_integrity_accepted"] is True,
        "Model measurement not accepted",
    )
    require(
        bench["pairs"] == 500 and len(bench["result"]["samples"]) == 1000, "Missing model pairs"
    )
    pairs = []
    for pair in range(500):
        order = ARMS if pair % 2 == 0 else ARMS[::-1]
        selected = bench["result"]["samples"][pair * 2 : pair * 2 + 2]
        require([row["arm"] for row in selected] == list(order), "Model order differs")
        values, traffic = {}, {}
        for row in selected:
            require(
                row["pair"] == pair and row["order"] == ("AB" if pair % 2 == 0 else "BA"),
                "Model pair differs",
            )
            require(math.isfinite(row["wall_ms"]) and row["wall_ms"] > 0, "Invalid model latency")
            require(len(row["layer_metrics"]) == 3, "Missing model layer metrics")
            for metric in row["layer_metrics"]:
                require(
                    metric["record_bytes"] == 1152
                    and metric["prefetched_records"] == 64
                    and metric["recalled_records"] + metric["resident_selection_records"] == 2048
                    and metric["host_to_device_bytes"]
                    == (metric["prefetched_records"] + metric["recalled_records"]) * 1152
                    and metric["device_to_host_bytes"] == 1152,
                    "Model traffic arithmetic differs",
                )
            values[row["arm"]] = row["wall_ms"]
            traffic[row["arm"]] = sum(m["host_to_device_bytes"] for m in row["layer_metrics"])
        delta = values["candidate"] - values["baseline"]
        current = {
            "pair": pair,
            "block": pair // 100,
            "order": "AB" if pair % 2 == 0 else "BA",
            "baseline_ms": values["baseline"],
            "candidate_ms": values["candidate"],
            "delta_ms": delta,
            "relative_delta_percent": delta / values["baseline"] * 100,
            "baseline_h2d_bytes": traffic["baseline"],
            "candidate_h2d_bytes": traffic["candidate"],
            "h2d_delta_bytes": traffic["candidate"] - traffic["baseline"],
        }
        pairs.append(current)
    saved = analysis["clean_timing"]["paired_samples"]
    require(len(saved) == 500, "Incomplete analysis pairs")
    for actual, original in zip(pairs, saved, strict=True):
        for key, value in original.items():
            if isinstance(value, (int, float)):
                close(actual[key], value)
            else:
                require(actual[key] == value, "Analysis pair label differs")
    return pairs


def describe(pairs):
    deltas = [row["delta_ms"] * 1000 for row in pairs]
    traffic = [row["h2d_delta_bytes"] for row in pairs]
    return {
        "pairs": len(pairs),
        "candidate_faster_pairs": sum(value < 0 for value in deltas),
        "median_delta_us": statistics.median(deltas),
        "mean_delta_us": statistics.mean(deltas),
        "median_h2d_delta_bytes": statistics.median(traffic),
        "mean_h2d_delta_bytes": statistics.mean(traffic),
        "min_h2d_delta_bytes": min(traffic),
        "max_h2d_delta_bytes": max(traffic),
    }


def model_statistics(pairs, review):
    groups = [("all", "all", pairs, review["overall"])]
    for order in ("AB", "BA"):
        groups.append(
            ("all", order, [r for r in pairs if r["order"] == order], review["orders"][order])
        )
    require(len(review["blocks"]) == 5, "Missing reviewed time blocks")
    for block, expected in enumerate(review["blocks"]):
        require(expected["block"] == block, "Block ordering differs")
        selected = [r for r in pairs if r["block"] == block]
        groups.append((block, "all", selected, expected))
        for order in ("AB", "BA"):
            groups.append(
                (
                    block,
                    order,
                    [r for r in selected if r["order"] == order],
                    expected["orders"][order],
                )
            )
    result = []
    for block, order, rows, expected in groups:
        stats = describe(rows)
        for key, value in stats.items():
            close(value, expected[key])
        stats.update(block=block, order=order)
        result.append(stats)
    distributions = {}
    for arm in ARMS:
        values = [r[arm + "_ms"] for r in pairs]
        distributions[arm] = {
            "samples": len(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "p95_ms": quantile(values, 0.95),
            "p99_ms": quantile(values, 0.99),
        }
        for key, value in distributions[arm].items():
            close(value, review["arm_latency_distribution"][arm][key])
    tail_counts = {
        "absolute_over_0_5_ms": sum(abs(row["delta_ms"]) > 0.5 for row in pairs),
        "negative_below_minus_0_5_ms": sum(row["delta_ms"] < -0.5 for row in pairs),
        "positive_above_0_5_ms": sum(row["delta_ms"] > 0.5 for row in pairs),
    }
    for key, value in tail_counts.items():
        require(value == review["large_paired_delta_counts"][key], "Tail count differs")
    return result, distributions


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(result, provenance):
    model = result["model"]
    distribution = model["latency_distribution"]
    baseline, candidate = (distribution[arm] for arm in ARMS)
    overall = model["statistics"][0]
    cold = [r for r in result["component"]["groups"] if r["policy"] == "zero"]
    cold_table = "\n".join(
        f"| L{index} | {r['baseline_us']:.3f} | {r['candidate_us']:.3f} | {r['paired_delta_us']:.3f} |"
        for index, r in enumerate(cold)
    )
    model_table = "\n".join(
        f"| {arm} | {r['median_ms']:.7f} | {r['mean_ms']:.7f} | {r['p95_ms']:.7f} | {r['p99_ms']:.7f} |"
        for arm, r in distribution.items()
    )
    strata_table = "\n".join(
        f"| {r['block']} | {r['order']} | {r['pairs']} | {r['median_delta_us']:.3f} | {r['mean_delta_us']:.3f} | {r['candidate_faster_pairs']} |"
        for r in model["statistics"]
        if r["block"] == "all" or r["order"] == "all"
    )
    return f"""# Q1 融合准备的 private 对照

该候选将 page64 打包、identity block table 和 host-token/staging 准备合并，保留官方
ECHO score/prefetch 核心、阈值、64 条 record 上限、精确 top-k、EMA 及后续发布和清理。
它目前只在实验私有入口中运行。本报告保留接入 production 前的有效优化对照；独立验收
和测量完整性检查通过，不等于生产版本已获得相同收益。

组件与完整模型使用不同计时边界。组件的冷缓存三层配对收益合计为
{result["component"]["cold_sum_paired_savings_us"]:.3f} µs；完整模型的配对中位数变化为
{overall["median_delta_us"]:.4f} µs。这两个数字不能相减或用来证明全部差值来自准备融合。

## 组件计时

输入取真实前三层 Q1 张量，H=65,536、N=65,537。每个 L0/L1/L2 ×
zero/small/partial/warm/empty 组合预热 20 次，再运行 100 对交替 AB/BA，共 3,000 个
arm 样本。每次在计时外恢复该策略的驻留状态。CUDA Graph 内的 GPU event 覆盖完整
`echo.logits` 和原 staging cleanup，包括打包、边界和页表准备、官方调度、stage reset、
融合分数、causal clean 和 promotion；caller FIFO 准备、精确 top-k 与 recall 在窗口外。

zero 从无驻留历史、阈值 0 开始；small 同样无驻留，阈值取 history 分数的第 17 大值。
partial 预先驻留隔一个 token 的历史，warm 驻留全部历史，两者阈值均为 0。
empty 无驻留且阈值为正无穷，因而不触发预测搬运。这些是明确定义的组件控制条件。

| 冷缓存 | baseline µs | candidate µs | 配对中位数变化 µs |
| --- | ---: | ---: | ---: |
{cold_table}

15 个组合的配对中位数均下降，范围为 0.976–2.800 µs；每个组合的 AB、BA 中位数也都
下降。完整组合数据见 [component_groups.csv](component_groups.csv)。原始 3,000 个
样本保留在组件 run 的 `summary.json` 及本报告 run 的输入副本中。

## 完整前三层模型计时

使用原始 FP8 权重、BF16 KV，真实 checkpoint L0–L2，H=65,536、A=1，history chunk
为 1,024，P=NH=65,600。baseline 与 candidate 是同一进程中依次准备的两个固定模型
实例，分别预热 5 次。正式计时运行预先规定的 500 对 AB/BA，各 250 对；每次恢复 cold
prefix/hint。窗口为完整 `forward(return_hidden=True)` 加 synchronize，恢复、绑定、
计数读取和诊断在窗口外。本表不使用 profile 的时长。

| arm | 中位数 ms | 均值 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
{model_table}

candidate 在 {overall["candidate_faster_pairs"]} / 500 对中较快，配对中位数下降
{abs(overall["median_delta_us"]):.4f} µs。下表保留全部时间区段；block 0–4 分别对应连续
100 对。AB 表示 baseline 先执行，BA 表示 candidate 先执行。差值均为 candidate 减
baseline，负值表示本次观察中 candidate 较快。

| block | 顺序 | 对数 | 配对中位数变化 µs | 配对均值变化 µs | candidate 较快对数 |
| --- | --- | ---: | ---: | ---: | ---: |
{strata_table}

两种顺序及五个区段的中位数都改善；每个区段内单独的 AB、BA 中位数也都改善。
均值和尾部并不一致：AB 均值上升 3.139 µs，block 0 和 4 均值上升；candidate p99
从 {baseline["p99_ms"]:.6f} 增至 {candidate["p99_ms"]:.6f} ms。81 对的绝对差值超过
0.5 ms，其中 35 对 candidate 较慢、46 对较快。没有删除异常值、裁剪早晚区段、
替换样本或重加权。[model_pairs.csv](model_pairs.csv)保留全部 500 对及各对实际 H2D，
[model_statistics.csv](model_statistics.csv)包括每个 block 的两种顺序。
五区段重采样只描述本次观察对区段变化的敏感性，不是独立运行的置信区间。

每层实际预取均为 64 条，每条 1,152 B；预测与精确 top-k 的重合会随调度变化，
剩余选择仍完整 recall。三层合计 H2D 的配对中位数增加 {overall["median_h2d_delta_bytes"]:,.0f} B，
均值增加 {overall["mean_h2d_delta_bytes"]:,.3f} B，范围
{overall["min_h2d_delta_bytes"]:,} 至 {overall["max_h2d_delta_bytes"]:,} B。
这些字节来自每次执行的实际 cache 计数，不能由 memcpy 行数推断，也不能用 profile
样本的流量替代。H2D 增加说明两条轨迹的工作量并非逐项相同。

两 arm 的 graph private reserved 均为 62,914,560 B。保存的 `graph.describe` 计数和
reservation 算术已核验，但没有保留 allocator 原始 segments，无法独立重建 private
segment 归属。allocated、reserved 和设备已用量分别保留在 [result.json](result.json)；
它们是依次准备模型时的进程快照，不能相减得到单个模型占用，也不是连续峰值或容量验收。

## 验收、来源与复现

两组运行都使用 GPU 1（NVIDIA H200，SM90），CPU 8–15，线程数 8，GPU UUID 为
`a2226185-cb05-a411-80da-f365154128fe`；Torch 2.12.1+cu130、CUDA 13.0、
Triton 3.7.1、FlashInfer 0.6.18。组件 receipt 覆盖 100 个 byte/layout case、210 个
完整 case，另有 6 个 malformed 子进程检查。模型 receipt 覆盖两个独立准备的 prefix、
三个变化 suffix token（111090、111091、111092）、eager、诊断 graph 和 clean graph，
包括输出、所有 offset 位、精确 scores/top-k 与实际 saturated-prefetch 转移。

独立 CPU 复核重新读取了 36 组 raw layer scores/top-k，重算 12 组 compact transition
proof，并检查映射、priority、free bit、clock 和计数。compact 证据没有完整 staged、
host、最终 KV payload；这些内容的逐字节比较仍依赖签名运行时验收。CPU 复核没有执行
GPU、重测性能、完整重哈希权重或重建 allocator segments。

来源 run ID：组件 `q1_fused_prepare_bench_20261008_02`，模型
`q1_fused_prepare_model_bench_20261008_02`，模型分析
`q1_fused_prepare_model_analysis_20261009_01`。receipt 签名分别为
`{provenance["component"]["receipt_signature"]}` 和
`{provenance["model"]["receipt_signature"]}`。完整输入、源码及 native 身份在原运行与
分析清单中；[provenance.json](provenance.json)给出来源文件哈希、关键源码/ELF、
三份不可变 FlashInfer ELF 和独立复核范围。FlashInfer 的源码、spec、include、toolchain
与编译器 `.d` 依赖闭包绑定到实际加载文件，check 和 bench 使用相同 ELF。官方 ECHO
headers 未修改，固定提交为 `bc1b75c1000010d0ac6f032ebaac283255c050b1`。
checkpoint 身份使用 metadata 哈希和全部 shard 的文件系统身份，未完整哈希权重内容。

报告生成器核验已接受分析及复核文件的来源哈希，重新统计全部组件和模型样本、顺序、
区段与流量算术，再逐字节核验选定副本。它不重复运行 GPU 验收，也不重新读取全部
tensor/native/source 依赖；该范围由已归档的独立检查和分析负责。

从仓库根目录运行下列命令；输出目录必须使用新的 run ID。完整报告原件和输入副本在
`experiments/deepseek_v32_echo_official/output/data/{result["run_id"]}/`，选定文件的哈希见
[publication.json](publication.json)。

```bash
python -m experiments.deepseek_v32_echo_official.src.report_q1_fused_prepare \\
  --component-dir experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_bench_20261008_02 \\
  --model-analysis-dir experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_model_analysis_20261009_01 \\
  --model-check-review /tmp/cxldsagr-checks/q1-fused-prepare-model/check_20261008_02_independent_review.json \\
  --output-dir experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_report_YYYYMMDD_NN
```

本次结论只覆盖 private 的真实前三层、固定输入和单进程配对计时。独立 source-bound
profile/work-equivalence 解释及 production 接入后的正式 cohort 另行验收；它不建立
完整 61 层、serving、session 容量或 SGLang 与本地模型数值等价的结论。
"""


def publish(args):
    component, model_dir, check_review_path, output = (
        Path(value).resolve()
        for value in (
            args.component_dir,
            args.model_analysis_dir,
            args.model_check_review,
            args.output_dir,
        )
    )
    require(
        output.is_relative_to(EXPERIMENT / "output" / "data"),
        "Report output must remain in output/data",
    )
    require(not output.exists(), "Use a new report run ID")
    report_dir = Path(args.report_dir).resolve() if args.report_dir else None
    if report_dir:
        require(
            report_dir.is_relative_to(EXPERIMENT / "report"),
            "Selected assets must remain in report",
        )
        require(not report_dir.exists(), "Selected report already exists")
    sources = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        value = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        require(
            expected is None or value["sha256"] == expected, "Source digest differs: " + str(path)
        )
        sources[str(path)] = value
        return value

    manifest = read(model_dir / "publication_manifest.json")
    bind(model_dir / "publication_manifest.json")
    for name, expected in manifest["files_sha256"].items():
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts, "Unsafe source manifest path"
        )
        bind(model_dir / relative, expected)
    summary, component_review = (
        read(component / name) for name in ("summary.json", "independent_review.json")
    )
    bind(component / "summary.json", ACCEPTED_COMPONENT_SHA256)
    bind(model_dir / "result.json", ACCEPTED_ANALYSIS_SHA256)
    bind(component / "summary.json", component_review["summary_sha256"])
    bind(component / "independent_review.json")
    analysis = read(model_dir / "result.json")
    review = read(model_dir / "independent_review.json")
    bind(model_dir / "result.json", review["analysis_result_sha256"])
    bench_path = Path(analysis["bindings"]["bench"]["path"])
    bind(bench_path, analysis["bindings"]["bench"]["sha256"])
    require(review["source_result_sha256"] == digest(bench_path), "Review source run differs")
    bench = read(bench_path)
    source, runtime = bench["identity"]["source"], bench["identity"]["timing_runtime"]
    expected_contract = {
        "history": 65536,
        "append": 1,
        "layers": [0, 1, 2],
        "chunk_size": 1024,
        "slots": 65600,
        "host_arena_tokens": 65600,
        "warmups": 5,
        "tokens": [111090, 111091, 111092],
        "candidate": "q1-fused-page-and-stage-prepare-v1",
    }
    require(
        all(source["contract"].get(k) == v for k, v in expected_contract.items()),
        "Wrong model contract",
    )
    require(
        source["gpu"]
        == {
            "capability": [9, 0],
            "name": "NVIDIA H200",
            "uuid": "a2226185-cb05-a411-80da-f365154128fe",
        }
        and source["affinity"] == list(range(8, 16))
        and source["threads"] == 8
        and source["torch"] == "2.12.1+cu130"
        and source["cuda"] == "13.0",
        "Wrong report environment",
    )
    require(
        summary["identity"]["physical_device"] == 1
        and summary["identity"]["cpu_affinity"] == list(range(8, 16))
        and summary["identity"]["triton"] == "3.7.1"
        and summary["identity"]["flashinfer"] == "0.6.18",
        "Wrong component environment",
    )
    require(
        source["component"]["benchmark"]["sha256"] == digest(component / "summary.json"),
        "Model/component binding differs",
    )
    check_review = read(check_review_path)
    bind(check_review_path)
    require(check_review["passed"] is True, "Independent saved-tensor review failed")
    require(
        check_review["receipt_signature"] == analysis["bindings"]["receipt_signature"]
        and check_review["identity_sha256"]
        == analysis["bindings"]["identity_sha256"]
        == identity_digest(bench["identity"]),
        "Saved-tensor review binding differs",
    )
    component_receipt_path = Path(source["component"]["receipt"]["path"])
    bind(component_receipt_path, source["component"]["receipt"]["sha256"])
    component_receipt = receipt_payload(
        component_receipt_path, summary["identity"], summary["receipt_sha256"]
    )
    require(
        component_review["receipt_sha256"] == summary["receipt_sha256"],
        "Component review receipt differs",
    )
    model_receipt_path = Path(analysis["bindings"]["receipt"]["path"])
    bind(model_receipt_path, analysis["bindings"]["receipt"]["sha256"])
    receipt_payload(model_receipt_path, bench["identity"], check_review["receipt_signature"])
    require(
        bench["result"]["receipt_sha256"] == check_review["receipt_signature"],
        "Timing receipt differs",
    )
    groups = component_rows(summary, component_review)
    pairs = model_pairs(bench, analysis, review)
    stats, distributions = model_statistics(pairs, review)
    with (model_dir / "paired_samples.csv").open(newline="") as stream:
        csv_pairs = list(csv.DictReader(stream))
    require(len(csv_pairs) == 500, "Published analysis CSV is incomplete")
    for actual, saved in zip(pairs, csv_pairs, strict=True):
        for key, value in saved.items():
            if key == "order":
                require(actual[key] == value, "CSV order differs")
            else:
                close(actual[key], float(value))
    result = {
        "schema": "private-q1-fused-prepare-report-v1",
        "run_id": output.name,
        "measurement_integrity_accepted": True,
        "production_promotion_accepted": False,
        "component": {
            "groups": groups,
            "arm_samples": 3000,
            "cold_sum_paired_savings_us": -sum(
                r["paired_delta_us"] for r in groups if r["policy"] == "zero"
            ),
            "boundary": summary["identity"]["boundary"],
        },
        "model": {
            "statistics": stats,
            "latency_distribution": distributions,
            "large_paired_delta_counts": review["large_paired_delta_counts"],
            "memory": review["memory"],
            "boundary": review["boundary"],
            "descriptive_block_sensitivity": analysis["clean_timing"][
                "descriptive_block_sensitivity"
            ],
        },
    }
    compact_check = {
        k: v for k, v in check_review.items() if k not in ("files", "analysis_sources")
    }
    provenance = {
        "schema": "private-q1-fused-prepare-report-provenance-v1",
        "run_id": output.name,
        "reporter": bind(__file__),
        "inputs": list(sources.values()),
        "publication_audit_scope": "Hashes of accepted analysis/reviews/receipts, receipt identity/signature, all 3000 component and 1000 model arm samples, order/block statistics, actual traffic arithmetic and selected copies. No GPU execution, tensor deserialization or full dependency rehash.",
        "component": {
            "run_id": component.name,
            "receipt_signature": summary["receipt_sha256"],
            "checks": component_receipt["checks"],
            "independent_review_boundary": component_review["boundary"],
            "inputs": summary["identity"]["inputs"],
            "source_files": len(summary["identity"]["sources"]),
            "runtime_files": len(summary["identity"]["runtime_files"]),
            "runtime_archives": summary["runtime_archives"],
        },
        "model": {
            "run_id": bench["run_id"],
            "analysis_run_id": analysis["run_id"],
            "identity_sha256": check_review["identity_sha256"],
            "receipt_signature": check_review["receipt_signature"],
            "contract": source["contract"],
            "gpu": source["gpu"],
            "cpu_affinity": source["affinity"],
            "request": source["request"],
            "checkpoint_boundary": source["checkpoint"]["boundary"],
            "source_files": len(source["sources"]),
            "mapped_native": runtime["mapped_native"],
            "private_source_sha256": {
                k: v for k, v in source["sources"].items() if "q1_fused_prepare" in k
            },
            "flashinfer": {
                name: {k: value[k] for k in ("cache_key", "library", "build_metadata", "manifest")}
                for name, value in runtime["private_flashinfer_native"].items()
            },
            "candidate_elf_sha256": runtime["candidate_native"]["artifact_sha256"],
            "official_elf_sha256": runtime["official_native"]["artifact_sha256"],
            "saved_tensor_review": compact_check,
        },
    }
    # Recheck source files before creating any publication output.
    for record in sources.values():
        require(
            digest(record["path"]) == record["sha256"], "Input changed during report generation"
        )
    output.mkdir(parents=True)
    input_dir = output / "inputs"
    input_dir.mkdir()
    for name, path in {
        "component_summary.json": component / "summary.json",
        "component_independent_review.json": component / "independent_review.json",
        "model_analysis.json": model_dir / "result.json",
        "model_analysis_manifest.json": model_dir / "publication_manifest.json",
        "model_analysis_input_hashes.json": model_dir / "input_hashes.json",
        "model_benchmark.json": bench_path,
        "model_independent_review.json": model_dir / "independent_review.json",
    }.items():
        shutil.copyfile(path, input_dir / name)
        require(digest(input_dir / name) == digest(path), "Input archive copy differs")
    write(input_dir / "model_saved_tensor_review_compact.json", compact_check)
    source_copy = output / "source" / Path(__file__).name
    source_copy.parent.mkdir()
    shutil.copyfile(__file__, source_copy)
    write(output / "result.json", result)
    write(output / "provenance.json", provenance)
    write_csv(output / "component_groups.csv", groups)
    write_csv(output / "model_statistics.csv", stats)
    write_csv(output / "model_pairs.csv", pairs)
    (output / "report.md").write_text(report_text(result, provenance))
    selected = (
        "report.md",
        "result.json",
        "component_groups.csv",
        "model_statistics.csv",
        "model_pairs.csv",
        "provenance.json",
    )
    publication = {
        "schema": "private-q1-fused-prepare-publication-v1",
        "run_id": output.name,
        "original_directory": str(output),
        "selected_files_sha256": {name: digest(output / name) for name in selected},
        "all_files_sha256": {
            str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()
        },
        "numerical_acceptance_scope": "Private component and local real L0-L2 candidate; no official SGLang cross-implementation numerical acceptance.",
        "production_promotion_accepted": False,
    }
    write(output / "publication.json", publication)
    if report_dir:
        report_dir.mkdir(parents=True)
        for name in (*selected, "publication.json"):
            shutil.copyfile(output / name, report_dir / name)
            require(
                digest(report_dir / name) == digest(output / name), "Selected asset copy differs"
            )
    return publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--model-analysis-dir", type=Path, required=True)
    parser.add_argument("--model-check-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    publication = publish(args)
    print(
        json.dumps(
            {
                "run_id": publication["run_id"],
                "selected_files_sha256": publication["selected_files_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
