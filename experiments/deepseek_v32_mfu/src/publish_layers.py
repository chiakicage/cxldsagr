"""Publish verified three-layer ECHO profiles without changing measured results."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from experiments.deepseek_v32_mfu.src.compare_backends import validate_result
from experiments.deepseek_v32_mfu.src.execution_utilization import (
    precision_normalized_utilization,
)
from experiments.deepseek_v32_mfu.src.profile_summary import (
    EXPERIMENT,
    collect_ncu,
    digest,
    read_json,
    require,
    run_directory,
    write_csv,
)
from experiments.deepseek_v32_mfu.src.run_contract import benchmark_view

MODES = ("resident", "offload")
PHASES = ("prefill_annotated", "extend_annotated")
STAGES = (
    "q_a_proj",
    "q_b_proj",
    "q_absorb",
    "kv_a_proj",
    "index_q_proj",
    "index_k_proj",
    "index_weights_proj",
    "indexer_qk",
    "indexer_fused",
    "mla_qk_pv",
    "v_expand",
    "o_proj",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "lm_head",
)
NCU_METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
    "launch__waves_per_multiprocessor",
    "derived__local_spilling_requests",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
)


def _end_to_end_utilization(result, ledger, peaks):
    """Bind complete dense-layer work to the matching formal wall samples."""
    scope = "checkpoint_layers_0_1_2_embedding_final_norm_last_token_lm_head"
    require(
        result["num_layers"] == 3 and result["scope"] == scope,
        "End-to-end utilization requires the measured three-layer single-device workload",
    )
    required = set(STAGES) - {"indexer_qk", "indexer_fused", "lm_head"}
    output = {}
    work_by_mode = {}
    for phase, annotated_phase in zip(("prefix", "extend"), PHASES):
        for mode in MODES:
            calls = [
                r for r in ledger["calls"] if (r["mode"], r["phase"]) == (mode, annotated_phase)
            ]
            matrix = [r for r in calls if r["useful_flops"] is not None]
            for layer in range(3):
                stages = {r["stage"] for r in matrix if r["layer"] == f"layer_{layer}"}
                indexer_stages = stages & {"indexer_qk", "indexer_fused"}
                require(
                    bool(indexer_stages)
                    and stages == required | indexer_stages
                    and (mode == "offload" or indexer_stages == {"indexer_qk"}),
                    "Incomplete or unsupported matrix ledger",
                )
            head = [r for r in matrix if r["layer"] == "shared"]
            require(
                len(head) == 1 and head[0]["stage"] == "lm_head",
                "Expected one shared last-token LM-head call",
            )
            require(
                all(r["layer"] in {"layer_0", "layer_1", "layer_2", "shared"} for r in matrix),
                "Matrix work lies outside the measured three layers",
            )
            work = defaultdict(int)
            for call in matrix:
                stage = (
                    "indexer" if call["stage"] in {"indexer_qk", "indexer_fused"} else call["stage"]
                )
                work[(call["layer"], stage, call["precision"])] += call["useful_flops"]
            work_by_mode[mode, phase] = dict(work)
            output.setdefault(mode, {})[phase] = precision_normalized_utilization(
                calls,
                result["measurements"][mode][phase + "_samples_ms"],
                peaks_tflops=peaks,
                scope=scope,
            )
            if "wall_time_denominator" in result:
                output[mode][phase]["wall_time_denominator"] = result["wall_time_denominator"]
        require(
            work_by_mode["resident", phase] == work_by_mode["offload", phase],
            "Resident/offload useful matrix work differs; inspect semantic equivalence",
        )
    return output


def _identity(row):
    return tuple(row[key] for key in ("mode", "phase", "layer", "stage"))


def _verify(directory, result, analysis, ledger, hashes):
    require(result["accepted"] is True and result["run_id"] == directory.name, "Run not accepted")
    require(result["num_layers"] == 3, "This publisher requires exactly three checkpoint layers")
    require(ledger["run_id"] == directory.name, "Call ledger run ID mismatch")
    require(analysis["ledger_metadata"]["run_id"] == directory.name, "Analysis run ID mismatch")
    require(
        digest(directory / "operator_calls.json", hashes) == analysis["calls_sha256"],
        "Ledger SHA mismatch",
    )
    require(
        digest(directory / "request.json", hashes) == result["request_sha256"],
        "Request SHA mismatch",
    )
    for name, expected in result["source_sha256"].items():
        require(
            digest(directory / "source" / name, hashes) == expected,
            f"Source snapshot SHA mismatch: {name}",
        )
    require(not analysis["calls_outside_selected_captures"], "Ledger has uncaptured calls")
    expected_captures = {(mode, phase) for mode in MODES for phase in PHASES}
    require(
        len(analysis["captures"]) == 4
        and {(row["mode"], row["phase"]) for row in analysis["captures"]} == expected_captures,
        "Expected one complete capture for each mode and phase",
    )
    coverage = []
    for capture in analysis["captures"]:
        audit = capture["audit"]
        require(
            audit["kernel_count_and_time_conserved"] and audit["metadata_call_counts_match"],
            "Attribution audit failed",
        )
        require(not audit["layer_unscoped_kernel_count"], "Unscoped kernels remain")
        require(
            all(
                not item["unattributed_count"] for item in audit["activity_counts_and_ns"].values()
            ),
            "Unattributed GPU work remains",
        )
        sqlite = directory / Path(capture["sqlite"]).name
        require(
            digest(sqlite, hashes) == capture["sqlite_sha256"], f"SQLite SHA mismatch: {sqlite}"
        )
        coverage.append(
            {
                "mode": capture["mode"],
                "phase": capture["phase"],
                "sqlite": str(sqlite),
                "sqlite_sha256": capture["sqlite_sha256"],
                "audit": audit,
                "devices": [
                    {k: v for k, v in device.items() if k != "largest_gaps"}
                    for device in capture["devices"]
                ],
                "capture_gpu_envelope": capture["capture_gpu_envelope"],
                "api_summary": capture["api_summary"],
            }
        )
    intervals = []
    for mode, phase in sorted(expected_captures):
        start = 0 if phase == PHASES[0] else result["prefix_tokens"]
        count = result["prefix_tokens"] if phase == PHASES[0] else result["extend_tokens"]
        for layer in range(3):
            for category, stages in (
                ("indexer", {"indexer_qk", "indexer_fused"}),
                ("mla", {"mla_qk_pv"}),
            ):
                calls = [
                    r
                    for r in ledger["calls"]
                    if (r["mode"], r["phase"], r["layer"]) == (mode, phase, f"layer_{layer}")
                    and r["stage"] in stages
                ]
                cursor = start
                for call in sorted(calls, key=lambda row: row["query_start"]):
                    require(
                        call["query_start"] == cursor and call["query_tokens"] > 0,
                        "Query interval has a gap or duplicate",
                    )
                    cursor += call["query_tokens"]
                require(cursor == start + count, "Capture does not cover every model query")
                intervals.append(
                    {
                        "mode": mode,
                        "phase": phase,
                        "layer": layer,
                        "operator": category,
                        "calls": len(calls),
                        "query_tokens": count,
                    }
                )
    return coverage, intervals


def _enrich(rows, calls):
    grouped = defaultdict(list)
    for call in calls:
        grouped[_identity(call)].append(call)
        grouped[(call["mode"], call["phase"], "all_layers", call["stage"])].append(call)
    output = []
    for original in rows:
        row = dict(original)
        group = grouped[_identity(row)]
        require(len(group) == row["metadata_call_count"], "Operator ledger count mismatch")
        if row["useful_flops"] is not None:
            require(
                sum(call["useful_flops"] for call in group) == row["useful_flops"],
                "Operator FLOPs mismatch",
            )
        padded = [call.get("executed_matmul_flops") for call in group]
        row["executed_matmul_flops"] = (
            sum(padded) if padded and all(x is not None for x in padded) else None
        )
        row["executed_matmul_utilization_percent"] = (
            row["executed_matmul_flops"] / row["kernel_ns"] / row["dense_peak_tflops"] / 10
            if row["executed_matmul_flops"] is not None
            and row["kernel_ns"]
            and row["dense_peak_tflops"]
            else None
        )
        for field in ("formula", "executed_formula", "notes"):
            row[field] = " | ".join(sorted({call[field] for call in group if call.get(field)}))
        output.append(row)
    return output


def _compact_ncu(run):
    output = {
        key: run[key]
        for key in (
            "run_id",
            "analysis_path",
            "analysis_sha256",
            "analyzer_sha256",
            "full_source_metadata_match",
        )
    }
    output["captures"] = {}
    for label, capture in run["captures"].items():
        meta = capture["metadata"]
        output["captures"][label] = {
            "metadata": {
                key: meta[key]
                for key in (
                    "kernel",
                    "layer",
                    "query_start",
                    "tensor_metadata",
                    "measurement_boundary",
                    "verification",
                    "input_sha256",
                    "source_sha256",
                    "selected_launches_inside_profiler_api_range",
                )
            },
            "metadata_sha256": capture["metadata_sha256"],
            "native_report": capture["native_report"],
            "native_report_sha256": capture["native_report_sha256"],
            "validation": capture["validation"],
            "actions": [
                {
                    "name": action["name"],
                    "metrics": {name: action["metrics"].get(name) for name in NCU_METRICS},
                    "extraction_errors": action["extraction_errors"],
                }
                for action in capture["actions"]
            ],
        }
    return output


def _plot(directory, summary, matrix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"FP8": "#376ab0", "BF16": "#218a81", "FP32": "#bd7428"}
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), layout="constrained")
    common_limit = max(1, max(row["kernel_mfu_percent"] or 0 for row in matrix)) * 1.23
    for i, mode in enumerate(MODES):
        for j, phase in enumerate(PHASES):
            ax = axes[i, j]
            rows = sorted(
                [r for r in matrix if (r["mode"], r["phase"]) == (mode, phase)],
                key=lambda r: STAGES.index(r["stage"]),
            )
            values = [r["kernel_mfu_percent"] or 0 for r in rows]
            bars = ax.barh(
                [r["stage"] for r in rows], values, color=[colors[r["precision"]] for r in rows]
            )
            ax.bar_label(
                bars,
                labels=[
                    "N/A" if r["kernel_mfu_percent"] is None else f"{r['kernel_mfu_percent']:.2f}%"
                    for r in rows
                ],
                padding=4,
                fontsize=9,
            )
            ax.invert_yaxis()
            ax.set_xlim(0, common_limit)
            ax.set_xlabel("Useful matrix FLOPs / (GPU kernel time × dense peak), %")
            tokens = summary["prefix_tokens"] if phase == PHASES[0] else summary["extend_tokens"]
            ax.set_title(
                f"{mode.title()} · {tokens:,}-token {'prefix' if phase == PHASES[0] else 'extend'}"
            )
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
    fig.suptitle(f"Checkpoint layers 0–2: measured operator MFU\n{summary['run_id']}", fontsize=14)
    fig.legend(
        handles=[Patch(color=color, label=dtype) for dtype, color in colors.items()],
        loc="outside lower center",
        ncols=3,
    )
    for ext in ("svg", "png"):
        fig.savefig(directory / f"mfu.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, phase in zip(axes, ("prefix", "extend")):
        medians = [summary["measurements"][mode][f"{phase}_median_ms"] for mode in MODES]
        bars = ax.bar(MODES, medians, color=[colors["FP8"], colors["BF16"]], width=0.58)
        ax.bar_label(bars, fmt="%.2f ms", padding=5)
        for i, mode in enumerate(MODES):
            samples = summary["measurements"][mode][f"{phase}_samples_ms"]
            ax.scatter([i] * len(samples), samples, color="#212121", s=16, zorder=3)
        ax.set_ylim(0, max(medians) * 1.2)
        ax.set_title(
            f"{phase.title()} · {summary['prefix_tokens'] if phase == 'prefix' else summary['extend_tokens']:,} tokens"
        )
        ax.set_ylabel("Uninstrumented synchronized wall time (ms)")
    fig.suptitle(f"Checkpoint layers 0–2 · median and individual samples\n{summary['run_id']}")
    for ext in ("svg", "png"):
        fig.savefig(directory / f"latency.{ext}", dpi=180)
    plt.close(fig)


def _format(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _markdown(summary, matrix, nonmatrix):
    lines = [
        f"## 前三层重新测量：`{summary['run_id']}`",
        "",
        (
            "完整 checkpoint 的第 0–2 层依次传播 hidden/residual，包含 embedding、final norm 和最后 token LM head。"
            f"Prefix={summary['prefix_tokens']:,}，extend={summary['extend_tokens']:,}，chunk={summary['chunk_size']:,}，"
            f"offload pool={summary['slots']:,} tokens/层。仅代表前三层。"
        ),
        "",
        "| 阶段 | Resident 中位延迟 (ms) | Offload 中位延迟 (ms) | 每种模式重复次数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    if "wall_time_denominator" in summary:
        denominator = summary["wall_time_denominator"]
        lines[4:4] = [
            (
                f"延迟与利用率分母来自独立 bench `{denominator['run_id']}`；"
                f"其 result SHA256 为 `{denominator['result_sha256']}`，源码清单见 summary.json。"
            ),
            "",
        ]
    for phase in ("prefix", "extend"):
        lines.append(
            f"| {phase} | {_format(summary['measurements']['resident'][phase + '_median_ms'])} | {_format(summary['measurements']['offload'][phase + '_median_ms'])} | {len(summary['measurements']['resident'][phase + '_samples_ms'])} |"
        )
    lines += [
        "",
        "![前三层无插桩延迟](report/layers3/latency.svg)",
        "",
        (
            "端到端精度归一化利用率 = 100 × Σ精度（useful matrix FLOPs / 对应精度 dense peak）/ "
            "无插桩同步 wall time。分子使用同一 run、同一阶段完整 annotated 调用账本中的"
            "逻辑矩阵工作量，按 FP8/BF16/FP32 分别换算理想计算时间；新 schema 的分母来自匹配的独立 bench，包含整个请求阶段的"
            "CPU 调度、非矩阵计算、搬运、等待和 launch gap。它是当前 absorbed-MLA 实现的"
            "三层工作负载指标，不外推完整 61 层，也不等于单一峰值 MFU 或 Tensor pipe active。"
        ),
        "",
        "| 阶段 | 理想矩阵计算时间 (ms) | Resident 端到端利用率 (%) | Offload 端到端利用率 (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for phase in ("prefix", "extend"):
        left = summary["end_to_end_utilization"]["resident"][phase]
        right = summary["end_to_end_utilization"]["offload"][phase]
        lines.append(
            f"| {phase} | {_format(left['ideal_compute_ms'])} | "
            f"{_format(left['utilization_at_median_wall_percent'], 2)} | "
            f"{_format(right['utilization_at_median_wall_percent'], 2)} |"
        )
    lines += [
        "",
        (
            "表中利用率由中位 wall time 计算；每次重复的比率、分精度 FLOPs 与理想计算时间见 "
            "[summary.json](report/layers3/summary.json) 的 `end_to_end_utilization`。"
        ),
        "",
        "下面保留 `operator_mfu` 数据字段名；其含义是算子 kernel 利用率 = useful matrix FLOPs /"
        "（同算子实际 GPU kernel duration 之和 × 对应精度 dense peak）。"
        "FMA 计 2 FLOPs；FP8/BF16/FP32 分母分别为 "
        + "/".join(str(summary["dense_peaks_tflops"][key]) for key in ("FP8", "BF16", "FP32"))
        + " TFLOPS；TF32 关闭。"
        "前缀汇总全部 chunk 与三层，extend 汇总三层的一次完整 query batch（含实际 offload leaf 拆分）。"
        "LM head 每个阶段仅执行最后一个 token。非矩阵算子的 MFU 为 N/A。",
        "",
        (
            "每种模式、每个阶段各采集一次独立 annotated profile；算子 MFU 没有重复采样置信区间。"
            "FP8Linear 的分母包含量化和 GEMM，indexer 包含 logits API 的辅助 kernel；"
            "这些是按完整算子口径计算的 MFU，与单 GEMM 或 Tensor pipe active 指标不同。"
            "算术与归因检查不证明计算实现或 cache 策略合理，MFU 及性能解释须独立验收。"
        ),
        "",
        "![四组逐算子 MFU](report/layers3/mfu.svg)",
        "",
    ]
    for phase in PHASES:
        lines += [
            f"### {'Prefix' if phase == PHASES[0] else 'Extend'} 矩阵算子",
            "",
            "| 算子 | 精度 | Resident kernel ms | MFU (%) | Offload kernel ms | MFU (%) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        lookup = {(r["mode"], r["stage"]): r for r in matrix if r["phase"] == phase}
        for stage in STAGES:
            left, right = lookup.get(("resident", stage), {}), lookup.get(("offload", stage), {})
            lines.append(
                f"| {stage} | {left.get('precision', right.get('precision', 'N/A'))} | {_format(left.get('kernel_ms'))} | {_format(left.get('kernel_mfu_percent'), 2)} | {_format(right.get('kernel_ms'))} | {_format(right.get('kernel_mfu_percent'), 2)} |"
            )
        lines += [
            "",
            (
                "indexer_qk 记录 resident logits 调用，indexer_fused 包含融合 prefetch。"
                "同一 offload 阶段可包含两种调用，分别列出；MLA 行共同计入 QK/PV。"
            ),
            "",
            (
                "下表按最内层 scope 归因；attention_projection、dense_mlp 等父 scope 仅保留"
                "未归入子算子的剩余 kernel。这里只列 kernel 时间，cache_write 的 memcpy、"
                "CPU API 等另见完整数据，0 ms kernel 不表示没有搬运或同步开销。"
            ),
            "",
            "| 非矩阵 scope（MFU N/A） | Resident kernel ms | Offload kernel ms |",
            "| --- | ---: | ---: |",
        ]
        lookup = {(r["mode"], r["stage"]): r for r in nonmatrix if r["phase"] == phase}
        for stage in sorted({key[1] for key in lookup}):
            lines.append(
                f"| {stage} | {_format(lookup.get(('resident', stage), {}).get('kernel_ms'))} | {_format(lookup.get(('offload', stage), {}).get('kernel_ms'))} |"
            )
        lines.append("")
    lines += [
        (
            "GPU kernel 时间、CPU API 时间、NVTX host 区间、无插桩 wall time 分别保存，不相加为端到端分解。"
            "Padding 工作另列 `executed_matmul_flops`，不计入 useful MFU；cuBLAS 内部 padding 未知，保留空值。"
        ),
        "",
        (
            "正确性与覆盖验收、硬件 identity、源码及输入 SHA256 见 "
            "[summary.json](report/layers3/summary.json)。逐层及 pooled 数据见 "
            "[operator_mfu.csv](report/layers3/operator_mfu.csv)、"
            "[operator_mfu_by_layer.csv](report/layers3/operator_mfu_by_layer.csv)、"
            "[nonmatrix.csv](report/layers3/nonmatrix.csv)。"
        ),
        "",
        (
            "完整 NCU full/source 的验证记录见 [summary.json](report/layers3/summary.json)。"
            "NCU 是同 run 对应真实层激活的独立 replay，不替代 NSYS 实际调用的 MFU 或正式 wall 延迟；"
            "其 cache 状态、排除项和报告 SHA256 单独保存。NCU run IDs："
            + ", ".join(f"`{run['run_id']}`" for run in summary["ncu"])
            + "。"
            if summary["ncu"]
            else "本次未采集 NCU replay；报告不包含 NCU full/source 指标或验证结论。"
        ),
        "",
        "生成命令：",
        "",
        "```bash",
        "python -m experiments.deepseek_v32_mfu.src.publish_layers "
        f"--run-id {summary['run_id']} "
        + " ".join(f"--ncu-run-id {run['run_id']}" for run in summary["ncu"])
        + " --publish",
        "```",
        "",
    ]
    # Keep this standalone artifact navigable; README embedding adds its directory prefix.
    return "\n".join(lines).replace("](report/layers3/", "](")


def generate(run_id, ncu_run_ids, *, publish=False):
    directory = run_directory(run_id)
    destination = directory / "publication"
    report = EXPERIMENT / "report" / "layers3"
    require(not destination.exists() or publish, f"Publication already exists: {destination}")
    require(not publish or not report.exists(), f"Report destination already exists: {report}")
    require(len(set(ncu_run_ids)) == len(ncu_run_ids), "Duplicate NCU run IDs")
    hashes = {}
    result = read_json(directory / "result.json")
    if result.get("schema_version", 1) == 2:
        require(result.get("mode") == "profile", "Formal report requires an independent profile")
        validate_result(directory, result)
        result = benchmark_view(directory, result)
    analysis = read_json(directory / "analysis" / "analysis.json")
    ledger = read_json(directory / "operator_calls.json")
    coverage, intervals = _verify(directory, result, analysis, ledger, hashes)
    rows = _enrich(analysis["operators"], ledger["calls"])
    by_layer = _enrich(analysis["operators_by_layer"], ledger["calls"])
    ncu = [collect_ncu(run_directory(name), run_id, hashes) for name in ncu_run_ids]
    for run in ncu:
        for capture in run["captures"].values():
            for name, expected in capture["metadata"]["source_sha256"].items():
                if name in result["source_sha256"]:
                    require(
                        result["source_sha256"][name] == expected,
                        f"NCU/model source differs: {name}",
                    )
    ncu_rows = [
        {
            "source_run_id": run_id,
            "ncu_run_id": run["run_id"],
            "capture": label,
            "kernel": capture["metadata"]["kernel"],
            "metric": name,
            **metric,
        }
        for run in ncu
        for label, capture in run["captures"].items()
        for action in capture["actions"]
        for name, metric in action["metrics"].items()
    ]
    summary = {key: value for key, value in result.items() if key != "hardware"}
    summary.update(
        hardware_identity=result["hardware"],
        dense_peaks_tflops=analysis["dense_peaks_tflops"],
        peak_reference=analysis["peak_reference"],
        coverage=coverage,
        query_coverage=intervals,
        end_to_end_utilization=_end_to_end_utilization(
            result, ledger, analysis["dense_peaks_tflops"]
        ),
        ncu=[_compact_ncu(run) for run in ncu],
        measurement_definitions=analysis["notes"],
        provenance={
            "result_sha256": digest(directory / "result.json", hashes),
            "analysis_sha256": digest(directory / "analysis" / "analysis.json", hashes),
            "calls_sha256": digest(directory / "operator_calls.json", hashes),
            "publisher_sha256": digest(__file__, hashes),
            "execution_utilization_sha256": digest(
                Path(__file__).with_name("execution_utilization.py"), hashes
            ),
            "nsys_analyzer_sha256": analysis["analyzer_sha256"],
            "attribution_source_sha256": analysis["attribution_source_sha256"],
            "source_snapshots_verified": True,
        },
    )
    for mode in MODES:
        for phase in ("prefix", "extend"):
            samples = result["measurements"][mode][phase + "_samples_ms"]
            require(
                samples and all(math.isfinite(x) and x > 0 for x in samples),
                "Invalid latency samples",
            )
            require(
                statistics.median(samples) == result["measurements"][mode][phase + "_median_ms"],
                "Latency median mismatch",
            )
    matrix = [row for row in rows if row["useful_flops"] is not None]
    nonmatrix = [row for row in rows if row["useful_flops"] is None]
    audit_path = directory / "postrun_audit.json"
    if audit_path.is_file():
        audit = read_json(audit_path)
        require(
            audit["run_id"] == run_id and audit["accepted"] is True, "Post-run audit not accepted"
        )
        summary["provenance"]["postrun_audit_sha256"] = digest(audit_path, hashes)
    if destination.exists():
        existing = read_json(destination / "summary.json")
        require(
            existing["provenance"] == summary["provenance"],
            "Existing publication input/source identity differs",
        )
        require(
            [run["run_id"] for run in existing["ncu"]] == ncu_run_ids,
            "Existing publication NCU selection differs",
        )
        for name, expected in existing["artifact_sha256"].items():
            require(
                digest(destination / name, hashes) == expected,
                f"Publication artifact SHA mismatch: {name}",
            )
        shutil.copytree(destination, report)
        return {
            "run_id": run_id,
            "publication": str(destination),
            "published": True,
            "reused_verified_publication": True,
        }
    with tempfile.TemporaryDirectory(prefix="echo-publication-") as temp:
        staging = Path(temp)
        for name, selected in (
            ("operator_mfu", matrix),
            ("operator_mfu_by_layer", [r for r in by_layer if r["useful_flops"] is not None]),
            ("nonmatrix", nonmatrix),
            ("nonmatrix_by_layer", [r for r in by_layer if r["useful_flops"] is None]),
            ("operator_inventory", rows),
            ("ncu_metrics", ncu_rows),
        ):
            write_csv(staging / f"{name}.csv", [{"run_id": run_id, **row} for row in selected])
        _plot(staging, summary, matrix)
        (staging / "results.md").write_text(_markdown(summary, matrix, nonmatrix))
        if audit_path.is_file():
            shutil.copyfile(audit_path, staging / audit_path.name)
        summary["artifact_sha256"] = {
            path.name: digest(path, hashes) for path in sorted(staging.iterdir())
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        shutil.copytree(staging, destination)
    if publish:
        shutil.copytree(destination, report)
    return {
        "run_id": run_id,
        "publication": str(destination),
        "ncu_runs": len(ncu),
        "matrix_rows": len(matrix),
        "nonmatrix_rows": len(nonmatrix),
        "published": publish,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ncu-run-id", action="append", default=[])
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    status = generate(args.run_id, args.ncu_run_id, publish=args.publish)
    print(json.dumps(status))


if __name__ == "__main__":
    main()
