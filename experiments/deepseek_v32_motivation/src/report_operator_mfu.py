"""Publish the operator-MFU tables from the existing motivation captures."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from experiments.deepseek_v32_motivation.src.analyze_pipeline import sha, write_json

SCHEMES = ("hbm", "echo", "serial_sparse", "dense_prefetch")
STAGES = (
    "q_a_proj",
    "q_b_proj",
    "q_absorb",
    "kv_a_proj",
    "index_q_proj",
    "index_k_proj",
    "index_weights_proj",
    "indexer",
    "mla_qk_pv",
    "v_expand",
    "o_proj",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "lm_head",
)


def render_report(summary, rows, *, calls_path=None):
    """Render current measured groups without historical counts or conclusions."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        stage = "indexer" if row["stage"] in ("indexer_qk", "indexer_fused") else row["stage"]
        grouped[row["phase"], row["segment"], stage, row["precision"]].append(row)
    schemes = [scheme for scheme in SCHEMES if any(r["scheme"] == scheme for r in rows)]
    lines = [
        "# 算子 MFU",
        "",
        f"诊断采集为 `{summary['profile_run_id']}`，对应正式运行 `{summary['formal_run_id']}`。",
        f"本报告处理 {len(summary['captures'])} 个请求 capture、{summary['matrix_calls']:,} 次矩阵调用。",
        "算子时间取每次 API 所属 GPU 活动的时间并集，包括内部量化、转置、归约和搬运，再按阶段累计。",
        "MFU 为累计理论计算时间除以累计实测时间；理论时间按每次调用的有效 FLOPs 和对应精度峰值计算。",
        "CPU 提交、API 外的 cache 管理和 GPU 发射间隙仍保留在端到端时间中，不计入此处的算子分母。",
        "",
    ]
    graphs = [capture.get("graph_attribution") for capture in summary["captures"]]
    replays = sum(item["replays"] for item in graphs if item)
    if replays:
        nodes = sum(item["gpu_node_activities"] for item in graphs if item)
        lines += [
            f"其中 {replays:,} 次 CUDA Graph 重放共记录 {nodes:,} 个 GPU 节点活动。",
            "图内 API 用捕获时记录的节点集合归属；实例化克隆关系取自 Nsight 的 originalGraphNodeId。",
            "分母使用本次重放的实际节点时间，未借用 eager 运行时间。图内 API 行不声明独立 CPU 范围。",
            "",
        ]
    peaks = ", ".join(
        f"{name.upper()}={value:g}" for name, value in summary["dense_peaks_tflops"].items()
    )
    lines += [f"参考 dense 峰值（TFLOPS）：{peaks}。精度设置和峰值来源保存在汇总 JSON。", ""]
    for phase, segment in sorted({(r["phase"], r["segment"]) for r in rows}):
        lines += [
            f"## {phase} / {segment}",
            "",
            "| 算子 | 精度 | " + " | ".join(schemes) + " |",
            "|---|---|" + "---:|" * len(schemes),
        ]
        for (row_phase, row_segment, stage, precision), group in sorted(grouped.items()):
            if (row_phase, row_segment) != (phase, segment):
                continue
            values = []
            for scheme in schemes:
                selected = [r for r in group if r["scheme"] == scheme]
                ideal = sum(float(r["ideal_ms"]) for r in selected)
                duration = sum(float(r["operator_gpu_active_ms"]) for r in selected)
                values.append(f"{100 * ideal / duration:.2f}%" if duration else "—")
            lines.append(f"| {stage} | {precision} | " + " | ".join(values) + " |")
        lines.append("")
    lines += [
        "Indexer 一行按实际 resident/fused 调用合并，使用总理论时间除以总 GPU 活动时间。",
        "融合 indexer 的矩阵 kernel 同时包含预取和标量工作，当前 trace 不能把这些工作拆开。",
        "主 kernel 仅计矩阵入口；split-K reduction、scale 转置等辅助节点保留在完整 API 时间中。",
        "",
        "每方案采集一个首访和一个复访请求。不同 history chunk 的上下文长度不同，不能当作同形状算子的独立重复测量。",
        "本报告反映诊断进程中的算子表现；正式端到端 MFU 使用对应未插桩运行的请求延迟。",
        "",
        "[完整数据](operator_mfu/operator_mfu.csv)、[逐层数据](operator_mfu/operator_mfu_by_layer.csv)、",
        "[kernel 清单](operator_mfu/operator_kernel_inventory.csv)、[汇总与定义](operator_mfu/operator_mfu_summary.json)、",
        "[独立复核](operator_mfu/crosscheck.json)和[发布来源](operator_mfu/publication.json)保留核验信息。",
        "",
        f"分析 ID：`{summary['analysis_run_id']}`。"
        + (
            f"逐调用数据位于 `{calls_path}`。"
            if calls_path is not None
            else "逐调用数据见分析目录中的 `operator_mfu_calls.jsonl`。"
        ),
        "",
    ]
    return "\n".join(lines)


def publish(analysis, output):
    summary = json.loads((analysis / "operator_mfu_summary.json").read_text())
    check = json.loads((analysis / "crosscheck.json").read_text())
    if (
        not check["passed"]
        or summary["matrix_calls"] != check["matrix_calls"]
        or summary["matrix_calls"] <= 0
    ):
        raise ValueError("complete verified operator-MFU analysis is required")
    for name, expected in summary["analysis_source_sha256"].items():
        if sha(name) != expected:
            raise ValueError(f"analysis source changed: {name}")
    rows = list(csv.DictReader((analysis / "operator_mfu.csv").open()))
    if sum(int(row["calls"]) for row in rows) != summary["matrix_calls"]:
        raise ValueError("operator rows do not conserve actual matrix calls")
    if sum(int(row["primary_kernel_count"]) for row in rows) != check["primary_kernels"]:
        raise ValueError("independent primary-kernel count check failed")
    if sum(int(row["gpu_activity_count"]) for row in rows) != check["matrix_scope_gpu_activities"]:
        raise ValueError("independent activity-count check failed")
    for expected in check["independent_spot_checks"]:
        row = next(
            row
            for row in rows
            if all(row[key] == expected[key] for key in ("capture_index", "segment", "stage"))
        )
        for column, field in (
            ("primary_kernel_mfu_pct", "primary_mfu_pct"),
            ("operator_gpu_active_mfu_pct", "gpu_active_mfu_pct"),
        ):
            if abs(float(row[column]) - expected[field]) > check["tolerance_percentage_points"]:
                raise ValueError("independent direct-SQL MFU check failed")
    text = render_report(summary, rows, calls_path=analysis / "operator_mfu_calls.jsonl")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "operator_mfu"
    assets.mkdir(exist_ok=True)
    selected = (
        "operator_mfu.csv",
        "operator_mfu_by_layer.csv",
        "operator_kernel_inventory.csv",
        "operator_mfu_summary.json",
        "crosscheck.json",
    )
    for name in selected:
        shutil.copyfile(analysis / name, assets / name)
    (output / "operator_mfu.md").write_text(text)
    write_json(
        assets / "publication.json",
        {
            "analysis_run_id": summary["analysis_run_id"],
            "profile_run_id": summary["profile_run_id"],
            "report_generator": str(Path(__file__).resolve()),
            "report_generator_sha256": sha(__file__),
            "published_sha256": {name: sha(assets / name) for name in selected},
            "report_sha256": sha(output / "operator_mfu.md"),
            "input_sha256": {name: sha(analysis / name) for name in selected},
        },
    )
    print(f"Published operator MFU: {output / 'operator_mfu.md'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    publish(args.analysis_dir, args.output_dir)


if __name__ == "__main__":
    main()
