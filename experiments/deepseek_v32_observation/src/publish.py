"""Publish complete, receipt-bound device-resident operator observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt
from experiments.deepseek_v32_observation.src.bandwidth import HBM_FIELDS, validate_bandwidth
from experiments.deepseek_v32_observation.src.contract import (
    GROUPS,
    PEAK_CONVENTION,
    PEAK_SOURCE,
    PEAKS_TFLOPS,
    S_VALUES,
    matrix,
    statistics,
    validate_result,
)
from experiments.deepseek_v32_observation.src.provenance import require_runtime_match

EXPERIMENT = Path(__file__).resolve().parents[1]
FIELDS = (
    "source_run_id",
    "group",
    "source_case_id",
    "target_shape_id",
    "measurement_kind",
    "B",
    "Q",
    "H",
    "S",
    "name",
    "scope",
    "operator_kind",
    "operator_family",
    "parent_api",
    "precision",
    "useful_flops",
    "formula",
    "time_us",
    "mean_us",
    "min_us",
    "max_us",
    "q1_us",
    "q3_us",
    "sample_count",
    "mfu_pct",
    "peak_tflops",
    "precision_flops_json",
    "precision_peaks_tflops_json",
    "mfu_na_reason",
    "timing_boundary",
    "metadata_json",
    "work_json",
    "timing_json",
    "identity_sha256",
    "receipt_file_sha256",
    *HBM_FIELDS,
)
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=_reject_constant)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing nonempty {field}")
    return value


def _number(value, field, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid finite {field}")
    if positive and value == 0:
        raise ValueError(f"Expected positive {field}")


def clean_entry(entry):
    """Remove only known profile fields before comparing the original result."""
    return {
        **entry,
        "operators": [
            {key: value for key, value in operator.items() if key not in HBM_FIELDS}
            for operator in entry["operators"]
        ],
    }


def validate_profile(entry, child_directory, receipt):
    path = child_directory / "hbm.json"
    profile = read(path)
    for field in ("group", "key", "identity", "receipt_path"):
        if profile.get(field) != entry.get(field):
            raise ValueError(f"Independent HBM profile differs from clean workload {field}")
    runtime = profile.get("runtime")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("Every HBM profile needs observed runtime/backend metadata")
    if runtime.get("uuid") != entry["runtime"].get("uuid"):
        raise ValueError("HBM profile GPU differs from its clean workload")
    if "runtime" not in receipt["artifact_paths"]:
        raise ValueError("HBM profile requires independently accepted runtime artifacts")
    checked_runtime = read(receipt["artifact_paths"]["runtime"])
    if checked_runtime.get("uuid") != runtime.get("uuid"):
        raise ValueError("HBM profile GPU differs from its independently accepted runtime")
    runtime_match = require_runtime_match(checked_runtime, runtime)
    observed = {operator["name"]: operator for operator in profile["operators"]}
    names = {operator["name"] for operator in entry["operators"]}
    if len(observed) != len(profile["operators"]) or set(observed) != names:
        raise ValueError("HBM profile must cover the exact clean operator inventory")
    artifacts = {}
    for operator in entry["operators"]:
        observation = observed[operator["name"]]
        validate_bandwidth(operator)
        validate_bandwidth(observation)
        if any(operator[field] != observation[field] for field in HBM_FIELDS):
            raise ValueError("Aggregate HBM metrics differ from independent profile")
        source = Path(observation["hbm_profile_path"]).resolve(strict=True)
        actual_sha256 = digest(source)
        if actual_sha256 != observation["hbm_profile_sha256"]:
            raise ValueError("NCU profile artifact content changed")
        artifacts[str(source)] = {"sha256": actual_sha256, "bytes": source.stat().st_size}
    return {
        "profile_result_path": str(path.resolve()),
        "profile_result_sha256": digest(path),
        "profile_runtime": runtime,
        "profile_runtime_match": runtime_match,
        "profile_artifacts": artifacts,
    }


def validate_operator_metadata(operator):
    metadata = operator["metadata"]
    kind = metadata.get("operator_kind")
    if kind not in {"gemm", "non_gemm", "api_aggregate"}:
        raise ValueError("Every operator requires an explicit GEMM/non-GEMM/API-aggregate kind")
    _text(metadata.get("operator_family"), "operator_family")
    _text(metadata.get("parent_api"), "parent_api")
    work = operator["work"]
    if kind == "non_gemm" and (
        work.get("useful_flops") is not None or operator.get("mfu_pct") is not None
    ):
        raise ValueError("Non-GEMM operators must retain N/A matrix FLOPs and MFU")
    if kind == "gemm" and work.get("useful_flops") is None:
        raise ValueError("GEMM leaves need their actual matrix FLOPs")
    if kind == "api_aggregate" and operator["scope"] == "operator":
        raise ValueError("Aggregate APIs must retain an explicit aggregate scope")
    if kind != "api_aggregate" and operator["scope"] != "operator":
        raise ValueError("Concrete leaves must use operator scope")


def validate_publication(result, results_path):
    """Reject incomplete shapes, altered arithmetic, or incompatible acceptance."""
    validate_result(result)
    _text(result.get("run_id"), "run_id")
    expected_shapes = {shape.key: shape.as_dict() for shape in matrix()}
    shapes = result.get("matrix")
    if not isinstance(shapes, list) or len(shapes) != len(expected_shapes):
        raise ValueError("Publication requires the complete 52-point B/Q/H matrix")
    if {shape.get("key"): shape for shape in shapes} != expected_shapes:
        raise ValueError("Result matrix differs from the frozen B/Q/H contract")
    if result.get("peaks_tflops") != PEAKS_TFLOPS or result.get("peak_source") != PEAK_SOURCE:
        raise ValueError("Publication arithmetic peaks or source differ from the contract")

    evidence = []
    execution_binding = None
    for group in GROUPS:
        for entry in result["groups"][group]:
            if entry.get("group") != group:
                raise ValueError("Workload entry group differs from its containing group")
            if group == "attention":
                shape = expected_shapes[entry["key"]]
                if any(entry.get(field) != shape[field] for field in ("B", "Q", "H", "S")):
                    raise ValueError("Attention B/Q/H/S differs from its workload key")
            elif (
                type(entry.get("S")) is not int
                or entry["S"] not in S_VALUES
                or entry["key"] != f"s{entry['S']}"
                or any(entry.get(field) is not None for field in ("B", "Q", "H"))
            ):
                raise ValueError("Common/MoE S or non-attention dimensions differ from the key")
            if not isinstance(entry.get("runtime"), dict) or not entry["runtime"]:
                raise ValueError("Every workload needs observed runtime/backend metadata")
            identity = entry["identity"]
            shared_identity = {
                field: identity.get(field)
                for field in (
                    "source_sha256",
                    "backend_sha256",
                    "checkpoint",
                    "cuda",
                    "precision_policy",
                    "environment",
                )
            }
            if execution_binding is None:
                execution_binding = shared_identity
            elif shared_identity != execution_binding:
                raise ValueError(
                    "Execution source/backend/checkpoint policy changed across workloads"
                )
            if identity.get("group") != group or identity.get("key") != entry["key"]:
                raise ValueError("Execution identity differs from its workload group/key")
            if not isinstance(identity.get("input"), dict) or not identity["input"]:
                raise ValueError("Execution identity needs a nonempty input identity")
            bound_operators = [
                {field: operator[field] for field in ("name", "work", "metadata")}
                for operator in entry["operators"]
            ]
            if identity.get("operators") != bound_operators:
                raise ValueError("Observed operator work/metadata differ from numerical acceptance")
            if identity.get("backend_sha256") != entry["runtime"].get("backend_sha256"):
                raise ValueError("Observed backend identity differs from numerical acceptance")
            child_result_path = Path(results_path).parent / group / entry["key"] / "result.json"
            if read(child_result_path) != clean_entry(entry):
                raise ValueError("Aggregate entry differs from its independent clean result")
            receipt = require_receipt(
                entry["receipt_path"],
                kind=f"deepseek-v32-observation-{group}",
                identity=entry["identity"],
            )
            receipt_file_sha256 = digest(receipt["receipt_path"])
            artifacts = {
                name: {
                    "path": path,
                    "sha256": digest(path),
                    "bytes": Path(path).stat().st_size,
                }
                for name, path in receipt["artifact_paths"].items()
            }
            profile_evidence = validate_profile(entry, child_result_path.parent, receipt)
            evidence.append(
                {
                    "group": group,
                    "key": entry["key"],
                    "identity_sha256": identity_digest(entry["identity"]),
                    "input_sha256": identity_digest(entry["identity"]["input"]),
                    "input_identity": entry["identity"]["input"],
                    "execution_identity": entry["identity"],
                    "runtime": entry["runtime"],
                    "clean_result_path": str(child_result_path.resolve()),
                    "clean_result_sha256": digest(child_result_path),
                    "receipt_path": receipt["receipt_path"],
                    "receipt_sha256": receipt["receipt_sha256"],
                    "receipt_file_sha256": receipt_file_sha256,
                    "acceptance_artifacts": artifacts,
                    **profile_evidence,
                }
            )
            for operator in entry["operators"]:
                _text(operator.get("name"), "operator name")
                _text(operator.get("scope"), "operator scope")
                _text(operator.get("timing_boundary"), "timing_boundary")
                if not isinstance(operator.get("metadata"), dict):
                    raise TypeError("Every operator needs explicit metadata")
                validate_operator_metadata(operator)
                if operator.get("profiler_enabled", False) is not False:
                    raise ValueError("Profiler timings cannot enter a clean publication")
                work = operator["work"]
                _text(work.get("formula"), "FLOPs formula")
                summary = statistics(operator["samples_us"])
                for field, value in summary.items():
                    if field in operator and not math.isclose(
                        operator[field], value, rel_tol=1e-12
                    ):
                        raise ValueError(f"Retained samples disagree with {field}")
                if work.get("useful_flops") is None:
                    if operator.get("peak_tflops") is not None:
                        raise ValueError("Non-matrix operators cannot have a matrix peak")
                else:
                    _text(work.get("precision"), "matrix arithmetic precision")
                    if work["precision"] == "mixed":
                        precision_flops = operator.get("precision_flops")
                        if not isinstance(precision_flops, dict) or not precision_flops:
                            raise ValueError(
                                "Mixed precision MFU needs explicit per-precision FLOPs"
                            )
                        if sum(precision_flops.values()) != work["useful_flops"]:
                            raise ValueError("Per-precision FLOPs do not cover mixed useful work")
                        if operator.get("peak_tflops") is not None:
                            raise ValueError("Mixed precision MFU has no single arithmetic peak")
                        expected_peaks = {
                            precision: PEAKS_TFLOPS[precision] for precision in precision_flops
                        }
                        if operator.get("precision_peaks_tflops") != expected_peaks:
                            raise ValueError("Mixed precision peaks differ from their arithmetic")
                        for precision, flops in precision_flops.items():
                            if work.get("dimensions", {}).get(f"{precision}_flops") != flops:
                                raise ValueError("Mixed FLOPs differ from accepted work dimensions")
                        expected_mfu = sum(
                            flops / (operator["time_us"] * 1e6 * PEAKS_TFLOPS[precision]) * 100
                            for precision, flops in precision_flops.items()
                        )
                        if not math.isclose(operator["mfu_pct"], expected_mfu, rel_tol=1e-12):
                            raise ValueError("Mixed precision MFU differs from ideal-time sum")
                    else:
                        peak = PEAKS_TFLOPS[work["precision"]]
                        if operator.get("peak_tflops") != peak:
                            raise ValueError("Operator arithmetic peak differs from its precision")
                    _number(operator["mfu_pct"], "matrix MFU", positive=True)
    return {
        "source_run_id": result["run_id"],
        "results_path": str(Path(results_path).resolve()),
        "results_sha256": digest(results_path),
        "peaks_tflops": result["peaks_tflops"],
        "peak_source": result["peak_source"],
        "peak_convention": result.get("peak_convention", PEAK_CONVENTION),
        "workloads": evidence,
        "source_sha256": execution_binding["source_sha256"],
        "backend_sha256": execution_binding["backend_sha256"],
        "metric_boundaries": {
            "clean": "Clean CUDA event API/leaf time; MFU uses this denominator",
            "hbm": "Independent NCU observed DRAM bytes and profile GPU time; cache-control none",
            "aggregation": "Concrete leaves and production API aggregates measured independently; not additive",
        },
    }


def rows(result, evidence):
    bindings = {(item["group"], item["key"]): item for item in evidence["workloads"]}
    output = {group: [] for group in GROUPS}
    for group in GROUPS:
        for entry in result["groups"][group]:
            binding = bindings[group, entry["key"]]
            for operator in entry["operators"]:
                work = operator["work"]
                timing = {
                    key: value
                    for key, value in operator.items()
                    if key not in ("name", "work", "metadata", "scope")
                }
                output[group].append(
                    {
                        "source_run_id": result["run_id"],
                        "group": group,
                        "source_case_id": entry["key"],
                        "target_shape_id": entry["key"] if group == "attention" else None,
                        "measurement_kind": "direct",
                        **{field: entry.get(field) for field in ("B", "Q", "H", "S")},
                        "name": operator["name"],
                        "scope": operator["scope"],
                        **{
                            field: operator["metadata"][field]
                            for field in ("operator_kind", "operator_family", "parent_api")
                        },
                        "precision": work.get("precision"),
                        "useful_flops": work.get("useful_flops"),
                        "formula": work["formula"],
                        **statistics(operator["samples_us"]),
                        "mfu_pct": operator["mfu_pct"],
                        "peak_tflops": operator.get("peak_tflops"),
                        "precision_flops_json": json.dumps(
                            operator.get("precision_flops"),
                            sort_keys=True,
                        ),
                        "precision_peaks_tflops_json": json.dumps(
                            operator.get("precision_peaks_tflops"),
                            sort_keys=True,
                        ),
                        "mfu_na_reason": operator.get("mfu_na_reason"),
                        "timing_boundary": operator["timing_boundary"],
                        "metadata_json": json.dumps(operator["metadata"], sort_keys=True),
                        "work_json": json.dumps(work, sort_keys=True),
                        "timing_json": json.dumps(timing, sort_keys=True),
                        "identity_sha256": binding["identity_sha256"],
                        "receipt_file_sha256": binding["receipt_file_sha256"],
                        **{
                            field: json.dumps(operator[field], sort_keys=True)
                            if field == "hbm_metrics"
                            else operator[field]
                            for field in HBM_FIELDS
                        },
                    }
                )
        output[group].sort(key=lambda row: (row["S"], row["Q"] or 0, row["H"] or 0, row["name"]))
    expanded = []
    for shape in matrix():
        for group in ("common", "moe"):
            for row in output[group]:
                if row["S"] == shape.S:
                    expanded.append(
                        {
                            **row,
                            "B": shape.B,
                            "Q": shape.Q,
                            "H": shape.H,
                            "target_shape_id": shape.key,
                            "measurement_kind": f"mapped_{group}",
                        }
                    )
        expanded.extend(row for row in output["attention"] if row["target_shape_id"] == shape.key)
    return output, expanded


def write_csv(path, records):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _plotting():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    import matplotlib.pyplot as plt

    return plt


def figure_groups(records):
    """Keep concrete functional families and measured aggregate APIs separate."""
    grouped = {}
    for row in records:
        family = row["operator_family"]
        if "projection" in family:
            family = "projection"
        if row["operator_kind"] == "api_aggregate":
            key = f"aggregate_{family}"
        else:
            key = family
        grouped.setdefault(key, []).append(row)
    priorities = {
        "projection": 0,
        "norm": 1,
        "rope": 2,
        "quantization": 3,
        "scale_layout": 4,
        "cast": 5,
        "scale": 6,
        "layout": 7,
        "moe_routing": 8,
        "moe_gemm": 9,
        "moe_activation": 10,
        "moe_cast": 11,
        "moe_reduce": 12,
    }
    return sorted(grouped.items(), key=lambda item: (priorities.get(item[0], 20), item[0]))


def _figure_family_title(family):
    names = {
        "projection": "Projection",
        "norm": "Normalization",
        "rope": "RoPE",
        "quantization": "Quantization",
        "scale_layout": "Scale layout",
        "cast": "Casts",
        "scale": "Scaling",
        "layout": "Layout and metadata",
        "moe_routing": "MoE routing",
        "moe_gemm": "MoE GEMM leaves",
        "moe_activation": "MoE activation",
        "moe_cast": "MoE casts",
        "moe_reduce": "MoE weighted reduction",
        "indexer": "Indexer",
        "attention": "Main attention",
        "mask": "Masks",
        "topk": "Exact top-k",
        "combine": "Attention combine",
        "dense_mlp": "Dense MLP",
        "lm_head": "LM head",
        "embedding": "Embedding",
    }
    aggregate = family.startswith("aggregate_")
    original = family.removeprefix("aggregate_")
    title = names.get(original, original.replace("_", " ").capitalize())
    return title + (" — non-additive API aggregates" if aggregate else " — concrete operators")


def operator_inventory(records):
    """Describe actual measured variants, without expanding mapped records."""
    inventory = {}
    for group, group_rows in records.items():
        for row in group_rows:
            key = (
                group,
                row["name"],
                row["operator_kind"],
                row["operator_family"],
                row["precision"],
                row["parent_api"],
            )
            entry = inventory.setdefault(key, {"rows": [], "conditions": set()})
            entry["rows"].append(row)
            metadata = json.loads(row["metadata_json"])
            if metadata.get("supported_condition"):
                entry["conditions"].add(metadata["supported_condition"])
    output = []
    for key, entry in sorted(inventory.items(), key=lambda item: tuple(str(x) for x in item[0])):
        group, name, kind, family, precision, parent = key
        values = entry["rows"]
        coverage = (
            "S=" + ",".join(str(s) for s in sorted({row["S"] for row in values}))
            if group != "attention"
            else f"{len({row['source_case_id'] for row in values})} B/Q/H points"
        )
        output.append(
            {
                "group": group,
                "name": name,
                "operator_kind": kind,
                "operator_family": family,
                "precision": precision or "N/A",
                "parent_api": parent,
                "coverage": coverage,
                "supported_condition": "; ".join(sorted(entry["conditions"])),
            }
        )
    return output


def render_inventory(inventory, *, aggregates):
    lines = [
        "| 分组 | 实际算子名称 | 类型 | 矩阵精度 | 所属 API | 直接测量覆盖 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in inventory:
        if (row["operator_kind"] == "api_aggregate") != aggregates:
            continue
        condition = f"; {row['supported_condition']}" if row["supported_condition"] else ""
        lines.append(
            f"| {row['group']}/{row['operator_family']} | `{row['name']}` | "
            f"{row['operator_kind']} | {row['precision']} | `{row['parent_api']}` | "
            f"{row['coverage']}{condition} |"
        )
    return lines


def plot_series(records, directory, *, group, horizontal, suffix=""):
    """Place clean time/MFU beside independent observed HBM bandwidth/utilization."""
    plt = _plotting()
    figures = []
    for family, family_rows in figure_groups(records):
        names = sorted({row["name"] for row in family_rows})
        # At most six operators per canvas keeps labels readable in Markdown.
        for page, start in enumerate(range(0, len(names), 6), start=1):
            selected_names = names[start : start + 6]
            columns = 4
            figure, axes = plt.subplots(
                len(selected_names),
                columns,
                figsize=(15.2, 2.7 * len(selected_names)),
                squeeze=False,
                layout="constrained",
            )
            metrics = (
                ("time_us", "Clean API time (µs)"),
                ("mfu_pct", "Clean useful matrix MFU (%)"),
                ("hbm_bandwidth_gbps", "NCU HBM bandwidth (GB/s)"),
                ("hbm_util_pct", "NCU HBM utilization (%)"),
            )
            for index, name in enumerate(selected_names):
                values = sorted(
                    (row for row in family_rows if row["name"] == name),
                    key=lambda row: row[horizontal],
                )
                for axis, (metric, ylabel) in zip(axes[index], metrics, strict=True):
                    eligible = [row for row in values if row[metric] is not None]
                    if eligible:
                        axis.plot(
                            [row[horizontal] for row in eligible],
                            [row[metric] for row in eligible],
                            color=PALETTE[index % len(PALETTE)],
                            marker=MARKERS[index % len(MARKERS)],
                            linewidth=1.5,
                            markersize=4,
                        )
                    else:
                        axis.text(
                            0.5,
                            0.5,
                            "N/A: non-GEMM",
                            ha="center",
                            va="center",
                            transform=axis.transAxes,
                        )
                    axis.set_title(name, fontsize=9)
                    axis.set_xscale("log", base=2)
                    ticks = sorted({row[horizontal] for row in values})
                    axis.set_xticks(
                        ticks,
                        [str(value) for value in ticks],
                        rotation=35 if horizontal == "S" else 0,
                    )
                    axis.set_xlabel(
                        "S = B × Q (log2)" if horizontal == "S" else "History tokens H (log2)"
                    )
                    axis.set_ylabel(ylabel, fontsize=9)
                    axis.set_ylim(bottom=0)
                    axis.margins(y=0.15)
                    axis.grid(axis="y", alpha=0.25)
            figure.suptitle(
                f"{group.capitalize()}: {_figure_family_title(family)}{suffix}", fontsize=12
            )
            stem = f"{group}{suffix.replace(' B=', '_b').replace(', Q=', '_q')}_{family}_{page}"
            for extension in ("svg", "png"):
                destination = directory / f"{stem}.{extension}"
                figure.savefig(destination, dpi=170)
                if extension == "svg":
                    destination.write_text(
                        "\n".join(line.rstrip() for line in destination.read_text().splitlines())
                        + "\n"
                    )
                figures.append(destination.name)
            plt.close(figure)
    return figures


def render_report(result, records, expanded, figures):
    counts = {group: len(result["groups"][group]) for group in GROUPS}
    operators = {group: len(records[group]) for group in GROUPS}
    leaves = [
        row
        for group_rows in records.values()
        for row in group_rows
        if row["operator_kind"] != "api_aggregate"
    ]
    aggregates = [
        row
        for group_rows in records.values()
        for row in group_rows
        if row["operator_kind"] == "api_aggregate"
    ]
    boundary = sorted(
        {row["timing_boundary"] for group_rows in records.values() for row in group_rows}
    )
    profile_boundary = sorted(
        {row["hbm_time_boundary"] for group_rows in records.values() for row in group_rows}
    )
    replay_modes = sorted(
        {row["hbm_replay_mode"] for group_rows in records.values() for row in group_rows}
    )
    inventory = operator_inventory(records)
    lines = [
        "# DeepSeek V3.2 device-resident operator observation",
        "",
        f"来源 run ID：`{result['run_id']}`。全部 {sum(counts.values())} 个工作负载完成匹配的独立数值验收、",
        "clean 计时与独立 NCU profile。公共算子与 MoE 各覆盖 7 个 S，attention 覆盖 52 个 B/Q/H 点。",
        "本报告测量具体 GEMM、非 GEMM 操作与另外列出的完整 API 聚合，不代表完整模型或 serving 延迟。",
        "",
        "Decode 使用 Q=1、B=[1,32,64,128,256,512,1024]；Extend 使用 B=1、",
        "Q=[32,64,128,256,512,1024]。H=[4096,16384,32768,65536]，S=B×Q。",
        "输入与 cache 均驻留设备，不执行 offload；权重来自 checkpoint，activation/KV 使用声明的合成输入。",
        "",
        "公共算子与 MoE 按 S 测量一次。`expanded.csv` 映射到对应 B/Q/H，`measurement_kind`",
        "与 `source_case_id` 保留原测量来源；映射不新增测量。",
        "",
        "## 指标与测量边界",
        "",
        "Clean 时间使用完整调用两端的 CUDA events，通过独立 CUDA Graph replay 测量。",
        "它包含调用内部 GPU 操作与间隙，不含 Python 提交、编译、数据生成和最终主机同步。",
        "MFU 使用对应具体矩阵工作的 useful FLOPs 与这一 clean 时间；FP8/BF16/FP32 的",
        "H200 SXM dense 峰值分别为 1979/989.5/67 TFLOP/s。非 GEMM 的矩阵 MFU 标为 N/A。",
        "混合精度 API 聚合先求各精度的理想矩阵时间之和，再除以完整 API 时间。",
        "",
        "HBM 指标来自另一次 NCU range/graph profile，`cache-control=none`。读取与写入",
        "字节均使用 DRAM 硬件计数器；每调用字节由 profile range 总字节除以该 range 的调用数得到。",
        "HBM bandwidth = 实测总字节 / profile range GPU 时间，utilization = bandwidth / NCU",
        "对应 HBM 峰值。它的时间分母与 clean 时间不同，不能把 NCU 时间替换进 clean MFU，",
        "也不能用 tensor 大小或 FLOPs 估算 HBM 实测流量。保留 runtime 的 cache 状态，",
        "不代表冷缓存带宽；cache 命中的重复调用可能使 HBM 字节少于逻辑 tensor 字节。",
        "",
        "NCU profile 是侵入式测量，clean 样本与其独立运行。profile range 的时间、inner、",
        "metric 名称、cache control 和原始 NCU 文件路径/哈希逐行保留。矩阵 MFU 和 HBM utilization",
        "都不是 Tensor Core 活跃率；完整 API 与其组件有重叠，不相加为模型时间或总流量。",
        "",
        f"峰值来源：[NVIDIA H200]({result['peak_source']})。",
        "",
        "Clean 计时边界：",
        "",
        *[f"- `{item}`" for item in boundary],
        "",
        "NCU 计时边界与 replay 模式：",
        "",
        *[f"- `{item}`" for item in profile_boundary],
        f"- replay：`{', '.join(replay_modes)}`",
        "",
        "| 表格 | 记录条数 | 含义 |",
        "| --- | ---: | --- |",
        f"| [common.csv](common.csv) | {operators['common']} | 公共 GEMM、非 GEMM 和 API 聚合 |",
        f"| [moe.csv](moe.csv) | {operators['moe']} | MoE GEMM、非 GEMM 和 API 聚合 |",
        f"| [attention.csv](attention.csv) | {operators['attention']} | 每个 B/Q/H 的 attention 操作 |",
        f"| [expanded.csv](expanded.csv) | {len(expanded)} | 公共映射与 attention 直接观测 |",
        "",
        f"其中具体操作为 {len(leaves)} 行，另列 API 聚合 {len(aggregates)} 行；每行均有独立 HBM profile。",
        "",
        "## Projection",
        "",
        "输入 projection、输出 `o_proj`、`q_absorb` 与 `v_expand` 均按实际 GEMM/BMM 叶操作单列。",
        "吸收 MLA 不额外执行普通 `kv_b_proj` API：对应 checkpoint 权重用于 Q absorb 与 value expand。",
        "完整 `attention_projection` 是输入准备聚合，包含输入投影、Q absorb、norm、casts、RoPE、",
        "量化、scale 与 KV concat；它不含 Indexer score、MLA、value expand 或输出 `o_proj`。",
        "聚合保留生产执行顺序与共享准备，不能用独立叶操作之和替代。",
        "",
    ]
    lines.extend(
        [
            "## 优先 Decode 配置的逐算子数值",
            "",
            "以下列出 S=128 的公共与 MoE 具体操作，以及 H=65536 的 Q1/B1、Q1/B128 attention。",
            "其他 S、H 与 Extend 配置见各 CSV 和后续图表。表中 MFU 使用 clean 时间，HBM 使用独立 profile 时间。",
            "",
        ]
    )
    for label, selected in (
        ("公共操作：S=128", [row for row in records["common"] if row["S"] == 128]),
        ("MoE 操作：S=128", [row for row in records["moe"] if row["S"] == 128]),
        (
            "Attention：B=1，Q=1，H=65536",
            [
                row
                for row in records["attention"]
                if (row["B"], row["Q"], row["H"]) == (1, 1, 65536)
            ],
        ),
        (
            "Attention：B=128，Q=1，H=65536",
            [
                row
                for row in records["attention"]
                if (row["B"], row["Q"], row["H"]) == (128, 1, 65536)
            ],
        ),
    ):
        lines.extend(
            [
                f"### {label}",
                "",
                "| 算子 | 类型 | 时间（µs） | MFU（%） | HBM（GB/s） | HBM 利用率（%） |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(
            selected,
            key=lambda item: (
                item["operator_kind"] == "api_aggregate",
                item["operator_family"],
                item["name"],
            ),
        ):
            mfu_value = "N/A" if row["mfu_pct"] is None else f"{row['mfu_pct']:.3f}"
            lines.append(
                f"| `{row['name']}` | {row['operator_kind']} | {row['time_us']:.3f} | {mfu_value} | {row['hbm_bandwidth_gbps']:.3f} | {row['hbm_util_pct']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 具体算子清单",
            "",
            "下表列出直接测量的算子名称、矩阵精度与所属 API；GEMM 包括融合的矩阵核心，",
            "`non_gemm` 的矩阵 MFU 为 N/A。支持条件和实际形状仍以每行 metadata 为准。",
            "",
            *render_inventory(inventory, aggregates=False),
            "",
            "## 单独测量的 API 聚合",
            "",
            "这些聚合保留生产调用的实际准备和执行顺序，与具体操作分别测量，不能重复相加。",
            "",
            *render_inventory(inventory, aggregates=True),
            "",
            "## Projection 图表",
            "",
        ]
    )
    projection = [
        name
        for name in figures
        if name.endswith(".png") and "projection" in name and not "aggregate_" in name
    ]
    lines.extend(f"![Projection](figures/{name})\n" for name in projection)
    lines.extend(
        [
            "## 其他具体操作与 API 聚合",
            "",
            "每行四列依次为 clean 时间、clean MFU、NCU HBM bandwidth 与 NCU HBM utilization。",
            "图中的 N/A 表示非 GEMM，不以零值代替；标题含 non-additive 的图是独立 API 聚合。",
            "",
        ]
    )
    lines.extend(
        f"![{name.removesuffix('.png')}](figures/{name})\n"
        for name in figures
        if name.endswith(".png") and name not in projection
    )
    lines.extend(
        [
            "MoE grouped GEMM 叶操作与 packing、量化、mapping、activation、加权归约分别展示。",
            "完整 MoE 聚合另列；useful FLOPs 只计实际 selected routes，padding capacity 不等于执行 FLOPs。",
            "",
            "来源、results SHA-256、具体工作负载的输入身份、数值验收 receipt、clean result、",
            "NCU profile result 与原始 profiler 文件哈希见 [provenance.json](provenance.json)。",
            "独立算子验收不代替完整模型或实际 serving 验证。",
            "",
        ]
    )
    return "\n".join(lines)


def publish(results_path, report_directory):
    source = Path(results_path).resolve(strict=True)
    target = Path(report_directory).resolve()
    if target.exists():
        raise FileExistsError(f"Publication directory already exists: {target}")
    result = read(source)
    evidence = validate_publication(result, source)
    records, expanded = rows(result, evidence)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="observation-report-", dir=target.parent) as temporary:
        directory = Path(temporary)
        figures_directory = directory / "figures"
        figures_directory.mkdir()
        for group in GROUPS:
            write_csv(directory / f"{group}.csv", records[group])
        write_csv(directory / "expanded.csv", expanded)
        inventory = operator_inventory(records)
        with (directory / "operator_inventory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
        figures = []
        for group in ("common", "moe"):
            figures.extend(
                plot_series(records[group], figures_directory, group=group, horizontal="S")
            )
        for q, b in sorted({(shape.Q, shape.B) for shape in matrix()}):
            selected = [row for row in records["attention"] if row["B"] == b and row["Q"] == q]
            figures.extend(
                plot_series(
                    selected,
                    figures_directory,
                    group="attention",
                    horizontal="H",
                    suffix=f" B={b}, Q={q}",
                )
            )
        (directory / "results.md").write_text(render_report(result, records, expanded, figures))
        evidence["coverage"] = {
            "workloads": {group: len(result["groups"][group]) for group in GROUPS},
            "direct_operator_records": {group: len(records[group]) for group in GROUPS},
            "expanded_records": len(expanded),
            "mapped_records_are_new_measurements": False,
            "profiled_operator_records": {group: len(records[group]) for group in GROUPS},
            "concrete_operator_records": {
                group: sum(row["operator_kind"] != "api_aggregate" for row in records[group])
                for group in GROUPS
            },
            "api_aggregate_records": {
                group: sum(row["operator_kind"] == "api_aggregate" for row in records[group])
                for group in GROUPS
            },
        }
        evidence["artifacts_sha256"] = {
            str(path.relative_to(directory)): digest(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        write(directory / "provenance.json", evidence)
        directory.rename(target)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id")
    source.add_argument("--results", type=Path)
    parser.add_argument("--report-dir", type=Path, default=EXPERIMENT / "report")
    arguments = parser.parse_args(argv)
    if arguments.run_id is not None and Path(arguments.run_id).name != arguments.run_id:
        raise ValueError("run-id must be a single path component")
    results_path = (
        arguments.results or EXPERIMENT / "output/data" / arguments.run_id / "results.json"
    )
    evidence = publish(results_path, arguments.report_dir)
    print(
        json.dumps({"source_run_id": evidence["source_run_id"], "coverage": evidence["coverage"]})
    )


if __name__ == "__main__":
    main()
