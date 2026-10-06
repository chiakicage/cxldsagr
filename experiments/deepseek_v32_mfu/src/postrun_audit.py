"""Recheck saved outputs and trace-summary conservation without CUDA or trace replay.

This is a lightweight consistency audit of result, ledger and analysis files.
It does not independently parse SQLite or establish native backend identities.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from experiments.deepseek_v32_mfu.src.compare_backends import (
    compare_tensors,
    digest,
    validate_result,
)
from experiments.deepseek_v32_mfu.src.execution_utilization import (
    precision_normalized_utilization,
)
from experiments.deepseek_v32_mfu.src.operator_report import read_calls
from experiments.deepseek_v32_mfu.src.run_contract import benchmark_view, control_directory

KEY = ("mode", "phase", "layer", "stage")
CAPTURES = {
    (mode, phase)
    for mode in ("resident", "offload")
    for phase in ("prefill_annotated", "extend_annotated")
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(
        type(actual) in (int, float)
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
        f"Inconsistent {label}: {actual} versus {expected}",
    )


def key(row):
    return tuple(row[field] for field in KEY)


def audit_work(row, calls, peaks):
    require(bool(calls), f"Missing ledger calls: {key(row)}")
    require(
        row["scope_count"] == row["metadata_call_count"] == len(calls)
        and row["metadata_call_count_matches"] is True,
        f"Metadata/scope count mismatch: {key(row)}",
    )
    matrix = [call for call in calls if call["useful_flops"] is not None]
    if not matrix:
        require(
            all(call.get("formula") == "N/A (no matrix multiply)" for call in calls)
            and row["useful_flops"] is None
            and row["kernel_mfu_percent"] is None,
            f"Nonmatrix work must retain null FLOPs/MFU: {key(row)}",
        )
        return
    require(len(matrix) == len(calls), f"Mixed known/unknown work: {key(row)}")
    precisions = {call["precision"] for call in calls}
    require(len(precisions) == 1, f"Mixed precision scope: {key(row)}")
    precision = precisions.pop()
    useful = sum(call["useful_flops"] for call in calls)
    require(row["precision"] == precision and row["useful_flops"] == useful, "Work mismatch")
    require(row["dense_peak_tflops"] == peaks[precision], "Matrix dense peak mismatch")
    if useful and row["kernel_ns"]:
        close(
            row["kernel_mfu_percent"],
            useful / (row["kernel_ns"] * peaks[precision] * 1000) * 100,
            "kernel MFU",
        )
    else:
        require(row["kernel_mfu_percent"] is None, "MFU requires nonzero work and kernel time")


def audit_analysis(result, analysis, calls):
    """Recompute arithmetic across independent ledger/row/inventory summaries."""
    require(not analysis["calls_outside_selected_captures"], "Unused ledger calls")
    captures = analysis["captures"]
    require(
        len(captures) == 4 and {(c["mode"], c["phase"]) for c in captures} == CAPTURES,
        "Expected one capture for each resident/offload prefix/extend execution",
    )
    call_groups, inventory_groups = defaultdict(list), defaultdict(list)
    for call in calls:
        call_groups[key(call)].append(call)
    rows = {key(row): row for row in analysis["operators_by_layer"]}
    require(len(rows) == len(analysis["operators_by_layer"]), "Duplicate operator rows")
    require(rows.keys() == call_groups.keys(), "Ledger/operator scope coverage differs")
    for item in analysis["kernel_inventory"]:
        require(
            key(item) in rows and item["unattributed_reason"] is None,
            "Unattributed kernel inventory entry",
        )
        require(
            type(item["count"]) is int
            and item["count"] > 0
            and type(item["total_ns"]) is int
            and item["total_ns"] > 0,
            "Invalid inventory count or duration",
        )
        close(item["total_ms"], item["total_ns"] / 1e6, "inventory time")
        inventory_groups[key(item)].append(item)
    for identity, row in rows.items():
        audit_work(row, call_groups[identity], analysis["dense_peaks_tflops"])
        for kind in ("kernel", "memcpy", "memset"):
            for suffix in ("count", "ns"):
                value = row[kind + "_" + suffix]
                require(type(value) is int and value >= 0, "Invalid activity count or duration")
            close(row[kind + "_ms"], row[kind + "_ns"] / 1e6, "operator activity time")
        for field, source in (("kernel_count", "count"), ("kernel_ns", "total_ns")):
            require(
                row[field] == sum(item[source] for item in inventory_groups[identity]),
                f"Kernel inventory/operator {field} mismatch: {identity}",
            )
    capture_checks, utilization = [], []
    work_by_capture = {}
    for capture in captures:
        identity = capture["mode"], capture["phase"]
        selected = [row for row in rows.values() if key(row)[:2] == identity]
        selected_calls = [call for call in calls if key(call)[:2] == identity]
        require(
            {row["layer"] for row in selected if row["layer"] != "shared"}
            == {f"layer_{i}" for i in range(result["num_layers"])},
            "Missing or unexpected layer coverage",
        )
        audit = capture["audit"]
        require(
            audit["kernel_count_and_time_conserved"] is True
            and audit["metadata_call_counts_match"] is True
            and not audit["metadata_call_count_mismatches"]
            and not audit["layer_unscoped_kernel_count"],
            "Analyzer reports incomplete attribution",
        )
        totals = {}
        for kind in ("kernel", "memcpy", "memset"):
            values = audit["activity_counts_and_ns"][kind]
            count = sum(row[kind + "_count"] for row in selected)
            duration = sum(row[kind + "_ns"] for row in selected)
            require(
                values["count"] == values["attributed_count"] == count
                and values["total_ns"] == values["attributed_ns"] == duration
                and values["unattributed_count"] == values["unattributed_ns"] == 0,
                f"Activity attribution conservation failed: {identity}/{kind}",
            )
            totals[kind] = {"count": count, "total_ns": duration}
        require(
            audit["operator_kernel_ns"]
            == audit["inventory_kernel_ns"]
            == totals["kernel"]["total_ns"],
            "Reported kernel conservation totals differ",
        )
        require(
            sum(row["api_count"] for row in selected)
            == capture["api_summary"]["non_event_api_count"],
            "CUDA API count conservation failed",
        )
        require(
            len(capture["devices"]) == 1, "Audit only supports the measured single-device workload"
        )
        device = capture["devices"][0]
        require(
            device["activity_count"]
            == device["attributed_activity_count"]
            == sum(v["count"] for v in totals.values()),
            "Device activity count mismatch",
        )
        close(
            device["activity_duration_sum_ms"],
            sum(v["total_ns"] for v in totals.values()) / 1e6,
            "device duration",
        )
        close(device["span_ms"], (device["end_ns"] - device["start_ns"]) / 1e6, "device span")
        close(device["busy_ms"] + device["gap_ms"], device["span_ms"], "busy + gap")
        close(
            device["busy_ms"] + device["overlap_ms"],
            device["activity_duration_sum_ms"],
            "busy + overlap",
        )
        require(
            0 <= device["kernel_busy_ms"] <= device["busy_ms"] <= device["span_ms"]
            and device["gap_ms"] >= 0
            and device["overlap_ms"] >= 0,
            "Impossible busy/gap/overlap intervals",
        )
        close(capture["capture_gpu_envelope"]["span_ms"], device["span_ms"], "capture envelope")
        phase = "prefix" if capture["phase"] == "prefill_annotated" else "extend"
        if phase + "_samples_ms" in result["measurements"][capture["mode"]]:
            utilization.append(
                {
                    "mode": capture["mode"],
                    "phase": phase,
                    **(
                        {"wall_time_denominator": result["wall_time_denominator"]}
                        if "wall_time_denominator" in result
                        else {}
                    ),
                    **precision_normalized_utilization(
                        selected_calls,
                        result["measurements"][capture["mode"]][phase + "_samples_ms"],
                        peaks_tflops=analysis["dense_peaks_tflops"],
                        scope=result["scope"],
                    ),
                }
            )
        else:
            utilization.append(
                {
                    "mode": capture["mode"],
                    "phase": phase,
                    "status": "pending_independent_bench",
                    "value": None,
                }
            )
        work = Counter()
        for call in selected_calls:
            if call["useful_flops"] is not None:
                stage = (
                    "indexer" if call["stage"] in ("indexer_qk", "indexer_fused") else call["stage"]
                )
                work[stage, call["precision"]] += call["useful_flops"]
        work_by_capture[identity] = work
        capture_checks.append(
            {
                "mode": identity[0],
                "phase": identity[1],
                "activity_totals": totals,
                "busy_gap_arithmetic_matches": True,
            }
        )
    for phase in ("prefill_annotated", "extend_annotated"):
        require(
            work_by_capture["resident", phase] == work_by_capture["offload", phase],
            "Resident/offload useful work differs",
        )
    pooled = {key(row): row for row in analysis["operators"]}
    groups = defaultdict(list)
    for row in rows.values():
        groups[(row["mode"], row["phase"], "all_layers", row["stage"])].append(row)
    require(
        len(pooled) == len(analysis["operators"]) and pooled.keys() == groups.keys(),
        "Pooled scope coverage differs",
    )
    for identity, group in groups.items():
        row = pooled[identity]
        grouped_calls = [call for member in group for call in call_groups[key(member)]]
        audit_work(row, grouped_calls, analysis["dense_peaks_tflops"])
        for kind in ("kernel", "memcpy", "memset"):
            for suffix in ("count", "ns"):
                field = kind + "_" + suffix
                require(
                    row[field] == sum(member[field] for member in group), f"Pooled {field} differs"
                )
    return {
        "captures": capture_checks,
        "operator_rows": len(rows),
        "kernel_count": sum(x["activity_totals"]["kernel"]["count"] for x in capture_checks),
        "end_to_end_precision_normalized_utilization": utilization,
    }


def audit_saved_outputs(directory, result):
    import torch

    names = (
        "resident_control",
        "offload_control",
        "resident_profile_output",
        "offload_profile_output",
    )
    controls = control_directory(directory, result)
    paths = {
        name: (controls if name.endswith("_control") else directory) / (name + ".pt")
        for name in names
    }
    saved = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in paths.items()
    }
    checks = []
    for name, values in saved.items():
        require(set(values) == {"hidden", "logits"}, "Saved output fields differ")
        for field, actual in values.items():
            expected = saved["resident_control"][field]
            rows = result["extend_tokens"] if field == "hidden" else 1
            require(actual.ndim == 2 and actual.shape[0] == rows, "Saved output coverage differs")
            measured = compare_tensors(actual, expected)
            require(measured["bitwise_equal"], "Saved internal outputs fail bitwise gate")
            checks.append({"file": name + ".pt", "field": field, **measured})
    for mode in ("resident", "offload"):
        for field in ("hidden", "logits"):
            require(
                result["correctness"][mode + "_profile_extend_" + field]["shape"]
                == list(saved[mode + "_control"][field].shape),
                "Correctness output shape mismatch",
            )
    return {
        "tensor_checks": checks,
        "file_sha256": {name + ".pt": digest(paths[name]) for name in names},
        "runtime_only_prefix_checks": [
            mode + "_profile_prefix_logits" for mode in ("resident", "offload")
        ],
    }


def audit_run(directory, analysis_path=None):
    directory = Path(directory)
    analysis_path = Path(analysis_path) if analysis_path else directory / "analysis/analysis.json"
    result = json.loads((directory / "result.json").read_text())
    analysis = json.loads(analysis_path.read_text())
    calls, metadata = read_calls(directory / "operator_calls.json")
    validate_result(directory, result, require_benchmark=False)
    result = benchmark_view(directory, result, required=False)
    require(
        result["run_id"] == metadata["run_id"] == analysis["ledger_metadata"]["run_id"],
        "Run ID mismatch",
    )
    require(
        analysis["calls_sha256"] == digest(directory / "operator_calls.json"), "Ledger SHA mismatch"
    )
    return {
        "schema_version": result.get("schema_version", 1),
        "mode": result.get("mode", "historical_combined"),
        "wall_time_denominator": result.get("wall_time_denominator"),
        "validation_receipt": result.get("validation_receipt"),
        "run_id": result["run_id"],
        "accepted": True,
        "auditor_sha256": digest(Path(__file__)),
        "input_sha256": {
            "result.json": digest(directory / "result.json"),
            "operator_calls.json": digest(directory / "operator_calls.json"),
            "analysis": digest(analysis_path),
        },
        "summary_conservation": audit_analysis(result, analysis, calls),
        "saved_output_checks": audit_saved_outputs(directory, result),
        "boundary": "Checks result/ledger/analysis consistency, reported activity count/time and busy/gap arithmetic, and all saved extend hidden/logit tensors on CPU. Does not independently parse native traces, inspect native kernels, or verify source/native-library hashes. Prefix numerical checks are runtime-only because prefix control tensors are not saved. Annotated kernel sums and unannotated wall time remain separate metrics.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a fresh audit output file")
    audit = audit_run(args.run_dir, args.analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "accepted": True, "run_id": audit["run_id"]}))


if __name__ == "__main__":
    main()
