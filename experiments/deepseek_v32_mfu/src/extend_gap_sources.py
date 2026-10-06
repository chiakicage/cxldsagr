"""Partition the accepted three-layer extend gap without running GPU work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

PRODUCTIVE = {"Compute", "Compute + IO", "IO"}
LABELS = {
    "scale_transpose": "DeepGEMM scale transpose",
    "projection_layout": "Projection metadata / layout",
    "cache_write": "Cache / index write and writeback source",
    "exact_recall": "Exact recall / resident selection metadata",
    "echo_prepare": "ECHO prefetch preparation",
    "echo_finalize": "ECHO prefetch finalization",
    "prefetch_hint": "Prefetch hint update",
    "dense_mapping": "Dense mapping clear / publish",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def control_category(row):
    stage = row["stage"] or ""
    if "deep_gemm::transpose_fp32" in row["name"]:
        return "scale_transpose"
    if stage.startswith("compute_graph_projection_"):
        return "projection_layout"
    if stage in {"cache_write", "index_cache_write", "compute_graph_owned_writeback_source"}:
        return "cache_write"
    if stage == "offload_exact_recall":
        return "exact_recall"
    if stage.startswith("dense_history_prefetch_layer_"):
        return "dense_mapping"
    direct = {
        "offload_prepare": "echo_prepare",
        "offload_finalize": "echo_finalize",
        "prefetch_hint": "prefetch_hint",
    }
    if stage in direct:
        return direct[stage]
    raise ValueError(f"Unclassified exposed control: {stage}, {row['name']}")


def api_lookup(method):
    lookup = {}
    for gap in method["gaps"]:
        for row in [gap["previous_gpu"], gap["next_gpu"], *gap["control_activities"]]:
            if row and row.get("api"):
                key = row["kind"], row["start"], row["end"], row["name"]
                api = row["api"]
                if key in lookup and lookup[key] != api:
                    raise ValueError("Conflicting launch correlation")
                lookup[key] = api
    return lookup


def analyze(panel, accepted_method):
    window = panel["window"]
    start, end = window["start_ns"], window["end_ns"]
    rows = [row for row in panel["rows"] if row["kind"] != "api"]
    boundaries = sorted(
        {start, end}
        | {max(start, row["raw_start_ns"]) for row in rows}
        | {min(end, row["raw_end_ns"]) for row in rows}
    )
    by_category, by_stage = defaultdict(int), defaultdict(int)
    idle = []
    control_total = productive_total = 0
    for left, right in pairwise(boundaries):
        active = [row for row in rows if row["raw_start_ns"] <= left < row["raw_end_ns"]]
        controls = [row for row in active if row["lane"] == "GPU control"]
        productive = any(row["lane"] in PRODUCTIVE for row in active)
        if controls:
            control_total += right - left
        if productive:
            productive_total += right - left
        elif controls:
            if len(controls) != 1:
                raise ValueError("Concurrent exposed control requires explicit attribution")
            by_category[control_category(controls[0])] += right - left
            by_stage[controls[0]["stage"]] += right - left
        elif active:
            raise ValueError("Unknown GPU activity lane")
        elif idle and idle[-1][1] == left:
            idle[-1][1] = right
        else:
            idle.append([left, right])
    lookup = api_lookup(accepted_method)
    idle_details = []
    idle_stages = defaultdict(int)
    submission = defaultdict(int)
    for left, right in idle:
        following = [row for row in rows if row["raw_start_ns"] == right]
        if len(following) != 1:
            raise ValueError("Idle endpoint does not identify a unique next GPU activity")
        row = following[0]
        key = row["kind"], row["raw_start_ns"], row["raw_end_ns"], row["name"]
        api = lookup[key]
        if row["correlation"] != api["correlation"]:
            raise ValueError("Next activity and launch API correlations differ")
        pieces = {
            "before_next_api": max(0, min(right, api["start"]) - left),
            "during_next_api": max(0, min(right, api["end"]) - max(left, api["start"])),
            "after_next_api": max(0, right - max(left, api["end"])),
        }
        if sum(pieces.values()) != right - left:
            raise ValueError("Idle/API partition is incomplete")
        for category, duration in pieces.items():
            submission[category] += duration
        idle_stages[row["stage"]] += right - left
        idle_details.append(
            {
                "start_ns": left,
                "end_ns": right,
                "duration_ns": right - left,
                "next_gpu": row,
                "next_api": api,
                "submission_partition_ns": pieces,
            }
        )
    idle_total = sum(right - left for left, right in idle)
    exposed_total = sum(by_category.values())
    for actual, field in (
        (idle_total, "gpu_idle_ms"),
        (exposed_total, "control_only_ms"),
        (idle_total + exposed_total, "gap_ms"),
        (productive_total, "compute_io_union_ms"),
    ):
        if actual != round(window[field] * 1e6):
            raise ValueError(f"Accepted {field} differs")
    return {
        "method": panel["method"],
        "window": window,
        "gap_ns": idle_total + exposed_total,
        "exposed_control_ns": exposed_total,
        "control_total_union_ns": control_total,
        "control_overlapped_by_productive_ns": control_total - exposed_total,
        "gpu_idle_ns": idle_total,
        "exposed_control_categories_ns": dict(by_category),
        "exposed_control_stages_ns": dict(by_stage),
        "idle_by_next_stage_ns": dict(idle_stages),
        "idle_submission_partition_ns": dict(submission),
        "idle_intervals": sorted(idle_details, key=lambda row: -row["duration_ns"]),
    }


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-dir", type=Path, required=True)
    parser.add_argument("--accepted-gap-audit", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_path = args.timeline_dir / "window_rows.json"
    receipt_path = args.timeline_dir / "compact_receipt.json"
    receipt = read(receipt_path)
    if sha(data_path) != receipt["artifacts_sha256"][data_path.name]:
        raise ValueError("Published timeline input changed")
    paths = [
        data_path,
        receipt_path,
        args.accepted_gap_audit,
        args.independent_audit,
        Path(__file__),
    ]
    bindings = {str(path.resolve()): sha(path) for path in paths}
    accepted = read(args.accepted_gap_audit)
    if accepted["run_id"] != receipt["profile_run_id"]:
        raise ValueError("Original capture identities differ")
    methods = [
        analyze(panel, next(x for x in accepted["methods"] if x["method"] == panel["method"]))
        for panel in read(data_path)["extend"]
    ]
    independent = read(args.independent_audit)
    for method in methods:
        expected = independent["methods"][method["method"]]
        for key, other in (
            ("gap_ns", "gap_ns"),
            ("gpu_idle_ns", "idle_ns"),
            ("exposed_control_ns", "control_ns"),
        ):
            if method[key] != expected[other]:
                raise ValueError(f"Independent {key} differs")
    if bindings != {str(path.resolve()): sha(path) for path in paths}:
        raise ValueError("Source changed during analysis")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = []
    controls = []
    for method in methods:
        summary.append(
            {
                "method": method["method"],
                "gap_us": method["gap_ns"] / 1000,
                "exposed_control_us": method["exposed_control_ns"] / 1000,
                "gpu_idle_us": method["gpu_idle_ns"] / 1000,
                "idle_share_of_gap_percent": 100 * method["gpu_idle_ns"] / method["gap_ns"],
                **{
                    key + "_us": value / 1000
                    for key, value in method["idle_submission_partition_ns"].items()
                },
                "control_total_union_us": method["control_total_union_ns"] / 1000,
                "control_overlapped_by_productive_us": method["control_overlapped_by_productive_ns"]
                / 1000,
            }
        )
        controls.extend(
            {
                "method": method["method"],
                "category": category,
                "description": LABELS[category],
                "exposed_us": method["exposed_control_categories_ns"].get(category, 0) / 1000,
            }
            for category in LABELS
        )
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "control_sources.csv", controls)
    report = {
        "schema": "v10-three-layer-extend-gap-sources-v1",
        "analysis_id": args.output_dir.name,
        "profile_run_id": receipt["profile_run_id"],
        "source_bindings": bindings,
        "new_gpu_run": False,
        "independent_control_and_idle_totals_match": True,
        "control_category_labels": LABELS,
        "notes": [
            "Only L0 first compute through L2 last compute; fused compute+IO is entirely productive.",
            "Only exposed control contributes to gap; concurrent productive work hides control.",
            "Idle is partitioned by the correlated next GPU activity's API start and return; this is temporal evidence, not a causal CPU/GPU/Python diagnosis.",
            "Idle-stage labels indicate the next GPU activity, not exclusive operation cost.",
            "Original intrusive NSYS profiles; no new GPU run or performance optimization.",
        ],
        "methods": methods,
    }
    (args.output_dir / "sources.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(__file__, args.output_dir / "analyzer.py")
    shutil.copy2(args.independent_audit, args.output_dir / "independent_audit.json")
    print(args.output_dir)


if __name__ == "__main__":
    main()
