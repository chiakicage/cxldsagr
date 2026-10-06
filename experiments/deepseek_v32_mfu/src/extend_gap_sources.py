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
    "input_norm_layout": "Input/residual norm helpers / layout",
    "indexer_layout": "Indexer helpers / metadata / layout",
    "topk_layout": "Exact top-k helpers / metadata / layout",
    "attention_output_layout": "Attention output helpers / layout",
    "attention_layout": "Sparse MLA helpers / layout",
    "post_norm_layout": "Post-attention norm helpers / layout",
    "mlp_layout": "MLP helpers / layout",
    "normalization_layout": "Normalization helpers / layout (source stage unspecified)",
    "endpoint_layout": "Embedding / final norm / LM-head helpers / layout",
    "rollback_backup": "ECHO rollback hint backup",
    "finish_layout": "Output / MLP graph helpers / layout",
}
SUBMISSION_KEYS = (
    "before_next_api",
    "during_next_api",
    "after_next_api",
    "ambiguous_next_api",
    "no_next_gpu",
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def control_category(row):
    stage = row["stage"] or ""
    source = row.get("source_stage") or stage
    if "deep_gemm::transpose_fp32" in row["name"]:
        return "scale_transpose"
    if stage.startswith("compute_graph_projection_") or source == "attention_projection":
        return "projection_layout"
    if stage.startswith("compute_graph_finish_"):
        return "finish_layout"
    source_categories = {
        "input_residual_norm": "input_norm_layout",
        "post_attention_residual_norm": "post_norm_layout",
        "attention_output": "attention_output_layout",
        "v_expand": "attention_output_layout",
        "o_proj": "attention_output_layout",
        "sparse_mla": "attention_layout",
        "mla_qk_pv": "attention_layout",
        "compute_graph_value_expansion": "attention_output_layout",
        "dense_mlp": "mlp_layout",
        "mlp": "mlp_layout",
        "mlp_gate": "mlp_layout",
        "mlp_up": "mlp_layout",
        "mlp_down": "mlp_layout",
        "indexer": "indexer_layout",
        "indexer_prefetch": "indexer_layout",
        "indexer_qk": "indexer_layout",
        "indexer_fused": "indexer_layout",
        "exact_topk": "topk_layout",
        "topk": "topk_layout",
        "residual_rms_norm": "normalization_layout",
        "rms_norm": "normalization_layout",
        "embedding": "endpoint_layout",
        "final_norm_lm_head": "endpoint_layout",
        "final_norm": "endpoint_layout",
        "lm_head": "endpoint_layout",
        "extend_graph_rollback_backup": "rollback_backup",
    }
    source_categories.update(
        dict.fromkeys(
            (
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj",
                "q_absorb",
                "index_q_proj",
                "index_k_proj",
                "index_weights_proj",
            ),
            "projection_layout",
        )
    )
    if source in source_categories:
        return source_categories[source]
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


def gpu_key(row):
    """Retain stream/node identity when simultaneous activities share timestamps."""
    return (
        row["kind"],
        row.get("raw_start_ns", row.get("start")),
        row.get("raw_end_ns", row.get("end")),
        row["name"],
        row.get("device_id"),
        row.get("stream", row.get("stream_id")),
        row.get("process"),
        row.get("graph_node_id"),
    )


def combination(values):
    values = sorted(set(values))
    return values[0] if len(values) == 1 else "concurrent:" + json.dumps(values)


def api_lookup(method):
    lookup = {}
    for gap in method["gaps"]:
        for row in [gap["previous_gpu"], gap["next_gpu"], *gap["control_activities"]]:
            if row and row.get("api"):
                key = gpu_key(row)
                api = row["api"]
                if key in lookup and lookup[key] != api:
                    raise ValueError("Conflicting launch correlation")
                lookup[key] = api
    return lookup


def idle_submission(left, right, following, lookup):
    """Partition once, retaining ambiguity when simultaneous successors disagree."""
    partitions = []
    apis = []
    for row in following:
        api = lookup[gpu_key(row)]
        if row["correlation"] != api["correlation"]:
            raise ValueError("Next activity and launch API correlations differ")
        if api["end"] < api["start"]:
            raise ValueError("Next API has a negative duration")
        apis.append(api)
        partitions.append(
            {
                "before_next_api": max(0, min(right, api["start"]) - left),
                "during_next_api": max(0, min(right, api["end"]) - max(left, api["start"])),
                "after_next_api": max(0, right - max(left, api["end"])),
            }
        )
    pieces = dict.fromkeys(SUBMISSION_KEYS, 0)
    if not following:
        pieces["no_next_gpu"] = right - left
        relation = "no_successor_in_window"
    elif all(part == partitions[0] for part in partitions):
        pieces.update(partitions[0])
        relation = "unique_successor" if len(following) == 1 else "simultaneous_successors_agree"
    else:
        pieces["ambiguous_next_api"] = right - left
        relation = "simultaneous_successors_disagree"
    if sum(pieces.values()) != right - left or any(
        sum(part.values()) != right - left for part in partitions
    ):
        raise ValueError("Idle/API partition is incomplete")
    return {
        "submission_partition_ns": pieces,
        "next_gpu": following[0] if len(following) == 1 else None,
        "next_gpu_candidates": following,
        "next_api": apis[0] if apis and all(api == apis[0] for api in apis) else None,
        "next_api_candidates": apis,
        "candidate_submission_partitions_ns": partitions,
        "successor_relation": relation,
    }


def analyze(panel, accepted_method):
    window = panel["window"]
    start, end = window["start_ns"], window["end_ns"]
    if end <= start:
        raise ValueError("Timeline window must have positive duration")
    rows = [
        row
        for row in panel["rows"]
        if row["kind"] != "api" and row["raw_start_ns"] < end and row["raw_end_ns"] > start
    ]
    boundaries = sorted(
        {start, end}
        | {max(start, row["raw_start_ns"]) for row in rows}
        | {min(end, row["raw_end_ns"]) for row in rows}
    )
    by_category, by_stage = defaultdict(int), defaultdict(int)
    by_source_stage = defaultdict(int)
    category_members = {}
    concurrent_controls = []
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
            categories = sorted({control_category(row) for row in controls})
            category = combination(categories)
            category_members[category] = categories
            by_category[category] += right - left
            if len(controls) > 1:
                concurrent_controls.append(
                    {
                        "start_ns": left,
                        "end_ns": right,
                        "duration_ns": right - left,
                        "category": category,
                        "activities": controls,
                    }
                )
            by_stage[combination(row["stage"] or "<unspecified>" for row in controls)] += (
                right - left
            )
            by_source_stage[
                combination(
                    row.get("source_stage") or row["stage"] or "<unspecified>" for row in controls
                )
            ] += right - left
        elif active:
            raise ValueError("Unknown GPU activity lane")
        elif idle and idle[-1][1] == left:
            idle[-1][1] = right
        else:
            idle.append([left, right])
    lookup = api_lookup(accepted_method)
    idle_details = []
    idle_stages = defaultdict(int)
    submission = dict.fromkeys(SUBMISSION_KEYS, 0)
    for left, right in idle:
        following = [row for row in rows if row["raw_start_ns"] == right]
        evidence = idle_submission(left, right, following, lookup)
        for category, duration in evidence["submission_partition_ns"].items():
            submission[category] += duration
        next_stage = (
            combination(row["stage"] or "<unspecified>" for row in following)
            if following
            else "<window-end>"
        )
        idle_stages[next_stage] += right - left
        idle_details.append(
            {
                "start_ns": left,
                "end_ns": right,
                "duration_ns": right - left,
                **evidence,
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
        "control_category_members": category_members,
        "concurrent_exposed_control_ns": sum(row["duration_ns"] for row in concurrent_controls),
        "concurrent_exposed_control_intervals": concurrent_controls,
        "control_category_labels": {
            key: LABELS[members[0]]
            if len(members) == 1
            else "Concurrent exposed controls (union): "
            + " + ".join(LABELS[item] for item in members)
            for key, members in category_members.items()
        },
        "exposed_control_stages_ns": dict(by_stage),
        "exposed_control_source_stages_ns": dict(by_source_stage),
        "idle_by_next_stage_ns": dict(idle_stages),
        "idle_submission_partition_ns": dict(submission),
        "observed_graph_ids": sorted({row["graph_id"] for row in rows if row.get("graph_id")}),
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
    labels = dict(LABELS)
    for method in methods:
        labels.update(method["control_category_labels"])
    for method in methods:
        summary.append(
            {
                "method": method["method"],
                "gap_us": method["gap_ns"] / 1000,
                "exposed_control_us": method["exposed_control_ns"] / 1000,
                "gpu_idle_us": method["gpu_idle_ns"] / 1000,
                "idle_share_of_gap_percent": (
                    100 * method["gpu_idle_ns"] / method["gap_ns"] if method["gap_ns"] else None
                ),
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
                "description": labels[category],
                "exposed_us": method["exposed_control_categories_ns"].get(category, 0) / 1000,
            }
            for category in labels
        )
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "control_sources.csv", controls)
    report = {
        "schema": "three-layer-extend-gap-sources-v2",
        "analysis_id": args.output_dir.name,
        "profile_run_id": receipt["profile_run_id"],
        "source_bindings": bindings,
        "new_gpu_run": False,
        "independent_control_and_idle_totals_match": True,
        "control_category_labels": labels,
        "notes": [
            "The selected three-layer window ends at L2 last compute and starts at L0 first compute, or its history H2D if earlier for dense extend; fused compute+IO is entirely productive.",
            "Only exposed control contributes to gap; concurrent productive work hides control.",
            "Idle is partitioned by the correlated next GPU activity's API start and return; this is temporal evidence, not a causal CPU/GPU/Python diagnosis.",
            "CUDA graph nodes can share one cudaGraphLaunch; their idle intervals do not represent separate Python kernel submissions.",
            "Simultaneous exposed controls use explicit combination buckets and are counted once; simultaneous next activities retain all API candidates and mark conflicting temporal partitions ambiguous.",
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
