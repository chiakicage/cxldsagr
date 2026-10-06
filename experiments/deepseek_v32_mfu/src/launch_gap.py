"""Audit complete extend overhead from native Nsight activity intervals.

The user-defined gap includes cache/control kernels and copies, as well as idle
time. Only model computation and host IO are excluded. This is separate from
the older timeline's all-GPU-activity idle gap.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from experiments.deepseek_v32_mfu.src.analyze_nsys import (
    _assign_scopes,
    _attribute,
    _read_capture,
    _union,
    _union_ns,
    kernel_category,
)
from experiments.deepseek_v32_mfu.src.operator_report import _SCOPE, read_calls
from experiments.deepseek_v32_mfu.src.timeline import (
    activity_lane,
    activity_layer,
    activity_stage,
    intersection_ns,
    select_forward_activities,
    select_forward_apis,
    select_layer,
    source_activity,
)
from experiments.deepseek_v32_motivation.src.graph_attribution import (
    attribute_graph_replays,
    read_lineage,
)

METHODS = ("hbm", "echo", "serial_sparse", "dense_prefetch")


def annotate_actual_io(activities, metrics, method, phase):
    """Bind transport nodes to this measured phase's synchronized layer counters.

    A zero nonnegative phase total proves every matching call empty. A positive
    total identifies one call only when that counter has one transport node.
    """
    phase_key = (
        "prefix_cache_per_layer" if phase == "prefill_annotated" else "extend_cache_per_layer"
    )
    if metrics is not None:
        for counts in metrics:
            for field in (
                "recalled_records",
                "prefetched_records",
                "host_to_device_bytes",
                "record_bytes",
            ):
                value = counts[field]
                if type(value) is not int or value < (1 if field == "record_bytes" else 0):
                    raise ValueError(
                        "actual IO evidence requires nonnegative integer record counters"
                    )
            if (
                counts["host_to_device_bytes"]
                != (counts["recalled_records"] + counts["prefetched_records"])
                * counts["record_bytes"]
            ):
                raise ValueError("actual IO byte and record counters disagree")
    groups = defaultdict(list)
    for row in activities:
        if row["kind"] != "kernel":
            continue
        family = kernel_category(row["name"])
        if family not in {"mapped_host_kv_gather", "indexer_fused_prefetch"}:
            continue
        owner = re.fullmatch(r"layer_(\d+)", activity_layer(row) or "")
        if owner is None:
            owner = re.fullmatch(r"dense_history_prefetch_layer_(\d+)", activity_stage(row) or "")
        if owner is None:
            raise ValueError("transport activity lacks a phase-layer counter owner")
        layer = int(owner[1])
        if metrics is not None and layer >= len(metrics):
            raise ValueError("transport layer has no measured counter evidence")
        groups[layer, family].append(row)
    audit = []
    for (layer, family), rows in sorted(groups.items()):
        field = "prefetched_records" if family == "indexer_fused_prefetch" else "recalled_records"
        total = metrics[layer][field] if metrics is not None else None
        # ECHO and serial_sparse recall counters exclusively count gathers.
        # Dense may mix DMA and mapped gathers under one recalled-record total.
        exclusive = family == "indexer_fused_prefetch" or method in {"echo", "serial_sparse"}
        status = (
            "phase_zero"
            if total == 0
            else "unique_positive"
            if total is not None and len(rows) == 1 and exclusive
            else "unresolved"
        )
        evidence = {
            "family": family,
            "layer": layer,
            "status": status,
            "records": total if status != "unresolved" else None,
            "bytes": total * metrics[layer]["record_bytes"] if status != "unresolved" else None,
            "phase_records": total,
            "matching_kernel_count": len(rows),
            "counter_path": f"result.json:measurements.{method}.{phase_key}[{layer}].{field}",
            "unresolved_reason": (
                "phase counters missing"
                if metrics is None
                else "per-call record counts unavailable"
                if len(rows) != 1
                else "phase counter also covers other transports"
            )
            if status == "unresolved"
            else None,
        }
        for row in rows:
            row["actual_io"] = evidence.copy()
        audit.append({**evidence, "activities": [source_activity(row) for row in rows]})
    return audit


def clipped(intervals, start, end):
    return [(max(start, a), min(end, b)) for a, b in intervals if a < end and b > start]


def complement(intervals, start, end):
    cursor = start
    gaps = []
    for left, right in _union(clipped(intervals, start, end)):
        if left > cursor:
            gaps.append((cursor, left))
        cursor = right
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def summarize_window(
    activities, start, end, *, threshold_percent=10.0, lane_classifier=activity_lane
):
    """Use interval unions; overlapping IO never erases simultaneous compute.

    Fused compute+IO is entirely productive and remains in the denominator.
    Only standalone IO outside both pure and fused computation is removed.
    """
    if end <= start:
        raise ValueError("gap window must have positive duration")
    lanes = defaultdict(list)
    unresolved_gathers = 0
    for activity in activities:
        if activity["end"] <= start or activity["start"] >= end:
            continue
        evidence = activity.get("actual_io", {})
        unresolved_gathers += (
            evidence.get("family") == "mapped_host_kv_gather"
            and evidence.get("status") == "unresolved"
        )
        lanes[lane_classifier(activity)].extend(
            clipped([(activity["start"], activity["end"])], start, end)
        )
    compute, io, fused = lanes["Compute"], lanes["IO"], lanes["Compute + IO"]
    productive = compute + io + fused
    all_gpu = productive + lanes["GPU control"]
    window = end - start
    productive_ns = _union_ns(productive)
    gap = window - productive_ns
    idle = window - _union_ns(all_gpu)
    control_only = gap - idle
    io_only = _union_ns(io) - intersection_ns(io, compute + fused)
    io_or_fused_only = _union_ns(io + fused) - intersection_ns(io + fused, compute)
    retained_denominator = window - io_only
    ratio = 100 * gap / retained_denominator if retained_denominator else None
    gate_pass = unresolved_gathers == 0 and ratio is not None and ratio < threshold_percent
    return {
        "start_ns": start,
        "end_ns": end,
        "window_ms": window / 1e6,
        "compute_union_ms": _union_ns(compute) / 1e6,
        "fused_union_ms": _union_ns(fused) / 1e6,
        "io_union_ms": _union_ns(io) / 1e6,
        "compute_io_union_ms": productive_ns / 1e6,
        "gpu_idle_ms": idle / 1e6,
        "control_only_ms": control_only / 1e6,
        "gap_ms": gap / 1e6,
        "gap_percent": 100 * gap / window,
        "pure_io_only_ms": io_only / 1e6,
        "io_or_fused_only_ms": io_or_fused_only / 1e6,
        "non_io_window_ms": retained_denominator / 1e6,
        "gap_no_io_percent": ratio,
        "gap_below_threshold": gate_pass,
        "fused_work_policy": "entire fused compute+IO interval is productive and retained in the ratio denominator",
        # Compatibility keys carry the same definite value, never a bound range.
        "non_io_window_lower_ms": retained_denominator / 1e6,
        "non_io_window_upper_ms": retained_denominator / 1e6,
        "gap_no_io_percent_lower_bound": ratio,
        "gap_no_io_percent_upper_bound": ratio,
        "fused_io_separately_identifiable": not bool(fused),
        "threshold_percent": threshold_percent,
        "unresolved_gather_count": unresolved_gathers,
        "gate_certifiable": unresolved_gathers == 0,
        "gate_uncertainty": "gather calls lack per-call transport evidence"
        if unresolved_gathers
        else None,
        "conservative_gate_pass": gate_pass,
    }


def _scope_name(row):
    scope = row.get("scope")
    return scope["label"] if scope else None


def gap_details(activities, apis, start, end):
    productive = [
        (row["start"], row["end"])
        for row in activities
        if activity_lane(row) in {"Compute", "Compute + IO", "IO"}
    ]
    gpu_intervals = [(row["start"], row["end"]) for row in activities]
    output = []
    for left, right in complement(productive, start, end):
        before = [row for row in activities if row["end"] <= left]
        after = [row for row in activities if row["start"] >= right]
        previous = max(before, key=lambda row: row["end"]) if before else None
        following = min(after, key=lambda row: row["start"]) if after else None
        api_rows = [row for row in apis if row["start"] < right and row["end"] > left]
        controls = [
            row
            for row in activities
            if row["start"] < right and row["end"] > left and activity_lane(row) == "GPU control"
        ]
        api_union = _union_ns(
            clipped([(row["start"], row["end"]) for row in api_rows], left, right)
        )
        output.append(
            {
                "start_ns": left,
                "end_ns": right,
                "duration_ms": (right - left) / 1e6,
                "idle_ms": (right - left - _union_ns(clipped(gpu_intervals, left, right))) / 1e6,
                "cpu_cuda_api_union_ms": api_union / 1e6,
                "cpu_outside_cuda_api_ms": (right - left - api_union) / 1e6,
                "previous_gpu": source_activity(previous) if previous else None,
                "next_gpu": source_activity(following) if following else None,
                "previous_scope": _scope_name(previous) if previous else None,
                "next_scope": _scope_name(following) if following else None,
                "control_activities": [
                    {**source_activity(row), "scope": _scope_name(row)} for row in controls
                ],
                "cpu_cuda_apis": [
                    {
                        key: row.get(key)
                        for key in ("name", "start", "end", "thread", "correlation", "source")
                    }
                    | {"scope": _scope_name(row)}
                    for row in api_rows
                ],
            }
        )
    return sorted(output, key=lambda row: row["duration_ms"], reverse=True)


def activity_inventory(activities):
    groups = defaultdict(list)
    for row in activities:
        stage = row.get("graph_stage")
        if stage is None and row.get("scope"):
            stage = row["scope"]["stage"]
        groups[activity_lane(row), row["kind"], stage, row["name"]].append(row)
    return [
        {
            "lane": lane,
            "kind": kind,
            "stage": stage,
            "name": name,
            "count": len(rows),
            "duration_sum_ms": sum(row["end"] - row["start"] for row in rows) / 1e6,
        }
        for (lane, kind, stage, name), rows in sorted(groups.items(), key=lambda item: str(item[0]))
    ]


def host_scope_gap_totals(gaps, scopes, apis):
    """Partition observed gap time by the innermost currently active CPU scope.

    This is temporal attribution, not proof of causation: a queued GPU task may
    still be executing while the host has advanced to another scope.
    """
    totals = defaultdict(lambda: {"gap_ns": 0, "cuda_api_ns": 0})
    api_intervals = [(row["start"], row["end"]) for row in apis]
    for gap in gaps:
        start, end = gap["start_ns"], gap["end_ns"]
        overlapping = [scope for scope in scopes if scope["start"] < end and scope["end"] > start]
        boundaries = sorted(
            {start, end}
            | {max(start, scope["start"]) for scope in overlapping}
            | {min(end, scope["end"]) for scope in overlapping}
        )
        for left, right in pairwise(boundaries):
            covering = [scope for scope in overlapping if scope["start"] <= left < scope["end"]]
            scope = min(covering, key=lambda row: row["end"] - row["start"]) if covering else None
            key = scope["stage"] or "layer_misc" if scope else "outside_nvtx"
            totals[key]["gap_ns"] += right - left
            totals[key]["cuda_api_ns"] += _union_ns(clipped(api_intervals, left, right))
    return [
        {
            "stage": stage,
            "gap_ms": row["gap_ns"] / 1e6,
            "cuda_api_ms": row["cuda_api_ns"] / 1e6,
            "outside_cuda_api_ms": (row["gap_ns"] - row["cuda_api_ns"]) / 1e6,
        }
        for stage, row in sorted(totals.items(), key=lambda item: item[1]["gap_ns"], reverse=True)
    ]


def analyze_capture(path, calls, parents, *, threshold_percent=10.0, metrics=None):
    scopes, apis, activities, tables = _read_capture(path.resolve(), scope_pattern=_SCOPE)
    identities = {(scope["mode"], scope["phase"]) for scope in scopes}
    if len(identities) != 1:
        raise ValueError("expected one method and phase per capture")
    method, phase = identities.pop()
    if phase != "extend_annotated" or method not in METHODS:
        raise ValueError("gap audit expects a four-method extend capture")
    if len({row["device_id"] for row in activities}) != 1:
        raise ValueError("gap audit requires exactly one GPU")
    _assign_scopes(apis, scopes)
    _attribute(apis, activities)
    activities, capture_boundary = select_forward_activities(scopes, activities)
    selected_calls = [row for row in calls if (row["mode"], row["phase"]) == (method, phase)]
    graph_audit = None
    full_graph = any(row.get("full_extend_graph") for row in selected_calls)
    if full_graph:
        from experiments.deepseek_v32_mfu.src.full_graph_profile import attribute_full_graph_replays

        graph_audit = attribute_full_graph_replays(
            activities,
            selected_calls,
            parents,
            apis=select_forward_apis(apis, capture_boundary),
        )
    elif any(row.get("graph_replay") for row in selected_calls):
        graph_audit = attribute_graph_replays(
            activities, selected_calls, parents, scopes=scopes, require_replays=True
        )
    actual_io = annotate_actual_io(activities, metrics, method, phase)
    if any(row.get("scope") is None for row in activities):
        raise ValueError("unattributed GPU activity prevents complete gap classification")
    forward = [
        scope for scope in scopes if scope["layer"] == "shared" and scope["stage"] == "forward_misc"
    ]
    if len(forward) != 1:
        raise ValueError("expected exactly one complete forward NVTX range")
    start, end = forward[0]["start"], forward[0]["end"]
    gpu_start = min(row["start"] for row in activities)
    gpu_end = max(row["end"] for row in activities)
    if gpu_start < start or gpu_end > end:
        raise ValueError("complete forward NVTX range does not enclose all GPU work")
    layer_scopes = [
        scope for scope in scopes if scope["layer"].startswith("layer_") and scope["stage"] is None
    ]
    layer_ids = (
        sorted(
            {
                int(activity_layer(row).removeprefix("layer_"))
                for row in activities
                if (activity_layer(row) or "").startswith("layer_")
            }
        )
        if full_graph
        else sorted(int(scope["layer"].removeprefix("layer_")) for scope in layer_scopes)
    )
    if layer_ids != list(range(len(layer_ids))) or not layer_ids:
        raise ValueError("extend must contain each layer exactly once in contiguous order")
    layer_windows = []
    cursor = start
    for layer in layer_ids:
        if full_graph:
            compute = [
                row
                for row in activities
                if activity_layer(row) == f"layer_{layer}"
                and activity_lane(row) in {"Compute", "Compute + IO"}
            ]
            if not compute:
                raise ValueError("full extend graph layer has no captured computation")
            layer_end = max(row["end"] for row in compute)
        else:
            summary, _ = select_layer(scopes, apis, activities, layer=layer)
            layer_end = summary["end_ns"]
        layer_windows.append(
            {
                "layer": layer,
                "boundary": "forward start for L0, otherwise previous layer compute end; through layer compute end",
                **summarize_window(
                    activities, cursor, layer_end, threshold_percent=threshold_percent
                ),
            }
        )
        cursor = layer_end
    tail = summarize_window(activities, cursor, end, threshold_percent=threshold_percent)
    full = summarize_window(activities, start, end, threshold_percent=threshold_percent)
    partitions = layer_windows + [tail]
    if sum(row["end_ns"] - row["start_ns"] for row in partitions) != end - start:
        raise RuntimeError("layer and tail windows do not cover the complete forward")
    gaps = gap_details(activities, apis, start, end)
    host_gap_totals = host_scope_gap_totals(gaps, scopes, apis)
    if abs(sum(row["gap_ms"] for row in host_gap_totals) - full["gap_ms"]) > 1e-9:
        raise RuntimeError("host scope partition does not conserve complete gap duration")
    return {
        "method": method,
        "phase": "extend",
        "sqlite": path.name,
        "sqlite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tables_read": tables,
        "graph_attribution": graph_audit,
        "capture_boundary": capture_boundary,
        "full_extend": {
            "boundary": "complete forward NVTX scope, including GPU synchronization",
            **full,
        },
        "gpu_envelope": {
            "boundary": "first through last GPU activity; excludes host-only leading and trailing time",
            **summarize_window(activities, gpu_start, gpu_end, threshold_percent=threshold_percent),
        },
        "layers": layer_windows,
        "shared_tail": {"boundary": "last layer compute end through forward end", **tail},
        "all_layer_gates_pass": all(row["conservative_gate_pass"] for row in layer_windows),
        "gaps": gaps,
        "host_scope_gap_totals": host_gap_totals,
        "host_scope_note": "temporal partition by innermost CPU NVTX scope active during gap; not a causal attribution or estimate of removable Python work",
        "activity_inventory": activity_inventory(activities),
        "actual_io_evidence": actual_io,
    }


def aggregate_windows(windows):
    """Sum disjoint windows, preserving duration-weighted ratios across chunks."""
    if not windows:
        raise ValueError("cannot aggregate an empty set of windows")
    result = {key: sum(row[key] for row in windows) for key in windows[0] if key.endswith("_ms")}
    result.update(
        window_count=len(windows),
        gate_certifiable=all(row["gate_certifiable"] for row in windows),
        unresolved_gather_count=sum(row["unresolved_gather_count"] for row in windows),
        gap_percent=100 * result["gap_ms"] / result["window_ms"],
        gap_no_io_percent=(
            100 * result["gap_ms"] / result["non_io_window_ms"]
            if result["non_io_window_ms"]
            else None
        ),
        gap_no_io_percent_lower_bound=(
            100 * result["gap_ms"] / result["non_io_window_upper_ms"]
            if result["non_io_window_upper_ms"]
            else None
        ),
        gap_no_io_percent_upper_bound=(
            100 * result["gap_ms"] / result["non_io_window_lower_ms"]
            if result["non_io_window_lower_ms"]
            else None
        ),
    )
    return result


def analyze_prefill_capture(path, calls, parents, *, metrics=None):
    scopes, apis, activities, tables = _read_capture(path.resolve(), scope_pattern=_SCOPE)
    identities = {(scope["mode"], scope["phase"]) for scope in scopes}
    if len(identities) != 1:
        raise ValueError("expected one method and phase per prefill capture")
    method, phase = identities.pop()
    if phase != "prefill_annotated" or method not in METHODS:
        raise ValueError("prefill audit expects one of the four methods")
    if len({row["device_id"] for row in activities}) != 1:
        raise ValueError("prefill audit requires exactly one GPU")
    _assign_scopes(apis, scopes)
    _attribute(apis, activities)
    activities, capture_boundary = select_forward_activities(scopes, activities)
    actual_io = annotate_actual_io(activities, metrics, method, phase)
    selected_calls = [row for row in calls if (row["mode"], row["phase"]) == (method, phase)]
    graph_audit = None
    if any(row.get("graph_replay") for row in selected_calls):
        graph_audit = attribute_graph_replays(
            activities, selected_calls, parents, scopes=scopes, require_replays=True
        )
    if any(row.get("scope") is None for row in activities):
        raise ValueError("unattributed prefill GPU activity prevents complete classification")
    forward = [
        scope for scope in scopes if scope["layer"] == "shared" and scope["stage"] == "forward_misc"
    ]
    if len(forward) != 1:
        raise ValueError("expected exactly one complete prefill forward NVTX range")
    start, end = forward[0]["start"], forward[0]["end"]
    gpu_start = min(row["start"] for row in activities)
    gpu_end = max(row["end"] for row in activities)
    if gpu_start < start or gpu_end > end:
        raise ValueError("prefill forward range does not enclose every GPU activity")
    layer_scopes = sorted(
        (
            scope
            for scope in scopes
            if scope["layer"].startswith("layer_") and scope["stage"] is None
        ),
        key=lambda row: row["start"],
    )
    layer_ids = sorted({int(scope["layer"].removeprefix("layer_")) for scope in layer_scopes})
    if layer_ids != list(range(len(layer_ids))) or not layer_ids:
        raise ValueError("prefill layer IDs must be contiguous")
    layer_count = len(layer_ids)
    if len(layer_scopes) % layer_count or any(
        scope["layer"] != f"layer_{index % layer_count}" for index, scope in enumerate(layer_scopes)
    ):
        raise ValueError("prefill chunks must execute every layer in order")
    starts = [row["start"] for row in layer_scopes]
    compute_ends = defaultdict(int)
    for activity in activities:
        if activity_lane(activity) not in {"Compute", "Compute + IO"}:
            continue
        scope = activity["scope"]
        if not scope["layer"].startswith("layer_"):
            continue
        index = bisect.bisect_right(starts, scope["start"]) - 1
        if (
            index < 0
            or not layer_scopes[index]["start"] <= scope["start"] < layer_scopes[index]["end"]
        ):
            raise ValueError("prefill computation has no enclosing layer launch")
        if layer_scopes[index]["layer"] != scope["layer"]:
            raise ValueError("prefill computation differs from its enclosing layer")
        compute_ends[index] = max(compute_ends[index], activity["end"])
    if set(compute_ends) != set(range(len(layer_scopes))):
        raise ValueError("a prefill layer/chunk has no attributed computation")
    windows = []
    cursor = start
    for index in range(len(layer_scopes)):
        layer_end = compute_ends[index]
        windows.append(
            {
                "chunk": index // layer_count,
                "layer": index % layer_count,
                **summarize_window(activities, cursor, layer_end),
            }
        )
        cursor = layer_end
    tail = summarize_window(activities, cursor, end)
    full = summarize_window(activities, start, end)
    if abs(sum(row["gap_ms"] for row in windows + [tail]) - full["gap_ms"]) > 1e-7:
        raise RuntimeError("prefill layer/chunk windows do not conserve full gap")
    return {
        "method": method,
        "phase": "prefill",
        "sqlite": path.name,
        "sqlite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tables_read": tables,
        "graph_attribution": graph_audit,
        "capture_boundary": capture_boundary,
        "layer_count": layer_count,
        "chunk_count": len(layer_scopes) // layer_count,
        "full_prefill": full,
        "gpu_envelope": summarize_window(activities, gpu_start, gpu_end),
        "layers": [
            {"layer": layer, **aggregate_windows([row for row in windows if row["layer"] == layer])}
            for layer in layer_ids
        ],
        "last_chunk": windows[-layer_count:],
        "layer_chunk_windows": windows,
        "shared_tail": tail,
        "partition_boundary": "forward start through each successive layer compute completion, followed by shared tail; L0 includes each chunk's embedding/scheduling",
        "activity_inventory": activity_inventory(activities),
        "actual_io_evidence": actual_io,
    }


def analyze_run(directory, *, threshold_percent=10.0):
    directory = Path(directory).resolve(strict=True)
    result = json.loads((directory / "result.json").read_text())
    calls, _ = read_calls(directory / "operator_calls.json")
    setup, captures, prefill_paths = [], [], []
    for index, label in enumerate(result["nsys_capture_order"], 1):
        path = directory / f"capture_{index}.sqlite"
        if label == "graph_setup" or label.endswith("/extend_graph_setup"):
            setup.append(path)
        elif label in {f"{method}/extend_annotated" for method in METHODS}:
            captures.append(
                (
                    path,
                    result.get("measurements", {})
                    .get(label.split("/")[0], {})
                    .get("extend_cache_per_layer"),
                )
            )
        elif label in {f"{method}/prefill_annotated" for method in METHODS}:
            prefill_paths.append(
                (
                    path,
                    result.get("measurements", {})
                    .get(label.split("/")[0], {})
                    .get("prefix_cache_per_layer"),
                )
            )
    parents = read_lineage(setup) if setup else None
    rows = [
        analyze_capture(path, calls, parents, threshold_percent=threshold_percent, metrics=metrics)
        for path, metrics in captures
    ]
    if len(rows) != len(METHODS) or {row["method"] for row in rows} != set(METHODS):
        raise ValueError("one extend capture is required for every method")
    prefills = [
        analyze_prefill_capture(path, calls, parents, metrics=metrics)
        for path, metrics in prefill_paths
    ]
    if len(prefills) != len(METHODS) or {row["method"] for row in prefills} != set(METHODS):
        raise ValueError("one prefill capture is required for every method")
    reference = next(row["full_prefill"] for row in prefills if row["method"] == "hbm")
    reference_gap = reference["gap_ms"]
    reference_ratio = reference["gap_no_io_percent_upper_bound"]
    if reference_gap <= 0 or reference_ratio is None or reference_ratio <= 0:
        raise ValueError("prefill comparison requires a positive identifiable HBM gap")
    for row in prefills:
        full = row["full_prefill"]
        ratio = full["gap_no_io_percent_upper_bound"]
        row["absolute_gap_over_hbm"] = full["gap_ms"] / reference_gap
        row["normalized_no_io_gap_over_hbm"] = (
            ratio / reference_ratio if ratio is not None else None
        )
        certifiable = full["gate_certifiable"] and reference["gate_certifiable"]
        row["absolute_gap_at_most_120_percent_hbm"] = (
            certifiable and full["gap_ms"] <= 1.2 * reference_gap
        )
        row["normalized_gap_at_most_120_percent_hbm"] = (
            certifiable and ratio is not None and ratio <= 1.2 * reference_ratio
        )
    all_layers_pass = all(row["all_layer_gates_pass"] for row in rows)
    extend_pass = (
        all(row["full_extend"]["conservative_gate_pass"] for row in rows) and all_layers_pass
    )
    prefill_pass = all(row["absolute_gap_at_most_120_percent_hbm"] for row in prefills)
    return {
        "schema_version": 2,
        "run_id": result["run_id"],
        "sqlite_path_base": "profile run directory containing result.json",
        "input_result_sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest(),
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "classifier_sha256": hashlib.sha256(
            Path(__file__).with_name("timeline.py").read_bytes()
        ).hexdigest(),
        "gap_definition": "window minus union of model Compute, host IO, and indivisible Compute + IO; GPU cache/control and idle are included",
        "no_io_definition": "remove only standalone IO outside pure and fused computation; fused compute+IO is productive and fully retained in the denominator, yielding one definite ratio",
        "gate_definition": "each complete extend and every layer must have a definite no-IO gap ratio strictly below threshold; complete extend includes startup and synchronization/commit; L0 includes forward startup; each complete offload prefill's absolute gap is at most 1.2 times HBM-only",
        "prefill_comparison_note": "absolute gap ratio and normalized non-IO gap ratio are distinct; both reported, primary prefill gate uses absolute gap at matched workload",
        "actual_io_definition": "same measured phase-layer record counters prove all matching calls empty when the total is zero; positive exclusive totals bind only a unique node; unresolved gather classification cannot certify an overlapping gate; fused calls remain productive in their entirety",
        "measurement_boundary": "single intrusive node-traced profile per method; includes profiler overhead; not independent benchmark wall time or SM utilization",
        "timed_output": result.get("timed_output"),
        "all_methods_pass": extend_pass and prefill_pass,
        "extend_gate_pass": extend_pass,
        "all_layers_gate_pass": all_layers_pass,
        "prefill_absolute_gap_gate_pass": prefill_pass,
        "methods": rows,
        "prefill_methods": prefills,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-percent", type=float, default=10.0)
    args = parser.parse_args()
    if not 0 < args.threshold_percent < 100:
        parser.error("threshold must be between 0 and 100 percent")
    result = analyze_run(args.profile_run, threshold_percent=args.threshold_percent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "all_methods_pass": result["all_methods_pass"]}))


if __name__ == "__main__":
    main()
