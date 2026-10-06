"""Measured CPU control, GPU computation, and IO for one checkpoint layer."""

from __future__ import annotations

import re
from itertools import pairwise

from experiments.deepseek_v32_mfu.src.analyze_nsys import (
    _assign_scopes,
    _attribute,
    _read_capture,
    _union,
    _union_ns,
    kernel_category,
)
from experiments.deepseek_v32_mfu.src.operator_report import _SCOPE


def intersection_ns(left, right):
    left, right = _union(left), _union(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def activity_stage(activity):
    scope = activity.get("scope") or {}
    return activity.get("graph_stage", scope.get("stage"))


def activity_layer(activity):
    """GPU capture ownership is distinct from the replay's actual CPU scope."""
    stage = activity_stage(activity) or ""
    target = re.fullmatch(r"dense_history_prefetch_layer_(\d+)", stage)
    if target:
        return f"layer_{target[1]}"
    scope = activity.get("scope") or {}
    return activity.get("graph_layer", scope.get("layer"))


def prefetch_transport(activity):
    if activity["kind"] == "memcpy" and activity["name"].lower() in {
        "host-to-device",
        "unified host-to-device",
    }:
        return "cuda_memcpy_h2d"
    if (
        activity["kind"] == "kernel"
        and kernel_category(activity["name"]) == "mapped_host_kv_gather"
    ):
        return "mapped_host_gather"
    return None


def source_activity(activity):
    """Keep the complete transport record and its launch, outside display clipping."""
    return {
        key: activity.get(key)
        for key in (
            "kind",
            "name",
            "start",
            "end",
            "device_id",
            "stream_id",
            "process",
            "correlation",
            "bytes",
            "graph_id",
            "graph_node_id",
            "api",
        )
    } | ({"actual_io": activity["actual_io"]} if "actual_io" in activity else {})


def phase_prefetch_totals(scopes, activities):
    """Count record transport in each layer's prefetch scopes over the whole phase."""
    totals = {}
    for scope in scopes:
        match = re.fullmatch(r"dense_history_prefetch_layer_(\d+)", scope["stage"] or "")
        if match:
            totals.setdefault(
                int(match[1]),
                {"h2d_memcpy_bytes": 0, "h2d_memcpy_count": 0, "mapped_gather_count": 0},
            )
    for activity in activities:
        scope = activity.get("scope")
        match = (
            re.fullmatch(r"dense_history_prefetch_layer_(\d+)", scope["stage"] or "")
            if scope
            else None
        )
        if match:
            row = totals[int(match[1])]
            transport = prefetch_transport(activity)
            if transport == "cuda_memcpy_h2d":
                if type(activity["bytes"]) is not int or activity["bytes"] <= 0:
                    raise ValueError("dense H2D activity lacks a positive native byte count")
                row["h2d_memcpy_bytes"] += activity["bytes"]
                row["h2d_memcpy_count"] += 1
            elif transport == "mapped_host_gather":
                row["mapped_gather_count"] += 1
    return totals


def activity_lane(activity):
    category = kernel_category(activity["name"])
    evidence = activity.get("actual_io")
    if evidence is not None and evidence.get("status") == "phase_zero":
        if (
            activity["kind"] != "kernel"
            or evidence.get("records") != 0
            or evidence.get("family") != category
            or category not in {"mapped_host_kv_gather", "indexer_fused_prefetch"}
        ):
            raise ValueError("zero IO annotation does not match its transport kernel")
        return "GPU control" if category == "mapped_host_kv_gather" else "Compute"
    scope_stage = activity["scope"]["stage"] if activity.get("scope") else ""
    stage = activity.get("graph_stage", scope_stage) or ""
    if category == "indexer_fused_prefetch":
        return "Compute + IO"
    if category == "mapped_host_kv_gather":
        return "IO"
    if activity["kind"] == "memcpy":
        # _read_capture obtains this label from ENUM_CUDA_MEMCPY_OPER.copyKind.
        # A D2D copy prepares device state; it is not host communication.
        return (
            "IO"
            if activity["name"].lower()
            in {
                "host-to-device",
                "device-to-host",
                "host-to-array",
                "array-to-host",
                "unified host-to-device",
                "unified device-to-host",
            }
            else "GPU control"
        )
    if activity["kind"] == "memset":
        return "GPU control"
    name = activity["name"]
    if (
        "gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda" in name
        or "CatArrayBatchedCopy" in name
        or "deep_gemm::transpose_fp32<" in name
        or "FillFunctor<" in name
    ):
        # Same-type contiguous copies, concatenation and scale transposes only
        # rearrange storage, even inside an operator or a captured graph.
        # LoadWithCast/StoreWithCast and bfloat16_copy perform numeric conversion.
        return "GPU control"
    if "arange_cuda_out" in name:
        # These kernels prepare local RoPE-table IDs or causal-mask indices.
        # The subsequent trig/rotation and causal masking remain computation.
        return "GPU control"
    projection_graph = scope_stage and scope_stage.startswith("compute_graph_projection_layer_")
    if projection_graph and (
        "CUDAFunctorOnSelf_add<long>" in name
        or (
            "unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda" in name
            and "[lambda(int)" in name
            and "LoadWithCast" in name
            and "StoreWithCast" in name
        )
    ):
        # The captured projection's integer +1 and destination-int cast form
        # exclusive causal ends. Floating position/value casts stay compute.
        return "GPU control"
    if stage == "prefetch_hint" or (
        stage == "exact_topk"
        and (
            activity["name"] in {"mask_count", "publish"}
            or "at::native::reduce_kernel" in activity["name"]
        )
    ):
        # Older traces put prediction-hint maintenance in the top-k scope.
        # The reduction computes a future-prefetch hint, not exact selection.
        return "GPU control"
    if stage == "indexer_fused" and "echo_native::clean(" in activity["name"]:
        # This kernel applies the causal -inf score mask, not cache metadata.
        return "Compute"
    if (
        activity.get("graph_call") is not None
        or stage.startswith("compute_graph_")
        and ("projection_layer_" in stage or "finish_layer_" in stage)
    ):
        return "Compute"
    if stage in {
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj",
        "index_q_proj",
        "index_k_proj",
        "o_proj",
        "index_weights_proj",
        "q_absorb",
        "v_expand",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
        "mla_qk_pv",
        "sparse_mla",
        "indexer_qk",
        "exact_topk",
        "topk",
        "mlp",
        "rms_norm",
        "residual_rms_norm",
        "quantize_index",
        "apply_rope_pair",
        "apply_rope",
        "embedding",
        "lm_head",
        "final_norm_lm_head",
        "attention_projection",
        "attention_output",
        "compute_graph_value_expansion",
    }:
        return "Compute"
    return "GPU control"


def activity_rows(activities, apis, start, end):
    """Keep measured endpoints, actual CPU scope and capture-time GPU ownership."""
    rows = []
    for activity in [*activities, *apis]:
        if activity["start"] >= end or activity["end"] <= start:
            continue
        scope = activity.get("scope") or {}
        kind = activity.get("kind", "api")
        stage = activity_stage(activity)
        rows.append(
            {
                "lane": "CPU CUDA API" if kind == "api" else activity_lane(activity),
                "kind": kind,
                "name": activity["name"],
                "scope_label": scope.get("label"),
                "stage": stage,
                "source_stage": activity.get("graph_source_stage", stage),
                "scope_path": activity.get("graph_scope_path", []),
                "layer": activity_layer(activity),
                "actual_io": activity.get("actual_io"),
                "correlation": activity.get("correlation"),
                "stream": activity.get("stream_id"),
                "device_id": activity.get("device_id"),
                "process": activity.get("process"),
                "graph_id": activity.get("graph_id"),
                "graph_node_id": activity.get("graph_node_id"),
                "bytes": activity.get("bytes"),
                "start_ns": max(start, activity["start"]),
                "end_ns": min(end, activity["end"]),
                "raw_start_ns": activity["start"],
                "raw_end_ns": activity["end"],
                "start_ms": (max(start, activity["start"]) - start) / 1e6,
                "end_ms": (min(end, activity["end"]) - start) / 1e6,
            }
        )
    return rows


def _select_graph_layer(scopes, apis, activities, *, layer, occurrence):
    from experiments.deepseek_v32_mfu.src.launch_gap import complement, summarize_window

    if occurrence not in (-1, 0):
        raise ValueError("one full extend graph contains exactly one occurrence of each layer")
    compute = [row for row in activities if activity_lane(row) in {"Compute", "Compute + IO"}]
    owned = [row for row in compute if activity_layer(row) == f"layer_{layer}"]
    if not owned:
        raise ValueError(f"full graph contains no computation for layer {layer}")
    if layer == 0:
        forward = [
            row for row in scopes if row["layer"] == "shared" and row["stage"] == "forward_misc"
        ]
        if len(forward) != 1:
            raise ValueError("full graph requires one complete measured forward boundary")
        start, boundary = forward[0]["start"], "complete forward start"
    else:
        previous = [row for row in compute if activity_layer(row) == f"layer_{layer - 1}"]
        if not previous:
            raise ValueError("full graph lacks previous layer computation")
        start, boundary = max(row["end"] for row in previous), "previous layer computation end"
    end = max(row["end"] for row in owned)
    if start > min(row["start"] for row in owned):
        raise ValueError("full graph layer computation overlaps its predecessor")
    visible = [row for row in activities if row["start"] < end and row["end"] > start]
    rows = activity_rows(visible, apis, start, end)
    gpu = [(max(start, row["start"]), min(end, row["end"])) for row in visible]
    pure_compute = [(row["start_ns"], row["end_ns"]) for row in rows if row["lane"] == "Compute"]
    io = [(row["start_ns"], row["end_ns"]) for row in rows if row["lane"] == "IO"]
    transfers = [
        row
        for row in activities
        if activity_stage(row) == f"dense_history_prefetch_layer_{layer + 1}"
        and activity_lane(row) == "IO"
    ]
    transfer_intervals = [(row["start"], row["end"]) for row in transfers]
    transfer_duration = _union_ns(transfer_intervals)
    window = summarize_window(visible, start, end)
    summary = {
        **window,
        "layer": layer,
        "occurrence": occurrence,
        "origin_ns": start,
        "origin_boundary": boundary,
        "boundary": f"{boundary} through layer GPU compute completion",
        "layer_ownership": "capture-time GPU node ownership; no replay-time CPU layer scopes",
        "gpu_busy_union_ms": _union_ns(gpu) / 1e6,
        "gpu_gap_ms": window["gpu_idle_ms"],
        "compute_io_gap_ms": window["gap_ms"],
        "largest_gpu_gap_ms": max((b - a for a, b in complement(gpu, start, end)), default=0) / 1e6,
        "compute_io_overlap_ms": intersection_ns(pure_compute, io) / 1e6,
        "next_layer_prefetch_overlap_ms": intersection_ns(pure_compute, transfer_intervals) / 1e6,
        "next_layer_prefetch_full_ms": transfer_duration / 1e6,
        "next_layer_prefetch_span_ms": (
            max(b for _, b in transfer_intervals) - min(a for a, _ in transfer_intervals)
        )
        / 1e6
        if transfer_intervals
        else 0,
        "next_layer_prefetch_h2d_memcpy_bytes": sum(
            row.get("bytes", 0) for row in transfers if row["kind"] == "memcpy"
        ),
        "next_layer_prefetch_transports": sorted({prefetch_transport(row) for row in transfers}),
        "next_layer_prefetch_activities": [source_activity(row) for row in transfers],
        "next_layer_prefetch_zero_transfer_activities": [],
        "next_layer_prefetch_unresolved_gather_count": 0,
        "next_layer_prefetch_overlap_certifiable": True,
        "next_layer_prefetch_overlap_percent": 100
        * intersection_ns(pure_compute, transfer_intervals)
        / transfer_duration
        if transfer_duration
        else None,
        "next_layer_prefetch_intervals_ns": transfer_intervals,
        "next_layer_prefetch_intervals_ms": [
            ((a - start) / 1e6, (b - start) / 1e6) for a, b in transfer_intervals
        ],
        "next_layer_prefetch_streams": sorted({row["stream_id"] for row in transfers}),
        "next_layer_prefetch_started_before_window": any(a < start for a, _ in transfer_intervals),
        "next_layer_prefetch_ends_after_window": any(b > end for _, b in transfer_intervals),
        "next_layer_prefetch_scope_label": None,
        "next_layer_prefetch_scope": None,
        "unattributed_gpu_activity_count": sum(row.get("scope") is None for row in visible),
        "fused_note": "Fused ECHO is productive in its entirety and retained in the gap denominator.",
    }
    return summary, rows


def select_layer(scopes, apis, activities, *, layer=1, occurrence=-1):
    """Select one launch scope, and include every activity overlapping its GPU span.

    Other-stream prefetch remains visible even when its CPU launch precedes the
    selected layer. CPU ranges are clipped to the same measured time window;
    they are independent lanes, never subtracted from another run's wall time.
    """
    from experiments.deepseek_v32_mfu.src.launch_gap import summarize_window

    if any(row.get("full_extend_graph") for row in activities):
        return _select_graph_layer(scopes, apis, activities, layer=layer, occurrence=occurrence)

    layers = sorted(
        (
            scope
            for scope in scopes
            if scope["layer"] == f"layer_{layer}" and scope["stage"] is None
        ),
        key=lambda scope: scope["start"],
    )
    if not layers:
        raise ValueError(f"capture has no layer {layer} launch scope")
    chosen = layers[occurrence]
    owned = [
        activity
        for activity in activities
        if activity.get("scope") is not None
        and chosen["start"] <= activity["scope"]["start"] < chosen["end"]
        and activity["scope"]["layer"] == f"layer_{layer}"
        and activity_lane(activity) in {"Compute", "Compute + IO"}
    ]
    if not owned:
        raise ValueError("selected layer has no attributed compute activity")
    # L0 starts at the complete forward boundary, or at the previous chunk's
    # final computation. Later layers start at their predecessor's computation.
    previous = [
        scope
        for scope in scopes
        if (
            scope["layer"].startswith("layer_")
            if layer == 0
            else scope["layer"] == f"layer_{layer - 1}"
        )
        and scope["stage"] is None
        and scope["end"] <= chosen["start"]
    ]
    origin = min(activity["start"] for activity in owned)
    origin_boundary = "first owned computation"
    if previous:
        previous = max(previous, key=lambda scope: scope["end"])
        previous_compute = [
            activity
            for activity in activities
            if activity.get("scope") is not None
            and previous["start"] <= activity["scope"]["start"] < previous["end"]
            and activity["scope"]["layer"] == previous["layer"]
            and activity_lane(activity) in {"Compute", "Compute + IO"}
        ]
        if previous_compute:
            origin = max(activity["end"] for activity in previous_compute)
            origin_boundary = (
                "previous chunk computation end" if layer == 0 else "previous layer computation end"
            )
        elif layer == 0:
            raise ValueError("L0 timeline cannot identify the previous chunk computation")
    elif layer == 0:
        forward = [
            scope
            for scope in scopes
            if scope["layer"] == "shared"
            and scope["stage"] == "forward_misc"
            and scope["start"] <= chosen["start"]
            and chosen["end"] <= scope["end"]
        ]
        if len(forward) != 1:
            raise ValueError("L0 timeline requires the complete measured forward boundary")
        origin = forward[0]["start"]
        origin_boundary = "complete forward start"
    end = max(activity["end"] for activity in owned)
    if end <= origin:
        raise ValueError("selected GPU layer window is empty")
    next_scopes = [
        scope
        for scope in scopes
        if scope["stage"] == f"dense_history_prefetch_layer_{layer + 1}"
        and scope["end"] <= chosen["start"]
    ]
    next_scope = max(next_scopes, key=lambda scope: scope["start"]) if next_scopes else None
    next_transport_calls = [
        activity
        for activity in activities
        if next_scope is not None
        and activity.get("scope") == next_scope
        and prefetch_transport(activity) is not None
    ]
    next_transfers = [item for item in next_transport_calls if activity_lane(item) == "IO"]
    unresolved_next_gathers = sum(
        item.get("actual_io", {}).get("status") == "unresolved" for item in next_transfers
    )
    next_intervals = [(activity["start"], activity["end"]) for activity in next_transfers]
    rows = []
    visible = [
        activity for activity in activities if activity["end"] > origin and activity["start"] < end
    ]
    for activity in visible:
        scope = activity.get("scope")
        stage = activity_stage(activity)
        rows.append(
            {
                "lane": activity_lane(activity),
                "kind": activity["kind"],
                "start_ns": max(origin, activity["start"]),
                "end_ns": min(end, activity["end"]),
                "raw_start_ns": activity["start"],
                "raw_end_ns": activity["end"],
                "stream": activity["stream_id"],
                "device_id": activity.get("device_id"),
                "process": activity.get("process"),
                "scope_label": scope.get("label") if scope else None,
                "correlation": activity.get("correlation"),
                "graph_id": activity.get("graph_id"),
                "graph_node_id": activity.get("graph_node_id"),
                "stage": stage,
                "source_stage": activity.get("graph_source_stage", stage),
                "scope_path": activity.get("graph_scope_path", []),
                "layer": activity_layer(activity),
                "name": activity["name"],
                "io_kind": "mapped_host_gather"
                if kernel_category(activity["name"]) == "mapped_host_kv_gather"
                else activity["name"]
                if activity["kind"] == "memcpy"
                else None,
                "bytes": activity.get("bytes"),
                "actual_io": activity.get("actual_io"),
            }
        )
    for api in apis:
        if api["end"] > origin and api["start"] < end:
            rows.append(
                {
                    "lane": "CPU CUDA API",
                    "kind": "api",
                    "start_ns": max(origin, api["start"]),
                    "end_ns": min(end, api["end"]),
                    "raw_start_ns": api["start"],
                    "raw_end_ns": api["end"],
                    "stream": None,
                    "device_id": None,
                    "process": api.get("process"),
                    "scope_label": api["scope"].get("label") if api.get("scope") else None,
                    "correlation": api.get("correlation"),
                    "graph_id": None,
                    "graph_node_id": None,
                    "stage": api["scope"]["stage"] if api.get("scope") else None,
                    "layer": api["scope"]["layer"] if api.get("scope") else None,
                    "name": api["name"],
                    "io_kind": None,
                    "bytes": None,
                    "actual_io": None,
                }
            )
    for scope in scopes:
        if scope["stage"] is not None and scope["end"] > origin and scope["start"] < end:
            rows.append(
                {
                    "lane": "CPU control",
                    "kind": "nvtx",
                    "start_ns": max(origin, scope["start"]),
                    "end_ns": min(end, scope["end"]),
                    "raw_start_ns": scope["start"],
                    "raw_end_ns": scope["end"],
                    "stream": None,
                    "device_id": None,
                    "process": None,
                    "scope_label": scope.get("label"),
                    "correlation": None,
                    "graph_id": None,
                    "graph_node_id": None,
                    "stage": scope["stage"],
                    "layer": scope["layer"],
                    "name": scope["stage"],
                    "io_kind": None,
                    "bytes": None,
                    "actual_io": None,
                }
            )
    gpu = [(max(origin, item["start"]), min(end, item["end"])) for item in visible]
    compute = [
        (max(origin, activity["start"]), min(end, activity["end"]))
        for activity in visible
        if activity_lane(activity) == "Compute"
    ]
    io = [(row["start_ns"], row["end_ns"]) for row in rows if row["lane"] == "IO"]
    window_metrics = summarize_window(visible, origin, end)
    target_io = next_intervals
    merged = _union(gpu)
    gaps = [(a[1], b[0]) for a, b in pairwise(merged)]
    if merged and merged[0][0] > origin:
        gaps.insert(0, (origin, merged[0][0]))
    for row in rows:
        row["start_ms"] = (row["start_ns"] - origin) / 1e6
        row["end_ms"] = (row["end_ns"] - origin) / 1e6
    return {
        **window_metrics,
        "layer": layer,
        "occurrence": occurrence,
        "origin_ns": origin,
        "origin_boundary": origin_boundary,
        "end_ns": end,
        "window_ms": (end - origin) / 1e6,
        "gpu_busy_union_ms": _union_ns(gpu) / 1e6,
        "gpu_gap_ms": window_metrics["gpu_idle_ms"],
        "gpu_gap_definition": "window minus union of all GPU activities; idle only",
        "compute_io_gap_ms": window_metrics["gap_ms"],
        "compute_io_gap_definition": "window minus union of model computation, IO and fused compute/IO; cache/control and idle remain gap",
        "largest_gpu_gap_ms": max((b - a for a, b in gaps), default=0) / 1e6,
        "compute_io_overlap_ms": intersection_ns(compute, io) / 1e6,
        "next_layer_prefetch_overlap_ms": intersection_ns(compute, target_io) / 1e6,
        "next_layer_prefetch_full_ms": _union_ns(next_intervals) / 1e6,
        "next_layer_prefetch_span_ms": (
            max(end for _, end in next_intervals) - min(start for start, _ in next_intervals)
        )
        / 1e6
        if next_intervals
        else 0.0,
        "next_layer_prefetch_h2d_memcpy_bytes": sum(
            activity["bytes"] for activity in next_transfers if activity["kind"] == "memcpy"
        ),
        "next_layer_prefetch_transports": sorted(
            {prefetch_transport(activity) for activity in next_transfers}
        ),
        "next_layer_prefetch_activities": [source_activity(item) for item in next_transfers],
        "next_layer_prefetch_zero_transfer_activities": [
            source_activity(item)
            for item in next_transport_calls
            if activity_lane(item) == "GPU control"
        ],
        "next_layer_prefetch_unresolved_gather_count": unresolved_next_gathers,
        "next_layer_prefetch_overlap_certifiable": unresolved_next_gathers == 0,
        "next_layer_prefetch_overlap_percent": 100
        * intersection_ns(compute, next_intervals)
        / _union_ns(next_intervals)
        if _union_ns(next_intervals)
        else None,
        "next_layer_prefetch_intervals_ns": next_intervals,
        "next_layer_prefetch_intervals_ms": [
            [(a - origin) / 1e6, (b - origin) / 1e6] for a, b in next_intervals
        ],
        "next_layer_prefetch_streams": sorted(
            {activity["stream_id"] for activity in next_transfers}
        ),
        "next_layer_prefetch_started_before_window": any(a < origin for a, _ in next_intervals),
        "next_layer_prefetch_ends_after_window": any(b > end for _, b in next_intervals),
        "next_layer_prefetch_scope_label": next_scope.get("label") if next_scope else None,
        "next_layer_prefetch_scope": next_scope,
        "overlap_boundary": "selected layer's standalone compute kernels intersect host-IO activity; next-layer metric uses the complete H2D memcpy or mapped-host gather activity belonging to this chunk's next-layer prefetch scope",
        "unattributed_gpu_activity_count": sum(
            activity.get("scope") is None for activity in visible
        ),
        "boundary": (
            f"{origin_boundary} through selected layer GPU compute completion"
            if layer == 0
            else "previous layer GPU compute completion through selected layer GPU compute completion"
        ),
        "fused_note": "Fused ECHO kernels with proven zero phase transfer are Compute; all others remain Compute + IO without inferring separate IO intervals.",
    }, rows


def select_forward_activities(scopes, activities):
    """Exclude explicitly separate traced warmup/setup, never in-window work."""
    forward = [
        scope for scope in scopes if scope["layer"] == "shared" and scope["stage"] == "forward_misc"
    ]
    if len(forward) != 1:
        raise ValueError("capture must identify exactly one measured forward NVTX scope")
    start, end = forward[0]["start"], forward[0]["end"]
    selected, excluded = [], []
    for activity in activities:
        scope = activity.get("scope")
        if scope is not None and start <= scope["start"] < end:
            if activity["start"] < start or activity["end"] > end:
                raise ValueError("measured GPU activity escapes synchronized forward scope")
            selected.append(activity)
        else:
            if activity["start"] < end and activity["end"] > start:
                raise ValueError("warmup/setup or unattributed GPU work overlaps measured forward")
            excluded.append(activity)
    if not selected:
        raise ValueError("measured forward has no attributed GPU work")
    return selected, {
        "measured_start_ns": start,
        "measured_end_ns": end,
        "capture_activity_count": len(activities),
        "measured_activity_count": len(selected),
        "outside_window_activity_count": len(excluded),
        "outside_window_activity_duration_sum_ms": sum(
            row["end"] - row["start"] for row in excluded
        )
        / 1e6,
        "outside_window_activity_overlap_ns": 0,
        "boundary": "only explicit measured forward; all excluded GPU work proved nonoverlapping; no measured leading/trailing gap removed",
    }


def select_forward_apis(apis, boundary):
    """Match graph-launch validation to the measured synchronized forward."""
    start, end = boundary["measured_start_ns"], boundary["measured_end_ns"]
    selected = [api for api in apis if api["start"] < end and api["end"] > start]
    for api in selected:
        if "GraphLaunch" in api["name"] and (
            api["start"] < start or api["end"] > end or api.get("scope") is None
        ):
            raise ValueError("measured graph launch must be fully inside an attributed forward")
    return selected


def extract(path, calls, parents, *, layer=1, measurements=None):
    from experiments.deepseek_v32_mfu.src.launch_gap import annotate_actual_io

    scopes, apis, activities, _ = _read_capture(path.resolve(), scope_pattern=_SCOPE)
    _assign_scopes(apis, scopes)
    _attribute(apis, activities)
    activities, capture_boundary = select_forward_activities(scopes, activities)
    identity = {(scope["mode"], scope["phase"]) for scope in scopes}
    if len(identity) != 1:
        raise ValueError("timeline requires one method and phase")
    method, phase = identity.pop()
    selected_calls = [row for row in calls if (row["mode"], row["phase"]) == (method, phase)]
    if any(row.get("full_extend_graph") for row in selected_calls):
        from experiments.deepseek_v32_mfu.src.full_graph_profile import attribute_full_graph_replays

        attribute_full_graph_replays(
            activities,
            selected_calls,
            parents,
            scopes=scopes,
            apis=select_forward_apis(apis, capture_boundary),
        )
    elif any(row.get("graph_replay") for row in selected_calls):
        from experiments.deepseek_v32_motivation.src.graph_attribution import (
            attribute_graph_replays,
        )

        attribute_graph_replays(
            activities, selected_calls, parents, scopes=scopes, require_replays=True
        )
    phase_key = (
        "prefix_cache_per_layer" if phase == "prefill_annotated" else "extend_cache_per_layer"
    )
    actual_io = annotate_actual_io(
        activities,
        measurements[method][phase_key] if measurements is not None else None,
        method,
        phase,
    )
    summary, rows = select_layer(scopes, apis, activities, layer=layer)
    return {
        "method": method,
        "phase": phase.removesuffix("_annotated"),
        "phase_dense_prefetch_totals": phase_prefetch_totals(scopes, activities),
        "capture_boundary": capture_boundary,
        "actual_io_evidence": actual_io,
        **summary,
    }, [{"method": method, "phase": phase.removesuffix("_annotated"), **row} for row in rows]


def computation_segments(rows):
    """Name consecutive compute families without changing any activity interval.

    The returned endpoints position labels only: their span is not a measured
    compute duration. Original kernel bars, including internal gaps, stay intact.
    """
    segments = []
    for row in sorted(rows, key=lambda item: item["start_ms"]):
        if row["lane"] not in {"Compute", "Compute + IO"}:
            continue
        stage = row.get("stage") or ""
        source_stage = row.get("source_stage", stage)
        scope = row.get("scope_label") or ""
        if stage == "embedding":
            label = "Embedding"
        elif source_stage in {"input_residual_norm", "attention_projection"}:
            label = "Projection / RoPE"
        elif source_stage in {"post_attention_residual_norm", "dense_mlp", "attention_output"}:
            label = "Output / MLP"
        elif "compute_graph_projection_layer_" in scope:
            label = "Projection / RoPE"
        elif "compute_graph_finish_layer_" in scope:
            label = "Output / MLP"
        elif stage == "indexer_fused":
            label = "Indexer logits" if row["lane"] == "Compute" else "Indexer + prefetch"
        elif stage in {"indexer_qk", "indexer"}:
            label = "Indexer logits"
        elif stage in {"exact_topk", "topk"}:
            label = "Exact top-k"
        elif stage in {"mla_qk_pv", "sparse_mla"}:
            label = "Sparse MLA"
        elif stage in {"v_expand", "compute_graph_value_expansion"}:
            label = "V expand"
        elif stage in {"o_proj", "mlp", "mlp_gate", "mlp_up", "mlp_down"}:
            label = "Output / MLP"
        elif stage in {"lm_head", "final_norm_lm_head"}:
            label = {
                "lm_head": "LM head",
                "final_norm_lm_head": "Final norm",
            }[stage]
        elif stage in {"rms_norm", "residual_rms_norm"}:
            label = "Norm / residual"
        elif stage in {
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj",
            "index_q_proj",
            "index_k_proj",
            "index_weights_proj",
            "q_absorb",
            "quantize_index",
            "apply_rope_pair",
            "apply_rope",
        }:
            label = "Projection / RoPE"
        else:
            label = stage.replace("_", " ") or "Model compute"
        if segments and segments[-1]["label"] == label:
            segments[-1]["end_ms"] = max(segments[-1]["end_ms"], row["end_ms"])
            segments[-1]["kernel_count"] += 1
        else:
            segments.append(
                {
                    "label": label,
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "kernel_count": 1,
                }
            )
    return segments


def _annotate_computation(axis, segments, *, left, right, y):
    # Keep labels in temporal order and separate their text boxes. Connectors
    # point to the actual kernels, so tiny top-k/V-expand segments remain named.
    if not segments:
        return
    width = right - left
    spacing = width * min(0.13, 0.87 / max(1, len(segments) - 1))
    centers = [(row["start_ms"] + row["end_ms"]) / 2 for row in segments]
    text_centers = []
    for index, center in enumerate(centers):
        minimum = text_centers[-1] + spacing if text_centers else left + width * 0.065
        maximum = right - width * 0.065 - (len(segments) - index - 1) * spacing
        text_centers.append(min(maximum, max(minimum, center)))
    for segment, center, text_center in zip(segments, centers, text_centers, strict=True):
        axis.annotate(
            segment["label"],
            xy=(center, y + 0.29),
            xytext=(text_center, y + 0.79),
            fontsize=8,
            ha="center",
            va="bottom",
            color="#263442",
            arrowprops={"arrowstyle": "-", "color": "#73808C", "lw": 0.55},
            annotation_clip=False,
        )


def draw(summaries, rows, output, *, phase):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {
        "Compute": "#0072B2",
        "Compute + IO": "#8F4C9B",
        "IO": "#D18A21",
        "GPU control": "#8B929A",
        "CPU CUDA API": "#608D49",
    }
    lanes = ["Compute", "IO", "GPU control", "CPU CUDA API"]
    chosen = [summary for summary in summaries if summary["phase"] == phase]
    display_left = min(
        [0.0]
        + [left for summary in chosen for left, _ in summary["next_layer_prefetch_intervals_ms"]]
    )
    display_right = max(
        [summary["window_ms"] for summary in chosen]
        + [right for summary in chosen for _, right in summary["next_layer_prefetch_intervals_ms"]]
    )
    fig, axes = plt.subplots(len(chosen), 1, figsize=(14, 11.5), squeeze=False)
    for axis, summary in zip(axes[:, 0], chosen, strict=True):
        selected_rows = [
            row for row in rows if (row["method"], row["phase"]) == (summary["method"], phase)
        ]
        for row in selected_rows:
            if row["lane"] == "CPU control":
                continue
            lane = "Compute" if row["lane"] == "Compute + IO" else row["lane"]
            y = len(lanes) - 1 - lanes.index(lane)
            axis.broken_barh(
                [(row["start_ms"], row["end_ms"] - row["start_ms"])],
                (y - 0.28, 0.56),
                facecolor=colors[row["lane"]],
                hatch="//" if row["lane"] == "Compute + IO" else None,
                edgecolor="#30343B",
                linewidth=0,
            )
        for start, end in summary["next_layer_prefetch_intervals_ms"]:
            y = len(lanes) - 1 - lanes.index("IO")
            axis.broken_barh(
                [(start, end - start)], (y - 0.28, 0.56), color=colors["IO"], linewidth=0
            )
        axis.set_yticks(range(len(lanes)), list(reversed(lanes)), fontsize=8)
        axis.set_xlim(display_left * 1.02, display_right * 1.02)
        axis.set_ylim(-0.5, len(lanes) + 0.48)
        _annotate_computation(
            axis,
            computation_segments(selected_rows),
            left=display_left,
            right=display_right,
            y=len(lanes) - 1,
        )
        axis.axvline(0, color="#4B5563", linewidth=0.6, alpha=0.4)
        axis.axvline(summary["window_ms"], color="#4B5563", linewidth=0.6, alpha=0.4, linestyle=":")
        overlap = (
            f"L{summary['layer'] + 1} prefetch overlap {summary['next_layer_prefetch_overlap_ms']:.3f} ms"
            if summary["method"] == "dense_prefetch"
            else f"separate IO overlap {summary['compute_io_overlap_ms']:.3f} ms"
        )
        lower = summary["gap_no_io_percent_lower_bound"]
        upper = summary["gap_no_io_percent_upper_bound"]
        gap_ratio = (
            "undefined"
            if lower is None or upper is None
            else f"{lower:.1f}–{upper:.1f}%"
            if abs(lower - upper) > 1e-9
            else f"{upper:.1f}%"
        )
        if not summary["gate_certifiable"]:
            gap_ratio += "; unverified gather IO"
        if not summary["next_layer_prefetch_overlap_certifiable"]:
            overlap += " (unverified gather IO)"
        axis.set_title(
            f"{summary['method']} | L{summary['layer']} | window {summary['window_ms']:.3f} ms | "
            f"gap {summary['compute_io_gap_ms']:.3f} ms ({gap_ratio} without IO) | {overlap}",
            loc="left",
            fontsize=10,
        )
        axis.grid(axis="x", alpha=0.15)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[-1, 0].set_xlabel("Time from layer window start (ms)")
    fig.suptitle(f"DeepSeek V3.2 · {phase} · measured single-layer timeline", x=0.06, ha="left")
    fig.legend(
        handles=[
            Patch(
                facecolor=color,
                edgecolor="#30343B" if label == "Compute + IO" else color,
                hatch="//" if label == "Compute + IO" else None,
                linewidth=0,
                label=label,
            )
            for label, color in colors.items()
        ],
        loc="lower center",
        ncol=5,
        fontsize=8,
    )
    fig.text(
        0.06,
        0.054,
        "Gap includes cache/control and idle. Proven-empty gather/fused calls are control/compute.\n"
        "Fused IO uses conservative bounds; unresolved gathers cannot certify the gap gate. Labels do not change kernel intervals.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.068, 1, 0.96))
    for suffix in ("svg", "png"):
        fig.savefig(output / f"timeline_{phase}.{suffix}", dpi=180)
    plt.close(fig)
