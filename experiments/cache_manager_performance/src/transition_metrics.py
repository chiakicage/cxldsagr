"""CPU submission and GPU execution intervals around exact sparse selection.

The input dictionaries are the attributed records returned by ``analyze_nsys``.
All timestamps remain on the original NSYS clock, in nanoseconds. This module
does not read a capture, infer transfer counters, or apply a model acceptance gate.
"""

from __future__ import annotations

from itertools import pairwise

from experiments.deepseek_v32_mfu.src.analyze_nsys import _PROCESS_MASK, _union, _union_ns

_LANES = {"Compute": "compute", "IO": "io", "Compute + IO": "fused", "GPU control": "control"}
_PARTITIONS = (
    "compute_only",
    "io_only",
    "fused_only",
    "compute_io_overlap",
    "compute_fused_overlap",
    "io_fused_overlap",
    "compute_io_fused_overlap",
    "exposed_control",
    "idle",
)


def _validate_records(records, kind):
    seen = set()
    for record in records:
        start, end = record.get("start"), record.get("end")
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError(f"Invalid {kind} timestamps: {start}, {end}")
        identity = record.get("id")
        if identity is None or identity in seen:
            raise ValueError(f"Missing or duplicate {kind} id: {identity}")
        seen.add(identity)


def _scope_id(record):
    scope = record.get("scope")
    return scope.get("id") if scope else None


def _overlaps(record, start, end):
    return (
        start < end
        and record["start"] < record["end"]
        and record["start"] < end
        and record["end"] > start
    )


def _clipped(record, start, end, **extra):
    # Retain original scope, launch, timestamps, correlation, bytes, and counters.
    return {
        **record,
        "clipped_start_ns": max(start, record["start"]),
        "clipped_end_ns": min(end, record["end"]),
        **extra,
    }


def _api_class(api):
    name = api["name"].lower()
    if "synchronize" in name or "streamwaitevent" in name:
        return "synchronization"
    if "event" in name:
        return "event"
    if "profiler" in name:
        return "profiler"
    return "other"


def _cpu_interval(apis, start, end, thread):
    rows = [
        _clipped(api, start, end, api_class=_api_class(api))
        for api in apis
        if api["thread"] == thread and _overlaps(api, start, end)
    ]
    intervals = [(row["clipped_start_ns"], row["clipped_end_ns"]) for row in rows]
    busy = _union_ns(intervals)
    return {
        "start_ns": start,
        "end_ns": end,
        "thread": thread,
        "wall_ms": (end - start) / 1e6,
        "api_union_ms": busy / 1e6,
        "non_api_residual_ms": (end - start - busy) / 1e6,
        "api_count": len(rows),
        "api_class_union_ms": {
            category: _union_ns(
                (row["clipped_start_ns"], row["clipped_end_ns"])
                for row in rows
                if row["api_class"] == category
            )
            / 1e6
            for category in ("synchronization", "event", "profiler", "other")
        },
        "api_union_intervals_ns": _union(intervals),
        "apis": rows,
    }


def _gpu_interval(activities, start, end, lane_classifier):
    rows = []
    for activity in activities:
        if not _overlaps(activity, start, end):
            continue
        lane = lane_classifier(activity)
        if lane not in _LANES:
            raise ValueError(f"Unknown GPU lane: {lane!r}")
        rows.append(_clipped(activity, start, end, lane=lane))
    intervals = {
        category: _union(
            (row["clipped_start_ns"], row["clipped_end_ns"])
            for row in rows
            if _LANES[row["lane"]] == category
        )
        for category in _LANES.values()
    }
    edges = sorted(
        {start, end, *(point for ranges in intervals.values() for x in ranges for point in x)}
    )
    partition = dict.fromkeys(_PARTITIONS, 0)
    segments = []
    for left, right in pairwise(edges):
        active = [
            category
            for category in ("compute", "io", "fused")
            if any(a <= left < b for a, b in intervals[category])
        ]
        if active:
            label = "_".join(active) + ("_only" if len(active) == 1 else "_overlap")
        else:
            label = (
                "exposed_control" if any(a <= left < b for a, b in intervals["control"]) else "idle"
            )
        partition[label] += right - left
        segments.append({"start_ns": left, "end_ns": right, "class": label})
    width = end - start
    covered = _union_ns(intervals["compute"] + intervals["io"] + intervals["fused"])
    busy = _union_ns([interval for ranges in intervals.values() for interval in ranges])
    return {
        "start_ns": start,
        "end_ns": end,
        "window_ms": width / 1e6,
        "gpu_busy_union_ms": busy / 1e6,
        "gpu_idle_ms": (width - busy) / 1e6,
        "compute_union_ms": _union_ns(intervals["compute"]) / 1e6,
        "io_union_ms": _union_ns(intervals["io"]) / 1e6,
        "fused_union_ms": _union_ns(intervals["fused"]) / 1e6,
        "gpu_control_union_ms": _union_ns(intervals["control"]) / 1e6,
        "exposed_control_ms": partition["exposed_control"] / 1e6,
        "compute_io_fused_union_ms": covered / 1e6,
        "gap_ms": (width - covered) / 1e6,
        "fused_internal_io_unresolved": bool(intervals["fused"]),
        "partition_ms": {name: duration / 1e6 for name, duration in partition.items()},
        "partition_segments": segments,
        "lane_union_intervals_ns": intervals,
        "activity_count": len(rows),
        "activities": rows,
    }


def _activity_process(activity):
    process = activity.get("process")
    api = activity.get("api")
    if process is None and api is not None:
        process = api.get("process")
    return process


def _idle_submission(gpu_interval, activities, topk_last):
    """Bound idle ending at a uniquely correlated kernel on the main stream."""
    stream = topk_last.get("stream_id")
    process = _activity_process(topk_last)
    rows = []
    known_before = known_after = unresolved = 0
    for segment in gpu_interval["partition_segments"]:
        if segment["class"] != "idle":
            continue
        start, end = segment["start_ns"], segment["end_ns"]
        idle_end_activities = [
            row
            for row in activities
            if row.get("device_id") == topk_last["device_id"]
            and _activity_process(row) == process
            and row["start"] == end
            and row["end"] > row["start"]
        ]
        candidates = (
            [row for row in idle_end_activities if row.get("stream_id") == stream]
            if stream is not None
            else []
        )
        following = candidates[0] if len(candidates) == 1 else None
        launch = following.get("api") if following else None
        reason = None
        if stream is None:
            reason = "missing_main_stream"
        elif len(candidates) > 1:
            reason = "ambiguous_next_main_stream_activity"
        elif following is None:
            reason = (
                "other_stream_ends_idle"
                if idle_end_activities
                else "missing_next_main_stream_activity"
            )
        elif following["kind"] != "kernel":
            reason = "non_kernel_ends_idle"
        elif (
            launch is None
            or launch.get("process") != process
            or launch.get("correlation") is None
            or launch.get("correlation") != following.get("correlation")
            or type(launch.get("start")) is not int
            or launch["start"] > following["start"]
        ):
            reason = "unknown_or_invalid_launch_correlation"
        before = max(0, min(end, launch["start"]) - start) if reason is None else None
        if before is None:
            unresolved += end - start
        else:
            known_before += before
            known_after += end - start - before
        rows.append(
            {
                "start_ns": start,
                "end_ns": end,
                "idle_ms": (end - start) / 1e6,
                "next_activity_id": following["id"] if following else None,
                "next_activity_start_ns": following["start"] if following else None,
                "next_launch_start_ns": launch.get("start") if launch else None,
                "next_activity": following,
                "next_api": launch,
                "idle_end_activities": idle_end_activities,
                "idle_before_next_launch_api_ms": before / 1e6 if before is not None else None,
                "unresolved_reason": reason,
            }
        )
    return {
        "main_stream_id": stream,
        "idle_before_next_launch_api_ms": known_before / 1e6,
        "idle_after_next_launch_api_ms": known_after / 1e6,
        "unresolved_idle_ms": unresolved / 1e6,
        "segments": rows,
        "note": (
            "For idle ending exactly at one unique main-stream kernel, the interval before "
            "its correlated launch starts is a lower bound on time during which the host "
            "has not submitted that kernel. Idle ending at another stream or non-kernel "
            "activity remains unresolved. This does not identify Python overhead or explain "
            "remaining idle, and does not prove the next kernel could safely have been "
            "submitted earlier."
        ),
    }


def analyze_transition(
    scopes,
    apis,
    activities,
    *,
    indexer_stage,
    topk_stage,
    consumer_stage,
    endpoint_kind,
    lane_classifier,
):
    """Analyze one layer, preserving overlapping work from other source scopes.

    ``scopes`` contains that layer's unique, simple stage names. ``activities``
    can contain the complete measured phase, including other layers. API/GPU
    records retain the scope IDs assigned by launch correlation. The classifier
    returns ``Compute``, ``IO``, ``Compute + IO``, or ``GPU control`` and is
    responsible for applying actual, checked transfer-counter evidence.

    ``attention_kernel`` ends at the unique actual ``sparse_attn_fwd_kernel``
    in the consumer scope. Wrapper preparation remains inside the GPU span.
    ``manager_ready`` is a conservative bound: the later of no-op consumer CPU
    entry and completion of all target-layer GPU work submitted in earlier
    scopes. It is not a measurement of an attention invocation.
    """
    scopes, apis, activities = list(scopes), list(apis), list(activities)
    for records, kind in ((scopes, "scope"), (apis, "API"), (activities, "activity")):
        _validate_records(records, kind)
    if endpoint_kind not in {"attention_kernel", "manager_ready"}:
        raise ValueError(f"Unsupported endpoint kind: {endpoint_kind!r}")
    if len({indexer_stage, topk_stage, consumer_stage}) != 3:
        raise ValueError("Boundary stages must be distinct")
    by_stage = {}
    for scope in scopes:
        stage = scope.get("stage")
        if not isinstance(stage, str) or not stage or "/" in stage:
            raise ValueError("Scopes require nonempty simple stage names")
        if stage in by_stage:
            raise ValueError(f"Duplicate stage scope: {stage}")
        by_stage[stage] = scope
    for stage in (indexer_stage, topk_stage, consumer_stage):
        if stage not in by_stage:
            raise ValueError(f"Missing boundary scope: {stage}")
    indexer, topk, consumer = (by_stage[x] for x in (indexer_stage, topk_stage, consumer_stage))
    threads = {scope.get("thread") for scope in scopes}
    if len(threads) != 1 or None in threads:
        raise ValueError("A layer transition requires a single CPU thread")
    thread = threads.pop()
    process = thread & _PROCESS_MASK
    if indexer["end"] > topk["start"] or topk["end"] > consumer["start"]:
        raise ValueError("Reversed or overlapping CPU boundary scopes")
    scope_ids = {scope["id"] for scope in scopes}
    target = [activity for activity in activities if _scope_id(activity) in scope_ids]
    grouped = {
        stage: [row for row in target if _scope_id(row) == scope["id"]]
        for stage, scope in by_stage.items()
    }
    for stage in (indexer_stage, topk_stage):
        if not grouped[stage]:
            raise ValueError(f"Missing correlated GPU activity for {stage}")
    indexer_last = max(grouped[indexer_stage], key=lambda row: (row["end"], row["id"]))
    topk_first = min(grouped[topk_stage], key=lambda row: (row["start"], row["id"]))
    topk_last = max(grouped[topk_stage], key=lambda row: (row["end"], row["id"]))
    consumer_gpu = grouped[consumer_stage]
    attention = None
    ready_activity = None
    if endpoint_kind == "attention_kernel":
        candidates = [
            row
            for row in consumer_gpu
            if row["kind"] == "kernel" and "sparse_attn_fwd_kernel" in row["name"]
        ]
        if len(candidates) != 1:
            raise ValueError("Expected one actual attention kernel; missing or split consumers")
        attention = candidates[0]
        launch = attention.get("api")
        if launch is None or _scope_id(launch) != consumer["id"]:
            raise ValueError("Attention kernel requires a correlated launch in its consumer scope")
        endpoint = attention["start"]
    else:
        if consumer_gpu:
            raise ValueError("manager_ready requires a no-op consumer with no GPU activities")
        earlier_ids = {
            scope["id"]
            for scope in scopes
            if scope["id"] != consumer["id"] and scope["end"] <= consumer["start"]
        }
        earlier_gpu = [row for row in target if _scope_id(row) in earlier_ids]
        ready_activity = max(earlier_gpu, key=lambda row: (row["end"], row["id"]))
        endpoint = max(consumer["start"], ready_activity["end"])
    if indexer_last["end"] > topk_first["start"] or topk_last["end"] > endpoint:
        raise ValueError("Reversed GPU transition boundaries")
    spans = ((indexer_last["end"], topk_first["start"]), (topk_last["end"], endpoint))
    relevant_gpu = [
        row
        for row in activities
        if _scope_id(row) in scope_ids or any(_overlaps(row, *span) for span in spans)
    ]
    devices = {row.get("device_id") for row in relevant_gpu}
    if len(devices) != 1 or None in devices:
        raise ValueError("Mixed or missing GPU device identity in measured transition")
    if any(_activity_process(row) != process for row in relevant_gpu):
        raise ValueError("Mixed or missing GPU process identity in measured transition")
    relevant_apis = [
        api
        for api in apis
        if _scope_id(api) in scope_ids
        or (api["thread"] == thread and _overlaps(api, topk["end"], consumer["start"]))
    ]
    if any(api.get("process") != process for api in relevant_apis):
        raise ValueError("Mixed or missing API process identity in measured transition")
    boundaries = {
        "endpoint_kind": endpoint_kind,
        "indexer_scope": indexer,
        "topk_scope": topk,
        "consumer_scope": consumer,
        "indexer_last_activity_id": indexer_last["id"],
        "topk_first_activity_id": topk_first["id"],
        "topk_last_activity_id": topk_last["id"],
        "attention_activity_id": attention["id"] if attention else None,
        "manager_last_activity_id": ready_activity["id"] if ready_activity else None,
        "indexer_last_activity": indexer_last,
        "topk_first_activity": topk_first,
        "topk_last_activity": topk_last,
        "attention_activity": attention,
        "manager_last_activity": ready_activity,
        "endpoint_ns": endpoint,
        "attention_launch_start_minus_topk_end_ms": (
            (attention["api"]["start"] - topk_last["end"]) / 1e6 if attention else None
        ),
        "process": process,
        "device_id": devices.pop(),
    }
    stages = []
    for scope in sorted(scopes, key=lambda row: (row["start"], row["id"])):
        calls = [api for api in apis if _scope_id(api) == scope["id"]]
        gpu = grouped[scope["stage"]]
        stages.append(
            {
                "stage": scope["stage"],
                "scope": scope,
                "cpu": _cpu_interval(calls, scope["start"], scope["end"], thread),
                "gpu_in_post_topk": _gpu_interval(gpu, *spans[1], lane_classifier),
                "correlated_api_count": len(calls),
                "correlated_gpu_activity_count": len(gpu),
                "gpu_first_start_ns": min((row["start"] for row in gpu), default=None),
                "gpu_last_end_ns": max((row["end"] for row in gpu), default=None),
            }
        )
    post_topk = _gpu_interval(activities, *spans[1], lane_classifier)
    return {
        "boundaries": boundaries,
        "cpu_topk_to_consumer": _cpu_interval(apis, topk["end"], consumer["start"], thread),
        "gpu_indexer_to_topk": _gpu_interval(activities, *spans[0], lane_classifier),
        "gpu_topk_to_consumer": post_topk,
        "post_topk_idle_submission": _idle_submission(post_topk, activities, topk_last),
        "stages": stages,
        "notes": [
            "CPU intervals describe submission scopes; GPU intervals describe execution. They are not additive.",
            "API unions include event and synchronization calls. API category unions can overlap because runtime and driver calls nest.",
            "Non-API CPU residual is uninstrumented host time, not an attribution to Python or evidence of its cause.",
            "GPU occupancy includes all supplied activity overlapping the span, with original launch scopes preserved; occupancy does not prove dependency or critical-path causation.",
            "GPU class unions can overlap. Only partition_ms is an additive breakdown; exposed control excludes compute, IO, and fused coverage.",
            "Fused compute/IO intervals remain unresolved internally; actual transfer counters are interpreted only by the supplied lane classifier.",
            "A negative attention launch offset means MLA was submitted before top-k finished on GPU, not that it executed before top-k.",
            "The manager_ready endpoint is a conservative completion bound for pre-consumer work; it does not measure a real MLA consumer.",
        ],
    }
