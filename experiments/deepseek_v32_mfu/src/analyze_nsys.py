"""Read-only Nsight SQLite analysis of one ECHO annotated extend capture.

GPU work is assigned through its runtime OR driver launch correlation to the
innermost ECHO NVTX range on that CPU thread. GPU timestamps need not fit inside
the CPU range: an asynchronous launch may finish after the range has closed.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from itertools import pairwise
from pathlib import Path

_PROCESS_MASK = 0xFFFFFFFFFF000000
_SCOPE = re.compile(
    r"^echo/(?P<mode>resident|offload)/extend_annotated/"
    r"(?P<layer>layer_\d+|shared)(?:/(?P<stage>[^/]+))?$"
)
_EVENT_API = re.compile(r"^(?:cuda|cu)(?:Event|Profiler)")
_NOTES = [
    (
        "Input must be a static Nsight export of one resident or offload extend capture. "
        "No SQL tables, indexes, views, or source files are modified."
    ),
    (
        "GPU attribution uses process/correlation IDs from both RUNTIME and DRIVER APIs, "
        "then the innermost ECHO CPU NVTX scope containing the launch API start. "
        "GPU execution is not clipped to the CPU NVTX duration."
    ),
    (
        "Kernel time is the sum of actual GPU kernel durations, not CUDA event elapsed time. "
        "Overlapping kernels or devices may make duration sums exceed wall time."
    ),
    (
        "api_ms is the union of non-event runtime/driver API intervals per CPU thread; "
        "nested runtime and driver calls are not double counted. API records/counts and "
        "per-source duration sums remain available separately."
    ),
    (
        "All cudaEvent/cuEvent and cudaProfiler/cuProfiler API records are excluded from stage "
        "API time and reported as bookkeeping. This also excludes framework dependency-event "
        "APIs; stream waits and synchronization APIs without these prefixes are retained."
    ),
    (
        "Per-device busy time unions KERNEL, MEMCPY, and MEMSET intervals across streams, "
        "including unattributed activities. Span runs from that device's first to last activity; "
        "gap = span - busy. These are activity occupancy measures, not SM utilization or stall time. "
        "CUDA event bookkeeping is not treated as GPU activity."
    ),
    (
        "Unattributed GPU work remains in capture/device/category totals and is listed explicitly. "
        "Capture GPU envelope excludes CPU work before the first or after the last GPU activity."
    ),
    (
        "Mapped-host KV reads inside the fused indexer and gather_records are kernels, not HtoD "
        "MEMCPY records. Actual KV transfer bytes come only from result.json cache counters; "
        "without that file they are unknown (null), never inferred to be zero."
    ),
]


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _union_ns(intervals):
    return sum(end - start for start, end in _union(intervals))


def _rows(connection, table):
    # Table names are selected from the fixed Nsight names below, not CLI input.
    return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]


def _name(row, strings, fields, fallback):
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
        if value is not None and value in strings:
            return strings[value]
    return fallback


def _time(row, table):
    start, end = row.get("start"), row.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise ValueError(f"Invalid or incomplete timestamp in {table}: {start}, {end}")
    return start, end


def _assign_scopes(apis, scopes):
    by_thread = defaultdict(list)
    for scope in scopes:
        by_thread[scope["thread"]].append(scope)
    api_threads = defaultdict(list)
    for api in apis:
        api_threads[api["thread"]].append(api)
    for thread, calls in api_threads.items():
        ranges = sorted(by_thread[thread], key=lambda x: (x["start"], -x["end"]))
        cursor, active = 0, []
        for api in sorted(calls, key=lambda x: x["start"]):
            while cursor < len(ranges) and ranges[cursor]["start"] <= api["start"]:
                scope = ranges[cursor]
                heapq.heappush(active, (-scope["start"], scope["end"], scope["id"], scope))
                cursor += 1
            while active and active[0][1] <= api["start"]:
                heapq.heappop(active)
            api["scope"] = active[0][-1] if active else None


def kernel_category(name):
    """Stable implementation families; full kernel names are also retained."""
    name = name.lower()
    if "sm90_fp8_mqa_logits_fuse_prefetch" in name:
        return "indexer_fused_prefetch"
    if "sm90_fp8_mqa_logits" in name:
        return "indexer"
    if "sparse_attn_fwd_kernel" in name or "sparse_mla" in name:
        return "sparse_mla"
    if "sm90_fp8_gemm_1d2d_impl" in name:
        return "deepgemm_fp8_gemm"
    if "grouped_fp8_linear" in name:
        return "grouped_expert_fp8"
    if "fp8_linear" in name:
        return "fp8_linear"
    if "reduce_fp8_split" in name:
        return "fp8_split_reduction"
    if "quantize_activation" in name:
        return "activation_quantization"
    if "gather_records" in name:
        return "mapped_host_kv_gather"
    if any(x in name for x in ("gemm", "cublas", "matmul")):
        return "blas_gemm"
    if any(x in name for x in ("topk", "top_k", "sort", "radix", "scan", "scatter", "gather")):
        return "selection_routing_indexing"
    if any(x in name for x in ("reduce", "reduction")):
        return "reduction"
    if any(x in name for x in ("elementwise", "pointwise", "vectorized")):
        return "pointwise"
    return "other"


def _read_capture(path, *, scope_pattern=None):
    pattern = _SCOPE if scope_pattern is None else scope_pattern
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "NVTX_EVENTS" not in tables:
            raise ValueError("SQLite does not contain NVTX_EVENTS")
        strings = (
            {row["id"]: row["value"] for row in _rows(connection, "StringIds")}
            if "StringIds" in tables
            else {}
        )
        scopes = []
        for row in _rows(connection, "NVTX_EVENTS"):
            text = _name(row, strings, ("text", "textId"), "")
            match = pattern.fullmatch(text)
            if not match:
                continue
            start, end = _time(row, "NVTX_EVENTS")
            if row.get("globalTid") is None:
                raise ValueError("ECHO NVTX range has no globalTid")
            scopes.append(
                {
                    "id": len(scopes),
                    "start": start,
                    "end": end,
                    "thread": row["globalTid"],
                    "label": text,
                    **match.groupdict(),
                }
            )
        apis, used_tables = [], ["NVTX_EVENTS"]
        if "StringIds" in tables:
            used_tables.append("StringIds")
        for source in ("RUNTIME", "DRIVER"):
            table = f"CUPTI_ACTIVITY_KIND_{source}"
            if table not in tables:
                continue
            used_tables.append(table)
            for row in _rows(connection, table):
                start, end = _time(row, table)
                thread = row.get("globalTid")
                if thread is None:
                    raise ValueError(f"{table} has no globalTid")
                name = _name(row, strings, ("name", "nameId"), "unknown_api")
                apis.append(
                    {
                        "id": len(apis),
                        "start": start,
                        "end": end,
                        "thread": thread,
                        "process": row["globalPid"]
                        if row.get("globalPid") is not None
                        else thread & _PROCESS_MASK,
                        "correlation": row.get("correlationId"),
                        "name": name,
                        "source": source,
                        "bookkeeping": bool(_EVENT_API.match(name)),
                    }
                )
        copy_labels = (
            {row["id"]: row["label"] for row in _rows(connection, "ENUM_CUDA_MEMCPY_OPER")}
            if "ENUM_CUDA_MEMCPY_OPER" in tables
            else {}
        )
        if "ENUM_CUDA_MEMCPY_OPER" in tables:
            used_tables.append("ENUM_CUDA_MEMCPY_OPER")
        activities = []
        for kind, alternatives in (
            ("kernel", ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")),
            ("memcpy", ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMCPY2")),
            ("memset", ("CUPTI_ACTIVITY_KIND_MEMSET",)),
        ):
            table = next((table for table in alternatives if table in tables), None)
            if table is None:
                continue
            used_tables.append(table)
            for row in _rows(connection, table):
                start, end = _time(row, table)
                copy_kind = copy_labels.get(row.get("copyKind"), f"copy_kind_{row.get('copyKind')}")
                name = (
                    _name(
                        row,
                        strings,
                        ("demangledName", "shortName", "name", "nameId", "mangledName"),
                        "unknown_kernel",
                    )
                    if kind == "kernel"
                    else copy_kind
                    if kind == "memcpy"
                    else "memset"
                )
                if row.get("deviceId") is None:
                    raise ValueError(f"{table} has no deviceId")
                activities.append(
                    {
                        "id": len(activities),
                        "kind": kind,
                        "start": start,
                        "end": end,
                        "device_id": row["deviceId"],
                        "stream_id": row.get("streamId"),
                        "process": row.get("globalPid"),
                        "correlation": row.get("correlationId"),
                        "name": name,
                        "bytes": row.get("bytes"),
                        "graph_id": row.get("graphId", 0),
                        "graph_node_id": row.get("graphNodeId", 0),
                    }
                )
    return scopes, apis, activities, used_tables


def _attribute(apis, activities):
    correlations = defaultdict(list)
    for api in apis:
        if api["correlation"] is not None:
            correlations[api["correlation"]].append(api)
    for activity in activities:
        candidates = correlations.get(activity["correlation"], [])
        reason = "no_correlated_api"
        if candidates and activity["process"] is not None:
            candidates = [api for api in candidates if api["process"] == activity["process"]]
            if not candidates:
                reason = "correlation_process_mismatch"
        # GPU execution may overlap the launch call, but not precede its start.
        eligible = [api for api in candidates if api["start"] <= activity["start"]]
        if candidates and not eligible:
            reason = "correlation_timestamp_mismatch"
        if (
            eligible
            and activity["process"] is None
            and len({api["process"] for api in eligible}) > 1
        ):
            eligible = []
            reason = "ambiguous_correlation_without_gpu_process"
        api = (
            max(eligible, key=lambda x: (x["start"], x["source"] == "DRIVER", -x["end"]))
            if eligible
            else None
        )
        activity["api"] = api
        activity["scope"] = api["scope"] if api else None
        activity["unattributed_reason"] = (
            None if activity["scope"] else "launch_outside_echo_scope" if api else reason
        )


def _group_summary(key, scopes, apis, activities):
    buckets = defaultdict(lambda: {"scopes": [], "apis": [], "gpu": []})
    for scope in scopes:
        buckets[key(scope)]["scopes"].append(scope)
    for api in apis:
        if api["scope"] and not api["bookkeeping"]:
            buckets[key(api["scope"])]["apis"].append(api)
    for activity in activities:
        if activity["scope"]:
            buckets[key(activity["scope"])]["gpu"].append(activity)
    result = []
    for group, data in sorted(buckets.items()):
        calls, gpu = data["apis"], data["gpu"]
        thread_intervals = defaultdict(list)
        for api in calls:
            thread_intervals[api["thread"]].append((api["start"], api["end"]))
        row = {
            "scope_count": len(data["scopes"]),
            "api_count": len(calls),
            "api_ms": sum(_union_ns(x) for x in thread_intervals.values()) / 1e6,
            "api_duration_sum_ms": sum(x["end"] - x["start"] for x in calls) / 1e6,
        }
        if isinstance(group, tuple):
            row.update(layer=group[0], stage=group[1])
        else:
            row["stage"] = group
        for source in ("RUNTIME", "DRIVER"):
            selected = [api for api in calls if api["source"] == source]
            row[source.lower() + "_api_count"] = len(selected)
            row[source.lower() + "_api_duration_sum_ms"] = (
                sum(x["end"] - x["start"] for x in selected) / 1e6
            )
        for kind in ("kernel", "memcpy", "memset"):
            selected = [activity for activity in gpu if activity["kind"] == kind]
            row[kind + "_count"] = len(selected)
            row[kind + "_ms"] = sum(x["end"] - x["start"] for x in selected) / 1e6
        result.append(row)
    return result


def _device_summary(activities):
    if not activities:
        return [], None
    first, last = min(x["start"] for x in activities), max(x["end"] for x in activities)
    output = []
    for device in sorted({x["device_id"] for x in activities}):
        selected = [x for x in activities if x["device_id"] == device]
        merged = _union((x["start"], x["end"]) for x in selected)
        busy = sum(end - start for start, end in merged)
        start, end = min(x["start"] for x in selected), max(x["end"] for x in selected)
        duration_sum = sum(x["end"] - x["start"] for x in selected)
        gaps = [(left[1], right[0]) for left, right in pairwise(merged)]
        output.append(
            {
                "device_id": device,
                "activity_count": len(selected),
                "attributed_activity_count": sum(x["scope"] is not None for x in selected),
                "start_ns": start,
                "end_ns": end,
                "span_ms": (end - start) / 1e6,
                "busy_ms": busy / 1e6,
                "gap_ms": (end - start - busy) / 1e6,
                "activity_duration_sum_ms": duration_sum / 1e6,
                "overlap_ms": (duration_sum - busy) / 1e6,
                "kernel_busy_ms": _union_ns(
                    (x["start"], x["end"]) for x in selected if x["kind"] == "kernel"
                )
                / 1e6,
                "idle_in_capture_gpu_envelope_ms": (last - first - busy) / 1e6,
                "largest_gaps": [
                    {"start_ns": a, "end_ns": b, "duration_ms": (b - a) / 1e6}
                    for a, b in sorted(gaps, key=lambda x: x[1] - x[0], reverse=True)[:20]
                ],
            }
        )
    return output, {"start_ns": first, "end_ns": last, "span_ms": (last - first) / 1e6}


def _kernel_summaries(activities):
    families, names = defaultdict(list), defaultdict(list)
    for activity in activities:
        if activity["kind"] == "kernel":
            families[kernel_category(activity["name"])].append(activity)
            names[activity["name"]].append(activity)

    def summarize(groups, label):
        output = []
        for name, rows in groups.items():
            durations = [row["end"] - row["start"] for row in rows]
            output.append(
                {
                    label: name,
                    "count": len(rows),
                    "total_ms": sum(durations) / 1e6,
                    "max_ms": max(durations) / 1e6,
                    "attributed_count": sum(row["scope"] is not None for row in rows),
                    "device_ids": sorted({row["device_id"] for row in rows}),
                }
            )
        return sorted(output, key=lambda row: row["total_ms"], reverse=True)

    return summarize(families, "category"), summarize(names, "name")


def _api_summary(apis):
    calls = [api for api in apis if not api["bookkeeping"]]
    by_name, by_thread, outside = defaultdict(list), defaultdict(list), defaultdict(list)
    for api in calls:
        interval = (api["start"], api["end"])
        by_name[api["source"], api["name"]].append(api)
        by_thread[api["thread"]].append(interval)
        if api["scope"] is None:
            outside[api["thread"]].append(interval)
    rows = [
        {
            "source": source,
            "name": name,
            "count": len(group),
            "duration_sum_ms": sum(api["end"] - api["start"] for api in group) / 1e6,
            "outside_echo_scope_count": sum(api["scope"] is None for api in group),
        }
        for (source, name), group in by_name.items()
    ]
    return {
        "non_event_api_count": len(calls),
        "thread_union_ms": sum(_union_ns(intervals) for intervals in by_thread.values()) / 1e6,
        "outside_echo_scope_thread_union_ms": sum(
            _union_ns(intervals) for intervals in outside.values()
        )
        / 1e6,
        "calls_by_name": sorted(rows, key=lambda row: row["duration_sum_ms"], reverse=True),
    }


def _find_result(sqlite_path):
    for ancestor in sqlite_path.parents:
        if ancestor.parent.name == "profile":
            candidate = ancestor.parent.parent / "data" / ancestor.name / "result.json"
            if candidate.is_file():
                return candidate
    return None


def _transfers(result_path, mode, activities):
    memcpy_bytes = Counter()
    for activity in activities:
        if activity["kind"] == "memcpy" and activity["bytes"] is not None:
            memcpy_bytes[activity["name"]] += activity["bytes"]
    output = {
        "result_path": None,
        "source_run_id": None,
        "host_to_device_bytes": None,
        "device_to_host_bytes": None,
        "reported_layers": None,
        "source_accepted": None,
        "source_sha256": None,
        "observed_memcpy_bytes_by_kind": dict(memcpy_bytes),
        "mapped_host_reads_are_memcpy_records": False,
    }
    if result_path is None:
        return output
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    try:
        rows = result["measurements"][mode]["cache_per_layer"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Full result lacks {mode} annotated extend cache counters") from error
    if not isinstance(rows, list) or not rows:
        raise ValueError("Full result must contain nonempty per-layer cache counters")
    for field in ("host_to_device_bytes", "device_to_host_bytes"):
        values = [row.get(field) for row in rows]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"Full result contains invalid {field} counters")
        output[field] = sum(values)
    output.update(
        result_path=str(result_path),
        source_run_id=result.get("run_id"),
        reported_layers=len(rows),
        counter_scope=f"measurements.{mode}.cache_per_layer",
        source_accepted=result.get("accepted"),
        source_sha256=hashlib.sha256(result_bytes).hexdigest(),
    )
    return output


def analyze_sqlite(sqlite_path, *, result_path=None, expected_mode=None):
    """Return serializable activity statistics without mutating the SQLite file."""
    sqlite_path = Path(sqlite_path).resolve(strict=True)
    scopes, apis, activities, tables = _read_capture(sqlite_path)
    modes = {scope["mode"] for scope in scopes}
    if len(modes) != 1:
        raise ValueError(
            "Expected ECHO extend_annotated NVTX ranges for exactly one mode per SQLite"
        )
    mode = modes.pop()
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"Expected mode {expected_mode}, found {mode}")
    if not activities:
        raise ValueError("SQLite contains no GPU kernel, memcpy, or memset activities")
    _assign_scopes(apis, scopes)
    _attribute(apis, activities)
    devices, envelope = _device_summary(activities)
    categories, kernels = _kernel_summaries(activities)
    unassigned = [row for row in activities if row["scope"] is None]
    bookkeeping = [api for api in apis if api["bookkeeping"]]
    excluded_names = Counter(api["name"] for api in bookkeeping)
    result_path = (
        Path(result_path).resolve(strict=True)
        if result_path is not None
        else _find_result(sqlite_path)
    )
    return {
        "schema_version": 1,
        "sqlite": str(sqlite_path),
        "mode": mode,
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tables_read": tables,
        "notes": _NOTES,
        "scope_prefix": f"echo/{mode}/extend_annotated/",
        "nvtx_scope_count": len(scopes),
        "cpu_nvtx_envelope": {
            "start_ns": min(x["start"] for x in scopes),
            "end_ns": max(x["end"] for x in scopes),
            "span_ms": (max(x["end"] for x in scopes) - min(x["start"] for x in scopes)) / 1e6,
        },
        "capture_gpu_envelope": envelope,
        "stages": _group_summary(
            lambda s: s["stage"] or "layer_unscoped", scopes, apis, activities
        ),
        "layer_stages": _group_summary(
            lambda s: (s["layer"], s["stage"] or "layer_unscoped"), scopes, apis, activities
        ),
        "devices": devices,
        "kernel_categories": categories,
        "kernels": kernels,
        "api_summary": _api_summary(apis),
        "attribution": {
            "gpu_activity_count": len(activities),
            "attributed_gpu_activity_count": len(activities) - len(unassigned),
            "unattributed_gpu_activity_count": len(unassigned),
            "unattributed_reasons": dict(Counter(row["unattributed_reason"] for row in unassigned)),
            "correlated_api_sources": dict(
                Counter(row["api"]["source"] for row in activities if row["api"])
            ),
            "non_event_api_outside_echo_scopes": sum(
                api["scope"] is None and not api["bookkeeping"] for api in apis
            ),
            "excluded_bookkeeping_api_count": len(bookkeeping),
            "excluded_bookkeeping_api_duration_sum_ms": sum(
                x["end"] - x["start"] for x in bookkeeping
            )
            / 1e6,
            "excluded_bookkeeping_api_names": dict(excluded_names),
            "unattributed_activities": [
                {
                    key: row[key]
                    for key in (
                        "kind",
                        "name",
                        "device_id",
                        "stream_id",
                        "start",
                        "end",
                        "process",
                        "correlation",
                        "unattributed_reason",
                    )
                }
                for row in unassigned
            ],
        },
        "kv_transfers": _transfers(result_path, mode, activities),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON analysis output")
    parser.add_argument(
        "--result",
        type=Path,
        help="Full run result.json; standard output/data layout is auto-detected",
    )
    parser.add_argument(
        "--mode", choices=("resident", "offload"), help="Assert the expected capture mode"
    )
    args = parser.parse_args()
    inputs = [args.sqlite, args.result] if args.result is not None else [args.sqlite]
    for path in inputs:
        if args.output.resolve() == path.resolve() or (
            args.output.exists() and path.exists() and args.output.samefile(path)
        ):
            parser.error("analysis output must not overwrite an input file")
    result = analyze_sqlite(args.sqlite, result_path=args.result, expected_mode=args.mode)
    # Also protect a result.json found automatically from the standard layout.
    source_result = result["kv_transfers"]["result_path"]
    if source_result is not None and (
        source_result == str(args.output.resolve())
        or (args.output.exists() and args.output.samefile(source_result))
    ):
        parser.error("analysis output must not overwrite the source result.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "output": str(args.output),
                "attributed": result["attribution"]["attributed_gpu_activity_count"],
                "unattributed": result["attribution"]["unattributed_gpu_activity_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
