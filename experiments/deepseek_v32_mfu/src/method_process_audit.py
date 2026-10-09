"""Read raw NSYS process/device ownership for one synchronized method forward.

This supplements graph/node attribution. It does not infer continuous GPU,
CPU or machine isolation from a profiler capture or from process metadata.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

PROCESS_MASK = 0xFFFFFFFFFF000000
GPU_TABLES = {
    "CUPTI_ACTIVITY_KIND_KERNEL": "kernel",
    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL": "kernel",
    "CUPTI_ACTIVITY_KIND_MEMCPY": "memcpy",
    "CUPTI_ACTIVITY_KIND_MEMCPY2": "memcpy",
    "CUPTI_ACTIVITY_KIND_MEMSET": "memset",
}


class CaptureProcessError(ValueError):
    """Keep the observed ownership evidence when a measured-window check fails."""

    def __init__(self, message, observations):
        super().__init__(message)
        self.observations = observations


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid raw Nsight {name}: {value!r}")
    return value


def _interval(row, table):
    start = _integer(row.get("start"), table + ".start")
    end = _integer(row.get("end"), table + ".end")
    if end <= start:
        raise ValueError(f"Nonpositive raw Nsight interval in {table}")
    return start, end


def _union_ns(intervals):
    total, previous_end = 0, None
    for start, end in sorted(intervals):
        total += max(0, end - max(start, start if previous_end is None else previous_end))
        previous_end = max(end, end if previous_end is None else previous_end)
    return total


def _summary(rows, process_map, *, window=None):
    intervals = [(row["start_ns"], row["end_ns"]) for row in rows]
    if window is not None:
        intervals = [(max(start, window[0]), min(end, window[1])) for start, end in intervals]
    global_pids = sorted({row["global_pid"] for row in rows})
    return {
        "activity_count": len(rows),
        "kind_counts": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "table_counts": dict(sorted(Counter(row["table"] for row in rows).items())),
        "serialized_process_ids": global_pids,
        "os_pids": sorted({process_map[pid] for pid in global_pids if pid in process_map}),
        "unmapped_serialized_process_ids": [pid for pid in global_pids if pid not in process_map],
        "device_ids": sorted({row["device_id"] for row in rows}),
        "interval_union_ns": _union_ns(intervals),
    }


def audit_capture_process(
    sqlite_path,
    *,
    target_pid,
    method,
    phase,
    expected_device=0,
    require_single_graph=False,
):
    """Bind raw activity to a recorded OS PID and CUDA device for one forward.

    ``phase`` is ``prefill`` or ``extend``. ``expected_device`` is the CUPTI
    device ID (GPU 0 for the fixed cohort); TARGET_INFO_CUDA_DEVICE must also
    map this GPU to this target process's CUDA-local device 0. Set
    ``require_single_graph`` for the full-extend graph; prefill may issue many
    compute-island graphs. Process identity comes from PROCESSES, not names or
    a guessed bit shift of the serialized GlobalId.

    GPU intervals use strict overlap with the synchronized forward NVTX range,
    including boundary-crossing rows from every recorded process/device.
    Foreign activity outside that range is reported and does not reject the
    measured forward. Graph GPU rows additionally need a process/correlation
    match to a GraphLaunch API issued by the forward's owning thread.
    """
    _integer(target_pid, "target PID", minimum=1)
    _integer(expected_device, "expected device")
    if method not in {"hbm", "echo", "serial_sparse", "dense_prefetch"}:
        raise ValueError("Unsupported method for process audit")
    if phase not in {"prefill", "extend"}:
        raise ValueError("Process audit phase must be prefill or extend")
    path = Path(sqlite_path).resolve(strict=True)
    before = _digest(path)
    scope_pattern = re.compile(
        rf"^echo/{method}/{phase}_annotated/shared/forward_misc/(?:call_\d+)$"
    )
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not {"PROCESSES", "NVTX_EVENTS", "TARGET_INFO_CUDA_DEVICE"} <= tables:
            raise ValueError("Raw process audit requires process, NVTX and CUDA device tables")
        strings = (
            dict(connection.execute("SELECT id,value FROM StringIds"))
            if "StringIds" in tables
            else {}
        )
        process_map = {}
        for row in connection.execute("SELECT globalPid,pid FROM PROCESSES"):
            global_pid = _integer(row["globalPid"], "PROCESSES.globalPid")
            pid = _integer(row["pid"], "PROCESSES.pid", minimum=1)
            if global_pid in process_map and process_map[global_pid] != pid:
                raise ValueError("Ambiguous raw serialized process identity")
            process_map[global_pid] = pid
        target_globals = {
            global_pid for global_pid, pid in process_map.items() if pid == target_pid
        }
        if len(target_globals) != 1:
            raise ValueError("Missing or ambiguous target PID in raw PROCESSES")
        (target_global,) = target_globals
        device_mapping = [
            dict(row)
            for row in connection.execute("SELECT * FROM TARGET_INFO_CUDA_DEVICE")
            if row["pid"] == target_pid
        ]
        if not any(
            row["gpuId"] == expected_device and row["cudaId"] == 0 for row in device_mapping
        ):
            raise ValueError("Target PID does not map expected GPU to CUDA-local device 0")
        scopes = []
        for item in connection.execute("SELECT * FROM NVTX_EVENTS"):
            row = dict(item)
            label = row.get("text") or strings.get(row.get("textId"), "")
            if not scope_pattern.fullmatch(label):
                continue
            start, end = _interval(row, "NVTX_EVENTS")
            thread = _integer(row.get("globalTid"), "NVTX_EVENTS.globalTid")
            scopes.append(
                {
                    "label": label,
                    "start_ns": start,
                    "end_ns": end,
                    "thread": thread,
                    "global_pid": thread & PROCESS_MASK,
                }
            )
        if len(scopes) != 1:
            raise ValueError("Expected exactly one measured forward NVTX scope")
        (scope,) = scopes
        if scope["global_pid"] != target_global:
            raise ValueError("Measured forward NVTX owner differs from the recorded target PID")
        gpu = []
        for table, kind in GPU_TABLES.items():
            if table not in tables:
                continue
            for item in connection.execute("SELECT * FROM " + table):
                row = dict(item)
                start, end = _interval(row, table)
                gpu.append(
                    {
                        "table": table,
                        "kind": kind,
                        "start_ns": start,
                        "end_ns": end,
                        "global_pid": _integer(row.get("globalPid"), table + ".globalPid"),
                        "device_id": _integer(row.get("deviceId"), table + ".deviceId"),
                        "correlation_id": row.get("correlationId"),
                        "graph_id": row.get("graphId") or 0,
                        "graph_node_id": row.get("graphNodeId") or 0,
                    }
                )
        launches = []
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            for item in connection.execute("SELECT * FROM " + table):
                row = dict(item)
                name = row.get("name") or strings.get(row.get("nameId"), "")
                if "GraphLaunch" not in name:
                    continue
                start, end = _interval(row, table)
                thread = _integer(row.get("globalTid"), table + ".globalTid")
                global_pid = thread & PROCESS_MASK
                if row.get("globalPid") is not None and row["globalPid"] != global_pid:
                    raise ValueError("Graph launch process and owning thread disagree")
                launches.append(
                    {
                        "table": table,
                        "name": name,
                        "start_ns": start,
                        "end_ns": end,
                        "thread": thread,
                        "global_pid": global_pid,
                        "correlation_id": row.get("correlationId"),
                    }
                )
    if _digest(path) != before:
        raise ValueError("Raw Nsight SQLite changed during process audit")
    start, end = scope["start_ns"], scope["end_ns"]
    measured = [row for row in gpu if row["end_ns"] > start and row["start_ns"] < end]
    outside = [row for row in gpu if row["end_ns"] <= start or row["start_ns"] >= end]
    foreign = lambda row: (
        row["global_pid"] != target_global
        or row.get("device_id", expected_device) != expected_device
    )
    measured_launches = [row for row in launches if start <= row["start_ns"] < end]
    overlapping_launches = [
        row for row in launches if row["end_ns"] > start and row["start_ns"] < end
    ]
    observations = {
        "schema": "deepseek-method-raw-process-audit-v1",
        "passed": False,
        "sqlite": str(path),
        "sqlite_sha256": before,
        "target_pid": target_pid,
        "target_serialized_process_id": target_global,
        "expected_device_id": expected_device,
        "target_cuda_device_mapping": device_mapping,
        "recorded_processes": [
            {"serialized_process_id": global_pid, "os_pid": pid}
            for global_pid, pid in sorted(process_map.items())
        ],
        "forward": {**scope, "os_pid": target_pid},
        "all_capture_gpu": _summary(gpu, process_map),
        "measured_forward_gpu": _summary(measured, process_map, window=(start, end)),
        "outside_forward_gpu": _summary(outside, process_map),
        "boundary_crossing_gpu": [
            row for row in measured if row["start_ns"] < start or row["end_ns"] > end
        ],
        "foreign_gpu_in_forward": [row for row in measured if foreign(row)],
        "foreign_gpu_outside_forward": [row for row in outside if foreign(row)],
        "all_capture_graph_launches": launches,
        "measured_forward_graph_launches": measured_launches,
        "foreign_graph_launches_in_forward": [row for row in overlapping_launches if foreign(row)],
        "foreign_graph_launches_outside_forward": [
            row for row in launches if not start <= row["start_ns"] < end and foreign(row)
        ],
        "boundary": "Recorded activity only. All CUPTI kernel/memcpy/memset tables are read; "
        "measured GPU rows overlap the synchronized forward NVTX interval and include boundary "
        "crossings. Foreign activity outside that interval remains explicit. No inference of "
        "continuous process, device, CPU or machine isolation.",
    }

    def require(condition, message):
        if not condition:
            raise CaptureProcessError(message, observations)

    require(measured, "Measured forward contains no target GPU activity")
    require(
        not observations["foreign_gpu_in_forward"],
        "Foreign PID/device GPU activity intersects measured forward",
    )
    require(
        not observations["foreign_graph_launches_in_forward"],
        "Foreign graph launch API intersects measured forward",
    )
    require(
        all(
            row["global_pid"] == target_global and row["thread"] == scope["thread"]
            for row in measured_launches
        ),
        "Measured graph launch belongs to another PID/thread",
    )
    by_launch = defaultdict(list)
    for row in measured_launches:
        require(type(row["correlation_id"]) is int, "Measured graph launch has no correlation ID")
        by_launch[row["global_pid"], row["correlation_id"]].append(row)
    graph_groups = defaultdict(list)
    for row in measured:
        if row["graph_id"] or row["graph_node_id"]:
            key = (row["global_pid"], row["correlation_id"])
            require(
                key in by_launch,
                "Measured graph GPU activity has no target GraphLaunch correlation",
            )
            graph_groups[key].append(row)
    require(graph_groups, "Measured forward has no process-bound graph GPU activity")
    if require_single_graph:
        require(
            len(graph_groups) == 1,
            "Full extend does not have exactly one process-bound graph launch",
        )
        require(
            len(
                {
                    row["graph_id"]
                    for rows in graph_groups.values()
                    for row in rows
                    if row["graph_id"]
                }
            )
            == 1,
            "Full extend contains missing or multiple executable graph IDs",
        )
    observations["graph_launch_bindings"] = [
        {
            "serialized_process_id": global_pid,
            "os_pid": target_pid,
            "correlation_id": correlation,
            "apis": by_launch[global_pid, correlation],
            "gpu_activity_count": len(rows),
            "device_ids": sorted({row["device_id"] for row in rows}),
            "graph_ids": sorted({row["graph_id"] for row in rows if row["graph_id"]}),
        }
        for (global_pid, correlation), rows in sorted(graph_groups.items())
    ]
    observations["passed"] = True
    return observations
