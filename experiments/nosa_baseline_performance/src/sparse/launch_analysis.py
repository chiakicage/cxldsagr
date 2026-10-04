"""Analyze active-time unions and late submission in a serial NOSA nsys trace.

Unlike module MFU, gaps require device intervals (including copies and memsets).
NVTX owns activities by correlated host submission, never by GPU timestamp.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import statistics
from bisect import bisect_left
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from experiments.nosa_baseline_performance.src.sparse.module_mfu import (
    ROOT_RANGE,
    read_trace,
    validate_trace_gpu,
)


def union_ns(intervals):
    total, right = 0, None
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError("Intervals must have positive duration")
        total += max(0, end - max(start, right if right is not None else start))
        right = max(end, right if right is not None else end)
    return total


def timeline(activities):
    """Return a GPU hull, union, and conservative pre-submission gap component.

    Requires serial work on a single stream. Small timestamp overlaps are
    handled with interval unions. A gap before the next CUDA API even starts
    cannot be device execution of that next operation. It may include Python,
    instrumentation, previous synchronization, allocator or scheduling work.
    The residual gap is deliberately not called pure CUDA launch latency.
    """
    if not activities:
        raise ValueError("No GPU activities")
    ordered = sorted(activities, key=lambda row: row["start"])
    if any(row["launch_start"] > row["start"] for row in ordered):
        raise ValueError("GPU work precedes its correlated submission")
    active = union_ns((r["start"], r["end"]) for r in ordered)
    hull = max(r["end"] for r in ordered) - ordered[0]["start"]
    right, late = ordered[0]["end"], 0
    for row in ordered[1:]:
        late += max(0, min(row["start"], row["launch_start"]) - right)
        right = max(right, row["end"])
    kernels = [r for r in ordered if r["kind"] == "kernel"]
    durations = [(r["end"] - r["start"]) / 1000 for r in kernels]
    return {
        "hull_ms": hull / 1e6,
        "gpu_active_union_ms": active / 1e6,
        "kernel_union_ms": union_ns((r["start"], r["end"]) for r in kernels) / 1e6,
        "idle_ms": (hull - active) / 1e6,
        "idle_pct": 100 * (hull - active) / hull,
        "before_next_submission_ms": late / 1e6,
        "remaining_gap_ms": (hull - active - late) / 1e6,
        "kernel_count": len(kernels),
        "copy_memset_count": len(ordered) - len(kernels),
        "kernel_median_us": statistics.median(durations) if durations else None,
        "kernels_le_5us": sum(d <= 5 for d in durations),
    }


def analyze(data_dir, output):
    data_dir, output = Path(data_dir), Path(output)
    trace = data_dir / "nsys.sqlite"
    scopes, kernels = read_trace(trace)
    activities = [dict(row, kind="kernel") for row in kernels]
    with sqlite3.connect(trace.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for kind in ("MEMCPY", "MEMSET"):
            if f"CUPTI_ACTIVITY_KIND_{kind}" not in tables:
                continue
            activities.extend(
                dict(row, kind=kind.lower())
                for row in conn.execute(
                    f"SELECT a.start,a.end,a.deviceId,a.streamId,a.globalPid,a.correlationId,"
                    f"r.start AS launch_start,r.globalTid FROM CUPTI_ACTIVITY_KIND_{kind} a "
                    "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON a.correlationId=r.correlationId"
                )
            )
        apis = [
            dict(row)
            for row in conn.execute(
                "SELECT r.start,r.end,r.globalTid,s.value AS name FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON r.nameId=s.id ORDER BY r.start"
            )
        ]
    for field in ("deviceId", "streamId", "globalPid", "globalTid"):
        if (
            not activities
            or len({a[field] for a in activities}) != 1
            or activities[0][field] is None
        ):
            raise ValueError(f"Expected one non-null {field}; multistream attribution unsupported")
    if any(a["launch_start"] is None for a in activities):
        raise ValueError("Uncorrelated device activity")
    if len({a["correlationId"] for a in activities}) != len(activities):
        raise ValueError("Expected unique operation correlation IDs")
    activities.sort(key=lambda a: a["launch_start"])
    if any(a["start"] > b["start"] for a, b in pairwise(activities)):
        raise ValueError("GPU execution order must follow serial submission order")
    launches = [a["launch_start"] for a in activities]
    api_starts = [a["start"] for a in apis]

    def owned(scope):
        return activities[
            bisect_left(launches, scope["start"]) : bisect_left(launches, scope["end"])
        ]

    roots = sorted((s for s in scopes if ROOT_RANGE.fullmatch(s["text"])), key=lambda s: s["start"])
    if not roots:
        raise ValueError("Missing NOSA profile roots")
    if any(root["end"] <= root["start"] for root in roots) or any(
        left["end"] > right["start"] for left, right in pairwise(roots)
    ):
        raise ValueError("Expected positive, nonoverlapping NOSA profile roots")
    if {root["globalTid"] for root in roots} != {activities[0]["globalTid"]}:
        raise ValueError("Profile roots must use the single CUDA launch thread")

    def api_summary(rows, *, clip=None):
        grouped = defaultdict(list)
        for api in rows:
            if api["globalTid"] != activities[0]["globalTid"]:
                continue
            start, end = api["start"], api["end"]
            if clip is not None:
                start, end = max(start, clip[0]), min(end, clip[1])
            if end > start:
                grouped[api["name"]].append(end - start)
        return {
            name: {"count": len(times), "duration_sum_ms": sum(times) / 1e6}
            for name, times in grouped.items()
        }

    runs = []
    for root in roots:
        selected = owned(root)
        result = {"scope": root["text"], **timeline(selected)}
        root_runtime = apis[
            bisect_left(api_starts, root["start"]) : bisect_left(api_starts, root["end"])
        ]
        result["root_cuda_apis"] = api_summary(root_runtime)
        result["cuda_apis"] = api_summary(
            root_runtime, clip=(min(a["start"] for a in selected), max(a["end"] for a in selected))
        )
        stage_rows, cis_kernels = defaultdict(list), defaultdict(list)
        for scope in scopes:
            if not root["start"] <= scope["start"] < scope["end"] <= root["end"]:
                continue
            match = re.fullmatch(r"NOSA/[^/]+/layer_\d+/([a-z_]+)/q\d+\+\d+", scope["text"])
            if not match:
                continue
            rows = owned(scope)
            if rows:
                stage_rows[match[1]].append(timeline(rows))
            if match[1] == "cis_projection":
                for row in rows:
                    if row["kind"] == "kernel":
                        cis_kernels[row["name"]].append((row["end"] - row["start"]) / 1000)
        additive = (
            "hull_ms",
            "gpu_active_union_ms",
            "idle_ms",
            "before_next_submission_ms",
            "remaining_gap_ms",
            "kernel_count",
            "copy_memset_count",
        )
        result["stages"] = {
            name: {"calls": len(rows), **{key: sum(r[key] for r in rows) for key in additive}}
            for name, rows in stage_rows.items()
        }
        result["cis_kernels"] = [
            {
                "name": name,
                "count": len(times),
                "mean_us": statistics.mean(times),
                "median_us": statistics.median(times),
            }
            for name, times in cis_kernels.items()
        ]
        runs.append(result)
    metadata = json.loads((data_dir / "metadata.json").read_text())
    profile_path = data_dir / "profile_metadata.json"
    profile_metadata = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    mode = profile_metadata.get("args", {}).get("mode", "profile")
    if mode == "timeline" and len(scopes) != len(roots):
        raise ValueError("Timeline capture must contain root-only NOSA NVTX scopes")
    validate_trace_gpu(
        trace,
        activities[0]["deviceId"],
        metadata["gpu"],
        global_pid=activities[0]["globalPid"],
    )
    report = {
        "source_run_id": metadata["run_id"],
        "capture_mode": mode,
        "definitions": {
            "hull": "First to last correlated GPU activity within a root; excludes NVTX setup/collect outside device hull",
            "active": "Union of kernel, memcpy and memset device intervals; not sum of durations",
            "before_next_submission": "Idle time before the next operation's host API begins; includes host/sync/instrumentation delay, not pure CUDA launch latency",
            "remaining_gap": "Idle after API entry: mixed submission, scheduling and device latency; not uniquely attributed",
            "stage": "Sum of per-call GPU hulls; child stages are inclusive and cannot be added to parents",
            "cuda_apis": "CUDA APIs on the launch thread, clipped to the GPU hull; includes boundary waits overlapping the hull",
            "root_cuda_apis": "CUDA APIs on the launch thread entered inside the root, including event setup and final completion waits",
            "limits": (
                "Root-only NVTX and nsys CUDA tracing, without module scopes"
                if mode == "timeline"
                else "Module-instrumented single-stream run"
            )
            + "; trace timing is profiled, not primary unprofiled latency; no subtraction from independent unprofiled wall time; no inferred HBM throughput",
        },
        "gpu": metadata["gpu"],
        "runs": runs,
        "provenance": {},
    }
    for path in (trace, data_dir / "metadata.json", Path(__file__)):
        with path.open("rb") as handle:
            report["provenance"][str(path)] = hashlib.file_digest(handle, "sha256").hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.data_dir, args.output)
    print(
        json.dumps(
            [
                {k: v for k, v in run.items() if k not in ("stages", "cuda_apis", "cis_kernels")}
                for run in report["runs"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
