"""Audit GPU activity gaps, CUDA APIs and disjoint NOSA NVTX module ranges."""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def activity_summary(activities, runtime):
    ordered = sorted(activities, key=lambda event: event["start"])
    end = ordered[0]["start"]
    busy = 0
    gaps = []
    late_host = 0
    for event in ordered:
        gap = max(0, event["start"] - end)
        if gap:
            gaps.append(gap)
            api = runtime.get(event["correlationId"])
            if api:
                late_host += min(gap, max(0, api["start"] - end))
        busy += max(0, event["end"] - max(end, event["start"]))
        end = max(end, event["end"])
    span = end - ordered[0]["start"]
    return {
        "gpu_span_ms": span / 1e6,
        "gpu_busy_ms": busy / 1e6,
        "gpu_gap_ms": sum(gaps) / 1e6,
        "gpu_busy_pct": busy / span * 100,
        "max_gap_us": max(gaps, default=0) / 1e3,
        "gaps_over_5us": sum(gap > 5000 for gap in gaps),
        "gap_before_next_cuda_api_ms": late_host / 1e6,
    }


def analyze(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    runtime = {
        row["correlationId"]: dict(row)
        for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME")
    }
    scopes = [
        dict(row)
        for row in connection.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL"
        )
    ]
    for scope in scopes:
        scope["text"] = scope["text"] or strings.get(scope["textId"], "")
    roots = [scope for scope in scopes if scope["text"].startswith("GR/")]
    module_scopes = defaultdict(list)
    for scope in scopes:
        if scope["text"].startswith("nosa::"):
            module_scopes[scope["globalTid"]].append(scope)
    starts = {}
    for tid, values in module_scopes.items():
        values.sort(key=lambda event: (event["start"], -event["end"]))
        starts[tid] = [event["start"] for event in values]
    activities = []
    for category in ("KERNEL", "MEMCPY", "MEMSET"):
        for row in connection.execute(f"SELECT * FROM CUPTI_ACTIVITY_KIND_{category}"):
            event = dict(row)
            event["category"] = category
            event["name"] = strings[event["demangledName"]] if category == "KERNEL" else category
            activities.append(event)
    results = []
    for root in roots:
        apis = {
            key: api
            for key, api in runtime.items()
            if api["globalTid"] == root["globalTid"] and root["start"] <= api["start"] < root["end"]
        }
        work = [event for event in activities if event["correlationId"] in apis]
        kernels = [event for event in work if event["category"] == "KERNEL"]
        result = {
            "range": root["text"],
            "wall_ms": (root["end"] - root["start"]) / 1e6,
            **activity_summary(work, runtime),
        }
        result["kernel_count"] = len(kernels)
        result["kernel_ms"] = sum(event["end"] - event["start"] for event in kernels) / 1e6
        result["streams"] = sorted({event["streamId"] for event in work})
        kernel_groups = defaultdict(lambda: {"count": 0, "gpu_ms": 0})
        api_groups = defaultdict(lambda: {"count": 0, "cpu_ms": 0})
        for event in kernels:
            item = kernel_groups[event["name"]]
            item["count"] += 1
            item["gpu_ms"] += (event["end"] - event["start"]) / 1e6
        for api in apis.values():
            item = api_groups[strings[api["nameId"]]]
            item["count"] += 1
            item["cpu_ms"] += (api["end"] - api["start"]) / 1e6
        result["kernels"] = dict(sorted(kernel_groups.items(), key=lambda pair: -pair[1]["gpu_ms"]))
        result["cuda_apis"] = dict(sorted(api_groups.items(), key=lambda pair: -pair[1]["cpu_ms"]))
        queue = [
            max(0, event["start"] - apis[event["correlationId"]]["end"]) / 1e3 for event in kernels
        ]
        result["median_launch_to_kernel_us"] = statistics.median(queue)
        result["modules"] = {}
        if root["text"].startswith("GR/detailed/"):
            groups = defaultdict(
                lambda: {"kernel_count": 0, "activity_count": 0, "gpu_ms": 0, "kernel_names": {}}
            )
            for event in work:
                api = apis[event["correlationId"]]
                tid = api["globalTid"]
                index = bisect.bisect_right(starts.get(tid, []), api["start"]) - 1
                module = None
                while index >= 0:
                    scope = module_scopes[tid][index]
                    if scope["start"] <= api["start"] < scope["end"]:
                        module = scope["text"].split("/")[1]
                        break
                    index -= 1
                if module is None:
                    raise RuntimeError(f"No module for {event['name']}")
                item = groups[module]
                item["activity_count"] += 1
                item["kernel_count"] += event["category"] == "KERNEL"
                item["gpu_ms"] += (event["end"] - event["start"]) / 1e6
                item["kernel_names"][event["name"]] = item["kernel_names"].get(event["name"], 0) + 1
            result["modules"] = dict(groups)
            assert sum(item["kernel_count"] for item in groups.values()) == len(kernels)
        results.append(result)
    assert len(results) == 6, "Expected light full + 3 extend + detailed full/extend"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.sqlite)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in row.items()
                    if key not in ("kernels", "cuda_apis", "modules")
                }
            )
        )


if __name__ == "__main__":
    main()
