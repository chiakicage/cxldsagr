"""Conservative CPU/CUDA attribution without summing overlapping device intervals."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict


def merged(intervals):
    result = []
    for begin, end in sorted(intervals):
        if not all(math.isfinite(value) for value in (begin, end)) or end < begin:
            raise ValueError("invalid timeline interval")
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((begin, end))
    return result


def duration(intervals):
    return sum(end - begin for begin, end in merged(intervals))


class _ScopeIndex:
    """Find the smallest containing scope with the original stable tie order."""

    def __init__(self, scopes):
        threads = defaultdict(list)
        for order, scope in enumerate(scopes):
            threads[scope.get("tid")].append((order, scope))
        self.threads = {}
        for tid, entries in threads.items():
            entries.sort(key=lambda entry: entry[1]["ts"])
            starts, prefix_ends, right = [], [], float("-inf")
            for _, scope in entries:
                starts.append(scope["ts"])
                right = max(right, scope["ts"] + scope["dur"])
                prefix_ends.append(right)
            self.threads[tid] = starts, prefix_ends, entries

    def owner(self, cpu):
        index = self.threads.get(cpu.get("tid"))
        if index is None:
            return None
        starts, prefix_ends, entries = index
        position = bisect_right(starts, cpu["ts"]) - 1
        end = cpu["ts"] + cpu["dur"]
        chosen = None
        while position >= 0 and prefix_ends[position] >= end:
            order, scope = entries[position]
            if end <= scope["ts"] + scope["dur"]:
                priority = scope["dur"], order
                if chosen is None or priority < chosen[0]:
                    chosen = priority, scope
            position -= 1
        return None if chosen is None else chosen[1]


def analyze_trace(trace, root_name):
    events = [
        event
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and "ts" in event and "dur" in event
    ]
    roots = [
        event
        for event in events
        if event.get("name") == root_name and event.get("cat") == "user_annotation"
    ]
    if len(roots) != 1:
        raise ValueError("profile must contain exactly one measured request root")
    root = roots[0]
    begin, end = root["ts"], root["ts"] + root["dur"]
    cpu_by_external = {
        event["args"]["External id"]: event
        for event in events
        if event.get("cat") == "cpu_op" and "External id" in event.get("args", {})
    }
    runtime_by_correlation = {
        event["args"]["correlation"]: event
        for event in events
        if event.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in event.get("args", {})
    }
    scopes = [
        event
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith("nosa::")
    ]
    scope_index, owners = _ScopeIndex(scopes), {}
    categories, inventory, orphaned = defaultdict(list), defaultdict(list), []
    activities = [
        event for event in events if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    if not activities:
        raise ValueError("profile contains no CUDA device activity")
    for event in activities:
        interval = (event["ts"], event["ts"] + event["dur"])
        if interval[0] < begin or interval[1] > end:
            raise ValueError("device activity escaped the synchronized request root")
        category = "support_kernel"
        owner = None
        cpu = runtime_by_correlation.get(event.get("args", {}).get("correlation"))
        if cpu is None:
            cpu = cpu_by_external.get(event.get("args", {}).get("External id"))
        if cpu is not None:
            identity = id(cpu)
            if identity not in owners:
                owners[identity] = scope_index.owner(cpu)
            owner = owners[identity]
        if owner is not None and owner["name"] == "nosa::diagnostics":
            category = "diagnostics"
        elif event["cat"] == "gpu_memcpy":
            category = "memcpy"
        elif event["cat"] == "gpu_memset":
            category = "memset"
        elif owner is not None:
            category = owner["name"].split("::")[1].split("/")[0]
        else:
            orphaned.append(event["name"])
        categories[category].append(interval)
        inventory[(category, event["name"])].append(event["dur"])
    device_intervals = [(event["ts"], event["ts"] + event["dur"]) for event in activities]
    union_us = duration(device_intervals)
    runtime = [
        event
        for event in events
        if event.get("cat") in ("cuda_runtime", "cuda_driver") and begin <= event["ts"] < end
    ]
    api_inventory = defaultdict(list)
    for event in runtime:
        api_inventory[event["name"]].append(event["dur"])
    return {
        "clock": "torch_profiler_chrome_microseconds",
        "instrumented_execute_wall_us": root["dur"],
        "gpu_activity_union_us": union_us,
        "gpu_activity_sum_us": sum(event["dur"] for event in activities),
        "gpu_idle_inside_request_us": root["dur"] - union_us,
        "gpu_activity_fraction": union_us / root["dur"],
        "category_union_us": {name: duration(values) for name, values in categories.items()},
        "category_sum_us": {
            name: sum(end - begin for begin, end in values) for name, values in categories.items()
        },
        "kernel_inventory": [
            {"category": category, "name": name, "calls": len(values), "sum_us": sum(values)}
            for (category, name), values in sorted(inventory.items())
        ],
        "cuda_api_inventory": [
            {"name": name, "calls": len(values), "sum_us": sum(values)}
            for name, values in sorted(api_inventory.items())
        ],
        "unattributed_kernel_names": sorted(set(orphaned)),
        "boundaries": (
            "Instrumented runner.execute including post-latency diagnostics. Category unions "
            "are non-additive. Matrix scopes are "
            "complete Linear APIs; indexer/attention scopes include helper work. A fused sparse "
            "kernel is one sparse_attention interval and is never duplicated as fetch plus math. "
            "GPU-idle time includes CPU submission, dependency waits and synchronization; it "
            "is not by itself a measurement of CPU launch cost. Internal work uses a separate "
            "device-globaltimer diagnostic and its absolute timestamps are never mixed here."
        ),
    }
