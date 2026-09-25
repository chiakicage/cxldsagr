"""Recover per-layer dense GPU activity durations from the existing nsys trace.

CUDA API launch times identify the enclosing host NVTX ranges. GPU event times
are used only for durations: queued kernels can start after their host ranges
have ended. Duration sums retain the original dense module MFU convention;
busy/overlap fields separately expose overlapping GPU activity such as MEMSET.
"""

from __future__ import annotations

import bisect
import sqlite3
from collections import defaultdict
from contextlib import closing
from itertools import pairwise
from numbers import Integral
from pathlib import Path

ROOT_RANGE = "GR/detailed/extend/0"
MODULE_PREFIX = "nosa::extend/"


def _activity_totals(events):
    total = sum(event["end"] - event["start"] for event in events)
    busy = 0
    cursor = None
    for event in sorted(events, key=lambda item: item["start"]):
        start, end = event["start"], event["end"]
        busy += end - start if cursor is None else max(0, end - max(start, cursor))
        cursor = end if cursor is None else max(cursor, end)
    return {
        "gpu_ms": total / 1e6,
        "gpu_busy_ms": busy / 1e6,
        "gpu_overlap_ms": (total - busy) / 1e6,
    }


def load_dense_layer_timing(sqlite_path, num_layers):
    """Read one detailed extend capture and return module activity sums in ms.

    ``layers`` separates decoder work from ``shared_modules`` (embedding,
    final norm, and any other work outside decoder scopes). Each layer must
    contain exactly one attention-core kernel, matching this dense baseline.
    The reader never modifies the SQLite database or its companion artifacts.
    """
    if isinstance(num_layers, bool) or not isinstance(num_layers, Integral) or num_layers < 1:
        raise ValueError("num_layers must be a positive integer")
    num_layers = int(num_layers)
    uri = Path(sqlite_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        strings = dict(connection.execute("SELECT id, value FROM StringIds"))
        scopes = []
        for row in connection.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL"
        ):
            scope = dict(row)
            scope["text"] = scope["text"] or strings.get(scope["textId"], "")
            scopes.append(scope)
        roots = [scope for scope in scopes if scope["text"] == ROOT_RANGE]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one {ROOT_RANGE} NVTX range, found {len(roots)}")
        root = roots[0]
        if root["start"] >= root["end"]:
            raise ValueError("Detailed extend NVTX range must have positive duration")
        modules = [
            scope
            for scope in scopes
            if scope["text"].startswith(MODULE_PREFIX)
            and scope["globalTid"] == root["globalTid"]
            and root["start"] <= scope["start"] < root["end"]
        ]
        modules.sort(key=lambda scope: (scope["start"], -scope["end"]))
        for scope in modules:
            parts = scope["text"].split("/")
            if len(parts) != 3 or not parts[1] or not parts[2]:
                raise ValueError(f"Malformed module NVTX label: {scope['text']}")
            if not scope["start"] <= scope["end"] <= root["end"]:
                raise ValueError("Module NVTX ranges must stay within the detailed extend range")
            scope["module"], scope["layer_label"] = parts[1:]
        decoders = [scope for scope in modules if scope["module"] == "decoder"]
        if len(decoders) != num_layers:
            raise ValueError(f"Expected {num_layers} decoder ranges, found {len(decoders)}")
        try:
            layer_ids = [int(scope["layer_label"]) for scope in decoders]
        except ValueError as error:
            raise ValueError("Decoder NVTX ranges must have integer layer IDs") from error
        if sorted(layer_ids) != list(range(num_layers)):
            raise ValueError("Decoder ranges must cover every layer exactly once")
        for previous, current in pairwise(decoders):
            if current["start"] < previous["end"]:
                raise ValueError("Overlapping decoder host ranges make layer attribution ambiguous")
        starts = [scope["start"] for scope in modules]
        decoder_starts = [scope["start"] for scope in decoders]

        apis = {}
        for row in connection.execute(
            "SELECT correlationId, start, end, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "WHERE globalTid = ? AND start >= ? AND start < ?",
            (root["globalTid"], root["start"], root["end"]),
        ):
            api = dict(row)
            correlation = api["correlationId"]
            if correlation in apis:
                raise ValueError("Duplicate CUDA correlation IDs inside the detailed extend range")
            apis[correlation] = api
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            raise ValueError("The source trace has no GPU kernel activity table")
        work = []
        for category in ("KERNEL", "MEMCPY", "MEMSET"):
            table = f"CUPTI_ACTIVITY_KIND_{category}"
            if table not in tables:
                continue
            for row in connection.execute(f"SELECT start, end, correlationId FROM {table}"):
                event = dict(row)
                if event["correlationId"] not in apis:
                    continue
                if event["start"] >= event["end"]:
                    raise ValueError("GPU activities must have positive durations")
                event["category"] = category
                work.append(event)

    layer_modules = [defaultdict(int) for _ in range(num_layers)]
    layer_events = [[] for _ in range(num_layers)]
    attention_events = [[] for _ in range(num_layers)]
    shared_modules = defaultdict(int)
    all_modules = defaultdict(int)
    for event in work:
        api = apis[event["correlationId"]]
        index = bisect.bisect_right(starts, api["start"]) - 1
        while index >= 0 and not modules[index]["start"] <= api["start"] < modules[index]["end"]:
            index -= 1
        if index < 0:
            raise ValueError(f"No module NVTX scope for CUDA correlation {event['correlationId']}")
        scope = modules[index]
        module = scope["module"]
        decoder_index = bisect.bisect_right(decoder_starts, api["start"]) - 1
        decoder = decoders[decoder_index] if decoder_index >= 0 else None
        if decoder is not None and not decoder["start"] <= api["start"] < decoder["end"]:
            decoder = None
        duration = event["end"] - event["start"]
        all_modules[module] += duration
        if decoder is None:
            if scope["layer_label"] != "shared" or module in ("attention_core", "rope_apply"):
                raise ValueError(f"Layer module {scope['text']} is outside a decoder NVTX range")
            shared_modules[module] += duration
        else:
            layer = int(decoder["layer_label"])
            if scope["layer_label"] not in ("shared", str(layer)):
                raise ValueError("Module layer ID disagrees with its enclosing decoder range")
            layer_modules[layer][module] += duration
            layer_events[layer].append(event)
            if module == "attention_core":
                attention_events[layer].append(event)
    for layer, events in enumerate(attention_events):
        if len(events) != 1 or events[0]["category"] != "KERNEL":
            raise ValueError(f"Layer {layer} must contain exactly one attention_core kernel")

    def durations(values):
        return {name: duration / 1e6 for name, duration in sorted(values.items())}

    totals = _activity_totals(work)
    return {
        "layers": [
            {
                "layer": layer,
                "modules": durations(layer_modules[layer]),
                **_activity_totals(layer_events[layer]),
            }
            for layer in range(num_layers)
        ],
        "shared_modules": durations(shared_modules),
        "modules": durations(all_modules),
        "total_gpu_ms": totals["gpu_ms"],
        "gpu_busy_ms": totals["gpu_busy_ms"],
        "gpu_overlap_ms": totals["gpu_overlap_ms"],
        "source_range": ROOT_RANGE,
        "duration_semantics": "sum of GPU activity durations, including overlapping MEMSET",
    }
