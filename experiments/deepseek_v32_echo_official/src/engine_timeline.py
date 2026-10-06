"""Read Engine NVTX/CUPTI exports and plot the last history chunk and extend.

Run from the repository root with the project environment:
python -B -m experiments.deepseek_v32_echo_official.src.engine_timeline --help
Only matplotlib is required; the separate ECHO environment is not imported.
Each --capture and --benchmark argument has the form CASE=/absolute/path.
No GPU code is imported or executed. All timestamps remain in nanoseconds in
the source tables; the figures use elapsed milliseconds and a shared phase axis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from itertools import pairwise
from pathlib import Path

PREFIX = "echo_engine|"
PROCESS_MASK = 0xFFFFFFFFFF000000
CASES = {"hbm": "SGLang Engine HBM-only", "echo": "SGLang Engine ECHO"}
COMPUTE_STAGES = {
    "projection": "Projection / RoPE",
    "embedding": "Projection / RoPE",
    "indexer": "Indexer / top-k",
    "topk": "Indexer / top-k",
    "indexer_prefetch": "Indexer / top-k",
    "attention": "Attention",
    "output_mlp": "Output / MLP",
    "final_norm": "Output / MLP",
    "lm_head": "Output / MLP",
}
COLORS = {
    "Projection / RoPE": "#56B4E9",
    "Indexer / top-k": "#0072B2",
    "Attention": "#009E73",
    "Output / MLP": "#CC79A7",
    "H2D": "#E69F00",
    "D2H": "#7B3294",
    "GPU idle": "#C43C39",
    "ECHO prepare": "#B38B42",
    "ECHO finalize": "#4D4D4D",
    "ECHO hint": "#8AAB38",
    "ECHO recall (IO unknown)": "#777777",
}
CONTROL_STAGES = {
    "echo_prepare": "ECHO prepare",
    "echo_finalize": "ECHO finalize",
    "echo_hint": "ECHO hint",
}
CONTROL_CATEGORIES = {*CONTROL_STAGES.values(), "ECHO recall (IO unknown)"}
COPY_DIRECTIONS = {
    "host-to-device": "H2D",
    "host-to-array": "H2D",
    "unified host-to-device": "H2D",
    "device-to-host": "D2H",
    "array-to-host": "D2H",
    "unified device-to-host": "D2H",
}
NOTES = [
    "Intrusive NSYS profile durations are separate from unprofiled Engine wall times.",
    (
        "GPU ownership follows process/correlation IDs to runtime or driver launch APIs, "
        "then the innermost same-thread Engine NVTX scope. GPU time is not clipped to CPU scopes."
    ),
    (
        "The display window starts at the first L0 compute kernel and ends at the last L2 "
        "compute kernel. Full phase GPU envelopes, CPU forward scopes, and Engine.generate "
        "request NVTX windows are retained separately, with their exact boundaries."
    ),
    (
        "GPU idle means no captured kernel, memcpy, or memset on the selected device in the "
        "window, including activity without a stage attribution. It is not GPU stall time."
    ),
    (
        "H2D/D2H intervals show CUPTI memcpy activities and destination-proven mapped-host writes. "
        "Hatched intervals identify dynamic mapped-host paths; fused prefetch and recall have guards "
        "and may transfer zero records. Recall without nonzero traffic evidence stays in the "
        "ECHO control row, not H2D. Per-call kernel transfer bytes are unknown and are never "
        "inferred from duration or launch shape. Destination-aware hooks identify writes."
    ),
    (
        "CUPTI memcpy bytes include metadata and output copies, not only main KV. Bytes "
        "are summed for complete observed copy records and never prorated at window edges."
    ),
    (
        "Other GPU control, D2D and memset activity is retained in source data and busy time "
        "but omitted from colored lanes. ECHO control rows show GPU activity, not CPU scope duration."
    ),
]


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def union(intervals):
    result = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


def duration(intervals):
    return sum(right - left for left, right in union(intervals))


def complement(intervals, start, end):
    cursor, result = start, []
    for left, right in union((max(start, a), min(end, b)) for a, b in intervals):
        if left > cursor:
            result.append([cursor, left])
        cursor = max(cursor, right)
    if cursor < end:
        result.append([cursor, end])
    return result


def parse_label(label):
    if not label.startswith(PREFIX):
        return None
    fields = {}
    for item in label[len(PREFIX) :].split("|"):
        key, separator, value = item.partition("=")
        if not separator or not key or key in fields:
            raise ValueError(f"Malformed Engine NVTX label: {label}")
        fields[key] = int(value) if re.fullmatch(r"-?\d+", value) else value
    if fields.get("kind") not in {"forward", "layer", "stage", "request", "diagnostic"}:
        raise ValueError(f"Unknown Engine NVTX kind: {label}")
    return fields


def read_capture(path):
    """Read the documented NSYS tables without mutating the export."""
    path = Path(path).resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        used = []

        def rows(table):
            # Names originate from fixed constants, never user SQL.
            if table not in tables:
                return []
            used.append(table)
            return [dict(row) for row in db.execute(f'SELECT * FROM "{table}"')]

        strings = {row["id"]: row["value"] for row in rows("StringIds")}

        def name(row, *keys):
            for key in keys:
                value = row.get(key)
                if isinstance(value, str) and value:
                    return value
                if value in strings:
                    return strings[value]
            raise ValueError(f"Missing string in NSYS record: {keys}")

        scopes = []
        for row in rows("NVTX_EVENTS"):
            label = (
                name(row, "text", "textId")
                if row.get("text") or row.get("textId") is not None
                else ""
            )
            fields = parse_label(label)
            if fields is None:
                continue
            if row.get("end") is None or row.get("globalTid") is None:
                raise ValueError(f"Incomplete Engine NVTX range: {label}")
            scopes.append(
                {
                    "scope_id": len(scopes),
                    "start_ns": row["start"],
                    "end_ns": row["end"],
                    "thread": row["globalTid"],
                    "label": label,
                    "fields": fields,
                }
            )
        apis = []
        for source in ("RUNTIME", "DRIVER"):
            for row in rows(f"CUPTI_ACTIVITY_KIND_{source}"):
                thread = row.get("globalTid")
                if thread is None:
                    raise ValueError("CUPTI API record has no globalTid")
                apis.append(
                    {
                        "api_id": len(apis),
                        "start_ns": row["start"],
                        "end_ns": row["end"],
                        "thread": thread,
                        "process": row.get("globalPid", thread & PROCESS_MASK),
                        "correlation": row.get("correlationId"),
                        "source": source,
                        "name": name(row, "name", "nameId"),
                    }
                )
        copies = {row["id"]: row["label"] for row in rows("ENUM_CUDA_MEMCPY_OPER")}
        activities = []
        for kind, alternatives in (
            ("kernel", ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")),
            ("memcpy", ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMCPY2")),
            ("memset", ("CUPTI_ACTIVITY_KIND_MEMSET",)),
        ):
            table = next((item for item in alternatives if item in tables), None)
            for row in rows(table) if table else []:
                activity_name = (
                    name(row, "demangledName", "shortName", "name", "nameId", "mangledName")
                    if kind == "kernel"
                    else copies.get(row.get("copyKind"), "Unknown")
                    if kind == "memcpy"
                    else "memset"
                )
                activities.append(
                    {
                        "activity_id": len(activities),
                        "kind": kind,
                        "name": activity_name,
                        "start_ns": row["start"],
                        "end_ns": row["end"],
                        "device": row["deviceId"],
                        "stream": row.get("streamId"),
                        "process": row.get("globalPid"),
                        "correlation": row.get("correlationId"),
                        "bytes": row.get("bytes"),
                        "graph_id": row.get("graphId"),
                        "graph_node_id": row.get("graphNodeId"),
                        "copy_kind": row.get("copyKind"),
                    }
                )
        for row in [*scopes, *apis, *activities]:
            if (
                type(row["start_ns"]) is not int
                or type(row["end_ns"]) is not int
                or row["end_ns"] < row["start_ns"]
            ):
                raise ValueError("Invalid NSYS timestamp")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "tables": used,
        "scopes": scopes,
        "apis": apis,
        "activities": activities,
    }


def attribute(capture):
    """Join nesting and asynchronous launch correlation, retaining all activity."""
    by_thread = defaultdict(list)
    for scope in capture["scopes"]:
        by_thread[scope["thread"]].append(scope)
    for scopes in by_thread.values():
        stack = []
        for scope in sorted(scopes, key=lambda row: (row["start_ns"], -row["end_ns"])):
            while stack and scope["start_ns"] >= stack[-1]["end_ns"]:
                stack.pop()
            if stack and scope["end_ns"] > stack[-1]["end_ns"]:
                raise ValueError("Engine NVTX scopes must be properly nested on each thread")
            scope["context"] = {**(stack[-1]["context"] if stack else {}), **scope["fields"]}
            stack.append(scope)
    api_threads = defaultdict(list)
    for api in capture["apis"]:
        api_threads[api["thread"]].append(api)
    for thread, apis in api_threads.items():
        scopes = sorted(by_thread[thread], key=lambda row: (row["start_ns"], -row["end_ns"]))
        cursor, active = 0, []
        for api in sorted(apis, key=lambda row: row["start_ns"]):
            while cursor < len(scopes) and scopes[cursor]["start_ns"] <= api["start_ns"]:
                scope = scopes[cursor]
                heapq.heappush(
                    active, (-scope["start_ns"], scope["end_ns"], scope["scope_id"], scope)
                )
                cursor += 1
            while active and active[0][1] <= api["start_ns"]:
                heapq.heappop(active)
            scope = active[0][-1] if active else None
            api["scope_id"] = scope["scope_id"] if scope else None
            api["context"] = scope["context"] if scope else {}
    correlations = defaultdict(list)
    for api in capture["apis"]:
        if api["correlation"] is not None:
            correlations[api["correlation"]].append(api)
    for row in capture["activities"]:
        matches = [
            api
            for api in correlations[row["correlation"]]
            if api["start_ns"] <= row["start_ns"]
            and (row["process"] is None or api["process"] == row["process"])
        ]
        if row["process"] is None and len({api["process"] for api in matches}) > 1:
            raise ValueError("Ambiguous process for CUPTI GPU activity")
        api = (
            max(
                matches,
                key=lambda item: (item["start_ns"], item["source"] == "DRIVER", -item["end_ns"]),
            )
            if matches
            else None
        )
        context = api["context"] if api else {}
        row.update(
            api_id=api["api_id"] if api else None,
            api_start_ns=api["start_ns"] if api else None,
            api_end_ns=api["end_ns"] if api else None,
            api_name=api["name"] if api else None,
            scope_id=api["scope_id"] if api else None,
            scope_kind=context.get("kind"),
            forward_id=context.get("id"),
            phase=context.get("phase"),
            layer=context.get("layer"),
            stage=context.get("stage"),
            direction=context.get("direction"),
            scope_case=context.get("case"),
        )
        classify(row)
    return capture


def classify(row):
    """Separate observed memcpy from source-identified, dynamically gated IO."""
    stage, name = row.get("stage"), row["name"].lower()
    row.update(
        lane="GPU control",
        category="GPU control",
        io_evidence=None,
        potential_io=False,
        measured_io_bytes=None,
    )
    if row["kind"] == "memcpy":
        direction = COPY_DIRECTIONS.get(name)
        if direction:
            if type(row["bytes"]) is not int or row["bytes"] < 0:
                raise ValueError("Host memcpy lacks a native nonnegative byte count")
            row.update(
                lane=direction,
                category=direction,
                measured_io_bytes=row["bytes"],
                io_evidence="CUPTI memcpy copyKind and bytes",
            )
        return
    if row["kind"] != "kernel":
        return
    if "sm90_fp8_mqa_logits_fuse_prefetch" in name:
        row.update(
            lane="Compute",
            category="Indexer / top-k",
            potential_io=True,
            io_evidence="Fused indexer contains dynamically gated mapped-host H2D; per-call bytes unknown",
        )
    elif "_recall_update_extend_kernel" in name or "_recall_update_with_prefetch_kernel" in name:
        row.update(
            lane="GPU control",
            category="ECHO recall (IO unknown)",
            potential_io=True,
            io_evidence="Source-identified recall kernel with dynamic miss count; per-call bytes unknown",
        )
    elif stage == "io_d2h" and "set_mla_kv_buffer_kernel" in name:
        if row.get("direction") != "D2H":
            raise ValueError("Mapped-host write lacks destination-aware direction metadata")
        row.update(
            lane="D2H",
            category="D2H",
            io_evidence="Destination-aware mapped-host transfer NVTX; per-call bytes unknown",
        )
    elif stage in CONTROL_STAGES:
        row.update(category=CONTROL_STAGES[stage])
    elif (
        stage is None
        and row.get("layer") == -1
        and "flashinfer::norm::fusedaddrmsnormkernel" in name
    ):
        row.update(
            lane="Compute",
            category="Output / MLP",
            classification_note="Shared final norm outside decoder-layer scopes; raw CUPTI overlap may clip its start into the L2 window",
        )
    elif stage in COMPUTE_STAGES:
        # Pure storage/layout and integer cache operations do not become model
        # computation merely because an outer Python operator encloses them.
        storage = (
            "catarraybatchedcopy",
            "fillfunctor<",
            "index_put_kernel",
            "indexkernel",
            "arange_cuda",
            "transpose_fp32",
        )
        pure_copy = "direct_copy_kernel_cuda" in name and "loadwithcast" not in name
        if not pure_copy and not any(token in name for token in storage):
            row.update(lane="Compute", category=COMPUTE_STAGES[stage])


def summarize_window(activities, start, end):
    if start >= end:
        raise ValueError("Timeline window is empty")
    selected = [row for row in activities if row["start_ns"] < end and row["end_ns"] > start]

    def intervals(rows):
        return [(max(start, row["start_ns"]), min(end, row["end_ns"])) for row in rows]

    busy = union(intervals(selected))
    idle = complement(busy, start, end)
    compute = duration(intervals(row for row in selected if row["lane"] == "Compute"))
    metrics = {
        "start_ns": start,
        "end_ns": end,
        "window_ms": (end - start) / 1e6,
        "gpu_busy_ms": duration(busy) / 1e6,
        "gpu_idle_ms": duration(idle) / 1e6,
        "compute_union_ms": compute / 1e6,
        "gpu_idle_intervals_ns": idle,
        "activity_count": len(selected),
        "kind_counts": dict(Counter(row["kind"] for row in selected)),
        "unattributed_activity_count": sum(row["scope_id"] is None for row in selected),
        "unclassified_kernel_count": sum(
            row["kind"] == "kernel" and row["stage"] is None and row["category"] == "GPU control"
            for row in selected
        ),
        "kernel_without_stage_scope_count": sum(
            row["kind"] == "kernel" and row["stage"] is None for row in selected
        ),
    }
    if duration(busy) + duration(idle) != end - start:
        raise ValueError("Busy + idle does not conserve the window")
    for direction in ("H2D", "D2H"):
        copies = [row for row in selected if row["lane"] == direction and row["kind"] == "memcpy"]
        metrics[f"{direction.lower()}_memcpy_union_ms"] = duration(intervals(copies)) / 1e6
        metrics[f"{direction.lower()}_memcpy_count"] = len(copies)
        metrics[f"{direction.lower()}_memcpy_bytes_whole_overlapping_records"] = sum(
            row["bytes"] for row in copies
        )
        metrics[f"{direction.lower()}_memcpy_crosses_window_edge"] = sum(
            row["start_ns"] < start or row["end_ns"] > end for row in copies
        )
        metrics[f"{direction.lower()}_mapped_kernel_count"] = sum(
            row["lane"] == direction and row["kind"] == "kernel" for row in selected
        )
    metrics["mapped_host_transfer_bytes"] = None
    metrics["fused_potential_io_kernel_count"] = sum(
        row["lane"] == "Compute" and row["potential_io"] for row in selected
    )
    metrics["potential_h2d_control_kernel_count"] = sum(
        row["lane"] == "GPU control" and row["potential_io"] for row in selected
    )
    metrics["category_union_ms"] = {
        category: duration(intervals(row for row in selected if row["category"] == category)) / 1e6
        for category in sorted({row["category"] for row in selected})
    }
    return metrics


def select_phases(capture, *, case, history, candidate, chunk, occurrence=-1):
    """Require a complete history sweep and a complete appended query batch."""
    forwards = sorted(
        (scope for scope in capture["scopes"] if scope["fields"]["kind"] == "forward"),
        key=lambda row: row["start_ns"],
    )
    if not forwards:
        raise ValueError(f"No Engine forward NVTX ranges in {capture['path']}")
    if any(scope["fields"].get("case") != case for scope in forwards):
        raise ValueError("Capture case differs from its Engine NVTX labels")
    if len({scope["fields"]["id"] for scope in forwards}) != len(forwards):
        raise ValueError("Forward IDs are not unique within the capture")
    sweeps, current = [], []
    for scope in forwards:
        fields = scope["fields"]
        if fields.get("phase") != "prefill":
            continue
        start, end, q = fields.get("start"), fields.get("end"), fields.get("q")
        if start == 0:
            if current:
                raise ValueError("Interrupted prefill history sweep")
            current = []
        if (
            start != (current[-1]["fields"]["end"] if current else 0)
            or q != end - start
            or q != min(chunk, history - start)
        ):
            raise ValueError(f"Incomplete/incorrect prefill query coverage: {fields}")
        current.append(scope)
        if end == history:
            sweeps.append(current)
            current = []
    if current:
        raise ValueError("Trailing incomplete prefill history sweep")
    extends = [
        scope
        for scope in forwards
        if scope["fields"].get("phase") == "extend"
        and scope["fields"].get("start") == history
        and scope["fields"].get("end") == history + candidate
        and scope["fields"].get("q") == candidate
    ]
    if not sweeps or not extends:
        raise ValueError("Capture needs a complete prefill sweep and an H+A extend")
    try:
        selection = {"prefill": sweeps[occurrence], "extend": [extends[occurrence]]}
    except IndexError as error:
        raise ValueError("Selected phase occurrence is absent") from error
    panels = {}
    for phase, scopes in selection.items():
        ids = {scope["fields"]["id"] for scope in scopes}
        owned = [row for row in capture["activities"] if row["forward_id"] in ids]
        forward_rows = []
        for scope in scopes:
            fid = scope["fields"]["id"]
            layer_ids = [
                row["fields"]["layer"]
                for row in capture["scopes"]
                if row["fields"]["kind"] == "layer" and row["fields"].get("id") == fid
            ]
            if sorted(layer_ids) != [0, 1, 2]:
                raise ValueError(f"Forward {fid} does not cover exactly layers 0, 1, 2")
            gpu = [row for row in owned if row["forward_id"] == fid]
            if not gpu:
                raise ValueError(f"Forward {fid} has no correlated GPU work")
            compute_layers = {row["layer"] for row in gpu if row["lane"] == "Compute"}
            if not {0, 1, 2} <= compute_layers:
                raise ValueError(f"Forward {fid} lacks compute kernels for a model layer")
            forward_rows.append(
                {
                    **scope["fields"],
                    "cpu_start_ns": scope["start_ns"],
                    "cpu_end_ns": scope["end_ns"],
                    "gpu_start_ns": min(row["start_ns"] for row in gpu),
                    "gpu_end_ns": max(row["end_ns"] for row in gpu),
                    "kind_counts": dict(Counter(row["kind"] for row in gpu)),
                    "gpu_duration_sum_ns": sum(row["end_ns"] - row["start_ns"] for row in gpu),
                    "layer_compute_kernel_counts": {
                        str(layer): sum(
                            row["layer"] == layer and row["lane"] == "Compute" for row in gpu
                        )
                        for layer in range(3)
                    },
                }
            )
        devices = {row["device"] for row in owned}
        if len(devices) != 1:
            raise ValueError(f"Expected one GPU for selected {phase}; got {devices}")
        device = next(iter(devices))
        all_device = [row for row in capture["activities"] if row["device"] == device]
        forward_id = scopes[-1]["fields"]["id"]
        chosen = [row for row in owned if row["forward_id"] == forward_id]
        endpoints = []
        for layer in range(3):
            rows = [row for row in chosen if row["layer"] == layer and row["lane"] == "Compute"]
            if not rows:
                raise ValueError(f"Missing {case} {phase} L{layer} compute attribution")
            first, last = (
                min(rows, key=lambda row: row["start_ns"]),
                max(rows, key=lambda row: row["end_ns"]),
            )
            endpoints.append(
                {
                    "layer": layer,
                    "start_ns": first["start_ns"],
                    "end_ns": last["end_ns"],
                    "first_activity_id": first["activity_id"],
                    "last_activity_id": last["activity_id"],
                    "first_kernel": first["name"],
                    "last_kernel": last["name"],
                    "compute_kernel_count": len(rows),
                }
            )
        layer_overlaps = [
            {
                "left_layer": left["layer"],
                "right_layer": right["layer"],
                "overlap_ns": max(0, left["end_ns"] - right["start_ns"]),
            }
            for left, right in pairwise(endpoints)
        ]
        start, end = endpoints[0]["start_ns"], endpoints[-1]["end_ns"]
        window = summarize_window(all_device, start, end)
        full_start, full_end = (
            min(row["start_ns"] for row in owned),
            max(row["end_ns"] for row in owned),
        )
        requests = [
            scope
            for scope in capture["scopes"]
            if scope["fields"]["kind"] == "request"
            and scope["fields"].get("case") == case
            and scope["fields"].get("phase") == phase
            and scope["start_ns"] <= scopes[0]["start_ns"]
            and scope["end_ns"] >= scopes[-1]["end_ns"]
        ]
        if len(requests) > 1:
            raise ValueError(f"Ambiguous enclosing Engine request NVTX range for {phase}")
        request = requests[0] if requests else None
        panels[phase] = {
            "case": case,
            "phase": phase,
            "device": device,
            "forward_id": forward_id,
            "query": scopes[-1]["fields"],
            "prefill_chunks": len(scopes) if phase == "prefill" else None,
            "window": window,
            "layers": endpoints,
            "adjacent_layer_compute_envelope_overlaps": layer_overlaps,
            "full_phase_gpu_envelope": summarize_window(all_device, full_start, full_end),
            "engine_request_nvtx_window": summarize_window(
                all_device, request["start_ns"], request["end_ns"]
            )
            if request
            else None,
            "engine_request_nvtx_status": "captured" if request else "missing_or_incomplete",
            "full_phase_forward_cpu_envelope": {
                "start_ns": scopes[0]["start_ns"],
                "end_ns": scopes[-1]["end_ns"],
                "wall_ms": (scopes[-1]["end_ns"] - scopes[0]["start_ns"]) / 1e6,
            },
            "phase_forward_ids": sorted(ids),
            "forwards": forward_rows,
            "phase_attributed_kind_counts": dict(Counter(row["kind"] for row in owned)),
            "rows": [row for row in all_device if row["start_ns"] < end and row["end_ns"] > start],
        }
    return panels


def draw_phase(panels, phase, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 7.0), squeeze=False)
    limit = max(panel["window"]["window_ms"] for panel in panels) * 1.025
    for ax, panel in zip(axes[:, 0], panels):
        start, end = panel["window"]["start_ns"], panel["window"]["end_ns"]
        control_row = panel["case"] == "echo"
        labels = ["GPU idle", *(["ECHO control"] if control_row else []), "D2H", "H2D", "Compute"]
        lane_y = {label: index for index, label in enumerate(labels)}
        for row in panel["rows"]:
            category = row["category"]
            dynamic_io = row["potential_io"]
            lane = (
                row["lane"]
                if row["lane"] in {"Compute", "H2D", "D2H"}
                else "ECHO control"
                if category in CONTROL_CATEGORIES
                else None
            )
            if lane is None:
                continue
            left, right = (
                (max(start, row["start_ns"]) - start) / 1e6,
                (min(end, row["end_ns"]) - start) / 1e6,
            )
            ax.broken_barh(
                [(left, right - left)],
                (lane_y[lane] - 0.27, 0.54),
                facecolors=COLORS[category],
                edgecolors="#FFFFFF" if dynamic_io else "none",
                linewidth=0.35,
                hatch="////" if dynamic_io else None,
            )
        for left, right in panel["window"]["gpu_idle_intervals_ns"]:
            ax.broken_barh(
                [((left - start) / 1e6, (right - left) / 1e6)],
                (-0.27, 0.54),
                facecolors=COLORS["GPU idle"],
            )
        for layer in panel["layers"]:
            left, right = (layer["start_ns"] - start) / 1e6, (layer["end_ns"] - start) / 1e6
            ax.text(
                (left + right) / 2,
                lane_y["Compute"] + 0.47,
                f"L{layer['layer']}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
            if layer["layer"]:
                ax.axvline(left, color="#777777", linewidth=0.6, linestyle=":")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlim(0, limit)
        ax.axvline(panel["window"]["window_ms"], color="#777777", linewidth=0.7, linestyle="--")
        ax.set_ylim(-0.55, lane_y["Compute"] + 1.0)
        ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(
            f"{CASES[panel['case']]}    {panel['window']['window_ms']:.3f} ms; GPU idle {panel['window']['gpu_idle_ms'] * 1000:.1f} µs",
            loc="left",
            fontsize=11,
        )
        ax.set_xlabel("Elapsed time from first L0 compute kernel (ms)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [
        Patch(facecolor=COLORS[key], label=key)
        for key in ("Projection / RoPE", "Indexer / top-k", "Attention", "Output / MLP")
    ]
    handles.extend(
        [
            Patch(
                facecolor="white",
                edgecolor="#555555",
                hatch="////",
                label="Dynamic mapped-host path",
            )
        ]
    )
    shown = {row["category"] for panel in panels for row in panel["rows"]}
    handles.extend(
        Patch(facecolor=COLORS[key], label=key) for key in ("H2D", "D2H") if key in shown
    )
    handles.extend(
        Patch(facecolor=COLORS[key], label=key)
        for key in COLORS
        if key in CONTROL_CATEGORIES and key in shown
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.125),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    query = panels[0]["query"]
    title = (
        f"Prefill | last chunk {panels[0]['prefill_chunks']}/{panels[0]['prefill_chunks']}, "
        f"{query['q']:,} tokens, L0–L2"
        if phase == "prefill"
        else f"Extend | {query['q']} candidate tokens, L0–L2"
    )
    fig.suptitle(title, y=0.98, fontsize=12)
    boundary = (
        "Window: L0 first compute start to L2 last compute end; shared time scale. "
        f"Token positions {query['start']:,}–{query['end'] - 1:,}."
    )
    fig.text(
        0.02,
        0.02,
        "Intrusive NSYS profile; displayed windows are not independent Engine benchmark times.\n"
        "Red: no captured GPU activity. H2D: host to GPU; D2H: GPU to host. Other GPU control remains uncolored.\n"
        "Hatched paths may transfer zero records; mapped-host transfer bytes are unknown. Unproven recall stays in ECHO control.\n"
        + boundary,
        fontsize=8.5,
        linespacing=1.45,
    )
    fig.tight_layout(rect=(0, 0.22, 1, 0.96), h_pad=1.3)
    for suffix in ("svg", "png"):
        fig.savefig(output / f"timeline_{phase}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing empty source table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def assignment(value):
    case, separator, filename = value.partition("=")
    if not separator or case not in CASES or not filename:
        raise argparse.ArgumentTypeError("Expected hbm=PATH or echo=PATH")
    return case, Path(filename).resolve(strict=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        action="append",
        type=assignment,
        required=True,
        help="CASE=NSYS_SQLITE (one capture containing both phases per case)",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        type=assignment,
        required=True,
        help="CASE=BENCHMARK_JSON; retained as separate unprofiled provenance",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory; existing data is never overwritten",
    )
    parser.add_argument("--history", type=int, default=65536)
    parser.add_argument("--candidate", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument(
        "--occurrence", type=int, default=-1, help="Complete phase occurrence; -1 selects the last"
    )
    args = parser.parse_args(argv)
    captures, benchmarks = dict(args.capture), dict(args.benchmark)
    if len(captures) != len(args.capture) or len(benchmarks) != len(args.benchmark):
        parser.error("Each case may appear only once per option")
    if set(captures) != set(CASES) or set(benchmarks) != set(CASES):
        parser.error("Both hbm and echo captures and benchmarks are required")
    if min(args.history, args.candidate, args.chunk) <= 0:
        parser.error("Token counts must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    all_panels, activity_rows, scope_rows, api_rows, sources = {}, [], [], [], {}
    for case in CASES:
        capture = attribute(read_capture(captures[case]))
        all_panels[case] = select_phases(
            capture,
            case=case,
            history=args.history,
            candidate=args.candidate,
            chunk=args.chunk,
            occurrence=args.occurrence,
        )
        activity_rows.extend({"case": case, **row} for row in capture["activities"])
        scope_rows.extend({"case": case, **row} for row in capture["scopes"])
        api_rows.extend({"case": case, **row} for row in capture["apis"])
        sources[case] = {
            "capture": {key: capture[key] for key in ("path", "sha256", "tables")},
            "benchmark": {
                "path": str(benchmarks[case]),
                "sha256": sha256(benchmarks[case]),
                "metadata": json.loads(benchmarks[case].read_text()),
            },
            "capture_counts": {key: len(capture[key]) for key in ("scopes", "apis", "activities")},
            "capture_activity_duration_sum_ns": sum(
                row["end_ns"] - row["start_ns"] for row in capture["activities"]
            ),
        }
    args.output_dir.mkdir(parents=True)
    windows = []
    for phase in ("prefill", "extend"):
        panels = [all_panels[case][phase] for case in CASES]
        draw_phase(panels, phase, args.output_dir)
        for panel in panels:
            windows.append(
                {
                    "case": panel["case"],
                    "phase": phase,
                    **{
                        key: value
                        for key, value in panel["window"].items()
                        if not isinstance(value, (list, dict))
                    },
                }
            )
    write_csv(args.output_dir / "activities.csv", activity_rows)
    write_csv(args.output_dir / "nvtx_scopes.csv", scope_rows)
    write_csv(args.output_dir / "cuda_apis.csv", api_rows)
    write_csv(args.output_dir / "windows.csv", windows)
    summary = {
        "schema": "echo-sglang-engine-timeline-v1",
        "notes": NOTES,
        "sources": sources,
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "selection": {
            key: getattr(args, key) for key in ("history", "candidate", "chunk", "occurrence")
        },
        "panels": {
            case: {
                phase: {key: value for key, value in panel.items() if key != "rows"}
                for phase, panel in phases.items()
            }
            for case, phases in all_panels.items()
        },
    }
    (args.output_dir / "timeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "windows": windows}, indent=2))


if __name__ == "__main__":
    main()
