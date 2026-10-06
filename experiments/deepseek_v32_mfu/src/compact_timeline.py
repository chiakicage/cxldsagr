"""Draw prefill and extend with one compute lane and explicit window bounds."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments.deepseek_v32_mfu.src import analyze_nsys, launch_gap, redraw_gap_timeline, timeline
from experiments.deepseek_v32_mfu.src.redraw_gap_timeline import RUN, gap_intervals, read, sha

GROUPS = {
    "Embedding": "Projection / RoPE",
    "Projection / RoPE": "Projection / RoPE",
    "Indexer logits": "Indexer / top-k",
    "Exact top-k": "Indexer / top-k",
    "Indexer + prefetch": "Indexer / top-k",
    "Sparse MLA": "Attention",
    "V expand": "Attention",
    "Output / MLP": "Output / MLP / head",
    "Final norm": "Output / MLP / head",
    "LM head": "Output / MLP / head",
    "Final norm / LM head": "Output / MLP / head",
}
COLORS = {
    "Projection / RoPE": "#56B4E9",
    "Indexer / top-k": "#0072B2",
    "Attention": "#009E73",
    "Output / MLP / head": "#CC79A7",
    "H2D: DRAM to GPU": "#E69F00",
    "D2H: GPU to DRAM": "#7B3294",
    "Gap": "#C43C39",
}
ANNOTATION_COLORS = {
    "GPU idle": "#C43C39",
    "ECHO prepare": "#B38B42",
    "ECHO finalize": "#4D4D4D",
    "ECHO hint": "#8AAB38",
}
ECHO_STAGES = {
    "offload_prepare": "ECHO prepare",
    "offload_finalize": "ECHO finalize",
    "prefetch_hint": "ECHO hint",
}


def idle_echo_annotations(panel):
    """Keep all-GPU-idle and selected ECHO GPU activity intervals separate."""
    window = panel["window"]
    start, end = window["start_ns"], window["end_ns"]
    active = sorted(
        (max(start, row["raw_start_ns"]), min(end, row["raw_end_ns"]))
        for row in panel["rows"]
        if row["kind"] != "api" and row["raw_start_ns"] < end and row["raw_end_ns"] > start
    )
    idle, cursor = [], start
    for left, right in active:
        if left > cursor:
            idle.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        idle.append((cursor, end))
    if sum(right - left for left, right in idle) != round(window["gpu_idle_ms"] * 1e6):
        raise ValueError("GPU idle differs from the accepted window")
    selected = {"GPU idle": idle}
    if panel["method"] == "echo":
        for stage, label in ECHO_STAGES.items():
            intervals = sorted(
                (max(start, row["raw_start_ns"]), min(end, row["raw_end_ns"]))
                for row in panel["rows"]
                if row["kind"] != "api"
                and row["stage"] == stage
                and row["raw_start_ns"] < end
                and row["raw_end_ns"] > start
            )
            merged = []
            for left, right in intervals:
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
                else:
                    merged.append((left, right))
            selected[label] = merged
    return selected


def io_direction(row):
    """Distinguish directions using retained memcpy kind or gather evidence."""
    name = row["name"].lower()
    if row["kind"] == "memcpy":
        if name in {"host-to-device", "unified host-to-device", "host-to-array"}:
            return "H2D: DRAM to GPU"
        if name in {"device-to-host", "unified device-to-host", "array-to-host"}:
            return "D2H: GPU to DRAM"
    evidence = row.get("actual_io") or {}
    if row["kind"] == "kernel" and evidence.get("family") == "mapped_host_kv_gather":
        return "H2D: DRAM to GPU"
    raise ValueError(f"Unrecognized actual IO direction: {row['kind']} {row['name']}")


def three_layer_panel(panel, phase):
    """Clip saved activities to the three-layer compute/IO window.

    Prefill selects only the last chunk using its accepted lower bound. Kernel
    lanes and actual-IO evidence are inherited without reclassification. Dense
    extend also includes L0 history H2D when it starts before L0 computation.
    """
    cutoff = panel["chunks"][-1]["start_ns"] if phase == "prefill" else panel["window"]["start_ns"]
    endpoints = []
    for layer in range(3):
        compute = [
            row
            for row in panel["rows"]
            if row["lane"] in {"Compute", "Compute + IO"}
            and row["layer"] == f"layer_{layer}"
            and row["raw_start_ns"] >= cutoff
        ]
        if not compute:
            raise ValueError(f"Missing {phase} L{layer} computation")
        first = min(compute, key=lambda row: row["raw_start_ns"])
        last = max(compute, key=lambda row: row["raw_end_ns"])
        endpoints.append(
            {
                "layer": layer,
                "compute_start_ns": first["raw_start_ns"],
                "compute_end_ns": last["raw_end_ns"],
                "first_compute": first,
                "last_compute": last,
                "compute_kernel_count": len(compute),
            }
        )
    start, end = endpoints[0]["compute_start_ns"], endpoints[-1]["compute_end_ns"]
    l0_history_h2d_start = None
    dense_extend = phase == "extend" and panel["method"] == "dense_prefetch"
    if dense_extend:
        history_h2d = [
            row["raw_start_ns"]
            for row in panel["rows"]
            if row["lane"] == "IO"
            and "dense_history_prefetch_layer_0" in {row["stage"], row.get("source_stage")}
            and io_direction(row) == "H2D: DRAM to GPU"
        ]
        if history_h2d:
            l0_history_h2d_start = min(history_h2d)
            if l0_history_h2d_start < panel["window"]["start_ns"]:
                raise ValueError("L0 history H2D starts outside the retained full activity window")
            start = min(start, l0_history_h2d_start)
    if end != panel["shared_tail"]["start_ns"]:
        raise ValueError("L2 compute end does not match the accepted shared-tail boundary")
    for previous, following in pairwise(endpoints):
        if (
            previous["compute_start_ns"] >= following["compute_start_ns"]
            or previous["compute_end_ns"] >= following["compute_end_ns"]
        ):
            raise ValueError("Expected increasing layer compute start and end timestamps")
    rows = [
        {
            **row,
            "start_ns": max(start, row["raw_start_ns"]),
            "end_ns": min(end, row["raw_end_ns"]),
            "start_ms": (max(start, row["raw_start_ns"]) - start) / 1e6,
            "end_ms": (min(end, row["raw_end_ns"]) - start) / 1e6,
        }
        for row in panel["rows"]
        if row["raw_start_ns"] < end and row["raw_end_ns"] > start
    ]
    activities = [
        {
            "start": row["raw_start_ns"],
            "end": row["raw_end_ns"],
            "lane": row["lane"],
            "actual_io": row.get("actual_io") or {},
        }
        for row in rows
        if row["kind"] != "api"
    ]
    window = launch_gap.summarize_window(
        activities, start, end, lane_classifier=lambda row: row["lane"]
    )
    window["gap_no_io_percent"] = window.pop("gap_no_io_percent_lower_bound")
    window["non_io_window_ms"] = window.pop("non_io_window_upper_ms")
    for name in (
        "gap_no_io_percent_upper_bound",
        "non_io_window_lower_ms",
        "io_or_fused_only_ms",
        "conservative_gate_pass",
    ):
        window.pop(name)
    window["gap_below_threshold"] = (
        window["gate_certifiable"]
        and window["gap_no_io_percent"] is not None
        and window["gap_no_io_percent"] < window["threshold_percent"]
    )
    window["fused_work_policy"] = (
        "Entire Compute + IO interval is productive and retained in the ratio denominator; "
        "only standalone IO outside Compute and Compute + IO is excluded"
    )
    window["boundary"] = (
        "Earlier of L0 first compute kernel start and L0 history H2D start (when present) "
        "to L2 last compute kernel end"
        if dense_extend
        else "L0 first compute kernel start to L2 last compute kernel end"
    )
    window["start_basis"] = (
        "L0 history H2D" if start < endpoints[0]["compute_start_ns"] else "L0 first compute"
    )
    window["l0_compute_start_ns"] = endpoints[0]["compute_start_ns"]
    window["l0_history_h2d_start_ns"] = l0_history_h2d_start
    layers = [
        {
            "layer": item["layer"],
            "start_ns": start if i == 0 else endpoints[i - 1]["compute_end_ns"],
            "end_ns": item["compute_end_ns"],
            "compute_start_ns": item["compute_start_ns"],
        }
        for i, item in enumerate(endpoints)
    ]
    return {
        "method": panel["method"],
        "phase": phase,
        "chunk": panel["chunks"][-1]["chunk"] if phase == "prefill" else None,
        "source_window": panel["window"],
        "window": window,
        "layers": layers,
        "endpoints": endpoints,
        "rows": rows,
    }


def extend_startup_panel(panel):
    """Include measured forward startup and embedding, retaining the L2 endpoint."""
    clipped = three_layer_panel(panel, "extend")
    start, end = panel["window"]["start_ns"], clipped["window"]["end_ns"]
    rows = [
        {
            **row,
            "start_ns": max(start, row["raw_start_ns"]),
            "end_ns": min(end, row["raw_end_ns"]),
            "start_ms": (max(start, row["raw_start_ns"]) - start) / 1e6,
            "end_ms": (min(end, row["raw_end_ns"]) - start) / 1e6,
        }
        for row in panel["rows"]
        if row["raw_start_ns"] < end and row["raw_end_ns"] > start
    ]
    embedding = [
        row
        for row in rows
        if row["kind"] == "kernel" and row["lane"] == "Compute" and row["stage"] == "embedding"
    ]
    if not embedding:
        raise ValueError("startup comparison requires measured embedding GPU activity")
    embedding_start = min(row["raw_start_ns"] for row in embedding)
    embedding_end = max(row["raw_end_ns"] for row in embedding)
    l0_start = clipped["endpoints"][0]["compute_start_ns"]
    if not start <= embedding_start < embedding_end <= l0_start:
        raise ValueError("embedding must follow forward entry and precede L0 computation")
    activities = [
        {
            "start": row["raw_start_ns"],
            "end": row["raw_end_ns"],
            "lane": row["lane"],
            "actual_io": row.get("actual_io") or {},
        }
        for row in rows
        if row["kind"] != "api"
    ]
    window = launch_gap.summarize_window(
        activities, start, end, lane_classifier=lambda row: row["lane"]
    )
    window["boundary"] = "Measured forward entry to L2 last compute kernel end"
    return {
        **clipped,
        "source_window": panel["window"],
        "cropped_window": clipped["window"],
        "window": window,
        "rows": rows,
        "startup": {
            "start_ns": start,
            "end_ns": embedding_start,
            "label": "Startup",
            "wall_ms": (embedding_start - start) / 1e6,
        },
        "embedding": {
            "start_ns": embedding_start,
            "end_ns": embedding_end,
            "kernel_union_ms": analyze_nsys._union_ns(
                [(row["raw_start_ns"], row["raw_end_ns"]) for row in embedding]
            )
            / 1e6,
        },
        "l0_offset_ms": (l0_start - start) / 1e6,
    }


def draw(
    prefill,
    extend,
    output,
    *,
    layout="combined",
    window_kind="full",
    io_layout="shared",
    annotations="gap",
):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    separate = layout == "separate"
    split_io = io_layout == "directions"
    selected_annotations = annotations == "idle-echo"
    startup_comparison = window_kind == "extend-startup"
    colors = {"Embedding": "#F0E442", **COLORS} if startup_comparison else COLORS
    fig, axes = plt.subplots(
        4, 1 if separate else 2, figsize=(13, 10) if separate else (18, 10), squeeze=False
    )
    figures = []
    metrics = {}
    phases = (
        [("extend", extend)] if startup_comparison else [("prefill", prefill), ("extend", extend)]
    )
    for column, (phase, panels) in enumerate(phases):
        if separate and column:
            fig, axes = plt.subplots(4, 1, figsize=(13, 10), squeeze=False)
        axis_column = 0 if separate else column
        limit = max(p["window"]["window_ms"] for p in panels) * 1.015
        for index, panel in enumerate(panels):
            axis = axes[index, axis_column]
            rows, window = panel["rows"], panel["window"]
            echo_lane = int(selected_annotations and panel["method"] == "echo")
            compute_y = (3 if split_io else 2) + echo_lane
            selected = idle_echo_annotations(panel) if selected_annotations else {}
            ranges, counts = defaultdict(list), Counter()
            for row in rows:
                if row["lane"] in {"Compute", "Compute + IO"}:
                    purpose = row.get("purpose") or timeline.computation_segments([row])[0]["label"]
                    category = (
                        "Embedding"
                        if startup_comparison and purpose == "Embedding"
                        else GROUPS[purpose]
                    )
                    fused = row["lane"] == "Compute + IO"
                    counts[category] += 1
                elif row["lane"] == "IO":
                    category, fused = io_direction(row), False
                else:
                    continue
                ranges[(category, fused)].append((row["start_ms"], row["end_ms"]))
            for (category, fused), intervals in ranges.items():
                if category.startswith("H2D:"):
                    vertical = (1.75, 0.50) if split_io else (1.02, 0.23)
                elif category.startswith("D2H:"):
                    vertical = (0.75, 0.50) if split_io else (0.75, 0.23)
                else:
                    vertical = (compute_y - 0.25, 0.50)
                if category.startswith(("H2D:", "D2H:")):
                    vertical = (vertical[0] + echo_lane, vertical[1])
                axis.broken_barh(
                    [(a, b - a) for a, b in intervals],
                    vertical,
                    facecolors=colors[category],
                    edgecolors="#FFFFFF" if fused else "none",
                    hatch="////" if fused else None,
                    linewidth=0,
                )
            gaps = gap_intervals(rows, window)
            if selected_annotations:
                for label, intervals in selected.items():
                    axis.broken_barh(
                        [((a - window["start_ns"]) / 1e6, (b - a) / 1e6) for a, b in intervals],
                        (-0.25, 0.50) if label == "GPU idle" else (0.75, 0.50),
                        facecolors=ANNOTATION_COLORS[label],
                        edgecolors="none",
                    )
            else:
                axis.broken_barh(
                    [(a, b - a) for a, b in gaps],
                    (-0.25, 0.50),
                    facecolors=COLORS["Gap"],
                    edgecolors="none",
                )
            if phase == "extend" or window_kind == "three-layers":
                parts = panel["layers"]
                if window_kind == "full":
                    parts = [*parts, panel["shared_tail"]]
                elif startup_comparison:
                    parts = [panel["startup"], *parts]
                for part in parts:
                    left = (part["start_ns"] - window["start_ns"]) / 1e6
                    right = (part["end_ns"] - window["start_ns"]) / 1e6
                    label = f"L{part['layer']}" if "layer" in part else part.get("label", "tail")
                    axis.text((left + right) / 2, compute_y + 0.52, label, ha="center", fontsize=9)
                    axis.axvline(right, color="#66717D", ls=":", alpha=0.5, lw=0.6)
                if startup_comparison:
                    embedding = panel["embedding"]
                    midpoint = (embedding["start_ns"] + embedding["end_ns"]) / 2
                    axis.scatter(
                        [(midpoint - window["start_ns"]) / 1e6],
                        [compute_y + 0.30],
                        marker="v",
                        s=32,
                        color=colors["Embedding"],
                        edgecolors="#665A00",
                        linewidths=0.5,
                        zorder=4,
                    )
                    axis.axvline(panel["l0_offset_ms"], color="#66717D", ls=":", alpha=0.5, lw=0.6)
            else:
                chunk = panel["chunks"][-1]
                left = (chunk["start_ns"] - window["start_ns"]) / 1e6
                right = (panel["shared_tail"]["start_ns"] - window["start_ns"]) / 1e6
                axis.plot(
                    [left, left, right, right], [2.32, 2.42, 2.42, 2.32], color="#52606D", lw=0.75
                )
                axis.text(right, 2.55, f"chunk {chunk['chunk'] + 1}", fontsize=8.5, ha="right")
            if "gap_no_io_percent" in window:
                ratio = f"{window['gap_no_io_percent']:.2f}%"
            else:
                lower, upper = [
                    window[f"gap_no_io_percent_{side}_bound"] for side in ("lower", "upper")
                ]
                ratio = (
                    f"{upper:.2f}%" if abs(upper - lower) < 1e-8 else f"{lower:.2f}-{upper:.2f}%"
                )
            annotation_title = (
                f"GPU idle {window['gpu_idle_ms']:.3f} ms"
                if selected_annotations
                else f"gap {window['gap_ms']:.3f} ms ({ratio})"
            )
            axis.set_title(
                f"{panel['method']}  |  {window['window_ms']:.3f} ms  |  {annotation_title}",
                loc="left",
                fontsize=10.5,
                pad=9,
            )
            axis.set_xlim(0, limit)
            axis.set_ylim(-0.58, compute_y + 0.98)
            labels = ["Compute", "H2D", "D2H"] if split_io else ["Compute", "IO"]
            if echo_lane:
                labels.append("ECHO ops")
            labels.append("GPU idle" if selected_annotations else "Gap")
            axis.set_yticks(list(range(compute_y, -1, -1)), labels)
            axis.tick_params(axis="both", labelsize=9)
            axis.axvline(window["window_ms"], color="#66717D", ls="--", lw=0.8)
            axis.grid(axis="x", alpha=0.12)
            axis.set_axisbelow(True)
            for edge in ("left", "top", "right"):
                axis.spines[edge].set_visible(False)
            expected = sum(r["lane"] in {"Compute", "Compute + IO"} for r in rows)
            if sum(counts.values()) != expected:
                raise ValueError("Compute purpose coverage mismatch")
            metrics[f"{phase}/{panel['method']}"] = {
                "window": window,
                "layer_windows_ns": panel.get("layers", []),
                "xlim_ms": list(axis.get_xlim()),
                "gap_intervals_ms": gaps,
                "compute_categories": dict(counts),
                "compute_lane_count": 1,
                "annotation_intervals_ns": selected,
                "annotation_totals_ms": {
                    label: sum(b - a for a, b in intervals) / 1e6
                    for label, intervals in selected.items()
                },
            }
            if startup_comparison:
                metrics[f"{phase}/{panel['method']}"]["startup"] = panel["startup"]
                metrics[f"{phase}/{panel['method']}"]["embedding"] = panel["embedding"]
                metrics[f"{phase}/{panel['method']}"]["l0_offset_ms"] = panel["l0_offset_ms"]
        xlabel = (
            "Time from L0 start: first compute or earlier history H2D for dense (ms)"
            if window_kind == "three-layers" and phase == "extend"
            else "Time from L0 first compute kernel start (ms)"
            if window_kind == "three-layers"
            else "Time from forward entry (ms)"
            if startup_comparison
            else f"Time from complete {phase} start (ms)"
        )
        axes[-1, axis_column].set_xlabel(xlabel, fontsize=10)
        if separate:
            figures.append((phase, fig))
    if not separate:
        figures.append(("prefill_extend", fig))
    handles = [
        Patch(
            facecolor=color,
            label="Output / MLP"
            if window_kind in {"three-layers", "extend-startup"} and name == "Output / MLP / head"
            else name,
        )
        for name, color in colors.items()
        if not selected_annotations or name != "Gap"
    ]
    if selected_annotations:
        handles.extend(
            Patch(facecolor=color, label=name) for name, color in ANNOTATION_COLORS.items()
        )
    handles.append(
        Patch(
            facecolor=COLORS["Indexer / top-k"],
            edgecolor="white",
            hatch="////",
            label="Fused compute + IO",
        )
    )
    notes = [
        "V10: one Compute lane. Within IO, orange H2D uses the upper band and purple D2H the lower band so concurrent transfers remain visible.",
        "Red gap = full window outside compute and actual IO. Percentages exclude IO-only time; ECHO ranges are fused-IO bounds, not confidence intervals.",
        "Complete windows include startup, final norm / head, synchronization and commit. Same time scale across the four methods within each phase.",
        "Intrusive NSYS profile; displayed times are not independent benchmark times.",
    ]
    if window_kind == "three-layers":
        notes[1] = (
            "Fused compute + IO is productive, never gap, and remains in the ratio denominator. Only standalone IO-only time is excluded."
        )
        notes[2] = (
            "Window ends at L2 last compute. Starts at L0 first compute, or earlier L0 history H2D for dense extend. Prefill uses only chunk 64."
        )
    if split_io:
        notes[0] = (
            "One Compute lane, colored by phase. Orange H2D: DRAM to GPU. Purple D2H: GPU to DRAM. Transfer directions use separate lanes."
        )
    if selected_annotations:
        notes[0] = (
            "Red marks GPU idle only. ECHO prepare, finalize and hint use a separate lane; other control activities are not highlighted."
        )
        notes[1] = (
            "Compute phases use color. H2D and D2H have separate lanes. Hatched fused compute + IO is active work, never idle."
        )
    if startup_comparison:
        notes[2] = (
            "Window: forward entry to L2 last compute end. Startup ends at embedding start; later pre-layer H2D remains IO."
        )
        notes.append(
            "Yellow marker locates the short embedding kernel (about 2 microseconds); all activity widths remain to scale."
        )
    prefill_title = (
        "Prefill | last chunk 64/64, 1,024 tokens, L0-L2"
        if window_kind == "three-layers"
        else "Prefill | 65,536 tokens, all 64 chunks"
    )
    for phase, fig in figures:
        if separate:
            title = prefill_title if phase == "prefill" else "Extend | 128 tokens, all 3 layers"
            if startup_comparison:
                title = "Extend | startup + embedding + L0-L2, 128 tokens"
            fig.text(0.09, 0.960, title, fontsize=15)
        else:
            fig.text(0.075, 0.960, prefill_title, fontsize=15)
            fig.text(0.555, 0.960, "Extend | 128 tokens, all 3 layers", fontsize=15)
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.097 if separate else 0.087),
            ncol=4 if separate else 7,
            frameon=False,
            fontsize=9,
        )
        for index, note in enumerate(notes):
            fig.text(
                0.09 if separate else 0.075,
                0.083 - index * (0.016 if startup_comparison else 0.018),
                note,
                fontsize=8 if separate else 8.5,
                va="top",
            )
        fig.subplots_adjust(
            left=0.09 if separate else 0.075,
            right=0.985,
            top=0.905,
            bottom=(0.235 if selected_annotations else 0.20) if separate else 0.16,
            hspace=0.50,
            wspace=0.16,
        )
        for ext in ("png", "svg"):
            name = "extend_with_startup" if startup_comparison else phase
            fig.savefig(output / f"{name}.{ext}", dpi=180, facecolor="white")
        plt.close(fig)
    return metrics


def extract_profile_panels(profile, gate):
    """Read one accepted profile into the existing final-pair renderer format."""
    from experiments.deepseek_v32_mfu.src.operator_report import _SCOPE, read_calls
    from experiments.deepseek_v32_motivation.src.graph_attribution import (
        attribute_graph_replays,
        read_lineage,
    )

    result = read(profile / "result.json")
    if not result["accepted"] or gate["run_id"] != result["run_id"]:
        raise ValueError("timeline requires one accepted matching profile and gap audit")
    if sha(profile / "result.json") != gate["input_result_sha256"]:
        raise ValueError("gap audit does not match the profile result")
    if (
        result["num_layers"],
        result["prefix_tokens"],
        result["chunk_size"],
        result["extend_tokens"],
    ) != (3, 65536, 1024, 128):
        raise ValueError("this final timeline pair requires L0-L2, H65536, chunk1024 and A128")
    calls, _ = read_calls(profile / "operator_calls.json")
    setup = [
        profile / f"capture_{index}.sqlite"
        for index, label in enumerate(result["nsys_capture_order"], 1)
        if label == "graph_setup" or label.endswith("/extend_graph_setup")
    ]
    parents = read_lineage(setup) if setup else None
    output = {}
    for phase, key in (("prefill", "prefill_methods"), ("extend", "methods")):
        panels = []
        for expected in gate[key]:
            method = expected["method"]
            path = profile / expected["sqlite"]
            if sha(path) != expected["sqlite_sha256"]:
                raise ValueError("profile capture changed after gap audit")
            scopes, apis, activities, _ = analyze_nsys._read_capture(
                path.resolve(), scope_pattern=_SCOPE
            )
            analyze_nsys._assign_scopes(apis, scopes)
            analyze_nsys._attribute(apis, activities)
            activities, boundary = timeline.select_forward_activities(scopes, activities)
            if boundary != expected["capture_boundary"]:
                raise ValueError("timeline measured forward boundary differs from gap audit")
            selected = [
                row
                for row in calls
                if (row["mode"], row["phase"]) == (method, f"{phase}_annotated")
            ]
            if any(row.get("full_extend_graph") for row in selected):
                from experiments.deepseek_v32_mfu.src.full_graph_profile import (
                    attribute_full_graph_replays,
                )

                attribute_full_graph_replays(
                    activities,
                    selected,
                    parents,
                    scopes=scopes,
                    apis=timeline.select_forward_apis(apis, boundary),
                )
            elif any(row.get("graph_replay") for row in selected):
                attribute_graph_replays(
                    activities, selected, parents, scopes=scopes, require_replays=True
                )
            counter_key = (
                "prefix_cache_per_layer" if phase == "prefill" else "extend_cache_per_layer"
            )
            launch_gap.annotate_actual_io(
                activities,
                result["measurements"][method][counter_key],
                method,
                f"{phase}_annotated",
            )
            if launch_gap.activity_inventory(activities) != expected["activity_inventory"]:
                raise ValueError("timeline activity classification differs from accepted gap audit")
            window = expected[f"full_{phase}"]
            rows = timeline.activity_rows(activities, apis, window["start_ns"], window["end_ns"])
            panel = {
                "method": method,
                "phase": phase,
                "window": window,
                "rows": rows,
                "shared_tail": expected["shared_tail"],
            }
            if phase == "prefill":
                panel["chunks"] = [
                    row for row in expected["layer_chunk_windows"] if row["layer"] == 0
                ]
                if [row["chunk"] for row in panel["chunks"]] != list(range(64)):
                    raise ValueError("prefill capture must contain every one of the 64 chunks")
            else:
                panel["layers"] = expected["layers"]
            panels.append(panel)
        if [panel["method"] for panel in panels] != list(launch_gap.METHODS):
            raise ValueError("final timeline requires each of the four methods in canonical order")
        output[phase] = panels
    return result["run_id"], output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-dir", type=Path)
    parser.add_argument("--extend-dir", type=Path)
    parser.add_argument("--profile-run", type=Path)
    parser.add_argument("--gap-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=("combined", "separate"), default="combined")
    parser.add_argument(
        "--window", choices=("full", "three-layers", "extend-startup"), default="full"
    )
    parser.add_argument("--io-layout", choices=("shared", "directions"), default="shared")
    parser.add_argument("--annotations", choices=("gap", "idle-echo"), default="gap")
    args = parser.parse_args()
    if args.window == "extend-startup" and args.layout != "separate":
        parser.error("--window extend-startup requires --layout separate")
    if args.profile_run:
        if args.gap_audit is None or args.prefill_dir is not None or args.extend_dir is not None:
            parser.error("--profile-run requires --gap-audit and excludes legacy extraction inputs")
        profile = args.profile_run.resolve()
        input_paths = [
            profile / "result.json",
            profile / "operator_calls.json",
            args.gap_audit.resolve(),
            *sorted(profile.glob("capture_*.sqlite")),
        ]
        inputs_before_extract = {str(path): sha(path) for path in input_paths}
        profile_run_id, panels = extract_profile_panels(profile, read(args.gap_audit))
        if inputs_before_extract != {str(path): sha(path) for path in input_paths}:
            raise ValueError("profile input changed during timeline extraction")
        prefill, extend = panels["prefill"], panels["extend"]
    else:
        if args.prefill_dir is None or args.extend_dir is None or args.gap_audit is not None:
            parser.error("provide --profile-run/--gap-audit or both --prefill-dir/--extend-dir")
        prefill_path = args.prefill_dir / "prefill_rows.json"
        extend_path = args.extend_dir / "complete_extend.json"
        prefill_receipt = read(args.prefill_dir / "prefill_receipt.json")
        extend_receipt = read(args.extend_dir / "binding.json")
        if prefill_receipt["profile_run_id"] != RUN or not extend_receipt["passed"]:
            raise ValueError("Expected accepted legacy extraction inputs")
        if sha(prefill_path) != prefill_receipt["artifacts_sha256"][prefill_path.name]:
            raise ValueError("Prefill extraction changed")
        if sha(extend_path) != extend_receipt["artifacts_sha256"][str(extend_path.resolve())]:
            raise ValueError("Extend extraction changed")
        input_paths = [
            prefill_path,
            extend_path,
            args.prefill_dir / "prefill_receipt.json",
            args.extend_dir / "binding.json",
        ]
        profile_run_id = RUN
        prefill, extend = read(prefill_path), read(extend_path)
    paths = [
        *input_paths,
        Path(__file__),
        Path(timeline.__file__),
        Path(redraw_gap_timeline.__file__),
        Path(launch_gap.__file__),
        Path(analyze_nsys.__file__),
        Path(__file__).with_name("full_graph_profile.py"),
    ]
    before = {str(p.resolve()): sha(p) for p in paths}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.window in {"three-layers", "extend-startup"}:
        if args.window == "three-layers":
            prefill = [three_layer_panel(panel, "prefill") for panel in prefill]
            extend = [three_layer_panel(panel, "extend") for panel in extend]
            panels = {"prefill": prefill, "extend": extend}
        else:
            extend = [extend_startup_panel(panel) for panel in extend]
            panels = {"extend": extend}
        (args.output_dir / "window_rows.json").write_text(json.dumps(panels, indent=2) + "\n")
        summaries = [
            {"phase": phase, "method": panel["method"], **panel["window"]}
            for phase, phase_panels in panels.items()
            for panel in phase_panels
        ]
        with (args.output_dir / "windows.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    metrics = draw(
        prefill,
        extend,
        args.output_dir,
        layout=args.layout,
        window_kind=args.window,
        io_layout=args.io_layout,
        annotations=args.annotations,
    )
    if args.annotations == "idle-echo":
        with (args.output_dir / "annotations.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["phase", "method", *ANNOTATION_COLORS])
            writer.writeheader()
            for key, item in metrics.items():
                phase, method = key.split("/")
                writer.writerow(
                    {
                        "phase": phase,
                        "method": method,
                        **{
                            label: item["annotation_totals_ms"].get(label, 0)
                            for label in ANNOTATION_COLORS
                        },
                    }
                )
    shutil.copy2(__file__, args.output_dir / "renderer.py")
    for source in paths:
        if source.suffix == ".py":
            target = args.output_dir / "renderer_sources" / source.relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if before != {str(p.resolve()): sha(p) for p in paths}:
        raise ValueError("Input or renderer changed")
    receipt = {
        "schema": "compact-prefill-extend-v2",
        "profile_run_id": profile_run_id,
        "layout": args.layout,
        "window_kind": args.window,
        "io_layout": args.io_layout,
        "annotations": args.annotations,
        "annotation_unit": "ms",
        "annotation_colors": ANNOTATION_COLORS if args.annotations == "idle-echo" else {},
        "io_direction_colors": {k: v for k, v in COLORS.items() if k.startswith(("H2D:", "D2H:"))},
        "prefill_annotation": (
            None
            if args.window == "extend-startup"
            else "Only final chunk 64 (1024 tokens), L0 first compute start through L2 last compute end"
            if args.window == "three-layers"
            else "Only final chunk 64 (1024 tokens), excluding shared tail; full prefill window retained"
        ),
        "inputs_and_sources_sha256": before,
        "metrics": metrics,
        "measurement": (
            "CPU reanalysis of accepted profile activities in the user-selected window; no new GPU measurement"
            if args.window in {"three-layers", "extend-startup"}
            else "Presentation only; no GPU measurement or optimization"
        ),
        "artifacts_sha256": {
            str(p.relative_to(args.output_dir)): sha(p)
            for p in sorted(args.output_dir.rglob("*"))
            if p.is_file()
        },
    }
    (args.output_dir / "compact_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(args.output_dir / "compact_receipt.json")


if __name__ == "__main__":
    main()
