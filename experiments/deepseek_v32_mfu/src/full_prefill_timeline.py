"""Draw all 64 prefill chunks from the accepted V10 profile, without GPU work."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from experiments.deepseek_v32_mfu.src import timeline
from experiments.deepseek_v32_mfu.src.analyze_nsys import _assign_scopes, _attribute, _read_capture
from experiments.deepseek_v32_mfu.src.launch_gap import annotate_actual_io
from experiments.deepseek_v32_mfu.src.operator_report import _SCOPE, read_calls
from experiments.deepseek_v32_mfu.src.redraw_gap_timeline import (
    COLORS,
    RUN,
    gap_intervals,
    read,
    sha,
)
from experiments.deepseek_v32_motivation.src.graph_attribution import (
    attribute_graph_replays,
    read_lineage,
)

PURPOSES = [
    "Embedding",
    "Projection / RoPE",
    "Indexer logits",
    "Exact top-k",
    "Sparse MLA",
    "V expand",
    "Output / MLP",
    "Final norm / LM head",
]
DISPLAY = [*PURPOSES, "IO", "GPU control", "CPU CUDA API", "Gap"]


def extract(profile, gate):
    calls, _ = read_calls(profile / "operator_calls.json")
    parents = read_lineage([profile / "capture_1.sqlite"])
    result = read(profile / "result.json")
    panels = []
    for expected in gate["prefill_methods"]:
        method = expected["method"]
        print(f"Read existing prefill capture: {method}", flush=True)
        scopes, apis, activities, _ = _read_capture(
            (profile / expected["sqlite"]).resolve(), scope_pattern=_SCOPE
        )
        _assign_scopes(apis, scopes)
        _attribute(apis, activities)
        activities, boundary = timeline.select_forward_activities(scopes, activities)
        if boundary != expected["capture_boundary"]:
            raise ValueError("Measured forward boundary changed")
        selected = [r for r in calls if (r["mode"], r["phase"]) == (method, "prefill_annotated")]
        attribute_graph_replays(activities, selected, parents, scopes=scopes, require_replays=True)
        annotate_actual_io(
            activities,
            result["measurements"][method]["prefix_cache_per_layer"],
            method,
            "prefill_annotated",
        )
        window = expected["full_prefill"]
        start, end = window["start_ns"], window["end_ns"]
        rows = []
        for activity in [*activities, *apis]:
            if activity["start"] >= end or activity["end"] <= start:
                continue
            scope = activity.get("scope") or {}
            kind = activity.get("kind", "api")
            lane = "CPU CUDA API" if kind == "api" else timeline.activity_lane(activity)
            row = {
                "lane": lane,
                "kind": kind,
                "name": activity["name"],
                "scope_label": scope.get("label"),
                "stage": activity.get("graph_stage", scope.get("stage")),
                "layer": scope.get("layer"),
                "actual_io": activity.get("actual_io"),
                "correlation": activity.get("correlation"),
                "stream": activity.get("stream_id"),
                "raw_start_ns": activity["start"],
                "raw_end_ns": activity["end"],
                "start_ms": (max(start, activity["start"]) - start) / 1e6,
                "end_ms": (min(end, activity["end"]) - start) / 1e6,
            }
            if lane in {"Compute", "Compute + IO"}:
                label = timeline.computation_segments([row])[0]["label"]
                row["purpose"] = (
                    "Final norm / LM head" if label in {"Final norm", "LM head"} else label
                )
                if row["purpose"] not in PURPOSES:
                    raise ValueError(f"Unexpected prefill compute purpose: {row['purpose']}")
            rows.append(row)
        inventory = Counter(
            (r["lane"], r["kind"], r["stage"], r["name"]) for r in rows if r["kind"] != "api"
        )
        accepted = Counter(
            {
                (r["lane"], r["kind"], r["stage"], r["name"]): r["count"]
                for r in expected["activity_inventory"]
            }
        )
        if inventory != accepted:
            raise ValueError("Activity inventory differs from accepted V10 audit")
        if any(r["lane"] == "Compute + IO" for r in rows):
            raise ValueError("This V10 prefill has no nonzero fused IO")
        chunks = [r for r in expected["layer_chunk_windows"] if r["layer"] == 0]
        if [r["chunk"] for r in chunks] != list(range(64)):
            raise ValueError("Expected all 64 chunks in chronological order")
        panels.append(
            {
                "method": method,
                "window": window,
                "rows": rows,
                "gap_intervals_ms": gap_intervals(rows, window),
                "chunks": chunks,
                "shared_tail": expected["shared_tail"],
                "absolute_gap_over_hbm": expected["absolute_gap_over_hbm"],
            }
        )
    return panels


def draw(panels, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(4, 1, figsize=(18, 16))
    xmax = max(p["window"]["window_ms"] for p in panels) * 1.012
    metrics = []
    for axis, panel in zip(axes, panels, strict=True):
        window = panel["window"]
        lanes = defaultdict(list)
        for row in panel["rows"]:
            lanes[row.get("purpose", row["lane"])].append((row["start_ms"], row["end_ms"]))
        lanes["Gap"] = panel["gap_intervals_ms"]
        for index, label in enumerate(DISPLAY):
            color = COLORS["Compute" if label in PURPOSES else label]
            axis.broken_barh(
                [(a, b - a) for a, b in lanes[label]],
                (len(DISPLAY) - index - 1 - 0.28, 0.56),
                facecolors=color,
                edgecolors="none",
            )
        for chunk in panel["chunks"][::8]:
            left = (chunk["start_ns"] - window["start_ns"]) / 1e6
            axis.axvline(left, color="#89929B", alpha=0.45, ls=":", lw=0.6)
            axis.text(
                left + 2,
                len(DISPLAY) - 0.12,
                f"chunk {chunk['chunk'] + 1}",
                fontsize=8.5,
                color="#52606D",
                va="bottom",
            )
        axis.axvline(window["window_ms"], color="#52606D", ls="--", lw=0.8)
        axis.set_xlim(0, xmax)
        axis.set_ylim(-0.7, len(DISPLAY) + 0.55)
        axis.set_yticks(list(reversed(range(len(DISPLAY)))), DISPLAY)
        axis.tick_params(axis="both", labelsize=9)
        axis.grid(axis="x", alpha=0.12)
        axis.set_axisbelow(True)
        for edge in ("left", "top", "right"):
            axis.spines[edge].set_visible(False)
        axis.set_title(
            f"{panel['method']}  |  profile window {window['window_ms']:.3f} ms"
            f"  |  gap {window['gap_ms']:.3f} ms"
            f"  |  gap / non-IO window {window['gap_no_io_percent_upper_bound']:.2f}%"
            f"  |  absolute gap / HBM {panel['absolute_gap_over_hbm']:.3f}x",
            loc="left",
            fontsize=11,
            pad=14,
        )
        metrics.append(
            {
                "method": panel["method"],
                "window": window,
                "xlim_ms": list(axis.get_xlim()),
                "activity_count": sum(r["kind"] != "api" for r in panel["rows"]),
                "compute_kernel_purposes": dict(
                    Counter(r["purpose"] for r in panel["rows"] if "purpose" in r)
                ),
                "gap_intervals_ms": panel["gap_intervals_ms"],
            }
        )
    fig.suptitle(
        "DeepSeek V3.2 V10 | Complete prefill: 65,536 tokens, 64 chunks, layers 0-2",
        x=0.135,
        ha="left",
        fontsize=15,
        y=0.978,
    )
    axes[-1].set_xlabel(
        "Time from complete prefill forward start (ms); common scale across methods", fontsize=11
    )
    fig.legend(
        handles=[
            Patch(facecolor=COLORS[x], label=x)
            for x in ["Compute", "IO", "GPU control", "CPU CUDA API", "Gap"]
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.082),
        ncol=5,
        frameon=False,
        fontsize=10,
    )
    notes = [
        "All 64 chunks, initial startup, final norm / LM head, synchronization and commit are included. Each blue lane names its computation purpose.",
        "Every chunk executes L0, L1 and L2; tick annotations show every eighth chunk start. Fine activities remain in the SVG; see layer views for local detail.",
        "Red gap = full window outside the union of computation and actual IO. Metadata, exposed D2D/layout, empty gathers and idle count as gap.",
        "Ratios exclude IO-only time. Counter-proven zero-IO fused indexer is compute. CPU CUDA API includes waiting; its duration is not itself gap.",
        "Existing single intrusive NSYS profile per method; these times differ from independent benchmark. No profiler-cost subtraction or new GPU measurement.",
    ]
    for index, note in enumerate(notes):
        fig.text(0.135, 0.076 - index * 0.012, note, fontsize=8.5, va="top")
    fig.subplots_adjust(left=0.135, right=0.985, top=0.93, bottom=0.135, hspace=0.42)
    for ext in ("png", "svg"):
        fig.savefig(output / f"complete_prefill.{ext}", dpi=160, facecolor="white")
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--accepted-timeline-dir", type=Path, required=True)
    parser.add_argument("--gap-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = read(args.accepted_timeline_dir / "source.json")
    gate = read(args.gap_audit)
    if gate["run_id"] != RUN or source["run_id"] != RUN:
        raise ValueError("Expected the accepted V10 profile identity")
    sources = {path: sha(path) for path in source["retained_before_after_sha256"]}
    if sources != source["retained_before_after_sha256"]:
        raise ValueError("Accepted profile or classifier sources changed")
    if sources[str(args.gap_audit.resolve())] != sha(args.gap_audit):
        raise ValueError("Gap audit identity mismatch")
    if sha(args.profile_run / "result.json") != gate["input_result_sha256"]:
        raise ValueError("Profile result identity mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    panels = extract(args.profile_run, gate)
    metrics = draw(panels, args.output_dir)
    if "torch" in sys.modules:
        raise RuntimeError("Presentation must not import Torch")
    if sources != {path: sha(path) for path in sources}:
        raise ValueError("Input changed during rendering")
    (args.output_dir / "prefill_rows.json").write_text(json.dumps(panels) + "\n")
    receipt = {
        "schema": "v10-full-prefill-timeline-v1",
        "profile_run_id": RUN,
        "renderer_sha256": sha(__file__),
        "inputs_sha256": sources,
        "metrics": metrics,
        "measurement": "Existing profile only; no GPU work or new measurement",
        "artifacts_sha256": {p.name: sha(p) for p in args.output_dir.iterdir()},
    }
    (args.output_dir / "prefill_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(args.output_dir / "prefill_receipt.json")


if __name__ == "__main__":
    main()
