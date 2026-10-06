"""Render accepted timeline rows with an explicit compute/IO-complement lane.

This is a CPU-only presentation entry point. It accepts the saved, bound timeline
extraction from an existing profile and never runs a model or reclassifies IO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

METHODS = ("hbm", "echo", "serial_sparse", "dense_prefetch")
COLORS = {
    "Compute": "#0072B2",
    "Compute + IO": "#8E529D",
    "IO": "#D58B22",
    "GPU control": "#88909A",
    "CPU CUDA API": "#628B4D",
    "Gap": "#C43C39",
}
LANES = {"Compute": 4, "Compute + IO": 4, "IO": 3, "GPU control": 2, "CPU CUDA API": 1, "Gap": 0}
PURPOSES = {
    "Embedding": 1,
    "Projection / RoPE": 2,
    "Indexer logits": 3,
    "Exact top-k": 4,
    "Sparse MLA": 5,
    "V expand": 6,
    "Output / MLP": 7,
    "Final norm / LM head": 8,
    "Final norm": 8,
    "LM head": 8,
    "Indexer + prefetch": 9,
}
RUN = "deepseek_gap_v10_a128_minimal_20261006_01"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def gap_intervals(rows, window):
    """Recompute the complement in integer nanoseconds; check accepted totals."""
    start, end = window["start_ns"], window["end_ns"]
    intervals = sorted(
        (max(start, row["raw_start_ns"]), min(end, row["raw_end_ns"]))
        for row in rows
        if row["lane"] in {"Compute", "Compute + IO", "IO"}
        and row["raw_start_ns"] < end
        and row["raw_end_ns"] > start
    )
    cursor, gaps = start, []
    for left, right in intervals:
        if left > cursor:
            gaps.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        gaps.append((cursor, end))
    actual = sum(right - left for left, right in gaps) / 1e6
    if abs(actual - window["gap_ms"]) > 1e-9:
        raise ValueError(f"Gap mismatch: {actual} != {window['gap_ms']}")
    return [((left - start) / 1e6, (right - start) / 1e6) for left, right in gaps]


def annotate(axis, segments, xmax, *, full):
    spacing = xmax * (0.015 if full else 0.095)
    centers = [(s["start_ms"] + s["end_ms"]) / 2 for s in segments]
    last = -spacing
    for index, (segment, center) in enumerate(zip(segments, centers, strict=True)):
        label = str(PURPOSES[segment["label"]]) if full else segment["label"]
        upper = xmax - spacing * (len(segments) - index - 0.5)
        text_x = min(upper, max(center, last + spacing, spacing / 2))
        last = text_x
        axis.annotate(
            label,
            xy=(center, 4.30),
            xytext=(text_x, 4.65 + (index % 2) * 0.43),
            fontsize=9 if full else 8.5,
            ha="center",
            va="bottom",
            arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "#73808C"},
            color="#263442",
        )


def draw(panels, output, name, *, phase, layer=None):
    full = layer is None
    xmax = (
        max(
            max(p["window"]["window_ms"], max((b for _, b in p.get("prefetch", [])), default=0))
            for p in panels
        )
        * 1.012
    )
    fig, axes = plt.subplots(4, 1, figsize=(17, 12.5))
    metrics = []
    for axis, panel in zip(axes, panels, strict=True):
        window, rows = panel["window"], panel["rows"]
        gaps = gap_intervals(rows, window)
        for lane, y in LANES.items():
            ranges = (
                gaps
                if lane == "Gap"
                else [(row["start_ms"], row["end_ms"]) for row in rows if row["lane"] == lane]
            )
            axis.broken_barh(
                [(a, b - a) for a, b in ranges],
                (y - 0.26, 0.52),
                facecolors=COLORS[lane],
                edgecolors="#4E3158" if lane == "Compute + IO" else "none",
                linewidth=0.25,
                hatch="///" if lane == "Compute + IO" else None,
            )
        # Preserve the complete next-layer DMA even when it crosses the window.
        if panel.get("prefetch"):
            axis.broken_barh(
                [(a, b - a) for a, b in panel["prefetch"]],
                (2.73, 0.54),
                facecolors="none",
                edgecolors="#815416",
                linewidth=0.8,
            )
        axis.axvline(window["window_ms"], color="#53606D", ls="--", lw=0.8)
        if full:
            for part in [*panel["layers"], panel["shared_tail"]]:
                left = (part["start_ns"] - window["start_ns"]) / 1e6
                right = (part["end_ns"] - window["start_ns"]) / 1e6
                axis.axvline(right, color="#6E7780", ls=":", lw=0.7)
                tag = f"L{part['layer']}" if "layer" in part else "shared tail"
                axis.text((left + right) / 2, -0.65, tag, fontsize=8, ha="center", va="top")
        lower, upper = (window[f"gap_no_io_percent_{side}_bound"] for side in ("lower", "upper"))
        ratio = f"{upper:.2f}%" if abs(upper - lower) < 1e-8 else f"{lower:.2f}-{upper:.2f}%"
        suffix = ""
        if not full and panel["method"] == "dense_prefetch":
            if layer == 2:
                suffix = " | no following-layer prefetch"
            elif panel.get("prefetch"):
                suffix = f" | L{layer + 1} prefetch outlined"
            else:
                suffix = f" | no L{layer + 1} H2D transfer"
        axis.set_title(
            f"{panel['method']}   |   profile window {window['window_ms']:.3f} ms"
            f"   |   gap {window['gap_ms']:.3f} ms   |   gap / non-IO window {ratio}{suffix}",
            loc="left",
            fontsize=10.5,
            pad=8,
        )
        axis.set_xlim(0, xmax)
        axis.set_ylim(-1.0 if full else -0.65, 5.65)
        axis.set_yticks([4, 3, 2, 1, 0], ["Compute", "IO", "GPU control", "CPU CUDA API", "Gap"])
        axis.tick_params(axis="both", labelsize=9)
        axis.grid(axis="x", alpha=0.13)
        axis.set_axisbelow(True)
        for edge in ("top", "right", "left"):
            axis.spines[edge].set_visible(False)
        annotate(axis, panel["segments"], xmax, full=full)
        count = sum(row["lane"] in {"Compute", "Compute + IO"} for row in rows)
        if count != sum(segment["kernel_count"] for segment in panel["segments"]):
            raise ValueError("Compute annotation coverage mismatch")
        metrics.append(
            {
                "method": panel["method"],
                "gap_ms": window["gap_ms"],
                "gap_intervals_ms": gaps,
                "compute_kernels_labeled": count,
                "xlim_ms": list(axis.get_xlim()),
            }
        )
    scope = (
        "Complete extend: startup + L0-L2 + synchronization / commit"
        if full
        else (
            f"Extend L{layer}: complete 128-token query batch"
            if phase == "extend"
            else f"Prefill L{layer}: last chunk 64/64, tokens 64,512-65,535 (Q=1,024)"
        )
    )
    fig.suptitle(f"DeepSeek V3.2 V10  |  {scope}", x=0.09, ha="left", fontsize=14)
    axes[-1].set_xlabel(
        "Time from complete forward start (ms); common scale across methods"
        if full
        else "Time from layer window start (ms); common scale across methods",
        fontsize=10,
    )
    handles = [
        Patch(
            facecolor=color,
            label=lane,
            hatch="///" if lane == "Compute + IO" else None,
            edgecolor="#4E3158" if lane == "Compute + IO" else "none",
        )
        for lane, color in COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.108),
        ncol=6,
        frameon=False,
        fontsize=9,
    )
    notes = [
        "Gap = window outside the union of compute and actual IO. Metadata, exposed D2D/layout, empty gathers and idle count as gap.",
        "Gap ratios exclude IO-only time; indivisible fused compute/IO yields bounds. Dashed line: measured window end. Kernel bars retain internal gaps.",
        "One intrusive NSYS profile per method/phase; profile time is not independent benchmark time. No profiler-cost subtraction.",
    ]
    if full:
        notes = [
            "1 Embedding   2 Projection / RoPE   3 Indexer logits   4 Exact top-k   5 Sparse MLA   6 V expand",
            "7 Output projection / MLP   8 Final norm / LM head   9 Fused indexer + prefetch. Shared tail includes final norm / head + sync / commit.",
            *notes,
        ]
    else:
        notes += [
            "L0 extend includes startup. Layer views end at layer compute completion; final norm / head and shared tail appear in the complete view."
            if phase == "extend"
            else "This is a last-chunk detail, not full prefill. Full-prefill acceptance uses all 64 chunks plus startup and shared tail."
        ]
    for index, note in enumerate(notes):
        fig.text(0.09, 0.101 - index * 0.015, note, fontsize=8.3, va="top")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.925, bottom=0.19, hspace=0.48)
    for extension in ("svg", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=160, facecolor="white")
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data_dir.resolve(), args.output_dir.resolve()
    binding = read(data / "binding.json")
    if not binding["passed"] or read(data / "source.json")["run_id"] != RUN:
        raise ValueError("Requires an accepted V10 timeline extraction")
    paths = [
        data / "complete_extend.json",
        data / "computation_segments.json",
        data / "source.json",
    ]
    paths += [
        data / f"layer_{layer}/{phase}_{kind}.json"
        for layer in range(3)
        for phase in ("extend", "prefill")
        for kind in ("summary", "activities")
    ]
    inputs = {str(path): sha(path) for path in paths}
    for path, digest in inputs.items():
        if binding["artifacts_sha256"][path] != digest:
            raise ValueError(f"Accepted input changed: {path}")
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    panels = read(data / "complete_extend.json")
    for panel in panels:
        panel["segments"] = panel["computation_segments"]
    metrics = {"complete_extend": draw(panels, output, "complete_extend", phase="extend")}
    segments = read(data / "computation_segments.json")
    for phase in ("extend", "prefill"):
        for layer in range(3):
            summaries = read(data / f"layer_{layer}/{phase}_summary.json")
            rows = read(data / f"layer_{layer}/{phase}_activities.json")
            key = f"layer_{layer}/timeline_{phase}"
            panels = [
                {
                    "method": s["method"],
                    "window": s,
                    "rows": [r for r in rows if r["method"] == s["method"]],
                    "segments": segments[key][s["method"]],
                    "prefetch": s["next_layer_prefetch_intervals_ms"],
                }
                for s in summaries
            ]
            name = f"{phase}_layer_{layer}"
            metrics[name] = draw(panels, output, name, phase=phase, layer=layer)
    if inputs != {str(path): sha(path) for path in paths}:
        raise ValueError("Input changed during rendering")
    receipt = {
        "schema": "v10-gap-lane-redraw-v1",
        "profile_run_id": RUN,
        "inputs_sha256": inputs,
        "input_binding_sha256": sha(data / "binding.json"),
        "renderer_sha256": sha(__file__),
        "metrics": metrics,
        "measurement": "CPU-only redraw of accepted V10 data; no new measurement",
        "artifacts_sha256": {p.name: sha(p) for p in sorted(output.iterdir())},
    }
    (output / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output / "render_receipt.json")


if __name__ == "__main__":
    main()
