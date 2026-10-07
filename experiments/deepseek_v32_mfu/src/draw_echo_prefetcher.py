"""Draw the SM90 ECHO inter-query prefetcher from implementation semantics.

This is a qualitative diagram, with no measured durations or performance data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
STEM = "echo_inter_query_prefetcher"
INK = "#233443"
MUTED = "#536471"
COLORS = {
    "compute": ("#E7F2FA", "#0072B2"),
    "transfer": ("#FFF0DB", "#AE6400"),
}
STAGES = 8
SOURCES = (
    "operators/deepseek_v32/indexer/echo.py",
    "operators/deepseek_v32/indexer/csrc/echo_indexer.cu",
    "operators/deepseek_v32/indexer/csrc/echo_logits.cuh",
    "operators/deepseek_v32/indexer/csrc/echo_cache.cuh",
)


def label(axis, x, y, text, *, size=14, color=INK, weight="normal", align="left"):
    return axis.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        weight=weight,
        ha=align,
        va="center",
        linespacing=1.18,
        zorder=5,
    )


def box(axis, x, y, width, height, text, kind="compute", *, size=14):
    fill, edge = COLORS[kind]
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.7",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.3,
            zorder=3,
        )
    )
    label(axis, x + width / 2, y + height / 2, text, size=size, align="center")


def arrow(axis, start, end, *, color=MUTED, style="-", connection="arc3"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            connectionstyle=connection,
            mutation_scale=13,
            linewidth=1.25,
            linestyle=style,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        )
    )


def render(output):
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "deepseek-echo-inter-query-v2",
            "pdf.fonttype": 42,
        }
    ):
        figure, axis = plt.subplots(figsize=(12, 3.5))
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        axis.set(xlim=(0, 120), ylim=(0, 35))
        axis.axis("off")

        label(axis, 4, 31, "ECHO inter-query pipeline", size=22, weight="bold")
        label(axis, 116, 31, "8 SCHEMATIC STAGES", size=14, align="right", color=MUTED)
        label(axis, 4, 20, "Indexer", size=17, weight="bold")
        label(axis, 4, 10, "Fetch", size=17, weight="bold")

        start_x, stride, width = 24, 10, 9.2
        overlap_left = start_x + stride
        overlap_right = start_x + (STAGES - 1) * stride + width
        axis.axvspan(
            overlap_left,
            overlap_right,
            ymin=6 / 35,
            ymax=24 / 35,
            facecolor="#F2F5F7",
            zorder=0,
        )
        label(
            axis,
            (overlap_left + overlap_right) / 2,
            26,
            "Indexer / fetch overlap",
            size=14,
            color=MUTED,
            align="center",
        )
        for stage in range(STAGES):
            x = start_x + stage * stride
            name = f"S{stage + 1}"
            box(axis, x, 17, width, 6, name, "compute", size=17)
            box(axis, x + stride, 7, width, 6, name, "transfer", size=17)
            arrow(
                axis,
                (x + width / 2, 17),
                (x + stride + width / 2, 13),
            )

        arrow(axis, (start_x, 3.8), (115, 3.8))
        label(axis, 69.5, 1.4, "Conceptual progression", size=13.5, color=MUTED, align="center")
        for suffix in ("svg", "pdf", "png"):
            metadata = {"Creator": "draw_echo_prefetcher.py"}
            if suffix == "svg":
                metadata["Date"] = None
            elif suffix == "pdf":
                metadata.update(CreationDate=None, ModDate=None)
            figure.savefig(output / f"{STEM}.{suffix}", dpi=180, metadata=metadata)
        plt.close(figure)


def draw(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    render(output)
    snapshots = {}
    for relative in (*SOURCES, str(Path(__file__).resolve().relative_to(ROOT))):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        snapshots[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    record = {
        "schema": "deepseek-echo-inter-query-schematic-v2",
        "schematic_stages": STAGES,
        "run_id": output.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "qualitative implementation schematic; no GPU measurement",
        "source_sha256": snapshots,
        "artifacts_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.glob(f"{STEM}.*"))
        },
        "boundary": (
            "Qualitative ECHO inter-query pipeline only. S1 through S8 label eight "
            "schematic query work stages on both lanes; fetch follows indexer by one "
            "stage, so Fetch(Si) overlaps Indexer(Si+1). The ninth horizontal position "
            "is the final fetch, not a ninth stage. Stage count, equal widths and spacing "
            "are illustrative, not native buffer counts or measured timings. "
            "Prefetch selection and subsequent completion are outside this diagram."
        ),
    }
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New diagram run directory")
    args = parser.parse_args()
    draw(args.output_dir)
    print(f"ECHO inter-query prefetcher: {args.output_dir}")


if __name__ == "__main__":
    main()
