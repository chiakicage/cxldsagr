"""Draw the C10 compute/fetch dependencies without using measured durations."""

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

INK = "#243447"
MUTED = "#596978"
COLORS = {
    "compute": ("#E6F1F8", "#0072B2"),
    "fetch": ("#FFF0DD", "#B56700"),
    "wait": ("#F0F2F4", "#7B8792"),
}
SOURCES = (
    "models/deepseek_v32/attention.py",
    "models/deepseek_v32/execution/pipeline.py",
    "models/deepseek_v32/execution/adapter.py",
    "models/deepseek_v32/cache/prefetch.py",
    "cache/sparse_token_cache.py",
    "operators/deepseek_v32/indexer/csrc/echo_logits.cuh",
    "operators/deepseek_v32/indexer/csrc/echo_cache.cuh",
)


def box(axis, x, y, width, text, kind="compute", height=6, size=12):
    fill, edge = COLORS[kind]
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.06,rounding_size=0.7",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.2,
            zorder=3,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        linespacing=1.15,
        zorder=4,
    )


def arrow(axis, start, end, color=MUTED):
    axis.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.15, color=color, zorder=2
        )
    )


def row_title(axis, y, name, note):
    axis.text(3, y, name, fontsize=15, weight="bold", color=INK, va="center")
    axis.text(3, y - 5, note, fontsize=11, color=MUTED, va="center", linespacing=1.3)


def draw(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "c10-compute-fetch",
            "pdf.fonttype": 42,
        }
    ):
        figure, axis = plt.subplots(figsize=(13.2, 8.6))
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        axis.set(xlim=(0, 132), ylim=(0, 86))
        axis.axis("off")
        axis.text(
            3,
            82,
            "Compute and fetch in sparse attention",
            fontsize=21,
            color=INK,
            weight="bold",
            va="center",
        )
        axis.text(
            129, 82, "SCHEMATIC · NOT TO SCALE", ha="right", va="center", fontsize=11, color=MUTED
        )
        axis.text(
            3,
            76.5,
            "DeepSeek C10 candidate execution · history KV misses · H ≤ P",
            fontsize=12,
            color=MUTED,
            va="center",
        )
        for x, kind, label in (
            (88, "compute", "Compute"),
            (103, "fetch", "Fetch"),
            (116, "wait", "GPU wait"),
        ):
            box(axis, x, 75, 2, "", kind, height=2.8)
            axis.text(x + 3, 76.4, label, va="center", fontsize=11, color=INK)
        for y in (71, 48, 28):
            axis.plot([3, 129], [y, y], color="#DCE3E8", linewidth=0.8)

        row_title(axis, 64, "ECHO", "In-layer fusion\nSM compute + host reads")
        box(axis, 23, 60, 13, "Projection")
        axis.add_patch(
            FancyBboxPatch(
                (39, 53),
                28,
                14,
                boxstyle="round,pad=0.1,rounding_size=0.9",
                facecolor="none",
                edgecolor=COLORS["compute"][1],
                linewidth=1.2,
                linestyle=(0, (4, 3)),
            )
        )
        axis.text(
            53,
            68.3,
            "ONE FUSED KERNEL",
            ha="center",
            fontsize=10.5,
            color=COLORS["compute"][1],
            weight="bold",
        )
        box(axis, 40, 60, 26, "Indexer scores")
        box(axis, 44, 54, 21, "Hint-based KV prefetch", "fetch", height=5, size=11)
        arrow(axis, (42, 60), (45, 59), COLORS["fetch"][1])
        box(axis, 71, 60, 11, "Exact\ntop-k")
        box(axis, 86, 60, 17, "Fetch remaining\nselected misses", "fetch", size=11.5)
        box(axis, 107, 60, 21, "Sparse attention\n+ output / MLP")
        for start, end in ((36, 39), (67, 71), (82, 86), (103, 107)):
            arrow(axis, (start, 63), (end, 63))
        axis.text(69, 68.3, "finalize", ha="center", fontsize=9, color=MUTED)
        axis.text(
            73,
            54.5,
            "Exact selection remains mandatory;\nfetch any remaining selected misses.",
            fontsize=11,
            color=MUTED,
        )

        row_title(axis, 42, "serial_sparse", "Select first\nthen fetch exact misses")
        box(axis, 23, 37, 13, "Projection")
        box(axis, 40, 37, 27, "Indexer scores")
        box(axis, 71, 37, 11, "Exact\ntop-k")
        box(axis, 86, 37, 17, "Fetch selected\nhistory misses", "fetch", size=11.5)
        box(axis, 107, 37, 21, "Sparse attention\n+ output / MLP")
        for start, end in ((36, 40), (67, 71), (82, 86), (103, 107)):
            arrow(axis, (start, 40), (end, 40))
        axis.text(
            40,
            32,
            "SM-based sparse fetch: attention starts after the selected KV is ready.",
            fontsize=11,
            color=MUTED,
        )

        row_title(axis, 21, "dense_prefetch", "Cross-layer DMA")
        axis.text(
            3,
            10.5,
            "Independent\ncopy stream",
            fontsize=11,
            color=COLORS["fetch"][1],
            va="center",
            linespacing=1.2,
        )
        box(axis, 23, 16, 12, "Wait for\nKV of layer i", "wait", height=7, size=11)
        box(
            axis,
            40,
            16,
            42,
            "Layer i compute\nProjection / indexer / exact top-k\nSparse attention / output / MLP",
            height=7,
            size=11.5,
        )
        box(axis, 95, 16, 12, "Wait for\nKV of i+1", "wait", height=7, size=11)
        box(axis, 111, 16, 17, "Layer i+1\ncompute", height=7)
        for start, end in ((35, 40), (82, 95), (107, 111)):
            arrow(axis, (start, 19.5), (end, 19.5))
        box(
            axis,
            40,
            8,
            61,
            "Full history KV for layer i+1 · DRAM → HBM (DMA)",
            "fetch",
            height=5.5,
            size=12,
        )
        arrow(axis, (38, 19.5), (40, 13.5), COLORS["fetch"][1])
        axis.text(36.5, 14, "launch", fontsize=9, color=COLORS["fetch"][1], ha="right")
        arrow(axis, (101, 13.5), (101, 16), COLORS["fetch"][1])
        axis.text(
            61,
            24.8,
            "Compute i can overlap fetch for i+1",
            ha="center",
            fontsize=11,
            color=COLORS["compute"][1],
            weight="bold",
        )
        axis.text(111, 10.5, "Same pipeline\nrepeats", fontsize=11, color=MUTED, va="center")

        axis.text(
            3,
            4.3,
            "History KV moves from CPU DRAM to HBM; candidate KV stays on the GPU.",
            fontsize=11,
            color=INK,
        )
        axis.text(
            3,
            1.5,
            "Widths are not timings. Resident history can bypass fetch. Cache bookkeeping and pipeline startup/drain are simplified.",
            fontsize=10.5,
            color=MUTED,
        )
        for suffix in ("png", "svg", "pdf"):
            metadata = {"Creator": "draw_compute_fetch.py"}
            if suffix == "svg":
                metadata["Date"] = None
            elif suffix == "pdf":
                metadata.update(CreationDate=None, ModDate=None)
            figure.savefig(output / f"compute_fetch_schematic.{suffix}", dpi=200, metadata=metadata)
        plt.close(figure)

    root = Path(__file__).resolve().parents[3]
    sources = (*SOURCES, str(Path(__file__).resolve().relative_to(root)))
    snapshots = {}
    for relative in sources:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
        snapshots[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    record = {
        "schema": "deepseek-c10-compute-fetch-schematic-v1",
        "run_id": output.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "qualitative implementation schematic",
        "schemes": ["echo", "serial_sparse", "dense_prefetch"],
        "source_sha256": snapshots,
        "artifacts_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.glob("compute_fetch_schematic.*"))
        },
        "boundary": "C10 fixed-P/NH candidate path with history misses and H<=P; no timing data, new measurements or performance claims. ECHO and serial fetch use SM host reads; only dense has an independent DMA stream. Dense waits before each whole layer, not the separate real-three-layer late-wait graph schedule.",
    }
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    draw(args.output_dir)
    print(f"Compute/fetch schematic: {args.output_dir}")


if __name__ == "__main__":
    main()
