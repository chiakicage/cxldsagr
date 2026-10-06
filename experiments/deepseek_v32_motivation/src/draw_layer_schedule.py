"""Draw the fixed-P/NH candidate dependency schedule, without timing data.

Source boundaries:
* models/deepseek_v32/attention.py: EchoAttentionRunner._forward / _consume
* models/deepseek_v32/execution/adapter.py: _forward_leased
* models/deepseek_v32/cache/prefetch.py: PoolHistoryPrefetch._submit / wait
* operators/deepseek_v32/indexer/csrc/echo_logits.cuh: fused warp groups

All horizontal positions are layout coordinates. They are neither measurements
nor performance estimates. The lower lane repeats traffic within its owning
kernel; only dense prefetch uses an independent stream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#233142"
MUTED = "#596775"
COMPUTE = "#E5F1F8"
COMPUTE_EDGE = "#0072B2"
TRANSFER = "#FFF0D0"
TRANSFER_EDGE = "#B97800"
CONTROL = "#F0F1F3"
CONTROL_EDGE = "#7C8793"
MAIN_Y = 1.12
TRAFFIC_Y = 0.25
BOX_HEIGHT = 0.56
END = 14.8


def box(axis, start, width, label, *, y=MAIN_Y, kind="compute", dashed=False, size=13):
    fill, edge = {
        "compute": (COMPUTE, COMPUTE_EDGE),
        "transfer": (TRANSFER, TRANSFER_EDGE),
        "control": (CONTROL, CONTROL_EDGE),
    }[kind]
    patch = FancyBboxPatch(
        (start, y),
        width,
        BOX_HEIGHT,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.1,
        linestyle=(0, (4, 3)) if dashed else "solid",
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(
        start + width / 2,
        y + BOX_HEIGHT / 2,
        label,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        linespacing=1.12,
        zorder=4,
    )


def arrow(axis, start, stop, *, color=MUTED, dashed=False, connection="arc3"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            stop,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
            linestyle=(0, (3, 3)) if dashed else "solid",
            connectionstyle=connection,
            zorder=2,
        )
    )


def stages(axis, items):
    for index, (start, width, label, kind) in enumerate(items):
        box(axis, start, width, label, kind=kind)
        if index:
            previous_start, previous_width, *_ = items[index - 1]
            arrow(
                axis,
                (previous_start + previous_width + 0.01, MAIN_Y + BOX_HEIGHT / 2),
                (start - 0.015, MAIN_Y + BOX_HEIGHT / 2),
            )


def panel(axis, number, name, note, *, main_label="GPU\nstream", traffic_label="Host\n→ HBM"):
    axis.set_xlim(-1.85, END + 0.15)
    axis.set_ylim(-0.17, 2.47)
    axis.axis("off")
    axis.text(-1.78, 2.19, f"{number}  {name}", fontsize=15.5, color=INK, weight="bold")
    axis.text(END, 2.19, note, fontsize=12, color=MUTED, ha="right")
    axis.text(
        -0.2,
        MAIN_Y + BOX_HEIGHT / 2,
        main_label,
        fontsize=11 if name == "Dense prefetch" else 12,
        color=MUTED,
        ha="right",
        va="center",
    )
    axis.text(
        -0.2,
        TRAFFIC_Y + BOX_HEIGHT / 2,
        traffic_label,
        fontsize=10.5 if name == "Dense prefetch" else 12,
        color=MUTED,
        ha="right",
        va="center",
    )
    axis.plot([0, END], [TRAFFIC_Y + BOX_HEIGHT / 2] * 2, color="#DCE1E5", lw=0.8, zorder=0)
    axis.plot([-1.78, END], [-0.14, -0.14], color="#DFE4E8", lw=0.8, clip_on=False)


def hbm_panel(axis):
    panel(axis, "a", "HBM-only", "History KV is already resident in this layer")
    stages(
        axis,
        [
            (0.0, 2.05, "Norm + Proj", "compute"),
            (2.25, 2.45, "Indexer logits", "compute"),
            (4.9, 2.1, "Exact top-k", "compute"),
            (7.2, 1.8, "KV write", "control"),
            (9.2, 2.25, "Sparse MLA", "compute"),
            (11.65, 3.15, "Out + Norm + MLP", "compute"),
        ],
    )
    axis.text(
        7.4,
        TRAFFIC_Y + BOX_HEIGHT / 2,
        "No history KV transfer",
        ha="center",
        va="center",
        fontsize=13,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 5},
    )


def echo_panel(axis):
    panel(axis, "b", "ECHO", "One fused kernel overlaps work across query groups")
    stages(
        axis,
        [
            (0.0, 1.8, "Norm +\nProj", "compute"),
            (2.0, 4.0, "Indexer logits\n(later query groups)", "compute"),
            (6.2, 1.55, "Exact\ntop-k", "compute"),
            (7.95, 1.1, "KV\nwrite", "control"),
            (9.25, 1.8, "Recall +\nremap", "control"),
            (11.25, 1.6, "Sparse\nMLA", "compute"),
            (13.05, 1.75, "Out + Norm\n+ MLP", "compute"),
        ],
    )
    enclosure = FancyBboxPatch(
        (1.91, TRAFFIC_Y - 0.095),
        4.18,
        MAIN_Y + BOX_HEIGHT - TRAFFIC_Y + 0.19,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        facecolor="none",
        edgecolor=COMPUTE_EDGE,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        zorder=5,
    )
    axis.add_patch(enclosure)
    axis.text(
        4,
        1.9,
        "ONE CUDA KERNEL",
        ha="center",
        va="center",
        fontsize=11,
        color=COMPUTE_EDGE,
        weight="bold",
    )
    box(
        axis,
        2.55,
        3.35,
        "Earlier-query misses\n(provable subset)",
        y=TRAFFIC_Y,
        kind="transfer",
        size=12.5,
    )
    box(axis, 9.25, 1.8, "Remaining\nexact misses", y=TRAFFIC_Y, kind="transfer", size=12)
    arrow(
        axis,
        (6.0, TRAFFIC_Y + BOX_HEIGHT / 2),
        (6.2, MAIN_Y),
        color=COMPUTE_EDGE,
        connection="angle,angleA=0,angleB=-90,rad=0",
    )
    arrow(axis, (10.9, TRAFFIC_Y + BOX_HEIGHT), (11.25, MAIN_Y + 0.1), color=TRANSFER_EDGE)


def serial_panel(axis):
    panel(axis, "c", "Serial sparse", "Exact selection → transfer → MLA on one stream")
    stages(
        axis,
        [
            (0.0, 1.8, "Norm +\nProj", "compute"),
            (2.0, 2.4, "Indexer logits", "compute"),
            (4.6, 1.65, "Exact\ntop-k", "compute"),
            (6.45, 1.15, "KV\nwrite", "control"),
            (7.8, 2.25, "Recall + remap", "control"),
            (10.25, 1.95, "Sparse MLA", "compute"),
            (12.4, 2.4, "Out + Norm\n+ MLP", "compute"),
        ],
    )
    box(axis, 7.8, 2.25, "Exact history misses", y=TRAFFIC_Y, kind="transfer", size=12)
    arrow(axis, (9.9, TRAFFIC_Y + BOX_HEIGHT), (10.25, MAIN_Y + 0.1), color=TRANSFER_EDGE)


def dense_body(axis):
    stages(
        axis,
        [
            (0.0, 1.05, "Wait\nL1", "control"),
            (1.25, 1.2, "Prepare\nL2", "control"),
            (2.65, 1.5, "Norm +\nProj", "compute"),
            (4.35, 1.7, "Indexer\nlogits", "compute"),
            (6.25, 1.35, "Exact\ntop-k", "compute"),
            (7.8, 1.6, "KV write\n+ remap", "control"),
            (9.6, 1.6, "Sparse\nMLA", "compute"),
            (11.4, 1.9, "Out + Norm\n+ MLP", "compute"),
            (13.5, 1.3, "Wait\nL2", "control"),
        ],
    )
    axis.text(
        0.53,
        TRAFFIC_Y + 0.14,
        "L1 ready",
        ha="center",
        va="center",
        fontsize=11,
        color=TRANSFER_EDGE,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
    )
    arrow(axis, (0.53, TRAFFIC_Y + 0.37), (0.53, MAIN_Y - 0.02), color=TRANSFER_EDGE)
    box(
        axis,
        2.65,
        10.1,
        "L2 history prefetch: all history misses  ·  DRAM → HBM  ·  H ≤ P",
        y=TRAFFIC_Y,
        kind="transfer",
        size=13,
    )
    arrow(axis, (2.2, MAIN_Y - 0.01), (2.66, TRAFFIC_Y + BOX_HEIGHT + 0.01), color=TRANSFER_EDGE)
    arrow(
        axis,
        (12.77, TRAFFIC_Y + BOX_HEIGHT / 2),
        (14.15, MAIN_Y - 0.01),
        color=TRANSFER_EDGE,
        connection="angle,angleA=0,angleB=-90,rad=0",
    )
    axis.plot(
        [13.4, 13.4],
        [MAIN_Y - 0.08, MAIN_Y + BOX_HEIGHT + 0.12],
        color=CONTROL_EDGE,
        lw=0.9,
        ls=(0, (2, 3)),
    )
    axis.text(14.15, 1.9, "L2 ENTRY", ha="center", va="center", fontsize=10.5, color=MUTED)


def dense_panel(axis):
    panel(
        axis,
        "d",
        "Dense prefetch",
        "L1 computation overlaps the prefetch of L2 history KV",
        main_label="L1 compute\n(main stream)",
        traffic_label="L2 prefetch\n(independent\nstream)",
    )
    dense_body(axis)


def save_figure(figure, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        metadata = {"Creator": "draw_layer_schedule.py"}
        if suffix == "svg":
            metadata["Date"] = None
        elif suffix == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        target = output_dir / f"{name}.{suffix}"
        figure.savefig(target, dpi=220, metadata=metadata)
        print(target)
    plt.close(figure)


def draw_dense_overlap(output_dir):
    figure, axis = plt.subplots(figsize=(15.6, 5.6))
    figure.subplots_adjust(left=0.025, right=0.985, top=0.80, bottom=0.22)
    axis.set_xlim(-3.9, 17.6)
    axis.set_ylim(-0.35, 2.2)
    axis.axis("off")
    figure.text(
        0.035,
        0.93,
        "Dense prefetch: L1 computation overlaps L2 history transfer",
        fontsize=22,
        color=INK,
        weight="bold",
    )
    figure.text(
        0.035,
        0.862,
        "DeepSeek V3.2 · fixed P/NH · candidate execution",
        fontsize=14,
        color=MUTED,
    )
    figure.text(
        0.965,
        0.862,
        "Scheduling schematic · not to scale",
        ha="right",
        fontsize=14,
        color=COMPUTE_EDGE,
    )
    for y, label, color in (
        (MAIN_Y, "L1 compute\n(main stream)", COMPUTE_EDGE),
        (TRAFFIC_Y, "L2 prefetch\n(independent stream)", TRANSFER_EDGE),
    ):
        axis.text(
            -0.3,
            y + BOX_HEIGHT / 2,
            label,
            ha="right",
            va="center",
            fontsize=14,
            color=color,
            weight="bold",
            linespacing=1.25,
        )
        axis.plot([0, 17.5], [y + BOX_HEIGHT / 2] * 2, color="#DCE1E5", lw=0.8, zorder=0)
    dense_body(axis)
    box(axis, 15.2, 2.3, "L2 computation", size=13)
    arrow(axis, (14.82, MAIN_Y + BOX_HEIGHT / 2), (15.18, MAIN_Y + BOX_HEIGHT / 2))
    axis.text(
        7.7,
        0.965,
        "OVERLAP",
        ha="center",
        va="center",
        fontsize=13,
        color=INK,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 3},
    )
    axis.text(1.85, 1.88, "Metadata", ha="center", va="center", fontsize=11, color=MUTED)
    axis.text(
        7.975,
        1.88,
        "L1 computation",
        ha="center",
        va="center",
        fontsize=13,
        color=COMPUTE_EDGE,
        weight="bold",
    )
    arrow(axis, (2.65, -0.14), (17.5, -0.14))
    axis.text(10.075, -0.27, "Time →", ha="center", va="top", fontsize=13, color=MUTED)
    figure.text(
        0.035,
        0.13,
        "Prepare L2 metadata, launch its history transfer, then compute L1 while L2 KV is fetched. "
        "Join the transfer before L2 computation starts.",
        fontsize=13.2,
        color=INK,
    )
    figure.text(
        0.035,
        0.073,
        "Only history misses are read from pinned DRAM into the L2 HBM pool; hits are reused. "
        "Candidate KV remains on the GPU (D2H = 0).",
        fontsize=12.5,
        color=MUTED,
    )
    save_figure(figure, output_dir, "dense_prefetch_overlap")


def draw(output_dir: Path):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "svg.fonttype": "none",
            "svg.hashsalt": "deepseek-one-layer-dependency-schematic",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(4, 1, figsize=(14.4, 11.0))
    figure.subplots_adjust(left=0.02, right=0.99, top=0.879, bottom=0.18, hspace=0.14)
    figure.text(
        0.031,
        0.966,
        "DeepSeek V3.2 · computation and communication in one layer",
        fontsize=20,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.031,
        0.932,
        "Fixed P/NH · candidate execution · local implementations",
        fontsize=14,
        color=MUTED,
    )
    figure.text(
        0.969,
        0.932,
        "DEPENDENCY SCHEMATIC · NOT TO SCALE",
        ha="right",
        fontsize=13,
        color=COMPUTE_EDGE,
        weight="bold",
    )
    for axis, builder in zip(axes, (hbm_panel, echo_panel, serial_panel, dense_panel), strict=True):
        builder(axis)
    axis = axes[-1]
    arrow(axis, (2.65, -0.31), (13.3, -0.31))
    axis.text(
        7.975,
        -0.44,
        "Execution order →   widths do not encode duration",
        ha="center",
        va="top",
        fontsize=12,
        color=MUTED,
        clip_on=False,
    )
    figure.text(
        0.031,
        0.131,
        "ECHO overlaps inside one kernel; serial sparse waits for exact misses; dense can "
        "prefetch the next layer.",
        fontsize=13.1,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.031,
        0.105,
        "The traffic lane is a semantic view of GPU loads from pinned host memory, not a DMA "
        "engine. Internal ECHO overlap is not timed here.",
        fontsize=12.1,
        color=MUTED,
    )
    figure.text(
        0.031,
        0.080,
        "Dense prefetch overlaps L1 computation with L2 history transfer on an independent stream. "
        "Hits are reused. All candidate KV D2H = 0.",
        fontsize=12.1,
        color=MUTED,
    )
    figure.text(
        0.031,
        0.055,
        "Proj includes main/indexer projections, RoPE and quantization; Out includes value "
        "expansion and output projection; MLP is dense.",
        fontsize=12.1,
        color=MUTED,
    )
    figure.text(
        0.031,
        0.030,
        "Shared indexer D2D staging, cache metadata and ECHO finalize are simplified. "
        "ECHO uses the resident indexer when all history is resident.",
        fontsize=12.1,
        color=MUTED,
    )
    save_figure(figure, output_dir, "schedule")
    draw_dense_overlap(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "report" / "single_layer",
        help="Directory for schedule and dense_prefetch_overlap in SVG, PDF, and PNG.",
    )
    args = parser.parse_args()
    draw(args.output_dir)


if __name__ == "__main__":
    main()
