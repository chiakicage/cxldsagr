"""Render reproducible figures from the derived twelve-shape statistics CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

HISTORIES = (4096, 16384, 65536)
EXTENDS = (128, 256, 512, 1024)
METHODS = ("hbm", "echo", "serial_sparse", "dense_prefetch")
OFFLOAD = METHODS[1:]
COLORS = {
    "hbm": "#68727D",
    "echo": "#82499B",
    "serial_sparse": "#087F8C",
    "dense_prefetch": "#D98120",
}
LABELS = {
    "hbm": "HBM resident",
    "echo": "ECHO",
    "serial_sparse": "Serial sparse",
    "dense_prefetch": "Dense prefetch",
}
MARKERS = {"hbm": "s", "echo": "D", "serial_sparse": "o", "dense_prefetch": "^"}
STYLES = {"hbm": "-", "echo": "--", "serial_sparse": "-", "dense_prefetch": "-."}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "svg.fonttype": "none",
    "svg.hashsalt": "deepseek-v32-shape-statistics-v1",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#A9AFB6",
    "axes.linewidth": 0.8,
    "text.color": "#24282E",
    "axes.labelcolor": "#24282E",
    "xtick.color": "#40464E",
    "ytick.color": "#40464E",
}


def _read(output, filename, *, phase=None):
    with (output / filename).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if phase is not None:
        rows = [row for row in rows if row["phase"] == phase]
    indexed = {}
    for row in rows:
        key = int(row["prefix_tokens"]), int(row["extend_tokens"]), row["method"]
        if key in indexed:
            raise ValueError(f"Duplicate statistics row in {filename}: {key}")
        indexed[key] = row
    expected = {(h, a, method) for h in HISTORIES for a in EXTENDS for method in METHODS}
    if indexed.keys() != expected:
        raise ValueError(f"Incomplete or unexpected shape/method coverage in {filename}")
    return indexed


def _number(row, name):
    value = float(row[name])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid nonnegative statistic {name}: {row[name]}")
    return value


def _interval(row, fields):
    median, low, high = (_number(row, name) for name in fields)
    if not low <= median <= high:
        raise ValueError(f"Median is outside the observed sample range: {fields}")
    return median, low, high


def _line_style(method):
    return {
        "color": COLORS[method],
        "linestyle": STYLES[method],
        "linewidth": 1.8 if method == "echo" else 2.0,
        "marker": MARKERS[method],
        "markersize": 7 if method == "echo" else 5.5,
        "markerfacecolor": "none" if method == "echo" else COLORS[method],
        "markeredgewidth": 1.3,
        "label": LABELS[method],
    }


def _legend(figure, methods):
    handles = [Line2D([], [], **_line_style(method)) for method in methods]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.061),
        ncol=len(methods),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.7,
    )


def _save(figure, output, name):
    paths = []
    try:
        for suffix in ("svg", "png"):
            path = output / f"{name}.{suffix}"
            metadata = (
                {"Date": None, "Creator": "DeepSeek V3.2 shape statistics"}
                if suffix == "svg"
                else {"Software": "DeepSeek V3.2 shape statistics"}
            )
            figure.savefig(path, dpi=180, facecolor="white", metadata=metadata)
            paths.append(path)
    finally:
        plt.close(figure)
    return paths


def _line_panels(output, rows, *, name, fields, methods, title, ylabel, note):
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 4.7), sharey=True)
    figure.subplots_adjust(left=0.072, right=0.987, bottom=0.255, top=0.805, wspace=0.09)
    figure.suptitle(title, y=0.968, fontsize=15)
    figure.text(0.5, 0.898, "DeepSeek V3.2 checkpoint L0–L2 · cold cache", ha="center", fontsize=11)
    upper = max(
        _interval(rows[h, a, method], fields)[2]
        for h in HISTORIES
        for a in EXTENDS
        for method in methods
    )
    for axis, history in zip(axes, HISTORIES):
        axis.set_title(f"History H = {history // 1024}K", pad=9)
        axis.set_xscale("log", base=2)
        axis.set_xticks(EXTENDS)
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.set_xlim(EXTENDS[0] / 1.14, EXTENDS[-1] * 1.14)
        axis.set_ylim(0, upper * 1.08 if upper else 1)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_xlabel("Extend tokens A (log₂ scale)", labelpad=9)
        axis.grid(axis="y", color="#E4E7EB", linewidth=0.7)
        axis.set_axisbelow(True)
        # Draw ECHO last: its hollow diamonds and dashes leave coincident teal visible.
        for method in [method for method in methods if method != "echo"] + ["echo"]:
            values = [_interval(rows[history, extend, method], fields) for extend in EXTENDS]
            median, low, high = zip(*values)
            axis.fill_between(EXTENDS, low, high, color=COLORS[method], alpha=0.12, linewidth=0)
            axis.plot(EXTENDS, median, **_line_style(method), zorder=3 if method == "echo" else 2)
    axes[0].set_ylabel(ylabel, labelpad=9)
    _legend(figure, methods)
    figure.text(0.5, 0.024, note, ha="center", va="bottom", fontsize=10)
    return _save(figure, output, name)


def _relative_panels(output, rows):
    values = [
        _number(rows[h, a, method], "relative_hbm")
        for method in OFFLOAD
        for h in HISTORIES
        for a in EXTENDS
    ]
    if min(values) < 1:
        raise ValueError("The requested heatmap scale starts at 1; an offload ratio is below 1")
    maximum = max(1.1, math.ceil(max(values) * 10) / 10)
    normalization = Normalize(vmin=1, vmax=maximum)
    cmap = plt.colormaps["YlOrBr"]
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 4.3), sharey=True)
    figure.subplots_adjust(left=0.068, right=0.91, bottom=0.23, top=0.78, wspace=0.13)
    figure.suptitle("Complete extend latency relative to HBM", y=0.965, fontsize=15)
    figure.text(
        0.5, 0.895, "Median wall time / HBM median at the same H and A", ha="center", fontsize=11
    )
    for axis, method in zip(axes, OFFLOAD):
        matrix = [[_number(rows[h, a, method], "relative_hbm") for a in EXTENDS] for h in HISTORIES]
        plot = axis.pcolormesh(
            range(len(EXTENDS) + 1),
            range(len(HISTORIES) + 1),
            matrix,
            cmap=cmap,
            norm=normalization,
            edgecolors="white",
            linewidth=0.8,
            rasterized=False,
        )
        axis.set_xlim(0, len(EXTENDS))
        axis.set_ylim(len(HISTORIES), 0)
        axis.set_title(LABELS[method], color=COLORS[method], pad=10)
        axis.set_xticks([i + 0.5 for i in range(len(EXTENDS))], [str(a) for a in EXTENDS])
        axis.set_yticks(
            [i + 0.5 for i in range(len(HISTORIES))], [f"{h // 1024}K" for h in HISTORIES]
        )
        axis.set_xlabel("Extend tokens A", labelpad=10)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                red, green, blue, _ = cmap(normalization(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                axis.text(
                    column + 0.5,
                    row + 0.5,
                    f"{value:.3f}×",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if luminance < 0.47 else "#24282E",
                )
    axes[0].set_ylabel("History tokens H", labelpad=10)
    color_axis = figure.add_axes((0.935, 0.23, 0.016, 0.55))
    colorbar = figure.colorbar(plot, cax=color_axis)
    colorbar.solids.set_rasterized(False)
    colorbar.ax.set_title("Ratio", fontsize=11, pad=9)
    colorbar.outline.set_visible(False)
    figure.text(
        0.5,
        0.06,
        "1.000× matches HBM; larger factors mean longer latency. One common color scale.",
        ha="center",
        fontsize=10,
    )
    return _save(figure, output, "relative_latency")


def render(output: Path):
    """Read derived CSVs and write three SVG/PNG pairs in the same directory."""
    output = Path(output)
    points = _read(output, "point_statistics.csv", phase="extend")
    comparisons = _read(output, "comparisons.csv")
    traffic = _read(output, "traffic_statistics.csv")
    counts = sorted({int(row["n"]) for row in points.values()})
    if not counts or counts[0] < 1:
        raise ValueError("Latency statistics require positive sample counts")
    count_label = "/".join(map(str, counts))
    with plt.rc_context(STYLE):
        paths = _line_panels(
            output,
            points,
            name="extend_latency",
            fields=("median_ms", "min_ms", "max_ms"),
            methods=METHODS,
            title="Complete extend wall time",
            ylabel="Wall time (ms)",
            note=f"Lines: medians. Shading: observed min–max over n = {count_label} samples per point, not confidence intervals.",
        )
        paths += _relative_panels(output, comparisons)
        paths += _line_panels(
            output,
            traffic,
            name="h2d_payload",
            fields=("h2d_mib_median", "h2d_mib_min", "h2d_mib_max"),
            methods=OFFLOAD,
            title="Cache host-to-device payload per complete extend",
            ylabel="H2D payload, L0–L2 total (MiB)",
            note="Lines: medians; shading: min–max. ECHO dashes and hollow diamonds distinguish overlapping serial-sparse payloads.",
        )
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
