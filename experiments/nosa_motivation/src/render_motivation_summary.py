"""Render CPU-only summary figures from the accepted NOSA motivation report.

This renderer reads published CSV/JSON assets directly. It does not import model,
benchmark, or torch modules, and it does not modify its input directory.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

METHODS = ("hbm", "dense_prefetch", "sync_sparse", "async_sparse")
LABELS = {
    "hbm": "HBM-only",
    "dense_prefetch": "Dense\nprefetch",
    "sync_sparse": "Sync\nsparse",
    "async_sparse": "Async\nsparse",
}
COLORS = {
    "hbm": "#586172",
    "dense_prefetch": "#d58531",
    "sync_sparse": "#4084b5",
    "async_sparse": "#39997a",
}
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "report" / "final"
GIB = 1 << 30


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(directory):
    names = ("summary.csv", "async_overlap.csv", "report_provenance.json", "publication.json")
    inputs = {name: sha256(directory / name) for name in names}
    publication = json.loads((directory / "publication.json").read_text())
    expected = {asset["path"]: asset["sha256"] for asset in publication["assets"]}
    for name in names[:-1]:
        if inputs[name] != expected[name]:
            raise ValueError(f"Published input hash mismatch: {name}")
    report = json.loads((directory / "report_provenance.json").read_text())
    if report["run_id"] != publication["measurement_run_id"]:
        raise ValueError("Measurement run IDs disagree")
    if report["source_sha256"] != publication["source_sha256"]:
        raise ValueError("Accepted source hashes disagree")

    with (directory / "summary.csv").open(newline="") as stream:
        raw_summary = list(csv.DictReader(stream))
    with (directory / "async_overlap.csv").open(newline="") as stream:
        raw_overlap = list(csv.DictReader(stream))
    summary = {}
    for row in raw_summary:
        key = (row["method"], row["visit_kind"])
        if key in summary:
            raise ValueError(f"Duplicate summary group: {key}")
        latency = float(row["latency_mean_ms"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"Invalid latency: {key}")
        summary[key] = {
            "method": row["method"],
            "visit_kind": row["visit_kind"],
            "requests": int(row["requests"]),
            "latency_mean_ms": latency,
            "latency_mean_s": latency / 1000,
            "prefix_hits": int(row["prefix_hits"]),
            "candidate_host_to_device_bytes": int(row["candidate_host_to_device_bytes"]),
        }
    expected_groups = {(method, visit) for method in METHODS for visit in ("first", "revisit")}
    if set(summary) != expected_groups:
        raise ValueError("Expected four methods with first/revisit groups")
    if {row["requests"] for row in summary.values()} != {16}:
        raise ValueError("This figure layout describes 16 requests per group")

    overlap, keys = [], set()
    for raw in raw_overlap:
        key = tuple(int(raw[name]) for name in ("request_id", "sample", "layer"))
        if key in keys:
            raise ValueError(f"Duplicate overlap sample: {key}")
        keys.add(key)
        miss_pages = int(raw["miss_pages"])
        ratios = [
            float(raw[name]) if raw[name] else None for name in ("page_ratio", "stripe_ratio")
        ]
        applicable = miss_pages > 0
        if applicable:
            if any(
                value is None or not math.isfinite(value) or not 0 <= value <= 1 for value in ratios
            ):
                raise ValueError(f"Invalid applicable overlap ratio: {key}")
            meets_target = all(value >= 0.9 for value in ratios)
            if raw["page_and_stripe_90pct"] != str(meets_target):
                raise ValueError(f"Overlap target flag disagrees: {key}")
        else:
            if miss_pages != 0 or ratios != [None, None] or raw["page_and_stripe_90pct"]:
                raise ValueError(f"No-fetch entry must have null ratios: {key}")
            meets_target = None
        overlap.append(
            {
                "request_id": key[0],
                "sample": key[1],
                "layer": key[2],
                "miss_pages": miss_pages,
                "page_ratio": ratios[0],
                "stripe_ratio": ratios[1],
                "page_and_stripe_90pct": meets_target,
                "included_in_plot": applicable,
                "plot_layer_with_sample_offset": key[2] + (key[1] - 1) * 0.24,
            }
        )
    if keys != {
        (request, sample, layer)
        for request in (0, 16)
        for sample in range(3)
        for layer in range(32)
    }:
        raise ValueError("Expected requests 0/16, three samples each, and all 32 layers")
    applicable = [row for row in overlap if row["included_in_plot"]]
    if len(applicable) != 96 or {row["request_id"] for row in applicable} != {16}:
        raise ValueError("This figure describes 96 applicable request-16 samples")
    if any(row["page_ratio"] != row["stripe_ratio"] for row in applicable):
        raise ValueError("Cannot combine unequal page-envelope and stripe-copy ratios")

    payload = [
        {
            "method": method,
            "visit_kind": "revisit",
            "requests": summary[(method, "revisit")]["requests"],
            "candidate_host_to_device_bytes": summary[(method, "revisit")][
                "candidate_host_to_device_bytes"
            ],
            "candidate_host_to_device_gib": summary[(method, "revisit")][
                "candidate_host_to_device_bytes"
            ]
            / GIB,
        }
        for method in METHODS[1:]
    ]
    if payload[1]["candidate_host_to_device_bytes"] != payload[2]["candidate_host_to_device_bytes"]:
        raise ValueError("The combined sparse reduction annotation requires equal payloads")
    ratios = [row["page_ratio"] for row in applicable]
    derived = {
        "sparse_payload_reduction_pct": 100
        * (
            1
            - payload[1]["candidate_host_to_device_bytes"]
            / payload[0]["candidate_host_to_device_bytes"]
        ),
        "async_revisit_latency_increase_vs_sync_pct": 100
        * (
            summary[("async_sparse", "revisit")]["latency_mean_ms"]
            / summary[("sync_sparse", "revisit")]["latency_mean_ms"]
            - 1
        ),
        "applicable_overlap_samples": len(applicable),
        "excluded_no_fetch_samples": len(overlap) - len(applicable),
        "samples_meeting_both_90pct_targets": sum(
            row["page_and_stripe_90pct"] for row in applicable
        ),
        "page_and_stripe_ratios_identical_for_every_applicable_sample": True,
        "overlap_ratio_min_median_max": [min(ratios), statistics.median(ratios), max(ratios)],
    }
    return publication, inputs, list(summary.values()), payload, overlap, derived


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", length=3, width=0.6)
    axis.grid(axis="y", linewidth=0.5, color="#e1e1e1")
    axis.set_axisbelow(True)


def bars(axis, methods, values, digits):
    rectangles = axis.bar(
        range(len(methods)), values, width=0.62, color=[COLORS[method] for method in methods]
    )
    axis.set_xticks(range(len(methods)), [LABELS[method] for method in methods])
    for rectangle, value in zip(rectangles, values, strict=True):
        axis.annotate(
            f"{value:.{digits}f}",
            (rectangle.get_x() + rectangle.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    style_axis(axis)


def save_figure(figure, directory, name):
    for extension in ("png", "svg", "pdf"):
        metadata = {"Creator": "NOSA motivation CPU summary renderer"}
        if extension == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        elif extension == "svg":
            metadata["Date"] = None
        figure.savefig(directory / f"{name}.{extension}", dpi=300, metadata=metadata)


def render_figures(directory, summary, payload, overlap, derived):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "nosa-motivation-summary-v1",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )
    lookup = {(row["method"], row["visit_kind"]): row for row in summary}
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.23, top=0.87, wspace=0.36)
    bars(axes[0], METHODS, [lookup[(method, "first")]["latency_mean_s"] for method in METHODS], 3)
    axes[0].set(
        title="(a) First visits: all methods", ylabel="Mean request latency (s)", ylim=(0, 3.4)
    )
    axes[0].set_yticks((0, 1, 2, 3))
    bars(
        axes[1],
        METHODS[1:],
        [lookup[(method, "revisit")]["latency_mean_ms"] for method in METHODS[1:]],
        2,
    )
    axes[1].set(
        title="(b) Revisits: offload methods", ylabel="Mean request latency (ms)", ylim=(0, 110)
    )
    axes[1].set_yticks((0, 20, 40, 60, 80, 100))
    hbm_ms = lookup[("hbm", "revisit")]["latency_mean_ms"]
    axes[1].text(
        0.98,
        0.98,
        f"HBM-only: {hbm_ms:.2f} ms\nhistory rebuilt; shown separately",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#333333",
    )
    figure.text(
        0.5,
        0.055,
        "One measured trace; 16 first visits and 16 revisits per method.",
        ha="center",
        fontsize=9,
    )
    save_figure(figure, directory, "latency_focus")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), gridspec_kw={"width_ratios": (0.9, 1.1)})
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.23, top=0.87, wspace=0.38)
    bars(axes[0], METHODS[1:], [row["candidate_host_to_device_gib"] for row in payload], 3)
    axes[0].set(
        title="(a) Revisit candidate transfer",
        ylabel="Total logical H2D payload (GiB)",
        ylim=(0, 40),
    )
    axes[0].set_yticks((0, 10, 20, 30, 40))
    axes[0].text(
        0.98,
        0.94,
        f"Sparse: −{derived['sparse_payload_reduction_pct']:.2f}%",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    for sample, marker in enumerate(("o", "s", "^")):
        selected = [row for row in overlap if row["included_in_plot"] and row["sample"] == sample]
        axes[1].scatter(
            [row["plot_layer_with_sample_offset"] for row in selected],
            [row["page_ratio"] for row in selected],
            marker=marker,
            s=16,
            linewidths=0.7,
            facecolors="none" if sample == 1 else COLORS["async_sparse"],
            edgecolors=COLORS["async_sparse"],
            label=f"Sample {sample}",
            zorder=3,
        )
    axes[1].axhline(0.9, color="#586172", linestyle="--", linewidth=0.9)
    axes[1].text(31.7, 0.923, "90% target", ha="right", va="bottom", fontsize=9, color="#586172")
    axes[1].set(
        title="(b) Async internal overlap",
        xlabel="Layer index",
        ylabel="Overlap ratio",
        xlim=(-1, 32),
        ylim=(0, 1),
        xticks=(0, 8, 16, 24, 31),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[1].text(
        0.98,
        0.25,
        f"{derived['samples_meeting_both_90pct_targets']} / {derived['applicable_overlap_samples']} meet 90%",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=9,
    )
    axes[1].legend(loc="lower left", ncol=1, frameon=False, borderpad=0.1, labelspacing=0.25)
    style_axis(axes[1])
    figure.text(
        0.5,
        0.055,
        "Payload: 16 revisits. Overlap: request 16, three samples × 32 layers.",
        ha="center",
        fontsize=9,
    )
    save_figure(figure, directory, "transfer_overlap")
    plt.close(figure)
    return matplotlib.__version__


def captions(publication):
    measurement = publication["measurement_run_id"]
    profile = publication["profile_run_id"]
    return {
        "latency_focus": (
            "Retaining histories reduces revisit latency in this 16-user trace, while async sparse "
            "fetch is slower than sync sparse fetch. Bars show synchronized mean request latency "
            "from one measured trace, with 16 first visits and 16 revisits per method; the left "
            "axis uses seconds and the right axis uses milliseconds, both starting at zero. "
            "The right panel includes only offload methods; the HBM-only revisit mean is printed "
            "separately because its history must be rebuilt. Full 32-layer BF16 NOSA uses "
            "H=P=65,536, A=128, chunk=1,024 and NH=16,777,216; candidates return normalized hidden "
            f"states without an LM head. Measurement run: {measurement}."
        ),
        "transfer_overlap": (
            "Sparse fetch reduces candidate logical H2D payload, but every applicable async "
            "profile sample remains below the 90% overlap target. The left panel sums main-KV "
            "host-to-device logical payload over 16 formal revisits (GiB = 2^30 bytes), not "
            "measured PCIe traffic; sync and async have identical payloads. The right panel "
            "plots all 96 applicable layer/sample pairs from profile request 16, with small "
            "horizontal offsets distinguishing the three samples; page-envelope and nonempty "
            "stripe-copy ratios are verified equal for every plotted pair and share one marker. "
            "The 96 request-0 samples have no fetch and null ratios and are excluded; overlap "
            "uses actual copy and softmax-update windows, not the full fused-kernel duration. "
            f"Measurement run: {measurement}; profile run: {profile}."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT, help="Accepted report directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New, empty render directory"
    )
    args = parser.parse_args()
    source, destination = args.input_dir.resolve(), args.output_dir.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Render output must be outside the accepted input directory")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Refusing to overwrite a nonempty render directory: {destination}")
    publication, inputs, summary, payload, overlap, derived = load_inputs(source)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "latency_plot_data.csv", summary)
    write_csv(destination / "payload_plot_data.csv", payload)
    write_csv(destination / "overlap_plot_data.csv", overlap)
    plot_data = {"latency": summary, "payload": payload, "overlap": overlap, "derived": derived}
    write_json(destination / "plot_data.json", plot_data)
    figure_captions = captions(publication)
    write_json(destination / "captions.json", figure_captions)
    version = render_figures(destination, summary, payload, overlap, derived)
    provenance = {
        "schema": "nosa-motivation-summary-render-v1",
        "render_id": destination.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "measurement_run_id": publication["measurement_run_id"],
        "profile_run_id": publication["profile_run_id"],
        "accepted_runtime_source_sha256": publication["source_sha256"],
        "renderer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "command_argv": sys.argv,
        "python_version": platform.python_version(),
        "matplotlib_version": version,
        "input_directory": str(source),
        "inputs_sha256": inputs,
        "published_input_hashes_verified": True,
        "output_directory": str(destination),
        "outputs": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(destination.iterdir())
            if path.is_file()
        ],
        "captions": figure_captions,
        "derived": derived,
        "design": {
            "figure_size_inches": [6.8, 3.4],
            "minimum_font_points_at_native_size": 8.5,
            "bar_y_axes_start_at_zero": True,
            "overlap_y_axis_limits": [0, 1],
            "method_color_encoding": COLORS,
            "method_text_labels": LABELS,
            "overlap_sample_markers": ["o", "s", "^"],
            "confidence_intervals": None,
            "png_dpi": 300,
            "vector_formats": ["svg", "pdf"],
        },
        "boundary": (
            "CPU rendering of accepted report assets only; no GPU execution or new performance "
            "measurement. Latency and payload use the formal trace; internal overlap uses a "
            "separate instrumented profile and does not establish end-to-end speedup. "
            "Null no-fetch ratios remain in plot_data.json and overlap_plot_data.csv."
        ),
    }
    write_json(destination / "provenance.json", provenance)
    print(json.dumps({"output_directory": str(destination), "derived": derived}, indent=2))


if __name__ == "__main__":
    main()
