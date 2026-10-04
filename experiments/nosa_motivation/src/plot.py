"""Render latency figures from independently audited NOSA motivation data."""

import argparse
import json
from pathlib import Path

from experiments.nosa_motivation.src.config import METHODS
from experiments.nosa_motivation.src.report import audit_run, summarize

LABELS = {
    "hbm": "HBM-only",
    "dense_prefetch": "Dense prefetch",
    "sync_sparse": "Sync sparse",
    "async_sparse": "Async sparse",
}
COLORS = ("#586172", "#d58531", "#4084b5", "#39997a")


def plot(directory, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory, destination = Path(directory), Path(destination)
    metadata, rows, _ = audit_run(directory)
    summary = summarize(rows)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for axis, visit in zip(axes, ("first", "revisit"), strict=True):
        selected = {r["method"]: r for r in summary if r["visit_kind"] == visit}
        means = [selected[method]["latency_mean_ms"] for method in METHODS]
        bars = axis.bar(range(4), means, color=COLORS, width=0.66)
        axis.set_xticks(range(4), [LABELS[method] for method in METHODS], rotation=15)
        axis.set_ylabel("Synchronized request latency (ms)")
        axis.set_title("First visit" if visit == "first" else "Revisit")
        for bar, value in zip(bars, means, strict=True):
            axis.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
        axis.set_ylim(0, max(means) * 1.2)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle(f"NOSA — {metadata['run_id']}")
    for suffix in ("png", "svg"):
        figure.savefig(destination / f"latency_summary.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for method, color in zip(METHODS, COLORS, strict=True):
        selected = sorted((r for r in rows if r["method"] == method), key=lambda r: r["request_id"])
        axis.plot(
            [r["request_id"] + 1 for r in selected],
            [r["latency_ms"] for r in selected],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=color,
            label=LABELS[method],
        )
    for boundary in range(1, metadata["config"]["rounds"]):
        axis.axvline(
            boundary * metadata["config"]["num_users"] + 0.5,
            color="#999999",
            linestyle="--",
            linewidth=0.8,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Request order (same cyclic user sequence for every method)")
    axis.set_ylabel("Synchronized request latency (ms, log scale)")
    axis.set_title(f"NOSA — {metadata['run_id']}")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    for suffix in ("png", "svg"):
        figure.savefig(destination / f"latency_per_request.{suffix}", dpi=180)
    plt.close(figure)
    (destination / "plot_provenance.json").write_text(
        json.dumps(
            {
                "run_id": metadata["run_id"],
                "source_sha256": metadata["source_sha256"],
                "workload_sha256": metadata["workload_sha256"],
                "generated_by": "python -m experiments.nosa_motivation.src.plot",
                "latency_scope": metadata["measurement_boundary"],
            },
            indent=2,
        )
        + "\n"
    )


def main():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("data_dir", type=Path)
    command.add_argument("--output-dir", required=True, type=Path)
    args = command.parse_args()
    plot(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
