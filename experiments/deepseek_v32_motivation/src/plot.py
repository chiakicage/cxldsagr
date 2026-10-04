"""Plot first visits, revisits, and the sequential finite-cache request trace."""

from __future__ import annotations

import argparse
import math
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from experiments.deepseek_v32_motivation.src.report import SCHEMES, read_jsonl, summarize

LABELS = {
    "hbm": "HBM-only",
    "echo": "ECHO",
    "serial_sparse": "Serial sparse",
    "dense_prefetch": "Dense prefetch",
}
COLORS = {
    "hbm": "#666666",
    "echo": "#0072B2",
    "serial_sparse": "#D55E00",
    "dense_prefetch": "#009E73",
}


def needs_log(values):
    return min(values) > 0 and max(values) / min(values) >= 20


def save_figure(figure, output, name):
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def summary_figure(summary, output, run_id):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.34
    for axis, visit, title in zip(
        axes, ("first", "revisit"), ("First visits", "Revisits"), strict=True
    ):
        by_scheme = {row["scheme"]: row for row in summary if row["visit_kind"] == visit}
        means = [by_scheme[scheme]["latency_mean_ms"] for scheme in SCHEMES]
        p95 = [by_scheme[scheme]["latency_p95_ms"] for scheme in SCHEMES]
        counts = {by_scheme[scheme]["requests"] for scheme in SCHEMES}
        logarithmic = visit == "revisit" and needs_log(means + p95)
        if logarithmic:
            axis.set_yscale("log")
        mean_bars = axis.bar(
            [index - width / 2 for index in range(len(SCHEMES))],
            means,
            width,
            color="#0072B2",
            label="Mean",
            zorder=3,
        )
        p95_bars = axis.bar(
            [index + width / 2 for index in range(len(SCHEMES))],
            p95,
            width,
            color="#E69F00",
            label="P95",
            zorder=3,
        )
        for bars, values in ((mean_bars, means), (p95_bars, p95)):
            axis.bar_label(
                bars,
                labels=[f"{value:.0f}" if value >= 1000 else f"{value:.1f}" for value in values],
                padding=3,
                fontsize=8,
            )
        axis.set_xticks(range(len(SCHEMES)), [LABELS[scheme] for scheme in SCHEMES], rotation=12)
        axis.set_title(
            f"{title} (n={next(iter(counts))} per scheme)" if len(counts) == 1 else title
        )
        axis.set_ylabel("E2E latency (ms, log scale)" if logarithmic else "E2E latency (ms)")
        if logarithmic:
            axis.set_ylim(min(means + p95) / 1.8, max(means + p95) * 2)
        else:
            axis.set_ylim(0, max(means + p95) * 1.2)
        axis.grid(axis="y", alpha=0.25, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right", ncol=2, fontsize=8)
    figure.suptitle(f"DeepSeek V3.2 finite-cache comparison\n{run_id}", fontsize=12)
    figure.text(
        0.5,
        0.015,
        "One measurement per request; P95 uses linear interpolation. Panel scales may differ.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(figure, output, "latency_summary")


def request_figure(rows, output, run_id):
    figure, axis = plt.subplots(figsize=(11, 4.6))
    reference = None
    for scheme in SCHEMES:
        group = sorted(
            (row for row in rows if row["scheme"] == scheme), key=lambda row: row["request_id"]
        )
        identities = [(row["request_id"], row["is_revisit"]) for row in group]
        if len({key for key, _ in identities}) != len(identities):
            raise ValueError(f"duplicate request IDs for {scheme}")
        if reference is None:
            reference = identities
        elif identities != reference:
            raise ValueError("schemes must contain the same ordered requests and visit labels")
        axis.plot(
            [row["request_id"] + 1 for row in group],
            [row["latency_ms"] for row in group],
            label=LABELS[scheme],
            color=COLORS[scheme],
            marker="o",
            markersize=3.5,
            linewidth=1.3,
        )
    logarithmic = needs_log([row["latency_ms"] for row in rows])
    if logarithmic:
        axis.set_yscale("log")
    else:
        axis.set_ylim(bottom=0)
    for previous, current in pairwise(reference):
        if previous[1] != current[1]:
            axis.axvline((previous[0] + current[0]) / 2 + 1, color="#888888", linestyle="--")
    axis.set_xlim(reference[0][0] + 0.5, reference[-1][0] + 1.5)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    axis.set_xlabel("Request order (first round, then revisits; dashed line marks the transition)")
    axis.set_ylabel("E2E latency (ms, log scale)" if logarithmic else "E2E latency (ms)")
    axis.set_title(f"Sequential request latency\n{run_id}")
    axis.grid(axis="y", which="major", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    figure.tight_layout()
    save_figure(figure, output, "latency_per_request")


def plot(run_dir, output_dir, *, per_request=True):
    rows = read_jsonl(Path(run_dir) / "measurements.jsonl")
    if not rows or any(
        not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0 for row in rows
    ):
        raise ValueError("plots require nonempty, positive finite latency measurements")
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("plots require one run ID")
    summary = summarize(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_id = next(iter(run_ids))
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        summary_figure(summary, output, run_id)
        if per_request:
            request_figure(rows, output, run_id)


def main(argv=None):
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--no-per-request", action="store_true", help="save only the summary figure"
    )
    args = command.parse_args(argv)
    plot(args.run_dir, args.output_dir, per_request=not args.no_per_request)
    print(f"figures: {args.output_dir}")


if __name__ == "__main__":
    main()
