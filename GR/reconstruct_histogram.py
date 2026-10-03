"""Reconstruct an integer user-weight histogram from saved cumulative heat points.

This is an approximation, not an exact histogram of the unavailable raw CSV.
It preserves every saved cumulative point and the total user/weight counts without
materializing individual users or sampling request events.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent / "analysis"
BIN_LIMITS = [(1, 1)] + [(2**power + 1, 2 ** (power + 1)) for power in range(1, 12)]
BIN_LIMITS[1] = (2, 4)
BIN_LIMITS.append((4097, None))
SCALED_BIN_LIMITS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 100)]


def _integer_count(fraction, scale):
    value = Decimal(fraction) * scale
    integer = int(value.to_integral_value())
    if abs(value - integer) > Decimal("0.000001"):
        raise ValueError("saved curve does not resolve integer user/visit counts")
    return integer


def reconstruct(points, users, visits):
    histogram = Counter()
    previous_rank = previous_visits = 0
    segments = []
    for point in points:
        rank = _integer_count(point["user_fraction"], users)
        cumulative = _integer_count(point["traffic_fraction"], visits)
        segment_users = rank - previous_rank
        segment_visits = cumulative - previous_visits
        if segment_users <= 0 or segment_visits < segment_users:
            raise ValueError("expected increasing ranks and at least one visit per user")
        low, high_users = divmod(segment_visits, segment_users)
        low_users = segment_users - high_users
        histogram[low] += low_users
        if high_users:
            histogram[low + 1] += high_users
        segments.append(
            {
                "first_rank": previous_rank + 1,
                "last_rank": rank,
                "users": segment_users,
                "visits": segment_visits,
                "floor_visits_per_user": low,
                "floor_users": low_users,
                "ceil_users": high_users,
            }
        )
        previous_rank, previous_visits = rank, cumulative
    if (previous_rank, previous_visits) != (users, visits):
        raise ValueError("curve endpoints disagree with saved population totals")
    assert sum(histogram.values()) == users
    assert sum(count * frequency for count, frequency in histogram.items()) == visits
    return histogram, segments


def _percentile(histogram, fraction):
    position = fraction * (sum(histogram.values()) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    running = 0
    values = {}
    for count, frequency in sorted(histogram.items()):
        for index in (lower, upper):
            if running <= index < running + frequency:
                values[index] = count
        running += frequency
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, bins, summary, points):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, StrMethodFormatter

    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    scaled = summary["field"] == "pv_scaled_1_100"
    value_name = "scaled_weight" if scaled else "visits_per_user"
    basename = "scaled_weight_per_user" if scaled else "visits_per_user"
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    x = list(range(len(bins)))
    values = [row["users"] for row in bins]
    bars = ax.bar(x, values, color="#D55E00" if scaled else "#0072B2", width=0.72)
    ax.set_yscale("log")
    ax.set_ylim(0.5, summary["users"] * (1.6 if scaled else 1))
    ax.set_xticks(x, [row[value_name] for row in bins], rotation=38, ha="right")
    ax.set_ylabel("Number of users (log scale)")
    ax.set_xlabel(
        "Scaled user weight (not visit counts)"
        if scaled
        else "Visits per user (including the first visit)"
    )
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.23,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.suptitle(
        "Industrial 10M: pv_scaled_1_100 — reconstructed reference"
        if scaled
        else "Industrial 10M: visits per user — reconstructed reference",
        fontsize=15,
        y=0.985,
    )
    ax.set_title(
        f"{summary['users']:,} users  |  weight sum {int(summary['total_weight']):,}  |  mean weight {summary['mean_weight']:g}"
        if scaled
        else f"{summary['users']:,} users  |  {int(summary['total_weight']):,} visits  |  "
        f"mean {summary['mean_weight']:g} visits/user  |  pv_int",
        fontsize=11,
        pad=15,
    )
    fig.text(
        0.5,
        0.014,
        f"Approximation from {points} saved cumulative-curve points; raw CSV unavailable. "
        "No random request sampling.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(path / f"{basename}.{extension}", dpi=190)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=("pv_int", "pv_scaled_1_100"), default="pv_int")
    parser.add_argument("--curve-path", type=Path, default=ANALYSIS / "heat_curves.csv")
    parser.add_argument("--summary-path", type=Path, default=ANALYSIS / "heat_summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary_rows = json.loads(args.summary_path.read_text())
    summary = next(
        row
        for row in summary_rows
        if row["dataset"] == "industrial_10M" and row["field"] == args.field
    )
    users, visits = summary["users"], int(summary["total_weight"])
    if type(users) is not int or users <= 0 or visits != summary["total_weight"]:
        raise ValueError("invalid saved population totals")
    with args.curve_path.open(newline="", encoding="utf-8") as handle:
        points = [
            row
            for row in csv.DictReader(handle)
            if row["dataset"] == "industrial_10M" and row["field"] == args.field
        ]
    histogram, segments = reconstruct(points, users, visits)
    scaled = args.field == "pv_scaled_1_100"
    value_name = "scaled_weight" if scaled else "visits_per_user"
    total_name = "weight_sum" if scaled else "visits"
    if scaled:
        segments = [
            {
                {"visits": "weight_sum", "floor_visits_per_user": "floor_scaled_weight"}.get(
                    key, key
                ): value
                for key, value in segment.items()
            }
            for segment in segments
        ]
    exact_counts = [
        {value_name: count, "users": frequency, total_name: count * frequency}
        for count, frequency in sorted(histogram.items())
    ]
    bins = []
    for low, high in SCALED_BIN_LIMITS if scaled else BIN_LIMITS:
        selected = {
            count: frequency
            for count, frequency in histogram.items()
            if count >= low and (high is None or count <= high)
        }
        frequency = sum(selected.values())
        bins.append(
            {
                value_name: str(low)
                if high == low
                else f"{low}+"
                if high is None
                else f"{low}–{high}",
                "users": frequency,
                "user_fraction": frequency / users,
                total_name: sum(count * number for count, number in selected.items()),
            }
        )
    assert sum(row["users"] for row in bins) == users
    assert sum(row[total_name] for row in bins) == visits
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _csv(args.output_dir / ("weight_counts.csv" if scaled else "visit_counts.csv"), exact_counts)
    _csv(args.output_dir / "histogram.csv", bins)
    _csv(args.output_dir / "curve_segments.csv", segments)
    _plot(args.output_dir, bins, summary, len(points))
    comparison = {
        "users": users,
        total_name: visits,
        "min": min(histogram),
        "mean": visits / users,
        "median": _percentile(histogram, 0.5),
        "p90": _percentile(histogram, 0.9),
        "p99": _percentile(histogram, 0.99),
        "max": max(histogram),
    }
    manifest = {
        "kind": "approximate curve-reconstructed histogram; not original raw CSV histogram",
        "field": args.field,
        "unit": "scaled user weight; not visits" if scaled else "visits including first access",
        "method": "For each rank segment, divide its exact integer weight sum among its users as floor/ceil of the segment mean, preserving both integer totals",
        "raw_events_sampled": False,
        "curve_points": len(points),
        "saved_full_population_summary": summary,
        "reconstructed_summary": comparison,
        "source_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.curve_path, args.summary_path, Path(__file__))
        },
        "artifact_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.output_dir.iterdir())
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
