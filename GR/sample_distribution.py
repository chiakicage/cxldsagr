"""Sample a user-access trace from a bundled heat curve, without loading a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from GR.heat import DEFAULT_HEAT_CURVES, HeatPopulation
from GR.scheduling import ScheduleConfig, _integer, _schedule

BUCKETS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257+", 257, math.inf),
)


def _write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values, fraction):
    position = fraction * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def sample_population(population, requests, schedule):
    """Return the actual trace, all-user counts, a histogram and scalar summaries."""
    _integer("requests", requests, 1)
    users = sorted(population.weights)
    ranked = sorted(users, key=lambda uid: (-population.weights[uid], uid))
    rank = {uid: index + 1 for index, uid in enumerate(ranked)}
    weight_sum = math.fsum(population.weights.values())
    counts, first, last, trace = Counter(), {}, {}, []
    for index, (uid, timestamp) in enumerate(
        _schedule(users, population.weights, requests, schedule)
    ):
        previous = last.get(uid)
        trace.append(
            {
                "request_id": index,
                "user_id": uid,
                "heat_rank": rank[uid],
                "visit_index": counts[uid],
                "is_revisit": int(counts[uid] > 0),
                "previous_request_id": previous,
                "reuse_distance_users": None
                if previous is None
                else sum(position > previous for position in last.values()),
                "synthetic_timestamp": timestamp,
            }
        )
        counts[uid] += 1
        first.setdefault(uid, index)
        last[uid] = index
    rows = [
        {
            "pool_users": len(users),
            "user_id": uid,
            "heat_rank": rank[uid],
            "probability": population.weights[uid] / weight_sum,
            "expected_visits": requests * population.weights[uid] / weight_sum,
            "visits": counts[uid],
            "revisits": max(0, counts[uid] - 1),
            "first_request_id": first.get(uid),
            "last_request_id": last.get(uid),
        }
        for uid in ranked
    ]
    histogram = [
        {
            "pool_users": len(users),
            "visit_count_bin": label,
            "users": sum(lo <= counts[uid] <= hi for uid in users),
            "user_fraction": sum(lo <= counts[uid] <= hi for uid in users) / len(users),
            "requests": sum(counts[uid] for uid in users if lo <= counts[uid] <= hi),
        }
        for label, lo, hi in BUCKETS
    ]
    values = sorted(counts[uid] for uid in users)
    top_count = math.ceil(0.1 * len(users))
    actual = sum(value > 0 for value in values)
    summary = {
        "pool_users": len(users),
        "requests": len(trace),
        "visited_users": actual,
        "unvisited_users": len(users) - actual,
        "one_visit_users": sum(value == 1 for value in values),
        "returning_users": sum(value > 1 for value in values),
        "first_visits": actual,
        "revisits": len(trace) - actual,
        "revisit_request_fraction": (len(trace) - actual) / len(trace),
        "visits_mean_all_pool_users": len(trace) / len(users),
        "visits_min_all_pool_users": values[0],
        "visits_median_all_pool_users": _quantile(values, 0.5),
        "visits_p95_all_pool_users": _quantile(values, 0.95),
        "visits_max": values[-1],
        "revisits_max": max(0, values[-1] - 1),
        "top10pct_users_ceiling": top_count,
        "top10pct_expected_request_fraction": sum(row["probability"] for row in rows[:top_count]),
        "top10pct_observed_request_fraction": sum(row["visits"] for row in rows[:top_count])
        / len(trace),
    }
    assert len(trace) == requests == sum(values)
    assert sum(row["users"] for row in histogram) == len(users)
    assert sum(row["requests"] for row in histogram) == requests
    assert sum(row["revisits"] for row in rows) == summary["revisits"]
    return trace, rows, histogram, summary


def plot_distribution(output, cases, source_curve):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    markers = ("o", "s", "^")
    bins = np.arange(len(BUCKETS))
    width = 0.8 / len(cases)
    for index, (rows, histogram, summary) in enumerate(cases):
        color = colors[index % len(colors)]
        label = f"N = {summary['pool_users']:,}"
        fractions = [row["user_fraction"] for row in histogram]
        axes[0].bar(
            bins + (index - (len(cases) - 1) / 2) * width,
            fractions,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        cumulative = np.concatenate(([0], np.cumsum([row["visits"] for row in rows])))
        ranks = np.arange(len(rows) + 1) / len(rows)
        axes[1].plot(
            ranks,
            cumulative / summary["requests"],
            color=color,
            label=label,
            marker=markers[index % len(markers)],
            markevery=max(1, len(rows) // 8),
            markersize=3,
            linewidth=1.5,
        )
    axes[0].set_xticks(bins, [label for label, _, _ in BUCKETS], rotation=40, ha="right")
    axes[0].set_xlabel("Visits per user (including the first visit)")
    axes[0].set_ylabel("Share of all users in the pool")
    axes[0].set_title("Access-count distribution; unvisited users included")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False)
    axes[1].plot(
        *zip(*source_curve, strict=True), color="#333333", linestyle="--", label="Input curve"
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Hottest users, ranked by input probability")
    axes[1].set_ylabel("Cumulative share of sampled requests")
    axes[1].set_title("Sampled concentration versus input heat")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, loc="lower right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"distribution.{suffix}", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="industrial_10M")
    parser.add_argument("--field", default="pv_share")
    parser.add_argument("--users", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--requests", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curve-path", type=Path, default=DEFAULT_HEAT_CURVES)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    _integer("requests", args.requests, 1)
    if len(args.users) != len(set(args.users)):
        parser.error("user pool sizes must be distinct")
    schedule = ScheduleConfig(seed=args.seed, arrival="constant", qps=1, max_revisits=None)
    cases, populations, traces = [], [], []
    for users in args.users:
        population = HeatPopulation.from_curve(
            args.curve_path, dataset=args.dataset, field=args.field, num_users=users, seed=args.seed
        )
        trace, rows, histogram, summary = sample_population(population, args.requests, schedule)
        cases.append((rows, histogram, summary))
        populations.append(population.metadata)
        traces.append(trace)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for (rows, histogram, summary), trace in zip(cases, traces, strict=True):
        users = summary["pool_users"]
        _write_csv(args.output_dir / f"users_{users}.csv", rows)
        _write_csv(args.output_dir / f"requests_{users}.csv", trace)
    _write_csv(args.output_dir / "summary.csv", [case[2] for case in cases])
    _write_csv(args.output_dir / "histogram.csv", [row for case in cases for row in case[1]])
    with args.curve_path.open(newline="", encoding="utf-8") as handle:
        source_curve = [(0.0, 0.0)] + [
            (float(row["user_fraction"]), float(row["traffic_fraction"]))
            for row in csv.DictReader(handle)
            if row["dataset"] == args.dataset and row["field"] == args.field
        ]
    plot_distribution(args.output_dir, cases, source_curve)
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("heat.py"),
        Path(__file__).with_name("scheduling.py"),
    ]
    metadata = {
        "dataset": args.dataset,
        "field": args.field,
        "users": args.users,
        "requests_per_population": args.requests,
        "population_seed": args.seed,
        "sampling_seed": args.seed,
        "schedule": asdict(schedule),
        "python": platform.python_version(),
        "populations": populations,
        "summaries": [case[2] for case in cases],
        "semantics": {
            "trace": "Independent weighted draws with replacement, without per-user caps or quotas",
            "identities": "Synthetic IDs; rank probabilities approximated from the cumulative curve",
            "time": "Synthetic request index only; no waiting or actual request-rate measurement",
            "content": "User access sequence only; no model token content or GPU execution",
            "industrial_provenance": "Archived synthetic population; not verified production traffic",
            "histogram": "Visit counts include first visits; all pool users including zero visits",
            "top10pct": "ceil(N * 0.1) users ranked by input probability, not by sampled counts",
            "quantiles": "All pool users including zeros; linear interpolation at (N - 1) * q",
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
        },
        "artifact_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.output_dir.iterdir())
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["summaries"], indent=2))


if __name__ == "__main__":
    main()
