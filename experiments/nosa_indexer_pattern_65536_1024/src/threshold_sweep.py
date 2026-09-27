"""Exhaustively scan sparse/dense coverage thresholds using existing fetch budgets.

Each classification interval is evaluated once. Minimize exposed transfer time
(also compute + exposed transfer), and separately maximize the hidden fraction.
The default attention MFU is half the source MFU; no GPU execution is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from experiments.nosa_indexer_pattern_65536_1024.src.analyze import _write_csv
from experiments.nosa_indexer_pattern_65536_1024.src.overlap import (
    ROOT,
    _hash,
    estimate_overlap,
)


def threshold_regions(coverages_pct):
    """Partition [0, 100]; equality remains dense, so upper bounds are closed."""
    values = sorted(set(coverages_pct))
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
        raise ValueError("Coverage must be finite and between 0 and 100 percent")
    regions = []
    lower = 0.0
    for upper in sorted({*values, 100.0}):
        representative = (lower + upper) / 2
        if regions and representative <= lower:
            # Adjacent floats can round the midpoint onto the excluded boundary.
            representative = upper
        regions.append(
            {
                "region_id": len(regions),
                "lower_pct": lower,
                "upper_pct": upper,
                "lower_inclusive": not regions,
                "upper_inclusive": True,
                "representative_threshold_pct": representative,
            }
        )
        lower = upper
    return regions


def sweep_thresholds(
    source_report, *, bandwidth_gbps=50.0, attention_mfu_scale=0.5, baseline_threshold_pct=30.0
):
    def evaluate(threshold):
        return estimate_overlap(
            source_report,
            threshold_pct=threshold,
            bandwidth_gbps=bandwidth_gbps,
            attention_mfu_scale=attention_mfu_scale,
        )

    baseline = evaluate(baseline_threshold_pct)
    regions = []
    for region in threshold_regions([head["coverage_pct"] for head in baseline["heads"]]):
        report = evaluate(region["representative_threshold_pct"])
        regions.append({**region, **report["totals"]})

    def best_ids(metric, choose):
        available = [row[metric] for row in regions if row[metric] is not None]
        if not available:
            return []
        best = choose(available)
        return [
            row["region_id"]
            for row in regions
            if row[metric] is not None
            and math.isclose(row[metric], best, abs_tol=1e-12, rel_tol=1e-12)
        ]

    min_unhidden = best_ids("unhidden_ms", min)
    max_efficiency = best_ids("overlap_efficiency", max)
    selected = [{"objective": "baseline", **baseline}]
    for objective, ids in (("min_unhidden", min_unhidden), ("max_efficiency", max_efficiency)):
        for region_id in ids:
            region = regions[region_id]
            selected.append(
                {
                    "objective": objective,
                    "region_id": region_id,
                    **evaluate(region["representative_threshold_pct"]),
                }
            )
    return {
        "parameters": {
            **source_report["parameters"],
            "bandwidth_gbps": bandwidth_gbps,
            "bytes_per_second": bandwidth_gbps * 1e9,
            "attention_mfu_scale": attention_mfu_scale,
            "baseline_threshold_pct": baseline_threshold_pct,
        },
        "regions": regions,
        "best_unhidden_region_ids": min_unhidden,
        "best_efficiency_region_ids": max_efficiency,
        "samples": [
            {"threshold_pct": threshold, **evaluate(threshold)["totals"]}
            for threshold in sorted({*range(101), baseline_threshold_pct})
        ],
        "selected_reports": selected,
    }


def interval_label(region):
    left = "[" if region["lower_inclusive"] else "("
    return f"{left}{region['lower_pct']:.6f}, {region['upper_pct']:.6f}]%"


def report_markdown(reports):
    lines = [
        "# Sparse/dense threshold sweep",
        "",
        "Sparse iff combined union/full KV coverage < threshold; equality is dense.",
        "Sparse fetches its union during attention; dense fetches full KV during the previous layer's other work.",
        "Attention FLOPs are unchanged by classification. Layer 0 has no dense prefetch window.",
        "All distinct classification intervals in [0, 100]% are evaluated, including ties and endpoints.",
        "Time objective: minimize unhidden fetch; compute is fixed, so compute + unhidden has the same optimum.",
        "Efficiency objective: maximize total hidden / total fetch. Its denominator varies with the threshold.",
        "Intervals are open on the left except at zero; printed bounds are rounded to six decimals.",
        "Ideal window budgets on one request, not measured DMA or end-to-end latency.",
        "",
        "| Blocks | Objective | Threshold / interval | Sparse / dense | Fetch ms | Unhidden ms | Overall overlap | Sparse overlap | Dense overlap | Compute + unhidden ms |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def percent(value):
        return "N/A" if value is None else f"{100 * value:.4f}%"

    for report in reports:
        for selection in report["selected_reports"]:
            totals = selection["totals"]
            threshold = (
                interval_label(report["regions"][selection["region_id"]])
                if "region_id" in selection
                else f"{selection['parameters']['threshold_pct']:g}%"
            )
            lines.append(
                f"| {report['parameters']['selection_slots']} | {selection['objective']} | {threshold} | "
                f"{totals['sparse_head_count']} / {totals['dense_head_count']} | "
                f"{totals['fetch_ms']:.6f} | {totals['unhidden_ms']:.6f} | "
                f"{percent(totals['overlap_efficiency'])} | {percent(totals['sparse_overlap_efficiency'])} | "
                f"{percent(totals['dense_overlap_efficiency'])} | {totals['compute_plus_unhidden_ms']:.6f} |"
            )
    return "\n".join(lines) + "\n"


def plot_sweep(output_dir, reports):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(reports), 2, figsize=(13, 4 * len(reports)), squeeze=False, constrained_layout=True
    )
    for index, report in enumerate(reports):
        rows = report["regions"]
        baseline = report["selected_reports"][0]
        baseline_pct = baseline["parameters"]["threshold_pct"]
        for axis, metric, scale, ylabel in (
            (axes[index, 0], "unhidden_ms", 1, "Unhidden fetch (ms, 32-layer sum)"),
            (axes[index, 1], "overlap_efficiency", 100, "Hidden / total fetch (%)"),
        ):
            # Each interval ends at x[i]; 'pre' keeps equality in the dense class.
            x = [0, *[row["upper_pct"] for row in rows]]
            y = [rows[0][metric], *[row[metric] for row in rows]]
            axis.step(
                x,
                [value * scale if value is not None else math.nan for value in y],
                where="pre",
                color="#17679a",
                linewidth=1.8,
            )
            axis.axvline(
                baseline_pct, color="#666666", linestyle="--", label=f"{baseline_pct:g}% baseline"
            )
            baseline_value = baseline["totals"][metric]
            if baseline_value is not None:
                axis.plot(baseline_pct, baseline_value * scale, "o", color="#666666")
            for ids, color, label in (
                (report["best_unhidden_region_ids"], "#2c9559", "Minimum unhidden fetch"),
                (report["best_efficiency_region_ids"], "#d88421", "Maximum hidden fraction"),
            ):
                for position, region_id in enumerate(ids):
                    region = rows[region_id]
                    axis.axvspan(
                        region["lower_pct"],
                        region["upper_pct"],
                        alpha=0.18,
                        color=color,
                        label=label if position == 0 else None,
                    )
                    value = region[metric]
                    if value is not None:
                        axis.plot(
                            region["representative_threshold_pct"],
                            value * scale,
                            "*",
                            color=color,
                            ms=10,
                        )
            axis.set(
                xlim=(0, 100),
                xlabel="Sparse/dense coverage threshold (%)",
                ylabel=ylabel,
                title=f"{report['parameters']['selection_slots']} blocks/query/head",
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, loc="best")
        axes[index, 0].set_ylim(bottom=0)
        axes[index, 1].set_ylim(0, 100)
    params = reports[0]["parameters"]
    figure.suptitle(
        f"NOSA threshold sweep | attention MFU scale {params['attention_mfu_scale']:g}, "
        f"shared {params['bandwidth_gbps']:g} GB/s link\n"
        "Dense-class heads fetch full KV; both classes keep sparse attention FLOPs | ideal window estimate"
    )
    for extension in ("png", "svg"):
        figure.savefig(output_dir / f"threshold_sweep.{extension}", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--estimate-data-dir",
        type=Path,
        action="append",
        required=True,
        help="Current valid estimate run; repeat to compare explicit inputs",
    )
    parser.add_argument("--baseline-threshold-pct", type=float, default=30.0)
    parser.add_argument("--bandwidth-gbps", type=float, default=50.0)
    parser.add_argument("--attention-mfu-scale", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    inputs = [path / "estimates.json" for path in args.estimate_data_dir]
    reports = []
    for path in inputs:
        source = json.loads(path.read_text())
        report = sweep_thresholds(
            source,
            baseline_threshold_pct=args.baseline_threshold_pct,
            bandwidth_gbps=args.bandwidth_gbps,
            attention_mfu_scale=args.attention_mfu_scale,
        )
        report["source_estimate_run_id"] = path.parent.name
        report["source_provenance"] = source["provenance"]
        reports.append(report)
    sources = [
        Path(__file__),
        Path(__file__).with_name("overlap.py"),
        Path(__file__).with_name("analyze.py"),
        Path(__file__).parent.parent / "scripts/threshold_sweep.sh",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    output = {
        "kind": "exhaustive_sparse_dense_threshold_window_budget",
        "assumptions": {
            "classification": "combined union/full < threshold is sparse; equality is dense; labels assumed ready",
            "fetch": "sparse union during current attention; dense full KV during previous-layer other work",
            "windows": "one shared link, no cross-layer window pooling; layer 0 dense fetch is exposed",
            "compute": "original sparse attention FLOPs at scaled MFU for both classes; other compute unchanged",
            "objectives": "minimize unhidden fetch (equivalent to compute + unhidden); separately maximize hidden/fetch",
            "scope": "offline tuning on one request, not executed DMA; excludes indexer, startup, first-tile waits and contention",
        },
        "configurations": reports,
        "provenance": {
            "run_id": args.run_id,
            "computed_at_utc": datetime.now(UTC).isoformat(),
            "analysis_host": platform.node(),
            "analysis_platform": platform.platform(),
            "python": platform.python_version(),
            "matplotlib": version("matplotlib"),
            "input_sha256": {str(path.resolve()): _hash(path) for path in inputs},
            "source_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in sources},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "threshold_sweep.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "threshold_sweep.md").write_text(report_markdown(reports))
    for key in ("regions", "samples"):
        _write_csv(
            args.output_dir / f"threshold_{key}.csv",
            [
                {
                    "estimate_run_id": report["source_estimate_run_id"],
                    "block_budget": report["parameters"]["selection_slots"],
                    **row,
                }
                for report in reports
                for row in report[key]
            ],
        )
    for key in ("layers", "heads"):
        _write_csv(
            args.output_dir / f"selected_{key}.csv",
            [
                {
                    "estimate_run_id": report["source_estimate_run_id"],
                    "block_budget": report["parameters"]["selection_slots"],
                    "objective": selected["objective"],
                    "threshold_pct": selected["parameters"]["threshold_pct"],
                    **row,
                }
                for report in reports
                for selected in report["selected_reports"]
                for row in selected[key]
            ],
        )
    plot_sweep(args.output_dir, reports)
    for source in sources:
        target = args.output_dir / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for index, path in enumerate(inputs):
        target = args.output_dir / "inputs" / f"{index}_{path.parent.name}" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    print(report_markdown(reports))


if __name__ == "__main__":
    main()
