"""Plot the distribution of combined KV union coverage across all layer/KV heads.

Each layer/KV-head pair contributes one sample. The denominator is that head's
full prefix + candidate cache. Sink/local and query-aware blocks are combined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.nosa_indexer_pattern_65536_1024.src.analyze import _write_csv, summarize

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/nosa_indexer_pattern_65536_1024/output/data"
DEFAULT_PATTERNS = (DATA / "query_aware_fp32_65536_1024_20260925_01",)


def load_distribution(pattern):
    """Revalidate original selections and return one coverage sample per head."""
    metadata = json.loads((pattern / "metadata.json").read_text())
    config, execution = metadata["model_config"], metadata["execution"]
    block_ids = np.load(pattern / "block_ids.npy", allow_pickle=False)
    valid = np.load(pattern / "valid_mask.npy", allow_pickle=False)
    if block_ids.shape != (
        config["num_hidden_layers"],
        execution["new_tokens"],
        config["num_key_value_heads"],
        metadata["indexer_policy"]["block_budget"],
    ):
        raise ValueError("Selection shape disagrees with captured model/policy")
    union, original = summarize(
        block_ids,
        valid,
        prefix_tokens=execution["prefix_tokens"],
        total_tokens=execution["total_tokens"],
        head_dim=config["head_dim"],
        element_size=metadata["element_size"],
        block_size=metadata["block_size"],
    )
    if not np.array_equal(union, np.load(pattern / "union_mask.npy", allow_pickle=False)):
        raise ValueError("Selections do not reproduce the saved combined union")
    budget = original["parameters"]["selection_slots"]
    heads = [
        {
            "pattern_run_id": metadata["run_id"],
            "block_budget": budget,
            "layer": row["layer"],
            "kv_head": row["kv_head"],
            "union_blocks": row["union_blocks"],
            "full_blocks": original["parameters"]["blocks_per_head"],
            "union_bytes": row["union_bytes"],
            "full_kv_bytes": row["kv_cache_bytes"],
            "coverage_pct": row["fraction"] * 100,
            "sparsity_pct": (1 - row["fraction"]) * 100,
        }
        for row in original["head_stats"]
    ]
    coverage = np.array([row["coverage_pct"] for row in heads])
    edges = np.arange(0, 101, 10)
    counts, _ = np.histogram(coverage, bins=edges)
    if counts.sum() != len(heads):
        raise ValueError("Histogram does not account for every layer/KV head")
    quantiles = np.quantile(coverage, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1], method="linear")
    stats = {
        "head_count": len(heads),
        "mean_pct": float(coverage.mean()),
        **dict(
            zip(
                ("min_pct", "p25_pct", "median_pct", "p75_pct", "p90_pct", "p95_pct", "max_pct"),
                quantiles.tolist(),
                strict=True,
            )
        ),
        "mean_sparsity_pct": float(100 - coverage.mean()),
        "heads_le_25_pct": int((coverage <= 25).sum()),
        "heads_le_50_pct": int((coverage <= 50).sum()),
        "heads_gt_75_pct": int((coverage > 75).sum()),
        "model_coverage_pct": original["summary"]["fraction"] * 100,
    }
    histogram = [
        {
            "pattern_run_id": metadata["run_id"],
            "block_budget": budget,
            "lower_pct_inclusive": int(lower),
            "upper_pct": int(upper),
            "upper_inclusive": bool(upper == 100),
            "head_count": int(count),
            "fraction_of_heads": int(count) / len(heads),
        }
        for lower, upper, count in zip(edges[:-1], edges[1:], counts, strict=True)
    ]
    return {
        "pattern_run_id": metadata["run_id"],
        "block_budget": budget,
        "parameters": original["parameters"],
        "coverage_statistics": stats,
        "heads": heads,
        "histogram": histogram,
    }, metadata


def plot_distribution(path, reports):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    figure, (histogram, ecdf) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = ("#17679a", "#db7924", "#538544", "#9964a6")
    width = 8 / len(reports)
    for index, report in enumerate(reports):
        color = colors[index % len(colors)]
        label = f"{report['block_budget']} blocks/query/head"
        centers = np.arange(5, 100, 10) + (index - (len(reports) - 1) / 2) * width
        counts = [row["head_count"] for row in report["histogram"]]
        bars = histogram.bar(centers, counts, width=width, label=label, color=color)
        histogram.bar_label(
            bars, labels=[str(c) if c else "" for c in counts], padding=3, fontsize=9
        )
        coverage = np.sort([row["coverage_pct"] for row in report["heads"]])
        ecdf.step(
            np.r_[0, coverage, 100],
            np.r_[0, np.arange(1, len(coverage) + 1) / len(coverage) * 100, 100],
            where="post",
            color=color,
            label=label,
            linewidth=2,
        )
    for axis in (histogram, ecdf):
        axis.set(xlim=(0, 100), xlabel="Selected KV union / full KV cache per head (%)")
        axis.set_xticks(np.arange(0, 101, 10))
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend(fontsize=9, loc="upper right" if axis is histogram else "lower right")
    histogram.set(
        title="Head counts by coverage (10 percentage-point bins)", ylabel="Number of KV heads"
    )
    histogram.yaxis.set_major_locator(MaxNLocator(integer=True))
    histogram.set_ylim(0, max(row["head_count"] for r in reports for row in r["histogram"]) * 1.18)
    ecdf.set(title="Cumulative distribution", ylabel="Fraction of KV heads", ylim=(0, 103))
    ecdf.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ecdf.axhline(50, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
    params = reports[0]["parameters"]
    figure.suptitle(
        f"NOSA combined KV unions: {params['num_layers']} layers × {params['num_kv_heads']} KV heads per configuration\n"
        f"Sink + local + query-aware, merged across {params['num_queries']:,} queries; lower coverage means sparser"
    )
    for extension in ("png", "svg"):
        figure.savefig(path / f"head_coverage_distribution.{extension}", dpi=180)
    plt.close(figure)


def report_markdown(reports):
    lines = [
        "# Combined KV union coverage distribution across heads",
        "",
        "One sample per (layer, KV head); union includes sink/local and query-aware blocks.",
        "Coverage = combined union bytes / that head's full prefix + candidate KV bytes.",
        "Lower coverage means sparser; sparsity = 100% minus coverage. Quantiles use linear interpolation.",
        "",
        "| Blocks/query/head | Heads | Mean | P25 | Median | P75 | P90 | P95 | Min | Max |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        stats = report["coverage_statistics"]
        values = " | ".join(
            f"{stats[key]:.2f}%"
            for key in (
                "mean_pct",
                "p25_pct",
                "median_pct",
                "p75_pct",
                "p90_pct",
                "p95_pct",
                "min_pct",
                "max_pct",
            )
        )
        lines.append(f"| {report['block_budget']} | {stats['head_count']} | {values} |")
    return "\n".join(lines) + "\n"


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern-data-dir",
        type=Path,
        action="append",
        help="Repeat to compare explicit inputs; defaults to the 64-block baseline",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    patterns = args.pattern_data_dir or DEFAULT_PATTERNS
    loaded = [load_distribution(path) for path in patterns]
    reports, metadata = zip(*loaded, strict=True)
    if len({r["block_budget"] for r in reports}) != len(reports):
        raise ValueError("Compare distinct block budgets; each curve must have one unique budget")
    for path, other in zip(patterns[1:], metadata[1:], strict=True):
        if any(
            other[key] != metadata[0][key]
            for key in ("model_config", "execution", "dtype", "checkpoint_sha256")
        ):
            raise ValueError("Pattern runs must share the model, execution, dtype and checkpoint")
        if (path / "request.json").read_bytes() != (patterns[0] / "request.json").read_bytes():
            raise ValueError("Compared pattern runs must use the exact same request")
    sources = [
        Path(__file__),
        Path(__file__).with_name("analyze.py"),
        Path(__file__).parent.parent / "scripts/distribution.sh",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    inputs = [
        path / name
        for path in patterns
        for name in (
            "metadata.json",
            "request.json",
            "block_ids.npy",
            "valid_mask.npy",
            "union_mask.npy",
        )
    ]
    report = {
        "metric": "combined KV union bytes / full KV bytes of each layer/KV head",
        "sampling": "one sample per layer/KV head within each configuration, equal weight",
        "quantile_method": "linear",
        "histogram_intervals": "[lower, upper), except the final interval includes 100%",
        "configurations": reports,
        "provenance": {
            "run_id": args.run_id,
            "computed_at_utc": datetime.now(UTC).isoformat(),
            "input_sha256": {str(path.resolve()): _hash(path) for path in inputs},
            "source_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in sources},
            "numpy_version": np.__version__,
            "model_executed": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plot_distribution(args.output_dir, reports)
    (args.output_dir / "distribution.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "distribution.md").write_text(report_markdown(reports))
    _write_csv(args.output_dir / "head_coverage.csv", [row for r in reports for row in r["heads"]])
    _write_csv(args.output_dir / "histogram.csv", [row for r in reports for row in r["histogram"]])
    for source in sources:
        target = args.output_dir / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    print(report_markdown(reports))


if __name__ == "__main__":
    main()
