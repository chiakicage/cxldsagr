"""Render completed synchronous-prototype replays; never infer online SLO capacity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

LABELS = {
    "resident": "Resident (capacity-rich reference)",
    "sparse_sync": "Blocking sparse fetch",
    "echo_gr_adapted": "ECHO (GR + fixes)",
    "dense_prefetch": "Dense prefetch (+ staging)",
}
COLORS = {
    "resident": "#68717b",
    "sparse_sync": "#c44747",
    "echo_gr_adapted": "#168578",
    "dense_prefetch": "#4263b5",
}
SHORT_LABELS = {
    "resident": "Resident\n(full GPU)",
    "sparse_sync": "Blocking\nsparse fetch",
    "echo_gr_adapted": "ECHO\n(GR + fixes)",
    "dense_prefetch": "Dense\nprefetch",
}


def load_runs(paths):
    runs = []
    for path in paths:
        summary = json.loads((path / "summary.json").read_text())
        if summary["artifact_type"] != "synchronous_gr_prototype_replay":
            raise ValueError("diagnostics are not performance results")
        rows = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
        if (
            len(rows) != summary["requests"]
            or len(rows) != summary["trace_metadata"]["stats"]["requests"]
        ):
            raise ValueError("only complete, count-verified replays can be reported")
        if summary["topk_order"] != "native; no correctness-test sorting hooks":
            raise ValueError("test-controlled top-k timings cannot be included")
        runs.append((path, summary, rows))
    if not runs:
        raise ValueError("at least one completed run is required")
    first = runs[0][1]
    keys = ("source_sha256", "measurement", "host_users_capacity", "configured_device_cache_tokens")
    for _, summary, rows in runs:
        if any(summary[key] != first[key] for key in keys):
            raise ValueError("measurement source, timing boundary or configured pools differ")
        if (
            summary["trace_metadata"]["requests_sha256"]
            != first["trace_metadata"]["requests_sha256"]
        ):
            raise ValueError("trace differs across runs")
        if [(r["ordinal"], r["user_id"], r["first_visit"], r["prefix_reused"]) for r in rows] != [
            (r["ordinal"], r["user_id"], r["first_visit"], r["prefix_reused"]) for r in runs[0][2]
        ]:
            raise ValueError("request order or prefix cache outcomes differ")
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("report output already exists")
    runs = load_runs(args.runs)
    root = Path(__file__).resolve().parents[3]
    existing_snapshot = runs[0][0] / "source_snapshot.json"
    snapshot = json.loads(existing_snapshot.read_text()) if existing_snapshot.exists() else {}
    use_saved_snapshot = bool(snapshot)
    for name, expected in runs[0][1]["source_sha256"].items():
        source = (
            snapshot[name]["text"].encode() if use_saved_snapshot else (root / name).read_bytes()
        )
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"source changed before archiving the measured snapshot: {name}")
        snapshot[name] = {"sha256": expected, "text": source.decode()}
    for path, _, _ in runs:
        destination = path / "source_snapshot.json"
        if destination.exists():
            if json.loads(destination.read_text()) != snapshot:
                raise ValueError("existing source snapshot differs")
        else:
            with destination.open("x") as handle:
                json.dump(snapshot, handle, indent=2)
    grouped = {mode: [run for run in runs if run[1]["mode"] == mode] for mode in LABELS}
    grouped = {mode: values for mode, values in grouped.items() if values}
    table = []
    for mode, values in grouped.items():
        row = {
            "mode": mode,
            "repetitions": len(values),
            "requests_per_run": values[0][1]["requests"],
        }
        metrics = {
            "first_visit_p50_ms": lambda s: s["first_visit"]["p50_ms"],
            "prefix_hit_p50_ms": lambda s: s["prefix_hit"]["p50_ms"],
            "prefix_hit_p95_ms": lambda s: s["prefix_hit"]["p95_ms"],
            "prefix_hit_p99_ms": lambda s: s["prefix_hit"]["p99_ms"],
            "all_mean_ms": lambda s: s["all"]["mean_ms"],
            "total_service_s": lambda s: s["all"]["sum_ms"] / 1000,
            "inverse_mean_service_rps": lambda s: s["inverse_mean_service_requests_per_s"],
            "peak_torch_allocated_gib": lambda s: s["peak_torch_allocated_bytes"] / 2**30,
        }
        for name, extract in metrics.items():
            numbers = [extract(value[1]) for value in values]
            row[name] = statistics.median(numbers)
            row[name + "_min"] = min(numbers)
            row[name + "_max"] = max(numbers)
        table.append(row)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for index, row in enumerate(table):
        mode = row["mode"]
        for offset, metric, alpha in (
            (-0.18, "prefix_hit_p50_ms", 1.0),
            (0.18, "prefix_hit_p95_ms", 0.45),
        ):
            value = row[metric]
            axes[0, 0].bar(index + offset, value, width=0.34, color=COLORS[mode], alpha=alpha)
            axes[0, 0].errorbar(
                index + offset,
                value,
                yerr=[[value - row[metric + "_min"]], [row[metric + "_max"] - value]],
                color="black",
                capsize=3,
                fmt="none",
            )
    axes[0, 0].set_xticks(range(len(table)), [SHORT_LABELS[row["mode"]] for row in table])
    axes[0, 0].set_ylabel("Request service time (ms)")
    axes[0, 0].set_title(
        "Prefix reuse: p50 (solid), p95 (light)\nMedian across runs; whiskers are run min/max"
    )

    for mode, values in grouped.items():
        # CDF pools full request observations; the CSV reports per-run quantiles separately.
        times = sorted(r["service_ms"] for _, _, rows in values for r in rows if r["prefix_reused"])
        axes[0, 1].plot(
            times,
            [(i + 1) / len(times) for i in range(len(times))],
            color=COLORS[mode],
            label=LABELS[mode],
        )
    axes[0, 1].set_xlabel("Prefix-hit service time (ms)")
    axes[0, 1].set_ylabel("Cumulative fraction")
    axes[0, 1].set_title("Reuse-request latency CDF (all repetitions)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    for mode, values in grouped.items():
        requests = values[0][2]
        medians = [
            statistics.median(run[2][i]["service_ms"] for run in values)
            for i in range(len(requests))
        ]
        axes[1, 0].plot(range(len(requests)), medians, color=COLORS[mode], linewidth=0.8, alpha=0.7)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Trace request ordinal")
    axes[1, 0].set_ylabel("Service time (ms, logarithmic)")
    axes[1, 0].set_title("Entire cold-start trace, including first-user prefill")
    axes[1, 0].grid(alpha=0.2)

    for row in table:
        mode = row["mode"]
        axes[1, 1].scatter(
            row["peak_torch_allocated_gib"],
            row["prefix_hit_p50_ms"],
            s=85,
            color=COLORS[mode],
            label=LABELS[mode],
        )
    axes[1, 1].set_xlabel("Peak PyTorch allocated GPU memory (GiB)")
    axes[1, 1].set_ylabel("Prefix-hit p50 (ms)")
    axes[1, 1].set_title("Memory / latency tradeoff (not an equal-budget sweep)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)
    first_summary = runs[0][1]
    metadata = first_summary["trace_metadata"]
    dataset = metadata["generator"]["heat"]["dataset"].title()
    seed = metadata["generator"]["schedule_config"]["seed"]
    counts = sorted({len(values) for values in grouped.values()})
    repetitions = str(counts[0]) if len(counts) == 1 else str(counts)
    fig.suptitle(
        f"{dataset} GR | {first_summary['requests']} requests | "
        f"{metadata['stable_prefix_tokens']} history + {metadata['candidate_suffix_tokens']} candidate | "
        "H100 PCIe | DeepSeek first 3 layers\n"
        f"{repetitions} run(s)/mode, seed {seed} | Synchronous serial prototype; native top-k; no queue/network/generation",
        fontsize=12,
    )
    args.output_dir.mkdir(parents=True)
    fig.savefig(args.output_dir / "overview.png", dpi=170)
    plt.close(fig)
    with (args.output_dir / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    manifest = {
        "run_ids": [path.name for path, _, _ in runs],
        "run_summary_sha256": {
            path.name: hashlib.sha256((path / "summary.json").read_bytes()).hexdigest()
            for path, _, _ in runs
        },
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "aggregation": "median of each run's statistic; min/max across repetitions, not confidence intervals",
        "limits": "one synthetic trace; synchronous three-layer prototype; ECHO patched; no equal-total-HBM enforcement, online arrivals, hit-rate or transfer instrumentation",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
