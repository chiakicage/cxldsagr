"""Paired dense scheduling comparison with unchanged math, inputs and budgets."""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

from experiments.gr_cache_serving.src.fixed_budget import PROFILE
from experiments.gr_cache_serving.src.report_capacity import load_capacity_runs
from experiments.gr_cache_serving.src.workload import _publish_directory

LABELS = {
    "layer_end": "Layer-end synchronous gather",
    "attention_window": "Attention-window worker",
}


def load_pairs(paths):
    runs = [load_capacity_runs([path], allow_partial=True)[0] for path in paths]
    points = {}
    for path, summary, rows in runs:
        schedule = summary["runner_provenance"].get("dense_prefetch", {}).get("schedule")
        key = (summary["workload_population_users"], schedule)
        if (
            summary["mode"] != "dense_prefetch"
            or summary["runner_provenance"]
            .get("dense_prefetch", {})
            .get("transport", "cpu_staging")
            != "cpu_staging"
            or summary.get("experiment_profile") != PROFILE
            or schedule not in LABELS
            or key in points
        ):
            raise ValueError("expected unique fixed-budget dense schedule pairs")
        points[key] = (path, summary, rows)
        reference = runs[0][1]
        for field in (
            "source_sha256",
            "measurement",
            "gpu",
            "environment",
            "budget_plan",
            "host_budget_plan",
            "pool_accounting",
            "transfer_instrumentation",
        ):
            if summary[field] != reference[field]:
                raise ValueError(f"paired runs differ in {field}")
        for field in ("checkpoint_metadata_sha256", "echo_revision", "echo_patch_sha256"):
            if summary["runner_provenance"][field] != reference["runner_provenance"][field]:
                raise ValueError(f"paired model provenance differs in {field}")
    if set(points) != {(n, s) for n in (64, 128) for s in LABELS}:
        raise ValueError("report requires both schedules at 64 and 128 users")
    for n in (64, 128):
        old, new = (points[n, s] for s in LABELS)
        if old[1]["trace_metadata"] != new[1]["trace_metadata"]:
            raise ValueError("paired input metadata differs")
        fields = (
            "ordinal",
            "user_id",
            "first_visit",
            "prefix_reused",
            "evicted_user_ids",
            "retained_prefix_tokens",
            "d2h_mla_payload_bytes",
        )
        outcomes = lambda rows, fields=fields: [[r[f] for f in fields] for r in rows]
        if outcomes(old[2]) != outcomes(new[2]):
            raise ValueError("paired request sequence or cache lifecycle differs")
    return [points[key] for key in sorted(points)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists")
    runs = load_pairs(args.runs)
    table = []
    for path, s, _ in runs:
        table.append(
            {
                "run_id": path.name,
                "population": s["workload_population_users"],
                "schedule": s["runner_provenance"]["dense_prefetch"]["schedule"],
                "requests": s["requests"],
                "visited_users": s["users"],
                "first_count": s["first_visit"]["count"],
                "revisit_count": s["revisit"]["count"],
                "first_mean_ms": s["first_visit"]["mean_ms"],
                "first_p50_ms": s["first_visit"]["p50_ms"],
                "first_p95_ms": s["first_visit"]["p95_ms"],
                "revisit_mean_ms": s["revisit"]["mean_ms"],
                "revisit_p50_ms": s["revisit"]["p50_ms"],
                "revisit_p95_ms": s["revisit"]["p95_ms"],
                "all_service_s": s["all"]["sum_ms"] / 1000,
                "prefill_total_s": s["prefill_total_ms"] / 1000,
                "revisit_reprefills": s["revisit_reprefill"]["count"],
                "history_hit_fraction": s["revisit_prefix_hit_fraction"],
                "prefix_evictions": s["host_evictions"],
                "nvml_peak_gib": s["memory_guard"]["peak_sampled_device_bytes"] / 2**30,
                "pinned_peak_gib": s["host_memory_observation"]["peak_pinned_reserved_bytes"]
                / 2**30,
                "process_peak_rss_gib": s["host_memory_observation"]["peak_process_rss_bytes"]
                / 2**30,
            }
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("first_mean_ms", "revisit_mean_ms", "all_service_s"),
        ("First visit mean (ms)", "All revisit mean (ms)", "512-request service sum (s)"),
        strict=True,
    ):
        for i, (schedule, label) in enumerate(LABELS.items()):
            values = sorted(
                (r for r in table if r["schedule"] == schedule), key=lambda r: r["population"]
            )
            ax.bar(
                [j + (i - 0.5) * 0.36 for j in range(2)],
                [r[metric] for r in values],
                width=0.36,
                label=label,
                color=("#68747d", "#168d82")[i],
            )
        ax.set_xticks([0, 1], ["64", "128"])
        ax.set(xlabel="User population", title=title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Five layers | 64K history + 1K candidate | fixed budgets | one paired run/size",
        fontsize=11,
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dense-report-", dir=args.output_dir.parent) as temp:
        stage = Path(temp) / "report"
        stage.mkdir()
        fig.savefig(stage / "overview.png", dpi=170)
        plt.close(fig)
        with (stage / "summary.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        manifest = {
            "run_ids": [p.name for p, _, _ in runs],
            "summary_sha256": {
                p.name: hashlib.sha256((p / "summary.json").read_bytes()).hexdigest()
                for p, _, _ in runs
            },
            "source_sha256": runs[0][1]["source_sha256"],
            "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits": "One seed/run; native top-k; serial cold-start synthetic GR; no timeline-derived overlap percentage or online throughput claim.",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _publish_directory(stage, args.output_dir)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
