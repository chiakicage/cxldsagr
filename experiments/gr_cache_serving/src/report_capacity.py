"""Compare capacity-limited runs without conditioning revisit latency on hits."""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

from experiments.gr_cache_serving.src.fixed_budget import PROFILE, validate_outcomes, validate_trace
from experiments.gr_cache_serving.src.host_memory import plan_host_cache
from experiments.gr_cache_serving.src.replay import MODES, summarize
from experiments.gr_cache_serving.src.report_replay import COLORS
from experiments.gr_cache_serving.src.workload import _publish_directory

LABELS = {
    "resident": "HBM-only user LRU",
    "sparse_sync": "DSA (blocking on-demand fetch)",
    "echo_gr_adapted": "ECHO (GR + fixes)",
    "dense_prefetch": "Dense prefetch",
}


def validate_fixed_summary(summary, rows):
    validate_trace(summary["trace_metadata"])
    resident = summary["mode"] == "resident"
    capacity = 93 if resident else 128
    validate_outcomes(rows, capacity)
    plan, host, observation = (
        summary["budget_plan"],
        summary["host_budget_plan"],
        summary["host_memory_observation"],
    )
    expected_host = plan_host_cache(
        users=128, layers=5, prefix=65536, suffix=1024, budget_bytes=66 * 2**30
    )
    expected_tokens = capacity * 65536 + 1088
    pool = summary["pool_accounting"]
    if (
        summary["requests"] != 512
        or summary["host_users_capacity"] != 128
        or summary["host_cache_capacity_users"] != (0 if resident else 128)
        or summary["workload_population_users"]
        != summary["trace_metadata"]["stats"]["population_users"]
        or summary["retained_users_capacity"] != capacity
        or plan["retained_users_capacity"] != capacity
        or plan["logical_pool_tokens"] != expected_tokens
        or plan["device_cache_tokens"] != (expected_tokens if resident else 66624)
        or (
            plan["total_hbm_bytes"],
            plan["non_torch_reserve_bytes"],
            plan["workspace_reserve_bytes"],
        )
        != (72 * 2**30, 2 * 2**30, 8 * 2**30)
        or host != expected_host
        or pool["host_mla_buffers_bytes"] != (0 if resident else host["main_kv_tensor_bytes"])
        or summary["transfer_instrumentation"]["enabled"]
        or not summary["host_availability_preflight"]
        or observation["samples"] < 513
        or observation["peak_process_rss_bytes"] > 98 * 2**30
        or max(0, observation["peak_pinned_reserved_bytes"] - host["max_pinned_reserved_bytes"])
        + observation["charged_cpu_metadata_bytes"]
        > host["metadata_reserve_bytes"]
        or observation["peak_pinned_reserved_bytes"]
        < (0 if resident else host["main_kv_rounded_pinned_bytes"])
    ):
        raise ValueError(
            "fixed-cache budget, actual pool or Host observation differs from protocol"
        )


def load_capacity_runs(paths, *, allow_partial=False):
    runs, points = [], {}
    for path in paths:
        summary = json.loads((path / "summary.json").read_text())
        rows = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
        metadata = summary["trace_metadata"]
        if (
            summary["artifact_type"] != "synchronous_gr_prototype_replay"
            or summary["topk_order"] != "native; no correctness-test sorting hooks"
            or summary["runner_provenance"]["num_layers"] != 5
            or metadata["stable_prefix_tokens"] != 65536
            or metadata["candidate_suffix_tokens"] != 1024
            or len(rows) != summary["requests"]
            or len(rows) != metadata["stats"]["requests"]
            or (
                not summary.get("transfer_instrumentation", {}).get("enabled")
                and summary.get("experiment_profile") != PROFILE
            )
            or not summary.get("budget_plan")
            or not summary.get("memory_guard")
        ):
            raise ValueError("expected complete, native-order, budget-guarded five-layer runs")
        recomputed = summarize(rows)
        if any(summary[key] != value for key, value in recomputed.items()):
            raise ValueError("summary statistics do not match the request rows")
        if summary.get("experiment_profile") == PROFILE:
            validate_fixed_summary(summary, rows)
        guard = summary["memory_guard"]
        if not 0 < guard["peak_sampled_device_bytes"] <= guard["common_total_hbm_limit_bytes"]:
            raise ValueError("invalid or over-budget memory observation")
        source = json.loads((path / "source_snapshot.json").read_text())
        for name, sha in summary["source_sha256"].items():
            if hashlib.sha256(source[name]["text"].encode()).hexdigest() != sha:
                raise ValueError("measured source snapshot does not match fingerprints")
        population = metadata["stats"].get("population_users", summary["host_users_capacity"])
        mode = summary["mode"]
        if mode not in MODES or (population, mode) in points:
            raise ValueError("unknown mode or duplicate population/mode point")
        points[population, mode] = (path, summary, rows)
        if runs:
            first = runs[0][1]
            if summary.get("experiment_profile") != first.get("experiment_profile"):
                raise ValueError("different experiment profiles")
            for key in ("source_sha256", "measurement", "gpu", "transfer_instrumentation"):
                if summary[key] != first[key]:
                    raise ValueError("source, timing, hardware or instrumentation differs")
            for key in ("total_hbm_bytes", "non_torch_reserve_bytes", "workspace_reserve_bytes"):
                if summary["budget_plan"][key] != first["budget_plan"][key]:
                    raise ValueError("common memory ceiling or reservation differs")
            if (
                summary["runner_provenance"]["checkpoint_metadata_sha256"]
                != first["runner_provenance"]["checkpoint_metadata_sha256"]
            ):
                raise ValueError("checkpoint differs")
        runs.append((path, summary, rows))
    if not runs:
        raise ValueError("no runs")
    dense_schedules = {
        summary["runner_provenance"].get("dense_prefetch", {}).get("schedule", "layer_end")
        for _, summary, _ in runs
        if summary["mode"] == "dense_prefetch"
    }
    if len(dense_schedules) > 1:
        raise ValueError("different dense schedules; use the paired dense report")
    dense_transports = {
        summary["runner_provenance"].get("dense_prefetch", {}).get("transport", "cpu_staging")
        for _, summary, _ in runs
        if summary["mode"] == "dense_prefetch"
    }
    if len(dense_transports) > 1:
        raise ValueError("different dense transports cannot share a capacity curve")
    for population in {key[0] for key in points}:
        present = [mode for mode in MODES if (population, mode) in points]
        if len(present) != len(MODES) and not allow_partial:
            raise ValueError("every reported population needs all four modes")
        reference = points[population, present[0]]
        for mode in present:
            _, summary, rows = points[population, mode]
            if (
                summary["trace_metadata"]["requests_sha256"]
                != reference[1]["trace_metadata"]["requests_sha256"]
            ):
                raise ValueError("different traces at the same population")
            # Cache outcomes MUST be allowed to differ in a capacity comparison.
            if [(r["ordinal"], r["user_id"], r["first_visit"]) for r in rows] != [
                (r["ordinal"], r["user_id"], r["first_visit"]) for r in reference[2]
            ]:
                raise ValueError("request sequence differs")
        offload = [points[population, mode] for mode in present if mode != "resident"]
        if reference[1].get("experiment_profile") == PROFILE and offload:
            outcomes = lambda rows: [
                (row["prefix_reused"], row["evicted_user_ids"], row["retained_prefix_tokens"])
                for row in rows
            ]
            if any(outcomes(run[2]) != outcomes(offload[0][2]) for run in offload[1:]):
                raise ValueError("offload Host LRU outcomes differ")
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("report already exists")
    runs = load_capacity_runs(args.runs, allow_partial=args.allow_partial)
    table = []
    for path, summary, rows in runs:
        table.append(
            {
                "run_id": path.name,
                "mode": summary["mode"],
                "population": summary["trace_metadata"]["stats"]["population_users"],
                "visited_users": summary["users"],
                "requests": summary["requests"],
                "retained_user_capacity": summary["retained_users_capacity"],
                "all_mean_ms": summary["all"]["mean_ms"],
                "revisit_mean_ms": summary["revisit"]["mean_ms"],
                "revisit_p50_ms": summary["revisit"]["p50_ms"],
                "revisit_p95_ms": summary["revisit"]["p95_ms"],
                "revisit_p99_ms": summary["revisit"]["p99_ms"],
                "revisit_hit_fraction": summary["revisit_prefix_hit_fraction"],
                "revisit_reprefills": summary["revisit_reprefill"]["count"],
                "prefill_total_s": summary["prefill_total_ms"] / 1000,
                "all_service_s": summary["all"]["sum_ms"] / 1000,
                "h2d_mla_gib": sum(row["h2d_mla_payload_bytes"] for row in rows) / 2**30
                if summary["transfer_instrumentation"]["enabled"]
                else None,
                "h2d_mla_mib_per_request": sum(row["h2d_mla_payload_bytes"] for row in rows)
                / len(rows)
                / 2**20
                if summary["transfer_instrumentation"]["enabled"]
                else None,
                "d2h_mla_gib": sum(row["d2h_mla_payload_bytes"] for row in rows) / 2**30,
                "peak_sampled_nvml_gib": summary["memory_guard"]["peak_sampled_device_bytes"]
                / 2**30,
                "peak_torch_allocated_gib": summary["peak_torch_allocated_bytes"] / 2**30,
                "peak_torch_reserved_gib": summary["peak_torch_reserved_bytes"] / 2**30,
                "first_visit_p50_ms": summary["first_visit"]["p50_ms"],
                "first_visit_mean_ms": summary["first_visit"]["mean_ms"],
                "first_visit_p95_ms": summary["first_visit"]["p95_ms"],
                "dense_transport": summary["runner_provenance"]
                .get("dense_prefetch", {})
                .get("transport", "cpu_staging")
                if summary["mode"] == "dense_prefetch"
                else None,
                "prefill_count": summary["prefill_count"],
                "prefix_evictions": summary["host_evictions"],
                "host_cache_capacity_users": summary.get("host_cache_capacity_users"),
                "peak_pinned_reserved_gib": summary.get("host_memory_observation", {}).get(
                    "peak_pinned_reserved_bytes", 0
                )
                / 2**30
                if summary.get("host_memory_observation")
                else None,
                "peak_cpu_cache_metadata_mib": summary["host_memory_observation"][
                    "charged_cpu_metadata_bytes"
                ]
                / 2**20
                if summary.get("host_memory_observation")
                else None,
                "process_peak_rss_gib": summary["process_max_rss_kib"] / 2**20,
            }
        )
    table.sort(key=lambda row: (row["population"], row["mode"]))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    fixed = runs[0][1].get("experiment_profile") == PROFILE
    panels = (
        ("revisit_p50_ms", "All revisit p50 (ms), including re-prefill")
        if fixed
        else ("all_mean_ms", "All request mean (ms), including cold first visits"),
        ("revisit_mean_ms", "All revisit mean (ms), including re-prefill"),
        ("revisit_p95_ms", "All revisit p95 (ms), including re-prefill"),
        ("revisit_hit_fraction", "History reuse fraction among all revisits"),
        ("peak_sampled_nvml_gib", "Peak sampled device memory (GiB)"),
        ("revisit_reprefills", "Revisit re-prefill count")
        if fixed
        else ("h2d_mla_mib_per_request", "Mean H2D MLA payload (MiB/request), all requests"),
    )
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for mode in MODES:
            values = [row for row in table if row["mode"] == mode]
            if not values:
                continue
            axis.plot(
                [row["population"] for row in values],
                [row[metric] for row in values],
                marker="o",
                color=COLORS[mode],
                label=LABELS[mode],
            )
        axis.set_xlabel("User population (actual visited counts in CSV)")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    budget_gib = runs[0][1]["budget_plan"]["total_hbm_bytes"] / 2**30
    axes[2, 0].axhline(budget_gib, color="black", linestyle="--", linewidth=1)
    fig.suptitle(
        f"DeepSeek first 5 layers | 64K history + 1K candidate | {budget_gib:g} GiB ceiling\n"
        + (
            "512 requests/size; fixed HBM/Host caches; one seed/run; not equal allocations"
            if fixed
            else "Serial cold-start GR replay; one seed/run; fixed small offload pool, not equal allocations"
        ),
        fontsize=11,
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".capacity-report-", dir=args.output_dir.parent
    ) as temp:
        stage = Path(temp) / "report"
        stage.mkdir()
        fig.savefig(stage / "overview.png", dpi=170)
        plt.close(fig)
        service_fig, service_axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for axis, metric, title in zip(
            service_axes,
            ("first_visit_mean_ms", "revisit_mean_ms", "all_service_s"),
            ("First visit mean (ms)", "All revisit mean (ms)", "Cumulative service time (s)"),
            strict=True,
        ):
            for mode in MODES:
                values = [row for row in table if row["mode"] == mode]
                if values:
                    axis.plot(
                        [r["population"] for r in values],
                        [r[metric] for r in values],
                        marker="o",
                        color=COLORS[mode],
                        label=LABELS[mode],
                    )
            axis.set(xlabel="User population", title=title)
            axis.grid(alpha=0.2)
        service_axes[0].legend(fontsize=8)
        service_fig.savefig(stage / "service_times.png", dpi=170)
        plt.close(service_fig)
        if fixed:
            for population in sorted({row["population"] for row in table}):
                detail, panels = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
                for _, summary, rows in runs:
                    if summary["workload_population_users"] != population:
                        continue
                    mode = summary["mode"]
                    values = sorted(row["service_ms"] for row in rows if not row["first_visit"])
                    panels[0].plot(
                        values,
                        [(i + 1) / len(values) for i in range(len(values))],
                        label=LABELS[mode],
                        color=COLORS[mode],
                    )
                    panels[1].plot(
                        [row["ordinal"] for row in rows],
                        [row["service_ms"] for row in rows],
                        label=LABELS[mode],
                        color=COLORS[mode],
                        alpha=0.75,
                        linewidth=0.7,
                    )
                panels[0].set(xlabel="All revisit latency (ms), including re-prefill", ylabel="CDF")
                panels[1].set(xlabel="Request ordinal", ylabel="Service latency (ms)")
                panels[0].legend(fontsize=7)
                detail.suptitle(f"Population {population} | 512 requests | fixed caches")
                detail.savefig(stage / f"u{population}_latency.png", dpi=160)
                plt.close(detail)
        with (stage / "summary.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        manifest = {
            "experiment_profile": runs[0][1].get("experiment_profile"),
            "partial_matrix": args.allow_partial,
            "run_ids": [path.name for path, _, _ in runs],
            "summary_sha256": {
                path.name: hashlib.sha256((path / "summary.json").read_bytes()).hexdigest()
                for path, _, _ in runs
            },
            "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits": "truncated model; one cold-start synthetic trace per size; no queue, quality, optimized pool allocation or overlap evidence; NVML sampling misses sub-interval peaks",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _publish_directory(stage, args.output_dir)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
