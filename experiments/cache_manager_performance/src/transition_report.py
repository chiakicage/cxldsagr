"""Report only exact-top-k completion to actual MLA compute start."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(windows):
    groups, totals = defaultdict(list), defaultdict(list)
    for row in windows:
        key = tuple(row[x] for x in ("source_kind", "run_id", "scheme", "phase"))
        groups[(*key, str(row["layer"]))].append(row["topk_to_mla_us"])
        totals[(*key, row["sample"])].append(row["topk_to_mla_us"])
    for (*key, _sample), values in totals.items():
        groups[(*key, "all")].append(sum(values))
    rows = []
    for key, values in sorted(groups.items()):
        q1, _, q3 = (
            statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
        )
        rows.append(
            {
                **dict(
                    zip(("source_kind", "run_id", "scheme", "phase", "layer"), key, strict=True)
                ),
                "samples": len(values),
                "median_us": statistics.median(values),
                "q1_us": q1,
                "q3_us": q3,
                "min_us": min(values),
                "max_us": max(values),
            }
        )
    return rows


def render(report, output):
    output = Path(output)
    windows, stages = [], []
    for sample in report["samples"]:
        if sample["boundaries"]["endpoint_kind"] != "attention_kernel":
            raise ValueError("post-top-k report requires a real MLA consumer, not readiness")
        identity = {
            key: sample[key]
            for key in ("source_kind", "run_id", "scheme", "phase", "layer", "sample")
        }
        gpu, cpu = sample["gpu_topk_to_consumer"], sample["cpu_topk_to_consumer"]
        if gpu["compute_union_ms"] != 0 or gpu["fused_union_ms"] != 0:
            raise ValueError("measured post-top-k window contains unexpected model computation")
        windows.append(
            {
                **identity,
                "topk_end_ns": gpu["start_ns"],
                "mla_start_ns": gpu["end_ns"],
                "topk_to_mla_us": gpu["window_ms"] * 1000,
                "io_union_us": gpu["io_union_ms"] * 1000,
                "exposed_control_us": gpu["exposed_control_ms"] * 1000,
                "gpu_idle_us": gpu["gpu_idle_ms"] * 1000,
                "non_io_gap_us": gpu["gap_ms"] * 1000,
                "idle_before_paired_launch_us": sample["post_topk_idle_submission"][
                    "idle_before_next_launch_api_ms"
                ]
                * 1000,
                "cpu_topk_to_wrapper_us": cpu["wall_ms"] * 1000,
                "cpu_api_union_us": cpu["api_union_ms"] * 1000,
                "cpu_non_api_residual_us": cpu["non_api_residual_ms"] * 1000,
                "mla_launch_minus_topk_end_us": sample["boundaries"][
                    "attention_launch_start_minus_topk_end_ms"
                ]
                * 1000,
            }
        )
        for stage in sample["stages"]:
            if stage["stage"] not in {
                "prefetch_hint",
                "cache_write",
                "offload_exact_recall",
                "sparse_mla",
            }:
                raise ValueError("report contains a stage outside the requested post-top-k scope")
            stages.append(
                {
                    **identity,
                    "stage": stage["stage"],
                    "cpu_scope_us": stage["cpu"]["wall_ms"] * 1000,
                    "cpu_api_union_us": stage["cpu"]["api_union_ms"] * 1000,
                    "cpu_non_api_residual_us": stage["cpu"]["non_api_residual_ms"] * 1000,
                    "gpu_busy_in_window_us": stage["gpu_in_post_topk"]["gpu_busy_union_ms"] * 1000,
                    "gpu_io_in_window_us": stage["gpu_in_post_topk"]["io_union_ms"] * 1000,
                    "gpu_control_in_window_us": stage["gpu_in_post_topk"]["gpu_control_union_ms"]
                    * 1000,
                }
            )
    latency = summarize(windows)
    _write_csv(output / "windows.csv", windows)
    _write_csv(output / "stages.csv", stages)
    _write_csv(output / "latency.csv", latency)
    lines = [
        "# Top-k completion to MLA compute start",
        "",
        f"Analysis ID: `{report['analysis_id']}`. Values come from actual NSYS GPU timestamps.",
        "",
        (
            "The sole measured interval starts after the last exact-top-k GPU activity, including invalid-ID masking, "
            "and ends at the first actual MLA compute kernel. Indexer, top-k and MLA computation are outside it. "
            "No stage-duration subtraction or extra mid-path synchronization is used. The interval retains hint updates, "
            "KV append/writeback, exact recall, mapping, GPU control, idle, and MLA preparation/launch delays."
        ),
        "",
        (
            "The independent replay executes the production MLA consumer. Complete-model captures, when supplied, "
            "are a separate workload. These are intrusive profile measurements, not unprofiled wall-clock benchmarks; "
            "profiler overhead remains."
        ),
        "",
    ]
    for kind in sorted({row["source_kind"] for row in latency}):
        selected = [row for row in latency if row["source_kind"] == kind]
        lines.extend(
            [
                f"## {kind}",
                "",
                "Source run: "
                + ", ".join(f"`{x}`" for x in sorted({r["run_id"] for r in selected}))
                + ".",
                "",
                (
                    "Times are µs. `all` first sums the separate layer windows within each sample, then takes the median; "
                    "it is not a complete request duration. Q1/Q3 use inclusive sample quantiles."
                ),
                "",
                "| Method | Phase | Layer | Samples | Median | Q1 | Q3 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            lines.append(
                f"| {row['scheme']} | {row['phase']} | {row['layer']} | {row['samples']} | "
                f"{row['median_us']:.3f} | {row['q1_us']:.3f} | {row['q3_us']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "![Measured post-top-k intervals](transition.svg)",
            "",
            "## Accounting",
            "",
            (
                "`windows.csv` retains every sample's endpoint timestamps, IO union, exposed GPU control, idle, and signed "
                "MLA launch offset. IO + exposed control + idle equals each window. Zero-record gathers remain control; "
                "persistent append D2H is IO even for warm history. `stages.csv` retains only hint, append, recall and MLA "
                "wrapper stages. CPU scope time is a separate submission view and must not be added to GPU execution intervals."
            ),
            "",
            (
                "A negative launch offset means MLA was already submitted when top-k completed on GPU. Idle-before-launch "
                "counts only idle ending at a unique main-stream kernel, before its correlated launch API begins. "
                "Other-stream endings remain unassigned. This does not attribute host residual time to Python, C++ checks, "
                "allocation or scheduling."
            ),
            "",
            (
                "Replay inputs use an append-built/restored prefix and zero initial hint. This differs from the full model's "
                "cache trajectory; differences between workloads are not wrapper-cost estimates. No GR transient candidate, "
                "H>P split query, model speedup or complete-model gap gate is claimed."
            ),
            "",
        ]
    )
    (output / "results.md").write_text("\n".join(lines))
    _plot(windows, output)


def _plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({(row["source_kind"], row["phase"]) for row in rows})
    figure, axes = plt.subplots(len(groups), 1, figsize=(9, 2.8 * len(groups)), squeeze=False)
    for axis, group in zip(axes[:, 0], groups, strict=True):
        selected = [r for r in rows if (r["source_kind"], r["phase"]) == group]
        keys = sorted({(r["scheme"], r["layer"]) for r in selected})
        values = [
            [r["topk_to_mla_us"] for r in selected if (r["scheme"], r["layer"]) == key]
            for key in keys
        ]
        axis.boxplot(
            values,
            orientation="horizontal",
            tick_labels=[f"{scheme} L{layer}" for scheme, layer in keys],
            showfliers=True,
        )
        for index, samples in enumerate(values, 1):
            axis.scatter(samples, [index] * len(samples), s=13, alpha=0.65, color="#4361a9")
        axis.set_title(
            f"{group[0]} / {group[1]}: top-k GPU end → MLA GPU start", loc="left", fontsize=10
        )
        axis.invert_yaxis()
        axis.set_xlabel("µs (intrusive NSYS samples; panels have independent scales)")
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output / "transition.svg")
    figure.savefig(output / "transition.png", dpi=150)
    plt.close(figure)
