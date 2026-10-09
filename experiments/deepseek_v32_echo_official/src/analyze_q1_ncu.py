"""Extract Q1 scheduling evidence from the four independent NCU reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def instances(metric):
    if metric.kind() in (metric.ValueKind_UINT32, metric.ValueKind_UINT64):
        getter = metric.as_uint64
    elif metric.kind() == metric.ValueKind_STRING:
        getter = metric.as_string
    else:
        getter = metric.as_double
    return [getter(i) for i in range(metric.num_instances())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["nonpaged", "paged"])
    parser.add_argument(
        "--boundary",
        default="One core kernel per report, profiler replay with cache flush and base "
        "clock control. These durations exclude packing/metadata and are not clean API timing.",
    )
    args = parser.parse_args()
    sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
    import ncu_report

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analyze_q1_ncu.py").write_bytes(Path(__file__).read_bytes())
    summary = {
        "ncu_python": ncu_report.__file__,
        "source_sha256": digest(Path(__file__)),
        "boundary": args.boundary,
        "reports": {},
    }
    keys = (
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum",
        "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
        "sm__cycles_active.avg",
        "sm__cycles_active.max",
        "sm__cycles_active.min",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "launch__grid_size",
        "launch__block_size",
    )
    for method in args.methods:
        for collection in ("full", "source"):
            tag = f"{method}_{collection}"
            path = args.reports / f"{tag}.ncu-rep"
            report = ncu_report.load_report(str(path))
            if report.num_ranges() != 1 or report.range_by_idx(0).num_actions() != 1:
                raise ValueError(f"Expected exactly one core kernel in {path}")
            action = report.range_by_idx(0).action_by_idx(0)
            metrics, samples, stalls = {}, {}, {}
            for name in sorted(action.metric_names()):
                metric = action[name]
                metrics[name] = {
                    "value": metric.value() if metric.has_value() else None,
                    "unit": metric.unit(),
                }
                if name.startswith("pmsampling:"):
                    samples[name] = {
                        **metrics[name],
                        "instances": instances(metric),
                        "correlation_ids": instances(metric.correlation_ids())
                        if metric.has_correlation_ids()
                        else None,
                    }
                if (
                    name.startswith("smsp__pcsamp_warps_issue_stalled")
                    and metric.has_correlation_ids()
                ):
                    grouped = defaultdict(int)
                    ids = metric.correlation_ids()
                    for i, count in enumerate(instances(metric)):
                        source = action.source_info(ids.as_uint64(i))
                        key = (source.file_name(), source.line()) if source else ("<unmapped>", 0)
                        grouped[key] += count
                    stalls[name] = [
                        {"file": file, "line": line, "samples": count}
                        for (file, line), count in sorted(
                            grouped.items(), key=lambda item: item[1], reverse=True
                        )
                        if count
                    ]
            write(args.output_dir / f"{tag}.metrics.json", metrics)
            write(args.output_dir / f"{tag}.sampling.json", samples)
            write(args.output_dir / f"{tag}.source_stalls.json", stalls)
            write(args.output_dir / f"{tag}.rules.json", action.rule_results_as_dicts())
            summary["reports"][tag] = {
                "path": str(path.resolve()),
                "sha256": digest(path),
                "kernel": action.name(),
                "metrics": {key: metrics.get(key) for key in keys},
                "metric_count": len(metrics),
                "pm_sampled_metrics": len(samples),
                "source_stall_metrics": len(stalls),
            }
    write(args.output_dir / "summary.json", summary)
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
