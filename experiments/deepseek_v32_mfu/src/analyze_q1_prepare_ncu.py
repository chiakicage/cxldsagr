"""Archive all kernels and source stalls from the current FIFO NCU captures."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from experiments.deepseek_v32_echo_official.src.analyze_q1_ncu import digest, instances


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), default="baseline")
    args = parser.parse_args()
    sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
    import ncu_report

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    keys = (
        "gpu__time_duration.sum",
        "launch__grid_size",
        "launch__block_size",
        "launch__waves_per_multiprocessor",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_eligible.avg.per_cycle_active",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__cycles_active.avg",
        "sm__cycles_active.min",
        "sm__cycles_active.max",
    )
    summary = {
        "analyzer_sha256": digest(Path(__file__)),
        "ncu_python": ncu_report.__file__,
        "variant": args.variant,
        "boundary": "Preparation kernels separately replayed with cache flush and base clocks. Excludes memsets, dispatch and gaps; no API latency or candidate gain is inferred from NCU duration sums.",
        "reports": {},
    }
    for collection in ("full", "source"):
        path = args.reports / f"{collection}.ncu-rep"
        report = ncu_report.load_report(str(path))
        actions = [
            report.range_by_idx(r).action_by_idx(a)
            for r in range(report.num_ranges())
            for a in range(report.range_by_idx(r).num_actions())
        ]
        expected = (
            (
                "prefetch_prepare_keys",
                "DeviceRadixSortHistogramKernel",
                "DeviceRadixSortExclusiveSumKernel",
                "DeviceRadixSortOnesweepKernel",
                "DeviceRadixSortOnesweepKernel",
                "prefetch_prepare_slots",
            )
            if args.variant == "baseline"
            else ("prepare_masks", "select_slots", "reset_publish")
        )
        if len(actions) != len(expected):
            raise RuntimeError(
                f"Expected {len(expected)} preparation kernels, found {len(actions)}"
            )
        rows = []
        for index, (action, fragment) in enumerate(zip(actions, expected, strict=True)):
            if fragment not in action.name():
                raise RuntimeError(f"Unexpected kernel {index}: {action.name()}")
            tag = f"{collection}_{index}_{fragment}"
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
                        location = (
                            (source.file_name(), source.line()) if source else ("<unmapped>", 0)
                        )
                        grouped[location] += count
                    stalls[name] = [
                        {"file": file, "line": line, "samples": count}
                        for (file, line), count in sorted(
                            grouped.items(), key=lambda item: -item[1]
                        )
                        if count
                    ]
            write(args.output / f"{tag}.metrics.json", metrics)
            write(args.output / f"{tag}.sampling.json", samples)
            write(args.output / f"{tag}.stalls.json", stalls)
            write(args.output / f"{tag}.rules.json", action.rule_results_as_dicts())
            rows.append(
                {
                    "tag": tag,
                    "name": action.name(),
                    "metrics": {name: metrics.get(name) for name in keys},
                    "metric_count": len(metrics),
                    "pm_sampled_metrics": len(samples),
                    "stall_samples_by_metric": {
                        name: sum(row["samples"] for row in values)
                        for name, values in stalls.items()
                    },
                }
            )
        summary["reports"][collection] = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "kernels": rows,
        }
    write(args.output / "summary.json", summary)
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
