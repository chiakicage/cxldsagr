"""Offline three-version serving comparisons with empirical repeat-noise bounds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import tempfile
from pathlib import Path

TIMINGS = ("latency_ms", "extend_ms", "admission_ms", "prefix_ms", "cleanup_ms")
ACCOUNTING = (
    "cache_hbm_bytes",
    "cache_dram_bytes",
    "reserved_hbm_bytes",
    "reserved_dram_bytes",
    "request_cache_hbm_bytes",
    "request_cache_dram_bytes",
)
MEMORY = ("torch_allocated_bytes", "torch_reserved_bytes", "device_used_bytes")


def memory_value(sample, name):
    if name == "device_used_bytes" and name not in sample:
        return sample["cuda_total_bytes"] - sample["cuda_free_bytes"]
    return sample[name]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(values):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("comparison requires finite nonnegative measurements")
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "sum": sum(values),
    }


def repeat_summary(values):
    center = statistics.median(values)
    return {
        "values": values,
        "median": center,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "mad": statistics.median(abs(value - center) for value in values),
    }


def trace_memory(run):
    return {
        **{
            name: max(group["sampled_memory"][name] for group in run["groups"].values())
            for name in MEMORY
        },
        **{
            name: max(case[name] for case in run["allocator_peaks"].values())
            for name in ("torch_peak_allocated_bytes", "torch_peak_reserved_bytes")
        },
    }


def assess_repeats(baseline, current):
    """A range is a detection floor, never a permitted percentage regression."""
    if min(len(baseline), len(current)) < 3:
        raise ValueError("at least three independent baseline and current runs are required")
    summarize(baseline)
    summarize(current)
    old, new = repeat_summary(baseline), repeat_summary(current)
    delta = new["median"] - old["median"]
    if min(current) > max(baseline) or delta > old["range"]:
        status = "investigate_slowdown"
    elif delta > 0:
        status = "increase_within_observed_repeat_range"
    else:
        status = "no_observed_slowdown"
    return {
        "baseline": old,
        "current": new,
        "delta": delta,
        "delta_pct": None if old["median"] == 0 else 100 * delta / old["median"],
        "status": status,
    }


def matching_contract(metadata):
    """Exclude version identities; retain actual workload and placement settings."""
    hardware = metadata["hardware"]
    return {
        "config": {
            key: value for key, value in metadata["config"].items() if key != "peak_bf16_tflops"
        },
        "workload_sha256": metadata["workload_sha256"],
        "measurement_boundary": metadata["measurement_boundary"],
        "model_boundary": metadata.get("model_boundary"),
        "memory_boundary": metadata.get("memory_boundary"),
        "precision_settings": metadata["precision_settings"],
        "execution_environment": metadata.get("execution_environment"),
        "checkpoint": metadata["checkpoint"],
        "hardware": {
            key: value
            for key, value in hardware.items()
            if key not in {"git_revision", "git_status", "submodule_status"}
        },
    }


def collect_run(directory, role, auditor, *, receipt_override=None):
    directory = Path(directory).resolve()
    metadata, rows, audit = auditor(directory, receipt_override=receipt_override)
    if metadata.get("status") != "accepted" or "-check-" in metadata["schema"]:
        raise ValueError("comparison requires accepted performance data")
    if role != "published" and metadata.get("mode") != "bench":
        raise ValueError("matched version comparison requires independent check/bench separation")
    groups = {}
    samples = []
    for row in rows:
        method = row.get("method", row["scheme"])
        group = (method, "revisit" if row["is_revisit"] else "first")
        groups.setdefault(group, []).append(row)
        samples.append(
            {
                "role": role,
                "run_id": metadata["run_id"],
                "method": method,
                "visit_kind": group[1],
                "request_id": row["request_id"],
                "prefix_cache_hit": row["prefix_cache_hit"],
                **{name: row[name] for name in TIMINGS},
                **{name: row[name] for name in ACCOUNTING},
                **{name: memory_value(row["memory_after"], name) for name in MEMORY},
            }
        )
    grouped = {
        f"{method}/{visit}": {
            "requests": len(selected),
            "prefix_hits": sum(row["prefix_cache_hit"] for row in selected),
            "timings": {name: summarize([row[name] for row in selected]) for name in TIMINGS},
            "accounting": {name: max(row[name] for row in selected) for name in ACCOUNTING},
            "sampled_memory": {
                name: max(memory_value(row["memory_after"], name) for row in selected)
                for name in MEMORY
            },
        }
        for (method, visit), selected in groups.items()
    }
    receipt = metadata.get("correctness_receipt", {})
    version = {
        "role": role,
        "run_id": metadata["run_id"],
        "directory": str(directory),
        "schema": metadata["schema"],
        "source_sha256": metadata["source_sha256"],
        "validation_identity_sha256": (
            identity_digest(metadata["validation_identity"])
            if "validation_identity" in metadata
            else None
        ),
        "receipt_sha256": receipt.get("sha256"),
        "receipt_path": receipt.get("path"),
        "receipt_resolved_path": audit.get("independent_correctness_receipt", {}).get("path"),
        "contract": matching_contract(metadata),
        "groups": grouped,
        "total_trace_latency_ms": sum(row["latency_ms"] for row in rows),
        "allocator_peaks": {
            case.get("method", case["scheme"]): {
                name: case[name]
                for name in ("torch_peak_allocated_bytes", "torch_peak_reserved_bytes")
            }
            for case in metadata["cases"]
        },
        "input_sha256": {
            name: digest(directory / name)
            for name in (
                "metadata.json",
                "measurements.jsonl",
                "source_manifest.json",
                "memory.json",
            )
        },
        "audit": audit,
    }
    return version, samples


def compare_versions(published, baseline, current):
    if min(len(baseline), len(current)) < 3:
        raise ValueError("at least three complete baseline and current runs are required")
    all_runs = [published, *baseline, *current]
    if len({run["run_id"] for run in all_runs}) != len(all_runs):
        raise ValueError("independent runs must have distinct run IDs")
    expected = baseline[0]["contract"]
    if any(run["contract"] != expected for run in [*baseline, *current]):
        raise ValueError(
            "baseline/current workload, placement, precision or timing boundary differs"
        )
    for family in (baseline, current):
        for key in ("source_sha256", "validation_identity_sha256"):
            if len({run[key] for run in family}) != 1:
                raise ValueError(f"repeated runs contain mixed {key} identities")
    if (
        len({published["source_sha256"], baseline[0]["source_sha256"], current[0]["source_sha256"]})
        != 3
    ):
        raise ValueError("published, frozen and refactored sources must have distinct identities")
    group_names = set(baseline[0]["groups"])
    if any(set(run["groups"]) != group_names for run in all_runs):
        raise ValueError("versions have different method/visit groups")
    comparisons = []
    accounting = []
    for group in sorted(group_names):
        method, visit = group.split("/")
        for field in ACCOUNTING:
            old = max(run["groups"][group]["accounting"][field] for run in baseline)
            new = max(run["groups"][group]["accounting"][field] for run in current)
            accounting.append(
                {
                    "method": method,
                    "visit_kind": visit,
                    "field": field,
                    "baseline_max": old,
                    "current_max": new,
                    "delta_bytes": new - old,
                    "increased": new > old,
                }
            )
        for metric in TIMINGS:
            for statistic in ("mean", "median", "p95", "sum"):
                repeated = [
                    [run["groups"][group]["timings"][metric][statistic] for run in family]
                    for family in (baseline, current)
                ]
                comparisons.append(
                    {
                        "method": method,
                        "visit_kind": visit,
                        "metric": metric,
                        "statistic": statistic,
                        "published": published["groups"][group]["timings"][metric][statistic],
                        **assess_repeats(*repeated),
                    }
                )
    return {
        "schema": "serving-three-version-comparison-v1",
        "runs": all_runs,
        "comparisons": comparisons,
        "accounting_comparisons": accounting,
        "trace_latency": assess_repeats(
            [run["total_trace_latency_ms"] for run in baseline],
            [run["total_trace_latency_ms"] for run in current],
        ),
        "published_contract_differences": [
            key for key in expected if published["contract"].get(key) != expected[key]
        ],
        "interpretation": (
            "Published values are descriptive historical controls. Only baseline/current share the "
            "validated workload, device/CPU/NUMA, precision and timing boundary. Repeat ranges and MAD "
            "are empirical detection floors, not permitted regression percentages or confidence "
            "intervals. Any stable increase or added synchronization requires investigation, including "
            "small increases within the observed range. Three repeats do not establish a tail-latency "
            "distribution. Memory samples are not process peaks; cache accounting is separate."
        ),
    }


def markdown(result, model, run_id):
    lines = [
        f"# {model} serving version comparison",
        "",
        f"Comparison run: `{run_id}`. All input runs passed their complete saved-evidence audit.",
        "",
        result["interpretation"],
        "",
        "Historical contract differences: "
        + ", ".join(result["published_contract_differences"] or ["none recorded"])
        + ".",
        "",
        "| Role | Run | Source SHA-256 | GPU UUID |",
        "| --- | --- | --- | --- |",
    ]
    for run in result["runs"]:
        lines.append(
            f"| {run['role']} | {run['run_id']} | {run['source_sha256']} | "
            f"{run['contract']['hardware'].get('uuid', 'unrecorded')} |"
        )
    lines += [
        "",
        (
            "Memory below is the maximum across methods, requests and each role's runs, in GiB. "
            "Post-request samples and CUDA allocator peak counters have separate columns; neither is "
            "a continuously observed device-used peak."
        ),
        "",
        "| Role | Sampled allocated | Sampled reserved | Sampled device used | Allocator peak allocated | Allocator peak reserved |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role in ("published", "baseline", "current"):
        observations = [trace_memory(run) for run in result["runs"] if run["role"] == role]
        values = [max(item[name] for item in observations) / 2**30 for name in observations[0]]
        lines.append(f"| {role} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += [
        "",
        "Values below are means within each run; baseline/current show the median of independent runs.",
        "All values are milliseconds. Positive change means slower execution.",
        "",
        "| Method | Visit | Metric | Published | Baseline | Current | Change | Baseline range | Assessment |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result["comparisons"]:
        if item["statistic"] != "mean" or item["metric"] not in ("latency_ms", "extend_ms"):
            continue
        lines.append(
            f"| {item['method']} | {item['visit_kind']} | {item['metric']} | {item['published']:.3f} | "
            f"{item['baseline']['median']:.3f} | {item['current']['median']:.3f} | "
            f"{item['delta']:+.3f} | {item['baseline']['range']:.3f} | {item['status']} |"
        )
    lines += [
        "",
        (
            "[Complete comparison](comparison.json) retains mean/median/p95/sum, allocation charges, "
            "sampled memory, complete contracts, evidence hashes and numerical audit results. "
            "[Request samples](request_samples.csv) retain all measured request values."
        ),
        "",
    ]
    return "\n".join(lines)


def main(*, model, experiment, auditor):
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id", required=True)
    command.add_argument("--published", type=Path, required=True)
    command.add_argument("--baseline", type=Path, action="append", required=True)
    command.add_argument("--current", type=Path, action="append", required=True)
    command.add_argument("--baseline-receipt", type=Path)
    command.add_argument("--current-receipt", type=Path)
    command.add_argument("--output-dir", type=Path)
    args = command.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run ID must contain letters, digits, underscores or hyphens")
    destination = args.output_dir or Path(experiment) / "output/data" / args.run_id
    if destination.exists():
        raise FileExistsError(destination)
    requests = []

    def collect(path, role, receipt_override=None):
        run, samples = collect_run(path, role, auditor, receipt_override=receipt_override)
        requests.extend(samples)
        return run

    result = compare_versions(
        collect(args.published, "published"),
        [collect(path, "baseline", args.baseline_receipt) for path in args.baseline],
        [collect(path, "current", args.current_receipt) for path in args.current],
    )
    result.update(model=model, run_id=args.run_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.run_id}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        (staging / "comparison.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        (staging / "results.md").write_text(markdown(result, model, args.run_id))
        with (staging / "request_samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(requests[0]))
            writer.writeheader()
            writer.writerows(requests)
        staging.rename(destination)
    print(f"accepted saved-evidence comparison: {destination}")
