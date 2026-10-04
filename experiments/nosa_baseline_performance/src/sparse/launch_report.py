"""Compare independent unprofiled timings with a root-only NOSA CUDA timeline."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.nosa_baseline_performance.src.sparse.analyze import (
    PHASES,
    TIMING_METRICS,
    _duration,
    _integer,
    _phase_runs,
    _stats,
)
from experiments.nosa_baseline_performance.src.sparse.launch_analysis import analyze
from experiments.nosa_baseline_performance.src.sparse.mfu import validate_kernel_backend
from experiments.nosa_baseline_performance.src.sparse.module_mfu import ROOT_RANGE

IDENTITY_FIELDS = (
    "model_config",
    "checkpoint_path",
    "checkpoint_files",
    "checkpoint_config_sha256",
    "request_sha256",
    "source_sha256",
    "torch",
    "cuda",
    "triton",
    "flashinfer",
    "tvm_ffi",
    "native_build",
    "gpu",
)
WORKLOAD_ARGS = (
    "prefix_tokens",
    "new_tokens",
    "chunk_size",
    "warmup",
    "repeats",
    "profile_repeats",
)
TRACE_METRICS = (
    "hull_ms",
    "gpu_active_union_ms",
    "kernel_union_ms",
    "idle_ms",
    "idle_pct",
    "before_next_submission_ms",
    "remaining_gap_ms",
    "kernel_count",
    "copy_memset_count",
    "kernel_median_us",
    "kernels_le_5us",
)


def validate_inputs(measurements, metadata, profile_metadata):
    """Reject changed workloads/sources and incomplete independently timed runs."""
    if measurements.get("schema_version") != 1:
        raise ValueError("Expected measurements schema_version=1")
    run_id = measurements.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("A nonempty run_id is required")
    for label, meta, mode in (
        ("benchmark", metadata, "benchmark"),
        ("timeline", profile_metadata, "timeline"),
    ):
        if meta.get("run_id") != run_id or meta.get("args", {}).get("run_id") != run_id:
            raise ValueError(f"{label} run_id differs from measurements")
        if meta["args"].get("mode") != mode:
            raise ValueError(f"Expected {label} metadata mode={mode}")
    if profile_metadata.get("module_scopes_enabled") is not False:
        raise ValueError("Timeline metadata must declare module_scopes_enabled=false")
    if measurements.get("profiles") != {}:
        raise ValueError("Root-only timeline must not contain module profile records")
    for key in IDENTITY_FIELDS:
        if (
            key not in metadata
            or key not in profile_metadata
            or metadata[key] != profile_metadata[key]
        ):
            raise ValueError(f"Timeline and benchmark differ in {key}")
    if not isinstance(metadata["source_sha256"], dict) or not metadata["source_sha256"]:
        raise ValueError("Nonempty source_sha256 provenance is required")
    workload = measurements.get("workload", {})
    validate_kernel_backend(workload, metadata)
    validate_kernel_backend(workload, profile_metadata)
    for key in WORKLOAD_ARGS:
        expected = _integer(workload.get(key), f"workload.{key}", minimum=1)
        if any(meta["args"].get(key) != expected for meta in (metadata, profile_metadata)):
            raise ValueError(f"Timeline and benchmark workload differ in {key}")
    if workload.get("total_tokens") != workload["prefix_tokens"] + workload["new_tokens"]:
        raise ValueError("total_tokens must equal prefix_tokens + new_tokens")
    if (
        workload["prefix_tokens"] % workload["chunk_size"]
        or workload["new_tokens"] > workload["chunk_size"]
    ):
        raise ValueError("Invalid chunk boundary for the benchmark workload")
    results = {}
    for label, rows, count in (
        ("benchmark", measurements.get("timings"), workload["repeats"]),
        ("timeline", profile_metadata.get("instrumented_timings"), workload["profile_repeats"]),
    ):
        _phase_runs(rows, f"{label} timings")
        results[label] = {}
        for phase in PHASES:
            if len(rows[phase]) != count:
                raise ValueError(f"{label}.{phase} timing count must equal {count}")
            for sample in rows[phase]:
                if not isinstance(sample, dict):
                    raise TypeError("Timing samples must be objects")
                for key in TIMING_METRICS:
                    _duration(sample.get(key), f"{label}.{phase}.{key}")
            results[label][phase] = {
                key: _stats([sample[key] for sample in rows[phase]]) for key in TIMING_METRICS
            }
    return results


def summarize(measurements, metadata, profile_metadata, launch):
    """Keep trace idle fractions and primary wall latency in their own processes."""
    timings = validate_inputs(measurements, metadata, profile_metadata)
    if (
        launch.get("source_run_id") != measurements["run_id"]
        or launch.get("capture_mode") != "timeline"
    ):
        raise ValueError("Launch analysis must come from this run's timeline capture")
    if launch.get("gpu") != metadata["gpu"]:
        raise ValueError("Launch analysis GPU differs from benchmark metadata")
    runs = {phase: {} for phase in PHASES}
    count = measurements["workload"]["profile_repeats"]
    for row in launch["runs"]:
        match = ROOT_RANGE.fullmatch(row.get("scope", ""))
        if match is None:
            raise ValueError("Invalid timeline root scope")
        phase, iteration = match[1], int(match[2])
        if iteration in runs[phase]:
            raise ValueError("Duplicate timeline root iteration")
        if row.get("stages"):
            raise ValueError("Timeline must not contain module stages")
        metrics = {}
        for key in TRACE_METRICS:
            value = row.get(key)
            metrics[key] = _duration(value, f"trace.{phase}.{key}", allow_zero=True)
        for label, key in (("root", "root_cuda_apis"), ("gpu_hull", "cuda_apis")):
            apis = row.get(key)
            if not isinstance(apis, dict):
                raise TypeError(f"Trace must contain {key}")
            sync = [api for name, api in apis.items() if "synchronize" in name.lower()]
            metrics[f"{label}_sync_api_count"] = sum(
                _integer(api.get("count"), "sync count") for api in sync
            )
            metrics[f"{label}_sync_api_duration_sum_ms"] = sum(
                _duration(api.get("duration_sum_ms"), "sync duration", allow_zero=True)
                for api in sync
            )
        metrics["before_next_submission_pct_of_idle"] = (
            100 * metrics["before_next_submission_ms"] / metrics["idle_ms"]
            if metrics["idle_ms"]
            else 0.0
        )
        runs[phase][iteration] = metrics
    for phase in PHASES:
        if set(runs[phase]) != set(range(count)):
            raise ValueError("Timeline roots must exactly cover profile_repeats in both phases")
    timeline = {
        phase: {key: _stats([row[key] for row in runs[phase].values()]) for key in runs[phase][0]}
        for phase in PHASES
    }
    return {
        "schema_version": 1,
        "run_id": measurements["run_id"],
        "workload": measurements["workload"],
        "gpu": metadata["gpu"],
        "benchmark": timings["benchmark"],
        "timeline_timings": timings["timeline"],
        "trace_overhead": {
            phase: {
                key: {
                    "median_ratio": timings["timeline"][phase][key]["median"]
                    / timings["benchmark"][phase][key]["median"],
                    "median_change_pct": 100
                    * (
                        timings["timeline"][phase][key]["median"]
                        / timings["benchmark"][phase][key]["median"]
                        - 1
                    ),
                }
                for key in TIMING_METRICS
            }
            for phase in PHASES
        },
        "timeline": timeline,
        "definitions": {
            **launch["definitions"],
            "benchmark": "Separate process without nsys or module scopes; wall includes CUDA completion; host_submit overlaps GPU work; CUDA events span launch gaps",
            "timeline_timings": "Separate process under nsys CUDA tracing and one root NVTX range per forward; no SparseScopes; these are profiled timings, not pure unprofiled wall latency",
            "trace_overhead": "Ratio of independently measured medians, including run variability and net tracing perturbation; not a per-launch cost or a correction factor",
            "sync": "Host synchronize API duration overlaps GPU execution and other waits; root totals include setup/final waits, hull totals clip to first/last GPU activity; not additive with GPU idle",
            "comparison": "Do not subtract trace GPU active time from independent benchmark wall time, or add host_submit to CUDA span; launch-bound evidence comes from gaps within the same trace",
            "limits": launch["definitions"]["limits"]
            + "; no module attribution, HBM-bandwidth diagnosis or pure launch-latency isolation",
        },
        "provenance": {
            "source_sha256": metadata["source_sha256"],
            "native_build": metadata["native_build"],
            "tvm_ffi": metadata["tvm_ffi"],
        },
    }


def markdown(report):
    lines = [
        f"# Root-only NOSA timeline: {report['run_id']}",
        "",
        (
            "Benchmark timings are from a separate process without nsys or module scopes. "
            "Timeline timings retain nsys CUDA tracing and one root NVTX range per forward. "
            "All values below are medians; CSV/JSON also preserve min/max and sample count."
        ),
        "",
        "| Phase | Benchmark wall / host / CUDA (ms) | Trace wall (ms) | Trace / benchmark wall | GPU hull / active / idle (ms) | Idle (%) | Before next submission (ms; % idle) | Kernels | Hull sync count / ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in PHASES:
        bench, trace = report["benchmark"][phase], report["timeline"][phase]

        def value(key, trace=trace):
            return trace[key]["median"]

        lines.append(
            f"| {phase} | "
            + " / ".join(f"{bench[key]['median']:.3f}" for key in TIMING_METRICS)
            + f" | {report['timeline_timings'][phase]['wall_ms']['median']:.3f}"
            + f" | {report['trace_overhead'][phase]['wall_ms']['median_ratio']:.3f}×"
            + " | "
            + " / ".join(
                f"{value(key):.3f}" for key in ("hull_ms", "gpu_active_union_ms", "idle_ms")
            )
            + f" | {value('idle_pct'):.2f}"
            + f" | {value('before_next_submission_ms'):.3f}; {value('before_next_submission_pct_of_idle'):.2f}%"
            + f" | {value('kernel_count'):.0f}"
            + f" | {value('gpu_hull_sync_api_count'):.0f} / {value('gpu_hull_sync_api_duration_sum_ms'):.3f} |"
        )
    lines.extend(["", "Definitions and limits:", ""])
    lines.extend(f"- **{key}**: {value}" for key, value in report["definitions"].items())
    return "\n".join(lines) + "\n"


def write_report(data_dir):
    data_dir = Path(data_dir)
    inputs = {
        name: json.loads((data_dir / f"{name}.json").read_text())
        for name in ("measurements", "metadata", "profile_metadata")
    }
    validate_inputs(inputs["measurements"], inputs["metadata"], inputs["profile_metadata"])
    launch = analyze(data_dir, data_dir / "launch_analysis.json")
    report = summarize(
        inputs["measurements"], inputs["metadata"], inputs["profile_metadata"], launch
    )
    report["provenance"]["input_sha256"] = {}
    for name in (*[f"{name}.json" for name in inputs], "launch_analysis.json"):
        with (data_dir / name).open("rb") as handle:
            report["provenance"]["input_sha256"][name] = hashlib.file_digest(
                handle, "sha256"
            ).hexdigest()
    report["provenance"]["launch_report_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (data_dir / "launch_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (data_dir / "launch_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("phase", "process", "metric", "count", "median", "min", "max")
        )
        writer.writeheader()
        for process in ("benchmark", "timeline_timings", "timeline"):
            for phase in PHASES:
                for metric, stats in report[process][phase].items():
                    writer.writerow({"phase": phase, "process": process, "metric": metric, **stats})
    (data_dir / "launch_summary.md").write_text(markdown(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args(argv)
    report = write_report(args.data_dir)
    print(markdown(report))


if __name__ == "__main__":
    main()
