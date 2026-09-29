"""Aggregate separate NOSA end-to-end benchmarks and inclusive event profiles."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from numbers import Real
from pathlib import Path

PHASES = ("full_prefill", "extend")
TIMING_METRICS = ("wall_ms", "host_submit_ms", "cuda_span_ms")
PRIMARY_STAGES = ("cis_projection", "indexer_total", "block_sparse_attention")
FUSED_STAGES = (
    "indexer_validate",
    "indexer_cache_update",
    "pooled_scores",
    "topk_qa",
    "prepare_cis",
    "topk_cis",
    "finish_selection",
)
NESTED_STAGES = (
    "compression_k",
    "compression_cis",
    "compressed_scores",
    "select_from_scores",
    *FUSED_STAGES,
    "native_indexer",
    "native_selection",
    "native_prepare",
    "native_prepare_ranked",
    "native_checked_indexer",
)
STAGES = (*PRIMARY_STAGES, *NESTED_STAGES)
WORKLOAD = {
    "prefix_tokens": 65536,
    "new_tokens": 1024,
    "total_tokens": 66560,
    "chunk_size": 1024,
}


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _duration(value, name, *, allow_zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or (not allow_zero and value == 0)
    ):
        condition = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {condition}")
    return float(value)


def _stats(values):
    result = {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    if any(not math.isfinite(value) for value in (*values, *result.values())):
        raise ValueError("Aggregated statistics must remain finite")
    return result


def _phase_runs(value, name):
    if not isinstance(value, dict) or set(value) != set(PHASES):
        raise ValueError(f"{name} must contain complete full_prefill and extend phases")
    for phase, runs in value.items():
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"{name}.{phase} must contain at least one complete measured run")
    if len(value[PHASES[0]]) != len(value[PHASES[1]]):
        raise ValueError(f"{name} phases must have the same number of measured runs")
    return value


def _expected_queries(phase):
    if phase == "extend":
        return {(WORKLOAD["prefix_tokens"], WORKLOAD["new_tokens"])}
    return {
        (start, min(WORKLOAD["chunk_size"], WORKLOAD["total_tokens"] - start))
        for start in range(0, WORKLOAD["total_tokens"], WORKLOAD["chunk_size"])
    }


def _validate_scopes(records):
    if not any("scope_id" in record for record in records):
        if any("parent_scope_id" in record for record in records):
            raise ValueError("parent_scope_id requires scope_id on all profile records")
        return
    scopes = {}
    for record in records:
        scope = _integer(record.get("scope_id"), "scope_id")
        if "parent_scope_id" not in record:
            raise ValueError("Every scope must declare parent_scope_id")
        if scope in scopes:
            raise ValueError("scope_id must be unique within one profile run")
        scopes[scope] = record
    ranked_parents = {
        _integer(record["parent_scope_id"], "parent_scope_id")
        for record in records
        if record["stage"] in ("native_prepare_ranked", "native_checked_indexer")
    }
    for record in records:
        parent_id = record["parent_scope_id"]
        if "prepared_ranking" in record and (
            record["stage"] not in ("native_indexer", "native_selection", "native_checked_indexer")
            or not isinstance(record["prepared_ranking"], bool)
        ):
            raise ValueError("prepared_ranking must be a boolean on native selection scopes")
        if record["stage"] in PRIMARY_STAGES:
            if parent_id is not None:
                raise ValueError("Primary profile stages must be top-level scopes")
            continue
        _integer(parent_id, "parent_scope_id")
        parent = scopes.get(parent_id)
        if parent is None or parent["stage"] != "indexer_total":
            raise ValueError("Nested profile stages require an indexer_total parent")
        for name in ("layer_idx", "query_start", "query_length", "call_id"):
            if record.get(name) != parent.get(name):
                raise ValueError(f"Nested stage and indexer_total must share {name}")
        if "prepared_ranking" in record and record["prepared_ranking"] != (
            parent_id in ranked_parents
        ):
            raise ValueError("Native selection prepared_ranking disagrees with its preparation")


def _profile_run(records, phase, num_layers):
    if not isinstance(records, list) or not records:
        raise ValueError("Each profile run must be a nonempty list of event records")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Each profile event must be an object")
    expected_queries = _expected_queries(phase)
    has_queries = any("query_start" in record or "query_length" in record for record in records)
    primary = defaultdict(list)
    totals = defaultdict(float)
    layer_totals = defaultdict(float)
    calls = Counter()
    layer_calls = Counter()
    for record in records:
        stage = record.get("stage")
        if stage not in STAGES:
            raise ValueError(f"Unknown NOSA profile stage: {stage!r}")
        layer = _integer(record.get("layer_idx"), "layer_idx")
        if layer >= num_layers:
            raise ValueError("Profile layer_idx is outside the model layer range")
        if record.get("phase", phase) != phase:
            raise ValueError("Profile phase does not match its containing run")
        if record.get("status", "ok") != "ok":
            raise ValueError("Failed profile events cannot be included in measured results")
        if record.get("inclusive", True) is not True:
            raise ValueError("This analysis requires inclusive CUDA event scopes")
        if "call_id" in record:
            _integer(record["call_id"], "call_id")
        if has_queries:
            start = _integer(record.get("query_start"), "query_start")
            length = _integer(record.get("query_length"), "query_length", minimum=1)
            if (start, length) not in expected_queries:
                raise ValueError(f"Unexpected {phase} query range: {(start, length)}")
        duration = _duration(
            record.get("cuda_elapsed_ms"),
            "cuda_elapsed_ms",
            allow_zero=stage in NESTED_STAGES,
        )
        totals[stage] += duration
        layer_totals[layer, stage] += duration
        calls[stage] += 1
        layer_calls[layer, stage] += 1
        if stage in PRIMARY_STAGES:
            primary[layer, stage].append(record)
    for layer in range(num_layers):
        for stage in PRIMARY_STAGES:
            rows = primary[layer, stage]
            if len(rows) != len(expected_queries):
                raise ValueError(
                    f"Incomplete {phase} run: layer {layer} {stage} needs "
                    f"{len(expected_queries)} calls, found {len(rows)}"
                )
            if has_queries:
                found = {(row["query_start"], row["query_length"]) for row in rows}
                if found != expected_queries:
                    raise ValueError(f"Duplicate or missing query range for layer {layer} {stage}")
    _validate_scopes(records)
    return totals, layer_totals, calls, layer_calls


def _summarize_profiles(profiles, num_layers):
    result = {}
    for phase, runs in profiles.items():
        accumulated = [_profile_run(records, phase, num_layers) for records in runs]
        present = {stage for totals, _, _, _ in accumulated for stage in totals}

        def stage_summary(stage, layer=None, *, samples=accumulated):
            durations, counts = [], []
            for totals, layer_totals, calls, layer_calls in samples:
                durations.append(totals[stage] if layer is None else layer_totals[layer, stage])
                counts.append(calls[stage] if layer is None else layer_calls[layer, stage])
            return {
                "parent_stage": "indexer_total" if stage in NESTED_STAGES else None,
                "cuda_elapsed_ms": _stats(durations),
                "calls_per_run": _stats(counts),
            }

        result[phase] = {
            "sample_count": len(runs),
            "stage_totals": {stage: stage_summary(stage) for stage in STAGES if stage in present},
            "per_layer": [
                {
                    "layer_idx": layer,
                    "stages": {
                        stage: stage_summary(stage, layer) for stage in STAGES if stage in present
                    },
                }
                for layer in range(num_layers)
            ],
            "unrecorded_stages": [stage for stage in STAGES if stage not in present],
        }
    return result


def summarize(measurements, *, num_layers=None):
    """Validate complete repetitions, then aggregate within each run first.

    Timings are uninstrumented end-to-end samples. Profiles contain one list
    of inclusive event records per independently instrumented repetition.
    Nested stages are reported separately and never added to indexer_total.
    """
    if (
        not isinstance(measurements, dict)
        or type(measurements.get("schema_version")) is not int
        or measurements["schema_version"] != 1
    ):
        raise ValueError("Expected measurements schema_version=1")
    run_id = measurements.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("measurements must declare a nonempty run_id")
    workload = measurements.get("workload")
    if not isinstance(workload, dict):
        raise TypeError("measurements must declare the experiment workload object")
    for name, expected in WORKLOAD.items():
        if _integer(workload.get(name), name, minimum=1) != expected:
            raise ValueError(f"This experiment requires {name}={expected}")
    timings = _phase_runs(measurements.get("timings"), "timings")
    profiles = _phase_runs(measurements.get("profiles"), "profiles")
    if "warmup" in workload:
        _integer(workload["warmup"], "warmup")
    for name, runs in (("repeats", timings), ("profile_repeats", profiles)):
        if name in workload:
            expected_count = _integer(workload[name], name, minimum=1)
            if any(len(samples) != expected_count for samples in runs.values()):
                raise ValueError(f"Measured run count must match workload.{name}={expected_count}")
    layer_source = "provided"
    if num_layers is None:
        layers = set()
        for runs in profiles.values():
            for records in runs:
                if not isinstance(records, list) or not records:
                    raise ValueError("Each profile run must contain its own event list")
                for record in records:
                    if not isinstance(record, dict):
                        raise TypeError("Each profile event must be an object")
                    layers.add(_integer(record.get("layer_idx"), "layer_idx"))
        num_layers = max(layers) + 1
        layer_source = "inferred_from_records"
    _integer(num_layers, "num_layers", minimum=1)
    timing_summary = {}
    for phase, runs in timings.items():
        values = {metric: [] for metric in TIMING_METRICS}
        for row in runs:
            if not isinstance(row, dict):
                raise TypeError("Each end-to-end repetition must be an object")
            for metric in TIMING_METRICS:
                values[metric].append(_duration(row.get(metric), f"{phase}.{metric}"))
        tokens = WORKLOAD["total_tokens"] if phase == "full_prefill" else WORKLOAD["new_tokens"]
        timing_summary[phase] = {
            "sample_count": len(runs),
            "tokens": tokens,
            **{metric: _stats(samples) for metric, samples in values.items()},
            "tokens_per_second": _stats([tokens * 1000 / ms for ms in values["wall_ms"]]),
        }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "workload": dict(workload),
        "num_layers": num_layers,
        "num_layers_source": layer_source,
        "definitions": {
            "full_prefill": "66560 tokens from an empty cache in 65 chunks of 1024 tokens",
            "extend": "One 1024-token forward with a 65536-token prefix already cached",
            "wall_ms": "Synchronized end-to-end benchmark elapsed time",
            "host_submit_ms": "Host submission interval, reported separately from CUDA time",
            "cuda_span_ms": "CUDA event span of the measured work, including possible launch gaps",
            "tokens_per_second": "Processed tokens * 1000 / wall_ms for each repetition",
            "profile_cuda_elapsed_ms": (
                "Inclusive CUDA event intervals from separate instrumented runs; includes possible "
                "launch gaps, not GPU active time or pure kernel duration"
            ),
            "profile_aggregation": (
                "Sum each stage within each run across calls/layers, then compute run statistics; "
                "per-layer statistics use the same order; nested stages remain inside indexer_total"
            ),
            "overlap": "Host and CUDA intervals overlap and must not be added",
        },
        "timings": timing_summary,
        "profiles": _summarize_profiles(profiles, num_layers),
    }


def _format_interval(stats):
    return f"{stats['median']:.3f} [{stats['min']:.3f}, {stats['max']:.3f}]"


def markdown_report(report):
    """Render summary tables while preserving benchmark/profile distinctions."""
    lines = [
        f"# NOSA sparse attention performance: {report['run_id']}",
        "",
        "P=65536, Q=1024, full sequence=66560; prefill chunks=1024 tokens.",
        "Values are median [min, max] across measured repetitions; warmups are excluded.",
        "",
        "## End-to-end benchmark",
        "",
        "| Phase | Tokens | Repeats | Wall ms | Host submit ms | CUDA span ms | Tokens/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in PHASES:
        row = report["timings"][phase]
        metrics = " | ".join(_format_interval(row[metric]) for metric in TIMING_METRICS)
        lines.append(
            f"| {phase} | {row['tokens']} | {row['sample_count']} | {metrics} | "
            f"{_format_interval(row['tokens_per_second'])} |"
        )
    lines.extend(
        [
            "",
            "Extend excludes prefix preparation. Host and CUDA intervals overlap; do not add them.",
            "",
            "## Module attribution from separate instrumented runs",
            "",
            (
                "CUDA events include possible launch gaps. These inclusive intervals are not GPU "
                "active time or pure kernel duration. Instrumented profiles are not the "
                "end-to-end benchmark."
            ),
            "Nested stages are already included in indexer_total and must not be added to it.",
            "Each stage is summed within a complete run before statistics are taken across runs.",
            "",
            "| Phase | Stage | Parent | Repeats | Total CUDA event ms | Calls per run |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for phase in PHASES:
        profile = report["profiles"][phase]
        for stage, row in profile["stage_totals"].items():
            lines.append(
                f"| {phase} | {stage} | {row['parent_stage'] or '—'} | "
                f"{profile['sample_count']} | {_format_interval(row['cuda_elapsed_ms'])} | "
                f"{_format_interval(row['calls_per_run'])} |"
            )
    for phase in PHASES:
        missing = report["profiles"][phase]["unrecorded_stages"]
        if missing:
            lines.extend(["", f"{phase} unrecorded stages: {', '.join(missing)}."])
    lines.extend(
        [
            "",
            "## Per-layer primary stages",
            "",
            "| Phase | Layer | CIS projection ms | Indexer total ms | Block sparse attention ms |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase in PHASES:
        for row in report["profiles"][phase]["per_layer"]:
            cells = " | ".join(
                _format_interval(row["stages"][stage]["cuda_elapsed_ms"])
                for stage in PRIMARY_STAGES
            )
            lines.append(f"| {phase} | {row['layer_idx']} | {cells} |")
    lines.extend(["", "Per-layer nested-stage statistics are also available in layers.csv.", ""])
    return "\n".join(lines)


def _csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def analyze(data_dir, *, num_layers=None):
    """Read a completed run and write summary.json, report.md, and three CSVs."""
    data_dir = Path(data_dir)
    measurements = json.loads((data_dir / "measurements.json").read_text(encoding="utf-8"))
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_id", measurements.get("run_id")) != measurements.get("run_id"):
            raise ValueError("metadata and measurements run_id must agree")
        declared_layers = metadata.get("model_config", {}).get("num_hidden_layers")
        if declared_layers is not None:
            if num_layers is not None and num_layers != declared_layers:
                raise ValueError("num_layers must match metadata.model_config.num_hidden_layers")
            num_layers = declared_layers
    report = summarize(measurements, num_layers=num_layers)
    timings, stages, layers = [], [], []
    for phase in PHASES:
        timing = report["timings"][phase]
        for metric in (*TIMING_METRICS, "tokens_per_second"):
            timings.append(
                {
                    "phase": phase,
                    "metric": metric,
                    "unit": "tokens/s" if metric == "tokens_per_second" else "ms",
                    "tokens": timing["tokens"],
                    **timing[metric],
                }
            )
        profile = report["profiles"][phase]
        for stage, row in profile["stage_totals"].items():
            stages.append(
                {
                    "phase": phase,
                    "stage": stage,
                    "parent_stage": row["parent_stage"],
                    **row["cuda_elapsed_ms"],
                    "min_calls": row["calls_per_run"]["min"],
                    "max_calls": row["calls_per_run"]["max"],
                }
            )
        for layer in profile["per_layer"]:
            for stage, row in layer["stages"].items():
                layers.append(
                    {
                        "phase": phase,
                        "layer_idx": layer["layer_idx"],
                        "stage": stage,
                        "parent_stage": row["parent_stage"],
                        **row["cuda_elapsed_ms"],
                        "min_calls": row["calls_per_run"]["min"],
                        "max_calls": row["calls_per_run"]["max"],
                    }
                )
    outputs = {
        "summary.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        "report.md": markdown_report(report),
        "timings.csv": _csv_text(timings),
        "stages.csv": _csv_text(stages),
        "layers.csv": _csv_text(layers),
    }
    for name, content in outputs.items():
        (data_dir / name).write_text(content, encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Completed run data directory")
    parser.add_argument(
        "--num-layers", type=int, help="Expected layers; normally read from metadata"
    )
    args = parser.parse_args(argv)
    report = analyze(args.data_dir, num_layers=args.num_layers)
    print(json.dumps(report["timings"], indent=2))


if __name__ == "__main__":
    main()
