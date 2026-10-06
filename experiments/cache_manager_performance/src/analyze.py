"""Audit NSYS intervals without treating cache metadata or D2D as computation."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

from experiments.cache_manager_performance.src.workload import PHASES, SCHEMES, file_sha256
from experiments.deepseek_v32_mfu.src.analyze_nsys import (
    _assign_scopes,
    _attribute,
    _read_capture,
    _union_ns,
)
from experiments.deepseek_v32_mfu.src.launch_gap import summarize_window
from experiments.deepseek_v32_mfu.src.timeline import activity_lane

SCOPE = re.compile(r"manager/(?P<scheme>[^/]+)/(?P<phase>[^/]+)/(?P<stage>.+)")
LAYER_SCOPE = re.compile(r"window/layer_(\d+)/(offload_exact_recall|indexer_prefetch)")


def category(row):
    name = row["name"].lower()
    normalized = name.replace("-", " ").replace("_", " ")
    stage = row["scope"]["stage"].split("/")[-1]
    if row.get("diagnostic_io_empty") is True:
        if row.get("diagnostic_io_records") != 0:
            raise ValueError("empty IO annotation requires a matched zero record count")
        if (
            row["kind"] == "kernel"
            and row.get("diagnostic_io_kind") == "exact_recall"
            and "gather_records" in name
        ):
            return "cache_management"
        if (
            row["kind"] == "kernel"
            and row.get("diagnostic_io_kind") == "fused_prefetch"
            and "sm90_fp8_mqa_logits_fuse_prefetch" in name
        ):
            return "indexer_compute"
        raise ValueError("empty IO annotation does not match its kernel family")
    if row["kind"] == "memcpy":
        if any(
            token in normalized for token in ("htod", "dtoh", "host to device", "device to host")
        ):
            return "io"
        return "d2d_control"
    if "sm90_fp8_mqa_logits_fuse_prefetch" in name:
        return "indexer_compute_and_io"
    if "gather_records" in name or "prefetch_ids_kernel" in name:
        return "io"
    if "sm90_fp8_mqa_logits" in name or "echo_native::clean" in name:
        return "indexer_compute"
    if stage == "indexer" and (
        "masked_fill_kernel" in name
        or "comparefunctor" in name
        or name == "_resident_causal_tail_mask"
    ):
        return "indexer_compute"
    if "prefetch_hint" in name:
        return "cache_management"
    if stage == "exact_topk":
        return "selection"
    return "cache_management"


def diagnostic_lane(row):
    """Keep exact selection compute and reuse MFU's pure-layout exceptions."""
    semantic = category(row)
    if row.get("diagnostic_io_empty") is True:
        # The complete zero-transport fused kernel still contains compute and
        # metadata. No internal duration split is inferred from its zero count.
        return "Compute" if semantic == "indexer_compute" else "GPU control"
    if semantic == "io":
        # Standalone also recognizes explicit mapped-host prefetch kernels.
        return "IO"
    stage = row["scope"]["stage"].split("/")[-1]
    if semantic == "indexer_compute":
        stage = "indexer_qk"
    adapted = {**row, "scope": {**row["scope"], "stage": stage}}
    return activity_lane(adapted)


def annotate_diagnostic_io(rows, sample, sample_index):
    """Bind unique per-layer transport calls to synchronized, checked counters."""
    metrics = sample["metrics"]
    checks = {check["layer"]: check for check in sample["checks"]}
    if len(checks) != len(sample["checks"]) or set(checks) != set(range(len(metrics))):
        raise ValueError("profile IO evidence omits or duplicates a checked layer")
    for layer, counts in enumerate(metrics):
        if checks[layer].get("exact_records") is not True:
            raise ValueError("profile IO evidence requires the completed exact-record check")
        for name in ("recalled_records", "prefetched_records", "record_bytes"):
            value = counts[name]
            if type(value) is not int or value < (1 if name == "record_bytes" else 0):
                raise ValueError("profile transport counters must be nonnegative integers")
            if value != checks[layer]["metrics"][name]:
                raise ValueError("profile transport counters differ from the checked result")
        if (
            counts["host_to_device_bytes"]
            != (counts["recalled_records"] + counts["prefetched_records"]) * counts["record_bytes"]
        ):
            raise ValueError("profile transport bytes do not match actual record counters")
    groups = {}
    for row in rows:
        row.update(
            diagnostic_io_kind=None,
            diagnostic_io_empty=None,
            diagnostic_io_records=None,
            diagnostic_io_bytes=None,
            diagnostic_io_layer=None,
            diagnostic_io_evidence=None,
        )
        name = row["name"].lower()
        if row["kind"] != "kernel":
            continue
        if "gather_records" in name:
            kind, stage = "exact_recall", "offload_exact_recall"
            if sample["scheme"] not in {"echo", "serial_sparse"}:
                raise ValueError("recall counters may include another transport path")
        elif "sm90_fp8_mqa_logits_fuse_prefetch" in name:
            kind, stage = "fused_prefetch", "indexer_prefetch"
        else:
            continue
        scope = LAYER_SCOPE.fullmatch(row["scope"]["stage"])
        if scope is None or scope[2] != stage or int(scope[1]) not in checks:
            raise ValueError("transport kernel lacks a matching per-layer production scope")
        groups.setdefault((int(scope[1]), kind), []).append(row)
    for (layer, kind), kernels in groups.items():
        field = "recalled_records" if kind == "exact_recall" else "prefetched_records"
        records = metrics[layer][field]
        evidence = f"result.json:samples[{sample_index}].metrics[{layer}].{field}"
        for row in kernels:
            row.update(diagnostic_io_kind=kind, diagnostic_io_layer=layer)
        if len(kernels) != 1:
            if kind == "exact_recall":
                raise ValueError("multiple recall kernels require per-call transport counters")
            # The layer total does not identify a unique fused invocation.
            # Preserve its unresolved compute/IO bounds instead of guessing.
            for row in kernels:
                row["diagnostic_io_evidence"] = f"unresolved: {evidence}; {len(kernels)} calls"
            continue
        kernels[0].update(
            diagnostic_io_empty=records == 0,
            diagnostic_io_records=records,
            diagnostic_io_bytes=records * metrics[layer]["record_bytes"],
            diagnostic_io_evidence=evidence,
        )


def diagnostic_gap_metrics(rows, start, end):
    """Use the complete replay window, with no complete-model acceptance gate."""
    metrics = summarize_window(rows, start, end, lane_classifier=diagnostic_lane)
    excluded = {
        "start_ns",
        "end_ns",
        "window_ms",
        "threshold_percent",
        "gap_below_threshold",
        "conservative_gate_pass",
        "gate_certifiable",
        "gate_uncertainty",
        "unresolved_gather_count",
    }
    return {
        **{f"diagnostic_{key}": value for key, value in metrics.items() if key not in excluded},
        "diagnostic_empty_gather_count": sum(
            row.get("diagnostic_io_kind") == "exact_recall"
            and row.get("diagnostic_io_empty") is True
            for row in rows
        ),
        "diagnostic_zero_io_fused_count": sum(
            row.get("diagnostic_io_kind") == "fused_prefetch"
            and row.get("diagnostic_io_empty") is True
            for row in rows
        ),
    }


def interval_metrics(rows, start, end):
    if end <= start:
        raise ValueError("profile window must be nonempty")

    def intervals(categories=None):
        return [
            (max(start, row["start"]), min(end, row["end"]))
            for row in rows
            if categories is None or row["category"] in categories
        ]

    width = end - start
    busy = _union_ns(intervals())
    io = _union_ns(intervals({"io"}))
    indexer = _union_ns(intervals({"indexer_compute", "indexer_compute_and_io"}))
    covered = _union_ns(intervals({"io", "indexer_compute", "indexer_compute_and_io"}))
    manager = _union_ns(intervals({"cache_management", "d2d_control", "selection"}))
    return {
        "window_ms": width / 1e6,
        "gpu_busy_union_ms": busy / 1e6,
        "gpu_idle_ms": (width - busy) / 1e6,
        "known_io_union_ms": io / 1e6,
        "indexer_union_ms": indexer / 1e6,
        "manager_selection_control_union_ms": manager / 1e6,
        "outside_indexer_and_io_ms": (width - covered) / 1e6,
        "window_minus_known_io_ms": (width - io) / 1e6,
        "fused_internal_io_unresolved": any(
            row["category"] == "indexer_compute_and_io" for row in rows
        ),
        "mfu": None,
        "whole_model_gap_ratio": None,
    }


def analyze(sqlite_path, profile_run, output):
    sqlite_path, profile_run, output = (
        Path(path).resolve() for path in (sqlite_path, profile_run, output)
    )
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((profile_run / "result.json").read_text())
    if metadata.get("mode") != "profile" or metadata.get("passed") is not True:
        raise ValueError("profile run did not finish its independent output checks")
    scopes, apis, activities, tables = _read_capture(sqlite_path, scope_pattern=SCOPE)
    _assign_scopes(apis, scopes)
    _attribute(apis, activities)
    windows = {
        (scope["phase"], scope["scheme"]): scope for scope in scopes if scope["stage"] == "window"
    }
    expected = {(phase, scheme) for phase in PHASES for scheme in SCHEMES}
    samples = {
        (sample["phase"], sample["scheme"]): (index, sample)
        for index, sample in enumerate(metadata["samples"])
    }
    if set(samples) != expected or len(samples) != len(metadata["samples"]):
        raise ValueError("profile result must contain one counter sample per measured window")
    if set(windows) != expected or len([s for s in scopes if s["stage"] == "window"]) != len(
        expected
    ):
        raise ValueError("profile must contain exactly one window for every scheme/phase")
    raw, summaries = [], []
    for phase, scheme in sorted(expected):
        window = windows[(phase, scheme)]
        rows = [
            row
            for row in activities
            if row["scope"] is not None
            and row["scope"]["phase"] == phase
            and row["scope"]["scheme"] == scheme
        ]
        if not rows:
            raise ValueError("profile window has no correlated GPU activities")
        # Synchronized transaction completion must include every submitted node.
        if any(row["start"] < window["start"] or row["end"] > window["end"] for row in rows):
            raise ValueError("GPU activity escaped its synchronized replay window")
        unexplained = [
            row
            for row in activities
            if row["scope"] is None
            and row["start"] < window["end"]
            and row["end"] > window["start"]
        ]
        if unexplained:
            raise ValueError("unattributed GPU activity overlaps the measured window")
        sample_index, sample = samples[(phase, scheme)]
        annotate_diagnostic_io(rows, sample, sample_index)
        for row in rows:
            row["category"] = category(row)
            raw.append(
                {
                    "phase": phase,
                    "scheme": scheme,
                    "kind": row["kind"],
                    "category": row["category"],
                    "diagnostic_lane": diagnostic_lane(row),
                    **{key: row[key] for key in row if key.startswith("diagnostic_io_")},
                    "name": row["name"],
                    "start_ns": row["start"],
                    "end_ns": row["end"],
                    "stream": row["stream_id"],
                    "device": row["device_id"],
                    "process": row["process"],
                    "correlation": row["correlation"],
                    "bytes": row["bytes"],
                    "scope": row["scope"]["stage"],
                }
            )
        summaries.append(
            {
                "phase": phase,
                "scheme": scheme,
                "start_ns": window["start"],
                "end_ns": window["end"],
                "activities": len(rows),
                **interval_metrics(rows, window["start"], window["end"]),
                **diagnostic_gap_metrics(rows, window["start"], window["end"]),
            }
        )
    with (output / "activities.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    helper_paths = (
        Path(__file__).resolve(),
        Path(_read_capture.__code__.co_filename).resolve(),
        Path(summarize_window.__code__.co_filename).resolve(),
        Path(activity_lane.__code__.co_filename).resolve(),
    )
    helper_archive = output / "analysis_sources"
    helper_archive.mkdir()
    helper_sources, helper_archives = {}, {}
    for path in helper_paths:
        destination = helper_archive / path.name
        destination.write_bytes(path.read_bytes())
        helper_sources[str(path)] = file_sha256(destination)
        helper_archives[str(path)] = str(destination.relative_to(output))
    identity = json.loads((profile_run / "identity.json").read_text())
    counter_sources = {}
    for relative in (
        "cache/sparse_token_cache.py",
        "operators/deepseek_v32/indexer/csrc/echo_cache.cuh",
        "operators/common/csrc/kv_transfer.cu",
        "experiments/cache_manager_performance/src/workload.py",
    ):
        digest = file_sha256(profile_run / "source" / relative)
        if digest != identity["sources"][relative]:
            raise ValueError("transport counter source differs from its execution identity")
        counter_sources[relative] = digest
    report = {
        "schema": "cache-manager-profile-v4",
        "run_id": metadata["run_id"],
        "profile_identity_sha256": metadata["identity_sha256"],
        "sqlite": {
            "path": os.path.relpath(sqlite_path, output),
            "sha256": file_sha256(sqlite_path),
        },
        "profile_result_sha256": file_sha256(profile_run / "result.json"),
        "analyzer_sources": helper_sources,
        "analyzer_source_archives": helper_archives,
        "diagnostic_io_counter_sources": counter_sources,
        "tables": tables,
        "samples": summaries,
        "definitions": {
            "window": "CPU NVTX range including manager submission and transaction completion",
            "outside_indexer_and_io": "window minus interval union of indexer compute, fused indexer/prefetch, and known pure IO; includes selection, cache metadata, D2D/control, and launch idle",
            "known_io": "host-direction memcpy or mapped-host gather with transport; checked zero-record gather calls are cache management in all categories and interval metrics",
            "fused": "checked zero-record fused prefetch is retained whole as indexer compute; nonzero or ambiguous fused transport remains compute+IO without invented internal decomposition",
            "causal_mask": "official CompareFunctor/masked_fill or _resident_causal_tail_mask under indexer, and fused clean, are indexer compute; bound construction and plain fill/reset remain management",
            "diagnostic_gap": "complete replay window minus the interval union of indexer and exact top-k mathematical compute, known host IO, and fused indexer/IO; retains idle and exposed GPU control",
            "diagnostic_compute": "exact top-k math is compute; same-dtype score packing, plain fills, cache FIFO sorting and metadata remain control under the shared MFU classifier",
            "diagnostic_actual_io": "unique per-layer exact-recall and fused-prefetch kernels bind synchronized record counters checked against the independent profile oracle; zero-record gather is control, zero-record fused is retained whole as compute, and nonzero/ambiguous fused calls retain unresolved compute/IO bounds",
            "actual_io_revision": "counter-based zero-IO classification applies to category, known_io_union_ms, outside_indexer_and_io_ms and diagnostic fields alike; raw activity timestamps and launch attribution are unchanged",
            "diagnostic_non_io_bounds": "shared MFU overlap-aware formula: lower ratio retains fused intervals; upper ratio removes IO/fused time outside standalone compute, without inventing fused internal IO duration",
            "diagnostic_scope": "standalone replay only; no complete-model threshold or gate; pure recall without model compute is evaluated by absolute wall/enqueue latency, not the complete-model gap percentage",
            "mfu": "N/A; this experiment does not execute the complete model",
            "gate": "diagnostic cache replay only; whole-model extend/prefill gates are evaluated in deepseek_v32_mfu",
        },
        "audit": {
            "all_windows_present": True,
            "all_window_activity_correlated": True,
            "duration_intervals_valid": True,
            "numerical_profile_checks_passed": True,
            "transport_counters_match_checked_layers": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--profile-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze(args.sqlite, args.profile_run, args.output_dir)
    print(json.dumps(report["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
