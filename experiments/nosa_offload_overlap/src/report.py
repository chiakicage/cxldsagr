"""Select verified NOSA offload timings and GPU overlap metrics for publication.

Unprofiled CUDA events provide latency; a separate matched Nsight run provides
GPU interval intersections. Unique logical payload bytes divided by copy windows
are effective payload rates, not measured PCIe/CXL wire bandwidth or read traffic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

from evaluation.validation import require_receipt
from experiments.nosa_mfu.src.phases import offload_config, validation_identity

MODES = ("resident", "serialized", "overlap")
TRAFFIC_CHECKS = (
    "CPU per-head first-use union equals GPU total and per-tile counts after "
    "validation, warmup and every measured iteration"
)
PROFILE_METRICS = (
    "fetch_union_us",
    "attention_union_us",
    "overlap_us",
    "fetch_without_attention_us",
    "fetch_hidden_fraction",
    "gpu_span_us",
)
WORK_METRICS = (
    "fetch_work_union_us",
    "softmax_union_us",
    "fetch_math_overlap_us",
    "fetch_without_softmax_us",
    "fetch_math_overlap_fraction",
    "instrumented_span_us",
)
STRIPE_METRICS = (
    "fetch_stripe_union_us",
    "fetch_stripe_math_overlap_us",
    "fetch_stripe_without_softmax_us",
    "fetch_stripe_math_overlap_fraction",
    "page_envelope_only_union_us",
    "page_envelope_only_math_overlap_us",
    "page_minus_stripe_math_overlap_fraction",
    "stripe_interval_count",
)


def _read(path):
    return json.loads(path.read_text())


def _sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, *, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _distribution(values, *, signed=False):
    if all(value is None for value in values):
        return None
    _require(
        all(
            type(value) in (int, float) and math.isfinite(value) and (signed or value >= 0)
            for value in values
        ),
        "Invalid numerical samples",
    )
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def _load_run(directory, *, profiled):
    metadata = _read(directory / "metadata.json")
    results = _read(directory / "results.json")
    _require(metadata.get("schema_version") == 1, "Unsupported measurement schema")
    _require(metadata["run_id"] == results["run_id"], "Result run ID mismatch")
    _require(metadata["args"]["run_id"] == metadata["run_id"], "Argument run ID mismatch")
    _require(
        metadata["measurement"]["profiled"] is profiled
        and metadata["args"]["profiled"] is profiled,
        f"Expected profiled={profiled}: {directory}",
    )
    _require(not metadata["args"]["synthetic"], "Publication requires captured model inputs")
    _require(metadata.get("source_sha256"), "Missing source fingerprints")
    for name, digest in metadata["source_sha256"].items():
        source = directory / "sources" / name
        _require(
            source.resolve().is_relative_to((directory / "sources").resolve()), "Bad source path"
        )
        _require(_sha256(source) == digest, f"Source snapshot hash mismatch: {name}")
    capture = metadata["input_capture"]
    _require(capture.get("metadata_sha256"), "Missing input-capture hash")
    _require(
        capture["metadata"]["kind"] == "actual_sparse_model_operator_inputs",
        "Expected actual sparse model capture",
    )
    checked_cases = None
    if "mode" in metadata:
        _require(
            metadata["mode"] == ("profile" if profiled else "bench"),
            "Only bench/profile results can support a performance report",
        )
        provenance = metadata["numerical_validation"]
        _require(provenance["source"] == "independent_check", "Missing independent acceptance")
        path = directory / provenance["file"]
        _require(path.resolve().is_relative_to(directory.resolve()), "Bad receipt path")
        _require(_sha256(path) == provenance["sha256"], "Receipt file hash mismatch")
        identity = validation_identity(
            metadata,
            config=offload_config(SimpleNamespace(**metadata["args"])),
            inputs={
                "metadata_sha256": capture["metadata_sha256"],
                "files": {
                    entry["file"]: entry["file_sha256"] for entry in capture["metadata"]["layers"]
                },
            },
            cache="cold_history_no_tags_full_logical_staging_gpu_suffix",
        )
        _require(identity == metadata["validation_identity"], "Validation identity mismatch")
        receipt = require_receipt(path, kind="nosa_offload_attention", identity=identity)
        checked_cases = receipt["checks"]["cases"]
    cases = {case["case"]: case for case in results["results"]}
    _require(len(cases) == len(results["results"]) > 0, "Empty or duplicate cases")
    for label, case in cases.items():
        _require(case["queries"] == metadata["args"]["queries"], f"Query count mismatch: {label}")
        _require(case["tile_size"] == metadata["args"]["tile_size"], f"Tile mismatch: {label}")
        if "fetch_ctas" in metadata["args"]:
            _require(
                case.get("fetch_ctas") == metadata["args"]["fetch_ctas"],
                f"Fetch CTA mismatch: {label}",
            )
        _require(case["input_kind"] == "captured_actual_sparse_model_inputs", "Wrong input kind")
        _require(case["serialized_overlap_exact_equal"] is True, f"Exact check failed: {label}")
        if checked_cases is None:
            _require(case["traffic_checks"] == TRAFFIC_CHECKS, f"Missing traffic checks: {label}")
        else:
            checked = checked_cases.get(label)
            _require(checked is not None, f"Missing independent case acceptance: {label}")
            for key in (
                "tensors",
                "acceptance",
                "prefix_transfer_bytes",
                "first_use_tile_bytes",
                "serialized_overlap_exact_equal",
            ):
                _require(case[key] == checked[key], f"Acceptance {key} mismatch: {label}")
            _require(
                case["numerical_acceptance_source"] == "independent_check",
                f"Missing acceptance source: {label}",
            )
            _require(
                case["traffic_checks"]
                == ("diagnostic_profile" if profiled else "independent_check"),
                f"Wrong traffic check phase: {label}",
            )
        tiles = case["first_use_tile_bytes"]
        _require(
            len(tiles) == (case["queries"] + case["tile_size"] - 1) // case["tile_size"]
            and all(type(value) is int and value >= 0 for value in tiles)
            and sum(tiles) == case["prefix_transfer_bytes"] <= case["dense_prefix_bytes"],
            f"Invalid first-use traffic accounting: {label}",
        )
        acceptance = case["acceptance"]
        rows = acceptance["rows"]
        _require(
            bool(acceptance["reference"])
            and bool(rows)
            and len(set(rows)) == len(rows)
            and all(type(row) is int and 0 <= row < case["queries"] for row in rows)
            and acceptance["rtol"] == acceptance["atol"] == 0.016
            and set(acceptance["errors"]) == set(MODES),
            f"Incomplete reference acceptance: {label}",
        )
        if checked_cases is not None or metadata["args"]["reference_all"]:
            _require(set(rows) == set(range(case["queries"])), f"Reference rows missing: {label}")
        for error in acceptance["errors"].values():
            _require(
                _number(error["max_abs"]) and _number(error["relative_l2"]),
                f"Invalid acceptance errors: {label}",
            )
        _require(set(case["modes"]) == set(MODES), f"Incomplete modes: {label}")
        for mode in MODES:
            for metric in ("cuda_ms", "wall_ms", "submit_ms"):
                timing = case["modes"][mode][metric]
                samples = timing["samples"]
                _require(
                    len(samples) == metadata["args"]["repeats"]
                    and all(_number(value, positive=True) for value in samples),
                    f"Invalid timing samples: {label}/{mode}/{metric}",
                )
                _require(
                    all(timing[key] == value for key, value in _distribution(samples).items()),
                    f"Timing summary differs from samples: {label}/{mode}/{metric}",
                )
    return metadata, cases


def _fused_report(measurement_dir, profile_dir, measurement, measured, profile, profiled, timeline):
    """Keep role-level evidence separate from whole-kernel CUDA intervals."""
    from experiments.nosa_offload_overlap.src.analyze import (
        stripe_interval_metrics,
        work_interval_metrics,
        work_profile_metadata,
    )

    fetch_ctas = measurement["args"].get("fetch_ctas")
    _require(
        type(fetch_ctas) is int
        and fetch_ctas > 0
        and fetch_ctas == profile["args"].get("fetch_ctas"),
        "Measurement/profile fetch CTA configuration mismatch",
    )
    phase_path = profile_dir / "work_intervals.json"
    _require(
        _sha256(phase_path) == timeline["work_intervals_sha256"], "Work interval hash mismatch"
    )
    phases = _read(phase_path)
    work_metadata = work_profile_metadata(phases)
    has_stripes = phases["schema_version"] == 3
    _require(
        timeline["schema_version"] == (3 if has_stripes else 2),
        "Timeline schema differs from raw work profile",
    )
    _require(
        phases["run_id"] == profile["run_id"],
        "Wrong work interval provenance or clock",
    )
    if phases["schema_version"] >= 2:
        for key, value in work_metadata.items():
            _require(timeline.get(key) == value, "Work interval semantics differ from raw profile")
    if has_stripes:
        for metadata in (measurement, profile):
            _require(
                metadata["native_build"].get("offload_fused", {}).get("fetch_stripes")
                == phases["fetch_stripes"],
                "Work interval stripe count differs from compiled native build",
            )
    work = {(r["case"], r["mode"], r["sample"]): r for r in phases["records"]}
    _require(len(work) == len(phases["records"]), "Duplicate work interval samples")
    expected = {
        (label, mode, sample)
        for label in measured
        for mode in MODES
        for sample in range(profile["args"]["repeats"])
    }
    records = {(r["case"], r["mode"], r["sample"]): r for r in timeline["records"]}
    _require(
        len(records) == len(timeline["records"]) and records.keys() == expected,
        "Incomplete fused timeline samples",
    )
    _require(
        work.keys() == {key for key in expected if key[1] != "resident"},
        "Incomplete work interval samples",
    )
    for key, row in records.items():
        label, mode, _ = key
        _require(_number(row["gpu_span_us"], positive=True), "Invalid GPU span")
        if mode == "overlap":
            _require(
                row["profile_method"] == "globaltimer_fetch_softmax"
                and row["fused_main_count"] == 1
                and row["kernel_intervals"] is None
                and _number(row["fused_main_us"], positive=True),
                "Fused candidate must have one main kernel and role-level evidence",
            )
        else:
            kernel = row["kernel_intervals"]
            _require(
                row["fused_main_count"] == 0 and kernel is not None,
                "Whole-query controls must use separate kernel evidence",
            )
            fetch, attention, overlap = (kernel[name] for name in PROFILE_METRICS[:3])
            _require(
                _number(fetch)
                and _number(attention, positive=True)
                and overlap == 0
                and kernel["fetch_without_attention_us"] == fetch
                and kernel["fetch_hidden_fraction"] == (None if fetch == 0 else 0),
                "Control kernel interval accounting is invalid or has overlap",
            )
            _require(
                fetch == 0 if mode == "resident" else fetch > 0,
                "Unexpected control fetch intervals",
            )
        if mode != "resident":
            case = phases["cases"][label]
            for field in ("prefix", "queries"):
                _require(case[field] == measured[label][field], "Work interval geometry mismatch")
            _require(
                case["expected_prefix_bytes"] == measured[label]["prefix_transfer_bytes"],
                "Work interval byte provenance mismatch",
            )
            derived = work_interval_metrics(case, work[key])
            _require(derived == row["work_metrics"], "Work interval metrics differ from raw events")
            if has_stripes:
                stripes = stripe_interval_metrics(case, work[key], phases["fetch_stripes"])
                _require(
                    "stripe_metrics" in row and stripes == row["stripe_metrics"],
                    "Stripe interval metrics differ from raw events",
                )
        else:
            _require(row["work_metrics"] is None, "Resident must not claim fused work metrics")
            if has_stripes:
                _require(
                    "stripe_metrics" in row and row["stripe_metrics"] is None,
                    "Resident must not claim stripe metrics",
                )
    selected, comparison = [], []
    for label, case in measured.items():
        for field in (
            "prefix",
            "queries",
            "tile_size",
            "input_kind",
            "tensors",
            "prefix_transfer_bytes",
            "first_use_tile_bytes",
            "dense_prefix_bytes",
            "suffix_gpu_bytes",
        ):
            _require(case[field] == profiled[label][field], f"Case {label}: {field} mismatch")
        modes = {}
        for mode in MODES:
            group = [records[label, mode, sample] for sample in range(profile["args"]["repeats"])]
            kernel = (
                None
                if mode == "overlap"
                else {
                    metric: _distribution([row["kernel_intervals"][metric] for row in group])
                    for metric in PROFILE_METRICS
                    if metric != "gpu_span_us"
                }
            )
            intra = None
            if mode == "overlap":
                intra = {
                    metric: _distribution([row["work_metrics"][metric] for row in group])
                    for metric in WORK_METRICS
                }
                intra["effective_fetch_work_GB_s"] = _distribution(
                    [
                        case["prefix_transfer_bytes"]
                        / (row["work_metrics"]["fetch_work_union_us"] * 1000)
                        for row in group
                    ]
                )
            modes[mode] = {
                "timing_ms": {
                    metric: {k: v for k, v in values.items() if k != "samples"}
                    for metric, values in case["modes"][mode].items()
                },
                "profile": {
                    "method": group[0]["profile_method"],
                    "gpu_span_us": _distribution([row["gpu_span_us"] for row in group]),
                    "kernel_intervals": kernel,
                    "intra_kernel_softmax": intra,
                },
            }
            if has_stripes:
                modes[mode]["profile"]["intra_kernel_stripes"] = (
                    {
                        metric: _distribution(
                            [row["stripe_metrics"][metric] for row in group],
                            signed=metric == "page_minus_stripe_math_overlap_fraction",
                        )
                        for metric in STRIPE_METRICS
                    }
                    if mode == "overlap"
                    else None
                )
        serial = modes["serialized"]["timing_ms"]["cuda_ms"]["median"]
        candidate = modes["overlap"]["timing_ms"]["cuda_ms"]["median"]
        item = {
            field: case[field]
            for field in (
                "case",
                "prefix",
                "queries",
                "tile_size",
                "input_kind",
                "tensors",
                "prefix_transfer_bytes",
                "first_use_tile_bytes",
                "dense_prefix_bytes",
                "suffix_gpu_bytes",
            )
        }
        item.update(
            fetch_ctas=fetch_ctas,
            execution_geometry="whole-query attention in both offload modes; tile_size is histogram width",
            serialized_over_overlap_speedup=serial / candidate,
            latency_reduction_fraction=1 - candidate / serial,
            acceptance={
                "measurement": case["acceptance"],
                "profile": profiled[label]["acceptance"],
                "serialized_overlap_exact_equal": True,
                "traffic_checks": {
                    "measurement": case["traffic_checks"],
                    "profile": profiled[label]["traffic_checks"],
                },
            },
            modes=modes,
        )
        if has_stripes:
            candidate_profile = modes["overlap"]["profile"]
            page_fractions = candidate_profile["intra_kernel_softmax"][
                "fetch_math_overlap_fraction"
            ]
            stripe_fractions = candidate_profile["intra_kernel_stripes"][
                "fetch_stripe_math_overlap_fraction"
            ]
            sample_evidence = []
            for sample in range(profile["args"]["repeats"]):
                sample_row = records[label, "overlap", sample]
                page_fraction = sample_row["work_metrics"]["fetch_math_overlap_fraction"]
                stripe_fraction = sample_row["stripe_metrics"]["fetch_stripe_math_overlap_fraction"]
                sample_evidence.append(
                    {
                        "sample": sample,
                        "page_fraction": page_fraction,
                        "stripe_fraction": stripe_fraction,
                        "meets_threshold": page_fraction >= 0.9 and stripe_fraction >= 0.9,
                    }
                )
            item["overlap_threshold_evidence"] = {
                "threshold": 0.9,
                "comparison": ">=",
                "page_and_stripe_medians_meet_threshold": min(
                    page_fractions["median"], stripe_fractions["median"]
                )
                >= 0.9,
                "page_and_stripe_all_samples_meet_threshold": all(
                    sample["meets_threshold"] for sample in sample_evidence
                ),
                "samples_meeting_threshold": sum(
                    sample["meets_threshold"] for sample in sample_evidence
                ),
                "samples": sample_evidence,
            }
        selected.append(item)
        row = {
            field: case[field] for field in ("case", "prefix", "queries", "prefix_transfer_bytes")
        }
        row["fetch_ctas"] = fetch_ctas
        row.update(
            {f"{mode}_cuda_ms": modes[mode]["timing_ms"]["cuda_ms"]["median"] for mode in MODES}
        )
        row.update(
            serialized_over_overlap_speedup=serial / candidate,
            latency_reduction_fraction=1 - candidate / serial,
            serialized_kernel_overlap_us=modes["serialized"]["profile"]["kernel_intervals"][
                "overlap_us"
            ]["median"],
        )
        row.update(
            {
                f"overlap_{metric}": values["median"]
                for metric, values in modes["overlap"]["profile"]["intra_kernel_softmax"].items()
            }
        )
        if has_stripes:
            row.update(
                {
                    f"overlap_{metric}": values["median"]
                    for metric, values in modes["overlap"]["profile"][
                        "intra_kernel_stripes"
                    ].items()
                }
            )
            row.update(
                {
                    key: value
                    for key, value in item["overlap_threshold_evidence"].items()
                    if key != "samples"
                }
            )
        comparison.append(row)
    sources = {}
    for name, directory, metadata in (
        ("measurement", measurement_dir, measurement),
        ("profile", profile_dir, profile),
    ):
        filenames = ["metadata.json", "results.json"]
        if "numerical_validation" in metadata:
            filenames.append(metadata["numerical_validation"]["file"])
        if name == "profile":
            filenames += ["overlap.json", "timeline.sqlite", "work_intervals.json"]
        sources[name] = {
            "run_id": metadata["run_id"],
            "recorded_at_utc": metadata["recorded_at_utc"],
            "args": metadata["args"],
            "numerical_validation": metadata.get("numerical_validation"),
            "artifact_sha256": {filename: _sha256(directory / filename) for filename in filenames},
        }
    report = {
        "schema_version": 3 if has_stripes else 2,
        "sources": sources,
        **{
            key: measurement[key]
            for key in (
                "gpu",
                "torch",
                "cuda",
                "dependencies",
                "native_build",
                "source_sha256",
                "input_capture",
                "git_commit",
            )
        },
        "measurement_boundary": measurement["measurement"],
        "profile_definition": timeline["definition"],
        **work_metadata,
        "fetch_math_overlap_definition": "Union intersection of instrumented copy windows and softmax-update intervals on device globaltimer, using the separately stated fetch definition; a lower bound on copy-window overlap with attention math, not full attention hidden time or PCIe/CXL wire occupancy. For batched copies, rows share batch start and their union counts the batch copy window once. Nsight separately verifies one fused main per candidate call and zero kernel overlap in the whole-query serialized control.",
        "effective_fetch_work_GB_s_definition": "Unique logical selected K+V payload bytes / union of instrumented copy windows, decimal GB/s. Batched rows share a start and have separate per-page completion times; their union measures batch copy windows, not independent page load latencies. Payload accounting does not establish physical host-read bytes or exclude hardware rereads. This is not PCIe/CXL wire bandwidth. Globaltimer and Nsight absolute timestamps are never mixed.",
        "aggregation": "Uninstrumented CUDA-event medians determine speedup against optimized whole-query sparse-union fetch plus full attention. Instrumented profile intervals determine role concurrency only; per-sample ratios precede median/min/max aggregation.",
        "cases": selected,
    }
    if has_stripes:
        report.update(
            fetch_math_overlap_definition="The original page-envelope union intersected with softmax-update intervals, divided by page-envelope union, on device globaltimer. Each envelope is exactly the min/max of its nonempty stripes and can contain gaps between them. This remains a lower bound on envelope overlap with attention math, not proof of stripe-copy overlap or wire occupancy. Nsight separately verifies one fused main per candidate call and zero kernel overlap in the serialized control.",
            fetch_stripe_math_overlap_definition="Union of all validated nonempty stripe copy windows intersected with the same softmax-update union, divided by stripe-window union. Every expected (page slot, stripe) occurs exactly once with its actual historical K+V bytes. Concurrency is counted once; global gaps present only in page envelopes are excluded. Softmax is a subset of attention math; this does not measure wire occupancy or full attention hidden time.",
            page_stripe_difference_definition="Page-envelope union minus stripe union, and their corresponding softmax intersections, in microseconds. The fraction difference is page fraction minus stripe fraction and may be positive or negative; an envelope-only ratio is not a one-sided bound on the stripe ratio.",
            overlap_threshold_definition="The >=90% acceptance check requires both the original page-envelope and independently validated stripe-copy ratios to be >=0.9 in every profiled sample of each case. Ratios and pass/fail are computed per sample before aggregation. Median checks are reported separately and cannot replace the all-sample check; envelope evidence alone cannot pass. These are measured window/softmax intersections, not a 50 GB/s model or wire-occupancy measurement.",
            effective_fetch_work_GB_s_definition="Unique logical selected K+V payload bytes / union of page envelopes, decimal GB/s. Envelopes can contain gaps between stripe windows. This retained page metric is not physical host-read traffic, PCIe/CXL wire bandwidth or an independent stripe-copy rate. Globaltimer and Nsight absolute timestamps are never mixed.",
        )
    return report, comparison


def build_report(measurement_dir, profile_dir):
    measurement, measured = _load_run(measurement_dir, profiled=False)
    profile, profiled = _load_run(profile_dir, profiled=True)
    _require(measurement["run_id"] != profile["run_id"], "Use independent measurement/profile runs")
    for key in ("source_sha256", "gpu", "torch", "cuda", "dependencies", "native_build"):
        _require(measurement[key] == profile[key], f"Measurement/profile {key} mismatch")
    for key in ("metadata_sha256", "metadata"):
        _require(
            measurement["input_capture"][key] == profile["input_capture"][key],
            f"Input capture {key} mismatch",
        )
    _require(measured.keys() == profiled.keys(), "Measurement/profile case mismatch")
    timeline = _read(profile_dir / "overlap.json")
    _require(timeline["schema_version"] in (1, 2, 3), "Unsupported overlap schema")
    _require(timeline["run_id"] == profile["run_id"], "Timeline run ID mismatch")
    _require(
        _sha256(profile_dir / "timeline.sqlite") == timeline["sqlite_sha256"],
        "Timeline SQLite hash mismatch",
    )
    if timeline["schema_version"] >= 2:
        return _fused_report(
            measurement_dir, profile_dir, measurement, measured, profile, profiled, timeline
        )
    groups = {(case, mode): [] for case in measured for mode in MODES}
    for record in timeline["records"]:
        key = record["case"], record["mode"]
        _require(key in groups, f"Unexpected timeline case/mode: {key}")
        fetch, attention, overlap = (record[name] for name in PROFILE_METRICS[:3])
        _require(
            all(_number(value) for value in (fetch, attention, overlap))
            and attention > 0
            and overlap <= min(fetch, attention)
            and math.isclose(record["fetch_without_attention_us"], fetch - overlap, abs_tol=1e-9),
            f"Invalid interval accounting: {key}",
        )
        fraction = record["fetch_hidden_fraction"]
        _require(
            fraction is None
            if fetch == 0
            else _number(fraction) and math.isclose(fraction, overlap / fetch, abs_tol=1e-12),
            f"Invalid hidden fraction: {key}",
        )
        if record["mode"] == "resident":
            _require(fetch == overlap == 0, "Resident control has host fetch")
        else:
            _require(fetch > 0, "Missing offload fetch intervals")
        if record["mode"] == "serialized":
            _require(overlap == 0, "Serialized control has GPU overlap")
        groups[key].append(record)
    selected, comparison = [], []
    for label, case in measured.items():
        for key in (
            "prefix",
            "queries",
            "tile_size",
            "input_kind",
            "tensors",
            "prefix_transfer_bytes",
            "first_use_tile_bytes",
            "dense_prefix_bytes",
            "suffix_gpu_bytes",
        ):
            _require(case[key] == profiled[label][key], f"Case {label}: {key} mismatch")
        modes = {}
        for mode in MODES:
            records = groups[label, mode]
            _require(
                len(records) == profile["args"]["repeats"]
                and {row["sample"] for row in records} == set(range(profile["args"]["repeats"])),
                f"Incomplete timeline samples: {label}/{mode}",
            )
            metrics = {
                metric: _distribution([row[metric] for row in records])
                for metric in PROFILE_METRICS
            }
            metrics["effective_fetch_GB_s"] = (
                None
                if mode == "resident"
                else _distribution(
                    [
                        case["prefix_transfer_bytes"] / (row["fetch_union_us"] * 1000)
                        for row in records
                    ]
                )
            )
            modes[mode] = {
                "timing_ms": {
                    metric: {key: value for key, value in values.items() if key != "samples"}
                    for metric, values in case["modes"][mode].items()
                },
                "profile": metrics,
            }
        speedup = (
            case["modes"]["serialized"]["cuda_ms"]["median"]
            / case["modes"]["overlap"]["cuda_ms"]["median"]
        )
        selected.append(
            {
                **{
                    key: case[key]
                    for key in (
                        "case",
                        "prefix",
                        "queries",
                        "tile_size",
                        "input_kind",
                        "tensors",
                        "prefix_transfer_bytes",
                        "first_use_tile_bytes",
                        "dense_prefix_bytes",
                        "suffix_gpu_bytes",
                    )
                },
                "serialized_over_overlap_speedup": speedup,
                "acceptance": {
                    "measurement": case["acceptance"],
                    "profile": profiled[label]["acceptance"],
                    "serialized_overlap_exact_equal": True,
                    "traffic_checks": {
                        "measurement": case["traffic_checks"],
                        "profile": profiled[label]["traffic_checks"],
                    },
                },
                "modes": modes,
            }
        )
        row = {
            key: case[key]
            for key in (
                "case",
                "prefix",
                "queries",
                "tile_size",
                "prefix_transfer_bytes",
                "dense_prefix_bytes",
            )
        }
        row.update(
            {f"{mode}_cuda_ms": modes[mode]["timing_ms"]["cuda_ms"]["median"] for mode in MODES}
        )
        row["serialized_over_overlap_speedup"] = speedup
        for mode in ("serialized", "overlap"):
            for metric in (*PROFILE_METRICS, "effective_fetch_GB_s"):
                row[f"{mode}_{metric}"] = modes[mode]["profile"][metric]["median"]
        comparison.append(row)
    artifacts = {}
    for name, directory, metadata in (
        ("measurement", measurement_dir, measurement),
        ("profile", profile_dir, profile),
    ):
        artifacts[name] = {
            "run_id": metadata["run_id"],
            "recorded_at_utc": metadata["recorded_at_utc"],
            "args": metadata["args"],
            "numerical_validation": metadata.get("numerical_validation"),
            "artifact_sha256": {
                filename: _sha256(directory / filename)
                for filename in ["metadata.json", "results.json"]
                + (
                    [metadata["numerical_validation"]["file"]]
                    if "numerical_validation" in metadata
                    else []
                )
            },
        }
    artifacts["profile"]["artifact_sha256"].update(
        {
            "overlap.json": _sha256(profile_dir / "overlap.json"),
            "timeline.sqlite": timeline["sqlite_sha256"],
        }
    )
    return {
        "schema_version": 1,
        "sources": artifacts,
        **{
            key: measurement[key]
            for key in (
                "gpu",
                "torch",
                "cuda",
                "dependencies",
                "native_build",
                "source_sha256",
                "input_capture",
                "git_commit",
            )
        },
        "measurement_boundary": measurement["measurement"],
        "profile_definition": timeline["definition"],
        "effective_fetch_GB_s_definition": "Selected historical K+V bytes / union of fetch-kernel execution windows; decimal GB/s. Includes kernel scanning and concurrent resource contention, excludes launch gaps. Not PCIe/CXL wire bandwidth and not a 50 GB/s rate limit.",
        "aggregation": "Latency: median of unprofiled CUDA-event samples. Profile: per-sample GPU interval union/intersection, then median/min/max; ratios are computed per sample before aggregation. Speedup: serialized median / overlap median. No aggregation across layers or extrapolation to all 32 layers.",
        "cases": selected,
    }, comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report, rows = build_report(args.measurement_dir, args.profile_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.output_dir / name for name in ("report.json", "comparison.csv")]
    _require(not any(path.exists() for path in targets), "Report output already exists")
    with targets[0].open("x") as destination:
        json.dump(report, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    with targets[1].open("x", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} cases; selected report written to {args.output_dir}")


if __name__ == "__main__":
    main()
