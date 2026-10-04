"""Report admission and aggregation checks using temporary CPU-only artifacts."""

import hashlib
import json

import pytest

from experiments.nosa_offload_overlap.src.report import (
    MODES,
    TRAFFIC_CHECKS,
    build_report,
    main,
)


def _write(path, value):
    path.write_text(json.dumps(value))


def _timing(values):
    return {"samples": values, "median": sorted(values)[1], "min": min(values), "max": max(values)}


@pytest.fixture
def runs(tmp_path):
    measurement, profile = tmp_path / "measurement", tmp_path / "profile"
    for directory, profiled in ((measurement, False), (profile, True)):
        source = directory / "sources/operators/sm90/example.py"
        source.parent.mkdir(parents=True)
        source.write_text("source fixture\n")
        metadata = {
            "schema_version": 1,
            "run_id": directory.name,
            "recorded_at_utc": "2026-09-30T00:00:00+00:00",
            "args": {
                "run_id": directory.name,
                "profiled": profiled,
                "synthetic": False,
                "queries": 8,
                "tile_size": 8,
                "reference_all": True,
                "repeats": 3,
            },
            "measurement": {"profiled": profiled, "scope": "Single-layer replay"},
            "source_sha256": {
                "operators/sm90/example.py": hashlib.sha256(source.read_bytes()).hexdigest()
            },
            "input_capture": {
                "metadata_sha256": "capture-sha",
                "metadata": {"kind": "actual_sparse_model_operator_inputs", "query_start": 65536},
            },
            "gpu": {"name": "Hopper fixture", "uuid": "GPU-fixture", "capability": [9, 0]},
            "torch": "fixture",
            "cuda": "fixture",
            "dependencies": {"flashinfer-python": "fixture"},
            "native_build": {"selected_backend": "native"},
            "git_commit": "fixture",
        }
        case = {
            "case": "layer_00",
            "prefix": 65536,
            "queries": 8,
            "tile_size": 8,
            "input_kind": "captured_actual_sparse_model_inputs",
            "prefix_transfer_bytes": 65536,
            "first_use_tile_bytes": [65536],
            "dense_prefix_bytes": 67108864,
            "suffix_gpu_bytes": 8192,
            "serialized_overlap_exact_equal": True,
            "traffic_checks": TRAFFIC_CHECKS,
            "tensors": {"q": {"shape": [8, 32, 128], "dtype": "torch.bfloat16"}},
            "acceptance": {
                "reference": "Independent FP32 reference",
                "rows": list(range(8)),
                "rtol": 0.016,
                "atol": 0.016,
                "errors": {mode: {"max_abs": 0.001, "relative_l2": 0.001} for mode in MODES},
            },
            "modes": {
                mode: {
                    metric: _timing([median / 2, median, median * 2])
                    for metric in ("cuda_ms", "wall_ms", "submit_ms")
                }
                for mode, median in zip(MODES, (1.0, 3.0, 2.0), strict=True)
            },
        }
        _write(directory / "metadata.json", metadata)
        _write(directory / "results.json", {"run_id": directory.name, "results": [case]})
    records = []
    for sample in range(3):
        for mode in MODES:
            fetch = 0 if mode == "resident" else (sample + 1) * 3.0
            overlap = fetch / 2 if mode == "overlap" else 0
            records.append(
                {
                    "case": "layer_00",
                    "mode": mode,
                    "sample": sample,
                    "fetch_union_us": fetch,
                    "attention_union_us": 5.0,
                    "overlap_us": overlap,
                    "fetch_without_attention_us": fetch - overlap,
                    "fetch_hidden_fraction": overlap / fetch if fetch else None,
                    "gpu_span_us": 15.0,
                }
            )
    sqlite = profile / "timeline.sqlite"
    sqlite.write_bytes(b"profile fixture whose digest is verified")
    _write(
        profile / "overlap.json",
        {
            "schema_version": 1,
            "run_id": "profile",
            "sqlite_sha256": hashlib.sha256(sqlite.read_bytes()).hexdigest(),
            "definition": "GPU interval intersection fixture",
            "records": records,
        },
    )
    return measurement, profile


def test_report_separates_latency_from_profile_and_preserves_provenance(runs, tmp_path):
    report, rows = build_report(*runs)
    case = report["cases"][0]
    assert case["serialized_over_overlap_speedup"] == 1.5
    assert rows[0]["resident_cuda_ms"] == 1
    assert rows[0]["overlap_fetch_hidden_fraction"] == 0.5
    assert rows[0]["overlap_effective_fetch_GB_s"] == pytest.approx(65536 / 6000)
    assert case["modes"]["resident"]["profile"]["effective_fetch_GB_s"] is None
    assert report["sources"]["measurement"]["run_id"] == "measurement"
    assert report["sources"]["profile"]["run_id"] == "profile"
    assert "Not PCIe/CXL wire bandwidth" in report["effective_fetch_GB_s_definition"]
    output = tmp_path / "selected"
    argv = [
        "--measurement-dir",
        str(runs[0]),
        "--profile-dir",
        str(runs[1]),
        "--output-dir",
        str(output),
    ]
    main(argv)
    assert json.loads((output / "report.json").read_text()) == report
    assert "overlap_effective_fetch_GB_s" in (output / "comparison.csv").read_text()
    with pytest.raises(ValueError, match="already exists"):
        main(argv)


@pytest.fixture
def receipt_runs(runs):
    from types import SimpleNamespace

    from evaluation.validation import write_receipt
    from experiments.nosa_kernel_mfu.src.phases import offload_config, validation_identity

    for directory in runs:
        metadata = json.loads((directory / "metadata.json").read_text())
        result = json.loads((directory / "results.json").read_text())
        case = result["results"][0]
        metadata["mode"] = "profile" if metadata["args"]["profiled"] else "bench"
        metadata["args"].update(
            layers=[0],
            prefix=65536,
            seed=42,
            fetch_ctas=96,
            mode=metadata["mode"],
            reference_all=False,
        )
        metadata["input_capture"]["metadata"]["layers"] = [
            {"layer": 0, "file": "layer_00.pt", "file_sha256": "input-sha"}
        ]
        identity = validation_identity(
            metadata,
            config=offload_config(SimpleNamespace(**metadata["args"])),
            inputs={"metadata_sha256": "capture-sha", "files": {"layer_00.pt": "input-sha"}},
            cache="cold_history_no_tags_full_logical_staging_gpu_suffix",
        )
        path = directory / "validation_receipt.json"
        write_receipt(
            path,
            kind="nosa_offload_attention",
            identity=identity,
            checks={"passed": True, "cases": {"layer_00": case}},
        )
        metadata["validation_identity"] = identity
        metadata["numerical_validation"] = {
            "kind": "nosa_offload_attention",
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": "independent_check",
        }
        case.update(
            fetch_ctas=96,
            numerical_acceptance_source="independent_check",
            traffic_checks="diagnostic_profile"
            if metadata["args"]["profiled"]
            else "independent_check",
        )
        _write(directory / "metadata.json", metadata)
        _write(directory / "results.json", result)
    return runs


def test_report_reuses_independent_acceptance_and_preserves_receipt(receipt_runs):
    report, rows = build_report(*receipt_runs)
    assert rows[0]["serialized_over_overlap_speedup"] == 1.5
    source = report["sources"]["measurement"]
    assert source["numerical_validation"]["source"] == "independent_check"
    assert "validation_receipt.json" in source["artifact_sha256"]
    assert report["cases"][0]["acceptance"]["traffic_checks"] == {
        "measurement": "independent_check",
        "profile": "diagnostic_profile",
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("fetch_ctas", "Validation identity mismatch"),
        ("tensor", "Acceptance tensors mismatch"),
        ("receipt", "Receipt file hash mismatch"),
    ],
)
def test_report_rejects_reused_acceptance_for_changed_execution(receipt_runs, mutation, match):
    directory = receipt_runs[0]
    metadata = json.loads((directory / "metadata.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    if mutation == "fetch_ctas":
        metadata["args"]["fetch_ctas"] = 12
    elif mutation == "tensor":
        result["results"][0]["tensors"]["q"]["dtype"] = "torch.float16"
    else:
        (directory / "validation_receipt.json").write_text("{}")
    _write(directory / "metadata.json", metadata)
    _write(directory / "results.json", result)
    with pytest.raises(ValueError, match=match):
        build_report(*receipt_runs)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("profiled", "profiled=True"),
        ("source", "source_sha256 mismatch"),
        ("capture", "Input capture metadata_sha256 mismatch"),
        ("shape", "tensors mismatch"),
        ("exact", "Exact check failed"),
        ("traffic", "Missing traffic checks"),
        ("reference", "Reference rows missing"),
        ("timing", "Timing summary differs"),
        ("samples", "Incomplete timeline samples"),
        ("intersection", "Serialized control has GPU overlap"),
        ("sqlite", "SQLite hash mismatch"),
    ],
)
def test_report_rejects_incompatible_or_incomplete_runs(runs, mutation, match):
    _, profile = runs
    metadata = json.loads((profile / "metadata.json").read_text())
    results = json.loads((profile / "results.json").read_text())
    timeline = json.loads((profile / "overlap.json").read_text())
    case = results["results"][0]
    if mutation == "profiled":
        metadata["args"]["profiled"] = False
    elif mutation == "source":
        source = profile / "sources/operators/sm90/example.py"
        source.write_text("a different measured implementation\n")
        metadata["source_sha256"]["operators/sm90/example.py"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    elif mutation == "capture":
        metadata["input_capture"]["metadata_sha256"] = "different-capture"
    elif mutation == "shape":
        case["tensors"]["q"]["shape"] = [8, 16, 128]
    elif mutation == "exact":
        case["serialized_overlap_exact_equal"] = False
    elif mutation == "traffic":
        case["traffic_checks"] = "not checked"
    elif mutation == "reference":
        case["acceptance"]["rows"].pop()
    elif mutation == "timing":
        case["modes"]["overlap"]["cuda_ms"]["median"] = 0.001
    elif mutation == "samples":
        timeline["records"].pop()
    elif mutation == "intersection":
        row = next(row for row in timeline["records"] if row["mode"] == "serialized")
        row["overlap_us"] = 1
        row["fetch_without_attention_us"] -= 1
        row["fetch_hidden_fraction"] = 1 / row["fetch_union_us"]
    elif mutation == "sqlite":
        (profile / "timeline.sqlite").write_bytes(b"modified trace")
    _write(profile / "metadata.json", metadata)
    _write(profile / "results.json", results)
    _write(profile / "overlap.json", timeline)
    with pytest.raises(ValueError, match=match):
        build_report(*runs)


@pytest.fixture(
    params=((1, "per_page_copy_windows"), (2, "batch_copy_windows"), (2, "per_page_copy_windows"))
)
def fused_runs(runs, request):
    from experiments.nosa_offload_overlap.src.analyze import (
        work_interval_metrics,
        work_profile_metadata,
    )

    _, directory = runs
    schema, coverage = request.param
    for run in runs:
        metadata = json.loads((run / "metadata.json").read_text())
        metadata["args"]["fetch_ctas"] = 4
        _write(run / "metadata.json", metadata)
        results = json.loads((run / "results.json").read_text())
        results["results"][0]["fetch_ctas"] = 4
        _write(run / "results.json", results)
    timeline = json.loads((directory / "overlap.json").read_text())
    case = {
        "prefix": 65536,
        "queries": 8,
        "kv_heads": 2,
        "expected_prefix_bytes": 65536,
        "expected_fetch_rows": [{"row": 0, "bytes": 32768}, {"row": 1, "bytes": 32768}],
    }
    phases = {
        "schema_version": schema,
        "run_id": "profile",
        "clock": "device_globaltimer_ns",
        "math_coverage": "softmax_only",
        "cases": {"layer_00": case},
        "records": [],
    }
    if schema == 2:
        phases.update(fetch_coverage=coverage, byte_accounting="logical_unique_kv_payload")
        timeline.update(work_profile_metadata(phases))
    for row in timeline["records"]:
        mode = row["mode"]
        kernel = {
            key: row.pop(key)
            for key in (
                "fetch_union_us",
                "attention_union_us",
                "overlap_us",
                "fetch_without_attention_us",
                "fetch_hidden_fraction",
            )
        }
        row["fused_main_count"] = 1 if mode == "overlap" else 0
        row["fused_main_us"] = 10 if mode == "overlap" else 0
        row["kernel_intervals"] = None if mode == "overlap" else kernel
        row["profile_method"] = (
            "globaltimer_fetch_softmax" if mode == "overlap" else "nsys_separate_kernel_intervals"
        )
        row["work_metrics"] = None
        if mode != "resident":
            begin = 10000 + row["sample"] * 1000
            work = {
                "case": row["case"],
                "mode": mode,
                "sample": row["sample"],
                "recorded_transfer_bytes": 65536,
                "intervals": []
                if mode == "serialized"
                else [
                    {
                        "row": 0,
                        "start_ns": begin,
                        "end_ns": begin + 3000,
                        "bytes": 32768,
                        "kind": 1,
                    },
                    {
                        "row": 1,
                        "start_ns": begin if coverage == "batch_copy_windows" else begin + 1000,
                        "end_ns": begin + 4000,
                        "bytes": 32768,
                        "kind": 1,
                    },
                    {
                        "row": 2048,
                        "start_ns": begin + 2000,
                        "end_ns": begin + 5000,
                        "bytes": 0,
                        "kind": 2,
                    },
                ],
            }
            phases["records"].append(work)
            row["work_metrics"] = work_interval_metrics(case, work)
    _write(directory / "work_intervals.json", phases)
    timeline["schema_version"] = 2
    timeline["work_intervals_sha256"] = hashlib.sha256(
        (directory / "work_intervals.json").read_bytes()
    ).hexdigest()
    _write(directory / "overlap.json", timeline)
    return runs


def test_fused_report_keeps_softmax_lower_bound_separate_from_kernel_intervals(fused_runs):
    report, rows = build_report(*fused_runs)
    assert report["schema_version"] == 2
    profile = report["cases"][0]["modes"]["overlap"]["profile"]
    assert profile["kernel_intervals"] is None
    assert profile["intra_kernel_softmax"]["fetch_math_overlap_fraction"]["median"] == 0.5
    assert "fetch_hidden_fraction" not in profile
    assert rows[0]["serialized_kernel_overlap_us"] == 0
    assert rows[0]["fetch_ctas"] == 4
    assert rows[0]["overlap_fetch_math_overlap_fraction"] == 0.5
    assert rows[0]["latency_reduction_fraction"] == pytest.approx(1 / 3)
    assert "lower bound" in report["fetch_math_overlap_definition"]
    assert report["byte_accounting"] == "logical_unique_kv_payload"
    phases = json.loads((fused_runs[1] / "work_intervals.json").read_text())
    assert report["fetch_coverage"] == phases.get("fetch_coverage", "per_page_copy_windows")
    assert "work_intervals.json" in report["sources"]["profile"]["artifact_sha256"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("raw_hash", "Work interval hash mismatch"),
        ("metric", "metrics differ from raw events"),
        ("main", "one main kernel"),
    ],
)
def test_fused_report_rejects_unverifiable_role_evidence(fused_runs, mutation, match):
    _, directory = fused_runs
    if mutation == "raw_hash":
        (directory / "work_intervals.json").write_text("{}")
    else:
        timeline = json.loads((directory / "overlap.json").read_text())
        row = next(row for row in timeline["records"] if row["mode"] == "overlap")
        if mutation == "metric":
            row["work_metrics"]["fetch_math_overlap_fraction"] = 1.0
        else:
            row["fused_main_count"] = 2
        _write(directory / "overlap.json", timeline)
    with pytest.raises(ValueError, match=match):
        build_report(*fused_runs)


def test_fused_report_requires_matching_fetch_cta_configuration(fused_runs):
    _, directory = fused_runs
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["args"]["fetch_ctas"] = 8
    _write(directory / "metadata.json", metadata)
    results = json.loads((directory / "results.json").read_text())
    results["results"][0]["fetch_ctas"] = 8
    _write(directory / "results.json", results)
    with pytest.raises(ValueError, match="fetch CTA configuration mismatch"):
        build_report(*fused_runs)


def test_fused_report_rejects_relabeling_batch_windows_as_page_latencies(fused_runs):
    from experiments.nosa_offload_overlap.src.analyze import work_profile_metadata

    _, directory = fused_runs
    phases = json.loads((directory / "work_intervals.json").read_text())
    phases.update(
        schema_version=2,
        fetch_coverage="batch_copy_windows",
        byte_accounting="logical_unique_kv_payload",
    )
    _write(directory / "work_intervals.json", phases)
    timeline = json.loads((directory / "overlap.json").read_text())
    timeline.update(work_profile_metadata(phases))
    timeline["work_intervals_sha256"] = hashlib.sha256(
        (directory / "work_intervals.json").read_bytes()
    ).hexdigest()
    timeline["fetch_coverage"] = "per_page_copy_windows"
    _write(directory / "overlap.json", timeline)
    with pytest.raises(ValueError, match="semantics differ from raw profile"):
        build_report(*fused_runs)


def _write_stripe_profile(directory, phases, timeline, *, derive=True):
    from experiments.nosa_offload_overlap.src.analyze import (
        stripe_interval_metrics,
        work_interval_metrics,
        work_profile_metadata,
    )

    if derive:
        raw = {(r["case"], r["mode"], r["sample"]): r for r in phases["records"]}
        for row in timeline["records"]:
            row["stripe_metrics"] = None
            if row["mode"] != "resident":
                record = raw[row["case"], row["mode"], row["sample"]]
                case = phases["cases"][row["case"]]
                row["work_metrics"] = work_interval_metrics(case, record)
                row["stripe_metrics"] = stripe_interval_metrics(
                    case, record, phases["fetch_stripes"]
                )
    _write(directory / "work_intervals.json", phases)
    timeline.update(
        schema_version=3,
        work_intervals_sha256=hashlib.sha256(
            (directory / "work_intervals.json").read_bytes()
        ).hexdigest(),
        **work_profile_metadata(phases),
    )
    _write(directory / "overlap.json", timeline)


@pytest.fixture
def stripe_runs(fused_runs):
    for run in fused_runs:
        metadata = json.loads((run / "metadata.json").read_text())
        metadata["native_build"]["offload_fused"] = {"fetch_stripes": 2}
        _write(run / "metadata.json", metadata)
    directory = fused_runs[1]
    phases = json.loads((directory / "work_intervals.json").read_text())
    phases.update(
        schema_version=3,
        fetch_coverage="per_page_copy_envelopes",
        stripe_coverage="nonempty_stripe_copy_windows",
        fetch_stripes=2,
        byte_accounting="logical_unique_kv_payload",
    )
    for record in phases["records"]:
        record["stripe_intervals"] = []
        for page in record["intervals"]:
            if page["kind"] != 1:
                continue
            midpoint = (page["start_ns"] + page["end_ns"]) // 2
            for stripe in range(2):
                record["stripe_intervals"].append(
                    {
                        "row": 2176 + page["row"] * 2 + stripe,
                        "start_ns": midpoint if stripe else page["start_ns"],
                        "end_ns": page["end_ns"] if stripe else midpoint,
                        "kind": 3,
                        "bytes": 16384,
                    }
                )
    timeline = json.loads((directory / "overlap.json").read_text())
    _write_stripe_profile(directory, phases, timeline)
    return fused_runs


def test_stripe_report_preserves_page_metrics_and_independent_stripe_evidence(stripe_runs):
    report, rows = build_report(*stripe_runs)
    case = report["cases"][0]
    profile = case["modes"]["overlap"]["profile"]
    assert report["schema_version"] == report["work_interval_schema_version"] == 3
    assert report["fetch_stripes"] == 2
    assert report["fetch_coverage"] == "per_page_copy_envelopes"
    assert profile["intra_kernel_softmax"]["fetch_math_overlap_fraction"]["median"] == 0.5
    assert profile["intra_kernel_stripes"]["fetch_stripe_math_overlap_fraction"]["median"] == 0.5
    assert profile["intra_kernel_stripes"]["stripe_interval_count"]["median"] == 4
    assert rows[0]["overlap_page_envelope_only_union_us"] == 0
    assert rows[0]["overlap_page_minus_stripe_math_overlap_fraction"] == 0
    assert rows[0]["overlap_fetch_stripe_math_overlap_fraction"] == 0.5
    assert not case["overlap_threshold_evidence"]["page_and_stripe_medians_meet_threshold"]
    assert case["modes"]["resident"]["profile"]["intra_kernel_stripes"] is None
    assert case["modes"]["serialized"]["profile"]["intra_kernel_stripes"] is None
    assert "envelope evidence alone cannot pass" in report["overlap_threshold_definition"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("metric", "Stripe interval metrics differ"),
        ("missing", "Missing selected nonempty stripe"),
        ("trace_count", "stripe count differs from compiled"),
        ("build_count", "stripe count differs from compiled"),
        ("semantics", "semantics differ from raw"),
        ("downgrade", "Timeline schema differs"),
        ("resident", "Resident must not claim stripe"),
        ("serialized", "Only fused overlap"),
    ],
)
def test_stripe_report_rejects_tampered_evidence(stripe_runs, mutation, match):
    directory = stripe_runs[1]
    phases = json.loads((directory / "work_intervals.json").read_text())
    timeline = json.loads((directory / "overlap.json").read_text())
    if mutation in ("metric", "resident"):
        mode = "overlap" if mutation == "metric" else "resident"
        row = next(row for row in timeline["records"] if row["mode"] == mode)
        row["stripe_metrics"] = {"fetch_stripe_math_overlap_fraction": 1}
    elif mutation == "missing":
        next(r for r in phases["records"] if r["mode"] == "overlap")["stripe_intervals"].pop()
    elif mutation == "trace_count":
        phases["fetch_stripes"] = 4
    elif mutation == "build_count":
        for run in stripe_runs:
            metadata = json.loads((run / "metadata.json").read_text())
            metadata["native_build"]["offload_fused"]["fetch_stripes"] = 4
            _write(run / "metadata.json", metadata)
    elif mutation == "serialized":
        phases["records"][0]["stripe_intervals"] = next(
            r["stripe_intervals"] for r in phases["records"] if r["mode"] == "overlap"
        )
    _write_stripe_profile(directory, phases, timeline, derive=False)
    if mutation in ("semantics", "downgrade"):
        timeline = json.loads((directory / "overlap.json").read_text())
        if mutation == "semantics":
            timeline["stripe_coverage"] = "per_page_copy_envelopes"
        else:
            timeline["schema_version"] = 2
        _write(directory / "overlap.json", timeline)
    with pytest.raises(ValueError, match=match):
        build_report(*stripe_runs)


@pytest.mark.parametrize(
    "scenario",
    [
        "envelope_false_positive",
        "negative_difference",
        "median_only",
        "all_pass",
        "exact_threshold",
    ],
)
def test_stripe_threshold_requires_both_metrics_and_preserves_signed_difference(
    stripe_runs, scenario
):
    directory = stripe_runs[1]
    phases = json.loads((directory / "work_intervals.json").read_text())
    timeline = json.loads((directory / "overlap.json").read_text())
    for record in phases["records"]:
        if record["mode"] != "overlap":
            continue
        for page in record["intervals"][:2]:
            page.update(start_ns=100, end_ns=1100)
        has_gap = scenario in ("envelope_false_positive", "negative_difference")
        for index, stripe in enumerate(record["stripe_intervals"]):
            if has_gap:
                stripe.update(
                    start_ns=100 if index % 2 == 0 else 1060, end_ns=140 if index % 2 == 0 else 1100
                )
            else:
                stripe.update(
                    start_ns=100 if index % 2 == 0 else 600, end_ns=600 if index % 2 == 0 else 1100
                )
        start, end = 140, 1100
        if scenario == "envelope_false_positive":
            end = 1060
        elif scenario == "negative_difference":
            start, end = 100, 140
        elif scenario == "median_only" and record["sample"] == 0:
            start = 300
        elif scenario == "exact_threshold":
            start = 200
        record["intervals"][2].update(start_ns=start, end_ns=end)
    _write_stripe_profile(directory, phases, timeline)
    report, rows = build_report(*stripe_runs)
    evidence = report["cases"][0]["overlap_threshold_evidence"]
    assert evidence["page_and_stripe_medians_meet_threshold"] == (
        scenario in ("median_only", "all_pass", "exact_threshold")
    )
    assert evidence["page_and_stripe_all_samples_meet_threshold"] == (
        scenario in ("all_pass", "exact_threshold")
    )
    if scenario == "envelope_false_positive":
        assert rows[0]["overlap_fetch_math_overlap_fraction"] == 0.92
        assert rows[0]["overlap_fetch_stripe_math_overlap_fraction"] == 0
    elif scenario == "negative_difference":
        assert rows[0]["overlap_page_minus_stripe_math_overlap_fraction"] < 0
