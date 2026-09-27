"""Timing provenance, complete runs, nested spans, and statistical aggregation."""

import csv
import json

import pytest

from experiments.indexer_block_sparse_profile.src.analyze import (
    NESTED_STAGES,
    PHASES,
    PRIMARY_STAGES,
    WORKLOAD,
    analyze,
    main,
    markdown_report,
    summarize,
)


def _profile(phase, cis_per_layer, *, nested=True):
    rows = []
    starts = range(0, 66560, 1024) if phase == "full_prefill" else (65536,)
    call_id = 0
    for start in starts:
        for layer, cis_duration in enumerate(cis_per_layer):

            def append(
                stage,
                duration,
                parent=None,
                *,
                current_call=call_id,
                layer_idx=layer,
                query_start=start,
            ):
                scope = len(rows)
                rows.append(
                    {
                        "phase": phase,
                        "stage": stage,
                        "scope_id": scope,
                        "parent_scope_id": parent,
                        "call_id": current_call,
                        "layer_idx": layer_idx,
                        "query_start": query_start,
                        "query_length": 1024,
                        "cuda_elapsed_ms": duration,
                        "inclusive": True,
                        "status": "ok",
                    }
                )
                return scope

            append("cis_projection", cis_duration)
            indexer = append("indexer_total", 10.0)
            if nested:
                append("compression_k", 1.0, indexer)
                append("compression_cis", 0.5, indexer)
                if start >= 4096:
                    # Internal query chunks are calls within a repetition,
                    # never independent statistical samples.
                    for _ in range(2):
                        append("compressed_scores", 0.25, indexer)
                        append("select_from_scores", 0.125, indexer)
            append("block_sparse_attention", 2.0)
            call_id += 1
    return rows


def measurements(*, nested=True):
    return {
        "schema_version": 1,
        "run_id": "fixture-001",
        "workload": {
            **WORKLOAD,
            "warmup": 2,
            "repeats": 3,
            "profile_repeats": 3,
            "block_size": 64,
        },
        "timings": {
            "full_prefill": [
                {"wall_ms": 100.0, "host_submit_ms": 80.0, "cuda_span_ms": 95.0},
                {"wall_ms": 200.0, "host_submit_ms": 140.0, "cuda_span_ms": 195.0},
                {"wall_ms": 300.0, "host_submit_ms": 210.0, "cuda_span_ms": 295.0},
            ],
            "extend": [
                {"wall_ms": 10.0, "host_submit_ms": 8.0, "cuda_span_ms": 9.5},
                {"wall_ms": 20.0, "host_submit_ms": 14.0, "cuda_span_ms": 19.5},
                {"wall_ms": 30.0, "host_submit_ms": 21.0, "cuda_span_ms": 29.5},
            ],
        },
        "profiles": {
            phase: [
                _profile(phase, per_layer, nested=nested)
                for per_layer in ((1.0, 100.0), (50.0, 50.0), (100.0, 1.0))
            ]
            for phase in PHASES
        },
    }


def test_metrics_remain_separate_and_throughput_uses_each_phase_token_count():
    report = summarize(measurements(), num_layers=2)
    full, extend = (report["timings"][phase] for phase in PHASES)
    assert full["wall_ms"] == {"count": 3, "median": 200.0, "min": 100.0, "max": 300.0}
    assert full["host_submit_ms"]["median"] == 140.0
    assert full["cuda_span_ms"]["median"] == 195.0
    assert full["tokens"] == 66560 and extend["tokens"] == 1024
    assert full["tokens_per_second"]["median"] == 332800
    assert full["tokens_per_second"]["min"] == pytest.approx(66560 * 1000 / 300)
    assert extend["tokens_per_second"]["median"] == 51200
    assert report["workload"]["block_size"] == 64
    assert report["num_layers_source"] == "provided"


def test_stage_totals_sum_within_run_before_taking_medians_and_keep_nesting():
    report = summarize(measurements(), num_layers=2)
    for phase, call_count in (("full_prefill", 65), ("extend", 1)):
        profile = report["profiles"][phase]
        cis = profile["stage_totals"]["cis_projection"]
        assert cis["cuda_elapsed_ms"] == {
            "count": 3,
            "median": 101 * call_count,
            "min": 100 * call_count,
            "max": 101 * call_count,
        }
        layer_medians = [
            layer["stages"]["cis_projection"]["cuda_elapsed_ms"]["median"]
            for layer in profile["per_layer"]
        ]
        assert layer_medians == [50 * call_count, 50 * call_count]
        assert sum(layer_medians) != cis["cuda_elapsed_ms"]["median"]
        indexer = profile["stage_totals"]["indexer_total"]
        assert indexer["cuda_elapsed_ms"]["median"] == 20 * call_count
        assert indexer["calls_per_run"]["median"] == 2 * call_count
        assert indexer["parent_stage"] is None
        for stage in NESTED_STAGES:
            if stage in profile["stage_totals"]:
                assert profile["stage_totals"][stage]["parent_stage"] == "indexer_total"
            else:
                assert stage in profile["unrecorded_stages"]
        score = profile["stage_totals"]["compressed_scores"]
        score_calls = 244 if phase == "full_prefill" else 4
        assert score["calls_per_run"]["median"] == score_calls
        assert score["cuda_elapsed_ms"]["count"] == 3
        assert score["cuda_elapsed_ms"]["median"] == score_calls * 0.25


def test_unrecorded_optional_stages_are_explicit_and_zero_nested_intervals_are_valid():
    raw = measurements(nested=False)
    report = summarize(raw)
    assert report["num_layers"] == 2
    assert report["num_layers_source"] == "inferred_from_records"
    for profile in report["profiles"].values():
        assert set(profile["stage_totals"]) == set(PRIMARY_STAGES)
        assert profile["unrecorded_stages"] == list(NESTED_STAGES)
    raw = measurements()
    for runs in raw["profiles"].values():
        for records in runs:
            for record in records:
                if record["stage"] == "compression_cis":
                    record["cuda_elapsed_ms"] = 0.0
    report = summarize(raw, num_layers=2)
    assert report["profiles"]["extend"]["stage_totals"]["compression_cis"]["cuda_elapsed_ms"] == {
        "count": 3,
        "median": 0,
        "min": 0,
        "max": 0,
    }


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1.0", None])
@pytest.mark.parametrize("metric", ["wall_ms", "host_submit_ms", "cuda_span_ms"])
def test_rejects_invalid_end_to_end_samples(value, metric):
    raw = measurements()
    raw["timings"]["extend"][1][metric] = value
    with pytest.raises(ValueError, match="finite and positive"):
        summarize(raw, num_layers=2)


@pytest.mark.parametrize("problem", ["zero", "nan", "error", "audit", "unknown_stage", "phase"])
def test_rejects_invalid_profile_samples(problem):
    raw = measurements()
    row = raw["profiles"]["extend"][0][0]
    if problem in ("zero", "nan", "audit"):
        row["cuda_elapsed_ms"] = {"zero": 0, "nan": float("nan"), "audit": None}[problem]
    elif problem == "error":
        row["status"] = "error"
    elif problem == "unknown_stage":
        row["stage"] = "arbitrary_kernel"
    else:
        row["phase"] = "full_prefill"
    with pytest.raises(ValueError):
        summarize(raw, num_layers=2)


@pytest.mark.parametrize(
    "problem",
    [
        "pending_profiles",
        "empty_run",
        "missing_phase",
        "missing_repeat",
        "mixed_runs",
        "missing_stage",
    ],
)
def test_rejects_incomplete_or_merged_profile_repetitions(problem):
    raw = measurements()
    if problem == "pending_profiles":
        raw["profiles"] = {}
    elif problem == "empty_run":
        raw["profiles"]["extend"][0] = []
    elif problem == "missing_phase":
        del raw["profiles"]["extend"]
    elif problem == "missing_repeat":
        for phase in PHASES:
            raw["profiles"][phase].pop()
    elif problem == "mixed_runs":
        raw["profiles"]["extend"][0] += raw["profiles"]["extend"][1]
    else:
        raw["profiles"]["extend"][0].pop(0)
    with pytest.raises(ValueError):
        summarize(raw, num_layers=2)


def test_rejects_missing_layer_and_duplicate_query_range_with_preserved_call_count():
    raw = measurements()
    with pytest.raises(ValueError, match="layer 2"):
        summarize(raw, num_layers=3)
    rows = raw["profiles"]["full_prefill"][0]
    cis = [row for row in rows if row["stage"] == "cis_projection" and row["layer_idx"] == 0]
    cis[-1]["query_start"] = cis[0]["query_start"]
    with pytest.raises(ValueError, match="Duplicate or missing query range"):
        summarize(raw, num_layers=2)


@pytest.mark.parametrize("problem", ["duplicate_scope", "wrong_parent", "wrong_layer", "exclusive"])
def test_rejects_invalid_scope_relationships(problem):
    raw = measurements()
    rows = raw["profiles"]["extend"][0]
    child = next(row for row in rows if row["stage"] == "compression_k")
    if problem == "duplicate_scope":
        child["scope_id"] = rows[0]["scope_id"]
    elif problem == "wrong_parent":
        child["parent_scope_id"] = rows[0]["scope_id"]
    elif problem == "wrong_layer":
        child["layer_idx"] = 1
    else:
        child["inclusive"] = False
    with pytest.raises(ValueError):
        summarize(raw, num_layers=2)


@pytest.mark.parametrize("problem", ["schema", "workload", "timing_repeats", "profile_repeats"])
def test_rejects_wrong_experiment_or_declared_repeat_counts(problem):
    raw = measurements()
    if problem == "schema":
        raw["schema_version"] = True
    elif problem == "workload":
        raw["workload"]["prefix_tokens"] = 32768
    elif problem == "timing_repeats":
        raw["workload"]["repeats"] = 5
    else:
        raw["workload"]["profile_repeats"] = 5
    with pytest.raises(ValueError):
        summarize(raw, num_layers=2)


def test_profile_repeat_count_can_differ_from_benchmark_repeat_count():
    raw = measurements()
    raw["workload"]["profile_repeats"] = 1
    raw["profiles"] = {phase: runs[:1] for phase, runs in raw["profiles"].items()}
    result = summarize(raw, num_layers=2)
    assert result["profiles"]["extend"]["sample_count"] == 1
    assert result["timings"]["extend"]["sample_count"] == 3


def test_file_analysis_uses_metadata_and_exports_traceable_tables(tmp_path, capsys):
    raw = measurements()
    (tmp_path / "measurements.json").write_text(json.dumps(raw))
    (tmp_path / "metadata.json").write_text(
        json.dumps({"run_id": raw["run_id"], "model_config": {"num_hidden_layers": 2}})
    )
    main([str(tmp_path)])
    console = json.loads(capsys.readouterr().out)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert console == report["timings"]
    assert report["num_layers_source"] == "provided"
    assert (tmp_path / "report.md").read_text() == markdown_report(report)
    with (tmp_path / "timings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    wall = next(
        row for row in rows if row["phase"] == "full_prefill" and row["metric"] == "wall_ms"
    )
    assert wall["tokens"] == "66560" and wall["median"] == "200.0"
    with (tmp_path / "stages.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 14
    assert all(
        row["parent_stage"] == "indexer_total" for row in rows if row["stage"] in NESTED_STAGES
    )
    with (tmp_path / "layers.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 28
    text = (tmp_path / "report.md").read_text()
    assert "Host and CUDA intervals overlap" in text
    assert "not GPU active time or pure kernel duration" in text
    assert "Nested stages are already included in indexer_total" in text


def test_analysis_does_not_publish_files_for_partial_capture_or_conflicting_metadata(tmp_path):
    raw = measurements()
    (tmp_path / "measurements.json").write_text(json.dumps(raw))
    (tmp_path / "metadata.json").write_text(json.dumps({"model_config": {"num_hidden_layers": 3}}))
    with pytest.raises(ValueError, match="layer 2"):
        analyze(tmp_path)
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "report.md").exists()
    with pytest.raises(ValueError, match="num_layers must match"):
        analyze(tmp_path, num_layers=2)
