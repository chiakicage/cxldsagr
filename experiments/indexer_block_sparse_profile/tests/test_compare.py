"""Native/Triton report ratios require matching capture and analysis provenance."""

import csv
import hashlib
import json

import pytest

from experiments.indexer_block_sparse_profile.src.capture import operator_workload
from experiments.indexer_block_sparse_profile.src.compare import MODULES, compare, write_report
from experiments.indexer_block_sparse_profile.src.mfu import (
    FUSED_NATIVE_SOURCES,
    PHASES,
    PRUNED_NATIVE_SOURCES,
    build_report,
)
from experiments.indexer_block_sparse_profile.tests.test_sparse_profile_mfu import native_inputs


def write_run(path, backend, *, scale=1.0, change=None, fused_revision=False, fa3=False):
    metadata, summary = native_inputs(backend=backend)
    run_id = f"fixture_{backend}"
    metadata["run_id"] = summary["run_id"] = run_id
    metadata["args"].update(run_id=run_id, warmup=2, profile_repeats=2, device="cuda:0")
    summary["workload"].update(warmup=2, profile_repeats=2)
    if fused_revision:
        summary["workload"].update(operator_workload(backend))
        summary["workload"].pop("attention_execution")
        if type(fused_revision) is int:
            summary["workload"]["native_kernel_revision"] = fused_revision
            if backend == "native":
                summary["workload"]["indexer_execution"] = f"cached_native_v{fused_revision}"
        if fused_revision == 3:
            summary["workload"]["native_kernel_revision"] = 3
            if backend == "native":
                summary["workload"].update(
                    indexer_execution="cached_native_v3",
                    indexer_preparation="native_guarded_ranked_v1",
                )
        for name in (
            FUSED_NATIVE_SOURCES
            | PRUNED_NATIVE_SOURCES
            | {"operators/sm90/csrc/nosa_indexer_checked.cu"}
        ):
            metadata["source_sha256"][name] = "3" * 64
            metadata["native_build"]["source_sha256"][name] = "3" * 64
    request = json.dumps({"input_ids": [1, 2, 3]}).encode()
    metadata.update(
        request_sha256=hashlib.sha256(request).hexdigest(),
        checkpoint_path="/weights/nosa",
        checkpoint_files={"model.safetensors": {"size": 1, "mtime_ns": 1}},
        checkpoint_config_sha256="c" * 64,
        torch="torch-version",
        cuda="cuda-version",
        triton="triton-version",
        flashinfer="flashinfer-version",
    )
    if fa3:
        from experiments.indexer_block_sparse_profile.tests.test_fa3_module_mfu import add_fa3_build

        version = fa3 if type(fa3) is int else 2
        add_fa3_build(metadata, version=version)
        summary["workload"]["attention_execution"] = (
            f"native_fa3_v{version}" if backend == "native" else "triton_v1"
        )
    metadata["gpu"]["uuid"] = "GPU-fixture"
    for phase in PHASES:
        summary["timings"][phase]["wall_ms"]["median"] *= scale
    if change == "request":
        request = json.dumps({"input_ids": [1, 2, 4]}).encode()
        metadata["request_sha256"] = hashlib.sha256(request).hexdigest()
    elif change == "source":
        metadata["source_sha256"]["models/nosa/model.py"] = "f" * 64
    elif change == "checkpoint":
        metadata["checkpoint_files"]["model.safetensors"]["size"] = 2
    elif change == "gpu":
        metadata["gpu"]["uuid"] = "GPU-other"
    elif change == "device":
        metadata["args"]["device"] = "cuda:1"
    elif change == "warmup":
        metadata["args"]["warmup"] = summary["workload"]["warmup"] = 3
    elif change == "build":
        metadata["native_build"]["compiler"]["version"] = "different compiler"
    elif change == "workload":
        summary["workload"]["indexer_execution"] = "different execution"
    path.mkdir()
    (path / "request.json").write_bytes(request)
    inputs = {"metadata.json": metadata, "summary.json": summary}
    for name, value in inputs.items():
        (path / name).write_text(json.dumps(value))
    for name in ("profile_metadata.json", "attention_audit.json"):
        (path / name).write_text("{}")
    digests = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in (*inputs, "profile_metadata.json", "attention_audit.json")
    }
    mfu = build_report(metadata, summary)
    mfu["input_sha256"] = {name: digests[name] for name in inputs}
    mfu["analysis_source_sha256"] = {"mfu.py": "a" * 64}
    if change == "analysis":
        mfu["analysis_source_sha256"]["mfu.py"] = "b" * 64
    modules = {
        key: mfu[key]
        for key in (
            "schema_version",
            "run_id",
            "gpu",
            "workload",
            "dimensions",
            "implementation",
            "peak_tflops",
        )
    }
    modules["phases"] = {
        phase: {
            "end_to_end": mfu["phases"][phase],
            "modules": {
                name: {
                    "kernel_ms_median": (index + 1) * scale,
                    "sample_count": 2,
                    "kernel_count_per_run": [32, 32],
                }
                for index, name in enumerate(MODULES)
            },
        }
        for phase in PHASES
    }
    modules["provenance"] = {
        "input_sha256": digests,
        "captured_source_sha256": metadata["source_sha256"],
        "analysis_source_sha256": {"module_mfu.py": "a" * 64},
    }
    if fused_revision:
        for phase in PHASES:
            values = modules["phases"][phase]["modules"]
            values["score_selection"] = {
                "kernel_ms_median": 0.5 if backend == "native" else 0,
                "sample_count": 2,
                "kernel_count_per_run": [32, 32] if backend == "native" else [0, 0],
            }
            if backend == "native" and phase == "extend":
                values["pooled_scores"].update(kernel_ms_median=0, kernel_count_per_run=[0, 0])
    (path / "mfu.json").write_text(json.dumps(mfu))
    (path / "module_mfu.json").write_text(json.dumps(modules))


def test_compare_preserves_run_ids_and_separates_timing_boundaries(tmp_path):
    native, triton = tmp_path / "native", tmp_path / "triton"
    write_run(native, "native")
    write_run(triton, "triton", scale=2)
    output = tmp_path / "comparison"
    report = write_report(native, triton, output)
    assert report["native_run_id"] == "fixture_native"
    assert report["triton_run_id"] == "fixture_triton"
    assert len(report["rows"]) == 8
    for row in report["rows"]:
        assert row["speedup_triton_over_native"] == 2
        assert (row["timing_boundary"] == "unprofiled wall time") == (row["metric"] == "end_to_end")
    assert json.loads((output / "comparison.json").read_text()) == report
    with (output / "comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {row["metric"] for row in rows} == {"end_to_end", *MODULES}
    with pytest.raises(FileExistsError):
        write_report(native, triton, output)


@pytest.mark.parametrize(
    "change",
    ["request", "source", "checkpoint", "gpu", "device", "warmup", "build", "workload", "analysis"],
)
def test_comparison_rejects_different_measurement_inputs(tmp_path, change):
    native, triton = tmp_path / "native", tmp_path / "triton"
    write_run(native, "native")
    write_run(triton, "triton", change=change)
    with pytest.raises(ValueError, match="differ"):
        compare(native, triton)


@pytest.mark.parametrize(
    "change", ["stale_input", "wrong_mfu", "incomplete", "nonfinite", "request"]
)
def test_comparison_rejects_stale_or_incomplete_reports(tmp_path, change):
    native, triton = tmp_path / "native", tmp_path / "triton"
    write_run(native, "native")
    write_run(triton, "triton")
    if change == "stale_input":
        path = native / "metadata.json"
        value = json.loads(path.read_text())
        value["changed_after_analysis"] = True
    elif change == "wrong_mfu":
        path = native / "mfu.json"
        value = json.loads(path.read_text())
        value["phases"]["extend"]["wall_ms"] *= 2
    elif change == "request":
        (native / "request.json").write_text("changed request")
        with pytest.raises(ValueError, match="Request content"):
            compare(native, triton)
        return
    else:
        path = native / "module_mfu.json"
        value = json.loads(path.read_text())
        row = value["phases"]["extend"]["modules"]["pooled_scores"]
        if change == "incomplete":
            row["kernel_count_per_run"].pop()
        else:
            row["kernel_ms_median"] = float("nan")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        compare(native, triton)


def test_comparison_rejects_backend_swap(tmp_path):
    native, triton = tmp_path / "native", tmp_path / "triton"
    write_run(native, "native")
    write_run(triton, "triton")
    with pytest.raises(ValueError, match="Expected kernel_backend"):
        compare(triton, native)


@pytest.mark.parametrize("revision", [3, 4, 5])
def test_fused_native_comparison_keeps_complete_scopes_without_partial_score_ratios(
    tmp_path, revision
):
    native, triton = tmp_path / "native", tmp_path / "triton"
    write_run(native, "native", fused_revision=revision)
    write_run(triton, "triton", scale=2, fused_revision=revision)
    report = compare(native, triton)
    assert len(report["rows"]) == 6
    assert {row["metric"] for row in report["rows"]} == {
        "end_to_end",
        "block_sparse_attention",
        "indexer_total",
    }
    assert all(row["speedup_triton_over_native"] == 2 for row in report["rows"])
    assert {(row["phase"], row["metric"]) for row in report["omitted_comparisons"]} == {
        (phase, "pooled_scores") for phase in PHASES
    }


@pytest.mark.parametrize("change", ["missing", "zero_time", "inconsistent_presence", "empty_qk"])
def test_fused_comparison_rejects_missing_or_inconsistent_kernel_work(tmp_path, change):
    native, triton = tmp_path / "native", tmp_path / "triton"
    write_run(native, "native", fused_revision=True)
    write_run(triton, "triton", fused_revision=True)
    path = native / "module_mfu.json"
    value = json.loads(path.read_text())
    rows = value["phases"]["extend"]["modules"]
    if change == "missing":
        rows.pop("score_selection")
    elif change == "zero_time":
        rows["score_selection"]["kernel_ms_median"] = 0
    elif change == "inconsistent_presence":
        rows["score_selection"]["kernel_count_per_run"] = [0, 32]
    else:
        rows["score_selection"].update(kernel_ms_median=0, kernel_count_per_run=[0, 0])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        compare(native, triton)


@pytest.mark.parametrize("version", [2, 3])
def test_fa3_native_and_triton_comparison_retains_complete_attention_denominator(tmp_path, version):
    native, triton = tmp_path / "native", tmp_path / "triton"
    write_run(native, "native", fused_revision=5, fa3=version)
    write_run(triton, "triton", fused_revision=5, fa3=version, scale=2.0)
    report = compare(native, triton)
    assert report["native_run_id"] == "fixture_native"
    assert any(row["metric"] == "block_sparse_attention" for row in report["rows"])
