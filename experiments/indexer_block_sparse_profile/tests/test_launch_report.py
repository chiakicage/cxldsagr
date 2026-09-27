"""Root-only analysis uses same-trace gaps and rejects mismatched measurement runs."""

import csv
import json
import sqlite3
from copy import deepcopy

import pytest

from experiments.indexer_block_sparse_profile.src.launch_analysis import analyze
from experiments.indexer_block_sparse_profile.src.launch_report import (
    PHASES,
    summarize,
    validate_inputs,
    write_report,
)
from experiments.indexer_block_sparse_profile.tests.test_sparse_profile_mfu import native_inputs


def inputs():
    native_metadata, native_summary = native_inputs()
    workload = {
        "prefix_tokens": 65536,
        "new_tokens": 1024,
        "total_tokens": 66560,
        "chunk_size": 1024,
        "warmup": 2,
        "repeats": 2,
        "profile_repeats": 2,
        "kernel_backend": native_summary["workload"]["kernel_backend"],
        "selection_backend": "flashinfer",
    }
    gpu = {"uuid": "GPU-fixture", "sm_count": 132, "capability": [9, 0]}
    metadata = {
        "run_id": "fixture",
        "args": {"run_id": "fixture", "mode": "benchmark", **workload, "kernel_backend": "native"},
        "model_config": {"num_hidden_layers": 32},
        "checkpoint_path": "/weights/nosa",
        "checkpoint_files": {"model.safetensors": {"size": 1, "mtime_ns": 1}},
        "checkpoint_config_sha256": "config",
        "request_sha256": "request",
        "source_sha256": {
            **native_metadata["source_sha256"],
            "models/nosa/model.py": "source",
        },
        "torch": "torch-version",
        "cuda": "cuda-version",
        "triton": "triton-version",
        "flashinfer": "flashinfer-version",
        "tvm_ffi": native_metadata["tvm_ffi"],
        "native_build": native_metadata["native_build"],
        "gpu": gpu,
    }

    def timings(values):
        return {
            phase: [
                {"wall_ms": value, "host_submit_ms": value - 0.2, "cuda_span_ms": value - 0.1}
                for value in values
            ]
            for phase in PHASES
        }

    profile = deepcopy(metadata)
    profile["args"]["mode"] = "timeline"
    profile["module_scopes_enabled"] = False
    profile["instrumented_timings"] = timings([4, 8])
    measurements = {
        "schema_version": 1,
        "run_id": "fixture",
        "workload": workload,
        "timings": timings([1, 3]),
        "profiles": {},
    }
    return measurements, metadata, profile


def create_trace(path):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1,'tiny_kernel'),(2,'cudaLaunchKernel'),
                (3,'cudaDeviceSynchronize');
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,
                globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,
                deviceId INTEGER,streamId INTEGER,contextId INTEGER,globalPid INTEGER,
                correlationId INTEGER,demangledName INTEGER);
            CREATE TABLE TARGET_INFO_GPU(id INTEGER,uuid TEXT,name TEXT,smCount INTEGER,
                computeMajor INTEGER,computeMinor INTEGER);
            INSERT INTO TARGET_INFO_GPU VALUES (0,'fixture','H100 fixture',132,9,0);
            """
        )
        for phase_index, phase in enumerate(PHASES):
            for iteration in range(2):
                number = phase_index * 2 + iteration
                offset = number * 2_000_000
                correlation = number * 10
                connection.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)",
                    (offset, offset + 1_000_000, f"NOSA/profile/{phase}/{iteration}", None, 1),
                )
                # The final kernel belongs to the output check outside the root.
                for operation, (launch, start, end) in enumerate(
                    (
                        (50_000, 100_000, 200_000),
                        (400_000, 500_000, 700_000),
                        (1_090_000, 1_100_000, 1_200_000),
                    ),
                    start=1,
                ):
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                        (offset + launch, offset + launch + 1000, 1, correlation + operation, 2),
                    )
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?)",
                        (offset + start, offset + end, 0, 7, 1, 1, correlation + operation, 1),
                    )
                # Setup sync, model-internal sync, completion sync; the latter
                # extends past the GPU hull and must be clipped for hull totals.
                for start, end, thread in (
                    (10_000, 20_000, 1),
                    (250_000, 350_000, 1),
                    (600_000, 800_000, 1),
                    (300_000, 900_000, 2),
                ):
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                        (offset + start, offset + end, thread, None, 3),
                    )


@pytest.fixture
def data_dir(tmp_path):
    for name, value in zip(("measurements", "metadata", "profile_metadata"), inputs(), strict=True):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    create_trace(tmp_path / "nsys.sqlite")
    return tmp_path


def test_end_to_end_root_only_report_preserves_process_boundaries(data_dir):
    report = write_report(data_dir)
    _, metadata, _ = inputs()
    assert report["provenance"]["native_build"] == metadata["native_build"]
    assert report["provenance"]["tvm_ffi"] == metadata["tvm_ffi"]
    for phase in PHASES:
        assert report["benchmark"][phase]["wall_ms"] == {
            "count": 2,
            "median": 2,
            "min": 1,
            "max": 3,
        }
        assert report["timeline_timings"][phase]["wall_ms"]["median"] == 6
        assert report["trace_overhead"][phase]["wall_ms"]["median_ratio"] == 3
        timeline = report["timeline"][phase]
        assert timeline["hull_ms"]["median"] == 0.6
        assert timeline["gpu_active_union_ms"]["median"] == 0.3
        assert timeline["idle_ms"]["median"] == 0.3
        assert timeline["idle_pct"]["median"] == 50
        assert timeline["before_next_submission_ms"]["median"] == 0.2
        assert timeline["kernel_count"]["median"] == 2
        assert timeline["root_sync_api_count"]["median"] == 3
        assert timeline["root_sync_api_duration_sum_ms"]["median"] == pytest.approx(0.31)
        assert timeline["gpu_hull_sync_api_count"]["median"] == 2
        assert timeline["gpu_hull_sync_api_duration_sum_ms"]["median"] == pytest.approx(0.2)
    saved = json.loads((data_dir / "launch_summary.json").read_text())
    assert saved == report
    with (data_dir / "launch_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["process"] for row in rows} == {"benchmark", "timeline_timings", "timeline"}
    assert "not pure unprofiled wall latency" in (data_dir / "launch_summary.md").read_text()
    assert "Do not subtract" in report["definitions"]["comparison"]


@pytest.mark.parametrize(
    "problem",
    [
        "source",
        "native_build",
        "tvm_ffi",
        "checkpoint",
        "workload",
        "run_id",
        "mode",
        "scopes",
        "records",
        "timing_count",
        "missing_timing",
        "nan",
    ],
)
def test_mismatched_and_incomplete_runs_are_rejected(problem):
    measurements, metadata, profile = inputs()
    if problem == "source":
        profile["source_sha256"]["models/nosa/model.py"] = "different"
    elif problem == "native_build":
        profile["native_build"]["compiler"]["version"] = "different"
    elif problem == "tvm_ffi":
        profile["tvm_ffi"] = "different"
    elif problem == "checkpoint":
        profile["checkpoint_files"]["model.safetensors"]["size"] = 2
    elif problem == "workload":
        profile["args"]["chunk_size"] = 2048
    elif problem == "run_id":
        profile["run_id"] = "different"
    elif problem == "mode":
        profile["args"]["mode"] = "profile"
    elif problem == "scopes":
        profile["module_scopes_enabled"] = True
    elif problem == "records":
        measurements["profiles"]["extend"] = [[{"stage": "cis_projection"}]]
    elif problem == "timing_count":
        profile["instrumented_timings"]["extend"].pop()
    elif problem == "missing_timing":
        del measurements["timings"]["extend"][0]["cuda_span_ms"]
    elif problem == "nan":
        measurements["timings"]["extend"][0]["wall_ms"] = float("nan")
    with pytest.raises(ValueError):
        validate_inputs(measurements, metadata, profile)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_iteration", "module_scope"])
def test_incomplete_or_instrumented_roots_are_rejected(data_dir, problem):
    if problem == "module_scope":
        with sqlite3.connect(data_dir / "nsys.sqlite") as connection:
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES (10000,20000,?,NULL,1)",
                ("NOSA/extend/layer_0/indexer_total/q65536+1024",),
            )
        with pytest.raises(ValueError, match="root-only"):
            write_report(data_dir)
        return
    launch = analyze(data_dir, data_dir / "launch_analysis.json")
    if problem == "missing":
        launch["runs"].pop()
    elif problem == "duplicate":
        launch["runs"].append(deepcopy(launch["runs"][0]))
    elif problem == "wrong_iteration":
        launch["runs"][0]["scope"] = "NOSA/profile/full_prefill/2"
    with pytest.raises(ValueError, match="iteration|profile_repeats"):
        summarize(*inputs(), launch)
