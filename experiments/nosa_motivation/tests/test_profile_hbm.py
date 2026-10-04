import json

import pytest

from experiments.nosa_motivation.src import profile_hbm
from experiments.nosa_motivation.src.profile_hbm import analyze_trace


def test_native_graph_nodes_use_launch_correlation_and_parent_annotation_is_not_work(tmp_path):
    events = [
        {
            "cat": "user_annotation",
            "name": "matrix_api/graph_project",
            "tid": 1,
            "ts": 0,
            "dur": 20,
        },
        {
            "cat": "cuda_runtime",
            "name": "cudaGraphLaunch",
            "tid": 1,
            "ts": 2,
            "dur": 1,
            "args": {"correlation": 42},
        },
        {"cat": "kernel", "name": "gemm", "ts": 5, "dur": 3, "args": {"correlation": 42}},
        {"cat": "kernel", "name": "activation", "ts": 7, "dur": 3, "args": {"correlation": 42}},
        {"cat": "gpu_user_annotation", "name": "parent", "ts": 0, "dur": 100},
    ]
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": events}))
    result = analyze_trace(trace)
    assert result["gpu_active_union_ms"] == 0.005
    assert result["api_gpu_union_ms"] == 0.005
    assert result["kernel_count"] == 2
    assert result["uncorrelated_activities"] == 0
    assert result["api_scope_counts"] == {"matrix_api/graph_project": 1}
    assert result["api_scopes_with_activity"] == result["api_scope_counts"]


def test_api_execution_span_retains_internal_gaps_without_counting_nested_scopes_twice(tmp_path):
    events = [
        {
            "cat": "user_annotation",
            "name": "matrix_api/project_helper",
            "tid": 1,
            "ts": 0,
            "dur": 30,
        },
        {"cat": "user_annotation", "name": "matrix_api/linear", "tid": 1, "ts": 1, "dur": 5},
        {
            "cat": "cuda_runtime",
            "name": "launch",
            "tid": 1,
            "ts": 2,
            "dur": 1,
            "args": {"correlation": 1},
        },
        {
            "cat": "cuda_runtime",
            "name": "launch",
            "tid": 1,
            "ts": 15,
            "dur": 1,
            "args": {"correlation": 2},
        },
        {"cat": "kernel", "name": "gemm", "ts": 5, "dur": 3, "args": {"correlation": 1}},
        {"cat": "kernel", "name": "helper", "ts": 20, "dur": 3, "args": {"correlation": 2}},
    ]
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": events}))
    result = analyze_trace(trace)
    assert result["api_gpu_union_ms"] == 0.006
    assert result["api_execution_span_union_ms"] == 0.018
    assert result["api_execution_span_ms"] == {
        "matrix_api/project_helper": 0.018,
        "matrix_api/linear": 0.003,
    }


def test_missing_native_launch_correlation_remains_explicit(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "kernel", "name": "unknown", "ts": 0, "dur": 2, "args": {}},
                ]
            }
        )
    )
    result = analyze_trace(trace)
    assert result["uncorrelated_activities"] == 1
    assert result["api_gpu_union_ms"] == 0


def test_failure_moves_only_new_experiment_artifacts_and_preserves_existing_run(
    tmp_path, monkeypatch
):
    experiment, external = tmp_path / "experiment", tmp_path / "failure"
    external.mkdir()
    data, profile = experiment / "data", experiment / "profile"
    profile.mkdir(parents=True)
    preserved = profile / "valid.json"
    preserved.write_text("unchanged")
    monkeypatch.setattr(profile_hbm, "EXPERIMENT", experiment)
    monkeypatch.setattr(profile_hbm.tempfile, "mkdtemp", lambda **kwargs: str(external))
    with pytest.raises(FileExistsError), profile_hbm._output_directories(data, profile):
        raise AssertionError("existing run directory was silently reused")
    assert preserved.read_text() == "unchanged"
    assert not data.exists()
    assert (external / "0").is_dir()
