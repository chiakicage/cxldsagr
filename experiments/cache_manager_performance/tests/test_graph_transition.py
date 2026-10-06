"""One graph submission must not become fabricated per-stage CPU timings."""

import copy
import csv

import pytest

from experiments.cache_manager_performance.src import transition_report
from experiments.cache_manager_performance.src.transition_metrics import analyze_graph_transition
from experiments.cache_manager_performance.tests.test_transition_metrics import (
    PROCESS,
    THREAD,
    make_trace,
)


def graph_trace():
    _, _, activities = make_trace()
    scope = {
        "id": 100,
        "stage": "extend_graph_replay_q_128",
        "layer": "shared",
        "thread": THREAD,
        "label": "echo/extend/full_graph",
        "start": 0,
        "end": 5,
    }
    launch = {
        "id": 100,
        "scope": scope,
        "thread": THREAD,
        "process": PROCESS,
        "start": 1,
        "end": 3,
        "name": "cudaGraphLaunch",
        "correlation": 20,
    }
    for row in activities:
        row.update(
            graph_stage=row["scope"]["stage"],
            graph_layer="layer_0",
            full_extend_graph=True,
            graph_node_id=row["id"] + 1,
            graph_id=9,
            scope=scope,
            api=launch,
            correlation=20,
        )
    activities.append(
        {
            **activities[1],
            "id": 20,
            "graph_node_id": 20,
            "name": "invalid_id_mask",
            "lane": "GPU control",
            "start": 55,
            "end": 58,
        }
    )
    return [launch], activities


def analyze(trace):
    return analyze_graph_transition(
        *trace,
        layer=0,
        indexer_stage="indexer_fused",
        topk_stage="exact_topk",
        consumer_stage="sparse_mla",
        lane_classifier=lambda row: row["lane"],
    )


def test_graph_uses_last_topk_node_and_real_mla_without_fabricating_cpu_scopes():
    trace = graph_trace()
    original = copy.deepcopy(trace)
    result = analyze(trace)
    assert result["cpu_topk_to_consumer"] is None
    assert result["boundaries"]["topk_scope"] is None
    assert result["boundaries"]["consumer_scope"] is None
    assert result["boundaries"]["attention_launch_start_minus_topk_end_ms"] is None
    assert result["boundaries"]["graph_launch_start_minus_topk_end_ms"] == -57 / 1e6
    gpu = result["gpu_topk_to_consumer"]
    assert (gpu["start_ns"], gpu["end_ns"]) == (58, 150)
    assert gpu["window_ms"] == 92 / 1e6
    assert gpu["io_union_ms"] == 30 / 1e6
    assert gpu["exposed_control_ms"] == 48 / 1e6
    assert gpu["gpu_idle_ms"] == 14 / 1e6
    assert all(stage["cpu"] is None for stage in result["stages"])
    assert all(row["scope"]["layer"] == "shared" for row in gpu["activities"])
    assert result["post_topk_idle_submission"]["idle_before_graph_launch_ms"] == 0
    assert result["post_topk_idle_submission"]["idle_after_graph_launch_ms"] == 14 / 1e6
    assert trace == original


def test_graph_interval_includes_concurrent_other_stream_work_once():
    apis, activities = graph_trace()
    activities.append(
        {
            **activities[4],
            "id": 21,
            "graph_node_id": 21,
            "graph_layer": "layer_1",
            "stream_id": 2,
            "start": 125,
            "end": 145,
        }
    )
    gpu = analyze((apis, activities))["gpu_topk_to_consumer"]
    assert gpu["exposed_control_ms"] == 55 / 1e6
    assert gpu["gpu_idle_ms"] == 7 / 1e6
    assert sum(gpu["partition_ms"].values()) == pytest.approx(gpu["window_ms"])


@pytest.mark.parametrize("change", ["missing_owner", "other_launch", "split_mla"])
def test_graph_rejects_unverified_or_ambiguous_transition(change):
    apis, activities = graph_trace()
    if change == "missing_owner":
        activities[0]["full_extend_graph"] = False
    elif change == "other_launch":
        activities[0]["api"] = {**apis[0], "correlation": 123}
    else:
        activities.append({**activities[6], "id": 21})
    with pytest.raises(ValueError):
        analyze((apis, activities))


def test_graph_report_leaves_absent_cpu_metrics_blank(tmp_path, monkeypatch):
    detail = analyze(graph_trace())
    sample = {
        **detail,
        "source_kind": "complete_model",
        "run_id": "full-graph",
        "scheme": "echo",
        "phase": "extend_cold",
        "layer": 0,
        "sample": 0,
    }
    sample["stages"] = [
        row
        for row in detail["stages"]
        if row["stage"] in {"prefetch_hint", "cache_write", "offload_exact_recall", "sparse_mla"}
    ]
    monkeypatch.setattr(transition_report, "_plot", lambda *args: None)
    transition_report.render({"samples": [sample], "analysis_id": "cpu-fixture"}, tmp_path)
    with (tmp_path / "windows.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["submission_kind"] == "complete_cuda_graph"
    assert row["cpu_topk_to_wrapper_us"] == ""
    assert row["mla_launch_minus_topk_end_us"] == ""
    assert float(row["graph_launch_minus_topk_end_us"]) == pytest.approx(-0.057)
    with (tmp_path / "stages.csv").open() as stream:
        assert all(row["cpu_scope_us"] == "" for row in csv.DictReader(stream))
