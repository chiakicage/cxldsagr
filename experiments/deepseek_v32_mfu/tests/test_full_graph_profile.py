from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_mfu.src import full_graph_profile as profile


def test_capture_assigns_innermost_stage_without_replay_cpu_scopes(monkeypatch):
    nodes = {}

    class Inspector:
        def __init__(self):
            self.provenance = {"fixture": True}

        def snapshot(self, stream):
            return 7, dict(nodes)

        def executable_id(self, graph):
            return 19

    monkeypatch.setattr(profile, "GraphInspector", Inspector)
    monkeypatch.setattr(profile.torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        profile.torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=3)
    )
    capture = profile.FullExtendGraphCapture("echo")
    with capture("extend_graph_body"):
        with capture("embedding"):
            nodes[1] = 0
        with capture("layer_0"):
            with capture("attention_projection"):
                nodes[2] = 0
            with capture("offload_prepare"):
                nodes[3] = 0
        with capture("final_norm_lm_head"):
            nodes[4] = 0
    template = capture.finalize(SimpleNamespace(graph=object(), describe=dict))
    assert template["node_owners"]["3"] == {
        "layer": "layer_0",
        "stage": "offload_prepare",
        "source_stage": "offload_prepare",
        "stage_path": ["extend_graph_body", "layer_0", "offload_prepare"],
    }
    replay = {"stage": "extend_graph_replay_q_128"}
    assert capture.expand_replay(replay, 1) == []
    assert replay["full_extend_graph"] and replay["graph_gpu_node_ids"] == [1, 2, 3, 4]


def replay_fixture():
    scope = {"label": "full/replay", "stage": "extend_graph_replay_q_128", "layer": "shared"}
    calls = [
        {
            "full_extend_graph": True,
            "nvtx": scope["label"],
            "graph_id": 19,
            "graph_gpu_node_ids": [1, 2],
            "graph_node_owners": {
                "1": {"stage": "attention_projection", "layer": "layer_0"},
                "2": {"stage": "offload_prepare", "layer": "layer_0"},
            },
        }
    ]
    activities = [
        {"kind": "kernel", "graph_id": 19, "graph_node_id": node, "scope": scope}
        for node in (101, 102)
    ]
    apis = [{"name": "cudaGraphLaunch_v10000", "scope": scope}]
    return scope, calls, activities, apis


def test_replay_requires_one_launch_complete_node_membership_and_preserves_cpu_scopes():
    scope, calls, activities, apis = replay_fixture()
    audit = profile.attribute_full_graph_replays(
        activities, calls, {101: 1, 102: 2}, scopes=[scope], apis=apis
    )
    assert audit["replays"] == 1 and audit["gpu_node_activities"] == 2
    assert activities[1]["graph_stage"] == "offload_prepare"
    assert activities[1]["graph_layer"] == "layer_0"
    assert activities[1]["graph_call"] is None
    assert activities[1]["scope"] is scope and scope["layer"] == "shared"
    for mutation, message in (
        (lambda a, p: a.pop(), "missing or duplicate"),
        (lambda a, p: a.append(dict(a[0])), "missing or duplicate"),
        (lambda a, p: p.append(dict(p[0])), "exactly one CUDA graph launch"),
    ):
        scope, calls, activities, apis = replay_fixture()
        mutation(activities, apis)
        with pytest.raises(ValueError, match=message):
            profile.attribute_full_graph_replays(activities, calls, {101: 1, 102: 2}, apis=apis)


def test_capture_ignores_warmup_and_rejects_unnamed_gpu_work(monkeypatch):
    capture = object.__new__(profile.FullExtendGraphCapture)
    monkeypatch.setattr(profile.torch.cuda, "is_current_stream_capturing", lambda: False)
    with capture("embedding"), nullcontext():
        pass
    capture.nodes = {1: 0}
    capture.owners = {1: {"stage": "extend_graph_body", "layer": "shared"}}
    capture.stack = []
    with pytest.raises(RuntimeError, match="outside every named stage"):
        capture.finalize(None)


def test_matrix_capture_retains_api_work_and_exclusive_nodes_without_changing_stages(monkeypatch):
    nodes = {}

    class Inspector:
        def __init__(self):
            self.provenance = {"fixture": True}

        def snapshot(self, stream):
            return 7, dict(nodes)

        def executable_id(self, graph):
            return 19

    monkeypatch.setattr(profile, "GraphInspector", Inspector)
    monkeypatch.setattr(profile.torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        profile.torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=3)
    )
    capture = profile.FullExtendGraphCapture("echo", record_operators=True)
    with capture("extend_graph_body"), capture("layer_0"), capture("attention_projection"):
        with capture("q_a_proj", useful_flops=128, precision="FP8", shape=[2, 8, 4]):
            nodes[1] = 0
            nodes[2] = 0
        with capture("rms_norm", useful_flops=None, precision=None):
            nodes[3] = 0
    template = capture.finalize(SimpleNamespace(graph=object(), describe=dict))
    assert {row["stage"] for row in template["node_owners"].values()} == {"attention_projection"}
    replay = {
        "stage": "extend_graph_replay_q_128",
        "call_id": 5,
        "nvtx": "full/replay",
        "layer": "shared",
    }
    (api,) = capture.expand_replay(replay, 6)
    assert api["layer"] == "layer_0" and api["stage"] == "q_a_proj"
    assert api["shape"] == [2, 8, 4] and api["useful_flops"] == 128
    assert api["graph_node_ids"] == [1, 2]
    assert api["graph_api"] and not api["graph_replay"] and not api["full_extend_graph"]
    scope = {"label": "full/replay", "stage": replay["stage"], "layer": "shared"}
    activities = [
        {"kind": "kernel", "graph_id": 19, "graph_node_id": node, "scope": scope}
        for node in (101, 102, 103)
    ]
    audit = profile.attribute_full_graph_replays(
        activities, [replay, api], {101: 1, 102: 2, 103: 3}
    )
    assert [row["graph_call"] for row in activities] == [api, api, None]
    assert audit["matrix_api_count"] == 1 and audit["matrix_api_gpu_node_activities"] == 2


@pytest.mark.parametrize("nodes", [[1, 1], [3], [], [1]])
def test_full_graph_rejects_duplicate_unknown_or_overlapping_api_nodes(nodes):
    _, calls, activities, _ = replay_fixture()
    api = {"graph_api": True, "graph_replay_nvtx": "full/replay", "graph_node_ids": nodes}
    calls += [api]
    if nodes == [1]:
        calls.append(dict(api))
    with pytest.raises(ValueError, match="matrix API"):
        profile.attribute_full_graph_replays(activities, calls, {101: 1, 102: 2})
