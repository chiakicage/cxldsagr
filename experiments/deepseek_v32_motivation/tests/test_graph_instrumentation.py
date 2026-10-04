"""Node ownership is captured without adding timing or event graph nodes."""

from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_motivation.src.graph_instrumentation import CaptureGraphOperators
from experiments.deepseek_v32_motivation.src.profile import Scopes, cpu_summary


class Inspector:
    def __init__(self):
        self.nodes = {}

    def snapshot(self, stream):
        assert stream == 123
        return 1, dict(self.nodes)


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=123)
    )
    monkeypatch.setattr(
        "experiments.deepseek_v32_motivation.src.graph_instrumentation.independent_weight_identity",
        lambda attention, block: {"weight": {"data_ptr": 1}},
    )
    recorder = CaptureGraphOperators(SimpleNamespace())
    recorder.inspector = Inspector()
    return recorder


def record_graph(recorder):
    bank = SimpleNamespace(device="cuda:0", attentions=[object()], blocks=[object()])
    pair = SimpleNamespace(layer=0, queries=128, residual_present=False)
    with recorder.graph_scope(bank, pair, "projection"):
        recorder.inspector.nodes[10] = 0  # Auxiliary normalization kernel.
        with recorder("q_a_proj", useful_flops=1024, precision="BF16"):
            recorder.inspector.nodes[11] = 0
            recorder.inspector.nodes[12] = 1  # Copy helper within the API.
        with recorder("reshape", useful_flops=None):
            recorder.inspector.nodes[13] = 2
    recorder.graphs["projection", 0, 128]["executable_graph_id"] = 7


def test_capture_ownership_and_replay_rows_preserve_flops_without_cpu_time(recorder):
    record_graph(recorder)
    graph = recorder.graphs["projection", 0, 128]
    assert graph["gpu_node_ids"] == [10, 11, 12, 13]
    assert graph["operators"][0]["graph_node_ids"] == [11, 12]
    scopes = Scopes("hbm", "cold", 2, nvtx=False, graph_operators=recorder)
    with scopes.context(segment="history", chunk=0, layer=0), scopes("request"):
        with scopes("compute_graph_projection_layer_0_q_128"):
            pass
        with scopes("after_graph"):
            pass
    request, replay, api, after = scopes.calls
    assert [call["call_id"] for call in scopes.calls] == list(range(4))
    assert api["parent_call_id"] == replay["call_id"]
    assert after["parent_call_id"] == request["call_id"]
    assert api["graph_api"] and not api["graph_replay"]
    assert api["cpu_inclusive_ns"] == 0
    assert replay["graph_id"] == 7
    assert api["graph_replay_nvtx"] == replay["nvtx"]
    assert api["nvtx"] != replay["nvtx"]
    summary = {row["stage"]: row for row in cpu_summary(scopes.calls)}
    assert summary["q_a_proj"]["useful_flops"] == 1024
    assert summary["q_a_proj"]["cpu_inclusive_ns"] == 0
    assert not scopes.active


def test_missing_matrix_nodes_and_nested_ownership_fail(recorder):
    bank = SimpleNamespace(device="cuda:0", attentions=[object()], blocks=[object()])
    pair = SimpleNamespace(layer=0, queries=128, residual_present=False)
    with (
        pytest.raises(RuntimeError, match="no GPU nodes"),
        recorder.graph_scope(bank, pair, "projection"),
        recorder("matrix", useful_flops=1),
    ):
        pass
    assert recorder.current is None
    with (
        pytest.raises(RuntimeError, match="ambiguous"),
        recorder.graph_scope(bank, pair, "projection"),
        recorder("outer", useful_flops=1),
        recorder("inner", useful_flops=1),
    ):
        recorder.inspector.nodes[10] = 0
    assert recorder.current is None


def test_scope_error_does_not_emit_replay_matrix_rows(recorder):
    record_graph(recorder)
    scopes = Scopes("hbm", "cold", 2, nvtx=False, graph_operators=recorder)
    with (
        pytest.raises(RuntimeError, match="failed"),
        scopes("compute_graph_projection_layer_0_q_128"),
    ):
        raise RuntimeError("replay failed")
    assert len(scopes.calls) == 1
    assert scopes.calls[0]["raised"]
    assert not scopes.active
