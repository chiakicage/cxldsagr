"""Lean scope balance, checked-source separation and post-capture graph ledger."""

from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_mfu.src.gap_profile import (
    ModelScopes,
    annotate,
    measurement_sources,
    profiler_capture,
)
from experiments.deepseek_v32_mfu.src.profile_layers import sources


def test_lean_scopes_preserve_layer_nesting_and_restore_after_failure(monkeypatch):
    events = []
    monkeypatch.setattr("torch.cuda.nvtx.range_push", lambda label: events.append(("push", label)))
    monkeypatch.setattr("torch.cuda.nvtx.range_pop", lambda: events.append(("pop", None)))
    scopes = ModelScopes("echo", "extend_annotated")
    with (
        pytest.raises(RuntimeError, match="failure"),
        scopes("forward_misc"),
        scopes("layer_2"),
        scopes("indexer_prefetch"),
    ):
        raise RuntimeError("failure")
    assert scopes.layer == "shared"
    assert [event[0] for event in events] == ["push", "push", "push", "pop", "pop", "pop"]
    calls = scopes.calls(None)
    assert [row["stage"] for row in calls] == ["forward_misc", "indexer_fused"]
    assert calls[1]["layer"] == "layer_2"
    assert calls[1]["useful_flops"] is None


def test_graph_ledger_expands_only_when_requested_after_scope_recording(monkeypatch):
    monkeypatch.setattr("torch.cuda.nvtx.range_push", lambda _: None)
    monkeypatch.setattr("torch.cuda.nvtx.range_pop", lambda: None)
    scopes = ModelScopes("hbm", "extend_annotated")
    with scopes("layer_0"), scopes("compute_graph_projection_layer_0_q_128"):
        pass

    class Capture:
        invocations = 0

        def expand_replay(self, row, call_id):
            self.invocations += 1
            row["graph_replay"] = True
            return [{"call_id": call_id, "graph_api": True}]

    capture = Capture()
    assert capture.invocations == 0
    calls = scopes.calls(capture)
    assert capture.invocations == 1
    assert calls[0]["graph_replay"] and calls[1]["graph_api"]


def test_measurement_helpers_have_separate_identity_from_checked_execution():
    measurement = measurement_sources()
    execution = sources()
    assert len(measurement) == 2
    assert all(len(value) == 64 for value in measurement.values())
    assert not measurement.keys() & execution.keys()


def test_profiler_stop_failure_retains_original_exception(monkeypatch):
    original, cleanup = RuntimeError("model failed"), RuntimeError("stop failed")

    def stop():
        raise cleanup

    runtime = SimpleNamespace(cudaProfilerStart=lambda: None, cudaProfilerStop=stop)
    monkeypatch.setattr("torch.cuda.cudart", lambda: runtime)
    with pytest.raises(BaseExceptionGroup) as caught, profiler_capture():
        raise original
    assert caught.value.exceptions == (original, cleanup)


def test_warmup_and_fresh_state_restoration_happen_with_profiler_active(monkeypatch):
    events = []
    runtime = SimpleNamespace(
        cudaProfilerStart=lambda: events.append("start"),
        cudaProfilerStop=lambda: events.append("stop"),
    )
    monkeypatch.setattr("torch.cuda.cudart", lambda: runtime)
    monkeypatch.setattr("torch.cuda.nvtx.range_push", lambda _: None)
    monkeypatch.setattr("torch.cuda.nvtx.range_pop", lambda: None)

    class Model:
        num_layers = 1

        def __init__(self):
            self.blocks = [SimpleNamespace(forward=lambda: (torch.ones(1), torch.ones(1)))]

        def synchronize(self):
            events.append("sync")

        def forward(self, ids, *, scope):
            events.append(scope.phase)
            with scope("layer_0"):
                first, second = self.blocks[0].forward()
            return first + second

    output, hidden, calls, wall = annotate(
        Model(),
        [1],
        "hbm",
        "prefill_annotated",
        prepare=lambda: events.append("restore"),
        trace_warmups=1,
    )
    assert events == [
        "sync",
        "start",
        "restore",
        "prefill_trace_warmup",
        "restore",
        "sync",
        "prefill_annotated",
        "stop",
    ]
    assert hidden is None and torch.equal(output, torch.full((1,), 2.0))
    assert wall >= 0 and len(calls) == 1
    assert calls[0]["phase"] == "prefill_annotated"
