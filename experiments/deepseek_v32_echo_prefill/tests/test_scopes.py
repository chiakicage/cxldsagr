from contextlib import nullcontext

import pytest
import torch

from experiments.deepseek_v32_echo_prefill.src.measure import Scopes


def test_retry_after_oversized_union_retains_failed_attempt_time(monkeypatch):
    class Event:
        clock = 0

        def __init__(self, **kwargs):
            self.time = None

        def record(self):
            self.time = Event.clock
            Event.clock += 1

        def elapsed_time(self, other):
            assert self.time is not None and other.time is not None
            return other.time - self.time

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _: nullcontext())
    scope = Scopes("extend")
    with scope("layer_0"):
        with pytest.raises(ValueError), scope("offload_exact_recall"):
            raise ValueError("union exceeds capacity")
        with scope("offload_exact_recall"):
            pass
        with scope("sparse_mla"):
            pass
    assert scope.layer == "shared" and not scope.stack
    result = scope.summary()
    assert result["stages_cuda_exclusive_ms"] == {"offload_exact_recall": 2, "sparse_mla": 1}
    assert len(result["stage_calls"]) == 3
