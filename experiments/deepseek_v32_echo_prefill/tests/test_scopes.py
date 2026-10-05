from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from experiments.deepseek_v32_echo_prefill.src import kernel_profile, measure, report
from experiments.deepseek_v32_echo_prefill.src.measure import Scopes


def test_runtime_collectors_include_complete_model_and_shared_operator_sources():
    model_sources = measure.source_manifest()
    replay_sources = kernel_profile.source_manifest()
    assert "models/attention_contracts.py" in model_sources
    assert not any(name.startswith("layers/") for name in model_sources)
    assert report.REQUIRED_SOURCES <= model_sources.keys()
    operator_sources = {
        name for name in report.REQUIRED_SOURCES if name.startswith("operators/")
    } | {"operators/deepseek_v32/attention/reference/torch.py"}
    assert operator_sources <= replay_sources.keys()
    for sources in (model_sources, replay_sources):
        assert {
            "experiments/deepseek_v32_echo_prefill/src/backend_provenance.py",
            "3rdparty/FlashMLA/csrc/kernels/sm90/prefill/sparse/phase1.cuh",
            "3rdparty/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh",
            "3rdparty/cutlass/include/cute/tensor.hpp",
        } <= sources.keys()
        assert all(Path(name).is_file() for name in sources)
        assert not any(name.startswith("operators/sm90/") for name in sources)
        assert not any("tests" in Path(name).parts for name in sources)


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
