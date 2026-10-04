"""Performance calls reuse exact acceptance without repeating comparisons."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest

from evaluation.validation import write_receipt
from experiments.nosa_kernel_mfu.src import measure, measure_modules, phases


class Event:
    def __init__(self, **kwargs):
        pass

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 2.0


def fail_comparison(*args, **kwargs):
    raise AssertionError("Performance samples must not run numerical comparisons")


def test_operator_bench_never_clones_or_compares_outputs(monkeypatch):
    monkeypatch.setattr(measure.torch.cuda, "Event", Event)
    monkeypatch.setattr(measure.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        measure.torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=lambda: None)
    )
    monkeypatch.setattr(measure.torch.cuda, "graph", lambda graph: nullcontext())
    monkeypatch.setattr(measure.torch.testing, "assert_close", fail_comparison)
    # A bare object has no clone(), so this also catches a hidden reference copy.
    result = measure.measure(lambda: object(), SimpleNamespace(warmup=1, repeats=3, graph_calls=2))
    assert result["graph"]["samples_ms"] == [1.0] * 3
    assert result["eager"]["samples_ms"] == [2.0] * 3


def test_module_bench_has_no_reference_or_profiler_between_samples(monkeypatch):
    monkeypatch.setattr(measure_modules.torch.cuda, "Event", Event)
    monkeypatch.setattr(measure_modules, "assert_same_output", fail_comparison)
    monkeypatch.setattr(measure_modules.torch.profiler, "profile", fail_comparison)
    lifecycle = []
    fixture = SimpleNamespace(
        prepare=lambda: lifecycle.append("prepare"),
        call=lambda module: lifecycle.append(module),
        cleanup=lambda: lifecycle.append("cleanup"),
    )
    args = SimpleNamespace(mode="bench", warmup=2, repeats=3, profile_repeats=7)
    result = measure_modules.measure_module(fixture, "indexer_total", args, "fixture")
    assert lifecycle == ["prepare", "indexer_total", "cleanup"] * 5
    assert result["eager_api"]["samples_ms"] == [2.0] * 3
    assert "profile_kernel_sum" not in result


def metadata():
    return {
        "source_sha256": {"operator.cu": "source-a"},
        "native_build": {"flags": ["sm90"], "headers": "header-a"},
        "gpu": {"capability": [9, 0], "uuid": "device-a"},
        "torch": "2",
        "cuda": "13",
        "dependencies": {"triton": "3"},
        "environment": {"CXLDSAGR_SM90_BACKEND": "native"},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", {"operator.cu": "source-b"}),
        ("native_build", {"flags": ["sm90"], "headers": "header-b"}),
        ("environment", {"CXLDSAGR_SM90_BACKEND": "triton"}),
        ("gpu", {"capability": [9, 0], "uuid": "device-b"}),
    ],
)
def test_bench_rejects_changed_runtime_before_loading_case(tmp_path, field, value):
    original = metadata()
    identity = phases.validation_identity(
        original, config={"queries": 8}, inputs=None, cache="resident"
    )
    path = tmp_path / "check.json"
    write_receipt(path, kind="fixture", identity=identity, checks={"passed": True, "cases": {}})
    changed = deepcopy(original)
    changed[field] = value
    args = SimpleNamespace(
        mode="bench", validation_receipt=path, input_dir=None, output_dir=tmp_path / "bench"
    )
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="execution identity"):
        phases.open_validation(
            args, changed, kind="fixture", config={"queries": 8}, cache="resident"
        )


def test_real_tensor_content_and_stride_must_match_receipt():
    checked = {
        "q": {
            "shape": [8, 32, 128],
            "stride": [4608, 128, 1],
            "dtype": "torch.bfloat16",
            "sha256": "q-a",
        }
    }
    args = SimpleNamespace(mode="bench", validation_cases={"layer_00": {"tensors": checked}})
    assert phases.check_case_identity(args, "layer_00", checked)["tensors"] == checked
    for key, value in (("stride", [4096, 128, 1]), ("sha256", "q-b"), ("dtype", "torch.float16")):
        changed = deepcopy(checked)
        changed["q"][key] = value
        with pytest.raises(ValueError, match="input/layout mismatch"):
            phases.check_case_identity(args, "layer_00", changed)
