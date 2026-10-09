"""CPU-only private dispatch, evidence and observer-lifetime guarantees."""

import json
from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_model as model


@pytest.mark.parametrize(
    "rows,columns,start,prefetch,selected",
    [
        (1, 65537, 65536, {"history_length": 65536, "max_prefetch": 64}, True),
        (1, 65537, 65536, None, False),
        (2, 65538, 65536, {"history_length": 65536, "max_prefetch": 8192}, False),
        (1, 257, 256, {"history_length": 256, "max_prefetch": 64}, False),
        (1, 65537, 65536, {"history_length": 65536, "max_prefetch": 63}, False),
        (1, 65537, 65536, {"history_length": 65535, "max_prefetch": 64}, False),
    ],
)
def test_candidate_only_routes_existing_support_and_preserves_arguments(
    monkeypatch, rows, columns, start, prefetch, selected
):
    calls = []
    original_result, candidate_result = object(), object()

    def original(*args, **kwargs):
        calls.append(("original", args, kwargs))
        return original_result

    def candidate(*args, **kwargs):
        calls.append(("candidate", args, kwargs))
        return candidate_result

    monkeypatch.setattr(model.echo, "logits", original)
    monkeypatch.setattr(model.candidate, "logits", candidate)
    binding = model.Binding()
    q, k = torch.empty(rows, 1), torch.empty(columns, 1)
    weights, scales, bounds = object(), object(), object()
    with binding.public():
        with binding.bound("baseline"):
            result = model.echo.logits(
                q, k, weights, scales, start, prefetch=prefetch, _bounds=bounds, _pad_to_stride=True
            )
            assert result is original_result
        with binding.bound("candidate"):
            result = model.echo.logits(
                q, k, weights, scales, start, prefetch=prefetch, _bounds=bounds, _pad_to_stride=True
            )
            assert result is (candidate_result if selected else original_result)
        assert binding.count == int(selected)
    assert model.echo.logits is original
    for _, args, kwargs in calls:
        assert args[:5] == (q, k, weights, scales, start)
        assert (args[5] if len(args) == 6 else kwargs["prefetch"]) is prefetch
        assert kwargs["_bounds"] is bounds and kwargs["_pad_to_stride"] is True


def test_arm_changes_preserve_observer_and_restore_original_on_error(monkeypatch):
    result = object()
    error = RuntimeError("injected model failure")
    original = lambda *args, **kwargs: result
    monkeypatch.setattr(model.echo, "logits", original)
    binding = model.Binding()
    with pytest.raises(RuntimeError) as raised, binding.public():
        entry = model.echo.logits

        def observer(*args, **kwargs):
            return entry(*args, **kwargs)

        with model.patch.object(model.echo, "logits", observer):
            assert binding.observer_entry() is observer
            with binding.bound("candidate"):
                assert model.echo.logits(None, None, None, None, None) is result
                assert model.echo.logits is observer
            assert binding.arm == "baseline"
            raise error
    assert raised.value is error
    assert model.echo.logits is original and not binding.installed


def test_model_cli_preserves_separate_modes_and_balanced_default():
    args = model.parser().parse_args(
        ["bench", "--request", "request.json", "--output-dir", "output/data/run"]
    )
    assert args.mode == "bench" and args.pairs == 100
    assert args.component_receipt == model.DEFAULT_COMPONENT
    assert args.component_bench == model.DEFAULT_BENCH
    assert model.CONTRACT["warmups"] == 5


@pytest.fixture
def component_evidence(tmp_path, monkeypatch):
    inputs = {f"/inputs/layer_{layer}.pt": "digest" for layer in range(3)}
    identity = {"sources": {}, "runtime_files": {}, "inputs": inputs}
    receipt = {
        "identity": identity,
        "checks": {"complete_cases": 210, "byte_cases": 100},
        "receipt_sha256": "signature",
    }
    bench = {
        "mode": "bench",
        "identity": identity,
        "receipt_sha256": "signature",
        "rows": [],
        "samples": [],
    }
    for layer in range(3):
        for policy in model.component.POLICIES:
            group = {"case": f"layer_{layer}", "policy": policy}
            bench["rows"].append({**group, "pairs": 100, "warmups": 20, "paired_delta_us": -10.0})
            for pair in range(100):
                order = list(model.ARMS if pair % 2 == 0 else model.ARMS[::-1])
                for arm, elapsed in (("baseline", 100.0), ("candidate", 90.0)):
                    bench["samples"].append(
                        {**group, "pair": pair, "order": order, "variant": arm, "gpu_us": elapsed}
                    )
    args = SimpleNamespace(
        component_receipt=tmp_path / "receipt.json", component_bench=tmp_path / "bench.json"
    )
    args.component_receipt.write_text(json.dumps(receipt))
    monkeypatch.setattr(model, "require_receipt", lambda *_args, **_kwargs: receipt)
    monkeypatch.setattr(model, "digest", lambda _path: "digest")
    return args, bench


def test_complete_component_evidence_is_accepted(component_evidence):
    args, bench = component_evidence
    args.component_bench.write_text(json.dumps(bench))
    assert model.component_binding(args)["identity"] == bench["identity"]


@pytest.mark.parametrize("bad", ["duplicate", "warmups", "nan", "infinity", "zero"])
def test_component_gate_rejects_incomplete_or_invalid_timing(component_evidence, bad):
    args, bench = component_evidence
    if bad == "duplicate":
        bench["rows"].append(dict(bench["rows"][0]))
    elif bad == "warmups":
        bench["rows"][0]["warmups"] = 19
    else:
        bench["samples"][0]["gpu_us"] = {"nan": float("nan"), "infinity": float("inf"), "zero": 0}[
            bad
        ]
    args.component_bench.write_text(json.dumps(bench))
    with pytest.raises(RuntimeError):
        model.component_binding(args)
