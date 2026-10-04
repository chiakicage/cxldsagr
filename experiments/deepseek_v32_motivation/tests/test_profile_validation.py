"""Captured requests must observe the validator selected before instrumentation."""

from contextlib import nullcontext

import pytest
import torch

from experiments.deepseek_v32_echo_prefill.src import operator_instrumentation
from experiments.deepseek_v32_motivation.src.profile import InstrumentServing, Scopes
from serving import token_validation
from serving.persistent import PersistentGRRunner
from serving.tests.test_persistent import Backend, request


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_profile_wraps_selected_validator_once_and_restores_it(monkeypatch, native, invalid):
    monkeypatch.setattr(token_validation, "prepare", lambda: token_validation.reference)
    monkeypatch.setattr(token_validation, "runtime_info", lambda: {"backend": "cpython_native"})
    # Validation runs real runner code. Matrix instrumentation is outside this
    # CPU test; the full InstrumentServing context still installs/restores hooks.
    monkeypatch.setattr(operator_instrumentation, "InstrumentOperators", lambda *_: nullcontext())
    backend = Backend()
    backend._forward = lambda *args, **kwargs: None
    backend.extend_candidate = backend.extend
    backend.session_metrics = lambda *_: {}
    backend.attentions, backend.blocks = (), ()
    backend.head_weight = torch.empty(0)
    scopes = Scopes("echo", "cold", 1, nvtx=False)
    with PersistentGRRunner(
        backend,
        hbm_budget_bytes=64,
        dram_budget_bytes=64,
        native_token_validation=native,
    ) as runner:
        original = runner._validate
        original_local = "_validate" in vars(runner)
        assert original_local is native
        instrument = InstrumentServing(backend, scopes, runner)
        value = request()
        if invalid:
            value["input_ids"] = [True, 2, 3, 4]
        expected = pytest.raises(ValueError) if invalid else nullcontext()
        with expected, instrument:
            runner._validate(value)
        assert len(scopes.calls) == 1
        assert scopes.calls[0]["stage"] == "request_validation"
        assert scopes.calls[0]["segment"] == "admission"
        assert bool(scopes.calls[0].get("raised")) is invalid
        assert not scopes.active
        assert instrument.targets.count("PersistentGRRunner._validate") == 1
        assert runner._validate.__func__ is original.__func__
        assert runner._validate.__self__ is runner
        assert ("_validate" in vars(runner)) is original_local
        if native:
            assert runner._validate is original
        runner._validate(request())
        assert len(scopes.calls) == 1
