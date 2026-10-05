"""Both native drivers obey one admission, output and transaction contract.

NOSA uses its tiny CPU reference model. DeepSeek uses the existing deterministic
block fixture with its real dense cache, resource provider and model loop. These
checks cover adapter behavior; checkpoint numerics remain independent GPU gates.
"""

from dataclasses import replace

import pytest
import torch

from cache.capacity import CacheFootprint, CapacityPolicy, allocation_footprint
from executor.contracts import OutputSpec, RequestShape
from executor.runtime import TokenRuntime


@pytest.fixture(params=("nosa", "deepseek"))
def native_runtime(request, monkeypatch):
    if request.param == "nosa":
        from models.nosa.execution.adapter import NosaServingBackend
        from models.nosa.tests.test_model import tiny_config
        from models.nosa.tests.test_sparse_model import initialized_sparse_model

        model = initialized_sparse_model(
            tiny_config(max_position_embeddings=64, num_hidden_layers=2)
        )
        backend = NosaServingBackend(model, "hbm", chunk_size=4)
        fail_target = model.model.layers[1]
        raw_mutation = lambda session: session.truncate(8)
    else:
        from models.deepseek_v32.tests.test_serving_backend import _dense_model

        backend = _dense_model(monkeypatch)
        backend.close()
        fail_target = backend.blocks[1]
        raw_mutation = lambda session: session.runners[0].cache.truncate(8)
    policy = CapacityPolicy.byte_budget(CacheFootprint(1 << 26, 1 << 26))
    runtime = TokenRuntime(backend.runtime_driver(policy))
    resource_plan = runtime.plan_resources(
        policy,
        {"max_session_capacity": 64, "max_history_tokens": 56, "max_candidate_tokens": 8},
    )
    owner = object()
    runtime.driver.bind_owner(owner)
    runtime.allocate_resources(resource_plan)
    plan = runtime.plan_session(RequestShape(8, 8), "stable-history")
    session = runtime.create_session(plan)
    yield runtime, backend, session, plan, fail_target, raw_mutation
    for state in tuple(runtime._sessions.values()):
        runtime.release_session(state.value)
    runtime.driver.unbind_owner(owner)
    runtime.close()


def test_native_candidate_preserves_history_outputs_and_one_session_plan(native_runtime):
    runtime, _, session, plan, _, _ = native_runtime
    assert plan.allocations
    assert allocation_footprint(plan.allocations).fits(plan.reservation)
    ids = torch.arange(16) % 19
    runtime.prefill(session, ids[:8])
    spec = OutputSpec(
        logits="last" if runtime.driver.output_spec.run_lm_head else "none",
        run_lm_head=runtime.driver.output_spec.run_lm_head,
    )
    observed = []
    first = runtime.candidate(session, ids[8:12], spec, observer=observed.append)
    saved = first.hidden.clone()
    saved_logits = None if first.logits is None else first.logits.clone()
    second = runtime.candidate(session, ids[10:16], replace(spec, hidden="last"))
    assert observed == ["executed", "cleaned"]
    assert first.hidden.shape[0] == 4 and second.hidden.shape[0] == 1
    assert runtime.driver.session_length(session) == 8
    torch.testing.assert_close(first.hidden, saved, rtol=0, atol=0)
    if saved_logits is not None:
        torch.testing.assert_close(first.logits, saved_logits, rtol=0, atol=0)
    assert runtime.session_usage(session).charged.fits(plan.reservation)


def test_native_owner_and_foreign_plan_reject_before_cache_mutation(native_runtime):
    runtime, backend, session, plan, _, raw_mutation = native_runtime
    with pytest.raises(ValueError, match="foreign"):
        runtime.create_session(replace(plan, resource_identity=object()))
    with pytest.raises(RuntimeError, match="owner"):
        backend.create_session(plan.retained_capacity)
    with pytest.raises(RuntimeError, match="owner"):
        runtime.driver.bind_owner(object())
    runtime.prefill(session, torch.arange(8))
    with pytest.raises(RuntimeError):
        backend.truncate(session, 8)
    with pytest.raises(RuntimeError):
        raw_mutation(session)
    with pytest.raises(ValueError):
        runtime.candidate(
            session,
            torch.tensor([9]),
            OutputSpec(run_lm_head=not runtime.driver.output_spec.run_lm_head),
        )
    assert runtime.driver.session_length(session) == 8


def test_native_append_commits_but_failed_runtime_session_cannot_retry(native_runtime, monkeypatch):
    runtime, _, session, _, target, _ = native_runtime
    runtime.prefill(session, torch.arange(8))
    result = runtime.append(session, torch.tensor([8, 9]))
    assert result.hidden.shape[0] == 2
    assert runtime.driver.session_length(session) == 10

    def fail(*args, **kwargs):
        raise RuntimeError("injected block failure")

    with monkeypatch.context() as patch:
        patch.setattr(target, "forward", fail)
        with pytest.raises(RuntimeError, match="injected block failure"):
            runtime.append(session, torch.tensor([10, 11]))
    assert runtime.driver.session_length(session) == 10
    with pytest.raises(RuntimeError, match="after failure"):
        runtime.append(session, torch.tensor([10, 11]))
