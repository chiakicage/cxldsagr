"""Token-only contract behavior through explicitly configured model adapters."""

from dataclasses import replace

import pytest
import torch

from cache.capacity import CacheFootprint, CapacityPolicy
from executor.contracts import OutputSpec, RequestShape
from executor.runtime import TokenRuntime
from serving.tests.test_persistent import Backend, ResidentSharedBackend


@pytest.fixture(params=["append_truncate", "gpu_transient"])
def runtime_case(request):
    backend = Backend() if request.param == "append_truncate" else ResidentSharedBackend()
    policy = CapacityPolicy.byte_budget(CacheFootprint(1024, 1024))
    runtime = TokenRuntime(backend.runtime_driver(policy))
    plan = runtime.plan_resources(
        policy,
        {"max_session_capacity": 6, "max_history_tokens": 2, "max_candidate_tokens": 4},
    )
    runtime.allocate_resources(plan)
    session_plan = runtime.plan_session(RequestShape(2, 4), (2, "history"))
    session = runtime.create_session(session_plan)
    yield runtime, backend, session, session_plan
    for state in tuple(runtime._sessions.values()):
        runtime.release_session(state.value)
    runtime.close()


def test_candidate_preserves_history_and_owned_outputs(runtime_case):
    runtime, backend, session, plan = runtime_case
    assert runtime.prefill(session, torch.tensor([1, 2])).hidden is None
    observations = []
    result = runtime.candidate(
        session,
        torch.tensor([3, 4]),
        observer=lambda stage: observations.append((stage, list(session["tokens"]))),
    )
    assert result.hidden.tolist() == [[6], [10]]
    assert observations[-1] == ("cleaned", [1, 2])
    during = [1, 2] if runtime.driver.candidate_mode == "gpu_transient" else [1, 2, 3, 4]
    assert observations[0] == ("executed", during)
    last = runtime.candidate(session, torch.tensor([5, 6, 7]), OutputSpec(hidden="last"))
    assert last.hidden.tolist() == [[21]]
    assert result.hidden.tolist() == [[6], [10]]
    assert session["tokens"] == [1, 2] and backend.built == 1
    assert runtime.session_usage(session).charged.fits(plan.reservation)


def test_plan_identity_and_invalid_input_reject_before_mutation(runtime_case):
    runtime, backend, session, plan = runtime_case
    with pytest.raises(ValueError, match="foreign"):
        runtime.create_session(replace(plan, resource_identity=object()))
    with pytest.raises(ValueError, match="integral"):
        runtime.prefill(session, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="planned history"):
        runtime.prefill(session, torch.tensor([1]))
    assert not session["tokens"] and backend.built == 0
    runtime.prefill(session, torch.tensor([1, 2]))
    for spec in (OutputSpec(run_lm_head=True), OutputSpec(ownership="borrowed")):
        with pytest.raises(ValueError):
            runtime.candidate(session, torch.tensor([3]), spec)
        assert session["tokens"] == [1, 2]
    with pytest.raises(ValueError, match="planned session capacity"):
        runtime.candidate(session, torch.tensor([3, 4, 5, 6, 7]))
    with pytest.raises(ValueError, match="foreign"):
        runtime.candidate(dict(session), torch.tensor([3]))


def test_append_retains_original_persistent_semantics(runtime_case):
    runtime, _, session, plan = runtime_case
    runtime.prefill(session, torch.tensor([1, 2]))
    if runtime.driver.candidate_mode == "gpu_transient":
        # A full H-only allocation cannot silently grow persistent history.
        with pytest.raises(ValueError, match="allocation capacity"):
            runtime.append(session, torch.tensor([3, 4]))
        assert session["tokens"] == [1, 2]
    else:
        output = runtime.append(session, torch.tensor([3, 4]))
        assert output.hidden.tolist() == [[6], [10]]
        assert session["tokens"] == [1, 2, 3, 4]
        with pytest.raises(ValueError, match="planned history"):
            runtime.candidate(session, torch.tensor([5]))
    assert session["capacity"] == plan.retained_capacity


def test_reporting_metadata_does_not_select_candidate_execution():
    backend = Backend()
    adapter = backend.runtime_driver(CapacityPolicy.byte_budget(CacheFootprint(100, 0)))
    # The explicit nonshared adapter does not consult a newly attached hook.
    backend.extend_candidate = lambda *_: pytest.fail("undeclared candidate capability")
    runtime = TokenRuntime(adapter)
    plan = runtime.plan_resources(CapacityPolicy.byte_budget(CacheFootprint(100, 0)), {})
    runtime.allocate_resources(plan)
    session = runtime.create_session(runtime.plan_session(RequestShape(2, 2), "prefix"))
    runtime.prefill(session, torch.tensor([1, 2]))
    assert runtime.candidate(session, torch.tensor([3, 4])).hidden.tolist() == [[6], [10]]
    assert plan.metadata["candidate_persistence"] == "committed"
    runtime.release_session(session)
    runtime.close()


def test_failed_execution_is_not_retried_and_release_is_explicit():
    backend = Backend()
    policy = CapacityPolicy.byte_budget(CacheFootprint(100, 0))
    runtime = TokenRuntime(backend.runtime_driver(policy))
    runtime.allocate_resources(runtime.plan_resources(policy, {}))
    session = runtime.create_session(runtime.plan_session(RequestShape(2, 2), "prefix"))
    runtime.prefill(session, torch.tensor([1, 2]))
    backend.fail = True
    with pytest.raises(RuntimeError, match="model failure"):
        runtime.candidate(session, torch.tensor([3, 4]))
    backend.fail = False
    with pytest.raises(RuntimeError, match="after failure"):
        runtime.candidate(session, torch.tensor([3, 4]))
    runtime.release_session(session)
    assert backend.released == 1
    with pytest.raises(ValueError, match="released"):
        runtime.release_session(session)
    runtime.close()
