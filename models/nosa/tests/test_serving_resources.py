"""Shared NOSA execution ownership, storage accounting and session isolation."""

from collections.abc import Mapping
from contextlib import contextmanager

import pytest
import torch

from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint
from models.nosa.execution.adapter import NosaServingBackend
from models.nosa.tests.test_model import tiny_config
from models.nosa.tests.test_sparse_model import initialized_sparse_model
from serving.persistent import PersistentGRRunner

SCHEMES = NosaServingBackend.schemes
BUDGET = CacheFootprint(1 << 30, 1 << 30)


def make_backend(scheme, *, chunk_size=32, context=192):
    model = initialized_sparse_model(
        tiny_config(max_position_embeddings=context, num_hidden_layers=4)
    )
    return NosaServingBackend(model, scheme, chunk_size=chunk_size)


def limits(capacity=160, candidate=81):
    return {"max_session_capacity": capacity, "max_candidate_tokens": candidate}


@contextmanager
def allocated_backend(scheme, *, capacity=160, candidate=81, chunk_size=32):
    backend = make_backend(scheme, chunk_size=chunk_size)
    plan = backend.plan_resources(BUDGET, limits(capacity, candidate))
    backend.allocate_shared(plan)
    try:
        yield backend, plan
    finally:
        # Also clean up sessions created before an assertion or injected failure.
        for session in tuple(backend.resources._sessions):
            backend.release_session(session)
        backend.close()


def storages(value, *, stop=()):
    """Enumerate actual tensor backing independently of production byte helpers."""
    found, visited = {}, {id(item) for item in stop}

    def visit(item):
        if id(item) in visited:
            return
        visited.add(id(item))
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            if storage.nbytes():
                found[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list, set)):
            for child in item:
                visit(child)
        elif hasattr(item, "__dict__") and not isinstance(item, torch.nn.Module):
            for name, child in vars(item).items():
                if name not in ("_sessions", "_active_session", "_admission_owner"):
                    visit(child)

    visit(value)
    return found


def tokens(length, user=0):
    return (torch.arange(length) * 7 + 3 + user * 11) % 43


@pytest.mark.parametrize("scheme", SCHEMES)
def test_planning_is_allocation_free_and_does_not_create_live_state(monkeypatch, scheme):
    backend = make_backend(scheme)

    def forbidden(*args, **kwargs):
        pytest.fail("pure resource planning must not allocate tensors or sessions")

    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "full"):
            patch.setattr(torch, name, forbidden)
        patch.setattr(backend, "create_session", forbidden)
        first = backend.plan_resources(BUDGET, limits())
        second = backend.plan_resources(BUDGET, limits())
    assert first == second
    assert backend.resources.plan is None
    assert backend.shared_bytes() == {"hbm": 0, "dram": 0}
    assert not backend.resources._sessions
    assert first.host_pages == 0
    assert first.metadata["max_session_capacity"] == 160
    assert first.metadata["max_candidate_tokens"] == 81
    assert first.metadata["max_query_tokens"] == 81
    assert first.metadata["stage_count"] == {"hbm": 0, "dense_prefetch": 2}.get(scheme, 1)
    assert first.metadata["pool_scope"] == "backend_workspace"
    assert first.metadata["host_scope"] == "session"
    assert first.metadata["trace_capacity"] == 0
    backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_allocator_configuration_change_rejects_before_session_allocation(monkeypatch, scheme):
    from models.nosa.execution import resources as serving_resources

    with allocated_backend(scheme) as (backend, plan):
        old = serving_resources.validate_allocator(backend.device)
        monkeypatch.setattr(
            serving_resources,
            "validate_allocator",
            lambda device, **kwargs: {**old, "configuration_key": "changed-effective-settings"},
        )

        def forbidden(*args, **kwargs):
            pytest.fail("Changed allocator reached cache allocation")

        monkeypatch.setattr(torch, "empty", forbidden)
        with pytest.raises(RuntimeError, match="allocator configuration changed"):
            backend.create_session(160)
        assert backend.resources.plan == plan
        assert not backend.resources._sessions


@pytest.mark.parametrize("scheme", SCHEMES)
def test_allocator_change_during_lease_poison_preserves_ownership(monkeypatch, scheme):
    from models.nosa.execution import resources as serving_resources

    backend = make_backend(scheme)
    backend.allocate_shared(backend.plan_resources(BUDGET, limits()))
    session = backend.create_session(160)
    original = serving_resources.validate_allocator
    old = original(backend.device)
    try:
        with (
            pytest.raises(RuntimeError, match="Unable to drain"),
            backend.resources.lease(session),
        ):
            monkeypatch.setattr(
                serving_resources,
                "validate_allocator",
                lambda device: {**old, "configuration_key": "changed-during-lease"},
            )
        assert backend.resources.poisoned
        assert backend.resources._active_session is session
        assert session in backend.resources._sessions
        with pytest.raises(RuntimeError, match="poisoned"):
            backend.release_session(session)
    finally:
        # Fault injection on CPU queued no asynchronous work. Restore solely
        # for fixture cleanup; production has no unpoison/recovery operation.
        monkeypatch.setattr(serving_resources, "validate_allocator", original)
        backend.resources.poisoned = False
        if backend.resources.staging_lease is not None:
            backend.resources.staging_lease = None
        backend.resources._active_session = None
        backend.release_session(session)
        backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_omitted_candidate_bound_reserves_full_capacity(scheme):
    backend = make_backend(scheme, chunk_size=32)
    plan = backend.plan_resources(BUDGET, {"max_session_capacity": 160})
    assert plan.metadata["max_candidate_tokens"] == 160
    assert plan.metadata["max_query_tokens"] == 160
    backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize(
    "requested",
    [
        limits(0, 1),
        limits(193, 1),
        limits(128, 0),
        limits(128, 129),
        limits(True, 1),
        limits(128, True),
        limits(128, 1.5),
    ],
)
def test_invalid_plan_limits_leave_existing_storage_unchanged(scheme, requested):
    with allocated_backend(scheme) as (backend, original):
        before = storages(backend.resources, stop=(backend, backend.model))
        with pytest.raises((TypeError, ValueError)):
            backend.plan_resources(BUDGET, requested)
        assert backend.resources.plan == original
        assert storages(backend.resources, stop=(backend, backend.model)) == before


@pytest.mark.parametrize("scheme", SCHEMES)
def test_shared_plan_budget_rejects_one_byte_short_before_allocation(scheme):
    backend = make_backend(scheme)
    plan = backend.plan_resources(BUDGET, limits())
    assert plan.shared.hbm > 0
    with pytest.raises(CacheBudgetExceeded):
        backend.plan_resources(CacheFootprint(plan.shared.hbm - 1, BUDGET.dram), limits())
    assert backend.shared_bytes() == {"hbm": 0, "dram": 0}
    backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_session_creation_requires_plan_and_enforces_capacity_before_allocation(
    monkeypatch, scheme
):
    backend = make_backend(scheme)
    with pytest.raises(RuntimeError):
        backend.create_session(160)
    plan = backend.plan_resources(BUDGET, limits())
    backend.allocate_shared(plan)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid session capacity reached tensor allocation")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", forbidden)
        for capacity in (0, 161, True, 1.5):
            with pytest.raises((TypeError, ValueError)):
                backend.create_session(capacity)
    assert not backend.resources._sessions
    backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_storage_is_shared_once_and_session_storage_is_disjoint(scheme):
    with allocated_backend(scheme) as (backend, plan):
        shared = storages(backend.resources, stop=(backend, backend.model))
        assert sum(shared.values()) == plan.shared.hbm
        assert backend.shared_bytes() == {"hbm": sum(shared.values()), "dram": 0}
        a, b = backend.create_session(160), backend.create_session(128)
        assert backend.estimate_session_host_pages(160) == backend.session_host_pages(a) == 0
        for session, prefix, user in ((a, 79, 0), (b, 63, 1)):
            backend.prefill(session, tokens(prefix, user))
            own = storages(session, stop=(backend, backend.model, backend.resources))
            assert set(own).isdisjoint(shared)
            assert sum(backend.session_bytes(session).values()) == sum(own.values())
        a_storage = storages(a, stop=(backend, backend.model, backend.resources))
        b_storage = storages(b, stop=(backend, backend.model, backend.resources))
        assert set(a_storage).isdisjoint(b_storage)
        assert storages(backend.resources, stop=(backend, backend.model)) == shared
        backend.release_session(a)
        assert not storages(a, stop=(backend, backend.model, backend.resources))
        assert storages(backend.resources, stop=(backend, backend.model)) == shared
        assert storages(b, stop=(backend, backend.model, backend.resources)) == b_storage


@pytest.mark.parametrize("scheme", SCHEMES)
def test_alternating_sessions_keep_independent_history_outputs_and_lengths(scheme):
    with allocated_backend(scheme) as (backend, _), allocated_backend("hbm") as (control, _):
        sessions = {0: backend.create_session(160), 1: backend.create_session(128)}
        controls = {0: control.create_session(160), 1: control.create_session(128)}
        for uid, prefix in ((0, 79), (1, 63)):
            backend.prefill(sessions[uid], tokens(prefix, uid))
            control.prefill(controls[uid], tokens(prefix, uid))
        # CPU resident reference recomputes compressed records without storing
        # them in IndexerCache; offload materializes them. Preserve each mode's
        # committed prefix state rather than changing that numerical policy.
        prefix_states = {
            uid: tuple(
                session.indexer_cache.layer_state(layer)
                for layer in range(backend.config.num_hidden_layers)
            )
            for uid, session in sessions.items()
        }
        fixed_shared = storages(backend.resources, stop=(backend, backend.model))
        retained_outputs = []
        for uid, prefix, query in ((0, 79, 33), (1, 63, 65), (0, 79, 81), (1, 63, 17)):
            candidate = tokens(query, uid + query)
            expected = control.extend(controls[uid], candidate)
            actual = backend.extend(sessions[uid], candidate)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            retained_outputs.append((actual, actual.clone()))
            backend.truncate(sessions[uid], prefix)
            control.truncate(controls[uid], prefix)
            assert sessions[uid].length == sessions[uid].indexer_cache.length == prefix
            for layer in range(backend.config.num_hidden_layers):
                assert sessions[uid].indexer_cache.layer_state(layer) == prefix_states[uid][layer]
            assert storages(backend.resources, stop=(backend, backend.model)) == fixed_shared
        for output, saved in retained_outputs:
            torch.testing.assert_close(output, saved, atol=0, rtol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_identical_plan_reuses_storage_but_live_sessions_prevent_plan_change(scheme):
    with allocated_backend(scheme) as (backend, plan):
        original = storages(backend.resources, stop=(backend, backend.model))
        generation = backend.resources.generation
        session = backend.create_session(128)
        backend.allocate_shared(plan)
        assert backend.resources.generation == generation
        assert storages(backend.resources, stop=(backend, backend.model)) == original
        changed = backend.plan_resources(BUDGET, limits(160, 82))
        with pytest.raises(RuntimeError):
            backend.allocate_shared(changed)
        assert backend.resources.plan == plan
        with pytest.raises(RuntimeError):
            backend.close()
        assert not session.released
        assert storages(backend.resources, stop=(backend, backend.model)) == original


@pytest.mark.parametrize("scheme", SCHEMES)
def test_changed_allocated_plan_requires_close_without_losing_old_storage(monkeypatch, scheme):
    with allocated_backend(scheme) as (backend, plan):
        original = storages(backend.resources, stop=(backend, backend.model))
        generation = backend.resources.generation
        changed = backend.plan_resources(BUDGET, limits(160, 82))

        def forbidden(*args, **kwargs):
            pytest.fail("replacement plan attempted allocation before explicit close")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", forbidden)
            with pytest.raises(RuntimeError):
                backend.allocate_shared(changed)
        assert backend.resources.plan == plan
        assert backend.resources.generation == generation
        assert storages(backend.resources, stop=(backend, backend.model)) == original
        backend.allocate_shared(plan)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_failed_initial_allocation_leaves_resources_reusable(monkeypatch, scheme):
    backend = make_backend(scheme)
    plan = backend.plan_resources(BUDGET, limits())
    original_empty = torch.empty
    calls = 0

    def fail_second_allocation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected allocation failure")
        return original_empty(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", fail_second_allocation)
        with pytest.raises(RuntimeError, match="injected allocation failure"):
            backend.allocate_shared(plan)
    assert calls == 2
    assert backend.resources.plan is None
    assert backend.shared_bytes() == {"hbm": 0, "dram": 0}
    assert not storages(backend.resources, stop=(backend, backend.model))
    backend.allocate_shared(plan)
    assert backend.shared_bytes()["hbm"] == plan.shared.hbm
    backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_close_drops_storage_and_replanning_invalidates_old_session(scheme):
    backend = make_backend(scheme)
    plan = backend.plan_resources(BUDGET, limits())
    backend.allocate_shared(plan)
    generation = backend.resources.generation
    old = backend.create_session(128)
    backend.release_session(old)
    backend.close()
    backend.close()
    assert backend.shared_bytes() == {"hbm": 0, "dram": 0}
    assert not storages(old, stop=(backend, backend.model, backend.resources))
    backend.allocate_shared(plan)
    assert backend.resources.generation > generation
    with pytest.raises((RuntimeError, ValueError)):
        backend.extend(old, tokens(1))
    backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_foreign_session_cannot_be_executed_counted_or_mutated(scheme):
    with allocated_backend(scheme) as (owner, _), allocated_backend(scheme) as (foreign, _):
        session = owner.create_session(160)
        owner.prefill(session, tokens(79))
        actions = (
            lambda: foreign.prefill(session, tokens(1)),
            lambda: foreign.extend(session, tokens(1)),
            lambda: foreign.truncate(session, 0),
            lambda: foreign.release_session(session),
            lambda: foreign.session_bytes(session),
        )
        for action in actions:
            with pytest.raises((RuntimeError, ValueError)):
                action()
            assert session.length == session.indexer_cache.length == 79
            assert not session.released


@pytest.mark.parametrize("scheme", SCHEMES)
def test_borrowed_cache_requires_exclusive_lease_and_recovers_after_exception(scheme):
    with allocated_backend(scheme) as (backend, _):
        a, b = backend.create_session(160), backend.create_session(128)
        with pytest.raises(RuntimeError):
            a.begin_step(1)
        assert a.length == 0 and a._pending_end is None
        with pytest.raises(RuntimeError, match="injected"), backend.execution_lease(a):
            for other in (a, b):
                with pytest.raises(RuntimeError), backend.execution_lease(other):
                    pytest.fail("nested/concurrent leases must be rejected")
            with pytest.raises(RuntimeError):
                backend.release_session(b)
            with pytest.raises(RuntimeError):
                backend.close()
            raise RuntimeError("injected execution failure")
        backend.prefill(b, tokens(63, 1))
        assert b.length == 63
        with pytest.raises(RuntimeError):
            backend.model(tokens(1), a, return_hidden=True)
        assert a._pending_end is None


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("candidate_limit", (17, 65))
def test_query_bounds_fail_before_begin_step_and_preserve_prefix(
    monkeypatch, scheme, candidate_limit
):
    with allocated_backend(scheme, candidate=candidate_limit) as (backend, _):
        session = backend.create_session(160)
        backend.prefill(session, tokens(79))
        expected = backend.extend(session, tokens(candidate_limit, 3))
        assert expected.shape[0] == candidate_limit  # A may exceed the prefill chunk.
        backend.truncate(session, 79)

        def forbidden(*args, **kwargs):
            pytest.fail("invalid query reached begin_step")

        with monkeypatch.context() as patch:
            patch.setattr(session, "begin_step", forbidden)
            for ids in (tokens(candidate_limit + 1), tokens(0), tokens(1).reshape(1, 1)):
                with pytest.raises((RuntimeError, ValueError)):
                    backend.extend(session, ids)
        assert session.length == session.indexer_cache.length == 79
        assert session._pending_end is None


@pytest.mark.parametrize("scheme", SCHEMES)
def test_prefill_and_direct_step_limits_reject_before_writing(monkeypatch, scheme):
    with allocated_backend(scheme, capacity=96, candidate=17) as (backend, plan):
        session = backend.create_session(96)

        def forbidden(*args, **kwargs):
            pytest.fail("an over-capacity prefill reached begin_step")

        with monkeypatch.context() as patch:
            patch.setattr(session, "begin_step", forbidden)
            with pytest.raises(ValueError):
                backend.prefill(session, tokens(97))
        with backend.execution_lease(session), pytest.raises(ValueError):
            session.begin_step(plan.metadata["max_query_tokens"] + 1)
        assert session.length == session.indexer_cache.length == 0
        assert session._pending_end is None


@pytest.mark.parametrize("scheme", SCHEMES)
def test_actual_pending_storage_fits_reservation_at_each_layer(scheme):
    with allocated_backend(scheme) as (backend, _):
        session = backend.create_session(160)
        reservation = backend.estimate_session_bytes(160, 79)
        backend.prefill(session, tokens(79))
        baseline = sum(backend.session_bytes(session).values())
        observations = []
        shared = storages(backend.resources, stop=(backend, backend.model))

        def observe(*args):
            actual = backend.session_bytes(session)
            owned = storages(session, stop=(backend, backend.model, backend.resources))
            assert sum(actual.values()) == sum(owned.values())
            assert set(owned).isdisjoint(shared)
            assert all(actual[tier] <= reservation[tier] for tier in ("hbm", "dram"))
            observations.append(sum(actual.values()))

        hooks = [layer.register_forward_hook(observe) for layer in backend.model.model.layers]
        try:
            backend.extend(session, tokens(81, 2))
        finally:
            for hook in hooks:
                hook.remove()
        assert len(observations) == backend.config.num_hidden_layers
        if scheme != "hbm":
            assert max(observations) > baseline
        assert storages(backend.resources, stop=(backend, backend.model)) == shared


@pytest.mark.parametrize("scheme", SCHEMES)
def test_admission_owner_is_distinct_from_execution_lease(scheme):
    with allocated_backend(scheme) as (backend, plan):
        owner, foreign = object(), object()
        shared = storages(backend.resources, stop=(backend, backend.model))
        backend.bind_owner(owner)
        try:
            with pytest.raises(RuntimeError):
                backend.bind_owner(foreign)
            with pytest.raises(ValueError):
                backend.unbind_owner(foreign)
            with pytest.raises(RuntimeError):
                backend.close()
            with pytest.raises(RuntimeError, match="admission owner"):
                backend.create_session(160)
            session = backend.create_session(160, owner=owner)
            try:
                with pytest.raises(RuntimeError):
                    backend.unbind_owner(owner)
                with pytest.raises(RuntimeError, match="admission owner"):
                    backend.prefill(session, tokens(79))
                backend.prefill(session, tokens(79), owner=owner)
                for mutation in (session.reset, session.release, lambda: session.truncate(0)):
                    with pytest.raises(RuntimeError, match="admission owner"):
                        mutation()
            finally:
                backend.release_session(session, owner=owner)
        finally:
            backend.unbind_owner(owner)
        assert backend.resources.plan == plan
        assert storages(backend.resources, stop=(backend, backend.model)) == shared
        backend.bind_owner(foreign)
        backend.unbind_owner(foreign)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_admission_cannot_take_over_direct_session(scheme):
    with allocated_backend(scheme) as (backend, _):
        backend.create_session(160)
        with pytest.raises(RuntimeError):
            backend.bind_owner(object())


@pytest.mark.parametrize("scheme", SCHEMES)
def test_model_failure_restores_attention_and_releases_shared_lease(monkeypatch, scheme):
    with allocated_backend(scheme) as (backend, _):
        a, b = backend.create_session(160), backend.create_session(128)
        backend.prefill(a, tokens(79))
        backend.prefill(b, tokens(63, 1))
        expected = backend.extend(a, tokens(33, 4)).clone()
        backend.truncate(a, 79)
        original_attention = backend.model.main_attention

        def fail(*args, **kwargs):
            raise RuntimeError("injected middle-layer failure")

        with monkeypatch.context() as patch:
            patch.setattr(backend.model.model.layers[2], "forward", fail)
            with pytest.raises(RuntimeError, match="injected"):
                backend.extend(a, tokens(33, 4))
        assert backend.model.main_attention is original_attention
        assert a.length == a.indexer_cache.length == 79
        backend.extend(b, tokens(17, 7))
        backend.truncate(b, 63)
        recovered = backend.extend(a, tokens(33, 4))
        torch.testing.assert_close(recovered, expected, atol=0, rtol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("capacity_in_users", (1, 2))
def test_exact_shared_plus_session_budget_controls_user_admission(scheme, capacity_in_users):
    backend = make_backend(scheme)
    bound = limits(96, 17)
    plan = backend.plan_resources(BUDGET, bound)
    backend.allocate_shared(plan)
    session_cost = backend.estimate_session_bytes(96, 79)
    caps = {
        tier: getattr(plan.shared, tier) + capacity_in_users * session_cost[tier]
        for tier in ("hbm", "dram")
    }
    requests = [
        {"user_id": uid, "input_ids": tokens(96, uid).tolist(), "stable_prefix_tokens": 79}
        for uid in (0, 1, 0)
    ]
    try:
        with PersistentGRRunner(
            backend,
            hbm_budget_bytes=caps["hbm"],
            dram_budget_bytes=caps["dram"],
            resource_limits=bound,
        ) as runner:
            results = list(runner.run(requests))
            assert len(runner.pool) == capacity_in_users
            assert results[2].metrics["prefix_cache_hit"] is (capacity_in_users == 2)
            assert results[2].metrics["is_revisit"]
            for result in results:
                for tier in ("hbm", "dram"):
                    assert result.metrics[f"shared_reserved_{tier}_bytes"] == getattr(
                        plan.shared, tier
                    )
                    assert result.metrics[f"reserved_{tier}_bytes"] <= caps[tier]
            torch.testing.assert_close(results[0].hidden, results[2].hidden, atol=0, rtol=0)
        assert backend.shared_bytes()["hbm"] == plan.shared.hbm
    finally:
        backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_foreign_resource_attach_preserves_both_backends_and_original_session(scheme):
    with allocated_backend(scheme) as (owner, _), allocated_backend(scheme) as (other, _):
        session = owner.create_session(160)
        owner.prefill(session, tokens(79))
        before = (
            session._serving_identity,
            session._execution_generation,
            session._execution_ready,
            owner.session_bytes(session),
            owner.shared_bytes(),
            other.shared_bytes(),
        )
        with pytest.raises(ValueError, match="[Ff]oreign"), other.execution_lease(session):
            pytest.fail("Foreign execution must be rejected")
        with pytest.raises(ValueError, match="foreign"):
            other.resources.attach(session)
        assert session in owner.resources._sessions
        assert not other.resources._sessions
        assert (
            session._serving_identity,
            session._execution_generation,
            session._execution_ready,
            owner.session_bytes(session),
            owner.shared_bytes(),
            other.shared_bytes(),
        ) == before
        assert session._execution_resources is owner.resources
        output = owner.extend(session, tokens(17, 3))
        assert output.shape[0] == 17
        owner.truncate(session, 79)
        owner.release_session(session)


@pytest.mark.parametrize(
    "scheme,short_tier",
    [(scheme, "hbm") for scheme in SCHEMES]
    + [(scheme, "dram") for scheme in SCHEMES if scheme != "hbm"],
)
def test_one_byte_below_minimum_user_capacity_cannot_allocate_or_evict(scheme, short_tier):
    backend = make_backend(scheme)
    bound = limits(96, 17)
    plan = backend.plan_resources(BUDGET, bound)
    backend.allocate_shared(plan)
    cost = backend.estimate_session_bytes(96, 79)
    caps = {tier: getattr(plan.shared, tier) + cost[tier] for tier in ("hbm", "dram")}
    caps[short_tier] -= 1
    request = {"user_id": 0, "input_ids": tokens(96).tolist(), "stable_prefix_tokens": 79}
    try:
        with PersistentGRRunner(
            backend,
            hbm_budget_bytes=caps["hbm"],
            dram_budget_bytes=caps["dram"],
            resource_limits=bound,
        ) as runner:
            with pytest.raises(CacheBudgetExceeded):
                runner.execute(request)
            assert len(runner.pool) == 0
            assert not runner.visits
            assert not backend.resources._sessions
    finally:
        backend.close()
