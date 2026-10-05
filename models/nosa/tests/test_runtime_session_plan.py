"""Native NOSA admission plans are pure and supply the allocated dimensions."""

from dataclasses import replace

import pytest
import torch

from cache.capacity import CacheFootprint, CapacityPolicy, allocation_footprint
from executor.contracts import RequestShape
from models.nosa.execution.adapter import NosaServingBackend
from models.nosa.execution.fixed import NosaFixedServingBackend
from models.nosa.tests.test_model import tiny_config
from models.nosa.tests.test_sparse_model import initialized_sparse_model


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("scheme", NosaServingBackend.schemes)
def test_native_session_plan_is_consumed_without_replanning(monkeypatch, fixed, scheme):
    model = initialized_sparse_model(tiny_config(max_position_embeddings=160, num_hidden_layers=2))
    options = {"chunk_size": 64}
    if fixed:
        backend = NosaFixedServingBackend(
            model, scheme, sparse_pool_tokens=128, host_arena_tokens=256, **options
        )
        policy = CapacityPolicy.fixed_pools()
    else:
        backend = NosaServingBackend(model, scheme, **options)
        policy = CapacityPolicy.byte_budget(CacheFootprint(1 << 30, 1 << 30))
    driver = backend.runtime_driver(policy)
    owner = object()
    driver.bind_owner(owner)
    resources = driver.plan_resources(
        policy,
        {"max_session_capacity": 144, "max_history_tokens": 128, "max_candidate_tokens": 16},
    )
    driver.allocate_resources(resources)
    session = None
    try:

        def forbidden(*args, **kwargs):
            pytest.fail("native session planning allocated or fell back to a scalar estimate")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", forbidden)
            patch.setattr(backend, "estimate_session_bytes", forbidden)
            plan = driver.plan_session(RequestShape(128, 8), ("user", "history"))
            from models.nosa.execution import session_budget

            patch.setattr(session_budget, "session_allocation_layout", forbidden)
            repeated = driver.plan_session(RequestShape(128, 8), ("other", "history"))
            with pytest.raises(ValueError, match="session capacity"):
                driver.plan_session(RequestShape(128, 17), ("other", "history"))
            with pytest.raises(ValueError, match="candidate"):
                driver.plan_session(RequestShape(64, 17), ("other", "history"))
            with pytest.raises(TypeError, match="unhashable"):
                driver.plan_session(RequestShape(128, 8), ["mutable"])
        assert repeated is not plan and repeated.model_plan is not plan.model_plan
        assert repeated.resource_identity is resources
        assert repeated.history_identity == ("other", "history")
        assert repeated.allocations is plan.allocations
        assert repeated.reservation is plan.reservation
        assert plan.resource_identity is resources
        assert plan.history_identity == ("user", "history")
        assert plan.retained_capacity == (128 if fixed else 136)
        assert plan.allocations
        assert allocation_footprint(plan.allocations) == plan.reservation
        assert {item.owner for item in plan.allocations} == {"session"}
        changed = (
            replace(plan, retained_capacity=plan.retained_capacity + 1),
            replace(plan, history_tokens=plan.history_tokens - 1),
            replace(plan, allocations=plan.allocations[:-1]),
            replace(plan, reservation=plan.reservation + CacheFootprint(1, 0)),
            replace(plan, host_pages=plan.host_pages + 1),
            replace(plan, hbm_tokens=plan.hbm_tokens + 1),
        )
        with monkeypatch.context() as patch:
            patch.setattr(backend, "_allocate_session", forbidden)
            for invalid in changed:
                with pytest.raises(ValueError, match="declaration"):
                    driver.create_session(invalid)
        with monkeypatch.context() as patch:
            patch.setattr(backend, "_session_geometry", forbidden)
            patch.setattr(backend, "estimate_session_bytes", forbidden)
            session = driver.create_session(plan)
        assert session._allocation_plan is plan
        assert session.max_seq_len == (144 if fixed else 136)
        declared = {item.name: item for item in plan.allocations}
        cis = session.cis_scores if scheme != "hbm" else session.buffers["cis_scores"]
        assert tuple(cis.shape) == declared["cis_scores"].shape
        for name in ("keys", "values"):
            key = name if scheme == "hbm" else f"host.{name}"
            assert tuple(session.buffers[name].shape) == declared[key].shape
        if fixed and scheme != "hbm":
            assert session.buffers["keys"].shape[1] == 128
            assert declared["host.keys"].charged_bytes > declared["host.keys"].storage_bytes
        if fixed:
            assert (plan.host_pages, plan.hbm_tokens) == ((0, 128) if scheme == "hbm" else (2, 0))
        else:
            assert plan.host_pages == plan.hbm_tokens == 0
    finally:
        if session is not None:
            driver.release_session(session)
        driver.unbind_owner(owner)
        backend.close()


def planning_only_backend():
    model = initialized_sparse_model(tiny_config(max_position_embeddings=320, num_hidden_layers=1))
    backend = NosaServingBackend(model, "overlap", chunk_size=64)
    policy = CapacityPolicy.byte_budget(CacheFootprint(1 << 30, 1 << 30))
    driver = backend.runtime_driver(policy)
    resources = driver.plan_resources(
        policy,
        {"max_session_capacity": 208, "max_history_tokens": 128, "max_candidate_tokens": 80},
    )
    return backend, driver, resources


def test_layout_cache_has_one_entry_and_keeps_resource_generation_identity():
    backend, driver, resources = planning_only_backend()
    shape = RequestShape(128, 8)
    first = driver.plan_session(shape, "first")
    changed_queries = driver.plan_session(RequestShape(64, 72), "second")
    assert changed_queries.retained_capacity == first.retained_capacity
    assert changed_queries.model_plan.queries != first.model_plan.queries
    assert changed_queries.allocations != first.allocations
    restored = driver.plan_session(shape, "third")
    assert restored.allocations == first.allocations
    assert restored.allocations is not first.allocations
    assert backend._session_layout_cache[1] is restored.allocations

    other_driver = backend.runtime_driver(resources.policy)
    other_resources = other_driver.plan_resources(
        resources.policy,
        {"max_session_capacity": 208, "max_history_tokens": 128, "max_candidate_tokens": 80},
    )
    other = other_driver.plan_session(shape, "fourth")
    assert other.allocations is restored.allocations
    assert other.resource_identity is other_resources and other_resources is not resources
    with pytest.raises(ValueError, match="different resources"):
        other_driver.create_session(restored)

    backend.resources.generation += 1
    advanced = other_driver.plan_session(shape, "fifth")
    assert advanced.allocations == other.allocations
    assert advanced.allocations is not other.allocations
    with pytest.raises(ValueError, match="foreign or stale"):
        other_driver.create_session(other)
    backend.resources.lifecycle.identity = object()
    replaced_provider = other_driver.plan_session(shape, "sixth")
    assert replaced_provider.allocations == advanced.allocations
    assert replaced_provider.allocations is not advanced.allocations
    with pytest.raises(ValueError, match="foreign or stale"):
        other_driver.create_session(advanced)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capacity", 144),
        ("queries", 72),
        ("num_hidden_layers", 2),
        ("num_key_value_heads", 1),
        ("num_attention_heads", 8),
        ("head_dim", 16),
        ("dtype", torch.bfloat16),
        ("device", torch.device("cpu:0")),
        ("scheme", "hbm"),
        ("host_capacity", 128),
        ("dense_counter", True),
    ],
)
def test_layout_cache_keys_every_layout_input(monkeypatch, field, value):
    from models.nosa.execution import session_budget

    backend, driver, _ = planning_only_backend()
    shape = RequestShape(128, 8)
    first = driver.plan_session(shape, "first")
    if field == "capacity":
        shape = RequestShape(128, value - 128)
    elif field == "queries":
        shape = RequestShape(136 - value, value)
    elif field in ("dtype", "device", "scheme"):
        monkeypatch.setattr(backend, field, value)
    elif field in ("host_capacity", "dense_counter"):
        original = backend._session_layout_options
        monkeypatch.setattr(
            backend,
            "_session_layout_options",
            lambda geometry: {**original(geometry), field: value},
        )
    else:
        monkeypatch.setattr(backend, "config", replace(backend.config, **{field: value}))
    declared = []
    original_layout = session_budget.session_allocation_layout

    def observe(**options):
        result = original_layout(**options)
        declared.append(result)
        return result

    monkeypatch.setattr(session_budget, "session_allocation_layout", observe)
    changed = driver.plan_session(shape, "second")
    assert len(declared) == 1
    assert changed.allocations is declared[0][0]
    assert changed.allocations is not first.allocations
    assert changed.reservation == CacheFootprint.from_mapping(declared[0][1])
    assert changed.reservation == allocation_footprint(changed.allocations)


def test_failed_layout_does_not_replace_valid_entry_or_bypass_typed_validation(monkeypatch):
    backend, driver, _ = planning_only_backend()
    shape = RequestShape(128, 8)
    first = driver.plan_session(shape, "first")
    with monkeypatch.context() as patch:
        # NosaConfig accepts bool as int; the allocation planner intentionally does not.
        patch.setattr(backend, "config", replace(backend.config, num_hidden_layers=True))
        with pytest.raises(ValueError, match="layers must be a positive integer"):
            driver.plan_session(shape, "invalid")
    restored = driver.plan_session(shape, "second")
    assert restored.allocations is first.allocations
    assert restored.history_identity == "second"
