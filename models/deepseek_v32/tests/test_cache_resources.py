"""CPU-only checks for shared-cache admission and allocation failure boundaries."""

from types import SimpleNamespace

import pytest
import torch

from cache.lifecycle import ResourceLifecycle
from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint
from cache.sparse_token_pool import SharedSparseTokenPool
from models.deepseek_v32.attention import EchoAttentionRunner
from models.deepseek_v32.execution.adapter import DeepSeekServingBackend
from models.deepseek_v32.execution.cache_resources import (
    check_dense_staging_allocation,
    dense_staging_allocation_bytes,
    execution_reservation,
)
from models.deepseek_v32.execution.pipeline import LocalPipeline


def planned_backend(*, host_tokens=256):
    backend = object.__new__(DeepSeekServingBackend)
    backend.lifecycle = ResourceLifecycle("DeepSeek test")
    # Resource estimation must not initialize CUDA or construct any device pools.
    backend.device = torch.device("cuda:0")
    backend.pipeline = LocalPipeline()
    backend._pipeline_resources = None
    backend.chunk_size = 8
    backend.scheme = "echo"
    backend.max_seq_len = 1024
    backend.num_layers = 3
    backend.slots = 32
    backend.workspace_query_tokens = 8
    backend.extend_chunk_size = None
    backend.host_arena_tokens = host_tokens
    backend.cfg = SimpleNamespace(
        kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128, index_topk=16
    )
    backend._shared_pool = None
    backend._dense_staging = None
    backend._dense_lease = None
    backend._dense_sources = []
    backend._allocation_failure = None
    backend.synchronize = lambda: None
    return backend


def test_explicit_arena_plan_checks_shared_and_private_bytes_before_allocation(monkeypatch):
    backend = planned_backend()

    def forbidden(*args, **kwargs):
        raise AssertionError("resource planning allocated a shared pool")

    monkeypatch.setattr(SharedSparseTokenPool, "__init__", forbidden)
    limits = {"max_session_capacity": 128}
    plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), limits)
    session = CacheFootprint.from_mapping(backend.estimate_session_bytes(128))
    exact = CacheFootprint(
        plan.shared.hbm + session.hbm,
        plan.shared.dram + max(session.dram, plan.host_pages * 4),
    )
    assert backend.plan_resources(exact, limits) == plan
    assert plan.host_pages == 4
    assert session.hbm > 0 and session.dram > 0
    # Each independent hard limit must reject even when the other has room.
    for budget in (
        CacheFootprint(exact.hbm - 1, exact.dram),
        CacheFootprint(exact.hbm, exact.dram - 1),
    ):
        with pytest.raises(CacheBudgetExceeded):
            backend.plan_resources(budget, limits)
    assert backend._shared_pool is None
    assert backend.lifecycle.plan is None
    assert not backend.lifecycle.sessions


@pytest.mark.parametrize("scheme", ["echo", "serial_sparse"])
def test_fixed_pools_plan_preserves_explicit_capacities_without_byte_caps(monkeypatch, scheme):
    backend = planned_backend()
    backend.scheme = scheme

    def forbidden(*args, **kwargs):
        raise AssertionError("resource planning allocated a shared pool")

    monkeypatch.setattr(SharedSparseTokenPool, "__init__", forbidden)
    limits = {"max_session_capacity": 128, "max_candidate_tokens": 8}
    budget_plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), limits)
    fixed_plan = backend.plan_resources(None, limits)
    assert fixed_plan.shared == budget_plan.shared
    assert fixed_plan.host_pages == 4
    assert fixed_plan.metadata["resource_mode"] == "fixed_pools"
    assert fixed_plan.metadata["host_arena_tokens"] == 256
    assert fixed_plan.metadata["sparse_pool_tokens"] == 32
    with pytest.raises(CacheBudgetExceeded):
        backend.plan_resources(CacheFootprint(), limits)
    backend.workspace_query_tokens = 16
    wider_plan = backend.plan_resources(None, limits)
    assert wider_plan.shared.hbm > fixed_plan.shared.hbm
    assert wider_plan.host_pages == fixed_plan.host_pages
    assert backend.lifecycle.plan is backend._shared_pool is None


@pytest.mark.parametrize("scheme", ["hbm", "dense_prefetch"])
def test_fixed_hbm_and_dense_require_one_history_and_keep_transient_candidates(scheme):
    backend = planned_backend()
    backend.scheme = scheme
    with pytest.raises(CacheBudgetExceeded, match="one complete history"):
        backend.plan_resources(None, {"max_session_capacity": 128})
    backend.slots = 128
    plan = backend.plan_resources(None, {"max_session_capacity": 128})
    assert plan.metadata["resource_mode"] == "fixed_pools"
    assert plan.metadata["candidate_persistence"] == "gpu_transient"
    assert plan.metadata["candidate_slots"] == backend.workspace_query_tokens
    if scheme == "hbm":
        assert plan.hbm_tokens == 128
        assert plan.host_pages == 0
    else:
        assert plan.host_pages == 4
        assert plan.metadata["shared_token_pool"]
        assert plan.metadata["dense_fetch_policy"] == "all_history_cache_misses_next_layer"
        assert plan.metadata.get("dense_staging_bytes", 0) == 0
    assert backend.lifecycle.plan is backend._shared_pool is None


@pytest.mark.parametrize(
    "attribute,value,match",
    [
        ("host_arena_tokens", None, "explicit host_arena_tokens"),
        ("host_arena_tokens", 64, "maximum-size session"),
        ("host_arena_tokens", 129, "multiple of 64"),
        ("slots", 8, "full exact selection"),
        ("workspace_query_tokens", 64, "per-layer device pool"),
    ],
)
def test_fixed_pools_still_reject_impossible_shapes_without_allocation(attribute, value, match):
    backend = planned_backend()
    setattr(backend, attribute, value)
    with pytest.raises(ValueError, match=match):
        backend.plan_resources(None, {"max_session_capacity": 128})
    assert backend.lifecycle.plan is backend._shared_pool is None


def test_auto_arena_preserves_private_cpu_page_metadata_in_dram_budget():
    backend = planned_backend(host_tokens=None)
    # Three pages use a 256-KiB bin per layer. Its exact byte budget leaves no
    # room for all CPU metadata, so only two pages can be admitted in that bin.
    budget = CacheFootprint(1 << 28, 3 * 262144 + 3 * 4 + 2 * 4 + 40)
    plan = backend.plan_resources(budget, {"max_session_capacity": 128})
    session = CacheFootprint.from_mapping(backend.estimate_session_bytes(128))
    assert plan.host_pages == 2
    assert (plan.shared + session).fits(budget)
    assert plan.shared.dram > plan.host_pages * 64 * 576 * 2 * backend.num_layers


def test_auto_arena_leaves_page_table_headroom_for_more_than_one_user():
    backend = planned_backend(host_tokens=None)
    # Three pages fit exactly including all owners' CPU page tables. The fourth
    # page would cross the per-layer pinned bin from 256 KiB to 512 KiB.
    budget = CacheFootprint(1 << 28, 3 * 262144 + 2 * 3 * 4 + 40)
    plan = backend.plan_resources(budget, {"max_session_capacity": 128})
    assert plan.host_pages == 3
    assert plan.shared.dram + plan.host_pages * 4 <= budget.dram
    small_session = CacheFootprint.from_mapping(backend.estimate_session_bytes(64))
    retained = plan.shared
    for _ in range(plan.host_pages):
        retained += small_session
    assert retained.fits(budget)


def test_auto_arena_search_respects_the_discontinuous_pinned_bin_boundary():
    backend = planned_backend(host_tokens=None)
    limits = {"max_session_capacity": 128}
    # NH=192 -> 262144 B/layer, NH=256 -> 524288 B/layer. Even a
    # near-double budget cannot admit the fourth page until its whole bin fits.
    below = backend.plan_resources(CacheFootprint(1 << 28, 3 * 524288 - 1), limits)
    above = backend.plan_resources(CacheFootprint(1 << 28, 3 * 524288 + 8 * 4 + 40), limits)
    assert below.host_pages == 3
    assert above.host_pages == 4
    assert above.shared.dram == 3 * 524288 + 4 * 4 + 40


def test_controlled_arena_charges_ten_independent_two_gib_pinned_allocations():
    backend = planned_backend(host_tokens=1050624)
    backend.num_layers = 10
    limits = {"max_session_capacity": 128}
    plan = backend.plan_resources(CacheFootprint(1 << 30, 1 << 36), limits)
    assert plan.shared.dram == 10 * (1 << 31) + 1050624 // 64 * 4 + 40
    logical_dram = 10 * 1050624 * 1152 + 2 * plan.host_pages * 4
    with pytest.raises(CacheBudgetExceeded, match="configured host arena"):
        backend.plan_resources(CacheFootprint(1 << 30, logical_dram), limits)


def test_dense_session_pinned_bin_is_reserved_before_admission():
    from cache.prefix_pool import PrefixSessionPool

    backend = planned_backend()
    backend.scheme = "dense_prefetch"
    reservation = CacheFootprint.from_mapping(backend.estimate_session_bytes(128))
    assert reservation.dram == backend.num_layers * 262144

    def forbidden(*args, **kwargs):
        raise AssertionError("impossible session reached allocation or eviction")

    pool = PrefixSessionPool(
        CacheFootprint(reservation.hbm, reservation.dram - 1),
        allocate=forbidden,
        release=forbidden,
        measure=forbidden,
    )
    with pytest.raises(CacheBudgetExceeded):
        pool.acquire("user", "prefix", 128, reservation)
    assert len(pool) == 0 and pool.evictions == 0


@pytest.mark.parametrize("scheme", ["hbm", "echo", "serial_sparse", "dense_prefetch"])
def test_all_schemes_reserve_observed_cpu_execution_scratch(scheme):
    backend = planned_backend()
    backend.scheme = scheme
    plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})
    assert {
        key: value for key, value in plan.metadata.items() if key.startswith("workspace_cpu")
    } == {
        "workspace_cpu_indexer_bytes": 8,
        "workspace_cpu_scalar_bytes": 8,
        "workspace_cpu_metrics_bytes": 24,
        "workspace_cpu_bytes": 40,
    }
    if scheme in ("echo", "serial_sparse"):
        assert plan.shared.dram == 3 * 524288 + 4 * 4 + 40
    else:
        assert plan.shared.dram == 40


def test_standalone_resident_cpu_scratch_has_a_real_dram_budget():
    from models.deepseek_v32.model import DeepSeekEchoModel

    model = object.__new__(DeepSeekEchoModel)
    model.cfg = planned_backend().cfg
    model.capacity = 128
    model.devices = [torch.device("cuda:0")]
    model.placement = model.devices * 3
    model.hbm_cache_budget_bytes = 1 << 28
    model.dram_cache_budget_bytes = 40
    model.execution_reservation = execution_reservation(8, 128, topk=16, width=576)
    plan = model._plan_cache_resources(False)
    assert plan["dram_bytes"] == plan["devices"]["cuda:0"]["dram"] == 40
    assert plan["devices"]["cuda:0"]["workspace_cpu_bytes"] == 40
    model.dram_cache_budget_bytes = 39
    with pytest.raises(CacheBudgetExceeded, match="DRAM bytes"):
        model._plan_cache_resources(False)


def test_workspace_is_reserved_for_fixed_maximum_chunk_across_scan():
    backend = planned_backend()
    backend.workspace_query_tokens = 32
    plans = []
    for chunk in (4, 8, 16, 32):
        backend.chunk_size = chunk
        plans.append(
            backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})
        )
    assert all(plan == plans[0] for plan in plans)
    backend.workspace_query_tokens = backend.slots + 1
    with pytest.raises(CacheBudgetExceeded, match="query workspace"):
        backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})


def test_source_window_is_drained_before_allocating_new_projection():
    query, width, inflight = 8, 576, 2
    reservation = execution_reservation(
        query, 128, topk=16, width=width, max_inflight_writes=inflight
    )
    pool = SharedSparseTokenPool(128, width, 1, 32, device="cpu", max_inflight_writes=inflight)
    session = pool.allocate_session(128)
    cache = session.layer(0)
    waited = []

    def synchronize():
        waited.append(True)

    for _ in range(inflight):
        pool._writes.append(
            SimpleNamespace(
                event=SimpleNamespace(query=lambda: False, synchronize=synchronize),
                source=torch.empty((query, width), dtype=torch.bfloat16),
            )
        )

    class ProjectionReached(Exception):
        pass

    def project(*args, **kwargs):
        assert waited == [True]
        current_source = torch.empty((query, width), dtype=torch.bfloat16)
        live_sources = [current_source, *(ticket.source for ticket in pool._writes)]
        actual = sum(source.untyped_storage().nbytes() for source in live_sources)
        assert actual <= reservation.copy_source_bytes
        raise ProjectionReached

    runner = object.__new__(EchoAttentionRunner)
    runner.cache = cache
    runner.attention = SimpleNamespace(project=project)
    with pytest.raises(ProjectionReached):
        runner.forward(torch.zeros(query, 1))
    pool.close()


def test_session_construction_failure_returns_reserved_pages(monkeypatch):

    backend = planned_backend()
    backend.lifecycle.plan = SimpleNamespace(metadata={"max_session_capacity": 128})
    released = []
    sparse_session = SimpleNamespace(
        layer=lambda index: index, release=lambda: released.append(True)
    )
    backend._shared_pool = SimpleNamespace(allocate_session=lambda capacity: sparse_session)
    backend.attentions = [object(), object(), object()]
    backend.chunk_size = 8
    created = []

    def failing_runner(*args, **kwargs):
        if created:
            raise RuntimeError("injected indexer allocation failure")
        created.append(object())
        return created[-1]

    monkeypatch.setattr(backend.pipeline, "layer_cache", lambda session, layer, resources: layer)
    monkeypatch.setattr(backend.pipeline, "create_runner", failing_runner)
    with pytest.raises(RuntimeError, match="indexer allocation failure"):
        backend.create_session(128)
    assert released == [True]
    assert not backend.lifecycle.sessions


@pytest.mark.parametrize("scheme", ["hbm", "echo", "serial_sparse", "dense_prefetch"])
def test_admission_owner_reuse_and_constructor_rollback_keep_the_right_plan(scheme):
    backend = planned_backend()
    backend.device = torch.device("cpu")
    backend.scheme = scheme
    plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})
    first, second = object(), object()
    with pytest.raises(ValueError, match="non-None"):
        backend.bind_owner(None)
    backend.bind_owner(first)
    with pytest.raises(RuntimeError, match="admission owner"):
        backend.bind_owner(second)
    with pytest.raises(ValueError, match="[Ff]oreign"):
        backend.unbind_owner(second)
    backend.allocate_shared(plan, owner=first)
    resource = backend._shared_pool or backend._dense_staging
    with pytest.raises(RuntimeError, match="admission owner"):
        backend.close()
    backend.unbind_owner(first)
    assert backend.lifecycle.plan is plan
    backend.bind_owner(second)
    backend.allocate_shared(plan, owner=second)
    backend.unbind_owner(second, rollback=True)
    assert backend.lifecycle.plan is plan
    assert (backend._shared_pool or backend._dense_staging) is resource
    changed = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 64})
    with pytest.raises(RuntimeError, match="prior shared plan"):
        backend.allocate_shared(changed)
    backend.close()
    backend.close()
    backend.bind_owner(first)
    backend.allocate_shared(changed, owner=first)
    backend.unbind_owner(first, rollback=True)
    assert backend.lifecycle.plan is backend._shared_pool is backend._dense_staging is None
    assert backend.lifecycle.admission_owner is None


def test_owner_refuses_existing_direct_sessions_and_keeps_binding_until_release():
    backend = planned_backend()
    owner = object()
    backend.lifecycle.register(
        object(), owner=backend.lifecycle.admission_owner, allow_unplanned=True
    )
    with pytest.raises(RuntimeError, match="direct sessions"):
        backend.bind_owner(owner)
    backend.lifecycle.sessions.clear()
    backend.bind_owner(owner)
    backend.lifecycle.register(
        object(), owner=backend.lifecycle.admission_owner, allow_unplanned=True
    )
    with pytest.raises(RuntimeError, match="[Rr]elease sessions"):
        backend.unbind_owner(owner, rollback=True)
    assert backend.lifecycle.admission_owner is owner


def test_failed_constructor_rollback_drain_retains_owner_storage_and_poison(monkeypatch):
    backend = planned_backend()
    backend.scheme, backend.device = "dense_prefetch", torch.device("cpu")
    plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})
    owner = object()
    backend.bind_owner(owner)
    backend.allocate_shared(plan, owner=owner)
    staging = backend._dense_staging
    storage = staging.storage_tensors()[0]

    def fail():
        raise RuntimeError("injected drain failure")

    monkeypatch.setattr(backend, "synchronize", fail)
    with pytest.raises(RuntimeError, match="injected drain failure"):
        backend.unbind_owner(owner, rollback=True)
    assert backend.lifecycle.poisoned and backend.lifecycle.admission_owner is owner
    assert backend.lifecycle.plan is plan and backend._dense_staging is staging
    assert staging.storage_tensors()[0] is storage


def test_partial_shared_allocation_failure_drains_before_dropping_traceback(monkeypatch):
    import models.deepseek_v32.execution.adapter as module

    backend = planned_backend()
    backend.scheme, backend.device = "dense_prefetch", torch.device("cpu")
    plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})
    owner = object()
    backend.bind_owner(owner)
    drained = []

    def fail(*args, **kwargs):
        raise RuntimeError("injected allocation failure")

    def drain():
        assert backend._allocation_failure is not None
        drained.append(True)

    monkeypatch.setattr(module.DoubleBufferStaging, "__init__", fail)
    monkeypatch.setattr(backend, "synchronize", drain)
    with pytest.raises(RuntimeError, match="allocation failure"):
        backend.allocate_shared(plan, owner=owner)
    assert drained == [True]
    assert backend.lifecycle.plan is backend._allocation_failure is None
    backend.unbind_owner(owner, rollback=True)
    assert backend.lifecycle.admission_owner is None


def test_pipeline_audit_and_cleanup_failure_retain_provider_and_owner():
    backend = planned_backend()
    backend.scheme, backend.device = "hbm", torch.device("cpu")
    plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})
    owner = object()
    backend.bind_owner(owner)

    class Resource:
        def close(self):
            raise RuntimeError("injected official cleanup failure")

    resource = Resource()

    class Pipeline(LocalPipeline):
        def allocate_resources(self, pool, plan, scheme):
            return resource

        def audit_resources(self, resources, plan):
            assert backend._pipeline_resources is resources is resource
            raise ValueError("injected official reservation failure")

    backend.pipeline = Pipeline()
    with pytest.raises(ExceptionGroup) as failed:
        backend.allocate_shared(plan, owner=owner)
    assert isinstance(failed.value.exceptions[0], ValueError)
    assert "reservation failure" in str(failed.value.exceptions[0])
    assert "official cleanup failure" in str(failed.value.exceptions[1])
    assert backend._pipeline_resources is resource
    assert backend._allocation_failure is failed.value.exceptions[0]
    assert backend.lifecycle.poisoned and backend.lifecycle.admission_owner is owner
    with pytest.raises(RuntimeError, match="poisoned"):
        backend.unbind_owner(owner, rollback=True)


def test_native_session_plan_is_consumed_without_reestimating(monkeypatch):
    from cache.capacity import SessionPlan
    from models.deepseek_v32.tests.test_serving_backend import _dense_model

    backend = _dense_model(monkeypatch)
    storage = backend._plan_session_storage(32, 16)
    plan = SessionPlan(
        backend.lifecycle.plan,
        "history",
        16,
        32,
        storage.reservation,
        allocations=storage.allocations,
        model_plan=storage,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("creation re-estimated the admitted storage")

    monkeypatch.setattr(backend, "_plan_session_storage", forbidden)
    session = backend.create_planned_session(plan)
    assert session.storage_plan is storage
    assert session.capacity == storage.capacity
    assert CacheFootprint.from_mapping(backend.session_bytes(session)).fits(plan.reservation)
    backend.release_session(session)
    backend.close()


@pytest.mark.parametrize("field", ["history_tokens", "host_pages", "hbm_tokens"])
def test_native_session_plan_rejects_changed_admission_before_allocation(monkeypatch, field):
    from dataclasses import replace

    from cache.capacity import CapacityPolicy
    from executor.contracts import ExecutionLimits, RequestShape

    backend = planned_backend()
    backend.scheme = "hbm" if field == "hbm_tokens" else "echo"
    limits = {"max_session_capacity": 24, "max_history_tokens": 16, "max_candidate_tokens": 8}
    native = backend.plan_resources(None, limits)
    backend.lifecycle.plan = native
    resources = replace(
        native,
        policy=CapacityPolicy.fixed_pools(),
        limits=ExecutionLimits.from_mapping(limits, context=1024, chunk_size=8),
    )
    plan = backend.plan_runtime_session(resources, RequestShape(16, 8), "history")

    def forbidden(*args, **kwargs):
        raise AssertionError("a changed declaration reached allocation")

    monkeypatch.setattr(backend, "_create_session_from_plan", forbidden)
    assert getattr(plan, field) > 0
    changed = replace(plan, **{field: getattr(plan, field) - 1})
    with pytest.raises(ValueError, match="native storage plan"):
        backend.create_planned_session(changed)


def test_dense_staging_is_reserved_once_before_any_session_allocation(monkeypatch):
    backend = planned_backend()
    backend.scheme = "dense_prefetch"
    plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})
    expected_staging = 2 * 128 * 576 * 2
    execution = execution_reservation(8, 128, topk=16, width=576)
    assert plan.shared.hbm == expected_staging + execution.hbm
    assert plan.metadata["dense_staging_bytes"] == expected_staging
    session = backend.estimate_session_bytes(64)
    assert session["hbm"] == 3 * (64 * (128 + 4 + 20) + 192)

    def forbidden(*args, **kwargs):
        raise AssertionError("unplanned dense session reached allocation")

    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(RuntimeError, match="plan_resources"):
        backend.create_session(64)


@pytest.mark.parametrize(
    ("logical", "expected"),
    [(1, 512), (512, 512), (513, 1024), (1048576, 1048576), (1048577, 2097664)],
)
def test_dense_staging_reserves_one_rounded_native_block_without_cuda(
    monkeypatch, logical, expected
):
    def forbidden(*args, **kwargs):
        raise AssertionError("allocation-free dense estimate touched CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    assert dense_staging_allocation_bytes(logical, "cuda:0") == expected
    assert dense_staging_allocation_bytes(logical, "cpu") == logical


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_dense_staging_bound_rejects_invalid_sizes(value):
    with pytest.raises(ValueError, match="positive integer"):
        dense_staging_allocation_bytes(value, "cuda:0")


def test_dense_staging_native_capacity_observation_rejects_unexpected_block(monkeypatch):
    tensor = SimpleNamespace(
        device=torch.device("cuda:0"),
        untyped_storage=lambda: SimpleNamespace(nbytes=lambda: 1048577, data_ptr=lambda: 4096),
    )
    reserved = dense_staging_allocation_bytes(1048577, "cuda:0")
    block = {"address": 4096, "size": reserved, "state": "active_allocated"}
    segments = [{"device": 0, "blocks": [block]}]
    calls = []

    def snapshot(settings):
        calls.append(settings)
        return {"segments": segments}

    monkeypatch.setattr(torch._C, "_cuda_memorySnapshot", snapshot)
    assert check_dense_staging_allocation(tensor, reserved) == reserved
    assert calls == [(0, 0, False)]
    block["size"] += 512
    with pytest.raises(RuntimeError, match="exceeds its reservation"):
        check_dense_staging_allocation(tensor, reserved)
    block["size"], block["state"] = reserved, "inactive"
    with pytest.raises(RuntimeError, match="one active allocator block"):
        check_dense_staging_allocation(tensor, reserved)
    block["state"] = "active_allocated"
    segments.append({"device": 0, "blocks": [dict(block)]})
    with pytest.raises(RuntimeError, match="one active allocator block"):
        check_dense_staging_allocation(tensor, reserved)


def test_dense_allocator_capacity_failure_drops_new_stage_after_drain(monkeypatch):
    import models.deepseek_v32.execution.adapter as module

    backend = planned_backend()
    backend.device, backend.scheme = torch.device("cpu"), "dense_prefetch"
    plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})
    owner = object()
    backend.bind_owner(owner)
    staging = []

    def unexpected_block(tensor, reserved):
        staging.append(backend._dense_staging)
        raise RuntimeError("dense staging allocator block exceeds its reservation")

    monkeypatch.setattr(module, "check_dense_staging_allocation", unexpected_block)
    with pytest.raises(RuntimeError, match="exceeds its reservation"):
        backend.allocate_shared(plan, owner=owner)
    assert staging[0].closed and not staging[0].storage_tensors()
    assert backend._dense_staging is backend.lifecycle.plan is None
    backend.unbind_owner(owner, rollback=True)
    assert backend.lifecycle.admission_owner is None


def test_direct_staging_lease_blocks_binding_and_close_without_sessions():
    backend = planned_backend()
    backend.device, backend.scheme = torch.device("cpu"), "dense_prefetch"
    plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})
    backend.allocate_shared(plan)
    with backend._dense_staging.lease(object()):
        with pytest.raises(RuntimeError, match="active execution"):
            backend.bind_owner(object())
        with pytest.raises(RuntimeError, match="execution"):
            backend.close()
    backend.close()


@pytest.mark.parametrize("scheme", ["hbm", "echo", "serial_sparse", "dense_prefetch"])
def test_explicit_candidate_bound_is_preserved_and_checked_in_every_plan(scheme):
    backend = planned_backend()
    backend.scheme = scheme
    budgets = CacheFootprint(1 << 28, 1 << 28)
    limits = {"max_session_capacity": 128, "max_candidate_tokens": 6}
    plan = backend.plan_resources(budgets, limits)
    assert plan.metadata["max_candidate_tokens"] == 6
    assert plan.metadata["workspace_query_tokens"] == 8
    with pytest.raises(CacheBudgetExceeded, match="whole-batch query workspace"):
        backend.plan_resources(budgets, {**limits, "max_candidate_tokens": 9})
    backend.extend_chunk_size = 4
    if scheme in ("echo", "serial_sparse"):
        # These backends always execute the complete transient candidate batch.
        with pytest.raises(CacheBudgetExceeded, match="whole-batch query workspace"):
            backend.plan_resources(budgets, {**limits, "max_candidate_tokens": 128})
    else:
        plan = backend.plan_resources(budgets, {**limits, "max_candidate_tokens": 128})
        assert plan.metadata["max_candidate_tokens"] == 128
    backend.extend_chunk_size = 9
    with pytest.raises(CacheBudgetExceeded, match="extend chunk"):
        backend.plan_resources(budgets, limits)


@pytest.mark.parametrize("candidate", [0, True, 1.5, 129])
def test_invalid_candidate_plan_rejects_without_allocation(candidate):
    backend = planned_backend()
    with pytest.raises(ValueError, match="0 < A <= C"):
        backend.plan_resources(
            CacheFootprint(1 << 28, 1 << 28),
            {"max_session_capacity": 128, "max_candidate_tokens": candidate},
        )
    assert backend.lifecycle.plan is backend._shared_pool is None


def test_foreign_or_under_reserved_dense_plan_rejects_before_tensor_allocation(monkeypatch):
    from dataclasses import replace

    backend = planned_backend()
    backend.scheme = "hbm"
    foreign = backend.plan_resources(
        CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128}
    )
    backend.scheme = "dense_prefetch"
    plan = backend.plan_resources(CacheFootprint(1 << 28, 1 << 28), {"max_session_capacity": 128})
    with pytest.raises(ValueError, match="allocations exceed"):
        replace(plan, shared=CacheFootprint(plan.shared.hbm - 1, plan.shared.dram))
    under = replace(
        plan, shared=CacheFootprint(plan.shared.hbm - 1, plan.shared.dram), allocations=()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid dense plan reached tensor allocation")

    monkeypatch.setattr(torch, "empty", forbidden)
    for invalid in (foreign, under):
        with pytest.raises(ValueError, match="does not match"):
            backend.allocate_shared(invalid)
    assert backend._dense_staging is backend.lifecycle.plan is None
