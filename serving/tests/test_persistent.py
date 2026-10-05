import pytest
import torch

from cache.capacity import ResourcePlan
from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint
from serving.persistent import PersistentGRRunner


class Backend:
    scheme = "hbm"
    device = torch.device("cpu")
    max_seq_len = 100

    def __init__(self):
        self.built = self.released = 0
        self.fail = False
        self.session_metrics = lambda session: {}

    def runtime_driver(self, policy):
        from executor.adapters import BackendAdapter

        return BackendAdapter(
            self,
            shared=False,
            candidate_mode="append_truncate",
            session_length=lambda session: len(session["tokens"]),
            chunk_size=self.max_seq_len,
            diagnostics=self.session_metrics,
        )

    def estimate_session_bytes(self, capacity, prefix_tokens):
        return {"hbm": capacity * 8, "dram": 0}

    def create_session(self, capacity):
        return {"tokens": [], "capacity": capacity}

    def prefill(self, session, ids):
        self.built += 1
        session["tokens"].extend(ids.tolist())

    def extend(self, session, ids):
        if self.fail:
            raise RuntimeError("model failure")
        outputs = []
        for token in ids.tolist():
            session["tokens"].append(token)
            outputs.append([sum(session["tokens"])])
        return torch.tensor(outputs)

    def truncate(self, session, prefix):
        del session["tokens"][prefix:]

    def session_bytes(self, session):
        return {"hbm": session["capacity"] * 8, "dram": 0}

    def release_session(self, session):
        session.clear()
        self.released += 1

    def synchronize(self):
        pass


def request(user=0, prefix=(1, 2), candidate=(3, 4)):
    return {"user_id": user, "input_ids": list(prefix + candidate), "stable_prefix_tokens": 2}


def test_revisits_and_suffix_rollback():
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=64) as runner:
        first = runner.execute(request())
        second = runner.execute(request(candidate=(5, 6)))
        assert first.hidden.tolist() == [[6], [10]]
        assert second.hidden.tolist() == [[8], [14]]
        assert not first.metrics["is_revisit"]
        assert second.metrics["is_revisit"]
        assert second.metrics["prefix_cache_hit"]
        assert second.metrics["prefix_hit_tier"] == "hbm"
        assert backend.built == 1
        assert second.metrics["latency_ms"] >= second.metrics["extend_ms"] >= 0
    assert backend.released == 1


def test_evicted_revisit_counts_as_revisit_but_miss():
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=32, dram_budget_bytes=64) as runner:
        runner.execute(request(0))
        runner.execute(request(1))
        result = runner.execute(request(0))
        assert result.metrics["visit_index"] == 1
        assert result.metrics["is_revisit"]
        assert not result.metrics["prefix_cache_hit"]
        assert result.metrics["evicted_users"] == [1]
        assert backend.built == 3


def test_changed_history_is_not_reused():
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=0) as runner:
        runner.execute(request())
        result = runner.execute(request(prefix=(7, 8)))
        assert result.hidden.tolist() == [[18], [22]]
        assert not result.metrics["prefix_cache_hit"]


def test_failed_candidate_discards_user_and_allows_recovery():
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=0) as runner:
        runner.execute(request())
        backend.fail = True
        with pytest.raises(RuntimeError, match="model failure"):
            runner.execute(request())
        assert len(runner.pool) == 0
        backend.fail = False
        result = runner.execute(request())
        assert not result.metrics["prefix_cache_hit"]
        assert result.metrics["visit_index"] == 1


def test_empty_request_never_allocates():
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=0) as runner:
        with pytest.raises(ValueError):
            runner.execute({"user_id": 0, "input_ids": [], "stable_prefix_tokens": 2})
        assert len(runner.pool) == 0


class SharedBackend(Backend):
    scheme = "echo"

    def __init__(self):
        super().__init__()
        self.allocations = 0
        self.plan = None
        self.owner = None
        self.initial_plan = None
        self.closed_resources = 0

    def runtime_driver(self, policy):
        from executor.adapters import BackendAdapter

        return BackendAdapter(
            self,
            shared=True,
            candidate_mode="append_truncate",
            session_length=lambda session: len(session["tokens"]),
            chunk_size=self.max_seq_len,
            diagnostics=self.session_metrics,
        )

    def bind_owner(self, owner):
        if owner is None or self.owner is not None:
            raise RuntimeError("backend already has an admission owner")
        self.initial_plan = self.plan
        self.owner = owner

    def unbind_owner(self, owner, *, rollback=False):
        if owner is not self.owner:
            raise ValueError("foreign admission owner")
        if rollback and self.initial_plan is None:
            self.plan = None
            self.closed_resources += 1
        self.owner = None

    def close(self):
        if self.owner is not None:
            raise RuntimeError("cannot close with admission owner")
        self.plan = None
        self.closed_resources += 1

    def plan_resources(self, budgets, limits):
        self.planned_budget = budgets
        self.planned_limits = limits
        return ResourcePlan(
            shared=CacheFootprint(16, 64),
            host_pages=2,
            page_size=4,
            metadata={"pool_scope": "backend"},
        )

    def allocate_shared(self, plan):
        if self.plan is None:
            self.allocations += 1
            self.plan = plan
        elif plan != self.plan:
            raise ValueError("cannot replace shared plan with live resources")

    def shared_bytes(self):
        return {"hbm": 16, "dram": 64}

    def estimate_session_host_pages(self, capacity):
        return (capacity + 3) // 4

    def session_host_pages(self, session):
        return self.estimate_session_host_pages(session["capacity"])


def test_shared_backend_accounts_arena_once_and_preserves_owner_lifetime():
    backend = SharedBackend()
    with PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64) as runner:
        runner.execute(request(0))
        result = runner.execute(request(1))
        assert result.metrics["cache_hbm_bytes"] == 80
        assert result.metrics["cache_dram_bytes"] == 64
        assert result.metrics["reserved_hbm_bytes"] == 80
        assert result.metrics["shared_cache_hbm_bytes"] == 16
        assert result.metrics["session_reserved_hbm_bytes"] == 64
        assert result.metrics["cache_host_pages"] == 2
        assert result.metrics["host_page_capacity"] == 2
        assert result.metrics["host_page_tokens"] == 4
        assert result.metrics["cache_pool_scope"] == "backend"
        assert backend.allocations == 1
        assert backend.planned_budget == CacheFootprint(80, 64)
        assert result.metrics["resource_mode"] == "budget"
        assert backend.planned_limits["max_session_capacity"] == 100
    assert backend.released == 2
    assert runner.pool.reserved == CacheFootprint(16, 64)
    with PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64) as runner:
        runner.execute(request(2))
    assert backend.allocations == 1


def test_fixed_pools_use_page_lru_report_unset_budgets_and_keep_owner_lifetime():
    backend = SharedBackend()
    with PersistentGRRunner(backend, resource_limits={"max_session_capacity": 4}) as runner:
        assert backend.owner is runner and backend.planned_budget is None
        runner.execute(request(0))
        runner.execute(request(1))
        hit = runner.execute(request(0))
        assert hit.metrics["prefix_cache_hit"]
        assert hit.metrics["resource_mode"] == "fixed_pools"
        assert hit.metrics["hbm_budget_bytes"] is None
        assert hit.metrics["dram_budget_bytes"] is None
        assert hit.metrics["cached_users"] == 2
        assert hit.metrics["cache_hbm_bytes"] == hit.metrics["reserved_hbm_bytes"] == 80
        assert runner.execute(request(2)).metrics["evicted_users"] == [1]
        miss = runner.execute(request(1))
        assert miss.metrics["visit_index"] == 1 and not miss.metrics["prefix_cache_hit"]
        assert miss.metrics["evicted_users"] == [0]
    assert backend.owner is None and backend.plan is not None
    assert backend.released == 4


@pytest.mark.parametrize("budgets", [(None, 64), (64, None)])
def test_one_missing_byte_budget_rejects_before_planning_or_allocation(budgets):
    backend = SharedBackend()
    with pytest.raises(ValueError, match="provide both"):
        PersistentGRRunner(backend, hbm_budget_bytes=budgets[0], dram_budget_bytes=budgets[1])
    assert not hasattr(backend, "planned_budget")
    assert backend.owner is None and backend.allocations == 0


def test_fixed_pools_require_shared_pages_and_one_maximum_session_before_allocation():
    with pytest.raises(ValueError, match="shared backend with host pages"):
        PersistentGRRunner(Backend())
    backend = SharedBackend()
    with pytest.raises(CacheBudgetExceeded, match="maximum-size session"):
        PersistentGRRunner(backend)
    assert backend.owner is None and backend.allocations == 0
    backend.plan_resources = lambda *_: ResourcePlan(shared=CacheFootprint(16, 64))
    with pytest.raises(ValueError, match="shared backend with host pages"):
        PersistentGRRunner(backend, resource_limits={"max_session_capacity": 4})
    assert backend.owner is None and backend.allocations == 0


def test_fixed_pools_reject_shared_allocation_overrun_and_roll_back_owner():
    backend = SharedBackend()
    backend.shared_bytes = lambda: {"hbm": 17, "dram": 64}
    with pytest.raises(RuntimeError, match="shared cache allocation exceeded"):
        PersistentGRRunner(backend, resource_limits={"max_session_capacity": 4})
    assert backend.owner is None and backend.plan is None
    assert backend.closed_resources == 1


def test_shared_plan_is_checked_before_allocation():
    backend = SharedBackend()
    with pytest.raises(CacheBudgetExceeded, match="shared cache"):
        PersistentGRRunner(backend, hbm_budget_bytes=15, dram_budget_bytes=64)
    assert backend.allocations == 0


def test_shared_actual_capacity_is_audited_before_requests():
    backend = SharedBackend()
    backend.shared_bytes = lambda: {"hbm": 17, "dram": 64}
    with pytest.raises(RuntimeError, match="shared cache allocation exceeded"):
        PersistentGRRunner(backend, hbm_budget_bytes=100, dram_budget_bytes=64)
    assert backend.built == backend.released == 0
    assert backend.owner is None
    assert backend.plan is None
    assert backend.closed_resources == 1


def test_two_empty_runners_cannot_bind_the_same_backend():
    backend = SharedBackend()
    with PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64) as first:
        with pytest.raises(RuntimeError, match="admission owner"):
            PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64)
        assert backend.owner is first
        assert backend.allocations == 1
        first.execute(request())
    assert backend.owner is None
    assert backend.plan is not None
    assert backend.closed_resources == 0


@pytest.mark.parametrize("existing_plan", [False, True])
@pytest.mark.parametrize("phase", ["allocation", "pool", "audit"])
def test_constructor_rollback_preserves_only_preexisting_resources(
    monkeypatch, existing_plan, phase
):
    backend = SharedBackend()
    if existing_plan:
        with PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64):
            pass
    original_plan = backend.plan

    def fail(*args, **kwargs):
        raise RuntimeError("injected initialization failure")

    with monkeypatch.context() as patch:
        if phase == "allocation":
            allocate = backend.allocate_shared

            def allocate_then_fail(plan):
                allocate(plan)
                fail()

            patch.setattr(backend, "allocate_shared", allocate_then_fail)
        elif phase == "pool":
            patch.setattr("serving.persistent.PrefixSessionPool", fail)
        else:
            patch.setattr(backend, "shared_bytes", fail)
        with pytest.raises(RuntimeError, match="injected initialization failure"):
            PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64)
    assert backend.owner is None
    assert backend.plan is original_plan
    assert backend.closed_resources == (0 if existing_plan else 1)
    with PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64) as runner:
        runner.execute(request())


def test_constructor_rollback_failure_retains_admission_owner(monkeypatch):
    backend = SharedBackend()

    def fail_unbind(owner, *, rollback=False):
        assert rollback
        assert owner is backend.owner
        raise RuntimeError("cannot establish completion")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "shared_bytes", lambda: {"hbm": 17, "dram": 64})
        patch.setattr(backend, "unbind_owner", fail_unbind)
        with pytest.raises(ExceptionGroup, match="cleanup failed; ownership retained") as error:
            PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64)
    assert isinstance(error.value.exceptions[0], RuntimeError)
    assert "allocation exceeded reservation" in str(error.value.exceptions[0])
    assert "cannot establish completion" in str(error.value.exceptions[1])
    assert backend.owner is not None
    with pytest.raises(RuntimeError, match="admission owner"):
        PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64)
    backend.owner.close()
    assert backend.owner is None
    assert backend.plan is None


def test_context_body_and_close_failures_remain_independently_inspectable(monkeypatch):
    backend = SharedBackend()
    runner = PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64)
    failure = ValueError("caller failed")
    close_failure = RuntimeError("cannot finish release")
    with monkeypatch.context() as patch:

        def fail_release(session):
            raise close_failure

        patch.setattr(runner.pool, "_release", fail_release)
        with pytest.raises(ExceptionGroup) as caught, runner:
            runner.execute(request())
            raise failure
    assert caught.value.exceptions == (failure, close_failure)
    assert backend.owner is runner
    runner.close()
    backend.close()


def test_failed_close_disables_execution_and_retains_owner_until_retry(monkeypatch):
    backend = SharedBackend()
    runner = PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64)
    runner.execute(request())
    release = runner.pool._release

    def fail(session):
        raise RuntimeError("release failed")

    monkeypatch.setattr(runner.pool, "_release", fail)
    with pytest.raises(RuntimeError, match="release failed"):
        runner.close()
    assert backend.owner is runner
    assert len(runner.pool) == 1
    with pytest.raises(RuntimeError, match="runner has been closed"):
        runner.execute(request())
    with pytest.raises(RuntimeError, match="admission owner"):
        PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64)
    monkeypatch.setattr(runner.pool, "_release", release)
    runner.close()
    runner.close()
    assert backend.owner is None
    assert backend.released == 1
    assert backend.plan is not None


def test_host_page_quota_limits_users_independently_of_bytes():
    backend = SharedBackend()
    with PersistentGRRunner(backend, hbm_budget_bytes=1000, dram_budget_bytes=64) as runner:
        runner.execute(request(0))
        runner.execute(request(1))
        result = runner.execute(request(2))
        assert result.metrics["evicted_users"] == [0]
        assert result.metrics["cache_host_pages"] == 2
        assert result.metrics["cache_dram_bytes"] == 64


def test_impossible_pages_do_not_evict_or_change_revisit_state():
    backend = SharedBackend()
    with PersistentGRRunner(backend, hbm_budget_bytes=1000, dram_budget_bytes=64) as runner:
        runner.execute(request(0))
        runner.execute(request(1))
        with pytest.raises(CacheBudgetExceeded, match="host pages"):
            runner.execute(request(0, prefix=(5, 6), candidate=tuple(range(10))))
        assert backend.released == 0
        result = runner.execute(request(2))
        assert result.metrics["evicted_users"] == [0]
        assert runner.visits[0] == 1


def test_planned_capacity_is_enforced_before_session_allocation():
    backend = SharedBackend()
    with PersistentGRRunner(
        backend,
        hbm_budget_bytes=1000,
        dram_budget_bytes=64,
        resource_limits={"max_session_capacity": 4},
    ) as runner:
        assert backend.planned_limits["max_session_capacity"] == 4
        with pytest.raises(ValueError, match="planned session capacity"):
            runner.execute(request(candidate=(3, 4, 5)))
        assert len(runner.pool) == 0
        assert backend.built == 0


def test_incomplete_shared_interface_fails_before_allocation():
    backend = SharedBackend()
    backend.session_host_pages = None
    with pytest.raises(TypeError, match="session_host_pages"):
        PersistentGRRunner(backend, hbm_budget_bytes=1000, dram_budget_bytes=64)
    assert backend.allocations == 0


@pytest.mark.parametrize("source", ["limits", "plan"])
def test_candidate_bound_rejects_before_evicting_reusable_prefix(source):
    backend = SharedBackend()
    options = {"resource_limits": {"max_candidate_tokens": 2}} if source == "limits" else {}
    if source == "plan":
        planner = backend.plan_resources

        def bounded_plan(budgets, limits):
            plan = planner(budgets, limits)
            return ResourcePlan(
                shared=plan.shared,
                host_pages=plan.host_pages,
                page_size=plan.page_size,
                metadata={**plan.metadata, "max_candidate_tokens": 2},
            )

        backend.plan_resources = bounded_plan
    with PersistentGRRunner(
        backend, hbm_budget_bytes=80, dram_budget_bytes=64, **options
    ) as runner:
        runner.execute(request(0))
        runner.execute(request(1))
        with pytest.raises(ValueError, match="candidate suffix"):
            runner.execute(request(2, candidate=(3, 4, 5)))
        assert backend.released == 0
        result = runner.execute(request(2))
        assert result.metrics["evicted_users"] == [0]


def test_optional_diagnostics_observe_completed_prefix_cleanup():
    backend = Backend()
    backend.session_metrics = lambda session: {"retained_tokens": len(session["tokens"])}
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=0) as runner:
        result = runner.execute(request())
    assert result.metrics["cache_diagnostics"] == {"retained_tokens": 2}


class ResidentSharedBackend(SharedBackend):
    scheme = "hbm"

    def runtime_driver(self, policy):
        from executor.adapters import BackendAdapter

        return BackendAdapter(
            self,
            shared=True,
            candidate_mode="gpu_transient",
            session_length=lambda session: len(session["tokens"]),
            chunk_size=self.max_seq_len,
            diagnostics=self.session_metrics,
        )

    def plan_resources(self, budgets, limits):
        self.planned_budget = budgets
        self.planned_limits = limits
        return ResourcePlan(shared=CacheFootprint(16, 0), hbm_tokens=2)

    def shared_bytes(self):
        return {"hbm": 16, "dram": 0}

    def retained_session_capacity(self, capacity, prefix_tokens):
        return prefix_tokens

    def estimate_session_host_pages(self, capacity):
        return 0

    def estimate_session_hbm_tokens(self, capacity):
        return capacity

    def session_hbm_tokens(self, session):
        return session["capacity"]

    def extend_candidate(self, session, ids):
        total = sum(session["tokens"])
        output = []
        for token in ids.tolist():
            total += token
            output.append([total])
        return torch.tensor(output)


def test_fixed_resident_pool_counts_only_history_as_candidates_change():
    backend = ResidentSharedBackend()
    limits = {"max_session_capacity": 5, "max_history_tokens": 2, "max_candidate_tokens": 3}
    with PersistentGRRunner(backend, resource_limits=limits) as runner:
        first = runner.execute(request())
        second = runner.execute(request(candidate=(5, 6, 7)))
        assert not first.metrics["prefix_cache_hit"]
        assert second.metrics["prefix_cache_hit"]
        assert second.metrics["prefix_hit_tier"] == "hbm"
        assert second.hidden.tolist() == [[8], [14], [21]]
        assert second.metrics["cache_hbm_tokens"] == second.metrics["hbm_token_capacity"] == 2
        assert second.metrics["session_hbm_tokens"] == 2
        assert second.metrics["cache_host_pages"] == second.metrics["host_page_capacity"] == 0
        assert second.metrics["cache_dram_bytes"] == 0
        assert second.metrics["resource_mode"] == "fixed_pools"
        assert second.metrics["hbm_budget_bytes"] is None
        assert second.metrics["cached_users"] == 1
        assert backend.built == 1
    assert backend.released == 1


def test_sixteen_users_two_rounds_with_one_history_hbm_pool_rebuild_every_request():
    backend = ResidentSharedBackend()
    limits = {"max_session_capacity": 4, "max_history_tokens": 2}
    with PersistentGRRunner(backend, resource_limits=limits) as runner:
        for visit in range(2):
            for user in range(16):
                result = runner.execute(request(user))
                assert not result.metrics["prefix_cache_hit"]
                assert result.metrics["visit_index"] == visit
                assert result.metrics["is_revisit"] == bool(visit)
                assert result.metrics["cached_users"] == 1
                assert result.metrics["cache_hbm_tokens"] == 2
                assert result.metrics["session_host_pages"] == 0
        assert backend.built == 32
        assert backend.released == 31
    assert backend.released == 32


def test_fixed_resident_max_history_rejected_before_allocation():
    backend = ResidentSharedBackend()
    with pytest.raises(CacheBudgetExceeded, match="fixed HBM pool"):
        PersistentGRRunner(
            backend, resource_limits={"max_session_capacity": 5, "max_history_tokens": 3}
        )
    assert backend.owner is None and backend.allocations == 0


@pytest.mark.parametrize("name", ["estimate_session_hbm_tokens", "session_hbm_tokens"])
def test_hbm_quota_requires_both_backend_hooks_before_allocation(name):
    backend = ResidentSharedBackend()
    setattr(backend, name, None)
    with pytest.raises(TypeError, match=name):
        PersistentGRRunner(
            backend, resource_limits={"max_session_capacity": 4, "max_history_tokens": 2}
        )
    assert backend.owner is None and backend.allocations == 0


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_shared_plan_rejects_invalid_hbm_token_quotas(value):
    with pytest.raises(ValueError, match="hbm_tokens"):
        ResourcePlan(hbm_tokens=value)


@pytest.mark.parametrize("scheme", ["hbm", "dense_prefetch"])
def test_explicit_transient_driver_controls_admission_before_backend_allocation(scheme):
    class DeferredModeBackend(ResidentSharedBackend):
        def plan_resources(self, budgets, limits):
            return ResourcePlan(
                shared=CacheFootprint(16, 0),
                host_pages=1 if scheme == "dense_prefetch" else 0,
                hbm_tokens=2 if scheme == "hbm" else 0,
                metadata={"candidate_persistence": "gpu_transient"},
            )

        def retained_session_capacity(self, capacity, prefix_tokens):
            return prefix_tokens if self.plan is not None else capacity

        def estimate_session_host_pages(self, capacity):
            return (capacity + 1) // 2 if scheme == "dense_prefetch" else 0

    backend = DeferredModeBackend()
    backend.scheme = scheme
    with PersistentGRRunner(
        backend, resource_limits={"max_session_capacity": 5, "max_history_tokens": 2}
    ) as runner:
        first = runner.execute(request())
        second = runner.execute(request(candidate=(5, 6, 7)))
        assert not first.metrics["prefix_cache_hit"]
        assert second.metrics["prefix_cache_hit"]
        assert second.metrics["session_hbm_tokens"] == (2 if scheme == "hbm" else 0)
        assert second.metrics["session_host_pages"] == (1 if scheme == "dense_prefetch" else 0)
        assert runner.pool._entries[0].capacity == 2


def test_disabled_plan_quotas_do_not_call_backend_token_hooks():
    backend = SharedBackend()
    backend.plan_resources = lambda *_: ResourcePlan(shared=CacheFootprint(16, 64))

    def forbidden(*args):
        raise AssertionError("disabled quota hook was called")

    for name in (
        "estimate_session_host_pages",
        "session_host_pages",
        "estimate_session_hbm_tokens",
        "session_hbm_tokens",
    ):
        setattr(backend, name, forbidden)
    with PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64) as runner:
        first = runner.execute(request())
        second = runner.execute(request())
        assert not first.metrics["prefix_cache_hit"]
        assert second.metrics["prefix_cache_hit"]
        assert second.metrics["cache_hbm_tokens"] == second.metrics["cache_host_pages"] == 0
