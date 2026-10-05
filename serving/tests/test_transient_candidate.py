"""CPU contracts for history-only admission and candidate failure cleanup."""

import pytest
import torch

from cache.capacity import ResourcePlan
from cache.prefix_pool import CacheFootprint
from serving.persistent import PersistentGRRunner
from serving.tests.test_persistent import SharedBackend


class TransientBackend(SharedBackend):
    """Minimal backend: history owns pages; candidate storage is shared."""

    max_seq_len = 1024

    def __init__(self, *, host_pages=1, plan_limits=None):
        super().__init__()
        self.host_pages = host_pages
        self.plan_limits = plan_limits or {}
        self.created_capacities = []
        self.estimate_calls = []
        self.candidate_calls = 0

    def runtime_driver(self, policy):
        from executor.adapters import BackendAdapter

        return BackendAdapter(
            self,
            shared=True,
            candidate_mode="gpu_transient",
            session_length=lambda session: len(session["tokens"]),
            chunk_size=self.max_seq_len,
        )

    def plan_resources(self, budgets, limits):
        self.planned_budget, self.planned_limits = budgets, dict(limits)
        return ResourcePlan(
            shared=CacheFootprint(16, 64),
            host_pages=self.host_pages,
            page_size=64,
            metadata={"pool_scope": "backend", **limits, **self.plan_limits},
        )

    def retained_session_capacity(self, capacity, prefix_tokens):
        return prefix_tokens

    def estimate_session_host_pages(self, capacity):
        return (capacity + 63) // 64

    def estimate_session_bytes(self, capacity, prefix_tokens):
        self.estimate_calls.append((capacity, prefix_tokens))
        return super().estimate_session_bytes(capacity, prefix_tokens)

    def create_session(self, capacity):
        self.created_capacities.append(capacity)
        return super().create_session(capacity)

    def extend(self, session, ids):
        raise AssertionError("runner bypassed the optional extend_candidate entrypoint")

    def extend_candidate(self, session, ids):
        self.candidate_calls += 1
        if self.fail:
            raise RuntimeError("candidate failure")
        running = sum(session["tokens"])
        rows = []
        for token in ids.tolist():
            running += token
            rows.append([running])
        return torch.tensor(rows)


def request(user="user", *, history=64, candidates=1):
    return {
        "user_id": user,
        "input_ids": list(range(1, history + 1)) + list(range(100, 100 + candidates)),
        "stable_prefix_tokens": history,
    }


def limits(**changes):
    return {
        "max_session_capacity": 256,
        "max_history_tokens": 64,
        "max_candidate_tokens": 96,
        **changes,
    }


@pytest.mark.parametrize("fixed_pools", [True, False])
def test_history_only_admission_keeps_same_session_when_candidate_grows(fixed_pools):
    backend = TransientBackend()
    budgets = {} if fixed_pools else {"hbm_budget_bytes": 528, "dram_budget_bytes": 64}
    with PersistentGRRunner(backend, resource_limits=limits(), **budgets) as runner:
        first = runner.execute(request(candidates=1))
        original = runner.pool._entries["user"].session
        grown = runner.execute(request(candidates=96))
        assert grown.metrics["prefix_cache_hit"]
        assert grown.metrics["prefix_hit_tier"] == "dram"
        assert grown.metrics["candidate_suffix_tokens"] == 96
        assert grown.hidden.shape == (96, 1)
        assert grown.hidden[-1, 0] == sum(range(1, 65)) + sum(range(100, 196))
        assert runner.pool._entries["user"].session is original
        assert original["tokens"] == list(range(1, 65))
        assert backend.created_capacities == [64]
        assert backend.estimate_calls == [(64, 64), (64, 64)]
        assert backend.built == 1 and backend.candidate_calls == 2
        assert first.metrics["cache_host_pages"] == grown.metrics["cache_host_pages"] == 1
        assert first.metrics["cache_hbm_bytes"] == grown.metrics["cache_hbm_bytes"] == 528
        assert grown.metrics["session_host_pages"] == grown.metrics["cached_users"] == 1
        assert not grown.metrics["evicted_users"]


def test_candidate_failure_raises_and_discards_session():
    backend = TransientBackend()
    with PersistentGRRunner(backend, resource_limits=limits()) as runner:
        runner.execute(request())
        backend.fail = True
        with pytest.raises(RuntimeError, match="candidate failure"):
            runner.execute(request(candidates=3))
        assert len(runner.pool) == 0
        assert runner.pool.reserved_host_pages == 0
        assert backend.released == 1


@pytest.mark.parametrize("same_user", [True, False])
@pytest.mark.parametrize(
    "over_limit,match",
    [({"candidates": 97}, "candidate suffix"), ({"history": 65}, "history exceeds")],
)
def test_request_shape_limits_reject_before_admission_or_eviction(same_user, over_limit, match):
    backend = TransientBackend()
    with PersistentGRRunner(backend, resource_limits=limits()) as runner:
        runner.execute(request())
        entry = runner.pool._entries["user"]
        calls = list(backend.estimate_calls)
        with pytest.raises(ValueError, match=match):
            runner.execute(request("user" if same_user else "new-user", **over_limit))
        assert runner.pool._entries == {"user": entry}
        assert entry.ready
        assert backend.estimate_calls == calls
        assert backend.created_capacities == [64]
        assert backend.released == runner.pool.evictions == 0
        assert runner.pool.reserved_host_pages == 1


@pytest.mark.parametrize(
    "over_limit,match",
    [({"candidates": 9}, "candidate suffix"), ({"history": 33}, "history exceeds")],
)
def test_stricter_backend_plan_limits_also_reject_before_eviction(over_limit, match):
    backend = TransientBackend(plan_limits={"max_history_tokens": 32, "max_candidate_tokens": 8})
    with PersistentGRRunner(backend, resource_limits=limits()) as runner:
        runner.execute(request(history=32))
        entry = runner.pool._entries["user"]
        bad = {"history": 32, "candidates": 1, **over_limit}
        with pytest.raises(ValueError, match=match):
            runner.execute(request("new-user", **bad))
        assert runner.pool._entries == {"user": entry}
        assert backend.created_capacities == [32]
        assert backend.released == 0
        assert backend.estimate_calls == [(32, 32)]
