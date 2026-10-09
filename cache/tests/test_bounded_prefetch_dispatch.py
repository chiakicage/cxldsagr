"""Eligibility and failure propagation for the official Q1 preparation provider."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_pool import MISSING, SharedSparseTokenPool
from operators.deepseek_v32.indexer.cache_ops import supports_free_q1_prepare


@pytest.mark.parametrize(
    ("history", "headroom", "queries", "limit", "sessions", "transient", "available", "bounded"),
    [
        (32767, 64, 1, 8192, 1, False, True, True),
        (32767, 64, 1, 64, 1, False, True, True),
        (32767, 63, 1, 8192, 1, False, True, False),
        (32766, 64, 1, 8192, 1, False, True, False),
        (32767, 64, 2, 8192, 1, False, True, False),
        (32767, 64, 1, 63, 1, False, True, False),
        (32767, 64, 1, 8192, 2, False, True, False),
        (32767, 64, 1, 8192, 1, True, True, False),
        (32767, 64, 1, 8192, 1, False, False, False),
    ],
)
def test_bounded_dispatch_support_boundary(
    monkeypatch, history, headroom, queries, limit, sessions, transient, available, bounded
):
    pool = SharedSparseTokenPool(
        33024, 7, 1, history + headroom, device="cpu", dtype=torch.float32, candidate_slots=2
    )
    cache = pool.allocate_session(32769).layer(0)
    if sessions == 2:
        pool.allocate_session(64)
    # Only dispatch is exercised: initialized host length with zero resident
    # records is legal capacity evidence and deliberately has no residency proof.
    cache.length = cache.written = cache.indexer_visible_end = history
    cache.begin_step(queries, transient=transient)
    cache.declare_indexer_visible(history + queries)
    calls = []
    token = SimpleNamespace(used=False)
    token.invalidate = lambda: setattr(token, "used", True)

    def full(*args, **kwargs):
        calls.append("full")
        return token

    def free(*args, **kwargs):
        assert pool._active == (cache.session.owner, 0) and pool._depth > 0
        assert not cache.all_history_resident
        calls.append("free")
        return token

    provider = SimpleNamespace(prepare_prefetch=full)
    if available:
        provider.prepare_prefetch_free = free
        provider.supports_free_q1_prepare = supports_free_q1_prepare
    # No GPU or fake numeric result is used; native dispatch is a recording stub.
    pool.allocation_log.fill_(MISSING)
    with monkeypatch.context() as patch:
        patch.setattr(SharedSparseTokenPool, "native_metadata", property(lambda self: provider))
        lease = cache.prepare_prefetch(history, queries, torch.zeros(16), limit=limit)
    assert calls == ["free" if bounded else "full"]
    assert lease["_prepared"] is token
    assert lease.get("prepared_limit") == (64 if bounded else None)
    assert lease["max_prefetch"] == (
        64 if bounded else min(limit, 8192, cache.slots - (0 if transient else queries))
    )
    cache.rollback()
    assert token.used
    pool.close()


def test_bounded_provider_failure_propagates_without_full_prepare(monkeypatch):
    pool = SharedSparseTokenPool(32832, 7, 1, 32831, device="cpu", dtype=torch.float32)
    cache = pool.allocate_session(32768).layer(0)
    cache.length = cache.written = cache.indexer_visible_end = 32767
    cache.begin_step(1)
    cache.declare_indexer_visible(32768)
    failure = RuntimeError("bounded provider failure")

    def fail(*args, **kwargs):
        raise failure

    def full(*args, **kwargs):
        raise AssertionError("failed bounded preparation must not invoke full preparation")

    provider = SimpleNamespace(
        supports_free_q1_prepare=supports_free_q1_prepare,
        prepare_prefetch_free=fail,
        prepare_prefetch=full,
    )
    with monkeypatch.context() as patch:
        patch.setattr(SharedSparseTokenPool, "native_metadata", property(lambda self: provider))
        with pytest.raises(RuntimeError) as raised:
            cache.prepare_prefetch(32767, 1, torch.zeros(16))
    assert raised.value is failure
    assert cache._prefetch is None and pool._pending_prefetch is None
    cache.rollback()
    pool.close()
