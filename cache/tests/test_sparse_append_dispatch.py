"""Dispatch and failure contracts for sole-session persistent free-slot append."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_pool import MISSING, SharedSparseTokenPool


def select_free(priority, free, d2h, count, timestamp):
    """Independent contract oracle; deliberately has no tile/warp scan logic."""
    if any(age < -1 or age > timestamp for age in priority[1:]):
        raise ValueError("invalid age")
    eligible = [slot for slot in range(1, len(priority)) if priority[slot] == -1]
    if len(eligible) < count:
        raise ValueError("insufficient free")
    chosen = eligible[:count]
    if any(not free[slot] or d2h[slot] != MISSING for slot in chosen):
        raise ValueError("free state disagrees")
    return chosen


class Provider:
    def __init__(self, *, free=True, failure=None):
        self.calls = []
        self.failure = failure
        if free:
            self.sparse_append_free = lambda *args, **kwargs: self.run("free", *args, **kwargs)

    def sparse_append(self, *args, **kwargs):
        return self.run("fifo", *args, **kwargs)

    @staticmethod
    def release_ids(ids, h2d, d2h, priority, free):
        for global_id in ids.tolist():
            slot = int(h2d[global_id])
            if slot == MISSING:
                continue
            assert int(d2h[slot]) == global_id
            h2d[global_id], d2h[slot] = MISSING, MISSING
            priority[slot], free[slot] = -1, True

    def run(
        self,
        method,
        source,
        records,
        pages,
        h2d,
        d2h,
        priority,
        free,
        clock,
        evictions,
        keys,
        chosen,
        *,
        start,
        timestamp,
    ):
        self.calls.append(method)
        if self.failure is not None:
            raise self.failure
        if method == "free":
            slots = select_free(
                priority.tolist(), free.tolist(), d2h.tolist(), len(source), timestamp
            )
        else:
            slots = (torch.argsort(priority[1:], stable=True)[: len(source)] + 1).tolist()
        for row, slot in enumerate(slots):
            global_id = int(pages[(start + row) // 64]) * 64 + (start + row) % 64
            assert int(h2d[global_id]) == MISSING
            old = int(d2h[slot])
            if old != MISSING:
                assert int(h2d[old]) == slot
                h2d[old] = MISSING
                evictions.add_(1)
            h2d[global_id], d2h[slot] = slot, global_id
            priority[slot], free[slot] = timestamp, False
            records[slot].copy_(source[row])
        priority[0], clock[0] = MISSING, timestamp + 1


@pytest.fixture
def setup_candidate(monkeypatch):
    monkeypatch.setattr(
        SharedSparseTokenPool, "native_metadata", property(lambda self: self.metadata_ops)
    )
    pools = []

    def make(*, slots=8, prefix=4, free=True, failure=None):
        provider = Provider(free=free, failure=failure)
        pool = SharedSparseTokenPool(
            256,
            7,
            1,
            slots,
            device="cpu",
            dtype=torch.float32,
            candidate_slots=4,
            metadata_ops=provider,
        )
        pools.append(pool)
        cache = pool.allocate_session(16).layer(0)
        cache.begin_step(prefix)
        cache.append(torch.arange(prefix * 7, dtype=torch.float32).reshape(prefix, 7))
        cache.commit()
        pool.restore(pool.snapshot())
        assert not cache.all_history_resident
        return SimpleNamespace(pool=pool, cache=cache, provider=provider)

    yield make
    for pool in pools:
        pool.close()


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("single", "free"),
        ("second_session", "fifo"),
        ("released_session", "free"),
        ("over_P", "fifo"),
        ("legacy_provider", "fifo"),
    ],
)
def test_actual_dispatch_tracks_live_eligibility(setup_candidate, scenario, expected):
    state = setup_candidate(
        prefix=8 if scenario == "over_P" else 4, free=scenario != "legacy_provider"
    )
    pool, cache, provider = state.pool, state.cache, state.provider
    if scenario in ("second_session", "released_session"):
        other = pool.allocate_session(16)
        if scenario == "released_session":
            other.release()
    start = cache.written
    rows = torch.full((1, 7), 91.0)
    before_clock = pool.layers[0].clock
    cache.begin_step(1)
    with cache.operation():
        assert pool._active == (cache.session.owner, 0) and pool._depth > 0
        cache.append(rows)
    cache.commit()
    assert provider.calls == [expected]
    assert pool._active is None and pool._depth == 0
    assert pool.layers[0].clock == before_clock + 1
    assert int(pool.layers[0].clock_tensor[0]) == before_clock + 1
    torch.testing.assert_close(cache.host_records()[start : start + 1], rows, rtol=0, atol=0)
    slot = int(cache.host_to_device[cache.logical_to_global(torch.tensor([start]))[0]])
    torch.testing.assert_close(cache.records[slot : slot + 1], rows, rtol=0, atol=0)
    assert not cache.all_history_resident


def test_free_failure_is_not_retried_and_source_write_is_not_submitted(
    setup_candidate, monkeypatch
):
    failure = RuntimeError("injected append submission failure")
    state = setup_candidate(failure=failure)
    cache, pool = state.cache, state.pool
    writes = []
    monkeypatch.setattr(pool, "write_host", lambda *args: writes.append(args))
    old_clock = pool.layers[0].clock
    cache.begin_step(1)
    with pytest.raises(RuntimeError) as caught:
        cache.append(torch.ones(1, 7))
    assert caught.value is failure
    assert state.provider.calls == ["free"] and writes == []
    assert pool.layers[0].clock == old_clock and cache.written == cache.length
    assert pool._active is None and pool._depth == 0
    cache.rollback()


def test_transient_append_does_not_dispatch_or_write_host(setup_candidate, monkeypatch):
    state = setup_candidate()
    writes = []
    monkeypatch.setattr(state.pool, "write_host", lambda *args: writes.append(args))
    state.cache.begin_transient(3)
    state.cache.append(torch.ones(3, 7))
    state.cache.discard_transient()
    assert state.provider.calls == [] and writes == []
