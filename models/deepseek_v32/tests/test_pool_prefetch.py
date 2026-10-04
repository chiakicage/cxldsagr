"""Finite-pool whole-history lookahead, including GPU-only candidate tails."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_cache import WorkingSetTooLarge
from cache.sparse_token_pool import PRIORITY_LIMIT, SharedSparseTokenPool
from models.deepseek_v32.pool_prefetch import PoolHistoryPrefetch


def make_pool(*, slots=8, layers=1, device="cpu"):
    return SharedSparseTokenPool(
        256,
        7,
        layers,
        slots,
        device=device,
        dtype=torch.float32,
        candidate_slots=3,
    )


def populate(cache, count, offset=0):
    values = torch.arange(
        offset, offset + count * cache.width, device=cache.device, dtype=cache.records.dtype
    ).reshape(count, cache.width)
    cache.begin_step(count)
    for chunk in values.split(cache.slots):
        cache.append(chunk)
    cache.commit()
    return values


def test_full_history_prefetch_copies_only_misses_and_preserves_candidate_tail():
    pool = make_pool()
    session = pool.allocate_session(8)
    cache = session.layer(0)
    history = populate(cache, 8)
    other = pool.allocate_session(4).layer(0)
    populate(other, 4, 1000)
    cache.reset_stats()
    cache.begin_transient(3)
    candidate = torch.arange(21, dtype=torch.float32).reshape(3, 7) + 2000
    cache.append(candidate)
    host_before = cache.host.clone()
    pool_bytes = pool.shared_bytes()
    records_pointer = cache.records.data_ptr()
    helper = PoolHistoryPrefetch("cpu")
    ticket = helper.prefetch(cache)
    assert (ticket.requested_records, ticket.resident_records, ticket.fetched_records) == (8, 4, 4)
    assert ticket.fetched_bytes == 4 * 7 * 4
    helper.wait(ticket)
    ids = torch.arange(11)
    physical = cache.ensure(ids)
    torch.testing.assert_close(cache.records[physical.long()], torch.cat((history, candidate)))
    torch.testing.assert_close(
        cache.host.view(torch.uint8), host_before.view(torch.uint8), rtol=0, atol=0
    )
    assert physical[-3:].tolist() == [9, 10, 11]
    assert cache.stats.recalled_records == 4
    assert cache.metrics()["device_to_host_bytes"] == 0
    assert pool.shared_bytes() == pool_bytes
    assert cache.records.data_ptr() == records_pointer
    helper.drain()
    cache.discard_transient()
    assert cache.length == cache.written == session.capacity == 8
    assert session.host_pages == 1
    helper.close()


def test_all_hit_history_is_requested_without_recopying():
    pool = make_pool()
    cache = pool.allocate_session(8).layer(0)
    history = populate(cache, 8)
    cache.reset_stats()
    helper = PoolHistoryPrefetch("cpu")
    ticket = helper.prefetch(cache)
    helper.wait(ticket)
    assert (ticket.requested_records, ticket.resident_records, ticket.fetched_records) == (8, 8, 0)
    physical = cache.ensure(torch.arange(8))
    torch.testing.assert_close(cache.records[physical.long()], history)
    assert cache.metrics()["host_to_device_bytes"] == 0
    helper.close()


def test_proven_all_hit_prefetch_omits_allocator_and_host_wait(monkeypatch):
    pool = make_pool()
    cache = pool.allocate_session(8).layer(0)
    populate(cache, 4)
    clock = cache._clock
    helper = PoolHistoryPrefetch("cpu")

    def unexpected(*args, **kwargs):
        raise AssertionError("proven all-hit prefetch must not allocate or wait for host")

    monkeypatch.setattr(cache, "_available_slots", unexpected)
    monkeypatch.setattr(pool, "wait_host", unexpected)
    ticket = helper.prefetch(cache)
    assert ticket._ready is None
    assert (ticket.requested_records, ticket.resident_records, ticket.fetched_records) == (4, 4, 0)
    helper.wait(ticket)
    assert cache._clock == clock + 2
    physical = cache.host_to_device[:4].long()
    assert cache.age[physical].tolist() == [clock] * 4
    assert pool.layers[0].append_owner == cache.session.owner
    helper.close()


def test_prefill_reads_only_initialized_history_and_allows_empty_first_chunk():
    pool = make_pool()
    cache = pool.allocate_session(8).layer(0)
    helper = PoolHistoryPrefetch("cpu")
    cache.begin_step(8)
    empty = helper.prefetch(cache)
    helper.wait(empty)
    assert empty.requested_records == empty.fetched_records == 0
    cache.append(torch.ones(4, 7))
    partial = helper.prefetch(cache)
    helper.wait(partial)
    assert partial.requested_records == partial.resident_records == 4
    assert partial.fetched_records == 0
    helper.drain()
    cache.rollback()
    helper.close()


def test_lookahead_tickets_are_layer_specific_and_expire_after_drain():
    pool = make_pool(layers=2)
    session = pool.allocate_session(8)
    other = pool.allocate_session(8)
    histories = [populate(session.layer(layer), 8, layer * 100) for layer in range(2)]
    for layer in range(2):
        populate(other.layer(layer), 8, 1000 + layer * 100)
    helper = PoolHistoryPrefetch("cpu")
    first = helper.prefetch(session.layer(0))
    helper.wait(first)
    second = helper.prefetch(session.layer(1))
    for layer, ticket in enumerate((first, second)):
        helper.wait(ticket)
        physical = session.layer(layer).ensure(torch.arange(8))
        torch.testing.assert_close(session.layer(layer).records[physical.long()], histories[layer])
        assert ticket.fetched_records == 8
    helper.drain()
    with pytest.raises(ValueError, match="expired"):
        helper.wait(second)
    helper.close()


def test_oversized_history_rejects_before_changing_metadata():
    pool = make_pool()
    cache = pool.allocate_session(9).layer(0)
    populate(cache, 9)
    before = cache.host_to_device.clone(), cache.device_to_host.clone()
    helper = PoolHistoryPrefetch("cpu")
    with pytest.raises(WorkingSetTooLarge, match="H <= P"):
        helper.prefetch(cache)
    torch.testing.assert_close(cache.host_to_device, before[0])
    torch.testing.assert_close(cache.device_to_host, before[1])
    helper.close()


def test_unconsumed_and_foreign_tickets_are_rejected():
    pool = make_pool()
    cache = pool.allocate_session(8).layer(0)
    populate(cache, 8)
    helper = PoolHistoryPrefetch("cpu")
    other = PoolHistoryPrefetch("cpu")
    ticket = helper.prefetch(cache)
    with pytest.raises(RuntimeError, match="prior history prefetch"):
        helper.prefetch(cache)
    with pytest.raises(ValueError, match="foreign"):
        other.wait(ticket)
    helper.drain()  # Also covers a submitted lookahead whose consumer was never reached.
    helper.close()
    other.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_prefetch_on_private_stream_joins_before_mixed_history_candidate_read():
    pool = make_pool(device="cuda", layers=2)
    session = pool.allocate_session(8)
    other = pool.allocate_session(8)
    caches = [session.layer(layer) for layer in range(2)]
    histories = [populate(cache, 8, layer * 100) for layer, cache in enumerate(caches)]
    for layer in range(2):
        populate(other.layer(layer), 8, 1000 + layer * 100)
    helper = PoolHistoryPrefetch("cuda")
    caller = torch.cuda.Stream()
    with torch.cuda.stream(caller):
        candidate = torch.arange(21, dtype=torch.float32, device="cuda").reshape(3, 7) + 2000
        for cache in caches:
            cache.reset_stats()
            cache.begin_transient(3)
            cache.append(candidate)
        first = helper.prefetch(caches[0])
        helper.wait(first)
        second = helper.prefetch(caches[1])
        observed = []
        for cache, ticket in zip(caches, (first, second), strict=True):
            helper.wait(ticket)
            physical = cache.ensure(torch.arange(11, device="cuda"))
            observed.append(cache.records[physical.long()].clone())
    caller.synchronize()
    helper.drain()
    for cache, ticket, result, history in zip(
        caches, (first, second), observed, histories, strict=True
    ):
        torch.testing.assert_close(result, torch.cat((history, candidate)))
        assert ticket.requested_records == ticket.fetched_records == 8
        assert cache.stats.recalled_records == 8
        assert cache.metrics()["device_to_host_bytes"] == 0
        cache.discard_transient()
    helper.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_dense_native_protection_keeps_fifo_events_and_append_order():
    from operators.deepseek_v32.indexer import cache_ops

    assert torch.cuda.get_device_capability()[0] == 9, "requires Hopper SM90"
    checked = SimpleNamespace(
        **{
            name: getattr(cache_ops, name)
            for name in ("protect", "mark_misses", "release_ids", "finalize_prefetch")
        }
    )
    pools = [make_pool(slots=193, device="cuda") for _ in range(2)]
    for pool, ops in zip(pools, (cache_ops, checked), strict=True):
        pool.metadata_ops = ops
    helpers = [PoolHistoryPrefetch("cuda") for _ in pools]
    caches = [pool.allocate_session(193).layer(0) for pool in pools]

    def compare():
        layers = [pool.layers[0] for pool in pools]
        for name in (
            "records",
            "host_to_device",
            "device_to_host",
            "priority",
            "free",
            "clock_tensor",
        ):
            torch.testing.assert_close(
                getattr(layers[0], name), getattr(layers[1], name), rtol=0, atol=0
            )
        assert layers[0].clock == layers[1].clock
        assert caches[0].metrics() == caches[1].metrics()

    try:
        for cache in caches:
            cache.begin_step(193)
        for start, end in ((0, 3), (3, 71), (71, 193)):
            records = torch.full((end - start, 7), float(start), device="cuda")
            for cache, helper in zip(caches, helpers, strict=True):
                cache.append(records)
                ticket = helper.prefetch(cache)
                assert ticket.fetched_records == 0
                helper.wait(ticket)
                helper.drain()
            compare()
            assert pools[0].layers[0].append_cursor == pools[1].layers[0].append_cursor == end
        for clock in (PRIORITY_LIMIT - 2, PRIORITY_LIMIT - 1, PRIORITY_LIMIT):
            for pool, cache, helper in zip(pools, caches, helpers, strict=True):
                pool.layers[0].clock = clock
                helper.wait(helper.prefetch(cache))
                helper.drain()
            compare()
        for cache in caches:
            cache.rollback()
        compare()
    finally:
        for helper in helpers:
            helper.close()
        for pool in pools:
            pool.close()
