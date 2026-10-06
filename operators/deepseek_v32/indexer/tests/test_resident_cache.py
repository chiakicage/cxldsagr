"""Native resident metadata differentials against checked cache operations."""

import pytest
import torch

from cache.sparse_token_pool import PRIORITY_LIMIT, SharedSparseTokenPool
from models.deepseek_v32.cache.prefetch import PoolHistoryPrefetch
from operators.deepseek_v32.indexer import cache_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def paired_pools(
    *, slots=96, candidate=129, width=7, dtype=torch.bfloat16, pages=8, dense_contiguous=False
):
    pools = [
        SharedSparseTokenPool(
            pages * 64,
            width,
            1,
            slots,
            candidate_slots=candidate,
            dtype=dtype,
            metadata_ops=native,
            dense_contiguous=dense_contiguous,
        )
        for native in (cache_ops, None)
    ]
    # Full-storage comparisons require defined contents in unused slots too.
    for pool in pools:
        pool.layers[0].records.zero_()
    return pools


def compare(pools, caches):
    for name in ("host_to_device", "device_to_host", "priority", "free", "records"):
        torch.testing.assert_close(
            getattr(pools[0].layers[0], name), getattr(pools[1].layers[0], name), rtol=0, atol=0
        )
    assert caches[0]._clock == caches[1]._clock
    assert pools[0].layers[0].clock_tensor.item() == pools[1].layers[0].clock_tensor.item()
    assert caches[0].metrics() == caches[1].metrics()
    # A repeated read must not fold the native totals into Python totals again.
    assert caches[0].metrics() == caches[1].metrics()


def append_pair(caches, values):
    for cache in caches:
        cache.append(values)


def consume_pair(caches, indices, *, checked=False):
    results = [
        cache.ensure(indices) if checked else cache._ensure_from_topk(indices) for cache in caches
    ]
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    return results


def protect_resident_history(cache, helper):
    if helper is not None:
        ticket = helper.prefetch(cache)
        helper.wait(ticket)
        return ticket.fetched_records
    # The checked cache provider has no CUDA DMA adapter. Exercise its two
    # public FIFO events for this certified all-hit history instead.
    assert cache.all_history_resident
    with cache.operation():
        pool = cache._pool
        ids = cache._global_range(0, cache.host_written_end)
        pool.protect(cache.layer_id, ids, preserve_append_plan=True)
        pool.stamp(
            cache.layer_id, pool.layers[cache.layer_id].append_order[:0], preserve_append_plan=True
        )
    return 0


@pytest.mark.parametrize(
    "width,dtype", [(7, torch.bfloat16), (19, torch.float32), (576, torch.bfloat16)]
)
def test_cuda_native_append_selection_and_dense_clock_match_checked_path(width, dtype, monkeypatch):
    pools = paired_pools(width=width, dtype=dtype, dense_contiguous=True)
    helpers = [PoolHistoryPrefetch("cuda"), None]
    for user in range(3):
        caches = [pool.allocate_session(64).layer(0) for pool in pools]
        values = (
            torch.arange(64 * width, device="cuda", dtype=torch.float32).reshape(64, width) % 251
            + user * 300
        ).to(dtype)
        for cache in caches:
            cache.begin_step(64)
        for start, stop in ((0, 13), (13, 44), (44, 64)):
            for cache, helper in zip(caches, helpers, strict=True):
                # The native branch must not materialize the logical range.
                with monkeypatch.context() as patch:
                    if cache is caches[0]:
                        patch.setattr(
                            cache,
                            "_global_range",
                            lambda *args: (_ for _ in ()).throw(
                                AssertionError("range materialized")
                            ),
                        )
                    fetched = protect_resident_history(cache, helper)
                assert fetched == 0
            append_pair(caches, values[start:stop])
            indices = (torch.arange(256 * 17, device="cuda").reshape(256, 17) % stop).int()
            indices[:, 3] = -7
            consume_pair(caches, indices)
            consume_pair(caches, indices[:2], checked=True)
            compare(pools, caches)
        for cache, helper in zip(caches, helpers, strict=True):
            cache.commit()
            if helper is not None:
                helper.drain()
            torch.testing.assert_close(cache.host_records(), values.cpu(), rtol=0, atol=0)
        candidate = torch.full((129, width), 900.0 + user, device="cuda", dtype=dtype)
        for cache in caches:
            cache.reset_stats()
            cache.begin_transient(129)
            cache.append(candidate)
        indices = (torch.arange(37 * 73, device="cuda").reshape(37, 73) % 193).int()
        indices[:, 2] = -1
        consume_pair(caches, indices)
        consume_pair(caches, torch.full((3, 7), -1, device="cuda", dtype=torch.int32))
        consume_pair(caches, torch.empty((0, 7), device="cuda", dtype=torch.int32))
        compare(pools, caches)
        assert caches[0].metrics()["selection_records"] == 193
        assert caches[0].metrics()["max_working_set"] == 193
        for cache in caches:
            cache.discard_transient()
        compare(pools, caches)
    for helper in helpers:
        if helper is not None:
            helper.close()
    for pool in pools:
        pool.close()


@pytest.mark.parametrize("clock", [PRIORITY_LIMIT - 1, PRIORITY_LIMIT, PRIORITY_LIMIT + 1])
def test_cuda_native_event_rollover_and_snapshot_reset(clock):
    pools = paired_pools(dense_contiguous=True)
    caches = [pool.allocate_session(64).layer(0) for pool in pools]
    for cache in caches:
        cache.begin_step(64)
        cache.append(torch.ones((64, 7), device="cuda", dtype=torch.bfloat16))
        cache.commit()
    for cache in caches:
        cache._clock = clock
    ids = torch.tensor([[0, 31, 32, 63, 63, -1]], device="cuda", dtype=torch.int32)
    consume_pair(caches, ids)
    compare(pools, caches)
    for cache in caches:
        cache._clock = clock
    helpers = [PoolHistoryPrefetch("cuda"), None]
    for helper, cache in zip(helpers, caches, strict=True):
        assert protect_resident_history(cache, helper) == 0
        if helper is not None:
            helper.close()
    compare(pools, caches)
    snapshots = [pool.snapshot() for pool in pools]
    consume_pair(caches, ids)
    for pool, snapshot in zip(pools, snapshots, strict=True):
        pool.restore(snapshot)
    consume_pair(caches, ids)  # Restore clears the certificate; native and checked recalls agree.
    compare(pools, caches)
    for pool in pools:
        pool.close()


def test_cuda_native_fragmented_pages_and_cross_stream_workspace_reuse():
    pools = paired_pools(slots=192, pages=6)
    caches = []
    for pool in pools:
        _, hole, _ = [pool.allocate_session(64) for _ in range(3)]
        hole.release()
        session = pool.allocate_session(130)
        assert session._pages.tolist() == [1, 3, 4]
        caches.append(session.layer(0))
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    values = torch.arange(129 * 7, device="cuda").reshape(129, 7).bfloat16()
    with torch.cuda.stream(first):
        for cache in caches:
            cache.begin_step(129)
        for start, end in ((0, 13), (13, 90), (90, 129)):
            append_pair(caches, values[start:end])
        ids = torch.arange(129, device="cuda", dtype=torch.int32).view(3, 43)
        physical = consume_pair(caches, ids)
        torch.cuda._sleep(2_000_000)
        observed = [
            cache.records[p.long()].clone() for cache, p in zip(caches, physical, strict=True)
        ]
    with torch.cuda.stream(second):
        consume_pair(caches, ids.flip(1).contiguous())
        for cache in caches:
            cache.commit()
    second.synchronize()
    compare(pools, caches)
    torch.testing.assert_close(observed[0], observed[1], rtol=0, atol=0)
    for cache in caches:
        torch.testing.assert_close(cache.host_records(), values.cpu(), rtol=0, atol=0)
        cache.begin_step(1)
        cache.append(torch.zeros((1, 7), device="cuda", dtype=torch.bfloat16))
        cache.rollback()
        cache.reset_stats()
        assert cache.metrics()["selection_records"] == cache.metrics()["evicted_records"] == 0
    for pool in pools:
        pool.close()


def test_cuda_native_fullshape_union_and_shared_allocation_accounting():
    pools = paired_pools(slots=65536, candidate=128, width=7, pages=1024)
    caches = [pool.allocate_session(65536).layer(0) for pool in pools]
    for cache in caches:
        cache.begin_step(65536)
        cache.append(torch.zeros((65536, 7), device="cuda", dtype=torch.bfloat16))
        cache.commit()
        cache.reset_stats()
    indices = (torch.arange(1024 * 2048, device="cuda").reshape(1024, 2048) % 65536).int()
    consume_pair(caches, indices)
    compare(pools, caches)
    assert caches[0].metrics()["selection_records"] == 65536
    indices.fill_(31)
    consume_pair(caches, indices)
    compare(pools, caches)
    assert caches[0].metrics()["selection_records"] == 65537
    assert caches[0].metrics()["max_working_set"] == 65536
    for pool in pools:
        assert pool.shared_bytes() == pool.estimate_shared_bytes(
            65536, 7, 1, 65536, candidate_slots=128
        )
        assert pool.union_bitmap.numel() * 4 + pool.union_count.numel() * 4 == 8216
        session = next(iter(pool._sessions.values()))
        assert session.session_bytes() == pool.estimate_session_bytes(65536)
        assert session._prefetch_totals.shape == (1, 3)
        assert session._prefetch_totals.untyped_storage().nbytes() == 8 * 8
        pool.close()


def test_cuda_native_empty_history_candidate_and_checked_miss_fallback(monkeypatch):
    pools = paired_pools(slots=4, candidate=129)
    caches = [pool.allocate_session(64).layer(0) for pool in pools]
    for cache in caches:
        cache.begin_transient(129)
        cache.append(torch.ones((129, 7), device="cuda", dtype=torch.bfloat16))
    indices = (torch.arange(32 * 129, device="cuda").reshape(32, 129) % 129).int()
    consume_pair(caches, indices)
    compare(pools, caches)
    assert caches[0].metrics()["max_working_set"] == 129
    assert caches[0].metrics()["host_written_records"] == 0
    for cache in caches:
        cache.discard_transient()
        cache.reset_stats()
        cache.begin_step(4)
        cache.append(torch.ones((4, 7), device="cuda", dtype=torch.bfloat16))
        cache.commit()
    for pool in pools:
        other = pool.allocate_session(4).layer(0)
        other.begin_step(4)
        other.append(torch.full((4, 7), 20.0, device="cuda", dtype=torch.bfloat16))
        other.commit()
    assert not caches[0].all_history_resident

    def unexpected(*args, **kwargs):
        raise AssertionError("a missing-history cache must use checked recall")

    monkeypatch.setattr(cache_ops, "resident_selection", unexpected)
    indices = torch.tensor([[0, 1, -1]], device="cuda", dtype=torch.int32)
    consume_pair(caches, indices)
    compare(pools, caches)
    assert caches[0].metrics()["recalled_records"] == 2
    before = caches[0].metrics()
    with pytest.raises(ValueError, match="unwritten"):
        caches[0].ensure(torch.tensor([[4]], device="cuda", dtype=torch.int32))
    assert caches[0].metrics() == before
    for pool in pools:
        pool.close()
