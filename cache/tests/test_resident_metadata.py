"""Certified native metadata must match checked finite-pool operations exactly."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_pool import PRIORITY_LIMIT, SharedSparseTokenPool
from operators.deepseek_v32.indexer import cache_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _pools(*, width=7, dtype=torch.float32, slots=193):
    assert torch.cuda.get_device_capability()[0] == 9, "requires Hopper SM90"
    checked = SimpleNamespace(
        **{
            name: getattr(cache_ops, name)
            for name in ("protect", "mark_misses", "release_ids", "finalize_prefetch")
        }
    )
    return [
        SharedSparseTokenPool(
            1024,
            width,
            2,
            slots,
            device="cuda",
            dtype=dtype,
            candidate_slots=35,
            metadata_ops=ops,
        )
        for ops in (cache_ops, checked)
    ]


def _equal(pools, caches):
    for layer_id in range(2):
        layers = [pool.layers[layer_id] for pool in pools]
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


@pytest.mark.parametrize("clock", [0, PRIORITY_LIMIT - 1, PRIORITY_LIMIT])
@pytest.mark.parametrize("preserve_append_plan", [False, True])
def test_cuda_empty_fifo_event_preserves_reference_state_and_rollover(clock, preserve_append_plan):
    pools = _pools(slots=8)
    try:
        caches = []
        for pool in pools:
            cache = pool.allocate_session(8).layer(0)
            cache.begin_step(8)
            cache.append(torch.ones(8, 7, device="cuda"))
            cache.commit()
            pool.layers[0].clock = clock
            pool.layers[0].clock_tensor.fill_(clock)
            with cache.operation():
                pool.stamp(
                    0,
                    pool.layers[0].append_order[:0],
                    preserve_append_plan=preserve_append_plan,
                )
            caches.append(cache)
        _equal(pools, caches)
        for pool, cache in zip(pools, caches, strict=True):
            layer = pool.layers[0]
            assert layer.clock == (2 if clock == PRIORITY_LIMIT else clock + 1)
            assert layer.append_owner == (cache.session.owner if preserve_append_plan else None)
            assert layer.resident_owner == cache.session.owner and layer.resident_end == 8
    finally:
        for pool in pools:
            pool.close()


def test_cuda_empty_fifo_event_failure_preserves_clock_and_error_identity(monkeypatch):
    pools = _pools(slots=8)
    try:
        pool = pools[0]
        cache = pool.allocate_session(8).layer(0)
        cache.begin_step(8)
        cache.append(torch.ones(8, 7, device="cuda"))
        cache.commit()
        layer = pool.layers[0]
        before = layer.clock, layer.clock_tensor.clone(), layer.priority.clone()
        failure = RuntimeError("injected empty FIFO launch failure")
        calls = []

        def fail(*args, **kwargs):
            calls.append(kwargs["timestamp"])
            raise failure

        monkeypatch.setattr(cache_ops, "empty_event", fail)
        with pytest.raises(RuntimeError) as caught, cache.operation():
            pool.stamp(0, layer.append_order[:0], preserve_append_plan=True)
        assert caught.value is failure and calls == [before[0]]
        assert layer.clock == before[0] and layer.append_owner == cache.session.owner
        torch.testing.assert_close(layer.clock_tensor, before[1], rtol=0, atol=0)
        torch.testing.assert_close(layer.priority, before[2], rtol=0, atol=0)
    finally:
        for pool in pools:
            pool.close()


@pytest.mark.parametrize(
    "dtype,width", [(torch.uint8, 7), (torch.float32, 7), (torch.bfloat16, 576)]
)
def test_cuda_cold_append_and_exact_union_match_checked_with_fragmented_pages(dtype, width):
    pools = _pools(width=width, dtype=dtype)
    try:
        # Leave holes so each owner has a noncontiguous host page table.
        for pool in pools:
            allocated = [pool.allocate_session(64) for _ in range(4)]
            allocated[0].release()
            allocated[2].release()
        rng = torch.Generator().manual_seed(821)
        for owner in range(3):
            sessions = [pool.allocate_session(193) for pool in pools]
            if owner == 0:
                assert len(sessions[0]._host_runs) > 1
            for layer_id in range(2):
                caches = [session.layer(layer_id) for session in sessions]
                data = (torch.arange(193 * width).reshape(193, width) % 197 + owner + layer_id).to(
                    device="cuda", dtype=dtype
                )
                for cache in caches:
                    cache.begin_step(193)
                for start, end in ((0, 63), (63, 134), (134, 193)):
                    ids = torch.randint(-3, end, (13, 47), generator=rng, dtype=torch.int32).cuda()
                    ids[:, :3] = 0
                    physical = []
                    for cache in caches:
                        cache.append(data[start:end])
                        physical.append(cache._ensure_from_topk(ids))
                    torch.testing.assert_close(*physical, rtol=0, atol=0)
                    _equal(pools, caches)
                for cache in caches:
                    cache.commit()
                for clock in (PRIORITY_LIMIT - 2, PRIORITY_LIMIT - 1, PRIORITY_LIMIT):
                    ids = torch.tensor(
                        [[0, 17, 17, -1, -7], [192, -1, 192, 0, 99]],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    for pool in pools:
                        pool.layers[layer_id].clock = clock
                    physical = [cache._ensure_from_topk(ids) for cache in caches]
                    torch.testing.assert_close(*physical, rtol=0, atol=0)
                    _equal(pools, caches)
                for cache in caches:
                    cache.reset_stats()
                    cache.begin_transient(35)
                    cache.append(data[:35])
                for ids in (
                    torch.full((3, 19), -1, device="cuda", dtype=torch.int32),
                    torch.empty((0, 17), device="cuda", dtype=torch.int32),
                    torch.randint(-2, 228, (31, 131), generator=rng, dtype=torch.int32).cuda(),
                ):
                    physical = [cache._ensure_from_topk(ids) for cache in caches]
                    torch.testing.assert_close(*physical, rtol=0, atol=0)
                    _equal(pools, caches)
                for cache in caches:
                    cache.discard_transient()
                _equal(pools, caches)
    finally:
        for pool in pools:
            pool.close()


def test_cuda_topk_falls_back_after_competing_user_eviction_and_public_ids_stay_checked():
    pools = _pools(slots=8)
    try:
        pairs = []
        for pool in pools:
            a, b = (pool.allocate_session(8).layer(0) for _ in range(2))
            for value, cache in enumerate((a, b)):
                cache.begin_step(8)
                cache.append(torch.full((8, 7), float(value), device="cuda"))
                cache.commit()
            assert not a.all_history_resident
            pairs.append((a, b))
        ids = torch.tensor([[0, -1, 0, 7], [1, 2, 3, 4]], device="cuda", dtype=torch.int32)
        for index in (0, 1, 0):
            caches = [pair[index] for pair in pairs]
            outputs = [cache._ensure_from_topk(ids) for cache in caches]
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
            _equal(pools, caches)
        for pair in pairs:
            with pytest.raises(ValueError, match="unwritten"):
                pair[0].ensure(torch.tensor([[8]], device="cuda", dtype=torch.int32))
    finally:
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("pattern", ["random", "duplicate", "padding"])
@pytest.mark.parametrize("queries", [128, 1024])
def test_cuda_resident_union_large_batch_matches_checked_across_cta_merges(pattern, queries):
    # Large selections use CTA-local unions; all CTAs can contend for one bit.
    pools = _pools(width=1, slots=65536)
    try:
        caches = [pool.allocate_session(128).layer(0) for pool in pools]
        for cache in caches:
            cache.begin_step(128)
            cache.append(torch.ones(128, 1, device="cuda"))
            cache.commit()
            cache.begin_transient(35)
            cache.append(torch.zeros(35, 1, device="cuda"))
        ids = torch.randint(-5, 163, (queries, 2048), device="cuda", dtype=torch.int32)
        if pattern == "duplicate":
            ids.fill_(128)
            ids[:, :128] = 3
        elif pattern == "padding":
            ids.fill_(-1)
        for _ in range(3):
            outputs = [cache._ensure_from_topk(ids) for cache in caches]
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
            _equal(pools, caches)
    finally:
        for pool in pools:
            pool.close()
