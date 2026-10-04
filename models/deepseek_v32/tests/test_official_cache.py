"""Original ECHO allocator/recall with shared history and GPU-only candidates."""

import pytest
import torch

from cache.sparse_token_pool import SharedSparseTokenPool
from models.deepseek_v32.official_cache import OfficialCacheState


def _require_hopper():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("official ECHO cache validation requires SM90")


def _populate(state, count, value):
    session = state.pool.allocate_session(count)
    cache = state.layer_cache(session.layer(0))
    records = torch.full((count, 576), value, device="cuda", dtype=torch.bfloat16)
    cache.begin_step(count)
    with cache.operation():
        cache.declare_indexer_visible(count)
        cache.append(records)
    cache.commit()
    return session, cache, records


def test_official_recall_host_zero_eviction_and_candidate_discard():
    _require_hopper()
    pool = SharedSparseTokenPool(384, 576, 1, 128, candidate_slots=16)
    state = OfficialCacheState(pool)
    assert state.shared_bytes() == state.estimate_extra_bytes(384, 128, 1)
    first, cache, history = _populate(state, 128, 1.25)
    second, _, _ = _populate(state, 128, -2.0)
    torch.testing.assert_close(cache.host_records(), history.cpu(), rtol=0, atol=0)
    with cache.operation():
        indices = torch.arange(128, device="cuda", dtype=torch.int32).unsqueeze(0)
        physical = cache.ensure(indices)
        torch.testing.assert_close(
            cache.records[physical.long()].reshape_as(history), history, rtol=0, atol=0
        )
    assert cache.metrics()["recalled_records"] == 128
    cache.reset_stats()
    cache.begin_transient(16)
    candidate = torch.full((16, 576), 3.5, device="cuda", dtype=torch.bfloat16)
    with cache.operation():
        cache.append(candidate)
        indices = torch.tensor([[0, 127, 128, 143, -1]], device="cuda", dtype=torch.int32)
        physical = cache.ensure(indices)
        assert physical[0, 2:].tolist() == [129, 144, -1]
        torch.testing.assert_close(
            cache.records[physical[0, :4].long()],
            torch.cat((history[[0, 127]], candidate[[0, 15]])),
            rtol=0,
            atol=0,
        )
    cache.discard_transient()
    assert cache.length == cache.written == 128
    assert cache.metrics()["device_to_host_bytes"] == 0
    torch.testing.assert_close(cache.host_records(), history.cpu(), rtol=0, atol=0)
    first.release()
    second.release()
    assert state.device_pool_allocator[0].available_size().item() == 128
    state.close()
    pool.close()


def test_official_fused_prefetch_then_original_residual_recall():
    _require_hopper()
    from operators.deepseek_v32.indexer.official import module

    pool = SharedSparseTokenPool(512, 576, 1, 256, candidate_slots=128)
    state = OfficialCacheState(pool)
    first, cache, history = _populate(state, 256, 1.0)
    second, _, _ = _populate(state, 256, -1.0)
    cache.reset_stats()
    cache.begin_transient(128)
    candidate = torch.full((128, 576), 2.0, device="cuda", dtype=torch.bfloat16)
    with cache.operation():
        cache.declare_indexer_visible(384)
        prefetch = cache.prepare_prefetch(256, 128)
        q = torch.randn(128, 64, 128, device="cuda").to(torch.float8_e4m3fn)
        k = torch.randn(384, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(384, device="cuda")
        weights = torch.rand(128, 64, device="cuda")
        starts = torch.zeros(128, device="cuda", dtype=torch.int32)
        ends = torch.arange(257, 385, device="cuda", dtype=torch.int32)
        logits = module().fp8_mqa_logits_fuse_prefetch(
            q,
            (k, scales),
            weights,
            starts,
            ends,
            **prefetch["arguments"],
            clean_logits=False,
        )
        cache.finalize_prefetch(prefetch, logits=logits)
        cache.append(candidate)
        indices = torch.arange(384, device="cuda", dtype=torch.int32).unsqueeze(0)
        physical = cache.ensure(indices)
        torch.testing.assert_close(
            cache.records[physical.long()].squeeze(0),
            torch.cat((history, candidate)),
            rtol=0,
            atol=0,
        )
    metrics = cache.metrics()
    assert metrics["prefetched_records"] + metrics["recalled_records"] == 256
    assert metrics["host_to_device_bytes"] == 256 * 1152
    assert metrics["device_to_host_bytes"] == 0
    cache.discard_transient()
    first.release()
    second.release()
    state.close()
    pool.close()
