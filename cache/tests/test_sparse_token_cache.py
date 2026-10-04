import pytest
import torch

from cache.sparse_token_cache import MISSING, SparseTokenCache, WorkingSetTooLarge

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_cuda_exact_recall_eviction_and_step_rollback():
    cache = SparseTokenCache(12, 16, slots=4)
    source = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    cache.begin_step(8)
    cache.append(source)
    with pytest.raises(RuntimeError, match="already active"):
        cache.begin_step(1)
    cache.commit()
    for selection in ([0, 3, 2], [3, 6, 7], [-1, 0, 6]):
        logical = torch.tensor([selection], device="cuda", dtype=torch.int32)
        physical = cache.ensure(logical)
        valid = logical >= 0
        torch.testing.assert_close(
            cache.records[physical[valid].long()], source[logical[valid].long()], rtol=0, atol=0
        )
        assert (physical[~valid] == -1).all()
    cache.begin_step(2)
    cache.append(torch.ones(2, 16, device="cuda", dtype=torch.bfloat16))
    cache.ensure(torch.tensor([8, 9], device="cuda"))
    cache.rollback()
    assert cache.length == cache.written == 8
    assert (cache.host_to_device[8:] == MISSING).all()
    assert ((cache.device_to_host < 8) | (cache.device_to_host == MISSING)).all()
    with pytest.raises(ValueError, match="unwritten"):
        cache.ensure(torch.tensor([8], device="cuda"))
    with pytest.raises(WorkingSetTooLarge):
        cache.ensure(torch.arange(5, device="cuda"))


@cuda
def test_cuda_prefetch_prepare_keeps_history_and_runs_before_current_append():
    cache = SparseTokenCache(16, 16, slots=8)
    cache.begin_step(12)
    source = torch.randn(12, 16, device="cuda", dtype=torch.bfloat16)
    cache.append(source[:10])
    cache.declare_indexer_visible(12)
    before = cache.host_to_device.clone()
    prefetch = cache.prepare_prefetch(10, 2, torch.zeros(16, device="cuda"), limit=4)
    torch.testing.assert_close(cache.host_to_device, before)
    assert (cache.host_to_device[10:12] == MISSING).all()
    assert prefetch["free_slots"].numel() == cache.slots
    assert not (prefetch["free_slots"] == 0).any()
    cache.finalize_prefetch()
    cache.append(source[10:])
    physical = cache.ensure(torch.arange(10, 12, device="cuda"))
    torch.testing.assert_close(cache.records[physical.long()], source[10:12], rtol=0, atol=0)
    cache.commit()
    cache.truncate(5)
    assert (cache.host_to_device[5:] == MISSING).all()


@cuda
def test_cuda_generic_record_width_and_dtype():
    cache = SparseTokenCache(8, 19, slots=4, dtype=torch.float32)
    x = torch.arange(8 * 19, dtype=torch.float32, device="cuda").view(8, 19)
    cache.begin_step(8)
    cache.append(x)
    cache.commit()
    ids = torch.tensor([7, 0, 3], device="cuda")
    physical = cache.ensure(ids)
    torch.testing.assert_close(cache.records[physical.long()], x[ids], rtol=0, atol=0)


@cuda
def test_cuda_recall_respects_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cache = SparseTokenCache(64, 576, slots=16)
        cache.begin_step(64)
        x = torch.randn(64, 576, device="cuda", dtype=torch.bfloat16)
        cache.append(x)
        physical = cache.ensure(torch.arange(16, device="cuda"))
        observed = cache.records[physical.long()].clone()
        cache.commit()
    stream.synchronize()
    torch.testing.assert_close(observed, x[:16], rtol=0, atol=0)
