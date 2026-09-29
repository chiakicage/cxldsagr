"""Model-owned transactions around the checked native ranked indexer."""

import pytest
import torch

from layers.attention import AttentionContext
from models.nosa.cache import NosaKVCache
from models.nosa.config import NosaConfig
from models.nosa.indexer import NosaIndexer
from operators.sm90 import _nosa_indexer_checked_cuda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Hopper CUDA required")


def fixture(rows):
    torch.manual_seed(875)
    prefix, total = 32768, 32768 + rows
    cfg = NosaConfig(
        hidden_size=4096,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=32,
        max_position_embeddings=total,
    )
    cache = NosaKVCache(cfg, total, device="cuda", dtype=torch.bfloat16, with_cis=True)
    q = torch.randn((rows, 32, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total, 2, 128), device="cuda", dtype=torch.bfloat16)
    cis = torch.randn((total, 2), device="cuda", dtype=torch.bfloat16)
    indexer = NosaIndexer(mode="nosa", backend="triton")
    cache.begin_step(prefix)
    cache.write_layer(0, keys=k[:prefix], values=k[:prefix], cis_scores=cis[:prefix])
    indexer(q[:1], cache, AttentionContext(0, prefix - 1, 1))
    cache.commit_step()
    return cache, indexer, q, k, cis, AttentionContext(0, prefix, rows)


def begin(cache, k, cis, context):
    cache.begin_step(context.query_length)
    cache.write_layer(
        0,
        keys=k[context.query_start :],
        values=k[context.query_start :],
        cis_scores=cis[context.query_start :],
    )


@pytest.mark.parametrize("rows", [64, 128, 1024])
@torch.inference_mode()
def test_cuda_checked_dispatch_keeps_fallback_and_publishes_only_via_model(monkeypatch, rows):
    cache, indexer, q, k, cis, context = fixture(rows)
    original = _nosa_indexer_checked_cuda.select_prepared_out
    calls = []

    def checked(*args, **kwargs):
        calls.append(1)
        assert cache.indexer_cache._reservations
        return original(*args, **kwargs)

    monkeypatch.setattr(_nosa_indexer_checked_cuda, "select_prepared_out", checked)
    try:
        begin(cache, k, cis, context)
        selected = indexer(q, cache, context)
        torch.cuda.synchronize()
        assert selected.block_ids.shape == (rows, 2, 64)
        assert len(calls) == int(rows >= 128)
        assert not cache.indexer_cache._reservations
        assert cache.length == context.query_start
        assert cache.indexer_cache.layer_state(0).validated_tokens == len(k)
        cache.commit_step()
        assert cache.length == len(k)
    finally:
        cache.release()
    assert cache._native_indexer_host_flag is None


@pytest.mark.parametrize("field", ["q", "keys", "cis_scores"])
@torch.inference_mode()
def test_cuda_checked_failure_aborts_reservation_and_preserves_committed_prefix(field):
    cache, indexer, q, k, cis, context = fixture(1024)
    try:
        state = cache.indexer_cache.layer_state(0)
        begin(cache, k, cis, context)
        expected = indexer(q, cache, context)
        torch.cuda.synchronize()
        expected_ids, expected_valid = expected.block_ids.clone(), expected.valid_mask.clone()
        cache.abort_step()
        before = {
            name: value.view(torch.uint8).clone()
            for name, value in cache.indexer_cache._buffers[0].items()
        }
        begin(cache, k, cis, context)
        tensor = q if field == "q" else cache.layer_view(0)[field]
        index = (-1,) * tensor.ndim
        saved = tensor[index].clone()
        tensor[index] = torch.nan
        with pytest.raises(ValueError, match="finite Q, K and CIS"):
            indexer(q, cache, context)
        assert cache.length == context.query_start
        assert cache.indexer_cache.layer_state(0) == state
        assert not cache.indexer_cache._reservations
        for name, value in cache.indexer_cache._buffers[0].items():
            torch.testing.assert_close(value.view(torch.uint8), before[name], rtol=0, atol=0)
        tensor[index] = saved
        cache.abort_step()
        begin(cache, k, cis, context)
        actual = indexer(q, cache, context)
        torch.testing.assert_close(actual.block_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual.valid_mask, expected_valid, rtol=0, atol=0)
        cache.commit_step()
    finally:
        cache.release()
