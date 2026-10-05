"""Owned model transactions keep eager dispatch and defer only guarded prefixes."""

import pytest
import torch

from models.attention_contracts import AttentionContext
from models.nosa.cache.resident import NosaKVCache
from models.nosa.config import NosaConfig
from models.nosa.execution.deferred_validation import DeferredValidation
from models.nosa.indexer import NosaIndexer
from operators.nosa.indexer import _indexer_checked_cuda, _indexer_deferred_cuda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Hopper CUDA required")


def inputs(length, rows):
    torch.manual_seed(1312)
    q = torch.randn((rows, 32, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((length, 2, 128), device="cuda", dtype=q.dtype)
    cis = torch.randn((length, 2), device="cuda", dtype=q.dtype)
    return q, k, cis


def independent_prefix(q, k, cis, prefix):
    config = NosaConfig(
        hidden_size=4096,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=32,
        max_position_embeddings=len(k),
    )
    cache = NosaKVCache(config, len(k), device="cuda", dtype=q.dtype, with_cis=True)
    indexer = NosaIndexer(mode="nosa", backend="triton")
    if prefix:
        cache.begin_step(prefix)
        cache.write_layer(0, keys=k[:prefix], values=k[:prefix], cis_scores=cis[:prefix])
        indexer(q[:1], cache, AttentionContext(0, prefix - 1, 1))
        cache.commit_step()
    return cache, indexer


def begin(cache, k, cis, prefix):
    cache.begin_step(len(k) - prefix)
    cache.write_layer(0, keys=k[prefix:], values=k[prefix:], cis_scores=cis[prefix:])


@pytest.mark.parametrize("length", [1024 * n for n in range(1, 33)])
@torch.inference_mode()
def test_independent_prefixes_match_and_joint_threshold_stays_at_2047(monkeypatch, length):
    rows, prefix = 1024, length - 1024
    q, k, cis = inputs(length, rows)
    reference, original_indexer = independent_prefix(q, k, cis, prefix)
    actual, deferred_indexer = independent_prefix(q, k, cis, prefix)
    context = AttentionContext(0, prefix, rows)
    validation = DeferredValidation(1, actual.device)
    validation.allocate()
    calls = []
    original_short = _indexer_deferred_cuda.select_prepared_out
    original_checked = _indexer_checked_cuda.select_prepared_out

    def short(*args, **kwargs):
        calls.append("short")
        return original_short(*args, **kwargs)

    def checked(*args, **kwargs):
        calls.append("checked")
        return original_checked(*args, **kwargs)

    try:
        begin(reference, k, cis, prefix)
        expected = original_indexer(q, reference, context)
        reference.commit_step()
        monkeypatch.setattr(_indexer_deferred_cuda, "select_prepared_out", short)
        monkeypatch.setattr(_indexer_checked_cuda, "select_prepared_out", checked)
        begin(actual, k, cis, prefix)
        validation.begin(actual)
        selected = deferred_indexer(q, actual, context)
        assert calls == (["short"] if length < 32768 else ["checked"])
        assert actual.length == prefix
        assert actual.indexer_cache.layer_state(0).validated_tokens == length
        validation.check(actual)
        actual.commit_step()
        assert actual.length == length
        torch.testing.assert_close(selected.block_ids, expected.block_ids, rtol=0, atol=0)
        torch.testing.assert_close(selected.valid_mask, expected.valid_mask, rtol=0, atol=0)
        for name, value in actual.indexer_cache._buffers[0].items():
            expected_buffer = reference.indexer_cache._buffers[0][name]
            torch.testing.assert_close(
                value.view(torch.uint8), expected_buffer.view(torch.uint8), rtol=0, atol=0
            )
    finally:
        validation.clear()
        validation.close()
        actual.release()
        reference.release()


@pytest.mark.parametrize("prefix", [0, 4096, 30720])
@pytest.mark.parametrize("field", ["q", "keys", "cis_scores"])
@torch.inference_mode()
def test_failed_deferred_short_layer_rolls_back_at_owner_check(prefix, field):
    rows = 1024
    q, k, cis = inputs(prefix + rows, rows)
    cache, indexer = independent_prefix(q, k, cis, prefix)
    validation = DeferredValidation(1, cache.device)
    validation.allocate()
    try:
        begin(cache, k, cis, prefix)
        # Allocate all records before byte snapshots, without committing any
        # derived metadata, so both empty and nonempty prefixes are covered.
        specs = cache.indexer_cache.specs
        reservation = cache.indexer_cache.reserve_layer(
            0, len(k), {name: spec.capacity for name, spec in specs.items()}
        )
        before = {
            name: value.view(torch.uint8).clone() for name, value in reservation.buffers.items()
        }
        state = reservation.previous
        cache.indexer_cache.abort_layer(0)
        tensor = q if field == "q" else cache.layer_view(0)[field]
        tensor[(-1,) * tensor.ndim] = torch.nan
        validation.begin(cache)
        selected = indexer(q, cache, AttentionContext(0, prefix, rows))
        assert cache.length == prefix
        with pytest.raises(ValueError, match="finite Q, K and CIS"):
            validation.check(cache)
        assert (selected.block_ids == -1).all()
        assert not selected.valid_mask.any()
        cache.abort_step()
        assert cache.length == prefix
        assert cache.indexer_cache.layer_state(0) == state
        for name, value in cache.indexer_cache._buffers[0].items():
            torch.testing.assert_close(value.view(torch.uint8), before[name], rtol=0, atol=0)
    finally:
        validation.clear()
        validation.close()
        cache.release()
