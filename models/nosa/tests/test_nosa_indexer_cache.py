"""Owned NOSA indexer data follows request appends, retries and borrowed views."""

import pytest
import torch

from layers.attention import AttentionContext, ResidentLayerView
from models.nosa.cache import NosaKVCache
from models.nosa.indexer import NosaIndexer, prepare_indexer_inputs
from models.nosa.scoring import NosaAttentionState, compress_sequence
from models.nosa.tests.test_model import tiny_config

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@requires_cuda
@torch.inference_mode()
def test_cuda_indexer_cache_append_abort_truncate_and_finite_prefix_reuse(monkeypatch):
    import operators.sm90.nosa_validation as validation

    config = tiny_config(hidden_size=256, head_dim=64, max_position_embeddings=4224)
    cache = NosaKVCache(config, 4224, device="cuda", dtype=torch.bfloat16, with_cis=True)
    indexer = NosaIndexer(mode="nosa", backend="triton")
    generator = torch.Generator(device="cuda").manual_seed(529)
    keys = torch.randn((2, 4224, 2, 64), generator=generator, device="cuda", dtype=cache.dtype)
    cis = torch.randn((2, 4224, 2), generator=generator, device="cuda", dtype=cache.dtype)
    query = torch.randn((2, 4224, 4, 64), generator=generator, device="cuda", dtype=cache.dtype)
    checked_lengths = []
    original_finite = validation.all_finite

    def finite(q, k, c):
        checked_lengths.append(len(k))
        return original_finite(q, k, c)

    monkeypatch.setattr(validation, "all_finite", finite)

    def append(end, *, abort=False, bad=False):
        old = cache.length
        cache.begin_step(end - old)
        for layer in range(2):
            new_cis = cis[layer, old:end].clone()
            if bad:
                new_cis[-1, 0] = torch.nan
            cache.write_layer(
                layer, keys=keys[layer, old:end], values=keys[layer, old:end], cis_scores=new_cis
            )
            start = max(old, end - 5)
            context = AttentionContext(layer, start, end - start)
            if bad:
                with pytest.raises(ValueError, match="finite"):
                    indexer(query[layer, start:end], cache, context)
                cache.abort_step()
                return
            actual = indexer(query[layer, start:end], cache, context)
            assert checked_lengths[-1] == end - old
            expected = indexer(
                query[layer, start:end],
                ResidentLayerView(layer, keys=keys[layer, :end], cis_scores=cis[layer, :end]),
                context,
            )
            torch.testing.assert_close(actual.block_ids, expected.block_ids, atol=0, rtol=0)
            torch.testing.assert_close(actual.valid_mask, expected.valid_mask, atol=0, rtol=0)
            assert cache.length == cache.indexer_cache.length == old
            if abort:
                cache.abort_step()
                return
        cache.commit_step()
        assert cache.length == cache.indexer_cache.length == end

    for end in (31, 32, 47, 48, 63, 64, 79, 80, 95, 96, 4097, 4217):
        append(end)
    long_state = cache.indexer_cache.layer_state(0)
    ck, cc, pool, _ = prepare_indexer_inputs(
        keys[0, :4111], cis[0, :4111], 5, indexer_cache=cache.indexer_cache, layer_idx=0
    )
    assert ck.shape[0] == cc.shape[0] == 255
    assert pool.shape[0] == 63
    torch.testing.assert_close(ck, compress_sequence(keys[0, :4111]))
    torch.testing.assert_close(cc, compress_sequence(cis[0, :4111]))
    assert cache.indexer_cache.layer_state(0) == long_state
    cache.truncate(4097)
    previous = [cache.indexer_cache.layer_state(layer) for layer in range(2)]
    saved = {name: value.clone() for name, value in cache.indexer_cache.layer_view(0).items()}
    append(4176, abort=True)
    # Retry different suffix contents after both new compressed records and
    # newly stable pools were written by the failed step.
    keys[:, 4097:4176].add_(0.25)
    cis[:, 4097:4176].sub_(1)
    append(4176, bad=True)
    assert [cache.indexer_cache.layer_state(layer) for layer in range(2)] == previous
    for name, value in cache.indexer_cache.layer_view(0).items():
        torch.testing.assert_close(value, saved[name], atol=0, rtol=0)
    append(4176)
    cache.reset()
    assert all(cache.indexer_cache.layer_state(layer).validated_tokens == 0 for layer in range(2))
    append(80)
    long_state = cache.indexer_cache.layer_state(0)
    # A shorter borrowed query range must not consume stable pools that depend
    # on the omitted suffix, or shorten the request's longer valid cache.
    result = indexer(query[0, 59:64], cache, AttentionContext(0, 59, 5))
    assert checked_lengths[-1] == 0
    assert cache.indexer_cache.layer_state(0) == long_state
    assert (result.block_ids[..., 0] == 0).all()
    override = cis[0, :80].clone()
    override[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        indexer(query[0, 75:80], cache, AttentionContext(0, 75, 5, NosaAttentionState(override)))
    assert checked_lengths[-1] == 80
    assert cache.indexer_cache.layer_state(0) == long_state
    cache.release()
    assert cache.stats()["resident_bytes"] == 0


@requires_cuda
@torch.inference_mode()
def test_cuda_indexer_lazily_materializes_a_committed_dense_prefix():
    config = tiny_config(hidden_size=256, head_dim=64, max_position_embeddings=128)
    cache = NosaKVCache(config, 128, device="cuda", dtype=torch.float16, with_cis=True)
    cache.begin_step(96)
    for layer in range(2):
        cache.write_layer(
            layer,
            keys=torch.zeros((96, 2, 64), device="cuda", dtype=cache.dtype),
            values=torch.zeros((96, 2, 64), device="cuda", dtype=cache.dtype),
            cis_scores=torch.zeros((96, 2), device="cuda", dtype=cache.dtype),
        )
    cache.commit_step()
    assert cache.indexer_cache.layer_state(0).validated_tokens == 0
    NosaIndexer(mode="nosa", backend="triton")(
        torch.zeros((1, 4, 64), device="cuda", dtype=cache.dtype),
        cache,
        AttentionContext(0, 95, 1),
    )
    assert cache.indexer_cache.layer_state(0).validated_tokens == 96
    assert cache.indexer_cache.layer_state(1).validated_tokens == 0
    assert cache.indexer_cache.layer_state(0).lengths == {
        "compressed_keys": 5,
        "compressed_cis": 5,
        "pooled_cis": 1,
    }
