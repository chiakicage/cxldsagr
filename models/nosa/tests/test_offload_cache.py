"""NOSA host history, compression boundaries, and derived-cache transactions."""

import pytest
import torch

from layers.attention import AttentionContext, ResidentLayerView
from models.nosa.config import NosaConfig
from models.nosa.indexer import NosaIndexer
from models.nosa.offload_cache import NosaOffloadCache
from models.nosa.scoring import NosaAttentionState, compress_sequence


def make_cache(length=4500, *, dtype=torch.float32):
    config = NosaConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=length,
    )
    return NosaOffloadCache(config, length, device="cpu", dtype=dtype)


def inputs(cache):
    generator = torch.Generator().manual_seed(592)
    return (
        torch.randn((2, cache.max_seq_len, 2, 4), dtype=cache.dtype, generator=generator),
        torch.randn((2, cache.max_seq_len, 2), dtype=cache.dtype, generator=generator),
        torch.randn((2, cache.max_seq_len, 4, 4), dtype=cache.dtype, generator=generator),
    )


def append(cache, keys, cis, end, *, layers=(0, 1)):
    start = cache.length
    cache.begin_step(end - start)
    for layer in layers:
        cache.write_layer(
            layer,
            keys=keys[layer, start:end],
            values=keys[layer, start:end] * 2,
            cis_scores=cis[layer, start:end],
        )


def assert_selection(indexer, cache, keys, cis, query, layer, end):
    start = max(0, end - 5)
    context = AttentionContext(layer, start, end - start)
    actual = indexer(query[layer, start:end], cache, context)
    expected = indexer(
        query[layer, start:end],
        ResidentLayerView(layer, keys=keys[layer, :end], cis_scores=cis[layer, :end]),
        context,
    )
    torch.testing.assert_close(actual.block_ids, expected.block_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual.valid_mask, expected.valid_mask, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_incremental_non_aligned_appends_match_full_compression_and_selection(dtype):
    cache = make_cache(192, dtype=dtype)
    keys, cis, query = inputs(cache)
    indexer = NosaIndexer(mode="nosa", backend="reference")
    for end in (1, 31, 32, 47, 48, 63, 64, 79, 80, 95, 96, 129, 191):
        old = cache.length
        append(cache, keys, cis, end)
        for layer in range(2):
            view = cache.offload_layer_view(layer)
            assert view["query_start"] == old
            assert view["host_keys"].shape == (192, 2, 4)
            assert view["host_keys"].is_contiguous()
            torch.testing.assert_close(view["suffix_keys"], keys[layer, old:end])
            torch.testing.assert_close(view["cis_scores"], cis[layer, :end])
            assert_selection(indexer, cache, keys, cis, query, layer, end)
            records = cache.indexer_cache.layer_view(layer)
            torch.testing.assert_close(
                records["compressed_keys"], compress_sequence(keys[layer, :end]), atol=0, rtol=0
            )
            compressed_cis = compress_sequence(cis[layer, :end])
            torch.testing.assert_close(records["compressed_cis"], compressed_cis, atol=0, rtol=0)
            stable = max(0, (end - 16) // 64)
            if stable:
                expected_pool = torch.stack(
                    [compressed_cis[max(0, 4 * b - 1) : 4 * b + 4].amax(0) for b in range(stable)]
                )
                torch.testing.assert_close(records["pooled_cis"], expected_pool, atol=0, rtol=0)
        assert cache.length == cache.indexer_cache.length == old
        cache.commit_step()
        assert cache.length == cache.indexer_cache.length == end
        assert all(cache.get_layer_state(layer) is None for layer in range(2))


def test_full_nosa_selection_uses_compressed_history_after_abort_truncate_and_retry():
    cache = make_cache()
    keys, cis, query = inputs(cache)
    indexer = NosaIndexer(mode="nosa", backend="reference")
    append(cache, keys, cis, 4097)
    cache.commit_step()  # Direct initialization also materializes derived records.
    before = cache.indexer_cache.layer_state(0)
    saved = {name: value.clone() for name, value in cache.indexer_cache.layer_view(0).items()}
    append(cache, keys, cis, 4217, layers=(0,))
    assert_selection(indexer, cache, keys, cis, query, 0, 4217)
    with pytest.raises(RuntimeError, match="Every layer"):
        cache.commit_step()
    cache.abort_step()
    assert cache.length == cache.indexer_cache.length == 4097
    assert cache.indexer_cache.layer_state(0) == before
    for name, value in cache.indexer_cache.layer_view(0).items():
        torch.testing.assert_close(value, saved[name], atol=0, rtol=0)
    keys[:, 4097:4217].add_(0.25)
    cis[:, 4097:4217].sub_(1)
    append(cache, keys, cis, 4217)
    for layer in range(2):
        assert_selection(indexer, cache, keys, cis, query, layer, 4217)
    cache.commit_step()
    cache.truncate(4111)
    assert cache.indexer_cache.layer_state(0).validated_tokens == 4111
    # Historical raw K outside the compression boundary is deliberately made
    # unreadable. Selection must use committed compressed records instead.
    cache.buffers["keys"][:, : cache.length - 31].fill_(torch.nan)
    append(cache, keys, cis, 4273)
    for layer in range(2):
        assert_selection(indexer, cache, keys, cis, query, layer, 4273)
    cache.commit_step()


@pytest.mark.parametrize("bad_input", ["q", "keys", "cis"])
def test_nonfinite_append_does_not_publish_derived_records(bad_input):
    cache = make_cache(160)
    keys, cis, query = inputs(cache)
    indexer = NosaIndexer(mode="nosa", backend="reference")
    append(cache, keys, cis, 80)
    cache.commit_step()
    state = cache.indexer_cache.layer_state(0)
    saved = {name: value.clone() for name, value in cache.indexer_cache.layer_view(0).items()}
    {"q": query, "keys": keys, "cis": cis}[bad_input][0, 95].fill_(torch.nan)
    append(cache, keys, cis, 96)
    with pytest.raises(ValueError, match="finite"):
        indexer(query[0, 91:96], cache, AttentionContext(0, 91, 5))
    assert cache.indexer_cache.layer_state(0) == state
    for name, value in cache.indexer_cache.layer_view(0).items():
        torch.testing.assert_close(value, saved[name], atol=0, rtol=0)
    cache.indexer_cache.validate_commit()
    cache.abort_step()
    assert cache.length == cache.indexer_cache.length == 80


def test_truncate_is_atomic_with_pending_derived_reservation_and_release_clears_storage():
    cache = make_cache(160)
    keys, cis, _ = inputs(cache)
    append(cache, keys, cis, 80)
    cache.commit_step()
    state = cache.indexer_cache.layer_state(0)
    cache.indexer_cache.reserve_layer(0, 80, state.lengths)
    with pytest.raises(RuntimeError, match="pending"):
        cache.truncate(79)
    assert cache.length == cache.indexer_cache.length == 80
    cache.indexer_cache.abort_layer(0)
    cache.length = 79
    assert cache.indexer_cache.layer_state(0).lengths["pooled_cis"] == 0
    append(cache, keys, cis, 80)
    cache.commit_step()
    assert cache.indexer_cache.layer_state(0).lengths["pooled_cis"] == 1
    cache.set_layer_state(0, object())
    with pytest.raises(ValueError, match="opaque"):
        cache.truncate(79)
    cache.reset()
    assert cache.length == cache.indexer_cache.length == 0
    assert cache.get_layer_state(0) is None
    assert cache.stats()["host_bytes"] > 0 and cache.stats()["resident_bytes"] > 0
    cache.release()
    cache.release()
    assert cache.stats()["host_bytes"] == cache.stats()["resident_bytes"] == 0
    with pytest.raises(RuntimeError, match="released"):
        cache.offload_layer_view(0)


def test_unsupported_policy_override_and_invalid_query_rejected():
    cache = make_cache(32)
    keys, cis, query = inputs(cache)
    append(cache, keys, cis, 32)
    context = AttentionContext(0, 0, 32)
    with pytest.raises(NotImplementedError, match="full NOSA"):
        NosaIndexer()(query[0], cache, context)
    indexer = NosaIndexer(mode="nosa")
    override = AttentionContext(0, 0, 32, NosaAttentionState(cis[0]))
    with pytest.raises(TypeError, match="CIS overrides"):
        indexer(query[0], cache, override)
    with pytest.raises(ValueError, match="query_length"):
        indexer(query[0], cache, AttentionContext(0, 0, 31))
    with pytest.raises(ValueError, match="cover"):
        indexer(query[0], cache, AttentionContext(0, 1, 32))
    assert not hasattr(cache, "layer_view")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("prefix,rows", [(63, 129), (65536, 1024)])
@torch.inference_mode()
def test_cuda_offload_compression_selection_and_retry_match_resident(monkeypatch, prefix, rows):
    from models.nosa.cache import NosaKVCache
    from models.nosa.indexer import prepare_indexer_inputs

    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    length = prefix + rows
    config = NosaConfig(
        hidden_size=4096,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=32,
        max_position_embeddings=length,
    )
    offload = NosaOffloadCache(config, length, device="cuda", dtype=torch.bfloat16)
    resident = NosaKVCache(config, length, device="cuda", dtype=torch.bfloat16, with_cis=True)
    generator = torch.Generator(device="cuda").manual_seed(2905)
    keys = torch.randn((length, 2, 128), device="cuda", dtype=offload.dtype, generator=generator)
    cis = torch.randn((length, 2), device="cuda", dtype=offload.dtype, generator=generator)
    query = torch.randn((rows, 32, 128), device="cuda", dtype=offload.dtype, generator=generator)
    for cache in (offload, resident):
        cache.begin_step(prefix)
        cache.write_layer(0, keys=keys[:prefix], values=keys[:prefix], cis_scores=cis[:prefix])
        if cache is resident:
            prepare_indexer_inputs(
                keys[:prefix], cis[:prefix], 1, indexer_cache=cache.indexer_cache, layer_idx=0
            )
        cache.commit_step()
    assert offload.host_layer_view(0)["keys"].is_pinned()
    torch.testing.assert_close(offload.host_layer_view(0)["keys"][:prefix], keys[:prefix].cpu())
    previous = offload.indexer_cache.layer_state(0)
    indexer = NosaIndexer(mode="nosa", backend="triton")
    for commit in (False, True):
        for cache in (offload, resident):
            cache.begin_step(rows)
            cache.write_layer(0, keys=keys[prefix:], values=keys[prefix:], cis_scores=cis[prefix:])
        context = AttentionContext(0, prefix, rows)
        actual = indexer(query, offload, context)
        expected = indexer(query, resident, context)
        torch.testing.assert_close(actual.block_ids, expected.block_ids, atol=0, rtol=0)
        torch.testing.assert_close(actual.valid_mask, expected.valid_mask, atol=0, rtol=0)
        for name, value in offload.indexer_cache.layer_view(0).items():
            torch.testing.assert_close(
                value, resident.indexer_cache.layer_view(0)[name], atol=0, rtol=0
            )
        for cache in (offload, resident):
            if commit:
                cache.commit_step()
                assert cache.length == cache.indexer_cache.length == length
            else:
                cache.abort_step()
                assert cache.length == cache.indexer_cache.length == prefix
        if not commit:
            assert offload.indexer_cache.layer_state(0) == previous
            keys[prefix:].add_(0.125)
            cis[prefix:].sub_(0.25)
    offload.release()
    resident.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_cuda_offload_compression_preserves_native_cancellation_and_overflow():
    from operators.nosa.indexer._prepare_cuda import PreparationScratch, prepare_out

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("NOSA offload checks require SM90/Hopper")
    config = NosaConfig(
        hidden_size=4096,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=32,
        max_position_embeddings=160,
    )
    cache = NosaOffloadCache(config, 160, device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros((160, 2, 128), device="cuda", dtype=torch.bfloat16)
    cis = torch.zeros((160, 2), device="cuda", dtype=torch.bfloat16)
    # Native D128 sums rows 0 and 8 before adding row 1. A generic reduction
    # that adds 0 and 1 first loses this small, representable compressed value.
    keys[0] = 2**25
    keys[1] = 1
    keys[8] = -(2**25)
    # All raw inputs are finite. Native CIS butterfly summation creates both
    # positive and negative infinity in window 0, hence a NaN compressed CIS.
    # Its five-window pool must ignore that NaN beside the finite zero windows.
    largest = torch.finfo(torch.bfloat16).max
    cis[0] = largest
    cis[16] = largest
    cis[1] = -largest
    cis[17] = -largest
    scratch = PreparationScratch.allocate("cuda")
    expected = {
        "compressed_keys": torch.empty((9, 2, 128), device="cuda", dtype=keys.dtype),
        "compressed_cis": torch.empty((9, 2), device="cuda", dtype=keys.dtype),
        "pooled_cis": torch.empty((2, 2), device="cuda", dtype=keys.dtype),
    }
    # Cross both incomplete K windows and the stable-pool boundary in multiple
    # appends, so the helper's local K indexing differs from global CIS indexing.
    for end in (17, 31, 32, 79, 80, 95, 96, 143, 144, 160):
        start = cache.length
        cache.begin_step(end - start)
        cache.write_layer(
            0, keys=keys[start:end], values=keys[start:end], cis_scores=cis[start:end]
        )
        actual = cache.prepare_indexer_inputs(0)
        finite = prepare_out(
            keys[:1],
            keys[:end],
            cis[:end],
            **expected,
            validated_start=0,
            compressed_start=0,
            pooled_start=0,
            scratch=scratch,
        )
        assert finite.item()
        for name, value in actual.items():
            torch.testing.assert_close(
                value, expected[name][: len(value)], atol=0, rtol=0, equal_nan=True
            )
        if end >= 32:
            assert torch.all(actual["compressed_keys"][0] == 1 / 32)
            assert torch.isnan(actual["compressed_cis"][0]).all()
        if end >= 80:
            assert torch.all(actual["pooled_cis"][0] == 0)
        cache.commit_step()
    cache.release()
