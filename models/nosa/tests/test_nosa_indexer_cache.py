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
    import operators.nosa.indexer._prepare_cuda as native_prepare
    from operators.nosa.indexer import validation

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
    original_prepare = native_prepare.prepare_out

    def prepare(q, k, c, *args, **kwargs):
        checked_lengths.append(len(k) - kwargs["validated_start"])
        return original_prepare(q, k, c, *args, **kwargs)

    monkeypatch.setattr(native_prepare, "prepare_out", prepare)

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


@requires_cuda
@pytest.mark.parametrize("bad_input", ["q", "keys", "cis"])
@torch.inference_mode()
def test_cuda_native_preparation_rejects_without_writes_and_allows_retry(monkeypatch, bad_input):
    from operators.nosa.indexer import validation

    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    config = tiny_config(hidden_size=512, head_dim=128, max_position_embeddings=160)
    cache = NosaKVCache(config, 160, device="cuda", dtype=torch.bfloat16, with_cis=True)
    indexer = NosaIndexer(mode="nosa", backend="triton")
    generator = torch.Generator(device="cuda").manual_seed(930)
    keys = torch.randn((128, 2, 128), generator=generator, device="cuda", dtype=cache.dtype)
    cis = torch.randn((128, 2), generator=generator, device="cuda", dtype=cache.dtype)
    query = torch.randn((5, 4, 128), generator=generator, device="cuda", dtype=cache.dtype)
    cache.begin_step(80)
    for layer in range(2):
        cache.write_layer(layer, keys=keys[:80], values=keys[:80], cis_scores=cis[:80])
    indexer(query, cache, AttentionContext(0, 75, 5))
    cache.commit_step()
    previous = cache.indexer_cache.layer_state(0)
    # Snapshot the whole allocation, including bytes beyond the valid prefix.
    # Rejection must not write even those unpublished append ranges.
    reservation = cache.indexer_cache.reserve_layer(0, 80, previous.lengths)
    for name, buffer in reservation.buffers.items():
        buffer[previous.lengths[name] :].fill_(7)
    buffers = dict(reservation.buffers)
    saved = {name: buffer.clone() for name, buffer in buffers.items()}
    cache.indexer_cache.abort_layer(0)

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("Supported owned inputs must use native preparation")

    monkeypatch.setattr(validation, "all_finite", unexpected_fallback)
    suffix_k, suffix_cis, bad_q = keys[80:].clone(), cis[80:].clone(), query.clone()
    {"q": bad_q, "keys": suffix_k, "cis": suffix_cis}[bad_input].flatten()[-1] = torch.nan
    cache.begin_step(48)
    cache.write_layer(0, keys=suffix_k, values=keys[80:], cis_scores=suffix_cis)
    with pytest.raises(ValueError, match="finite"):
        indexer(bad_q, cache, AttentionContext(0, 123, 5))
    assert cache.indexer_cache.layer_state(0) == previous
    cache.indexer_cache.validate_commit()  # No leaked per-layer reservation.
    for name, buffer in buffers.items():
        torch.testing.assert_close(buffer, saved[name], rtol=0, atol=0)
    cache.abort_step()

    cache.begin_step(48)
    cache.write_layer(0, keys=keys[80:], values=keys[80:], cis_scores=cis[80:])
    actual = indexer(query, cache, AttentionContext(0, 123, 5))
    assert cache.indexer_cache.layer_state(0).validated_tokens == 128
    assert actual.valid_mask[..., :2].all()
    assert (actual.block_ids[..., :2] == torch.tensor([0, 1], device="cuda")).all()
    ck, cc, _, _ = prepare_indexer_inputs(keys, cis, len(query))
    materialized = cache.indexer_cache.layer_view(0)
    torch.testing.assert_close(materialized["compressed_keys"], ck, rtol=0, atol=0)
    torch.testing.assert_close(materialized["compressed_cis"], cc, rtol=0, atol=0)
    cache.abort_step()
    assert cache.indexer_cache.layer_state(0) == previous


@requires_cuda
@pytest.mark.parametrize("prefix,rows", [(8192, 128), (65536, 1024)])
@torch.inference_mode()
def test_cuda_ranked_preparation_owns_disjoint_scratch_and_retries(monkeypatch, prefix, rows):
    import operators.nosa.indexer._indexer_checked_cuda as checked
    import operators.nosa.indexer._prepare_ranked_cuda as ranked
    import operators.nosa.indexer.api as selection

    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    length = prefix + rows
    config = tiny_config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=2,
        num_hidden_layers=1,
        head_dim=128,
        max_position_embeddings=length,
    )
    cache = NosaKVCache(config, length, device="cuda", dtype=torch.bfloat16, with_cis=True)
    generator = torch.Generator(device="cuda").manual_seed(830)
    keys = torch.randn((length, 2, 128), device="cuda", dtype=cache.dtype, generator=generator)
    cis = torch.randn((length, 2), device="cuda", dtype=cache.dtype, generator=generator)
    query = torch.randn((rows, 32, 128), device="cuda", dtype=cache.dtype, generator=generator)
    cache.begin_step(prefix)
    cache.write_layer(0, keys=keys[:prefix], values=keys[:prefix], cis_scores=cis[:prefix])
    prepare_indexer_inputs(
        keys[:prefix], cis[:prefix], 1, indexer_cache=cache.indexer_cache, layer_idx=0
    )
    cache.commit_step()
    previous = cache.indexer_cache.layer_state(0)
    indexer = NosaIndexer(mode="nosa", backend="triton")
    context = AttentionContext(0, prefix, rows)
    expected = indexer(query, ResidentLayerView(0, keys=keys, cis_scores=cis), context)
    original_prepare, original_select = (
        ranked.prepare_ranked_out,
        selection.select_contiguous_blocks,
    )
    prepared, consumed = [], []

    def assert_ranking_disjoint(ranking, workspace):
        assert ranking is not None
        assert (
            ranking.data_ptr()
            >= workspace.data_ptr() + workspace.numel() * workspace.element_size()
        )

    def record_prepare(*args, **kwargs):
        ranking = kwargs["ranking"]
        before = ranking.view(torch.uint8).clone()
        flag = original_prepare(*args, **kwargs)
        if not flag:
            torch.testing.assert_close(ranking.view(torch.uint8), before, atol=0, rtol=0)
        else:
            prepared.append(ranking.clone())
        return flag

    def record_select(*args, **kwargs):
        ranking, workspace = kwargs["prepared_ranking"], kwargs["workspace"]
        assert_ranking_disjoint(ranking, workspace)
        torch.testing.assert_close(ranking, prepared[-1], atol=0, rtol=0)
        result = original_select(*args, **kwargs)
        torch.testing.assert_close(ranking, prepared[-1], atol=0, rtol=0)
        consumed.append("separate")
        return result

    original_checked = checked.select_prepared_out

    def record_checked(*args, **kwargs):
        # The combined entry owns preparation and selection; the separate
        # Python selector is intentionally absent from this dispatch.
        ranking, workspace = kwargs["ranking"], kwargs["workspace"]
        assert_ranking_disjoint(ranking, workspace)
        before = ranking.view(torch.uint8).clone()
        try:
            result = original_checked(*args, **kwargs)
        except ValueError:
            torch.testing.assert_close(ranking.view(torch.uint8), before, atol=0, rtol=0)
            raise
        consumed.append("checked")
        return result

    monkeypatch.setattr(ranked, "prepare_ranked_out", record_prepare)
    monkeypatch.setattr(selection, "select_contiguous_blocks", record_select)
    monkeypatch.setattr(checked, "select_prepared_out", record_checked)
    reservation = cache.indexer_cache.reserve_layer(0, prefix, previous.lengths)
    buffers = dict(reservation.buffers)
    cache.indexer_cache.abort_layer(0)
    for target in ("query", "keys", "cis"):
        q, suffix_k, suffix_cis = query.clone(), keys[prefix:].clone(), cis[prefix:].clone()
        {"query": q, "keys": suffix_k, "cis": suffix_cis}[target].flatten()[-1] = torch.nan
        before = {name: tensor.view(torch.uint8).clone() for name, tensor in buffers.items()}
        cache.begin_step(rows)
        cache.write_layer(0, keys=suffix_k, values=keys[prefix:], cis_scores=suffix_cis)
        with pytest.raises(ValueError, match="finite"):
            indexer(q, cache, context)
        assert cache.indexer_cache.layer_state(0) == previous
        assert not consumed
        cache.indexer_cache.validate_commit()
        for name, tensor in buffers.items():
            torch.testing.assert_close(tensor.view(torch.uint8), before[name], atol=0, rtol=0)
        cache.abort_step()
    for commit in (False, True):
        cache.begin_step(rows)
        cache.write_layer(0, keys=keys[prefix:], values=keys[prefix:], cis_scores=cis[prefix:])
        actual = indexer(query, cache, context)
        torch.testing.assert_close(actual.block_ids, expected.block_ids, atol=0, rtol=0)
        torch.testing.assert_close(actual.valid_mask, expected.valid_mask, atol=0, rtol=0)
        if commit:
            cache.commit_step()
            assert cache.length == cache.indexer_cache.length == length
        else:
            cache.abort_step()
            assert cache.indexer_cache.layer_state(0) == previous
    assert consumed == ["checked" if prefix == 65536 else "separate"] * 2
