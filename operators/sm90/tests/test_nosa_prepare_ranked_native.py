"""A prepared CIS ranking has no dependency on another CTA's pending writes."""

import pytest
import torch

from operators.sm90._nosa_prepare_cuda import PreparationScratch
from operators.sm90._nosa_prepare_ranked_cuda import prepare_ranked_out, supports_ranked
from operators.sm90.nosa_compression import update_compressed_cache

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _reference(pool, prefix):
    bits = pool[:prefix].view(torch.int16).to(torch.int32) & 65535
    bits = torch.where((bits & 32767) == 0, 0, bits)
    keys = torch.where((bits & 32768) != 0, bits ^ 65535, bits ^ 32768).T
    ids = keys.argsort(dim=-1, descending=True, stable=True)[:, :64].sort(dim=-1).values
    return ((keys.gather(1, ids) << 12) | (4095 - ids)).to(torch.int32)


def _inputs(prefix, offset=0, rows=128, heads=2, *, pattern="random", compressed_lag=0):
    torch.manual_seed(625)
    start = prefix * 64 + offset
    length = start + rows
    query = torch.randn((rows, heads, 16, 128), device="cuda", dtype=torch.bfloat16)
    keys = torch.randn((length, heads, 128), device="cuda", dtype=torch.bfloat16)
    cis = torch.randn((length, heads), device="cuda", dtype=torch.bfloat16)
    if pattern == "ties":
        cis.copy_((cis * 2).round())
    elif pattern == "extrema":
        maximum = torch.finfo(torch.bfloat16).max
        cis.copy_(torch.where(cis > 0, maximum, -maximum))
        cis[::5] = 0
    count, stable = length // 16 - 1, (length - 16) // 64
    old_c, old_s = start // 16 - 1, (start - 16) // 64
    outputs = tuple(
        torch.full(shape, 123, device="cuda", dtype=torch.bfloat16)
        for shape in ((count + 3, heads, 128), (count + 3, heads), (stable + 3, heads))
    )
    update_compressed_cache(keys[:start], cis[:start], *outputs, compressed_start=0, pooled_start=0)
    expected = tuple(t.clone() for t in outputs)
    old_c -= compressed_lag
    update_compressed_cache(keys, cis, *expected, compressed_start=old_c, pooled_start=old_s)
    ranking = torch.full((heads, 64), 71, device="cuda", dtype=torch.int32)
    kwargs = {
        "query_start": start,
        "validated_start": start,
        "compressed_start": old_c,
        "pooled_start": old_s,
        "scratch": PreparationScratch.allocate("cuda"),
    }
    return query, keys, cis, outputs, expected, ranking, kwargs


@requires_cuda
@pytest.mark.parametrize("prefix,rows,heads", [(64, 128, 1), (128, 128, 3), (1024, 1024, 2)])
@pytest.mark.parametrize("offset", [0, 15, 16, 63])
@pytest.mark.parametrize("pattern", ["random", "ties", "extrema"])
@torch.inference_mode()
def test_ranked_prepare_exactly_matches_stable_prefix_and_compression(
    prefix, rows, heads, offset, pattern
):
    query, keys, cis, outputs, expected, ranking, kwargs = _inputs(
        prefix, offset, rows, heads, pattern=pattern
    )
    assert prepare_ranked_out(query, keys, cis, *outputs, ranking, **kwargs)
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(ranking, _reference(expected[2], prefix), atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize("lag", [1, 4, 5, 17])
@torch.inference_mode()
def test_ranked_prepare_recomputes_every_pending_cis_window_locally(lag):
    query, keys, cis, outputs, expected, ranking, kwargs = _inputs(128, compressed_lag=lag)
    # Poison uncommitted capacity so any cross-CTA read has an observable effect.
    outputs[0][kwargs["compressed_start"] :].fill_(torch.nan)
    outputs[1][kwargs["compressed_start"] :].fill_(torch.nan)
    outputs[2][kwargs["pooled_start"] :].fill_(torch.nan)
    count, stable = len(keys) // 16 - 1, (len(keys) - 16) // 64
    for _ in range(5):
        assert prepare_ranked_out(query, keys, cis, *outputs, ranking, **kwargs)
        for actual, reference, size in zip(outputs, expected, (count, count, stable), strict=True):
            torch.testing.assert_close(actual[:size], reference[:size], atol=0, rtol=0)
        torch.testing.assert_close(ranking, _reference(expected[2], 128), atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize("target", [0, 1, 2])
@torch.inference_mode()
def test_ranked_prepare_nonfinite_preserves_all_buffers_and_ranking(target):
    query, keys, cis, outputs, _expected, ranking, kwargs = _inputs(128)
    before = tuple(t.view(torch.uint8).clone() for t in (*outputs, ranking))
    tensor = (query, keys, cis)[target]
    for bad in (torch.nan, torch.inf, -torch.inf):
        tensor.flatten()[-1] = bad
        assert not prepare_ranked_out(query, keys, cis, *outputs, ranking, **kwargs)
        for actual, reference in zip((*outputs, ranking), before, strict=True):
            torch.testing.assert_close(actual.view(torch.uint8), reference, atol=0, rtol=0)


@requires_cuda
@torch.inference_mode()
def test_ranked_prepare_async_graph_and_nondefault_stream():
    query, keys, cis, outputs, expected, ranking, kwargs = _inputs(128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert prepare_ranked_out(query, keys, cis, *outputs, ranking, **kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            flag = prepare_ranked_out(query, keys, cis, *outputs, ranking, **kwargs)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert flag
    torch.testing.assert_close(ranking, _reference(expected[2], 128), atol=0, rtol=0)
    before = tuple(t.view(torch.uint8).clone() for t in (*outputs, ranking))
    keys[-1, -1, -1] = torch.nan
    graph.replay()
    assert not flag
    for actual, reference in zip((*outputs, ranking), before, strict=True):
        torch.testing.assert_close(actual.view(torch.uint8), reference, atol=0, rtol=0)


@requires_cuda
@torch.inference_mode()
def test_ranked_prepare_rejects_cold_cache_readonly_and_invalid_ranking_before_writes():
    query, keys, cis, outputs, _expected, ranking, kwargs = _inputs(128)
    before = tuple(t.clone() for t in outputs)
    for change in ({"pooled_start": 0}, {"validated_start": len(keys)}):
        options = kwargs | change
        assert not supports_ranked(
            query,
            keys,
            cis,
            **{key: options[key] for key in ("query_start", "validated_start", "pooled_start")},
        )
        with pytest.raises(ValueError, match="Unsupported native ranked"):
            prepare_ranked_out(query, keys, cis, *outputs, ranking, **options)
    with pytest.raises(ValueError, match="separate contiguous"):
        prepare_ranked_out(query, keys, cis, *outputs, ranking[:, :63], **kwargs)
    for actual, reference in zip(outputs, before, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
