"""Native preparation preserves Triton rounding and failed-call buffer ownership."""

import pytest
import torch

from operators.nosa.indexer._prepare_cuda import (
    PreparationScratch,
    prepare_checked,
    prepare_out,
    supports,
)
from operators.nosa.indexer.compression import update_compressed_cache

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _outputs(length, heads, dim, dtype, extra=3):
    count, stable = max(0, length // 16 - 1), max(0, (length - 16) // 64)
    return tuple(
        torch.full(shape, 123, device="cuda", dtype=dtype)
        for shape in ((count + extra, heads, dim), (count + extra, heads), (stable + extra, heads))
    )


def _assert_exact(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def _layout(tensor, kind):
    if kind == "strided":
        shape = tuple(n * 2 if i < 2 else n for i, n in enumerate(tensor.shape))
        storage = torch.full(shape, torch.nan, device="cuda", dtype=tensor.dtype)
        result = storage[::2, ::2]
        result.copy_(tensor)
        return result
    if kind == "misaligned":
        storage = torch.empty(tensor.numel() + 1, device="cuda", dtype=tensor.dtype)
        result = storage[1:].reshape(tensor.shape)
        result.copy_(tensor)
        return result
    if kind == "odd_stride":
        storage = torch.full(
            (tensor.shape[0], tensor[0].numel() + 1), torch.nan, device="cuda", dtype=tensor.dtype
        )
        result = storage[:, :-1].view(tensor.shape)
        result.copy_(tensor)
        return result
    if kind == "broadcast":
        return tensor[:1, :1].expand_as(tensor)
    return tensor


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("layout", ["dense", "strided", "broadcast"])
@torch.inference_mode()
def test_native_prepare_matches_triton_exactly(dtype, dim, layout):
    torch.manual_seed(931)
    length, heads = 4193, 3
    keys = _layout(torch.randn((length, heads, dim), device="cuda", dtype=dtype), layout)
    cis = _layout(torch.randn((length, heads), device="cuda", dtype=dtype), layout)
    query = _layout(torch.randn((11, heads, dim), device="cuda", dtype=dtype), layout)
    actual = _outputs(length, heads, dim, dtype)
    expected = tuple(t.clone() for t in actual)
    update_compressed_cache(keys, cis, *expected, compressed_start=0, pooled_start=0)
    prepare_checked(
        query,
        keys,
        cis,
        *actual,
        validated_start=0,
        compressed_start=0,
        pooled_start=0,
        scratch=PreparationScratch.allocate("cuda"),
    )
    _assert_exact(actual, expected)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [64, 128, 256])
@torch.inference_mode()
def test_native_prepare_finite_extrema_match_triton_derived_nan_pool_semantics(dtype, dim):
    torch.manual_seed(411)
    length, heads = 417, 2
    maximum = torch.finfo(dtype).max
    keys = torch.where(torch.rand((length, heads, dim), device="cuda") > 0.5, maximum, -maximum).to(
        dtype
    )
    cis = torch.where(torch.rand((length, heads), device="cuda") > 0.5, maximum, -maximum).to(dtype)
    cis[::5] = 0
    query = torch.zeros((3, heads, dim), device="cuda", dtype=dtype)
    actual = _outputs(length, heads, dim, dtype)
    expected = tuple(t.clone() for t in actual)
    update_compressed_cache(keys, cis, *expected, compressed_start=0, pooled_start=0)
    prepare_checked(
        query,
        keys,
        cis,
        *actual,
        validated_start=0,
        compressed_start=0,
        pooled_start=0,
        scratch=PreparationScratch.allocate("cuda"),
    )
    _assert_exact(actual, expected)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [64, 128, 256])
@torch.inference_mode()
def test_native_prepare_incremental_tail_pool_catchup_and_rollback(dtype, dim):
    torch.manual_seed(681)
    length, heads = 4217, 2
    keys = _layout(torch.randn((length, heads, dim), device="cuda", dtype=dtype), "strided")
    cis = _layout(torch.randn((length, heads), device="cuda", dtype=dtype), "odd_stride")
    query = _layout(torch.randn((7, heads, 3, dim), device="cuda", dtype=dtype), "misaligned")
    actual = _outputs(length, heads, dim, dtype)
    expected = tuple(t.clone() for t in actual)
    scratch = PreparationScratch.allocate("cuda")
    old_c = old_s = validated = 0
    ends = (0, 1, 31, 32, 47, 48, 63, 64, 79, 80, 95, 96, 129, 4095, 4096, 4097, 4112, length)
    for end in ends:
        before = tuple(t[:n].clone() for t, n in zip(actual, (old_c, old_c, old_s), strict=True))
        update_compressed_cache(
            keys[:end], cis[:end], *expected, compressed_start=old_c, pooled_start=old_s
        )
        prepare_checked(
            query,
            keys[:end],
            cis[:end],
            *actual,
            validated_start=validated,
            compressed_start=old_c,
            pooled_start=old_s,
            scratch=scratch,
        )
        _assert_exact(actual, expected)
        _assert_exact((actual[0][:old_c], actual[1][:old_c], actual[2][:old_s]), before)
        old_c, old_s = max(0, end // 16 - 1), max(0, (end - 16) // 64)
        validated = end
    # A model cache may roll back valid cursors while retaining overwritten
    # suffix capacity. The next append must reproduce a fresh full update.
    keys[4000:].mul_(2)
    cis[4000:].mul_(2)
    old_c, old_s = 4000 // 16 - 1, (4000 - 16) // 64
    prepare_checked(
        query,
        keys,
        cis,
        *actual,
        validated_start=4000,
        compressed_start=old_c,
        pooled_start=old_s,
        scratch=scratch,
    )
    update_compressed_cache(keys, cis, *expected, compressed_start=old_c, pooled_start=old_s)
    _assert_exact(actual, expected)
    # Existing compressed records can outlive a shorter valid stable-pool prefix.
    actual[2].fill_(123)
    expected[2].fill_(123)
    count = length // 16 - 1
    prepare_checked(
        query,
        keys,
        cis,
        *actual,
        validated_start=length,
        compressed_start=count,
        pooled_start=0,
        scratch=scratch,
    )
    update_compressed_cache(keys, cis, *expected, compressed_start=count, pooled_start=0)
    _assert_exact(actual, expected)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("target", ["query", "keys", "cis"])
@pytest.mark.parametrize("bad", [torch.nan, torch.inf, -torch.inf])
@torch.inference_mode()
def test_native_prepare_bad_inputs_leave_every_derived_byte_unchanged(dtype, target, bad):
    query = torch.ones((5, 2, 128), device="cuda", dtype=dtype)
    keys = torch.ones((129, 2, 128), device="cuda", dtype=dtype)
    cis = torch.ones((129, 2), device="cuda", dtype=dtype)
    {"query": query, "keys": keys, "cis": cis}[target].flatten()[-1] = bad
    outputs = _outputs(len(keys), 2, 128, dtype)
    before = tuple(t.view(torch.uint8).clone() for t in outputs)
    scratch = PreparationScratch.allocate("cuda")
    kwargs = {"validated_start": 128, "compressed_start": 0, "pooled_start": 0, "scratch": scratch}
    assert not prepare_out(query, keys, cis, *outputs, **kwargs)
    _assert_exact(tuple(t.view(torch.uint8) for t in outputs), before)
    with pytest.raises(ValueError, match="finite Q, K and CIS"):
        prepare_checked(query, keys, cis, *outputs, **kwargs)
    _assert_exact(tuple(t.view(torch.uint8) for t in outputs), before)


@requires_cuda
@torch.inference_mode()
def test_native_prepare_scan_respects_logical_views_and_trusted_prefix():
    dtype = torch.bfloat16
    query = _layout(torch.ones((5, 3, 128), device="cuda", dtype=dtype), "strided")
    keys = _layout(torch.ones((129, 3, 128), device="cuda", dtype=dtype), "strided")
    cis = _layout(torch.ones((129, 3), device="cuda", dtype=dtype), "misaligned")
    outputs = _outputs(len(keys), 3, 128, dtype)
    scratch = PreparationScratch.allocate("cuda")
    prepare_checked(
        query,
        keys,
        cis,
        *outputs,
        validated_start=0,
        compressed_start=0,
        pooled_start=0,
        scratch=scratch,
    )
    before = tuple(t.clone() for t in outputs)
    keys[0, 0, 0] = torch.inf
    cis[0, 0] = torch.nan
    prepare_checked(
        query,
        keys,
        cis,
        *outputs,
        validated_start=128,
        compressed_start=7,
        pooled_start=1,
        scratch=scratch,
    )
    _assert_exact(outputs, before)
    query[0, 0, 0] = torch.nan
    assert not prepare_out(
        query,
        keys,
        cis,
        *outputs,
        validated_start=128,
        compressed_start=7,
        pooled_start=1,
        scratch=scratch,
    )
    _assert_exact(outputs, before)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_native_prepare_vector_scan_checks_every_packed_lane_and_ignores_row_padding(dtype):
    # Match the model's Q view: contiguous heads/features with a packed-KV gap
    # between rows. Unused row storage must not participate in validation.
    storage = torch.full((3, 4608), torch.nan, device="cuda", dtype=dtype)
    query = storage[:, :4096].view(3, 2, 16, 128)
    query.fill_(1)
    keys = torch.ones((129, 2, 128), device="cuda", dtype=dtype)
    cis = torch.ones((129, 2), device="cuda", dtype=dtype)
    outputs = _outputs(len(keys), 2, 128, dtype)
    scratch = PreparationScratch.allocate("cuda")
    kwargs = {"validated_start": 0, "compressed_start": 0, "pooled_start": 0, "scratch": scratch}
    prepare_checked(query, keys, cis, *outputs, **kwargs)
    before = tuple(t.view(torch.uint8).clone() for t in outputs)
    for index in range(16 // query.element_size()):
        for bad in (torch.nan, torch.inf, -torch.inf):
            query[1, 1, 7, index] = bad
            assert not prepare_out(query, keys, cis, *outputs, **kwargs)
            _assert_exact(tuple(t.view(torch.uint8) for t in outputs), before)
            query[1, 1, 7, index] = 1


@requires_cuda
@torch.inference_mode()
def test_native_prepare_nondefault_stream_and_async_graph_replay():
    dtype = torch.float32
    query = torch.randn((5, 3, 128), device="cuda", dtype=dtype)
    keys = torch.randn((129, 3, 128), device="cuda", dtype=dtype)
    cis = torch.randn((129, 3), device="cuda", dtype=dtype)
    actual = _outputs(len(keys), 3, 128, dtype)
    expected = tuple(t.clone() for t in actual)
    update_compressed_cache(keys, cis, *expected, compressed_start=0, pooled_start=0)
    scratch = PreparationScratch.allocate("cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    kwargs = {"validated_start": 0, "compressed_start": 0, "pooled_start": 0, "scratch": scratch}
    with torch.cuda.stream(stream):
        prepare_checked(query, keys, cis, *actual, **kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            flag = prepare_out(query, keys, cis, *actual, **kwargs)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert flag.data_ptr() == scratch.finite.data_ptr() and flag
    _assert_exact(actual, expected)
    for tensor in actual:
        tensor.fill_(321)
    before = tuple(t.clone() for t in actual)
    keys[-1, -1, -1] = torch.nan
    graph.replay()
    assert not flag
    _assert_exact(actual, before)


@requires_cuda
@pytest.mark.parametrize("layout", ["misaligned", "odd_stride"])
@torch.inference_mode()
def test_native_prepare_rejects_layouts_with_different_triton_rounding(layout):
    query = torch.ones((5, 2, 128), device="cuda", dtype=torch.float16)
    keys = _layout(torch.ones((129, 2, 128), device="cuda", dtype=torch.float16), layout)
    cis = torch.ones((129, 2), device="cuda", dtype=torch.float16)
    outputs = _outputs(len(keys), 2, 128, keys.dtype)
    before = tuple(t.clone() for t in outputs)
    assert not supports(query, keys, cis)
    with pytest.raises(ValueError, match="Unsupported native NOSA preparation"):
        prepare_out(
            query,
            keys,
            cis,
            *outputs,
            validated_start=0,
            compressed_start=0,
            pooled_start=0,
            scratch=PreparationScratch.allocate("cuda"),
        )
    _assert_exact(outputs, before)


@requires_cuda
@torch.inference_mode()
def test_native_prepare_metadata_error_precedes_all_derived_writes():
    query = torch.ones((5, 2, 128), device="cuda", dtype=torch.float16)
    keys = torch.ones((129, 2, 128), device="cuda", dtype=torch.float16)
    cis = torch.ones((129, 2), device="cuda", dtype=torch.float16)
    outputs = _outputs(len(keys), 2, 128, keys.dtype)
    before = tuple(t.clone() for t in outputs)
    scratch = PreparationScratch.allocate("cuda")
    for change in (
        {"validated_start": 130},
        {"compressed_start": -1},
        {"pooled_start": 2},
        {"scratch": PreparationScratch(scratch.partial[:1], scratch.finite)},
    ):
        kwargs = {
            "validated_start": 0,
            "compressed_start": 0,
            "pooled_start": 0,
            "scratch": scratch,
        } | change
        with pytest.raises(ValueError):
            prepare_out(query, keys, cis, *outputs, **kwargs)
        _assert_exact(outputs, before)
