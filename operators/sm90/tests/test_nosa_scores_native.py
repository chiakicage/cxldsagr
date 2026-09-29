"""Native QK normalization, pooled rounding, and stream correctness."""

import pytest
import torch

from models.nosa.indexer import compressed_scores_reference

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("count", [1, 127, 128, 129, 256, 257, 4159])
@torch.inference_mode()
def test_cuda_native_scores_strides_causality_and_pooled_rounding(dtype, count):
    from operators.sm90._nosa_scores_cuda import scores_out
    from operators.sm90.nosa_indexer import _pool_qa

    generator = torch.Generator(device="cuda").manual_seed(357)
    # Model QKV views and cache records need not have contiguous outer strides.
    query = torch.randn(6, 4608, device="cuda", dtype=dtype, generator=generator)
    query = query[1:, :4096].view(5, 2, 16, 128)
    keys = torch.randn(count + 1, 2, 144, device="cuda", dtype=dtype, generator=generator)
    keys = keys[1:, ..., :128]
    length = (count + 1) * 16
    positions = torch.tensor([0, 31, length - 1, 1023, 63], device="cuda")
    scores = torch.full((5, 2, count), float("nan"), device="cuda", dtype=dtype)
    scores_out(query, keys, positions, scores, 0, 0, pool_output=False)
    expected = compressed_scores_reference(query, keys, positions)
    torch.testing.assert_close(scores.float(), expected.float(), rtol=0.012, atol=0.001)
    assert scores[0].count_nonzero() == 0

    blocks = (length + 63) // 64
    pooled = torch.full((5, 2, blocks), float("nan"), device="cuda", dtype=torch.float32)
    expected_pool = torch.empty_like(pooled)
    scores_out(query, keys, positions, pooled, 0, blocks, pool_output=True)
    # Independent existing pooling implementation consumes the materialized,
    # model-dtype scores. Equality proves rounding happens before max pooling.
    _pool_qa[(10, (blocks + 127) // 128)](
        scores, positions, expected_pool, count, 2, blocks, 0, False, 128
    )
    torch.testing.assert_close(pooled, expected_pool, rtol=0, atol=0)

    # The actual select_blocks/select_contiguous_blocks pipeline uses a flat
    # row/head dimension for FlashInfer top-k; both layouts share the ABI.
    flat_pool = torch.full((10, blocks), float("nan"), device="cuda", dtype=torch.float32)
    scores_out(query, keys, positions, flat_pool, 0, blocks, pool_output=True)
    torch.testing.assert_close(flat_pool.view_as(pooled), pooled, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_cuda_native_scores_strided_groups_single_split_and_uniform_logits(dtype):
    from operators.sm90._nosa_scores_cuda import scores_out

    query = torch.randn(130, 2, 32, 128, device="cuda", dtype=dtype)[1:, :, ::2]
    query[::2].zero_()
    keys = torch.randn(290, 2, 128, device="cuda", dtype=dtype)[1:]
    # 129 queries use one normalizer split; early rows have no eligible key.
    positions = torch.arange(129, device="cuda", dtype=torch.int32) * 35
    output = torch.empty(129, 2, 289, device="cuda", dtype=dtype)
    scores_out(query, keys, positions, output, 0, 0, pool_output=False)
    expected = compressed_scores_reference(query, keys, positions)
    torch.testing.assert_close(output.float(), expected.float(), rtol=0.012, atol=0.001)


@requires_cuda
@torch.inference_mode()
def test_cuda_native_scores_broadcast_strides_use_triton(monkeypatch):
    from operators.sm90._nosa_scores_cuda import supports
    from operators.sm90.nosa_indexer import compressed_scores

    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    query = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.bfloat16).expand(3, 2, 16, 128)
    keys = torch.randn(17, 1, 128, device="cuda", dtype=torch.bfloat16).expand(17, 2, 128)
    positions = torch.tensor([31, 150, 287], device="cuda")
    output = torch.empty(3, 2, 17, device="cuda", dtype=query.dtype)
    assert not supports(query, keys, output)
    actual = compressed_scores(query, keys, positions)
    expected = compressed_scores_reference(query, keys, positions)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.012, atol=0.001)


@requires_cuda
@torch.inference_mode()
def test_cuda_native_scores_contiguous_range_on_nondefault_stream_and_graph():
    from operators.sm90._nosa_scores_cuda import scores_out

    query = torch.randn(7, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(289, 2, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(4000, 4007, device="cuda", dtype=torch.int32)
    expected = torch.empty(7, 2, 289, device="cuda", dtype=query.dtype)
    actual = torch.empty_like(expected)
    scores_out(query, keys, positions, expected, 0, 0, pool_output=False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        scores_out(query, keys, None, actual, 4000, 0, pool_output=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            scores_out(query, keys, None, actual, 4000, 0, pool_output=False)
        actual.fill_(float("nan"))
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["native", "triton"])
@torch.inference_mode()
def test_cuda_native_scores_large_uniform_logits_preserve_normalization(
    dtype, backend, monkeypatch
):
    from operators.sm90.nosa_indexer import _launch_scores as scores_out

    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", backend)

    count = 4159
    # Large but finite logits expose lost precision from folding log(sum) into
    # max, or fusing scale/subtract when the normalizer rounded scale first.
    magnitude = torch.tensor([64256.0, 65504.0], device="cuda", dtype=dtype)
    query = magnitude[:, None, None, None].expand(-1, 2, 16, 128).contiguous()
    keys = torch.ones(count, 2, 128, device="cuda", dtype=dtype)
    positions = torch.full((2,), (count + 1) * 16 - 1, device="cuda", dtype=torch.int64)
    actual = torch.empty(2, 2, count, device="cuda", dtype=dtype)
    scores_out(query, keys, positions, actual, 0, 0, pool_output=False)
    # Every Q head has exactly uniform softmax over all compressed windows.
    expected = torch.full_like(actual, 16.0 / count)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    blocks = ((count + 1) * 16 + 63) // 64
    pooled = torch.empty(2, 2, blocks, device="cuda", dtype=torch.float32)
    scores_out(query, keys, positions, pooled, 0, blocks, pool_output=True)
    torch.testing.assert_close(
        pooled[..., 1 : blocks - 17],
        expected[..., :1].float().expand_as(pooled[..., 1 : blocks - 17]),
        rtol=0,
        atol=0,
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["native", "triton"])
@torch.inference_mode()
def test_cuda_native_scores_full_prefill_match_fp32_reference(dtype, backend, monkeypatch):
    from operators.sm90.nosa_indexer import _launch_scores as scores_out
    from operators.sm90.nosa_indexer import _pool_qa

    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", backend)

    generator = torch.Generator(device="cuda").manual_seed(913)
    query = torch.randn(1024, 4608, device="cuda", dtype=dtype, generator=generator)
    query = query[:, :4096].view(1024, 2, 16, 128)
    keys = torch.randn(4159, 2, 144, device="cuda", dtype=dtype, generator=generator)
    keys = keys[..., :128]
    positions = torch.arange(65536, 66560, device="cuda")
    actual = torch.empty(1024, 2, 4159, device="cuda", dtype=dtype)
    scores_out(query, keys, None, actual, 65536, 0, pool_output=False)
    expected = compressed_scores_reference(query, keys, positions)
    # An absolute 1e-3 tolerance would hide large relative errors at this
    # context length. Allow one dtype rounding step with a small zero floor.
    rtol = 0.008 if dtype == torch.bfloat16 else 0.0011
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=1e-7)

    # The fused large-query path writes pooled scores directly. Compare it
    # against independently materialized, model-dtype reference scores so
    # the GQA rounding and five-window halo are both exercised.
    pooled = torch.empty(1024, 2, 1040, device="cuda", dtype=torch.float32)
    expected_pool = torch.empty_like(pooled)
    scores_out(query, keys, None, pooled, 65536, 1040, pool_output=True)
    _pool_qa[(2048, 9)](expected, positions, expected_pool, 4159, 2, 1040, 0, False, 128)
    torch.testing.assert_close(pooled, expected_pool, rtol=rtol, atol=1e-7)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_cuda_native_fused_pooled_tail_uniform_logits_and_empty_rows(dtype):
    from operators.sm90._nosa_scores_cuda import scores_out
    from operators.sm90.nosa_indexer import _pool_qa

    rows, count, blocks = 1025, 4159, 1040
    query = torch.empty(rows + 1, 4608, device="cuda", dtype=dtype)[1:, :4096]
    query = query.view(rows, 2, 16, 128)
    query[::2].fill_(-64256.0)
    query[1::2].fill_(64256.0)
    keys = torch.ones(count, 2, 144, device="cuda", dtype=dtype)[..., :128]
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) * 64
    positions[:4] = torch.tensor([0, 15, 31, 66559], device="cuda")

    # Uniform positive/negative logits have an exact analytical softmax.
    # Explicit positions cover empty rows and nonmonotonic causal boundaries.
    eligible = ((positions - 31) // 16 + 1).clamp(0, count)
    values = 16.0 / eligible.clamp_min(1).float()
    reference_scores = torch.where(
        torch.arange(count, device="cuda")[None, :] < eligible[:, None],
        values[:, None],
        0.0,
    ).to(dtype)
    reference_scores = reference_scores[:, None, :].expand(-1, 2, -1).contiguous()
    expected = torch.empty(rows, 2, blocks, device="cuda", dtype=torch.float32)
    _pool_qa[(rows * 2, 9)](reference_scores, positions, expected, count, 2, blocks, 0, False, 128)
    actual = torch.full_like(expected, float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        scores_out(query, keys, positions, actual, 0, blocks, pool_output=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            scores_out(query, keys, positions, actual, 0, blocks, pool_output=True)
        actual.fill_(float("nan"))
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [2, 1025])
@pytest.mark.parametrize("sign", [-1, 1])
@torch.inference_mode()
def test_cuda_native_scores_preserve_small_gaps_at_large_common_offset(dtype, rows, sign):
    from operators.sm90._nosa_scores_cuda import scores_out
    from operators.sm90.nosa_indexer import _pool_qa

    count, blocks, start = 4159, 1040, 66559
    query = torch.zeros(rows, 2, 16, 128, device="cuda", dtype=dtype)
    query[..., 0], query[..., 1] = 256, 1
    keys = torch.zeros(count, 2, 128, device="cuda", dtype=dtype)
    keys[..., 0] = sign * 65504
    keys[..., 1] = ((torch.arange(count, device="cuda") % 2) * 2)[:, None]
    positions = torch.arange(start, start + rows, device="cuda")
    # Each dot product is exactly representable in FP32. The natural-unit
    # scaled logits retain a small gap on top of a large shared offset; scaling
    # to base two before subtracting the maximum changes that gap.
    expected = compressed_scores_reference(query[:1], keys, positions[:1])
    expected = expected.expand(rows, -1, -1).contiguous()
    actual = torch.empty_like(expected)
    scores_out(query, keys, None, actual, start, 0, pool_output=False)
    tolerance = 0.008 if dtype == torch.bfloat16 else 0.0011
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=1e-7)

    expected_pool = torch.empty(rows, 2, blocks, device="cuda", dtype=torch.float32)
    _pool_qa[(rows * 2, 9)](expected, positions, expected_pool, count, 2, blocks, 0, False, 128)
    actual_pool = torch.empty_like(expected_pool)
    scores_out(query, keys, None, actual_pool, start, blocks, pool_output=True)
    torch.testing.assert_close(actual_pool, expected_pool, rtol=tolerance, atol=1e-7)
