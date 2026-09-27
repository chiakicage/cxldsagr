"""Native Hopper attention contract, including causal tails and stream ordering."""

import pytest
import torch

from layers.attention import BlockSelection
from operators.sm90._nosa_attention_cuda import supports_native_attention
from operators.sm90.nosa_attention import (
    nosa_block_sparse_attention,
    reference_nosa_block_sparse_attention,
)


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("Native NOSA CUDA operator checks require SM90/Hopper")


@pytest.fixture(autouse=True)
def force_native_backend(monkeypatch):
    # These tests must exercise the migration even if the caller selects the
    # Triton control for an experiment or for another group of tests.
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("query_start", [0, 322])
@pytest.mark.parametrize("bias_dtype", [None, torch.float32, torch.bfloat16, torch.float16])
def test_cuda_native_causal_gqa16_strides_invalid_ids_and_cis(dtype, query_start, bias_dtype):
    require_sm90()
    torch.manual_seed(4921)
    q = torch.randn(5, 64, 128, device="cuda", dtype=dtype)[:, ::2]
    keys = torch.randn(327, 4, 128, device="cuda", dtype=dtype)[:, ::2]
    values = torch.randn_like(keys)
    # Unsorted, noncontiguous selection/mask with invalid and masked IDs.
    ids = torch.tensor([0, 12, 5, -1, 2, 1, 3, 17], device="cuda")
    ids = ids.expand(5, 2, -1).clone()[..., ::2]
    mask = torch.ones(5, 2, 8, dtype=torch.bool, device="cuda")[..., ::2]
    mask[..., 3] = False
    mask[0, 1] = False
    keys[192:256] = float("nan")
    values[192:256] = float("nan")
    bias = None
    if bias_dtype is not None:
        bias = torch.randn(327, 4, device="cuda", dtype=bias_dtype)[:, ::2]
        bias[:, 1] = -float("inf")
        bias[192:256] = float("nan")
    selection = BlockSelection(ids, 64, mask)
    assert supports_native_attention(q, keys, values)
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, query_start, bias)
    actual = nosa_block_sparse_attention(q, keys, values, selection, query_start, bias)
    tolerance = 0.016 if dtype == torch.bfloat16 else 0.002
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.equal(actual[0, 16:], torch.zeros_like(actual[0, 16:]))


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_cuda_native_broadcast_empty_rows_graph_and_nondefault_stream(ids_dtype):
    require_sm90()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q = torch.randn(3, 32, 128, device="cuda", dtype=torch.bfloat16)
        keys = torch.randn(131, 2, 128, device="cuda", dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        ids = torch.tensor([[[1, 2, -1, 99]]], device="cuda", dtype=ids_dtype)
        bias = torch.randn(131, 2, device="cuda")
        bias[:, 1] = -float("inf")
        selection = BlockSelection(ids, 64)
        expected = reference_nosa_block_sparse_attention(q, keys, values, selection, 128, bias)
        nosa_block_sparse_attention(q, keys, values, selection, 128, bias)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = nosa_block_sparse_attention(q, keys, values, selection, 128, bias)
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.016)
    assert torch.equal(actual[:, 16:], torch.zeros_like(actual[:, 16:]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_budget", [32, 64])
def test_cuda_native_long_unsorted_selection_and_empty_first_tile(dtype, block_budget):
    require_sm90()
    torch.manual_seed(743)
    q = torch.randn(3, 32, 128, device="cuda", dtype=dtype)
    keys = torch.randn(80 * 64 + 23, 2, 128, device="cuda", dtype=dtype)
    values = torch.randn_like(keys)
    bias = torch.randn(len(keys), 2, device="cuda")
    bias[:64] = -float("inf")
    # Exercise many producer/consumer barrier phase transitions, causal tails,
    # and an initial tile with no contribution to the online softmax state.
    ids = torch.cat((torch.tensor([0, 80]), torch.randperm(79)[: block_budget - 2] + 1))
    ids = ids.to("cuda").reshape(1, 1, -1)
    selection = BlockSelection(ids, 64)
    start = len(keys) - len(q)
    assert supports_native_attention(q, keys, values)
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, start, bias)
    actual = nosa_block_sparse_attention(q, keys, values, selection, start, bias)
    reverse = nosa_block_sparse_attention(
        q, keys, values, BlockSelection(ids.flip(-1), 64), start, bias
    )
    tolerance = 0.001 if dtype == torch.bfloat16 else 0.0002
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.016)
    torch.testing.assert_close(reverse, expected, atol=tolerance, rtol=0.016)


@pytest.mark.parametrize("query_start", [0, 63, 64, 126])
def test_cuda_native_does_not_read_future_nan_values_in_causal_tile(query_start):
    require_sm90()
    torch.manual_seed(143)
    q = torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(128, 1, 128, device="cuda", dtype=q.dtype)
    values = torch.randn_like(keys)
    bias = torch.randn(128, 1, device="cuda")
    keys[query_start + 1 :] = float("nan")
    values[query_start + 1 :] = float("nan")
    bias[query_start + 1 :] = float("nan")
    # A valid current block, a future block, and an enormous int64 ID must all
    # preserve the logical causal mask without overflow or NaN contamination.
    ids = torch.tensor([[[query_start // 64, 2**62, query_start // 64 + 1]]], device="cuda")
    selection = BlockSelection(ids, 64)
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, query_start, bias)
    actual = nosa_block_sparse_attention(q, keys, values, selection, query_start, bias)
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.016)


def test_cuda_native_broadcast_qkv_strides_and_zero_queries():
    require_sm90()
    torch.manual_seed(352)
    q = torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16).expand(3, 16, 128)
    keys = torch.randn(1, 1, 128, device="cuda", dtype=q.dtype).expand(131, 1, 128)
    values = torch.randn_like(keys)
    selection = BlockSelection(torch.tensor([[[0, 1, 2]]], device="cuda"), 64)
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, 128)
    actual = nosa_block_sparse_attention(q, keys, values, selection, 128)
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.016)
    empty = nosa_block_sparse_attention(q[:0], keys[:0], values[:0], selection)
    assert empty.shape == (0, 16, 128)


def test_cuda_native_unsupported_tma_alignment_uses_triton():
    require_sm90()
    torch.manual_seed(591)
    # Slicing a packed dimension creates odd row/head strides and an unaligned
    # pointer. This public layout remains supported through the Triton path.
    q = torch.randn(2, 16, 129, device="cuda", dtype=torch.bfloat16)[..., 1:]
    keys = torch.randn(70, 1, 129, device="cuda", dtype=q.dtype)[..., 1:]
    values = torch.randn_like(keys)
    selection = BlockSelection(torch.tensor([[[0, 1]]], device="cuda"), 64)
    assert not supports_native_attention(q, keys, values)
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, 68)
    actual = nosa_block_sparse_attention(q, keys, values, selection, 68)
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.016)
