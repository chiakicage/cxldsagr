"""Native Hopper attention contract, including causal tails and stream ordering."""

import pytest
import torch

from models.attention_contracts import BlockSelection
from operators.nosa.attention.device_only._cuda import supports_native_attention
from operators.nosa.attention.device_only.api import nosa_block_sparse_attention
from operators.nosa.attention.reference.torch import reference_nosa_block_sparse_attention


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("queries", [4, 5])
@pytest.mark.parametrize("block_budget", [32, 64])
def test_cuda_native_grouped_independent_unsorted_masks_and_causal_blocks(
    dtype, queries, block_budget
):
    require_sm90()
    torch.manual_seed(8031)
    query_start = 79 * 64 + 62
    tokens = 80 * 64 + 23
    q = torch.randn(queries, 4608, device="cuda", dtype=dtype)[:, :4096]
    q = q.view(queries, 32, 128)
    keys = torch.randn(tokens + 1, 2, 144, device="cuda", dtype=dtype)[1:, :, :128]
    values = torch.randn_like(keys)
    bias = torch.randn(tokens, 2, device="cuda")
    bias[:64] = -float("inf")
    # Query rows 0/1 end in block 79, rows 2+ in block 80. Every query/head
    # chooses its own history; block 3 is excluded and may contain poison.
    generator = torch.Generator().manual_seed(934)
    history = torch.tensor([2, *range(4, 79)])
    ids = torch.full((queries, 2, block_budget * 2), -1, dtype=torch.int64)[..., ::2]
    mask = torch.zeros(queries, 2, block_budget * 2, dtype=torch.bool)[..., ::2]
    for row in range(queries):
        for head in range(2):
            selected = torch.cat(
                (
                    torch.tensor([79, 80, 81, -1, 2**62, 3, 0, 1]),
                    history[torch.randperm(len(history), generator=generator)[: block_budget - 8]],
                )
            )
            valid = (torch.arange(block_budget) + row + head) % 5 != 0
            valid[:3] = True
            valid[5] = False
            valid[6] = True
            order = torch.randperm(block_budget, generator=generator)
            ids[row, head] = selected[order]
            mask[row, head] = valid[order]
    mask[1, 1] = False
    # Preserve strided selection and mask views on CUDA as well.
    ids_storage = torch.empty(queries, 2, block_budget * 2, device="cuda", dtype=ids.dtype)
    mask_storage = torch.empty_like(ids_storage, dtype=torch.bool)
    ids_storage[..., ::2] = ids
    mask_storage[..., ::2] = mask
    selection = BlockSelection(ids_storage[..., ::2], 64, mask_storage[..., ::2])
    keys[192:256] = float("nan")
    values[192:256] = float("nan")
    bias[192:256] = float("nan")
    assert supports_native_attention(q, keys, values)
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, query_start, bias)
    actual = nosa_block_sparse_attention(q, keys, values, selection, query_start, bias)
    assert torch.isfinite(expected).all()
    tolerance = 0.001 if dtype == torch.bfloat16 else 0.0002
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.016)
    assert torch.equal(actual[1, 16:], torch.zeros_like(actual[1, 16:]))


@pytest.mark.parametrize("queries", [4, 5])
@pytest.mark.parametrize("block_budget", [32, 64])
def test_cuda_native_grouped_disjoint_selection_exercises_entire_union(queries, block_budget):
    require_sm90()
    query_start = queries * block_budget * 64
    tokens = query_start + 64
    q = torch.zeros(queries, 16, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros(tokens, 1, 128, device="cuda", dtype=q.dtype)
    block_values = (torch.arange(tokens // 64, device="cuda") % 17 - 8).to(q.dtype) / 8
    values = block_values.repeat_interleave(64)[:, None, None].expand(-1, 1, 128).contiguous()
    ids = torch.arange(queries * block_budget, device="cuda").view(queries, 1, block_budget)
    selection = BlockSelection(ids.flip(-1), 64)
    # Four disjoint 64-block selections make the largest 256-block union.
    # The fifth query exercises the final partially filled query package.
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, query_start)
    actual = nosa_block_sparse_attention(q, keys, values, selection, query_start)
    # Uniform probabilities and power-of-two values are exactly representable;
    # dropping/duplicating a union block cannot hide inside a loose tolerance.
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("queries", [4, 5])
@pytest.mark.parametrize("block_budget", [32, 64])
def test_cuda_native_grouped_all_masked_rows_return_zero(queries, block_budget):
    require_sm90()
    q = torch.ones(queries, 32, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.full((192, 2, 128), float("nan"), device="cuda", dtype=q.dtype)
    values = torch.full_like(keys, float("nan"))
    bias = torch.full((192, 2), float("nan"), device="cuda")
    ids = torch.arange(block_budget, device="cuda").expand(queries, 2, -1).clone()
    selection = BlockSelection(ids, 64, torch.zeros_like(ids, dtype=torch.bool))
    actual = nosa_block_sparse_attention(q, keys, values, selection, 126, bias)
    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("queries", [4, 5])
@pytest.mark.parametrize("poison", ["future_for_early_queries", "unselected_history", "past_group"])
def test_cuda_native_grouped_nan_values_do_not_leak_between_queries(queries, poison):
    require_sm90()
    torch.manual_seed(6081)
    query_start = 126
    q = torch.randn(queries, 32, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(192, 2, 128, device="cuda", dtype=q.dtype)
    values = torch.randn_like(keys)
    bias = torch.randn(192, 2, device="cuda")
    ids = torch.tensor([2, 0, 1, -1], device="cuda").expand(queries, 2, -1).clone()
    mask = torch.ones_like(ids, dtype=torch.bool)
    if poison == "future_for_early_queries":
        # Token 128 is future for queries 126/127 and legitimately visible to
        # the other rows. Only those later rows may propagate its NaN in dim 17.
        values[128, :, 17] = float("nan")
    elif poison == "unselected_history":
        # The union includes block 0, but query 0 did not select it. Zero P
        # alone must not let 0*NaN in a shared PV tile poison that query.
        mask[0, :, 1] = False
        values[0, :, 17] = float("nan")
    else:
        first_future = query_start + queries
        keys[first_future:] = float("nan")
        values[first_future:] = float("nan")
        bias[first_future:] = float("nan")
    selection = BlockSelection(ids, 64, mask)
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, query_start, bias)
    actual = nosa_block_sparse_attention(q, keys, values, selection, query_start, bias)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.016, equal_nan=True)
    if poison == "future_for_early_queries":
        assert torch.isfinite(expected[:2]).all()
        assert torch.isnan(expected[2:, :, 17]).all()
    elif poison == "unselected_history":
        assert torch.isfinite(expected[0]).all()
        assert torch.isnan(expected[1:, :, 17]).all()
    else:
        assert torch.isfinite(expected).all()
