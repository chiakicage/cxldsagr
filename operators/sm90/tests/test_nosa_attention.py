"""Numerical checks for resident NOSA selection, CIS and fused online softmax."""

import math

import pytest
import torch

from layers.attention import BlockSelection
from operators.sm90.nosa_attention import (
    nosa_block_sparse_attention,
    reference_nosa_block_sparse_attention,
)


def dense_masked_oracle(q, keys, values, selection, query_start, cis_bias=None):
    """Independent dense masked attention, with a logical token membership mask."""
    queries, query_heads, dimension = q.shape
    kv_heads = keys.shape[1]
    groups = query_heads // kv_heads
    output = torch.zeros_like(q)
    for row in range(queries):
        for head in range(query_heads):
            kv_head = head // groups
            selection_row = row if selection.block_ids.shape[0] > 1 else 0
            selection_head = kv_head if selection.block_ids.shape[1] > 1 else 0
            ids = selection.block_ids[selection_row, selection_head].tolist()
            valid = (
                [True] * len(ids)
                if selection.valid_mask is None
                else selection.valid_mask[selection_row, selection_head].tolist()
            )
            blocks = {block for block, included in zip(ids, valid, strict=True) if included}
            tokens = [
                token
                for token in range(len(keys))
                if token <= query_start + row and token // 64 in blocks
            ]
            if not tokens:
                continue
            logits = torch.tensor(
                [
                    float(torch.dot(q[row, head].float(), keys[token, kv_head].float()))
                    / math.sqrt(dimension)
                    + (0.0 if cis_bias is None else float(cis_bias[token, kv_head]))
                    for token in tokens
                ],
                device=q.device,
            )
            if logits.isneginf().all():
                continue
            probabilities = logits.softmax(dim=-1)
            output[row, head] = torch.einsum(
                "t,td->d", probabilities, values[tokens, kv_head].float()
            ).to(q.dtype)
    return output


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
def test_reference_matches_independent_causal_gqa_oracle(with_bias, broadcast):
    generator = torch.Generator().manual_seed(702)
    q = torch.randn(5, 6, 16, generator=generator)[..., ::2]
    keys = torch.randn(197, 2, 16, generator=generator)[..., ::2]
    values = torch.randn(197, 2, 16, generator=generator)[..., ::2]
    cis_bias = torch.randn(197, 2, generator=generator) if with_bias else None
    ids = torch.tensor([[[3, -1, 0, 1, 2, 100]]])
    valid = torch.tensor([[[True, False, True, False, True, True]]])
    if not broadcast:
        ids = ids.expand(5, 2, -1).clone()
        valid = valid.expand_as(ids).clone()
        valid[0, 1] = False
        ids[1:, 1, 4] = 1
    selection = BlockSelection(ids, 64, valid)
    expected = dense_masked_oracle(q, keys, values, selection, 192, cis_bias)
    actual = reference_nosa_block_sparse_attention(q, keys, values, selection, 192, cis_bias)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_reference_none_validity_and_zero_rows_do_not_load_nan_padding():
    q = torch.zeros(2, 2, 8)
    keys = torch.zeros(128, 1, 8)
    values = torch.ones_like(keys)
    values[:64] = float("nan")
    bias = torch.zeros(128, 1)
    bias[:64] = float("nan")
    selection = BlockSelection(torch.tensor([[[2, -1]], [[1, -1]]]), 64)
    actual = reference_nosa_block_sparse_attention(q, keys, values, selection, 126, bias)
    torch.testing.assert_close(actual[0], torch.zeros_like(q[0]))
    torch.testing.assert_close(actual[1], torch.ones_like(q[1]))
    bias[64:] = -float("inf")
    actual = reference_nosa_block_sparse_attention(q, keys, values, selection, 126, bias)
    assert torch.equal(actual, torch.zeros_like(actual))


def test_reference_cis_changes_token_weights_per_kv_head():
    q = torch.zeros(1, 4, 4)
    keys = torch.zeros(2, 2, 4)
    values = torch.zeros_like(keys)
    values[1] = 1
    selection = BlockSelection(torch.zeros(1, 1, 1, dtype=torch.long), 64)
    bias = torch.tensor([[0.0, math.log(3)], [math.log(3), 0.0]])
    result = reference_nosa_block_sparse_attention(q, keys, values, selection, 1, bias)
    torch.testing.assert_close(result[0, :2], torch.full((2, 4), 0.75))
    torch.testing.assert_close(result[0, 2:], torch.full((2, 4), 0.25))


def test_reference_fp32_semantics_ignore_outer_autocast():
    generator = torch.Generator().manual_seed(532)
    q = torch.randn(1, 4, 8, generator=generator)
    keys = torch.randn(70, 2, 8, generator=generator)
    values = torch.randn(70, 2, 8, generator=generator)
    selection = BlockSelection(torch.tensor([[[1, 0]]]), 64)
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, 69)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = reference_nosa_block_sparse_attention(q, keys, values, selection, 69)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad_input", ["gqa", "query_start", "mask", "block_size", "bias"])
def test_attention_rejects_invalid_geometry(bad_input):
    q = torch.zeros(1, 4, 8)
    keys = values = torch.zeros(2, 2, 8)
    ids = torch.zeros(1, 2, 2, dtype=torch.long)
    selection = BlockSelection(ids, 64)
    start, bias = 1, None
    if bad_input == "gqa":
        q = q[:, :3]
    elif bad_input == "query_start":
        start = 2
    elif bad_input == "mask":
        selection = BlockSelection(ids, 64, torch.ones_like(ids))
    elif bad_input == "block_size":
        selection = BlockSelection(ids, 32)
    else:
        bias = torch.zeros(2)
    with pytest.raises(ValueError):
        reference_nosa_block_sparse_attention(q, keys, values, selection, start, bias)


def test_fused_requires_hopper_and_reference_supports_empty_queries():
    q = torch.empty(0, 2, 64)
    keys = values = torch.empty(0, 1, 64)
    selection = BlockSelection(torch.zeros(1, 1, 1, dtype=torch.long), 64)
    assert reference_nosa_block_sparse_attention(q, keys, values, selection).shape == q.shape
    with pytest.raises(NotImplementedError, match="SM90"):
        nosa_block_sparse_attention(q, keys, values, selection)


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("NOSA CUDA operator checks require SM90/Hopper")


@pytest.mark.parametrize(
    ("dtype", "dimension", "groups"),
    [
        (torch.bfloat16, 64, 4),
        (torch.float16, 128, 1),
        (torch.bfloat16, 128, 3),
        (torch.float16, 64, 32),
    ],
)
def test_cuda_fused_matches_fp32_reference_for_gqa_cis_and_partial_blocks(dtype, dimension, groups):
    require_sm90()
    generator = torch.Generator().manual_seed(947)
    # Strided row/head views model packed projection and preallocated cache layouts.
    q = torch.randn(5, 2 * groups * 2, dimension, generator=generator).to("cuda", dtype)[:, ::2]
    keys = torch.randn(327, 4, dimension, generator=generator).to("cuda", dtype)[:, ::2]
    values = torch.randn(327, 4, dimension, generator=generator).to("cuda", dtype)[:, ::2]
    bias = torch.randn(327, 4, generator=generator).to("cuda")[:, ::2] * 2
    ids = torch.tensor([-1, 5, 0, 2, 3, 7], device="cuda").expand(5, 2, -1).clone()
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[..., 4] = False
    valid[0, 1] = False
    # NaNs in an explicitly masked block must never enter the online state.
    keys[192:256] = float("nan")
    values[192:256] = float("nan")
    bias[192:256] = float("nan")
    selection = BlockSelection(ids, 64, valid)
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, 322, bias)
    actual = nosa_block_sparse_attention(q, keys, values, selection, 322, bias)
    tolerance = 1.6e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.equal(actual[0, groups:], torch.zeros_like(actual[0, groups:]))


@pytest.mark.parametrize("with_bias", [False, True])
def test_cuda_broadcast_selection_masked_tiles_and_extreme_bias(with_bias):
    require_sm90()
    q = torch.zeros(3, 6, 64, device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros(131, 2, 64, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    selection = BlockSelection(torch.tensor([[[-1, 1, 2]]], device="cuda"), 64)
    keys[:64] = float("nan")
    values[:64] = float("nan")
    bias = None
    if with_bias:
        bias = torch.full((131, 2), -1e4, device="cuda")
        bias[:64] = float("nan")
        bias[128:, 0] = -2e4
        bias[:, 1] = -float("inf")
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, 128, bias)
    actual = nosa_block_sparse_attention(q, keys, values, selection, 128, bias)
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=8e-3)


@pytest.mark.parametrize("block_budget", [32, 64])
def test_cuda_long_selection_is_order_independent_and_matches_fp32(block_budget):
    require_sm90()
    generator = torch.Generator().manual_seed(916)
    q = torch.randn(1, 8, 128, generator=generator).to("cuda", torch.bfloat16)
    keys = torch.randn(80 * 64 + 23, 2, 128, generator=generator).to("cuda", torch.bfloat16)
    values = torch.randn(keys.shape, generator=generator).to("cuda", torch.bfloat16)
    bias = torch.randn(len(keys), 2, generator=generator).to("cuda", torch.bfloat16)
    ids = torch.cat(
        (torch.tensor([80]), torch.randperm(80, generator=generator)[: block_budget - 1])
    )
    ids = ids.to("cuda").reshape(1, 1, block_budget)
    selection = BlockSelection(ids, 64)
    expected = reference_nosa_block_sparse_attention(
        q, keys, values, selection, len(keys) - 1, bias
    )
    actual = nosa_block_sparse_attention(q, keys, values, selection, len(keys) - 1, bias)
    reversed_selection = BlockSelection(ids.flip(-1), 64)
    reversed_output = nosa_block_sparse_attention(
        q, keys, values, reversed_selection, len(keys) - 1, bias
    )
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1.6e-2)
    torch.testing.assert_close(reversed_output, expected, atol=1e-3, rtol=1.6e-2)
