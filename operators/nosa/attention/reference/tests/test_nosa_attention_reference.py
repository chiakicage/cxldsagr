"""Independent mathematical checks for the NOSA FP32 reference."""

import math

import pytest
import torch

from layers.attention import BlockSelection
from operators.nosa.attention.reference.torch import reference_nosa_block_sparse_attention


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
