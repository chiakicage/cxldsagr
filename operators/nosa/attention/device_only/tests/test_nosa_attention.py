"""Numerical checks for device-resident NOSA sparse attention."""

import pytest
import torch

from models.attention_contracts import BlockSelection
from operators.nosa.attention.device_only.api import nosa_block_sparse_attention
from operators.nosa.attention.reference.torch import reference_nosa_block_sparse_attention


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
