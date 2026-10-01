"""Sparse MLA correctness; these tests are not performance experiments."""

import math

import pytest
import torch

from operators.sm90.deepseek_mla import reference_sparse_mla, sparse_mla


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("DeepSeek sparse MLA checks require SM90/Hopper")


def test_reference_selected_slots_include_duplicates_and_ignore_padding():
    q = torch.zeros(3, 2, 24)
    kv = torch.zeros(3, 24)
    kv[0] = float("nan")
    kv[1, :16] = 2
    kv[2, :16] = 5
    indices = torch.tensor([[1, 2, 2, -1], [-1, -1, 10, -1], [1, -1, -1, -1]])
    actual = reference_sparse_mla(q, kv, indices, 1.0, value_dim=16)
    torch.testing.assert_close(actual[0], torch.full((2, 16), 4.0))
    torch.testing.assert_close(actual[1], torch.zeros(2, 16))
    torch.testing.assert_close(actual[2], torch.full((2, 16), 2.0))


def test_reference_position_dimensions_affect_scores_but_are_not_values():
    q = torch.zeros(1, 1, 24)
    q[..., 23] = 1
    kv = torch.zeros(2, 24)
    kv[1, :16] = 1
    kv[1, 23] = math.log(3)
    output = reference_sparse_mla(q, kv, torch.tensor([[0, 1]]), 1.0, value_dim=16)
    torch.testing.assert_close(output, torch.full((1, 1, 16), 0.75))


def test_kernel_explicitly_requires_hopper():
    with pytest.raises(NotImplementedError, match="SM90"):
        sparse_mla(torch.zeros(1, 2, 24), torch.zeros(5, 24), torch.zeros(1, 4).int(), 1, 16)


@pytest.mark.parametrize("bad_case", ["dimension", "queries", "dtype", "ids", "value", "scale"])
def test_invalid_inputs_are_rejected(bad_case):
    q, kv, ids, scale, value_dim = (
        torch.zeros(2, 3, 24),
        torch.zeros(4, 24),
        torch.zeros(2, 4, dtype=torch.int32),
        0.2,
        16,
    )
    if bad_case == "dimension":
        kv = kv[:, :20]
    elif bad_case == "queries":
        ids = ids[:1]
    elif bad_case == "dtype":
        kv = kv.double()
    elif bad_case == "ids":
        ids = ids.float()
    elif bad_case == "value":
        value_dim = 25
    else:
        scale = float("inf")
    with pytest.raises(ValueError):
        reference_sparse_mla(q, kv, ids, scale, value_dim)


@pytest.mark.parametrize(
    ("dimension", "value_dim", "heads", "selected", "dtype"),
    [
        (24, 16, 3, 79, torch.bfloat16),
        (32, 16, 19, 129, torch.float16),
        (32, 32, 16, 65, torch.bfloat16),
        (576, 512, 128, 137, torch.bfloat16),
        (576, 512, 16, 2048, torch.bfloat16),
    ],
)
def test_cuda_matches_fp32_reference(dimension, value_dim, heads, selected, dtype):
    require_sm90()
    torch.manual_seed(427)
    # Exercise noncontiguous projections, cache rows, and index slots.
    q = torch.randn(3, heads * 2, dimension * 2, device="cuda", dtype=dtype)[:, ::2, ::2]
    kv = torch.randn(4234, dimension * 2, device="cuda", dtype=dtype)[::2, ::2]
    ids = torch.randint(1, len(kv), (3, selected * 2), device="cuda", dtype=torch.int32)[:, ::2]
    kv[0] = float("nan")
    ids[0] = -1
    ids[1, ::7] = -1
    ids[2, ::11] = len(kv) + 10
    scale = dimension**-0.5
    expected = reference_sparse_mla(q, kv, ids, scale, value_dim)
    actual = sparse_mla(q, kv, ids, scale, value_dim)
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=2e-2)
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))


def test_cuda_physical_remapping_preserves_attention():
    require_sm90()
    torch.manual_seed(734)
    q = torch.randn(4, 19, 576, device="cuda", dtype=torch.bfloat16)
    resident = torch.randn(1031, 576, device="cuda", dtype=torch.bfloat16)
    logical = torch.randint(0, len(resident), (4, 129), device="cuda", dtype=torch.int32)
    logical[:, -4:] = -1
    unique = torch.unique(logical[logical >= 0]).long()
    physical = torch.searchsorted(unique, logical).int().masked_fill(logical < 0, -1)
    compact = resident[unique]
    resident_output = sparse_mla(q, resident, logical, 0.07)
    compact_output = sparse_mla(q, compact, physical, 0.07)
    torch.testing.assert_close(compact_output, resident_output, atol=0, rtol=0)


def test_cuda_empty_tiles_and_large_logits_keep_online_softmax_stable():
    require_sm90()
    q = torch.zeros(1, 3, 24, device="cuda", dtype=torch.bfloat16)
    q[..., 23] = 100
    kv = torch.zeros(3, 24, device="cuda", dtype=torch.bfloat16)
    kv[0] = float("nan")
    kv[1, :16], kv[1, 23] = 2, 100
    kv[2, :16], kv[2, 23] = 5, -100
    ids = torch.full((1, 193), -1, device="cuda", dtype=torch.int32)
    ids[0, 65:67] = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    actual = sparse_mla(q, kv, ids, 1.0, 16)
    torch.testing.assert_close(actual, torch.full_like(actual, 2.0), atol=0, rtol=0)


@pytest.mark.parametrize(("queries", "tokens", "selected"), [(0, 5, 3), (2, 0, 3), (2, 5, 0)])
def test_cuda_empty_inputs(queries, tokens, selected):
    require_sm90()
    q = torch.empty(queries, 3, 24, device="cuda", dtype=torch.bfloat16)
    kv = torch.empty(tokens, 24, device="cuda", dtype=torch.bfloat16)
    indices = torch.full((queries, selected), -1, device="cuda", dtype=torch.int32)
    actual = sparse_mla(q, kv, indices, 0.2, 16)
    assert actual.shape == (queries, 3, 16)
    assert torch.equal(actual, torch.zeros_like(actual))
