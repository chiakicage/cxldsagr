"""Sparse MLA device correctness; these are not performance experiments."""

import pytest
import torch

from operators.deepseek_v32.attention.device_only.mla import sparse_mla
from operators.deepseek_v32.attention.reference.torch import reference_sparse_mla


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("DeepSeek sparse MLA checks require SM90/Hopper")


def test_kernel_explicitly_requires_hopper():
    with pytest.raises(NotImplementedError, match="SM90"):
        sparse_mla(torch.zeros(1, 2, 24), torch.zeros(5, 24), torch.zeros(1, 4).int(), 1, 16)


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
