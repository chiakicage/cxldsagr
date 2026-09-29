"""Fully selected causal pages preserve finite PV limits and subnormals."""

import math

import pytest
import torch

from layers.attention import BlockSelection
from operators.sm90.nosa_attention import nosa_block_sparse_attention


@pytest.mark.parametrize("query_start", [127, 128])
@pytest.mark.parametrize(
    "case",
    ["positive", "negative", "upward_positive", "upward_negative", "mixed_sign", "subnormal"],
)
def test_cuda_attention_common_pages_preserve_finite_pv(monkeypatch, query_start, case):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("Native NOSA CUDA operator checks require SM90/Hopper")
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    q = torch.zeros(8, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(256, 2, 128, device="cuda", dtype=q.dtype)
    maximum = torch.finfo(q.dtype).max
    value = -maximum if case.endswith("negative") else maximum
    v = torch.full_like(k, value)
    cis = torch.zeros(256, 2, device="cuda")
    expected = torch.full_like(q, value)
    if case.startswith("upward"):
        cis[1:128] = math.log(0.50025)
    elif case == "mixed_sign":
        v[1::2].neg_()
        expected.zero_()
    elif case == "subnormal":
        bits = torch.tensor([1, 2, 63, 64, 127, 128, 129, 255], dtype=torch.int16)
        values = bits.view(torch.bfloat16).repeat(16).cuda()
        v[:] = values
        v[:, :, 1::2].neg_()
        expected = v[0].repeat_interleave(16, dim=0).expand_as(q)
    # Both pages are selected by every query; equality at token 127 is valid.
    ids = torch.tensor([0, 1], device="cuda", dtype=torch.int64).reshape(1, 1, 2)
    selection = BlockSelection(ids, 64)
    actual = nosa_block_sparse_attention(q, k, v, selection, query_start, cis)
    assert torch.isfinite(actual).all()
    if case == "subnormal":
        error = (actual.double() - expected.double()).abs().max().item()
        assert error <= 512 * 2.0**-133
    else:
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
