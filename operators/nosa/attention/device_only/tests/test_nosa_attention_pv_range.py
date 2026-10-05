"""Finite V must not overflow the unnormalized PV sum or final dtype cast."""

import math

import pytest
import torch

from models.attention_contracts import BlockSelection
from operators.nosa.attention.device_only.api import nosa_block_sparse_attention


@pytest.mark.parametrize("queries", [1, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", ["uniform", "upward_probability", "mixed_sign"])
def test_cuda_attention_preserves_finite_pv_range(monkeypatch, queries, dtype, case):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("Native NOSA CUDA operator checks require SM90/Hopper")
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    q = torch.zeros(queries, 32, 128, dtype=dtype, device="cuda")
    k = torch.zeros(128, 2, 128, dtype=dtype, device="cuda")
    maximum = torch.finfo(dtype).max
    v = torch.full_like(k, maximum)
    cis = torch.zeros(128, 2, device="cuda")
    expected = torch.full_like(q, maximum)
    if case == "upward_probability":
        # FP16 rounding raises the probability mass above the FP32 denominator.
        cis[1:64] = math.log(0.50025)
    elif case == "mixed_sign":
        v[32:64] = -maximum
        expected.zero_()
    selection = BlockSelection(torch.zeros(1, 1, 1, device="cuda", dtype=torch.int64), 64)
    actual = nosa_block_sparse_attention(q, k, v, selection, 128 - queries, cis)
    assert torch.isfinite(actual).all()
    tolerance = 0.016 if dtype == torch.bfloat16 else 0.002
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf")])
def test_cuda_fa3_graph_repair_tracks_mutating_nonfinite_values(monkeypatch, poison):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("FA3 attention graph checks require SM90/Hopper")
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    q = torch.zeros(8, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(256, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.ones_like(k)
    ids = torch.tensor([0, 3], device="cuda").reshape(1, 1, 2)
    selection = BlockSelection(ids, 64)
    start = 232
    for _ in range(3):
        nosa_block_sparse_attention(q, k, v, selection, start)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = nosa_block_sparse_attention(q, k, v, selection, start)
    # Selected, wholly future, unselected, then finite again on the same graph.
    for token in (8, 250, 72, None):
        v.fill_(1)
        expected = torch.ones_like(q)
        if token is not None:
            v[token, 1, 127] = poison
        if token == 8:
            expected[:, 16:, 127] = poison
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016, equal_nan=True)
