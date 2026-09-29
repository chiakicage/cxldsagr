"""FP32 attention score rounding and shift-invariant softmax regressions."""

import pytest
import torch

from layers.attention import BlockSelection
from operators.sm90.nosa_attention import (
    nosa_block_sparse_attention,
    reference_nosa_block_sparse_attention,
)


@pytest.mark.parametrize("queries", [1, 8])
@pytest.mark.parametrize("case", ["cancellation", "positive_shift", "negative_shift"])
def test_cuda_native_attention_preserves_small_logits_after_large_terms(monkeypatch, queries, case):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("Native NOSA CUDA operator checks require SM90/Hopper")
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    torch.manual_seed(9821)
    tokens = 512
    q = torch.zeros(queries, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(tokens, 2, 128, device="cuda", dtype=q.dtype)
    if case == "cancellation":
        q[..., 0] = 4096
        k[..., 0] = ((torch.arange(tokens, device="cuda") % 17 - 8).float() * 65536).unsqueeze(-1)
        v = torch.randn_like(k)
        # The FP32 reference rounds QK * scale, then adds CIS. These terms cancel.
        bias = -(k[..., 0].float() * 4096 * (128**-0.5))
    else:
        parity = (torch.arange(tokens, device="cuda") % 2).float()
        v = (parity * 2 - 1)[:, None, None].expand(tokens, 2, 128).contiguous().to(q.dtype)
        shift = 1e7 if case == "positive_shift" else -1e7
        # The one-unit gap is exactly representable in FP32 at either offset.
        bias = (parity + shift)[:, None].expand(tokens, 2).contiguous()
    ids = torch.arange(tokens // 64, device="cuda").reshape(1, 1, -1)
    selection = BlockSelection(ids, 64)
    start = tokens - queries
    expected = reference_nosa_block_sparse_attention(q, k, v, selection, start, bias)
    actual = nosa_block_sparse_attention(q, k, v, selection, start, bias)
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
