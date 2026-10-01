"""Verify empty-query handling and nonempty FA3 dispatch against FP32."""

import pytest
import torch

from layers.attention import BlockSelection
from operators.nosa.attention.device_only import _fa3 as fa3
from operators.nosa.attention.reference.torch import reference_nosa_block_sparse_attention


@pytest.mark.parametrize("queries", [0, 1, 2, 3, 4, 8])
def test_cuda_fa3_short_query_dispatch_preserves_numerics(monkeypatch, queries):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("FA3 attention dispatch checks require SM90/Hopper")
    torch.manual_seed(719)
    q = torch.randn(queries, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(256, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    cis = torch.randn(256, 2, device="cuda")
    ids = torch.tensor([3, 0, 1], device="cuda").expand(queries, 2, -1).clone()
    valid = torch.ones_like(ids, dtype=torch.bool)
    if queries:
        valid[0, 1] = False
    selection = BlockSelection(ids, 64, valid)
    compiled = fa3._module()
    calls = []

    class Recorder:
        def forward(self, *args):
            calls.append("native")
            return compiled.forward(*args)

        def fa3_forward(self, *args):
            calls.append("fa3")
            return compiled.fa3_forward(*args)

    monkeypatch.setattr(fa3, "_module", lambda: Recorder())
    actual = fa3.launch_nosa_fa3_attention(q, k, v, selection, 248, cis)
    if not queries:
        assert actual.shape == q.shape
        assert calls == []
        return
    assert calls == ["fa3"]
    expected = reference_nosa_block_sparse_attention(q, k, v, selection, 248, cis)
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
    assert torch.equal(actual[0, 16:], torch.zeros_like(actual[0, 16:]))
