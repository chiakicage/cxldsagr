"""Check runtime work ordering with logical outputs and poisoned graph scratch."""

import pytest
import torch
import tvm_ffi

from operators.sm90 import _nosa_attention_fa3 as fa3


@pytest.mark.parametrize("queries", [1017, 1024])
def test_cuda_fa3_work_order_preserves_logical_mapping(queries):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("FA3 work ordering checks require SM90/Hopper")
    q = torch.zeros(queries, 32, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(32768, 2, 128, dtype=q.dtype, device=q.device)
    v = torch.ones_like(k)
    v[:, 1] = 2
    bias = torch.zeros(32768, 2, device=q.device)
    slots = torch.arange(64, device=q.device)
    ids = (slots % 4).expand(queries, 2, -1).clone()
    valid = torch.ones_like(ids, dtype=torch.bool)
    output = torch.empty_like(q)
    flags = torch.empty(((queries + 3) // 4, 2), dtype=torch.int32, device=q.device)
    pages = torch.empty((256, 512), dtype=torch.int32, device=q.device)
    members = torch.empty_like(pages)
    counts = torch.empty(512, dtype=torch.int32, device=q.device)
    compiled = fa3._module()

    def call():
        with tvm_ffi.use_torch_stream():
            compiled.fa3_forward(
                q, k, v, ids, valid, bias, output, flags, pages, members, counts, 32768
            )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for mode in ("mixed", "all_empty", "all_fallback", "mixed"):
        ids.copy_((slots % 4).expand_as(ids))
        valid.fill_(True)
        expected_flags = torch.zeros_like(flags)
        expected_counts = torch.full((256,), 4, dtype=torch.int32)
        if mode == "mixed":
            valid[:8, 0] = False
            ids[80:88, 1] = torch.arange(512, device=q.device).reshape(8, 64)
            expected_flags[20:22, 1] = 1
            expected_counts[0] = 0
            expected_counts[21] = 0
        elif mode == "all_empty":
            valid.fill_(False)
            expected_counts.zero_()
        else:
            ids.copy_((torch.arange(queries, device=q.device) % 8)[:, None, None] * 64 + slots)
            expected_flags.fill_(1)
            expected_counts.zero_()
            if queries == 1017:
                # The final group contains one query and only64 unique blocks.
                expected_flags[-1] = 0
                expected_counts[-2:] = 64
        expected = torch.ones_like(q)
        expected[:, 16:] = 2
        if mode == "mixed":
            expected[:8, :16] = 0
        elif mode == "all_empty":
            expected.zero_()
        output.fill_(float("nan"))
        for scratch in (flags, pages, members, counts):
            scratch.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.testing.assert_close(flags, expected_flags, atol=0, rtol=0)
        torch.testing.assert_close(counts[:256].cpu(), expected_counts, atol=0, rtol=0)
        expected_order = sorted(
            range(256), key=lambda i: (-((int(expected_counts[i]) + 1) // 2), i)
        )
        assert counts[256:].cpu().tolist() == expected_order
