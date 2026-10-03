"""Indexer score equivalence, causal tails, and mapped-host prefetch invariants."""

import pytest
import torch

from operators.deepseek_v32.indexer.echo import build_info, logits


def test_echo_metadata_and_cpu_rejection():
    info = build_info()
    assert len(info["upstream_revision"]) == 40
    assert "operators/deepseek_v32/indexer/csrc/echo_logits.cuh" in info["source_sha256"]
    with pytest.raises(ValueError, match="CUDA Q"):
        logits(torch.zeros(1, 64, 128), torch.zeros(8, 128), None, None, 7)


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.fail("ECHO numerical tests require Hopper SM90")


def inputs(rows, columns):
    torch.manual_seed(521)
    q = torch.randn(rows, 64, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(columns, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 64, device="cuda") / 64
    scales = torch.rand(columns, device="cuda") + 0.5
    return q, k, weights, scales


def prefetch_state(columns, rows, limit):
    pool_size = rows + limit
    host = torch.randn(columns, 576, dtype=torch.bfloat16, pin_memory=True)
    pool = torch.zeros(pool_size, 576, device="cuda", dtype=torch.bfloat16)
    h2d = torch.full((columns,), 2**31 - 1, device="cuda", dtype=torch.int32)
    d2h = torch.full((pool_size,), 2**31 - 1, device="cuda", dtype=torch.int64)
    # Every newly appended record is resident and protected from the prefetch pool.
    pool[:rows].copy_(host[-rows:])
    h2d[-rows:] = torch.arange(rows, device="cuda", dtype=torch.int32)
    d2h[:rows] = torch.arange(columns - rows, columns, device="cuda")
    return {
        "host": host,
        "device": pool,
        "host_to_device": h2d,
        "device_to_host": d2h,
        "free_slots": torch.arange(rows, pool_size, device="cuda", dtype=torch.int32),
        "counter": torch.zeros(1, device="cuda", dtype=torch.uint32),
        "offset": torch.zeros(16, device="cuda"),
        "max_prefetch": limit,
    }


@pytest.mark.parametrize("rows,columns", [(1, 17), (5, 4099), (302, 4096)])
def test_cuda_echo_logits_match_independent_fp32_oracle(rows, columns):
    require_sm90()
    q, k, weights, scales = inputs(rows, columns)
    result = logits(q, k, weights, scales, columns - rows)
    # Chunk the independent oracle to keep its [Q,heads,K] intermediates small.
    for start in range(0, rows, 16):
        stop = min(rows, start + 16)
        raw = torch.einsum("qhd,nd->qhn", q[start:stop].float(), k.float()).relu()
        expected = (raw * weights[start:stop, :, None]).sum(1) * scales
        positions = torch.arange(columns, device="cuda")
        causal = (
            positions[None] <= columns - rows + torch.arange(start, stop, device="cuda")[:, None]
        )
        expected.masked_fill_(~causal, -torch.inf)
        # Hopper FP8 WGMMA partial accumulation is not an IEEE FP32 dot product.
        # Resident and fused paths are required to agree bit-for-bit separately.
        torch.testing.assert_close(result[start:stop], expected, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "rows,columns,limit",
    [(1, 17, 16), (5, 4099, 57), (302, 4096, 2048), (1024, 65536 + 1024, 8192)],
)
def test_cuda_echo_prefetch_is_exact_topk_subset_and_preserves_new_records(rows, columns, limit):
    require_sm90()
    q, k, weights, scales = inputs(rows, columns)
    state = prefetch_state(columns, rows, limit)
    resident = logits(q, k, weights, scales, columns - rows)
    fused = logits(q, k, weights, scales, columns - rows, prefetch=state)
    torch.testing.assert_close(fused, resident, atol=0, rtol=0)
    h2d = state["host_to_device"].cpu()
    d2h = state["device_to_host"].cpu()
    pool = state["device"].cpu()
    fetched = torch.nonzero(h2d[: columns - rows] != 2**31 - 1).flatten()
    assert 0 < len(fetched) <= limit
    assert len(fetched) == min(state["counter"].item(), limit)
    assert not (h2d == -1).any(), "A failed atomic reservation must restore the miss sentinel"
    assert len(torch.unique(h2d[fetched])) == len(fetched)
    torch.testing.assert_close(pool[h2d[fetched].long()], state["host"][fetched], atol=0, rtol=0)
    torch.testing.assert_close(d2h[h2d[fetched].long()], fetched)
    torch.testing.assert_close(pool[:rows], state["host"][-rows:], atol=0, rtol=0)
    selected_union = torch.unique(resident.topk(min(2048, columns), dim=-1).indices).cpu()
    assert torch.isin(fetched, selected_union).all()


def test_cuda_echo_prefetch_zero_budget_and_tied_scores():
    require_sm90()
    q, k, weights, scales = inputs(3, 3072)
    weights.zero_()
    state = prefetch_state(3072, 3, 32)
    result = logits(q, k, weights, scales, 3069, prefetch=state)
    assert state["counter"].item() == 0, "Tied threshold-bin keys must wait for exact top-k"
    assert result[0, :3070].eq(0).all()
    state["max_prefetch"] = 0
    weights.fill_(1)
    logits(q, k, weights, scales, 3069, prefetch=state)
    assert state["counter"].item() == 0


def test_cuda_echo_budget_exhaustion_remains_uniform_across_prefetch_warps():
    require_sm90()
    q, k, weights, scales = inputs(302, 4096)
    # Repeatedly exhaust the budget while persistent CTAs still have more Q
    # blocks. An upstream shared-flag race made some warps skip a named barrier.
    for _ in range(24):
        state = prefetch_state(4096, 302, 57)
        logits(q, k, weights, scales, 3794, prefetch=state)
        torch.cuda.synchronize()
        assert min(state["counter"].item(), 57) == 57
