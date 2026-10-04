"""Indexer score equivalence, causal tails, and mapped-host prefetch invariants."""

import pytest
import torch

from operators.deepseek_v32.indexer import echo
from operators.deepseek_v32.indexer.echo import build_info, logits


def test_echo_metadata_and_cpu_rejection():
    info = build_info()
    assert len(info["upstream_revision"]) == 40
    assert "operators/deepseek_v32/indexer/csrc/echo_logits.cuh" in info["source_sha256"]
    assert "3rdparty/cutlass/include/cute/tensor.hpp" in info["shared_header_sha256"]
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
    # Deliberately nonidentity, noncontiguous physical pages: two sessions can
    # have the same logical token IDs without sharing their host records.
    pages = (columns + 63) // 64
    page_table = torch.arange(pages - 1, -1, -1, device="cuda", dtype=torch.int32) * 2 + 1
    host = torch.randn(pages * 2 * 64, 576, dtype=torch.bfloat16, pin_memory=True)
    pool = torch.zeros(pool_size + 1, 576, device="cuda", dtype=torch.bfloat16)
    h2d = torch.full((len(host),), 2**31 - 1, device="cuda", dtype=torch.int32)
    d2h = torch.full((pool_size + 1,), 2**31 - 1, device="cuda", dtype=torch.int64)
    return {
        "host": host,
        "device": pool,
        "host_to_device": h2d,
        "device_to_host": d2h,
        "free_slots": torch.arange(1, pool_size + 1, device="cuda", dtype=torch.int32),
        "counter": torch.zeros(1, device="cuda", dtype=torch.uint32),
        "allocation_log": torch.full((pool_size + 1,), 2**31 - 1, device="cuda", dtype=torch.int64),
        "prefetch_stats": torch.zeros(3, device="cuda", dtype=torch.int64),
        "offset": torch.zeros(16, device="cuda"),
        "page_table": page_table,
        "history_length": columns - rows,
        "max_prefetch": limit,
    }


def global_ids(state, columns):
    logical = torch.arange(columns, device="cuda")
    return state["page_table"][logical // 64].long() * 64 + logical % 64


@pytest.mark.parametrize(
    "rows,columns,query_start",
    [(1, 17, 16), (5, 4099, 4094), (302, 4096, 3794), (5, 23, 2), (17, 4099, 0)],
)
def test_cuda_echo_logits_match_independent_fp32_oracle(rows, columns, query_start, monkeypatch):
    require_sm90()

    def forbidden_local_resident():
        raise AssertionError("Resident indexer must use official DeepGEMM without loading ECHO")

    monkeypatch.setattr(echo, "_module", forbidden_local_resident)
    q, k, weights, scales = inputs(rows, columns)
    result = logits(q, k, weights, scales, query_start)
    # Chunk the independent oracle to keep its [Q,heads,K] intermediates small.
    for start in range(0, rows, 16):
        stop = min(rows, start + 16)
        raw = torch.einsum("qhd,nd->qhn", q[start:stop].float(), k.float()).relu()
        expected = (raw * weights[start:stop, :, None]).sum(1) * scales
        positions = torch.arange(columns, device="cuda")
        causal = positions[None] <= query_start + torch.arange(start, stop, device="cuda")[:, None]
        expected.masked_fill_(~causal, -torch.inf)
        # Hopper FP8 WGMMA partial accumulation is not an IEEE FP32 dot product.
        # Resident and fused paths are required to agree bit-for-bit separately.
        torch.testing.assert_close(result[start:stop], expected, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "rows,columns,query_start",
    [(1, 1, 0), (1, 17, 0), (1, 17, 16), (5, 23, 2), (5, 23, 18), (17, 17, 0), (33, 65, 3)],
)
def test_cuda_resident_mask_preserves_prefix_and_padded_storage(
    rows, columns, query_start, monkeypatch
):
    require_sm90()
    import deep_gemm

    from operators.deepseek_v32.indexer.selection import exact_topk

    # Simulate the official API's strided output and deliberately dirty tail.
    backing = torch.arange((rows + 2) * (columns + 11), device="cuda", dtype=torch.float32)
    backing = backing.remainder(19).reshape(rows + 2, columns + 11)
    scores = backing[1:-1, 3 : 3 + columns]
    expected = backing.clone()
    expected_scores = expected[1:-1, 3 : 3 + columns]
    ends = query_start + torch.arange(1, rows + 1, device="cuda")
    expected_scores.masked_fill_(
        torch.arange(columns, device="cuda")[None, :] >= ends[:, None], -torch.inf
    )

    def official(*args, max_seqlen_k):
        assert max_seqlen_k == columns
        assert args[3].tolist() == [0] * rows
        assert args[4].tolist() == ends.tolist()
        return scores

    monkeypatch.setattr(deep_gemm, "fp8_fp4_mqa_logits", official)
    monkeypatch.setattr(echo, "_module", lambda: pytest.fail("resident path loaded ECHO"))
    result = logits(*inputs(rows, columns), query_start)
    assert result is scores
    # Compare storage bits, including untouched padding and adjacent rows.
    torch.testing.assert_close(backing.view(torch.int32), expected.view(torch.int32))
    actual_topk = exact_topk(result, min(16, columns))
    reference_topk = exact_topk(expected_scores, min(16, columns))
    for actual, reference in zip(actual_topk, reference_topk, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "rows,columns,limit,offset",
    [
        (1, 17, 16, 0.0),
        (5, 4099, 57, 0.0),
        (302, 4096, 2048, 0.0),
        (302, 4096, 2048, 13.5),
        (1024, 65536 + 1024, 8192, 0.0),
    ],
)
def test_cuda_echo_prefetch_is_exact_topk_subset_and_excludes_pending_suffix(
    rows, columns, limit, offset
):
    require_sm90()
    q, k, weights, scales = inputs(rows, columns)
    state = prefetch_state(columns, rows, limit)
    state["offset"].fill_(offset)
    resident = logits(q, k, weights, scales, columns - rows)
    fused = logits(q, k, weights, scales, columns - rows, prefetch=state)
    torch.testing.assert_close(fused, resident, atol=0, rtol=0)
    h2d = state["host_to_device"].cpu()
    d2h = state["device_to_host"].cpu()
    pool = state["device"].cpu()
    ids = global_ids(state, columns).cpu()
    fetched = torch.nonzero(h2d != 2**31 - 1).flatten()
    fetched_logical = torch.nonzero(h2d[ids] != 2**31 - 1).flatten()
    assert 0 < len(fetched) <= limit
    assert len(fetched) == state["prefetch_stats"][0].item()
    assert len(fetched) == (state["allocation_log"] != 2**31 - 1).sum().item()
    assert not (h2d == -1).any(), "A failed atomic reservation must restore the miss sentinel"
    assert len(torch.unique(h2d[fetched])) == len(fetched)
    torch.testing.assert_close(pool[h2d[fetched].long()], state["host"][fetched], atol=0, rtol=0)
    torch.testing.assert_close(d2h[h2d[fetched].long()], fetched)
    assert h2d[ids[-rows:]].eq(2**31 - 1).all()
    assert pool[0].eq(0).all()
    assert d2h[0].item() == 2**31 - 1
    selected_union = torch.unique(resident.topk(min(2048, columns), dim=-1).indices).cpu()
    assert torch.isin(fetched_logical, selected_union).all()


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
        assert state["prefetch_stats"][0].item() == 57


def test_cuda_echo_rejects_query_over_pool_and_prefetch_over_headroom_before_compile(monkeypatch):
    require_sm90()
    q, k, weights, scales = inputs(8, 64)
    state = prefetch_state(64, 4, 0)
    monkeypatch.setattr(
        echo, "_module", lambda: pytest.fail("invalid lease reached native compiler")
    )
    with pytest.raises(ValueError, match="Q <= P"):
        logits(q, k, weights, scales, 56, prefetch=state)
    state = prefetch_state(64, 8, 0)
    state["max_prefetch"] = 1
    with pytest.raises(ValueError, match="P-Q"):
        logits(q, k, weights, scales, 56, prefetch=state)
