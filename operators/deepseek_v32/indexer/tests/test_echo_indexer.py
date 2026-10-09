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


def test_prepared_lease_rejects_reuse_changed_rows_and_replaced_buffers():
    values = [torch.empty(4) for _ in range(4)]
    scratch = torch.tensor([3, 0, 0, 0], dtype=torch.int32).view(torch.int64)
    lease = dict(zip(("free_slots", "allocation_log", "counter", "prefetch_stats"), values))
    token = echo._PreparedPrefetch(3, *values, scratch)
    with pytest.raises(ValueError, match="resized"):
        token.consume(lease, 2)
    with pytest.raises(ValueError, match="replaced"):
        token.consume({**lease, "counter": values[2].clone()}, 3)
    lengths, requests = token.consume(lease, 3)
    assert lengths.tolist() == [3] and requests.tolist() == [0, 0, 0]
    assert lengths.untyped_storage().data_ptr() == scratch.untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="consumed"):
        token.consume(lease, 3)
    token = echo._PreparedPrefetch(3, *values, scratch)
    token.invalidate()
    with pytest.raises(ValueError, match="consumed"):
        token.consume(lease, 3)


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
@pytest.mark.parametrize("borrow_bounds", [False, True])
@pytest.mark.parametrize("column_stride", [1, 2])
def test_cuda_resident_mask_preserves_prefix_and_padded_storage(
    rows, columns, query_start, monkeypatch, borrow_bounds, column_stride
):
    require_sm90()
    import deep_gemm

    from operators.deepseek_v32.indexer.selection import exact_topk

    # Simulate the official API's strided output and deliberately dirty tail.
    width = columns * column_stride + 11
    backing = torch.arange((rows + 2) * width, device="cuda", dtype=torch.float32)
    backing = backing.remainder(19).reshape(rows + 2, width)
    scores = backing[1:-1, 3 : 3 + columns * column_stride : column_stride]
    expected = backing.clone()
    expected_scores = expected[1:-1, 3 : 3 + columns * column_stride : column_stride]
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
    bounds = (torch.zeros(rows, dtype=torch.int32, device="cuda"), ends.int())
    result = logits(*inputs(rows, columns), query_start, _bounds=bounds if borrow_bounds else None)
    assert result is scores
    # Compare storage bits, including untouched padding and adjacent rows.
    torch.testing.assert_close(backing.view(torch.int32), expected.view(torch.int32))
    actual_topk = exact_topk(result, min(16, columns))
    reference_topk = exact_topk(expected_scores, min(16, columns))
    for actual, reference in zip(actual_topk, reference_topk, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "rows,columns,query_start,k",
    [(1, 1, 0, 1), (5, 23, 2, 16), (5, 4099, 4094, 2048), (128, 65664, 65536, 2048)],
)
@torch.inference_mode()
def test_cuda_resident_padded_view_preserves_selection_and_logical_hint(
    rows, columns, query_start, k
):
    require_sm90()
    from operators.deepseek_v32.indexer.prefetch_hint import update_prefetch_hint
    from operators.deepseek_v32.indexer.selection import exact_topk

    data = inputs(rows, columns)
    original = logits(*data, query_start)
    padded = logits(*data, query_start, _pad_to_stride=True)
    assert original.shape == (rows, columns)
    assert padded.shape == (rows, (columns + 255) // 256 * 256)
    assert padded.is_contiguous()
    assert original.stride() == padded.stride()
    torch.testing.assert_close(padded[:, :columns].view(torch.int32), original.view(torch.int32))
    assert torch.isneginf(padded[:, columns:]).all()
    for actual, expected in zip(exact_topk(padded, k), exact_topk(original, k), strict=True):
        torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32))
    expected_hint = torch.arange(16, device="cuda", dtype=torch.float32)
    actual_hint = expected_hint.clone()
    update_prefetch_hint(original, expected_hint)
    update_prefetch_hint(padded[:, :columns], actual_hint)
    torch.testing.assert_close(actual_hint.view(torch.int32), expected_hint.view(torch.int32))


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


def test_cuda_echo_prefetch_scores_stay_exact_across_repeated_stage_reuse():
    require_sm90()
    q, k, weights, scales = inputs(1024, 66560)
    state = prefetch_state(66560, 1024, 8192)
    initial = {
        name: value.clone()
        for name, value in state.items()
        if isinstance(value, torch.Tensor) and value.is_cuda
    }
    expected = logits(q, k, weights, scales, 65536)
    # This shape spans multiple persistent query iterations. A single launch
    # can miss a race between scale reads and reuse of the three KV stages.
    for _ in range(64):
        for name, value in initial.items():
            state[name].copy_(value)
        actual = logits(q, k, weights, scales, 65536, prefetch=state)
        torch.testing.assert_close(
            actual.view(torch.int32), expected.view(torch.int32), atol=0, rtol=0
        )


def test_cuda_echo_prefetch_zero_budget_and_tied_scores():
    require_sm90()
    q, k, weights, scales = inputs(3, 3072)
    weights.zero_()
    state = prefetch_state(3072, 3, 32)
    result = logits(q, k, weights, scales, 3069, prefetch=state)
    assert state["counter"].item() == 0, "Tied threshold-bin keys must wait for exact top-k"
    assert result[0, :3070].eq(0).all()
    state["max_prefetch"] = 0
    state["counter"].fill_(17)
    state["allocation_log"].zero_()
    state["prefetch_stats"].fill_(19)
    weights.fill_(1)
    logits(q, k, weights, scales, 3069, prefetch=state)
    assert state["counter"].item() == 0
    assert state["allocation_log"].eq(2**31 - 1).all()
    assert state["prefetch_stats"].eq(0).all()


@pytest.mark.parametrize("rows,columns,limit", [(1, 17, 16), (5, 4099, 0), (128, 8192, 256)])
def test_cuda_prepared_prefetch_reuses_scratch_without_resetting_journal(rows, columns, limit):
    require_sm90()
    from torch.utils._python_dispatch import TorchDispatchMode

    from operators.deepseek_v32.indexer import cache_ops

    state = prefetch_state(columns, rows, limit)
    slots = len(state["free_slots"])
    priority = torch.full((slots + 1,), -1, dtype=torch.int64, device="cuda")
    priority[0] = 2**31 - 1
    scratch = torch.empty(slots, dtype=torch.int64, device="cuda")
    state["_prepared"] = cache_ops.prepare_prefetch(
        priority,
        state["free_slots"],
        state["allocation_log"],
        state["counter"],
        state["prefetch_stats"],
        scratch,
        timestamp=0,
        host_capacity=len(state["host"]),
        query_count=rows,
    )
    lengths, requests = state["_prepared"].metadata
    assert lengths.tolist() == [rows] and requests.eq(0).all()
    assert lengths.untyped_storage().data_ptr() == scratch.untyped_storage().data_ptr()
    protected = {state[name].data_ptr() for name in ("counter", "allocation_log", "prefetch_stats")}

    class NoJournalReset(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if (
                func in (torch.ops.aten.zero_.default, torch.ops.aten.fill_.Scalar)
                and args[0].data_ptr() in protected
            ):
                raise AssertionError("prepared journal was redundantly reset")
            return func(*args, **(kwargs or {}))

    data = inputs(rows, columns)
    reference = logits(*data, columns - rows)
    with NoJournalReset():
        result = logits(*data, columns - rows, prefetch=state)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert state["_prepared"].used
    with pytest.raises(ValueError, match="consumed"):
        logits(*data, columns - rows, prefetch=state)
    fetched = torch.where(state["host_to_device"] != 2**31 - 1)[0]
    assert len(fetched) <= limit
    mapped = state["host_to_device"][fetched].long()
    torch.testing.assert_close(
        state["device"][mapped].cpu(), state["host"][fetched.cpu()], rtol=0, atol=0
    )


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


@pytest.mark.parametrize("limit", [-1, 65, 8192, True, 1.0, None])
def test_bounded_prepared_lease_rejects_unprepared_consumer_limit(limit):
    values = [torch.empty(128) for _ in range(4)]
    scratch = torch.zeros(128, dtype=torch.int64)
    token = echo._BoundedPreparedPrefetch(1, *values, scratch)
    lease = dict(zip(("free_slots", "allocation_log", "counter", "prefetch_stats"), values))
    with pytest.raises(ValueError, match="prepared slot limit"):
        token.consume(lease, 1, consumer_limit=limit)
    with pytest.raises(TypeError):
        token.consume(lease, 1)
    assert type(token) is not echo._PreparedPrefetch
    token.consume(lease, 1, consumer_limit=64)
    with pytest.raises(ValueError, match="consumed"):
        token.consume(lease, 1, consumer_limit=64)


@pytest.mark.parametrize("field", range(5))
@pytest.mark.parametrize("change", ["resize", "set", "shape"])
def test_bounded_prepared_lease_rejects_inplace_storage_or_geometry_change(field, change):
    values = [torch.empty(128) for _ in range(4)]
    scratch = torch.zeros(128, dtype=torch.int64)
    token = echo._BoundedPreparedPrefetch(1, *values, scratch)
    lease = dict(zip(("free_slots", "allocation_log", "counter", "prefetch_stats"), values))
    tensor = (*values, scratch)[field]
    if change == "resize":
        tensor.resize_(129)
    elif change == "set":
        tensor.set_(tensor.clone())
    else:
        tensor.unsqueeze_(0)
    with pytest.raises(ValueError, match="resized"):
        token.consume(lease, 1, consumer_limit=64)


@pytest.mark.parametrize(
    "changed",
    [
        {"rows": 2},
        {"columns": 65538},
        {"query_start": 65535},
        {"history_length": 65535},
        {"limit": 63},
        {"limit": 64.0},
    ],
)
def test_free_prepare_provider_uses_exact_official_q1_predicate(changed):
    from operators.deepseek_v32.indexer.cache_ops import supports_free_q1_prepare

    args = {"rows": 1, "columns": 65537, "query_start": 65536, "history_length": 65536, "limit": 64}
    assert supports_free_q1_prepare(**args)
    assert not supports_free_q1_prepare(**(args | changed))
    assert not supports_free_q1_prepare(
        rows=1, columns=32767, query_start=32766, history_length=32766, limit=64
    )
    assert supports_free_q1_prepare(
        rows=1, columns=32768, query_start=32767, history_length=32767, limit=64
    )
