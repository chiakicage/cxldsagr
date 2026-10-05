"""Exact sparse transfers, streaming lifetime and NOSA offload numerical checks."""

import pytest
import torch

from models.attention_contracts import BlockSelection
from operators.nosa.attention.device_only.api import nosa_block_sparse_attention
from operators.nosa.attention.offload.api import NosaFetchWorkspace
from operators.nosa.attention.reference.torch import reference_nosa_block_sparse_attention


@pytest.fixture
def hopper(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("NOSA offload checks require SM90/Hopper")
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    torch.manual_seed(620)


def _inputs(prefix, queries):
    q = torch.randn(queries, 32, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(prefix + queries, 2, 128, device="cuda", dtype=q.dtype)
    values = torch.randn_like(keys)
    host_keys = torch.full(
        (prefix + queries + 23, 2, 128), float("nan"), dtype=q.dtype, pin_memory=True
    )
    host_values = torch.full_like(host_keys, float("nan"), pin_memory=True)
    # No suffix token is initialized on the host. Fetching a prefix tail must
    # stop exactly at prefix, without overwriting the GPU-resident suffix.
    host_keys[:prefix].copy_(keys[:prefix])
    host_values[:prefix].copy_(values[:prefix])
    cis = torch.randn(prefix + queries, 2, device="cuda", dtype=torch.float32) * 0.3
    return q, keys, values, host_keys, host_values, cis


def _call(workspace, inputs, selection, prefix):
    q, keys, values, host_keys, host_values, cis = inputs
    return workspace.run(
        q, selection, host_keys, host_values, keys[prefix:], values[prefix:], cis, prefix
    )


def _expected_first_use(selection, prefix, queries, tile_size):
    # An independent host set is only a test oracle; runtime IDs never leave GPU.
    ids = selection.block_ids.cpu()
    mask = selection.valid_mask.cpu() if selection.valid_mask is not None else None
    first = {}
    for query in range(queries):
        row = query if ids.shape[0] != 1 else 0
        for head in range(2):
            selected_head = head if ids.shape[1] != 1 else 0
            for slot, block in enumerate(ids[row, selected_head].tolist()):
                if mask is not None and not mask[row, selected_head, slot]:
                    continue
                if block < 0 or block >= (prefix + 63) // 64:
                    continue
                # FA3 physically reads page 0 for odd-union padding. A selected
                # page 0 must be ready before any such speculative read.
                first.setdefault((block, head), 0 if block == 0 else query // tile_size)
    return first


def _expected_tile_bytes(selection, prefix, queries, tile_size):
    first = _expected_first_use(selection, prefix, queries, tile_size)
    result = [0] * ((queries + tile_size - 1) // tile_size)
    for (block, _head), tile in first.items():
        result[tile] += min(64, prefix - block * 64) * 128 * 2 * 2
    return result


@pytest.mark.parametrize("prefix", [0, 63, 64, 70, 129, 257])
def test_cuda_overlap_preserves_tail_cis_and_exact_fetch_union(hopper, prefix):
    queries, tile_size = 19, 8
    inputs = _inputs(prefix, queries)
    ids = torch.full((queries, 2, 6), -1, dtype=torch.int64)
    valid = torch.ones_like(ids, dtype=torch.bool)
    for query in range(queries):
        for head in range(2):
            row = list(dict.fromkeys([0, head + 1, query // 8 + 1, (prefix + query) // 64]))
            row.extend([-1, 2**40])
            ids[query, head, : len(row)] = torch.tensor(row)
    valid[0, 1] = False
    valid[1, 0, 0] = False
    # Exercise independent noncontiguous ID and validity strides.
    ids_storage = torch.zeros((queries, 2, 12), device="cuda", dtype=torch.int64)
    mask_storage = torch.zeros((queries, 2, 18), device="cuda", dtype=torch.bool)
    ids_storage[..., ::2] = ids.cuda()
    mask_storage[..., ::3] = valid.cuda()
    selection = BlockSelection(ids_storage[..., ::2], 64, mask_storage[..., ::3])
    workspace = NosaFetchWorkspace(
        prefix + queries + 32,
        2,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        query_tile_size=tile_size,
    )
    actual = _call(workspace, inputs, selection, prefix)
    q, keys, values, _, _, cis = inputs
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, prefix, cis)
    resident = nosa_block_sparse_attention(q, keys, values, selection, prefix, cis)
    workspace.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
    torch.testing.assert_close(actual, resident, atol=0.016, rtol=0.016)
    expected_bytes = _expected_tile_bytes(selection, prefix, queries, tile_size)
    assert workspace.last_tile_transfer_bytes.cpu().tolist() == expected_bytes
    assert workspace.last_transfer_bytes.item() == sum(expected_bytes)
    torch.testing.assert_close(workspace.keys[prefix : prefix + queries], keys[prefix:])
    torch.testing.assert_close(workspace.values[prefix : prefix + queries], values[prefix:])


@pytest.mark.parametrize("fetch_ctas,queries", [(1, 9), (7, 3), (96, 9)])
@pytest.mark.parametrize("bounded", [False, True])
def test_cuda_overlap_stripe_tail_trace_and_profile_reuse(hopper, fetch_ctas, queries, bounded):
    from operators.nosa.attention.offload._fused import build_info

    prefix, tile_size, heads = 145, 8, 2
    stripes = build_info()["fetch_stripes"]
    stripe_tokens = 64 // stripes
    inputs = _inputs(prefix, queries)
    q, keys, values, _, _, cis = inputs
    ids = torch.tensor([0, 1, 2, 2, -1, 2**40], device="cuda", dtype=torch.int64)
    ids = ids.reshape(1, 1, 6).expand(queries, heads, -1).clone()
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[:, 1, 1] = False
    # The partial last page sends head 0 through numerical repair. Keep head 1
    # on complete page 0 so small CTA caps also exercise live TMA ready waits.
    valid[:, 1, 2:4] = False
    selection = BlockSelection(ids, 64, valid)
    workspace = NosaFetchWorkspace(
        512,
        heads,
        128,
        device="cuda",
        dtype=q.dtype,
        query_tile_size=tile_size,
        fetch_ctas=fetch_ctas,
        **(
            {
                "bounded": True,
                "max_queries": queries,
                "trace_capacity": NosaFetchWorkspace.estimate_trace_rows(512, queries, heads),
            }
            if bounded
            else {}
        ),
    )
    fetch_slots = ((prefix + 63) // 64) * heads
    batches = ((queries + 7) // 8) * heads
    stripe_offset = fetch_slots + batches * 64
    # The 7/96 caps exceed these two/four-batch grids. Terminal queue claims
    # expose the number of participating CTAs without relying on trace timing.
    grid = min(batches, torch.cuda.get_device_properties(q.device).multi_processor_count)
    fetch_workers = min(fetch_ctas, grid)
    for iteration, profile in enumerate((True, False, True)):
        if iteration == 2:
            # Re-enabling tracing must clear rows left by the larger selection.
            valid[:, 1] = False
        workspace.profile_work_intervals = profile
        actual = _call(workspace, inputs, selection, prefix)
        resident = nosa_block_sparse_attention(q, keys, values, selection, prefix, cis)
        workspace.synchronize()
        assert torch.equal(actual.view(torch.int16), resident.view(torch.int16))
        first = _expected_first_use(selection, prefix, queries, tile_size)
        slots = {block * heads + head for block, head in first}
        expected_bytes = _expected_tile_bytes(selection, prefix, queries, tile_size)
        assert workspace.last_tile_transfer_bytes.cpu().tolist() == expected_bytes
        assert workspace.last_transfer_bytes.item() == sum(expected_bytes)
        expected_ready = torch.zeros(workspace._ready_blocks.shape, dtype=torch.int32)
        for block, head in first:
            expected_ready[block, head] = stripes
        assert torch.equal(workspace._ready_blocks.cpu(), expected_ready)
        queue = workspace._fetch_queue.cpu().tolist()
        assert queue[0] == len(slots)
        assert queue[1] == len(slots) * stripes + fetch_workers
        assert len(set(queue[2 : 2 + queue[0]])) == len(slots)
        assert set(queue[2 : 2 + queue[0]]) == slots
        if not profile:
            assert workspace.last_work_intervals.numel() == 0
            assert workspace.work_intervals() == workspace.stripe_work_intervals() == []
            continue
        work_rows = workspace.work_intervals()
        if iteration == 0:
            assert any(row["kind"] == 2 for row in work_rows)
        pages = [row for row in work_rows if row["kind"] == 1]
        stripe_rows = workspace.stripe_work_intervals()
        assert len(pages) == len(slots)
        assert {row["row"] for row in pages} == slots
        expected_stripes = {}
        for slot in slots:
            tokens = min(64, prefix - (slot // heads) * 64)
            for stripe in range(stripes):
                count = min(stripe_tokens, tokens - stripe * stripe_tokens)
                if count > 0:
                    expected_stripes[slot, stripe] = count * 128 * 2 * 2
        observed_stripes = {}
        for row in stripe_rows:
            slot, stripe = divmod(row["row"] - stripe_offset, stripes)
            assert (slot, stripe) not in observed_stripes
            assert (slot, stripe) in expected_stripes
            assert row["kind"] == 3
            assert 0 < row["start_ns"] < row["end_ns"]
            assert row["bytes"] == expected_stripes[slot, stripe]
            observed_stripes[slot, stripe] = row
        assert observed_stripes.keys() == expected_stripes.keys()
        assert sum(row["bytes"] for row in stripe_rows) == sum(expected_bytes)
        for page in pages:
            children = [row for (slot, _), row in observed_stripes.items() if slot == page["row"]]
            assert page["bytes"] == sum(row["bytes"] for row in children)
            # Empty tail stripes publish readiness but must not widen the copy
            # envelope or contribute a zero-byte copy interval.
            assert page["start_ns"] == min(row["start_ns"] for row in children)
            assert page["end_ns"] == max(row["end_ns"] for row in children)


@pytest.mark.parametrize("prefix,layout", [(63, "aligned"), (65, "misaligned"), (63, "broadcast")])
def test_cuda_overlap_initialization_stages_strided_suffix(hopper, prefix, layout):
    from operators.nosa.attention.offload._fused import build_info

    queries, tile_size = 17, 8
    q, keys, values, host_keys, host_values, cis = _inputs(prefix, queries)
    width, column = (131, 1) if layout == "misaligned" else (144, 0)
    storage = torch.empty((queries, 3, width), device="cuda", dtype=q.dtype)
    suffix_keys = storage[:, :2, column : column + 128]
    suffix_keys.copy_(keys[prefix:])
    if layout == "broadcast":
        source = torch.randn((1, 1, 128), device="cuda", dtype=q.dtype)
        suffix_values = source.expand(queries, 2, 128)
        values[prefix:].copy_(suffix_values)
    else:
        value_storage = torch.empty((queries, 3, 137), device="cuda", dtype=q.dtype)
        suffix_values = value_storage[:, :2, 1:129]
        suffix_values.copy_(values[prefix:])
    selection = BlockSelection(torch.ones((1, 1, 1), device="cuda", dtype=torch.int64), 64)
    workspace = NosaFetchWorkspace(
        512, 2, 128, device="cuda", dtype=q.dtype, query_tile_size=tile_size
    )
    workspace.keys.fill_(7)
    workspace.values.fill_(-19)
    workspace._first_use.fill_(-73)
    workspace._ready_blocks.fill_(29)
    workspace._tile_bytes.fill_(37)
    workspace.last_transfer_bytes.fill_(43)
    expected_keys, expected_values = workspace.keys.clone(), workspace.values.clone()
    first = _expected_first_use(selection, prefix, queries, tile_size)
    for expected, source in ((expected_keys, keys), (expected_values, values)):
        expected[: min(prefix, 64)] = 0
        expected[prefix : prefix + queries] = source[prefix:]
        for block, head in first:
            start, end = block * 64, min(prefix, (block + 1) * 64)
            expected[start:end, head] = source[start:end, head]
    actual = workspace.run(
        q, selection, host_keys, host_values, suffix_keys, suffix_values, cis, prefix
    )
    resident = nosa_block_sparse_attention(q, keys, values, selection, prefix, cis)
    workspace.synchronize()
    assert torch.equal(actual.view(torch.int16), resident.view(torch.int16))
    assert torch.equal(workspace.keys.view(torch.int16), expected_keys.view(torch.int16))
    assert torch.equal(workspace.values.view(torch.int16), expected_values.view(torch.int16))
    expected_first = torch.full(
        workspace._first_use.shape, torch.iinfo(torch.int32).max, dtype=torch.int32
    )
    expected_ready = torch.zeros(workspace._ready_blocks.shape, dtype=torch.int32)
    for (block, head), tile in first.items():
        expected_first[block, head] = tile
        expected_ready[block, head] = build_info()["fetch_stripes"]
    assert torch.equal(workspace._first_use.cpu(), expected_first)
    assert torch.equal(workspace._ready_blocks.cpu(), expected_ready)
    expected_bytes = _expected_tile_bytes(selection, prefix, queries, tile_size)
    assert workspace.last_transfer_bytes.item() == sum(expected_bytes)
    assert workspace.last_tile_transfer_bytes.cpu().tolist() == expected_bytes
    assert torch.count_nonzero(workspace._tile_bytes[len(expected_bytes) :]).item() == 0


@pytest.mark.parametrize("broadcast", ["head", "query", "both"])
@pytest.mark.parametrize("bounded", [False, True])
def test_cuda_overlap_broadcast_selection_and_serial_control(hopper, broadcast, bounded):
    prefix, queries, tile_size = 512, 25, 11
    inputs = _inputs(prefix, queries)
    ids = torch.tensor([1, 5, 8, -1], device="cuda", dtype=torch.int32).reshape(1, 1, 4)
    if broadcast == "head":
        ids = ids.expand(queries, 1, -1)
    elif broadcast == "query":
        ids = ids.expand(1, 2, -1)
    selection = BlockSelection(ids, 64)
    outputs, counts = [], []
    for overlap in (False, True):
        workspace = NosaFetchWorkspace(
            prefix + queries,
            2,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            query_tile_size=tile_size,
            overlap=overlap,
            **({"bounded": True, "max_queries": queries} if bounded else {}),
        )
        # Unselected page 0 is used by odd-page FA3 padding and must have safe
        # initialized contents, even if a previous layer left NaN in staging.
        workspace.keys.fill_(float("nan"))
        workspace.values.fill_(float("nan"))
        outputs.append(_call(workspace, inputs, selection, prefix))
        workspace.synchronize()
        counts.append(workspace.last_tile_transfer_bytes.cpu().tolist())
        assert workspace.last_transfer_bytes.item() == 2 * 2 * 64 * 128 * 2 * 2
        assert torch.isnan(workspace.keys[2 * 64 : 3 * 64]).all()
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert counts[0] == counts[1] == _expected_tile_bytes(selection, prefix, queries, tile_size)
    q, keys, values, _, _, cis = inputs
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, prefix, cis)
    torch.testing.assert_close(outputs[1], expected, atol=0.016, rtol=0.016)


def test_cuda_overlap_late_page_zero_and_masked_nan_preserve_repair(hopper):
    prefix, queries = 320, 24
    q, keys, values, host_keys, host_values, cis = _inputs(prefix, queries)
    q.zero_()
    keys.zero_()
    values.fill_(1)
    values[:64, :, 17] = float("nan")
    keys[192:256] = float("nan")
    values[192:256] = float("nan")
    cis.zero_()
    cis[192:256] = float("nan")
    host_keys[:prefix].copy_(keys[:prefix])
    host_values[:prefix].copy_(values[:prefix])
    ids = torch.full((queries, 2, 1), 1, device="cuda", dtype=torch.int64)
    ids[8:16] = 2
    ids[16:] = 0
    selection = BlockSelection(ids, 64)
    workspace = NosaFetchWorkspace(
        prefix + queries, 2, 128, device="cuda", dtype=q.dtype, query_tile_size=8
    )
    actual = _call(workspace, (q, keys, values, host_keys, host_values, cis), selection, prefix)
    expected = reference_nosa_block_sparse_attention(q, keys, values, selection, prefix, cis)
    workspace.synchronize()
    assert torch.isfinite(actual[:16]).all()
    assert torch.isnan(actual[16:, :, 17]).all()
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016, equal_nan=True)
    assert workspace.last_tile_transfer_bytes.cpu().tolist() == [131072, 65536, 0]


@pytest.mark.parametrize("bounded", [False, True])
def test_cuda_overlap_workspace_reuse_across_caller_streams_and_empty_step(hopper, bounded):
    workspace = NosaFetchWorkspace(
        512,
        2,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        query_tile_size=8,
        **({"bounded": True, "max_queries": 32} if bounded else {}),
    )
    inputs = [_inputs(257, 17), _inputs(129, 9)]
    current = torch.cuda.current_stream()
    results, selections = [], []
    for data, prefix in zip(inputs, [257, 129], strict=True):
        selection = BlockSelection(
            torch.tensor([0, 2], device="cuda", dtype=torch.int64).reshape(1, 1, 2), 64
        )
        selections.append(selection)
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            results.append(_call(workspace, data, selection, prefix))
    workspace.synchronize()
    for actual, data, selection, prefix in zip(
        results, inputs, selections, [257, 129], strict=True
    ):
        q, keys, values, _, _, cis = data
        expected = reference_nosa_block_sparse_attention(q, keys, values, selection, prefix, cis)
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
    # The second invocation has one token in block 2; counts are per invocation.
    assert workspace.last_transfer_bytes.item() == (64 + 1) * 2 * 128 * 2 * 2
    # Poison the entire reusable capacity, including tiles/pages outside the
    # current query. Empty preparation must reset it before returning as well.
    workspace._first_use.fill_(-19)
    workspace._ready_blocks.fill_(23)
    workspace._fetch_queue.fill_(29)
    workspace._tile_bytes.fill_(31)
    workspace.last_transfer_bytes.fill_(37)
    empty = _inputs(64, 0)
    selection = BlockSelection(torch.zeros(1, 1, 1, device="cuda", dtype=torch.int64), 64)
    assert _call(workspace, empty, selection, 64).shape == (0, 32, 128)
    workspace.synchronize()
    assert workspace.last_transfer_bytes.item() == 0
    assert workspace.last_tile_transfer_bytes.numel() == 0
    assert (workspace._first_use == torch.iinfo(torch.int32).max).all().item()
    assert torch.count_nonzero(workspace._ready_blocks).item() == 0
    assert torch.count_nonzero(workspace._tile_bytes).item() == 0
    assert workspace._fetch_queue[:2].cpu().tolist() == [0, 0]


def test_cuda_overlap_reused_workspace_retains_outputs_with_varying_queries(hopper):
    prefix, tile_size = 129, 128
    workspace = NosaFetchWorkspace(
        512, 2, 128, device="cuda", dtype=torch.bfloat16, query_tile_size=tile_size
    )
    retained = []
    for iteration, queries in enumerate((129, 1, 0, 17, 129)):
        inputs = _inputs(prefix, queries)
        q, keys, values, host_keys, host_values, cis = inputs
        q.zero_()
        keys.zero_()
        value = 2.0**iteration
        values.fill_(value)
        cis.zero_()
        host_keys[:prefix].copy_(keys[:prefix])
        host_values[:prefix].copy_(values[:prefix])
        ids = torch.tensor([0, 2], device="cuda", dtype=torch.int64).reshape(1, 1, 2)
        valid = torch.full_like(ids, queries != 17, dtype=torch.bool)
        selection = BlockSelection(ids, 64, valid)
        actual = _call(workspace, inputs, selection, prefix)
        # Retain the returned tensor itself: cloning would hide output aliasing
        # with reusable scratch overwritten by the next call.
        retained.append((actual, 0.0 if queries == 17 else value))
        workspace.synchronize()
        expected_bytes = _expected_tile_bytes(selection, prefix, queries, tile_size)
        assert workspace.last_tile_transfer_bytes.cpu().tolist() == expected_bytes
        assert workspace.last_transfer_bytes.item() == sum(expected_bytes)
    for output, expected_value in retained:
        torch.testing.assert_close(output, torch.full_like(output, expected_value), atol=0, rtol=0)


@pytest.mark.parametrize("bounded", [False, True])
def test_cuda_overlap_reused_scratch_crosses_fa3_work_order_boundary(hopper, bounded):
    prefix, tile_size = 2048, 128
    workspace = NosaFetchWorkspace(
        4096,
        2,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        query_tile_size=tile_size,
        **({"bounded": True, "max_queries": 1032} if bounded else {}),
    )
    initial_bytes = workspace.capacity_bytes
    pointers = [tensor.data_ptr() for tensor in workspace.tensors()]
    # With two KV heads, 1017..1024 queries require 256 work items and the
    # separate counts/order region; adjacent lengths use a different layout.
    for queries in (1016, 1017, 1024, 1025, 1017):
        inputs = _inputs(prefix, queries)
        q, keys, values, host_keys, host_values, cis = inputs
        q.zero_()
        keys.zero_()
        cis.zero_()
        head = torch.arange(2, device="cuda")
        codes = torch.arange(len(keys), device="cuda")[:, None] // 64 + head[None, :] * 32
        values.copy_(codes[..., None])
        host_keys[:prefix].copy_(keys[:prefix])
        host_values[:prefix].copy_(values[:prefix])
        row = torch.arange(queries, device="cuda")[:, None]
        group = row // 8
        ids = ((group + row % 8 * (group % 4) + head[None, :] * 11) % 32)[..., None]
        selection = BlockSelection(ids, 64)
        # Every query selects one full constant-valued block. Varying union
        # sizes across groups exercise work sorting without a tolerance hiding
        # wrong output-row or scratch-order indexing.
        expected = (ids[..., 0] + head[None, :] * 32).to(q.dtype)
        expected = expected.repeat_interleave(16, dim=1)[..., None].expand_as(q)
        actual = _call(workspace, inputs, selection, prefix)
        workspace.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        expected_bytes = _expected_tile_bytes(selection, prefix, queries, tile_size)
        assert workspace.last_tile_transfer_bytes.cpu().tolist() == expected_bytes
        assert workspace.last_transfer_bytes.item() == sum(expected_bytes)
        if bounded:
            assert workspace.capacity_bytes == initial_bytes
            assert [tensor.data_ptr() for tensor in workspace.tensors()] == pointers


def test_cuda_overlap_repair_flags_reset_after_poison_and_empty_rows(hopper):
    prefix, queries, tile_size = 320, 24, 8
    inputs = _inputs(prefix, queries)
    q, keys, values, host_keys, host_values, cis = inputs
    q.zero_()
    keys.zero_()
    cis.zero_()
    host_keys[:prefix].copy_(keys[:prefix])
    ids = torch.zeros(queries, 2, 1, device="cuda", dtype=torch.int64)
    ids[8:16] = 1
    ids[16:] = 5  # The selected physical tail contains only the current suffix.
    valid = torch.ones_like(ids, dtype=torch.bool)
    selection = BlockSelection(ids, 64, valid)
    workspace = NosaFetchWorkspace(
        prefix + queries, 2, 128, device="cuda", dtype=q.dtype, query_tile_size=tile_size
    )
    for poison, all_masked in (
        (float("nan"), False),
        (None, False),
        (float("inf"), False),
        (None, True),
        (None, False),
    ):
        values.fill_(1)
        if poison is not None:
            values[0, 0, 17] = poison
        host_values[:prefix].copy_(values[:prefix])
        valid.fill_(not all_masked)
        valid[0, 1] = False
        actual = _call(workspace, inputs, selection, prefix)
        expected = reference_nosa_block_sparse_attention(q, keys, values, selection, prefix, cis)
        workspace.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016, equal_nan=True)
        assert torch.equal(actual[0, 16:], torch.zeros_like(actual[0, 16:]))
        assert torch.isfinite(actual[8:]).all()
        if poison is None:
            assert torch.isfinite(actual).all()
        expected_bytes = _expected_tile_bytes(selection, prefix, queries, tile_size)
        assert workspace.last_tile_transfer_bytes.cpu().tolist() == expected_bytes
        assert workspace.last_transfer_bytes.item() == sum(expected_bytes)


def test_cuda_overlap_caller_stream_consumption_survives_temporary_input_reuse(hopper):
    workspace = NosaFetchWorkspace(
        512, 2, 128, device="cuda", dtype=torch.bfloat16, query_tile_size=8
    )
    default = torch.cuda.current_stream()
    originals = [_inputs(257, 17), _inputs(129, 9)]
    ids = torch.tensor([0, 2], device="cuda", dtype=torch.int64).reshape(1, 1, 2)
    selection = BlockSelection(ids, 64)
    references = [
        reference_nosa_block_sparse_attention(q, keys, values, selection, prefix, cis)
        for (q, keys, values, _, _, cis), prefix in zip(originals, (257, 129), strict=True)
    ]
    outputs, consumed, streams, pressure = [], [], [], []
    # The pinned backing in originals remains alive until all caller work is
    # complete; only temporary device inputs are freed to stress stream ownership.
    for inputs, prefix in zip(originals, (257, 129), strict=True):
        temporary = tuple(tensor.clone() if tensor.is_cuda else tensor for tensor in inputs)
        temporary_selection = BlockSelection(ids.clone(), 64)
        stream = torch.cuda.Stream()
        streams.append(stream)
        stream.wait_stream(default)
        with torch.cuda.stream(stream):
            # Keep the GPU consumer pending while the host submits allocator
            # pressure on the allocating stream after releasing temporary inputs.
            torch.cuda._sleep(1_000_000)
            output = _call(workspace, temporary, temporary_selection, prefix)
            outputs.append(output)
            consumed.append(output.float().clone())
        del temporary, temporary_selection
        pressure.extend(torch.empty_like(tensor).fill_(23) for tensor in inputs if tensor.is_cuda)
    for stream in streams:
        stream.synchronize()
    for output, immediate, expected in zip(outputs, consumed, references, strict=True):
        torch.testing.assert_close(output, expected, atol=0.016, rtol=0.016)
        torch.testing.assert_close(immediate, output.float(), atol=0, rtol=0)
    workspace.synchronize()
    assert workspace.last_transfer_bytes.item() == (64 + 1) * 2 * 128 * 2 * 2


@pytest.mark.parametrize(
    "bad", ["host", "suffix", "gqa", "q_stride", "cis", "q_alias", "suffix_alias", "triton"]
)
def test_cuda_overlap_rejects_unsupported_inputs(hopper, monkeypatch, bad):
    inputs = list(_inputs(64, 8))
    workspace = NosaFetchWorkspace(128, 2, 128, device="cuda", dtype=torch.bfloat16)
    selection = BlockSelection(torch.zeros(1, 1, 1, device="cuda", dtype=torch.int64), 64)
    if bad == "host":
        inputs[3] = torch.empty(inputs[3].shape, dtype=inputs[3].dtype)
    elif bad == "suffix":
        inputs[2] = inputs[2][:-1]
    elif bad == "gqa":
        inputs[0] = inputs[0][:, :16]
    elif bad == "q_stride":
        inputs[0] = inputs[0][:1].expand(8, -1, -1)
    elif bad == "cis":
        inputs[5] = inputs[5][:-1]
    elif bad == "q_alias":
        inputs[0] = workspace.values.reshape(8, 32, 128)
    elif bad == "suffix_alias":
        inputs[1] = workspace.keys[:72]
    else:
        monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "triton")
    with pytest.raises((ValueError, NotImplementedError)):
        _call(workspace, inputs, selection, 64)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_seq_len": 0},
        {"kv_heads": True},
        {"head_dim": 64},
        {"dtype": torch.float16},
        {"query_tile_size": 0},
        {"overlap": 1},
    ],
)
def test_offload_workspace_rejects_invalid_configuration(kwargs):
    config = {
        "max_seq_len": 128,
        "kv_heads": 2,
        "head_dim": 128,
        "device": "cpu",
        "dtype": torch.bfloat16,
    }
    config.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        NosaFetchWorkspace(**config)


def test_offload_workspace_requires_gpu():
    with pytest.raises(NotImplementedError, match="SM90"):
        NosaFetchWorkspace(128, 2, 128, device="cpu", dtype=torch.bfloat16)
