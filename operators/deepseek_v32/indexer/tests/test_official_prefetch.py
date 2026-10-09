"""Exact publication and cleanup of official Q1 prefetch staging."""

import pytest
import torch

from operators.deepseek_v32.indexer import cache_ops, official_decode, official_prefetch

MISSING = 2**31 - 1


def _require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; explicit GPU entry requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("Official prefetch adapter validation requires SM90")


def _lease(*, slots=96, resident=0, history=128, host_capacity=384):
    _require_sm90()
    device = torch.device("cuda", torch.cuda.current_device())
    host = (
        torch.arange(host_capacity * 576, dtype=torch.float32)
        .remainder(997)
        .to(torch.bfloat16)
        .reshape(host_capacity, 576)
        .pin_memory()
    )
    records = torch.zeros((slots + 1, 576), dtype=torch.bfloat16, device=device)
    h2d = torch.full((host_capacity,), MISSING, dtype=torch.int32, device=device)
    d2h = torch.full((slots + 1,), MISSING, dtype=torch.int64, device=device)
    ages = torch.full((slots + 1,), -1, dtype=torch.int64, device=device)
    ages[0] = MISSING
    if resident:
        ids = torch.arange(host_capacity - resident, host_capacity, device=device)
        physical = torch.arange(1, resident + 1, device=device)
        records[physical] = host[ids.cpu()].to(device)
        h2d[ids] = physical.int()
        d2h[physical] = ids
        ages[physical] = 7
    return {
        "host": host,
        "device": records,
        "host_to_device": h2d,
        "device_to_host": d2h,
        "free_slots": (ages[1:].argsort(stable=True) + 1).int(),
        "allocation_log": torch.full_like(d2h, MISSING),
        "counter": torch.tensor([99], dtype=torch.uint32, device=device),
        "prefetch_stats": torch.full((3,), 99, dtype=torch.int64, device=device),
        "page_table": torch.tensor([0, 2, 1, 3, 4, 5], dtype=torch.int32, device=device),
        "offset": torch.zeros(16, dtype=torch.float32, device=device),
        "history_length": history,
        "max_prefetch": slots - 1,
        "priority": ages,
        "free_bitmap": (d2h == MISSING) & (torch.arange(slots + 1, device=device) > 0),
        "clock": torch.tensor([8], dtype=torch.int64, device=device),
    }


def _context(lease):
    return torch.tensor(
        [lease["history_length"] + 1], dtype=torch.int32, device=lease["device"].device
    )


def _stage(lease, state, attempts):
    count = min(attempts, 64)
    ids = torch.arange(count, dtype=torch.int32, device=lease["device"].device)
    state.host_ids[0, :count] = ids
    state.records[0, :count] = lease["host"][:count].to(lease["device"].device)
    lease["host_to_device"][ids.long()] = state.pool_rows + ids
    state.counter.fill_(attempts)
    return ids.long()


@pytest.mark.parametrize("resident", [0, 1, 32])
def test_official_prepare_consumes_bounded_free_token(resident):
    lease = _lease(resident=resident)
    scratch = torch.empty(96, dtype=torch.int64, device=lease["device"].device)
    token = cache_ops.prepare_prefetch_free(
        lease["priority"],
        lease["free_bitmap"],
        lease["device_to_host"],
        lease["free_slots"],
        lease["allocation_log"],
        lease["counter"],
        lease["prefetch_stats"],
        scratch,
        timestamp=8,
    )
    lease.update(_prepared=token, prepared_limit=64, max_prefetch=64)
    original_maps = lease["host_to_device"].clone(), lease["device_to_host"].clone()
    state = official_prefetch._prepare(lease, _context(lease), 192)
    torch.cuda.synchronize()
    assert token.used
    assert lease["free_slots"][:64].cpu().tolist() == list(range(resident + 1, resident + 65))
    assert lease["free_slots"][64:].eq(MISSING).all()
    assert lease["allocation_log"].eq(MISSING).all()
    assert lease["counter"].item() == 0 and lease["prefetch_stats"].eq(0).all()
    assert state.counter.item() == 0
    assert torch.equal(original_maps[0], lease["host_to_device"])
    assert torch.equal(original_maps[1], lease["device_to_host"])
    with pytest.raises(ValueError, match="staging already belongs"):
        official_prefetch._prepare(lease, _context(lease), 192)
    with pytest.raises(ValueError, match="consumed"):
        token.consume(lease, 1, consumer_limit=64)


@pytest.mark.parametrize("attempts", [0, 1, 17, 64, 91])
@pytest.mark.parametrize("resident", [0, 80, 96])
def test_promote_exact_records_maps_fifo_journal_and_counters(attempts, resident):
    lease = _lease(resident=resident)
    original_h2d = lease["host_to_device"].cpu().clone()
    original_d2h = lease["device_to_host"].cpu().clone()
    original_records = lease["device"].cpu().clone()
    ages = lease["priority"].cpu().clone()
    state = official_prefetch._prepare(lease, _context(lease), 192)
    expected_tokens = torch.cat((torch.arange(64), torch.arange(128, 192), torch.zeros(64)))
    assert torch.equal(state.page_table.cpu()[0], expected_tokens.int())
    assert state.host_ids.cpu().eq(-1).all()
    assert state.counter.item() == 0
    ids = _stage(lease, state, attempts)
    count = len(ids)
    slots = lease["free_slots"][:count].cpu().long()
    old = original_d2h[slots]
    evicted = old[old != MISSING]
    official_prefetch._promote(state, lease)
    lease[official_prefetch.CLEANUP_KEY]()
    lease[official_prefetch.CLEANUP_KEY]()
    expected_h2d, expected_d2h = original_h2d.clone(), original_d2h.clone()
    expected_h2d[evicted] = MISSING
    expected_h2d[ids.cpu()] = slots.int()
    expected_d2h[slots] = ids.cpu()
    expected_records = original_records.clone()
    expected_records[slots] = lease["host"][:count]
    expected_log = torch.full_like(original_d2h, MISSING)
    expected_log[slots] = ids.cpu()
    assert torch.equal(lease["device"].cpu(), expected_records)
    assert torch.equal(lease["host_to_device"].cpu(), expected_h2d)
    assert torch.equal(lease["device_to_host"].cpu(), expected_d2h)
    assert torch.equal(lease["allocation_log"].cpu(), expected_log)
    assert lease["prefetch_stats"].tolist() == [count, len(evicted), attempts - count]
    assert torch.equal(lease["priority"].cpu(), ages), "promotion must leave finalization to cache"
    cache_ops.finalize_prefetch(
        lease["priority"], lease["free_bitmap"], lease["allocation_log"], lease["clock"]
    )
    ages[slots] = 8
    assert torch.equal(lease["priority"].cpu(), ages)
    expected_free = (expected_d2h == MISSING) & (torch.arange(len(expected_d2h)) > 0)
    assert torch.equal(lease["free_bitmap"].cpu(), expected_free)
    assert lease["clock"].item() == 9


@pytest.mark.parametrize("published", [0, 1, 31, 64])
def test_cleanup_keeps_published_mappings_and_clears_host_zero(published):
    lease = _lease()
    state = official_prefetch._prepare(lease, _context(lease), 129)
    _stage(lease, state, 97)
    ids = torch.arange(published, device=lease["device"].device)
    lease["host_to_device"][ids] = (ids + 1).int()
    before_d2h = lease["device_to_host"].clone()
    before_records = lease["device"].clone()
    state.clear()
    state.clear()
    expected = torch.full_like(lease["host_to_device"], MISSING)
    expected[ids] = (ids + 1).int()
    assert torch.equal(lease["host_to_device"], expected)
    assert torch.equal(lease["device_to_host"], before_d2h)
    assert torch.equal(lease["device"], before_records)


def test_wrapper_registers_cleanup_before_raw_bridge_failure(monkeypatch):
    lease = _lease()
    expected_error = RuntimeError("injected failure after stage publication")

    def fail(*args):
        assert callable(lease[official_prefetch.CLEANUP_KEY])
        _stage(lease, lease[official_prefetch.STATE_KEY], 64)
        raise expected_error

    monkeypatch.setattr(official_decode, "logits", fail)
    with pytest.raises(RuntimeError) as raised:
        official_prefetch.logits(None, None, None, _context(lease), None, None, 129, lease)
    assert raised.value is expected_error
    assert lease["host_to_device"][0].item() == len(lease["device"])
    lease[official_prefetch.CLEANUP_KEY]()
    assert lease["host_to_device"].eq(MISSING).all()
    assert lease["allocation_log"].eq(MISSING).all()


def test_prepared_token_is_consumed_and_low_headroom_rejected():
    lease = _lease()
    scratch = torch.empty(96, dtype=torch.int64, device=lease["device"].device)
    lease["_prepared"] = cache_ops.prepare_prefetch(
        lease["priority"],
        lease["free_slots"],
        lease["allocation_log"],
        lease["counter"],
        lease["prefetch_stats"],
        scratch,
        timestamp=8,
        host_capacity=len(lease["host"]),
        query_count=1,
    )
    official_prefetch._prepare(lease, _context(lease), 129)
    assert lease["_prepared"].used
    with pytest.raises(ValueError, match="already belongs"):
        official_prefetch._prepare(lease, _context(lease), 129)
    lease = _lease(slots=64)
    with pytest.raises(ValueError, match="64 prepared"):
        official_prefetch._prepare(lease, _context(lease), 129)


def test_actual_official_scores_staging_and_promotion():
    lease = _lease(slots=320, history=256)
    device = lease["device"].device
    torch.manual_seed(7301)
    n = 257
    pages = (n + 63) // 64
    keys = torch.randn((n, 128), device=device).to(torch.float8_e4m3fn)
    scales = torch.ones(n, dtype=torch.float32, device=device)
    q = torch.randn((1, 1, 64, 128), device=device).to(torch.float8_e4m3fn)
    weights = torch.ones((1, 64), dtype=torch.float32, device=device)
    packed = torch.zeros((pages, 64 * 132), dtype=torch.uint8, device=device)
    for page in range(pages):
        count = min(64, n - page * 64)
        packed[page, : count * 128] = (
            keys[page * 64 : page * 64 + count].view(torch.uint8).flatten()
        )
        packed[page, 8192 : 8192 + count * 4] = scales[page * 64 : page * 64 + count].view(
            torch.uint8
        )
    blocks = torch.arange(pages, dtype=torch.int32, device=device).view(1, -1)
    context = _context(lease)
    schedule = official_decode.metadata(context)
    scores = official_prefetch.logits(q, packed, weights, context, blocks, schedule, n, lease)
    state = lease[official_prefetch.STATE_KEY]
    count = min(state.counter.item(), 64)
    assert count == 64
    actual_ids = state.host_ids[0, :count].long()
    assert actual_ids.unique().numel() == count
    assert torch.isin(actual_ids.cpu(), state.page_table.cpu()[0, :256]).all()
    selected_slots = lease["host_to_device"][actual_ids].long()
    assert selected_slots.ge(1).all() and selected_slots.lt(len(lease["device"])).all()
    assert torch.equal(lease["device_to_host"][selected_slots], actual_ids)
    assert torch.equal(lease["device"][selected_slots].cpu(), lease["host"][actual_ids.cpu()])
    assert torch.equal(state.records[0, :count].cpu(), lease["host"][actual_ids.cpu()])
    assert lease["prefetch_stats"][0].item() == count
    state.clear()
    assert lease["host_to_device"].max().item() == MISSING
    reference = official_decode.logits(
        q,
        packed,
        weights,
        context,
        blocks,
        schedule,
        n,
        state.page_table,
        lease["device"],
        lease["host"],
        torch.full_like(state.host_ids, -1),
        torch.empty_like(state.records),
        torch.full_like(lease["host_to_device"], 1),
        torch.zeros_like(state.counter),
        lease["offset"][1:2],
    )
    assert torch.equal(scores, reference)
