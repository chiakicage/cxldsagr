"""GPU-only suffix ownership and exact mixed history/candidate addressing."""

import pytest
import torch

from cache.sparse_token_cache import SparseTokenCache, WorkingSetTooLarge
from cache.sparse_token_pool import MISSING, SharedSparseTokenPool


def make_pool(*, slots=4, candidate_slots=5, layers=1, device="cpu"):
    pool = SharedSparseTokenPool(
        128,
        7,
        layers,
        slots,
        device=device,
        dtype=torch.float32,
        candidate_slots=candidate_slots,
    )
    for layer in pool.layers:
        layer.host.fill_(-77)
    return pool


def populate(cache, count, offset=0):
    values = torch.arange(
        offset, offset + count * cache.width, dtype=cache.records.dtype, device=cache.device
    ).reshape(count, cache.width)
    cache.begin_step(count)
    for chunk in values.split(cache.slots):
        cache.append(chunk)
    cache.commit()
    return values


def assert_selection(cache, logical, values):
    physical = cache.ensure(logical)
    valid = logical >= 0
    torch.testing.assert_close(cache.records[physical[valid].long()], values[logical[valid]])
    assert (physical[~valid] == -1).all()
    return physical


def test_candidate_storage_adds_only_device_records_not_host_or_maps():
    pool = make_pool(layers=2)
    estimate = pool.estimate_shared_bytes(
        128, 7, 2, 4, dtype=torch.float32, device="cpu", candidate_slots=5
    )
    assert pool.shared_bytes() == estimate
    plain = pool.estimate_shared_bytes(128, 7, 2, 4, dtype=torch.float32, device="cuda")
    extended = pool.estimate_shared_bytes(
        128, 7, 2, 4, dtype=torch.float32, device="cuda", candidate_slots=5
    )
    assert extended["hbm"] - plain["hbm"] == 2 * 5 * 7 * 4
    assert extended["dram"] == plain["dram"]
    for layer in pool.layers:
        assert layer.records.shape == (10, 7)
        assert layer.host.shape == (128, 7)
        assert layer.host_to_device.shape == (128,)
        assert layer.device_to_host.shape == layer.priority.shape == layer.free.shape == (5,)
    session = pool.allocate_session(64)
    assert session.capacity == session.host_tokens == 64
    assert session.host_pages == 1


def test_multichunk_candidates_never_write_host_and_preserve_history(monkeypatch):
    pool = make_pool()
    session = pool.allocate_session(64)
    cache = session.layer(0)
    history = populate(cache, 64)
    host_before = cache.host.clone()
    maps_before = cache.host_to_device.clone(), cache.device_to_host.clone()
    cache.reset_stats()

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate append attempted host writeback")

    monkeypatch.setattr(pool, "write_host", forbidden)
    candidate = torch.arange(35, dtype=torch.float32).reshape(5, 7) + 1000
    cache.begin_transient(5)
    assert cache.transient_start == cache.host_written_end == cache.length == 64
    cache.declare_indexer_visible(66)
    cache.append(candidate[:2])
    torch.testing.assert_close(cache.host_to_device, maps_before[0])
    torch.testing.assert_close(cache.device_to_host, maps_before[1])
    values = torch.cat((history, candidate))
    first = torch.tensor([[0, 63, 64, 65, -1]])
    assert_selection(cache, first, values)
    cache.declare_indexer_visible(69)
    cache.append(candidate[2:])
    mixed = torch.tensor([[1, 62, 64, 65, 66, 67, 68, 64, -2]])
    physical = assert_selection(cache, mixed, values)
    assert physical[0, 2:7].tolist() == [5, 6, 7, 8, 9]
    assert cache.written == cache.indexer_visible_end == 69
    assert cache.length == cache.host_written_end == 64
    assert cache.stats.transient_written_records == cache.stats.written_records == 5
    assert cache.metrics()["device_to_host_bytes"] == 0
    assert pool.pending_source_bytes == {"hbm": 0, "dram": 0}
    torch.testing.assert_close(cache.host, host_before)
    torch.testing.assert_close(cache.host_records(), history)
    with pytest.raises(ValueError, match="host-backed"):
        cache.host_records(65)
    with pytest.raises(ValueError, match="GPU-only"):
        cache.logical_to_global(torch.tensor([64]))
    with pytest.raises(RuntimeError, match="cannot be committed"):
        cache.commit()
    cache.discard_transient()
    assert cache.length == cache.written == cache.indexer_visible_end == 64
    assert cache.transient_start is None
    assert pool._transient_owners == {}
    torch.testing.assert_close(cache.host, host_before)
    with pytest.raises(ValueError, match="unwritten"):
        cache.ensure(torch.tensor([64]))
    assert session.host_pages == 1


def test_exact_recall_splits_only_history_union_and_cannot_evict_candidate_tail():
    pool = make_pool(slots=2)
    cache = pool.allocate_session(8).layer(0)
    history = populate(cache, 8)
    candidate = torch.randn(5, 7)
    cache.begin_step(5, transient=True)
    # The standalone cache operation can write more candidates than P: these
    # rows are separately allocated and do not displace any history slot.
    cache.append(candidate)
    values = torch.cat((history, candidate))
    for history_ids in ([0, 1], [6, 7], [2, 3]):
        ids = torch.tensor([history_ids + list(range(8, 13))])
        physical = assert_selection(cache, ids, values)
        assert physical[0, 2:].tolist() == [3, 4, 5, 6, 7]
        torch.testing.assert_close(cache.records[3:], candidate)
    with pytest.raises(WorkingSetTooLarge, match="3 selected history"):
        cache.ensure(torch.tensor([0, 1, 2, 8, 9, 10, 11, 12]))
    assert cache.stats.max_working_set == 7
    cache.rollback()
    assert cache.length == cache.written == 8
    assert all(int(x) < 8 or int(x) == MISSING for x in cache.device_to_host)


def test_prefetch_uses_history_view_and_never_fetches_prior_candidate_chunks():
    pool = make_pool()
    cache = pool.allocate_session(8).layer(0)
    history = populate(cache, 8)
    host_before = cache.host.clone()
    cache.begin_transient(3)
    cache.declare_indexer_visible(9)
    state = cache.prepare_prefetch(8, 1, torch.zeros(16))
    assert state["device"].shape == (5, 7)
    assert state["device"].data_ptr() == cache.records.data_ptr()
    assert state["history_length"] == 8
    assert state["transient_suffix"] is True
    assert state["max_prefetch"] == 4
    assert len(state["page_table"]) == 1
    cache.prefetch_reference([0, 8, 9, 100])
    assert int(state["counter"]) == 1
    cache.finalize_prefetch(state)
    first = torch.full((1, 7), 900.0)
    cache.append(first)
    cache.declare_indexer_visible(11)
    state = cache.prepare_prefetch(9, 2, torch.zeros(16))
    assert state["history_length"] == 8
    cache.prefetch_reference([1, 8, 9, 10])
    assert int(state["counter"]) == 1
    cache.finalize_prefetch(state)
    rest = torch.full((2, 7), 901.0)
    cache.append(rest)
    assert_selection(cache, torch.tensor([0, 1, 8, 9, 10]), torch.cat((history, first, rest)))
    torch.testing.assert_close(cache.host, host_before)
    cache.discard_transient()


def test_per_layer_candidate_ownership_lasts_between_chunks_until_discard():
    pool = make_pool(layers=2)
    a, b = pool.allocate_session(8), pool.allocate_session(8)
    ca, cb = a.layer(0), b.layer(0)
    populate(ca, 4)
    populate(cb, 4, offset=100)
    ca.begin_transient(2)
    ca.append(torch.full((1, 7), 10.0))
    with pytest.raises(RuntimeError, match="another session owns transient"):
        cb.begin_step(1)
    assert cb._step_end is None
    assert cb.length == cb.written == cb.indexer_visible_end == 4
    with pytest.raises(RuntimeError, match="another session owns transient"):
        cb.begin_transient(1)
    with pytest.raises(RuntimeError, match="another session owns transient"):
        cb.ensure(torch.tensor([0]))
    # Each layer owns distinct candidate storage; another layer is independent.
    b.layer(1).begin_transient(1)
    b.layer(1).append(torch.full((1, 7), 30.0))
    ca.append(torch.full((1, 7), 20.0))
    torch.testing.assert_close(
        ca.records[ca.ensure(torch.tensor([4, 5])).long(), 0], torch.tensor([10.0, 20.0])
    )
    ca.discard_transient()
    cb.begin_transient(1)
    cb.append(torch.full((1, 7), 40.0))
    assert float(cb.records[cb.ensure(torch.tensor([4])).long(), 0]) == 40.0
    b.release()
    assert pool._transient_owners == {}
    assert cb.records is None
    a.release()
    pool.close()
    assert pool.shared_bytes() == {"hbm": 0, "dram": 0}


def test_partial_rollback_finalizes_pending_prefetch_and_preserves_snapshot():
    pool = make_pool()
    cache = pool.allocate_session(8).layer(0)
    history = populate(cache, 8)
    snapshot = pool.snapshot()
    cache.begin_transient(3)
    cache.append(torch.ones(1, 7))
    cache.declare_indexer_visible(10)
    cache.prepare_prefetch(9, 1, torch.zeros(16))
    cache.prefetch_reference([0, 8, 9])
    with pytest.raises(RuntimeError, match="committed"):
        pool.snapshot()
    cache.rollback()
    assert cache.length == cache.written == cache.indexer_visible_end == 8
    assert pool._pending_prefetch is None
    assert pool._transient_owners == {}
    torch.testing.assert_close(cache.host_records(), history)
    pool.restore(snapshot)
    cache.begin_transient(1)
    cache.append(torch.full((1, 7), 777.0))
    assert float(cache.records[cache.ensure(torch.tensor([8])).long(), 0]) == 777.0
    cache.discard_transient()


def test_candidate_bounds_reject_without_mutating_ownership_or_host():
    pool = make_pool()
    cache = pool.allocate_session(8).layer(0)
    populate(cache, 4)
    for count in (0, -1, 6, True):
        with pytest.raises(ValueError, match="candidate slot"):
            cache.begin_transient(count)
        assert pool._transient_owners == {}
        assert cache.transient_start is None
    cache.begin_step(1)
    with pytest.raises(RuntimeError, match="already active"):
        cache.begin_transient(1)
    cache.rollback()
    cache.begin_transient(2)
    with pytest.raises(ValueError, match="unwritten"):
        cache.ensure(torch.tensor([4]))
    with pytest.raises(ValueError, match="invalid appended"):
        cache.append(torch.ones(3, 7))
    assert cache.written == 4
    cache.rollback()
    with pytest.raises(RuntimeError, match="no transient"):
        cache.discard_transient()
    resident = SparseTokenCache(8, 7, device="cpu")
    with pytest.raises(ValueError, match="candidate slot"):
        resident.begin_transient(1)


@pytest.mark.parametrize("history_length", [4, 8])
def test_resident_transient_suffix_keeps_history_and_maps_with_changing_candidates(history_length):
    cache = SparseTokenCache(8, 7, device="cpu", dtype=torch.float32, candidate_slots=5)
    history = populate(cache, history_length)
    assert cache.capacity == cache.slots == 8
    assert cache.records.shape == (13, 7)
    assert cache.host is cache.page_table is None
    maps = [value.clone() for value in (cache.host_to_device, cache.device_to_host, cache.age)]
    assert all(len(value) == 8 for value in maps)
    for candidate_count in (2, 5, 1):
        cache.reset_stats()
        candidate = torch.arange(candidate_count * 7, dtype=torch.float32).reshape(-1, 7) + 1000
        cache.begin_transient(candidate_count)
        cache.declare_indexer_visible(history_length + candidate_count)
        cache.append(candidate)
        assert cache.length == cache.transient_start == history_length
        assert cache.written == cache.indexer_visible_end == history_length + candidate_count
        selection = torch.tensor(
            [[0, history_length - 1, *range(history_length, history_length + candidate_count), -1]]
        )
        physical = assert_selection(cache, selection, torch.cat((history, candidate)))
        assert torch.equal(physical, selection.int())
        for actual, before in zip(
            (cache.host_to_device, cache.device_to_host, cache.age), maps, strict=True
        ):
            assert torch.equal(actual, before)
        assert torch.equal(cache.records[:history_length], history)
        metrics = cache.metrics()
        assert metrics["written_records"] == metrics["transient_written_records"] == candidate_count
        assert metrics["device_to_host_bytes"] == metrics["host_to_device_bytes"] == 0
        assert metrics["host_record_bytes"] == 0
        with pytest.raises(RuntimeError, match="cannot be committed"):
            cache.commit()
        cache.discard_transient()
        assert cache.length == cache.written == cache.indexer_visible_end == history_length
        assert cache.transient_start is cache._step_end is None
        assert torch.equal(cache.records[:history_length], history)
        with pytest.raises(ValueError, match="unwritten"):
            cache.ensure(torch.tensor([history_length]))
    cache.reset()
    assert cache.length == cache.written == 0


def test_resident_candidate_capacity_does_not_expand_persistent_append_capacity():
    cache = SparseTokenCache(8, 7, device="cpu", dtype=torch.float32, candidate_slots=5)
    populate(cache, 8)
    with pytest.raises(ValueError, match="exceeds capacity"):
        cache.begin_step(1)
    for candidate_count in (0, 6, True):
        with pytest.raises(ValueError, match="candidate slot"):
            cache.begin_transient(candidate_count)
        assert cache.transient_start is cache._step_end is None
    cache.reset()


@pytest.mark.parametrize("candidate_slots", [-1, True, MISSING])
def test_candidate_allocation_bound_is_checked_before_allocation(candidate_slots):
    with pytest.raises(ValueError):
        SharedSparseTokenPool.estimate_shared_bytes(64, 7, 1, 4, candidate_slots=candidate_slots)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_transient_candidates_cross_stream_recall_and_zero_host_write(monkeypatch):
    pool = make_pool(device="cuda", slots=4)
    cache = pool.allocate_session(64).layer(0)
    history = populate(cache, 64)
    expected_host = cache.host_records()
    cache.reset_stats()

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU candidate append attempted D2H")

    monkeypatch.setattr(pool, "write_host", forbidden)
    candidate = torch.randn(5, 7, device="cuda")
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    first.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(first):
        cache.begin_transient(5)
        cache.append(candidate[:2])
    with torch.cuda.stream(second), cache.operation():
        cache.append(candidate[2:])
        ids = torch.tensor([0, 63, 64, 65, 66, 67, 68], device="cuda")
        physical = cache.ensure(ids)
        observed = cache.records[physical.long()].clone()
    cache.discard_transient()
    torch.testing.assert_close(observed, torch.cat((history, candidate))[ids])
    torch.testing.assert_close(cache.host_records(), expected_host)
    assert cache.metrics()["device_to_host_bytes"] == 0
    assert not pool._writes
    pool.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_native_prefetch_view_and_metadata_with_transient_suffix(monkeypatch):
    from operators.deepseek_v32.indexer import cache_ops

    pool = SharedSparseTokenPool(
        64, 576, 1, 4, device="cuda", candidate_slots=3, metadata_ops=cache_ops
    )
    cache = pool.allocate_session(64).layer(0)
    history = populate(cache, 64)
    host_before = cache.host_records()
    cache.reset_stats()

    def forbidden(*args, **kwargs):
        raise AssertionError("native candidate path attempted host writeback")

    monkeypatch.setattr(pool, "write_host", forbidden)
    candidate = torch.randn(3, 576, dtype=torch.bfloat16, device="cuda")
    cache.begin_transient(3)
    with cache.operation():
        for start, end in ((0, 1), (1, 3)):
            cache.declare_indexer_visible(64 + end)
            state = cache.prepare_prefetch(64 + start, end - start, torch.zeros(16, device="cuda"))
            assert state["device"].shape == (5, 576)
            assert state["history_length"] == 64
            assert state["transient_suffix"]
            assert state["max_prefetch"] == 4
            ids = cache.logical_to_global(torch.tensor([start, start + 1], device="cuda"))
            cache_ops.prefetch_ids(ids, state)
            cache.finalize_prefetch(state)
            cache.append(candidate[start:end])
        selected = torch.tensor([0, 1, 64, 65, 66], device="cuda")
        physical = cache.ensure(selected)
        observed = cache.records[physical.long()].clone()
    cache.discard_transient()
    torch.testing.assert_close(observed, torch.cat((history, candidate))[selected])
    torch.testing.assert_close(cache.host_records(), host_before)
    assert cache.metrics()["device_to_host_bytes"] == 0
    assert cache.host_to_device.shape == (64,)
    assert cache.device_to_host.shape == (5,)
    pool.close()
