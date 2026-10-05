"""Shared cache ownership, recorded FIFO events, transactions and transport."""

import pytest
import torch

from cache.sparse_token_cache import WorkingSetTooLarge
from cache.sparse_token_pool import MISSING, PRIORITY_LIMIT, SharedSparseTokenPool


def make_pool(*, slots=4, pages=4, layers=1, device="cpu"):
    return SharedSparseTokenPool(pages * 64, 7, layers, slots, device=device, dtype=torch.float32)


def test_host_estimate_rounds_each_pinned_layer_and_not_cpu_reference():
    estimate = SharedSparseTokenPool.estimate_shared_bytes(
        64, 7, 3, 4, dtype=torch.float32, device="cuda"
    )
    assert estimate["dram"] == 3 * 2048 + 4
    pool = make_pool(pages=1, layers=3)
    assert all(layer.host.untyped_storage().nbytes() == 1792 for layer in pool.layers)
    assert pool.shared_bytes()["dram"] == estimate["hbm"] + 3 * 1792 + 4
    pool.close()


def append(cache, values):
    cache.begin_step(len(values))
    for piece in values.split(cache.slots):
        cache.append(piece)
    cache.commit()


def check_maps(pool):
    for layer in pool.layers:
        assert int(layer.device_to_host[0]) == MISSING
        assert int(layer.priority[0]) == MISSING
        assert not bool(layer.free[0])
        live = torch.where(layer.device_to_host != MISSING)[0]
        global_ids = layer.device_to_host[live]
        torch.testing.assert_close(layer.host_to_device[global_ids].long(), live)
        host_live = torch.where(layer.host_to_device != MISSING)[0]
        torch.testing.assert_close(
            layer.device_to_host[layer.host_to_device[host_live].long()], host_live
        )
        torch.testing.assert_close(layer.free[1:], layer.device_to_host[1:] == MISSING)


def test_page_allocator_uses_noncontiguous_holes_and_impossible_request_is_pure():
    pool = make_pool(pages=4, layers=2)
    a, b, c = [pool.allocate_session(64) for _ in range(3)]
    assert a.page_table.tolist() == [0]
    assert b.page_table.tolist() == [1]
    assert c.page_table.tolist() == [2]
    b.release()
    before = pool.free_host_pages
    with pytest.raises(ValueError, match="insufficient host pages"):
        pool.allocate_session(193)
    assert pool.free_host_pages == before
    d = pool.allocate_session(65)
    assert set(d.page_table.tolist()) == {1, 3}
    cache = d.layer(0)
    ids = torch.tensor([0, 63, 64])
    expected = d.page_table[ids // 64].long() * 64 + ids % 64
    torch.testing.assert_close(cache.logical_to_global(ids), expected)
    contents = torch.arange(65 * 7, dtype=torch.float32).reshape(65, 7)
    append(cache, contents)
    torch.testing.assert_close(cache.host_records(), contents)
    assert pool.shared_bytes() == pool.estimate_shared_bytes(
        256, 7, 2, 4, dtype=torch.float32, device="cpu"
    )
    assert d.session_bytes() == pool.estimate_session_bytes(65, layers=2, device="cpu")
    assert a.layer(0).records.data_ptr() == c.layer(0).records.data_ptr()
    assert a.layer(0).records.data_ptr() != a.layer(1).records.data_ptr()


def test_cross_session_eviction_release_and_reused_host_ids():
    pool = make_pool()
    a, b = pool.allocate_session(64), pool.allocate_session(64)
    ca, cb = a.layer(0), b.layer(0)
    xa = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    xb = xa + 100
    append(ca, xa)
    append(cb, xb)
    assert (ca.host_to_device[ca.logical_to_global(torch.arange(4))] == MISSING).all()
    torch.testing.assert_close(ca.records[ca.ensure(torch.arange(4)).long()], xa)
    torch.testing.assert_close(cb.records[cb.ensure(torch.arange(4)).long()], xb)
    original_pages = a.page_table.clone()
    a.release()
    with pytest.raises(RuntimeError, match="released"):
        ca.ensure(torch.arange(1))
    torch.testing.assert_close(cb.records[cb.ensure(torch.arange(4)).long()], xb)
    c = pool.allocate_session(64)
    assert c.page_table.tolist() == original_pages.tolist()
    cc = c.layer(0)
    append(cc, xa + 200)
    torch.testing.assert_close(cc.records[cc.ensure(torch.arange(4)).long()], xa + 200)
    torch.testing.assert_close(cb.records[cb.ensure(torch.arange(4)).long()], xb)
    check_maps(pool)


def test_released_handles_drop_private_storage_and_close_drops_shared_storage():
    pool = make_pool()
    session = pool.allocate_session(64)
    cache = session.layer(0)
    append(cache, torch.ones(1, 7))
    session.release()
    assert session.page_table is session._pages is session._prefetch_totals is None
    assert session._layers == {}
    assert cache.records is cache.host is cache._prefetch_totals is None
    with pytest.raises(RuntimeError, match="released"):
        cache.metrics()
    pool.close()
    assert pool.shared_bytes() == {"hbm": 0, "dram": 0}
    with pytest.raises(RuntimeError, match="closed"):
        pool.allocate_session(64)


def test_partial_free_recall_preserves_hits_and_only_evicts_deficit():
    pool = make_pool(slots=4)
    a, b = pool.allocate_session(64), pool.allocate_session(64)
    ca, cb = a.layer(0), b.layer(0)
    x = torch.arange(42, dtype=torch.float32).reshape(6, 7)
    append(ca, x)
    # A ends resident on 2..5. B replaces 2 and 3, then releases one slot.
    append(cb, torch.ones(2, 7))
    cb.truncate(1)
    assert int(pool.layers[0].free.sum()) == 1
    before = ca.stats.evicted_records
    ids = torch.tensor([0, 1, 4, 4, -1])
    p = ca.ensure(ids)
    torch.testing.assert_close(ca.records[p[:4].long()], x[ids[:4]])
    assert p[-1] == -1
    assert ca.stats.evicted_records - before == 1
    check_maps(pool)


def test_fifo_event_trace_all_hit_protection_and_empty_finalize():
    # Fixed reference event sequence, matching memory_pool_host.py:1432-1444:
    # append allocation stamp0; exact protection stamp1; empty recall stamp2;
    # empty prefetch finalize stamp3. This is not an access-frequency counter.
    pool = make_pool()
    cache = pool.allocate_session(64).layer(0)
    append(cache, torch.ones(2, 7))
    physical = cache.host_to_device[:2].long()
    assert cache._clock == 1
    assert cache.age[physical].tolist() == [0, 0]
    cache.ensure(torch.tensor([0, 0]))
    assert cache._clock == 3
    assert cache.age[physical].tolist() == [1, 0]
    cache.begin_step(1)
    cache.declare_indexer_visible(3)
    state = cache.prepare_prefetch(2, 1, torch.zeros(16), limit=0)
    assert state is None  # Proven all-hit path retains the empty allocation event.
    assert cache._clock == 4
    assert cache.age[physical].tolist() == [1, 0]
    cache.rollback()
    check_maps(pool)


def test_prefetch_claims_only_real_misses_never_suffix_and_records_exact_count():
    pool = make_pool()
    cache = pool.allocate_session(64).layer(0)
    x = torch.arange(56, dtype=torch.float32).reshape(8, 7)
    append(cache, x)
    cache.begin_step(1)
    cache.declare_indexer_visible(9)
    before_h2d = cache.host_to_device.clone()
    state = cache.prepare_prefetch(8, 1, torch.zeros(16), limit=2)
    torch.testing.assert_close(cache.host_to_device, before_h2d)
    assert len(state["free_slots"]) == cache.slots
    assert not (state["free_slots"] == 0).any()
    cache.prefetch_reference([4, 0, 0, 1, 2, 3, -1, 8, 9])
    assert int(state["counter"].item()) > 2
    cache.finalize_prefetch(state)
    assert cache.metrics()["prefetched_records"] == 2
    assert cache.metrics()["prefetch_capacity_failures"] == 2
    selected = torch.tensor([0, 1])
    torch.testing.assert_close(cache.records[cache.ensure(selected).long()], x[selected])
    assert int(cache.host_to_device[8]) == MISSING
    assert cache.prefetch_counts == []
    cache.rollback()
    check_maps(pool)


def test_new_kv_direct_write_zero_compulsory_recall_and_q_boundaries():
    pool = make_pool()
    cache = pool.allocate_session(64).layer(0)
    cache.begin_step(4)
    cache.declare_indexer_visible(4)
    state = cache.prepare_prefetch(0, 4, torch.zeros(16))
    assert state is None
    x = torch.randn(4, 7)
    cache.append(x)
    physical = cache.ensure(torch.arange(4))
    torch.testing.assert_close(cache.records[physical.long()], x)
    assert cache.metrics()["host_to_device_bytes"] == 0
    cache.commit()
    cache.begin_step(5)
    cache.declare_indexer_visible(9)
    with pytest.raises(WorkingSetTooLarge, match="Q > P"):
        cache.prepare_prefetch(4, 5, torch.zeros(16))
    with pytest.raises(WorkingSetTooLarge, match="query batch"):
        cache.append(torch.ones(5, 7))
    cache.append(torch.ones(4, 7))
    with pytest.raises(WorkingSetTooLarge):
        cache.ensure(torch.arange(5))
    cache.rollback()


def test_consumer_lease_excludes_other_layer_and_pending_prefetch_owns_workspace():
    pool = make_pool(layers=2)
    session = pool.allocate_session(64)
    a, b = session.layer(0), session.layer(1)
    with (
        a.operation(),
        a.operation(),
        pytest.raises(RuntimeError, match="exclusive"),
        b.operation(),
    ):
        pass
    with a.operation():
        with a.operation():
            pass
        with pytest.raises(RuntimeError, match="exclusive"), b.operation():
            pass
    a.begin_step(1)
    a.declare_indexer_visible(1)
    # Unknown nonempty history uses the shared fused workspace.
    a.append(torch.ones(1, 7))
    a.commit()
    pool.invalidate_residency(0)
    a.begin_step(1)
    a.declare_indexer_visible(2)
    a.prepare_prefetch(1, 1, torch.zeros(16))
    with pytest.raises(RuntimeError, match="pending fused"), b.operation():
        pass
    a.rollback()
    with b.operation():
        pass


@pytest.mark.parametrize("body_fails", [False, True])
@pytest.mark.parametrize("failure_point", ["create", "record"])
def test_operation_retains_body_and_completion_errors_and_poisoned_owner(
    monkeypatch, body_fails, failure_point
):
    pool = make_pool()
    session = pool.allocate_session(64)
    body_error = ValueError("attention failed")
    completion_error = KeyboardInterrupt("event failed")
    stream = object()

    class Event:
        def __init__(self):
            if failure_point == "create":
                raise completion_error

        def record(self, actual_stream):
            assert actual_stream is stream
            raise completion_error

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    expected = BaseExceptionGroup if body_fails else KeyboardInterrupt
    with pytest.raises(expected) as caught, pool.operation(session, 0):
        pool.device = torch.device("cuda:0")
        if body_fails:
            raise body_error
    if body_fails:
        assert caught.value.exceptions == (body_error, completion_error)
    else:
        assert caught.value is completion_error
    assert pool.poisoned
    assert pool._active == (session.owner, 0)
    assert pool._sessions[session.owner] is session
    assert pool.layers
    with pytest.raises(RuntimeError, match="poisoned"), pool.operation(session, 0):
        pass


def test_failed_previous_stream_dependency_poison_retains_pool(monkeypatch):
    pool = make_pool()
    session = pool.allocate_session(64)
    pool.device = torch.device("cuda:0")
    previous = object()
    pool._last_stream = previous
    failure = KeyboardInterrupt("dependency interrupted")

    class Event:
        def record(self, stream):
            assert stream is previous
            raise failure

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(KeyboardInterrupt) as caught, pool.operation(session, 0):
        pytest.fail("dependency failure must prevent model execution")
    assert caught.value is failure
    assert pool.poisoned and pool.layers
    assert pool._sessions[session.owner] is session
    assert pool._last_stream is previous


def test_cold_append_plan_matches_original_fifo_with_old_sessions(monkeypatch):
    optimized, reference = make_pool(slots=12), make_pool(slots=12)
    # Keep the pre-optimization allocator as an independent state-machine oracle.
    reference.append_slots = lambda cache, start, count: None
    original_sort = torch.argsort
    sorts = [0, 0]
    active = 0

    def counted_sort(*args, **kwargs):
        sorts[active] += 1
        return original_sort(*args, **kwargs)

    monkeypatch.setattr(torch, "argsort", counted_sort)
    for owner in range(3):
        caches = [pool.allocate_session(12).layer(0) for pool in (optimized, reference)]
        values = torch.arange(84, dtype=torch.float32).reshape(12, 7) + owner * 100
        for cache in caches:
            cache.begin_step(12)
        for start in (0, 4, 8):
            for active, cache in enumerate(caches):
                cache.append(values[start : start + 4])
                selected = torch.tensor([0, start + 3, start + 3, -1])
                physical = cache.ensure(selected)
                torch.testing.assert_close(cache.records[physical[:3].long()], values[selected[:3]])
            assert caches[0].all_history_resident
            for name in ("host_to_device", "device_to_host", "priority", "free"):
                torch.testing.assert_close(
                    getattr(optimized.layers[0], name), getattr(reference.layers[0], name)
                )
            assert caches[0]._clock == caches[1]._clock
            assert vars(caches[0].stats) == vars(caches[1].stats)
        for cache in caches:
            cache.commit()
        check_maps(optimized)
    assert sorts == [3, 9]


def test_residency_proof_invalidates_on_interleave_truncate_restore_and_fused():
    pool = make_pool(slots=8)
    first, second = pool.allocate_session(8), pool.allocate_session(8)
    a, b = first.layer(0), second.layer(0)
    append(a, torch.ones(4, 7))
    assert a.all_history_resident
    with b.operation():
        pass
    assert not a.all_history_resident
    # A restored diagnostic snapshot proves tensor state, not a retained plan.
    snapshot = pool.snapshot()
    pool.restore(snapshot)
    assert not a.all_history_resident
    append(b, torch.ones(4, 7))
    assert b.all_history_resident
    b.truncate(2)
    assert not b.all_history_resident
    b.begin_step(1)
    b.declare_indexer_visible(3)
    state = b.prepare_prefetch(2, 1, torch.zeros(16))
    assert state is not None
    b.finalize_prefetch(state)
    assert not b.all_history_resident
    b.rollback()
    second.release()
    assert not a.all_history_resident


def test_host_copy_uses_cached_fragmented_spans_with_unaligned_appends():
    pool = make_pool(slots=160, pages=5)
    a, hole, c = [pool.allocate_session(64) for _ in range(3)]
    hole.release()
    session = pool.allocate_session(129)
    assert session._pages.tolist() == [1, 3, 4]
    assert session._host_runs == ((0, 64, 64), (64, 192, 192))
    cache = session.layer(0)
    values = torch.arange(129 * 7, dtype=torch.float32).reshape(129, 7)
    cache.begin_step(129)
    for start, end in ((0, 13), (13, 90), (90, 129)):
        cache.append(values[start:end])
    cache.commit()
    torch.testing.assert_close(cache.host_records(), values)
    assert a.host_pages == c.host_pages == 1


def test_empty_allocation_elides_sort_without_changing_metadata(monkeypatch):
    pool = make_pool()
    cache = pool.allocate_session(64).layer(0)
    append(cache, torch.ones(2, 7))
    before = pool.layers[0].priority.clone(), cache._clock, vars(cache.stats).copy()

    def unexpected(*args, **kwargs):
        raise AssertionError("zero-count allocation must not sort")

    monkeypatch.setattr(torch, "argsort", unexpected)
    assert cache._available_slots(torch.empty(0, dtype=torch.int64), 0).numel() == 0
    torch.testing.assert_close(pool.layers[0].priority, before[0])
    assert cache._clock == before[1]
    assert vars(cache.stats) == before[2]


def test_all_hit_recall_does_not_wait_for_host_writeback(monkeypatch):
    pool = make_pool()
    cache = pool.allocate_session(64).layer(0)
    append(cache, torch.ones(4, 7))

    def unexpected(*args, **kwargs):
        raise AssertionError("resident records do not depend on D2H completion")

    monkeypatch.setattr(pool, "wait_host", unexpected)
    cache.ensure(torch.tensor([0, 2, 2, -1]))


def test_priority_rollover_preserves_equal_priority_groups():
    pool = make_pool()
    cache = pool.allocate_session(64).layer(0)
    append(cache, torch.ones(4, 7))
    cache.age[1:] = torch.tensor([500, 500, 700, 900])
    cache._clock = PRIORITY_LIMIT
    pool.stamp(0, torch.tensor([], dtype=torch.int64))
    assert cache.age.tolist() == [MISSING, 0, 0, 1, 2]
    assert cache._clock == 4


def test_snapshot_restores_global_residency_and_rejects_stale_host_prefix():
    pool = make_pool(layers=2)
    a, b = pool.allocate_session(64), pool.allocate_session(64)
    for i in range(2):
        append(a.layer(i), torch.full((3, 7), float(i)))
        append(b.layer(i), torch.full((3, 7), float(i + 10)))
    snapshot = pool.snapshot()
    assert snapshot.diagnostic_bytes["dram"] > 0
    for i in range(2):
        cache = a.layer(i)
        append(cache, torch.ones(1, 7))
        cache.truncate(3)
        cache.ensure(torch.arange(3))
    pool.restore(snapshot)
    for layer, state in zip(pool.layers, snapshot.layers, strict=True):
        for name in ("records", "host_to_device", "device_to_host", "priority", "free"):
            torch.testing.assert_close(getattr(layer, name), state[name])
        assert layer.clock == state["clock"]
    a.layer(0).truncate(2)
    with pytest.raises(ValueError, match="prefix was truncated"):
        pool.restore(snapshot)
    snapshot2 = pool.snapshot()
    b.release()
    pool.allocate_session(64)
    with pytest.raises(ValueError, match="ownership changed"):
        pool.restore(snapshot2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_nondefault_stream_immediate_host_read_and_id_reuse():
    from operators.deepseek_v32.indexer import cache_ops

    pool = make_pool(slots=8, pages=3, device="cuda")
    pool.metadata_ops = cache_ops
    a, b = pool.allocate_session(64), pool.allocate_session(64)
    ca, cb = a.layer(0), b.layer(0)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(s1):
        xa = torch.arange(56, device="cuda", dtype=torch.float32).view(8, 7)
        ca.begin_step(8)
        ca.append(xa)
    with torch.cuda.stream(s2):
        cb.begin_step(8)
        cb.append(torch.full((8, 7), 900.0, device="cuda"))
        observed = ca.records[ca.ensure(torch.arange(8, device="cuda")).long()].clone()
    # Both source storages and writes have bounded lifetimes across layers.
    assert len(pool._writes) <= pool.max_inflight_writes
    pool.drain()
    torch.testing.assert_close(observed, xa)
    ca.commit()
    cb.commit()
    old_pages = a.page_table.cpu().clone()
    a.release()
    c = pool.allocate_session(64)
    torch.testing.assert_close(c.page_table.cpu(), old_pages)
    append(c.layer(0), torch.full((8, 7), 700.0, device="cuda"))
    torch.testing.assert_close(
        cb.records[cb.ensure(torch.arange(8, device="cuda")).long()],
        torch.full((8, 7), 900.0, device="cuda"),
    )
    check_maps(pool)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_native_prefetch_finalize_protect_and_release_integration():
    from operators.deepseek_v32.indexer import cache_ops

    pool = SharedSparseTokenPool(128, 576, 1, 8, device="cuda", metadata_ops=cache_ops)
    assert pool.layers[0].host.shape == (128, 576)
    assert pool.layers[0].host.is_pinned()
    assert pool.layers[0].host.untyped_storage().nbytes() == 262144
    assert pool.shared_bytes()["dram"] == 262144 + 2 * 4
    session = pool.allocate_session(64)
    cache = session.layer(0)
    x = torch.arange(16, device="cuda", dtype=torch.bfloat16)[:, None].expand(-1, 576).contiguous()
    append(cache, x)
    cache.begin_step(1)
    with cache.operation():
        cache.declare_indexer_visible(17)
        state = cache.prepare_prefetch(16, 1, torch.zeros(16, device="cuda"), limit=3)
        ids = cache.logical_to_global(torch.tensor([0, 1, 0, 2, 3, 4], device="cuda"))
        cache_ops.prefetch_ids(ids, state)
        cache.finalize_prefetch(state)
        assert cache.metrics()["prefetched_records"] == 3
        chosen = torch.tensor([0, 1, 2, 15], device="cuda")
        p = cache.ensure(chosen)
        torch.testing.assert_close(cache.records[p.long()], x[chosen])
    cache.rollback()
    pool.drain()
    assert int(pool.layers[0].clock_tensor.item()) == cache._clock
    check_maps(pool)
    session.release()
    assert (pool.layers[0].host_to_device == MISSING).all()
    assert pool.layers[0].free[1:].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_source_reservation_drains_before_next_projection_allocation():
    import weakref

    pool = SharedSparseTokenPool(64, 7, 1, 4, device="cuda", max_inflight_writes=1)
    cache = pool.allocate_session(64).layer(0)
    cache.begin_step(2)
    with cache.operation():
        # Delay the transfer stream so the source really is still live when
        # reserve_append_source runs; an already-complete copy is insufficient.
        with torch.cuda.stream(pool._copy_stream):
            torch.cuda._sleep(50_000_000)
        base = torch.ones((2, 7), device="cuda", dtype=torch.bfloat16)
        source = base[:1]
        expected_storage = base.untyped_storage().nbytes()
        ref = weakref.ref(source)
        cache.append(source)
        assert len(pool._writes) == 1
        assert pool.pending_source_bytes == {"hbm": expected_storage, "dram": 0}
        del source, base
        assert ref() is not None
        cache.reserve_append_source()
        assert len(pool._writes) == 0
        assert ref() is None
        # Only now is the next projection source allocated.
        cache.append(torch.full((1, 7), 2.0, device="cuda", dtype=torch.bfloat16))
    cache.commit()
    torch.testing.assert_close(
        cache.host_records()[:, 0], torch.tensor([1, 2], dtype=torch.bfloat16)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_same_stream_then_cross_stream_consumer_ordering():
    pool = make_pool(slots=8, device="cuda")
    a, b = [pool.allocate_session(8).layer(0) for _ in range(2)]
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(first):
        values = torch.arange(56, device="cuda", dtype=torch.float32).reshape(8, 7)
        append(a, values)
        physical = a.ensure(torch.arange(8, device="cuda"))
        # This consumer is submitted after ensure's operation lease exits.
        torch.cuda._sleep(2_000_000)
        observed = a.records[physical.long()].clone()
        a.ensure(torch.tensor([0, 1], device="cuda"))
    with torch.cuda.stream(second):
        append(b, torch.full((8, 7), -10.0, device="cuda"))
    second.synchronize()
    torch.testing.assert_close(observed, values)
    assert not a.all_history_resident
    check_maps(pool)
