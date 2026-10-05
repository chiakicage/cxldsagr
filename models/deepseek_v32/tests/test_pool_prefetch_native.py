"""Native dense preparation must retain private ticket IDs across scratch reuse."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_pool import PRIORITY_LIMIT, SharedSparseTokenPool
from models.deepseek_v32.cache.prefetch import PoolHistoryPrefetch
from models.deepseek_v32.tests.test_pool_prefetch import make_pool, populate
from operators.common import kv_transfer
from operators.deepseek_v32.indexer import cache_ops


def test_metadata_reservation_failure_disables_helper_before_copy(monkeypatch):
    pool = make_pool()
    a, b = [pool.allocate_session(8).layer(0) for _ in range(2)]
    populate(a, 8)
    other = populate(b, 8, 1000)
    helper = PoolHistoryPrefetch("cpu")
    original = pool.stamp

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected post-reservation failure")

    with monkeypatch.context() as patch:
        patch.setattr(pool, "stamp", fail)
        with pytest.raises(RuntimeError, match="post-reservation"):
            helper.prefetch(a)
    assert helper.failed and not helper._tickets
    with pytest.raises(RuntimeError, match="failed"):
        helper.prefetch(a)
    helper.drain()
    a.session.release()
    output = b.ensure(torch.arange(8))
    torch.testing.assert_close(b.records[output.long()], other)
    helper.close()
    pool.close()


@pytest.mark.parametrize("case", ["shared_prefix", "shared_ids", "oversized_storage", "offset"])
def test_native_dense_ticket_binding_rejects_borrowed_storage(monkeypatch, case):
    monkeypatch.setattr(cache_ops, "_resident_maps", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(
        cache_ops, "_call", lambda *args: pytest.fail("invalid ticket reached CUDA")
    )
    prefix = torch.empty(8, dtype=torch.int64)
    victims = torch.empty(8, dtype=torch.int64)
    misses = torch.empty(3, dtype=torch.int64)
    chosen = torch.empty(3, dtype=torch.int64)
    if case == "shared_prefix":
        misses = prefix[:3]
    elif case == "shared_ids":
        chosen = misses
    elif case == "oversized_storage":
        chosen = torch.empty(8, dtype=torch.int64)[:3]
    else:
        chosen = torch.empty(4, dtype=torch.int64)[1:]
    with pytest.raises(ValueError, match="distinct storage|own exactly"):
        cache_ops.dense_history_reserve(
            prefix,
            victims,
            torch.tensor([0], dtype=torch.int32),
            torch.zeros(64, dtype=torch.int32),
            torch.zeros(9, dtype=torch.int64),
            torch.zeros(9, dtype=torch.int64),
            torch.zeros(9, dtype=torch.bool),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            misses,
            chosen,
            history=8,
            timestamp=2,
        )


def _pools(*, slots=193, width=576, dtype=torch.bfloat16):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    assert torch.cuda.get_device_capability() == (9, 0), "Hopper SM90 required"
    checked = SimpleNamespace(
        **{
            name: getattr(cache_ops, name)
            for name in (
                "protect",
                "mark_misses",
                "release_ids",
                "finalize_prefetch",
                "planned_append",
                "resident_selection",
                "protect_resident_history",
                "sparse_selection_classify",
                "sparse_selection_compact",
                "sparse_selection_publish",
                "sparse_selection_map",
            )
        }
    )
    return [
        SharedSparseTokenPool(
            1024, width, 2, slots, device="cuda", dtype=dtype, candidate_slots=35, metadata_ops=ops
        )
        for ops in (cache_ops, checked)
    ]


def _equal(pools, sessions):
    for layer_id in range(2):
        states = [pool.layers[layer_id] for pool in pools]
        for field in (
            "records",
            "host_to_device",
            "device_to_host",
            "priority",
            "free",
            "clock_tensor",
        ):
            torch.testing.assert_close(
                getattr(states[0], field), getattr(states[1], field), rtol=0, atol=0
            )
        for field in (
            "clock",
            "map_generation",
            "resident_owner",
            "resident_end",
            "append_owner",
            "append_cursor",
            "lease_owner",
        ):
            assert getattr(states[0], field) == getattr(states[1], field), field
        assert sessions[0].layer(layer_id).metrics() == sessions[1].layer(layer_id).metrics()


@pytest.mark.parametrize("history,other", [(137, 150), (193, 193)])
@pytest.mark.parametrize(
    "dtype,width", [(torch.uint8, 7), (torch.float32, 7), (torch.bfloat16, 576)]
)
def test_cuda_dense_private_tickets_survive_delayed_copy_and_shared_scratch_overwrite(
    history, other, dtype, width
):
    pools = _pools(width=width, dtype=dtype)
    helpers = [PoolHistoryPrefetch("cuda") for _ in pools]
    sessions = []
    try:
        # Warm compilation before installing the delay used to expose aliases.
        cache_ops._module()
        kv_transfer._module()
        for pool, helper in zip(pools, helpers, strict=True):
            small = [pool.allocate_session(64) for _ in range(4)]
            small[0].release()
            small[2].release()
            a, b = pool.allocate_session(history), pool.allocate_session(other)
            assert bool((a._pages[1:] < a._pages[:-1]).any())
            for layer in range(2):
                populate(a.layer(layer), history, layer * 100)
                populate(b.layer(layer), other, 1000 + layer * 100)
                if history < pool.slots:
                    b.layer(layer).truncate(80)
            snapshot = pool.snapshot()
            # Also warm CUDA's lazy modules for classify/scan/sort/transport;
            # a first-use module load can synchronize past an installed sleep.
            warm_tickets = [helper.prefetch(a.layer(layer)) for layer in range(2)]
            for ticket in warm_tickets:
                helper.wait(ticket)
            helper.drain()
            pool.restore(snapshot)
            for layer in range(2):
                a.layer(layer).reset_stats()
                a.layer(layer).begin_transient(35)
                a.layer(layer).append(torch.full((35, width), 7, device="cuda", dtype=dtype))
            sessions.append(a)
        for pool, session, helper in zip(pools, sessions, helpers, strict=True):
            delayed = torch.cuda.Event()
            with torch.cuda.stream(helper._copy_stream):
                torch.cuda._sleep(1_000_000_000)
                delayed.record()
            tickets = [helper.prefetch(session.layer(layer)) for layer in range(2)]
            for ticket in tickets:
                for tensor in (ticket._host_ids, ticket._slots):
                    assert tensor.storage_offset() == 0
                    assert tensor.untyped_storage().nbytes() == ticket.fetched_records * 8
                    assert tensor.untyped_storage().data_ptr() not in {
                        pool.allocation_log.untyped_storage().data_ptr(),
                        pool.miss_scratch.untyped_storage().data_ptr(),
                        pool.free_slots.untyped_storage().data_ptr(),
                    }
            for scratch in (
                pool.allocation_log,
                pool.miss_scratch,
                pool.free_slots,
                pool.union_bitmap,
                pool.counter,
            ):
                scratch.fill_(19)
            torch.cuda.current_stream().synchronize()
            assert not delayed.query(), "copy delay ended before the overwrite test"
            for ticket in tickets:
                helper.wait(ticket)
            torch.cuda.current_stream().synchronize()
            for cache in session._layers.values():
                assert cache.metrics()["selection_records"] == 0
                assert cache.metrics()["max_working_set"] == 0
            helper.drain()
        _equal(pools, sessions)
        for session in sessions:
            for cache in session._layers.values():
                cache.discard_transient()
        _equal(pools, sessions)
    finally:
        for helper in helpers:
            helper.close()
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("history", [0, 5, 193])
@pytest.mark.parametrize("clock", [0, PRIORITY_LIMIT - 2, PRIORITY_LIMIT - 1, PRIORITY_LIMIT])
def test_cuda_dense_uncertified_all_hit_and_empty_preserve_two_events(history, clock):
    pools = _pools(width=7, dtype=torch.float32)
    helpers = [PoolHistoryPrefetch("cuda") for _ in pools]
    sessions = []
    try:
        for pool, helper in zip(pools, helpers, strict=True):
            session = pool.allocate_session(193)
            cache = session.layer(0)
            if history:
                populate(cache, history)
            layer = pool.layers[0]
            layer.resident_owner = None
            layer.clock = max(layer.clock, clock)
            ticket = helper.prefetch(cache)
            assert ticket.fetched_records == 0 and ticket._ready is None
            helper.wait(ticket)
            helper.drain()
            sessions.append(session)
        _equal(pools, sessions)
    finally:
        for helper in helpers:
            helper.close()
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("stage", ["metadata", "copy"])
def test_cuda_dense_failed_submission_retains_copy_inputs_and_blocks_reuse(monkeypatch, stage):
    pools = _pools(slots=8, width=7, dtype=torch.float32)
    pool = pools[0]
    helper = PoolHistoryPrefetch("cuda")
    try:
        a, b = [pool.allocate_session(8).layer(0) for _ in range(2)]
        populate(a, 8)
        other = populate(b, 8, 1000)
        target, name = (
            (cache_ops, "dense_history_reserve")
            if stage == "metadata"
            else (kv_transfer, "gather_host_records")
        )
        original = getattr(target, name)

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected submission failure")

        with monkeypatch.context() as patch:
            patch.setattr(target, name, fail)
            with pytest.raises(RuntimeError, match="submission failure"):
                helper.prefetch(a)
        assert helper.failed
        if stage == "copy":
            assert helper._tickets and helper._tickets[0]._host_ids is not None
        with pytest.raises(RuntimeError, match="failed"):
            helper.prefetch(a)
        helper.drain()
        a.session.release()
        physical = b._ensure_from_topk(torch.arange(8, device="cuda", dtype=torch.int32)[None])
        torch.testing.assert_close(b.records[physical.long()], other[None])
    finally:
        helper.close()
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("boundary", ["metadata_ready", "copy_ready"])
@pytest.mark.parametrize("failure", ["constructor", "record"])
def test_cuda_dense_event_failure_retains_enqueued_copy_until_drain(monkeypatch, boundary, failure):
    import sys

    pools = _pools(slots=8, width=7, dtype=torch.float32)
    pool = pools[0]
    helper = PoolHistoryPrefetch("cuda")
    try:
        a, b = [pool.allocate_session(8).layer(0) for _ in range(2)]
        populate(a, 8)
        other = populate(b, 8, 1000)
        original_event = torch.cuda.Event
        submit_events = 0
        target_event = 1 if boundary == "metadata_ready" else 2

        def fail_record(*args, **kwargs):
            raise RuntimeError("injected event record failure")

        def event_factory(*args, **kwargs):
            nonlocal submit_events
            # Pool lease-completion events must remain real and successful.
            if sys._getframe(1).f_code.co_name == "_submit":
                submit_events += 1
                if submit_events == target_event:
                    if failure == "constructor":
                        raise RuntimeError("injected event constructor failure")
                    return SimpleNamespace(record=fail_record)
            return original_event(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "Event", event_factory)
            with pytest.raises(RuntimeError, match="injected event"):
                helper.prefetch(a)
        assert submit_events == target_event and helper.failed
        assert bool(helper._tickets) == (boundary == "copy_ready")
        if helper._tickets:
            ticket = helper._tickets[0]
            assert ticket._host_ids is not None and ticket._slots is not None
            assert ticket._host_ids.untyped_storage().nbytes() == ticket.fetched_records * 8
        with pytest.raises(RuntimeError, match="failed"):
            helper.prefetch(a)
        helper.drain()
        assert not helper._tickets
        a.session.release()
        physical = b._ensure_from_topk(torch.arange(8, device="cuda", dtype=torch.int32)[None])
        torch.testing.assert_close(b.records[physical.long()], other[None])
    finally:
        helper.close()
        for pool in pools:
            pool.close()
