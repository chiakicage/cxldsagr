"""DMA ownership, direct-map publication, and asynchronous failure boundaries."""

import pytest
import torch

from cache.sparse_token_pool import MISSING
from models.deepseek_v32.cache.prefetch import PoolHistoryPrefetch
from models.deepseek_v32.tests.test_pool_prefetch import make_pool, populate
from operators.common import kv_transfer
from operators.deepseek_v32.indexer import cache_ops


def test_metadata_clear_failure_retains_owners_and_disables_helper(monkeypatch):
    pool = make_pool()
    a, b = [pool.allocate_session(8).layer(0) for _ in range(2)]
    populate(a, 8)
    populate(b, 8, 1000)
    helper = PoolHistoryPrefetch("cpu")
    original = helper._clear_reference
    error = RuntimeError("injected post-clear failure")

    def fail(*args):
        original(*args)
        raise error

    monkeypatch.setattr(helper, "_clear_reference", fail)
    with pytest.raises(RuntimeError) as caught:
        helper.prefetch(a)
    assert caught.value is error
    assert helper.failed and len(helper._tickets) == 1
    ticket = helper._tickets[0]
    assert ticket._host is a.host and ticket._records is a.records
    assert not pool.dense_history_resident(a)
    with pytest.raises(RuntimeError, match="failed"):
        helper.prefetch(a)
    helper.drain()
    assert ticket._host is ticket._records is None
    helper.close()
    pool.close()


def test_failed_drain_preserves_both_errors_and_borrowed_owners():
    pool = make_pool()
    cache = pool.allocate_session(8).layer(0)
    populate(cache, 8)
    helper = PoolHistoryPrefetch("cpu")
    ticket = helper.prefetch(cache)
    copy_error = RuntimeError("copy completion failed")
    caller_error = RuntimeError("caller completion failed")

    class FailingStream:
        def __init__(self, error):
            self.error = error

        def synchronize(self):
            raise self.error

    helper.failed = True
    helper._copy_stream = FailingStream(copy_error)
    helper._caller_stream = FailingStream(caller_error)
    with pytest.raises(BaseExceptionGroup) as caught:
        helper.drain()
    assert caught.value.exceptions == (copy_error, caller_error)
    assert helper._tickets == [ticket] and ticket._active
    assert ticket._host is cache.host and ticket._records is cache.records
    # The fake streams performed no asynchronous work. Release test-owned data.
    helper._copy_stream = helper._caller_stream = None
    helper.close()
    pool.close()


def test_wait_rejects_mapping_generation_change_without_certifying_residency():
    pool = make_pool()
    cache = pool.allocate_session(8).layer(0)
    populate(cache, 8)
    helper = PoolHistoryPrefetch("cpu")
    ticket = helper.prefetch(cache)
    pool.invalidate_residency(cache.layer_id)
    with pytest.raises(RuntimeError, match="maps changed"):
        helper.wait(ticket)
    assert helper.failed and not pool.dense_history_resident(cache)
    helper.close()
    pool.close()


@pytest.mark.parametrize("host_start,history", [(-1, 3), (True, 3), (128, 1), (126, 3)])
def test_native_binding_rejects_invalid_contiguous_host_span(monkeypatch, host_start, history):
    monkeypatch.setattr(cache_ops, "_resident_maps", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(cache_ops, "_call", lambda *args: pytest.fail("invalid span reached CUDA"))
    with pytest.raises(ValueError, match="host span"):
        cache_ops.dense_history_clear(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.empty(128, dtype=torch.int32),
            torch.empty(9, dtype=torch.int64),
            torch.empty(9, dtype=torch.int64),
            torch.empty(9, dtype=torch.bool),
            torch.empty(1, dtype=torch.int64),
            torch.empty(1, dtype=torch.int64),
            history=history,
            host_start=host_start,
            timestamp=0,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_permuted_existing_maps_publish_directly_without_losing_unrelated_suffix(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pool = make_pool(slots=8, device=device)
    cache, other = [pool.allocate_session(3).layer(0) for _ in range(2)]
    history = populate(cache, 3)
    other_history = populate(other, 3, 1000)
    pool.drain()
    incoming = cache.session._host_runs[0][2]
    unrelated = other.session._host_runs[0][2]
    layer = pool.layers[0]
    layer.host_to_device.fill_(MISSING)
    layer.device_to_host.fill_(MISSING)
    layer.priority.fill_(-1)
    layer.priority[0] = MISSING
    layer.free.fill_(True)
    layer.free[0] = False
    # Incoming IDs are permuted, one is outside the target. Another user's
    # record in target slot3 must be evicted; its slot8 record must survive.
    for slot, global_id in (
        (1, incoming + 2),
        (2, incoming),
        (5, incoming + 1),
        (3, unrelated + 1),
        (8, unrelated),
    ):
        layer.host_to_device[global_id] = slot
        layer.device_to_host[slot] = global_id
        layer.priority[slot] = 0
        layer.free[slot] = False
        layer.records[slot].copy_(layer.host[global_id])
    pool.invalidate_residency(0)
    cache.reset_stats()
    helper = PoolHistoryPrefetch(device)
    ticket = helper.prefetch(cache)
    assert (ticket.resident_records, ticket.fetched_records) == (0, 3)
    assert helper.pending_bytes == 0
    assert not pool.dense_history_resident(cache)
    helper.wait(ticket)
    assert pool.dense_history_resident(cache)
    helper.drain()
    assert layer.host_to_device[incoming : incoming + 3].tolist() == [1, 2, 3]
    assert layer.device_to_host[1:4].tolist() == [incoming, incoming + 1, incoming + 2]
    assert int(layer.device_to_host[5]) == MISSING and bool(layer.free[5])
    assert int(layer.host_to_device[unrelated + 1]) == MISSING
    assert int(layer.host_to_device[unrelated]) == 8
    torch.testing.assert_close(layer.records[1:4], history, rtol=0, atol=0)
    torch.testing.assert_close(layer.records[8], other_history[0], rtol=0, atol=0)
    assert cache.metrics()["evicted_records"] == 1
    assert cache.metrics()["host_to_device_bytes"] == 3 * cache.record_bytes
    helper.close()
    pool.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_dma_no_scalar_reads_and_shared_scratch_reuse(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("dense DMA synchronized a GPU scalar")
            return func(*args, **(kwargs or {}))

    pool = make_pool(slots=193, device="cuda", layers=2)
    session, other = [pool.allocate_session(193) for _ in range(2)]
    histories = [populate(session.layer(i), 193, i * 2000) for i in range(2)]
    for i in range(2):
        populate(other.layer(i), 193, 10000 + i * 2000)
    helper = PoolHistoryPrefetch("cuda")
    monkeypatch.setattr(
        kv_transfer, "gather_host_records", lambda *args, **kw: pytest.fail("DMA used SM gather")
    )
    with NoScalarRead():
        tickets = [helper.prefetch(session.layer(i)) for i in range(2)]
        for scratch in (pool.free_slots, pool.miss_scratch, pool.counter, pool.allocation_log):
            scratch.zero_()
        for ticket in tickets:
            helper.wait(ticket)
    helper.drain()
    for i, ticket in enumerate(tickets):
        assert ticket.fetched_records == 193
        torch.testing.assert_close(session.layer(i).records[1:194], histories[i], rtol=0, atol=0)
    assert helper.pending_bytes == 0
    helper.close()
    pool.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("boundary", ["copy", "publication"])
def test_cuda_dma_submission_failure_retains_storage_until_drain(monkeypatch, boundary):
    pool = make_pool(device="cuda")
    cache, other = [pool.allocate_session(8).layer(0) for _ in range(2)]
    populate(cache, 8)
    populate(other, 8, 1000)
    helper = PoolHistoryPrefetch("cuda")
    module, name = (
        (kv_transfer, "copy_host_records_async")
        if boundary == "copy"
        else (cache_ops, "dense_history_publish")
    )
    original = getattr(module, name)
    error = RuntimeError("injected after " + boundary)

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(module, name, fail)
        with pytest.raises(RuntimeError) as caught:
            helper.prefetch(cache)
    assert caught.value is error and helper.failed
    ticket = helper._tickets[0]
    assert ticket._host is cache.host and ticket._records is cache.records
    assert not pool.dense_history_resident(cache)
    helper.drain()
    assert ticket._host is ticket._records is None
    helper.close()
    pool.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_wait_requires_original_caller_stream():
    pool = make_pool(device="cuda")
    cache = pool.allocate_session(8).layer(0)
    populate(cache, 8)
    helper = PoolHistoryPrefetch("cuda")
    ticket = helper.prefetch(cache)
    with torch.cuda.stream(torch.cuda.Stream()), pytest.raises(RuntimeError, match="caller stream"):
        helper.wait(ticket)
    helper.wait(ticket)
    helper.close()
    pool.close()
