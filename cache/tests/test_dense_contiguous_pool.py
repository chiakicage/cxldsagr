"""Dense DMA placement is explicit across allocation, append and snapshots."""

import pytest
import torch

from cache.sparse_token_pool import MISSING, SharedSparseTokenPool


def make_pool(*, pages=8, slots=128, layers=1, candidate_slots=3):
    return SharedSparseTokenPool(
        pages * 64,
        7,
        layers,
        slots,
        device="cpu",
        dtype=torch.float32,
        candidate_slots=candidate_slots,
        dense_contiguous=True,
    )


def append(cache, data, *, split=None):
    cache.begin_step(len(data))
    for piece in data.split(split or len(data)):
        cache.append(piece)
    cache.commit()


def assert_direct(cache, data):
    length = len(data)
    ids = cache.logical_to_global(torch.arange(length))
    expected = torch.arange(1, length + 1)
    torch.testing.assert_close(cache.host_to_device[ids].long(), expected)
    torch.testing.assert_close(cache.device_to_host[expected], ids)
    torch.testing.assert_close(cache.records[1 : length + 1], data)
    assert cache.all_history_resident
    assert cache._pool.dense_history_resident(cache)


def test_recycled_dense_host_pages_keep_one_increasing_run():
    pool = make_pool()
    a, b = pool.allocate_session(128), pool.allocate_session(128)
    assert a._pages.tolist() == [0, 1]
    a.release()
    c = pool.allocate_session(128)
    assert c._pages.tolist() == [0, 1]
    assert c._host_runs == ((0, 128, 0),)
    b.release()
    c.release()
    assert pool._dense_free_runs == ((0, 8),)
    pool.close()


def test_fragmented_dense_admission_fails_without_mutating_ownership():
    pool = make_pool(pages=4)
    sessions = [pool.allocate_session(64) for _ in range(4)]
    sessions[0].release()
    sessions[2].release()
    before = (
        pool.free_host_pages,
        pool._dense_free_runs,
        tuple(pool._sessions),
        pool._next_owner,
        pool._topology,
    )
    with pytest.raises(ValueError, match="contiguous host pages"):
        pool.allocate_session(128)
    assert before == (
        pool.free_host_pages,
        pool._dense_free_runs,
        tuple(pool._sessions),
        pool._next_owner,
        pool._topology,
    )
    sessions[1].release()
    contiguous = pool.allocate_session(128)
    assert contiguous._pages.tolist() == [0, 1]
    pool.close()


def test_dense_admission_metadata_failure_leaves_free_runs_unchanged(monkeypatch):
    import cache.sparse_token_pool as module

    pool = make_pool()
    before = (pool._dense_free_runs, pool.free_host_pages, pool._topology)
    failure = RuntimeError("metadata construction failed")

    def fail(*args):
        raise failure

    monkeypatch.setattr(module, "SparseTokenSession", fail)
    with pytest.raises(RuntimeError) as caught:
        pool.allocate_session(128)
    assert caught.value is failure
    assert (pool._dense_free_runs, pool.free_host_pages, pool._topology) == before
    assert not pool._sessions
    pool.close()


def test_dense_append_uses_logical_slots_despite_old_fifo_priorities():
    pool = make_pool(slots=16)
    a, b = (pool.allocate_session(16).layer(0) for _ in range(2))
    data = torch.arange(56, dtype=torch.float32).reshape(8, 7)
    tail = pool.layers[0].records[17:]
    tail.fill_(901)
    append(a, data, split=3)
    assert_direct(a, data)
    a.ensure(torch.tensor([0, 2, 0, 6]))
    pool.layers[0].priority[1:].copy_(torch.arange(16, 0, -1))
    append(b, data + 100, split=3)
    assert_direct(b, data + 100)
    assert not a.all_history_resident
    assert not pool.dense_history_resident(a)
    assert tail.eq(901).all()
    assert pool.layers[0].records[0].eq(0).all()
    assert pool.layers[0].device_to_host[0] == MISSING
    assert not pool.layers[0].free[0]
    pool.close()


def test_generic_residency_fields_cannot_certify_dense_layout():
    pool = make_pool(slots=16)
    cache = pool.allocate_session(16).layer(0)
    data = torch.ones(8, 7)
    append(cache, data)
    pool.invalidate_residency(0)
    layer = pool.layers[0]
    layer.resident_owner, layer.resident_end = cache.session.owner, 8
    assert not cache.all_history_resident
    before = cache.records.clone()
    cache.begin_step(1)
    with pytest.raises(RuntimeError, match="resident contiguous prefix"):
        cache.append(torch.ones(1, 7))
    torch.testing.assert_close(cache.records, before)
    cache.rollback()
    pool.close()


def test_warm_snapshot_restores_direct_layout_proof_and_append():
    pool = make_pool(slots=16, layers=2)
    a, b = pool.allocate_session(16), pool.allocate_session(16)
    data = torch.arange(56, dtype=torch.float32).reshape(8, 7)
    for index in range(2):
        append(a.layer(index), data + index, split=3)
    snapshot = pool.snapshot()
    for index in range(2):
        append(b.layer(index), data + 200 + index, split=3)
        assert not a.layer(index).all_history_resident
    pool.restore(snapshot)
    for index in range(2):
        cache = a.layer(index)
        with cache.operation():
            assert_direct(cache, data + index)
        suffix = torch.full((2, 7), 99.0)
        append(cache, suffix)
        assert_direct(cache, torch.cat((data + index, suffix)))
    pool.close()


def test_dense_capacity_and_tensor_accounting_match_existing_reservation():
    pool = make_pool(slots=128, layers=2)
    with pytest.raises(ValueError, match="must not exceed P"):
        pool.allocate_session(129)
    assert pool.shared_bytes() == pool.estimate_shared_bytes(
        512, 7, 2, 128, dtype=torch.float32, device="cpu", candidate_slots=3
    )
    session = pool.allocate_session(65)
    assert session._host_runs == ((0, 128, 0),)
    assert session.session_bytes() == pool.estimate_session_bytes(65, layers=2, device="cpu")
    pool.close()
    assert pool.shared_bytes() == {"hbm": 0, "dram": 0}


@pytest.mark.parametrize("invalid", [1, None, "true"])
def test_dense_mode_requires_explicit_boolean(invalid):
    with pytest.raises(ValueError, match="boolean"):
        SharedSparseTokenPool(64, 7, 1, 16, device="cpu", dense_contiguous=invalid)
