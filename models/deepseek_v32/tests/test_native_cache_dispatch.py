"""Serving ownership guards preserve the explicitly opted-in native metadata path."""

from types import SimpleNamespace

import pytest
import torch

from cache.lifecycle import ResourceLifecycle
from cache.sparse_token_pool import SharedSparseTokenPool
from models.deepseek_v32.cache.prefetch import PoolHistoryPrefetch
from models.deepseek_v32.cache.session import DeepSeekServingSession, ServingSparseTokenCache


def test_new_subclass_does_not_inherit_native_metadata_authorization():
    class DifferentLayout(ServingSparseTokenCache):
        pass

    cache = object.__new__(DifferentLayout)
    cache._pool = SimpleNamespace(
        native_metadata=SimpleNamespace(resident_selection=True, dense_history_classify=True)
    )
    ids = torch.tensor([[0]], dtype=torch.int32)
    calls = []
    cache.ensure = lambda selected: calls.append(selected) or selected
    assert cache._ensure_from_topk(ids) is ids
    assert calls == [ids]
    helper = PoolHistoryPrefetch("cpu")
    assert helper._reserve_native(cache, None, 1) is None
    helper.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_guarded_serving_cache_runs_native_resident_sparse_and_dense_paths(monkeypatch):
    from operators.deepseek_v32.indexer import cache_ops

    assert torch.cuda.get_device_capability() == (9, 0)
    calls = []
    for name in ("resident_selection", "sparse_selection_classify", "dense_history_classify"):
        original = getattr(cache_ops, name)

        def observe(*args, name=name, original=original, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(cache_ops, name, observe)
    lifecycle = ResourceLifecycle("native serving cache")
    owner = object()
    lifecycle.bind_owner(owner)
    lifecycle.allocated(object(), owner=owner)
    pool = SharedSparseTokenPool(192, 576, 1, 64, device="cuda", metadata_ops=cache_ops)
    backing = pool.allocate_session(64)
    cache = ServingSparseTokenCache.for_layer(backing, 0)
    session = DeepSeekServingSession(64, "echo", [], SimpleNamespace(lifecycle=lifecycle))
    session.registration = lifecycle.register(session, owner=owner)
    cache.bind_serving(session, lifecycle)
    backing.bind_release_guard(session.check_backing_release)
    helper = PoolHistoryPrefetch("cuda")
    others = []
    records = torch.arange(8 * 576, device="cuda").reshape(8, 576).bfloat16()
    ids = torch.tensor([[0, 7, -1]], device="cuda", dtype=torch.int32)
    try:
        with lifecycle.execution(session, session.registration, owner=owner):
            cache.begin_step(8)
            cache.append(records)
            cache.commit()
            assert cache.all_history_resident
            physical = cache._ensure_from_topk(ids)
            torch.testing.assert_close(
                cache.records[physical[0, :2].long()], records[[0, 7]], rtol=0, atol=0
            )
        assert calls == ["resident_selection"]
        for path in ("sparse", "dense"):
            other = pool.allocate_session(64)
            others.append(other)
            layer = other.layer(0)
            layer.begin_step(64)
            layer.append(torch.full((64, 576), 123, device="cuda", dtype=torch.bfloat16))
            layer.commit()
            assert not cache.all_history_resident
            with lifecycle.execution(session, session.registration, owner=owner):
                if path == "sparse":
                    physical = cache._ensure_from_topk(ids)
                    torch.testing.assert_close(
                        cache.records[physical[0, :2].long()], records[[0, 7]], rtol=0, atol=0
                    )
                else:
                    ticket = helper.prefetch(cache)
                    helper.wait(ticket)
                    assert ticket.fetched_records == 8
                    physical = cache.host_to_device[
                        cache.logical_to_global(torch.arange(8, device="cuda"))
                    ]
                    torch.testing.assert_close(
                        cache.records[physical.long()], records, rtol=0, atol=0
                    )
                    helper.drain()
        assert calls == [
            "resident_selection",
            "sparse_selection_classify",
            "dense_history_classify",
        ]
    finally:
        helper.close()
        with lifecycle.mutation(session, session.registration, owner=owner, releasing=True):
            backing.release()
            lifecycle.detach(session)
        for other in others:
            other.release()
        lifecycle.unbind_owner(owner)
        lifecycle.close(pool.close)
