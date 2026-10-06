"""Sole-session allocation eligibility follows the live pool lifecycle."""

import pytest
import torch

from cache.sparse_token_pool import SharedSparseTokenPool
from cache.tests.test_sparse_recall_metadata import (
    _append,
    _capture,
    _LegacyMetadataProvider,
    _validate_recall,
)
from operators.deepseek_v32.indexer import cache_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def pool():
    pool = SharedSparseTokenPool(
        1024,
        7,
        1,
        193,
        device="cuda",
        dtype=torch.float32,
        candidate_slots=35,
        metadata_ops=cache_ops,
    )
    try:
        yield pool
    finally:
        pool.close()


def _release_history(cache):
    with cache.operation():
        cache._pool.release_ids(cache.layer_id, cache.session.global_ids())


def _recall(cache, calls, expected, *, stream=None):
    cache.reset_stats()
    ids = torch.arange(cache.written, device=cache.device, dtype=torch.int32)[None]
    before = _capture(cache)
    count = len(calls)
    with torch.cuda.stream(stream or torch.cuda.default_stream(cache.device)):
        output = cache._ensure_from_topk(ids)
    assert calls[count:] == [expected]
    _validate_recall(cache, ids, output, before)


@pytest.mark.parametrize("complete_dispatch", [False, True])
def test_cuda_free_allocation_tracks_session_release_snapshot_and_transient(
    pool, monkeypatch, complete_dispatch
):
    calls = []
    if complete_dispatch:
        original = cache_ops.sparse_selection_complete

        def complete(*args, **kwargs):
            calls.append(
                "sparse_selection_allocate_free"
                if kwargs["free_only"]
                else "sparse_selection_allocate"
            )
            return original(*args, **kwargs)

        monkeypatch.setattr(cache_ops, "sparse_selection_complete", complete)
        names = ()
    else:
        pool.metadata_ops = _LegacyMetadataProvider()
        names = ("sparse_selection_allocate", "sparse_selection_allocate_free")
    for name in names:
        original = getattr(cache_ops, name)

        def tracked(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cache_ops, name, tracked)

    fragments = [pool.allocate_session(64) for _ in range(4)]
    fragments[0].release()
    fragments[2].release()
    cache = pool.allocate_session(137).layer(0)
    assert bool((cache.session._pages[1:] < cache.session._pages[:-1]).any())
    fragments[1].release()
    fragments[3].release()
    _append(cache, torch.arange(137 * 7, device="cuda", dtype=torch.float32).reshape(137, 7))
    old_snapshot = pool.snapshot()
    _release_history(cache)
    _recall(cache, calls, "sparse_selection_allocate_free")

    other = pool.allocate_session(137).layer(0)
    _append(other, torch.full((137, 7), 2.0, device="cuda"))
    _recall(cache, calls, "sparse_selection_allocate")
    other.session.release()
    assert len(pool._sessions) == 1
    with pytest.raises(ValueError, match="session"):
        pool.restore(old_snapshot)
    _release_history(cache)
    _recall(cache, calls, "sparse_selection_allocate_free")

    pool.restore(pool.snapshot())
    assert not cache.all_history_resident
    _recall(cache, calls, "sparse_selection_allocate_free")
    assert cache.metrics()["recalled_records"] == 0

    _release_history(cache)
    cache.begin_transient(35)
    cache.append(torch.full((35, 7), 3.0, device="cuda"))
    _recall(cache, calls, "sparse_selection_allocate_free", stream=torch.cuda.Stream())
    cache.discard_transient()


def test_cuda_bounded_sole_session_without_free_provider_retains_fifo(pool, monkeypatch):
    provider = _LegacyMetadataProvider("sparse_selection_allocate_free")
    pool.metadata_ops = provider
    calls = []
    original = provider.sparse_selection_allocate

    def allocate(*args, **kwargs):
        calls.append("sparse_selection_allocate")
        return original(*args, **kwargs)

    monkeypatch.setattr(provider, "sparse_selection_allocate", allocate)
    cache = pool.allocate_session(137).layer(0)
    _append(cache, torch.ones(137, 7, device="cuda"))
    _release_history(cache)
    _recall(cache, calls, "sparse_selection_allocate")
