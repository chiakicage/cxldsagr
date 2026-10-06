"""CPU boundary tests for pool initialization and complete recall dispatch."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_cache import SparseTokenCache
from cache.sparse_token_pool import PRIORITY_LIMIT, SharedSparseTokenPool
from operators.common import kv_transfer


@pytest.mark.parametrize("mode", ["missing", "success", "error", "none", "noncallable"])
def test_cuda_metadata_initialization_precedes_pool_storage(monkeypatch, mode):
    pool = object.__new__(SharedSparseTokenPool)
    events = []
    initialize_error = KeyboardInterrupt("initializer failed")
    storage_boundary = RuntimeError("first allocation reached")

    def initialize(device):
        events.append(("initialize", device))
        assert not any(
            hasattr(pool, name) for name in ("_free_pages", "layers", "_sessions", "free_slots")
        )
        if mode == "error":
            raise initialize_error

    provider = SimpleNamespace()
    if mode != "missing":
        provider.initialize = (
            None if mode == "none" else 123 if mode == "noncallable" else initialize
        )

    def first_allocation(*args, **kwargs):
        events.append(("allocate",))
        raise storage_boundary

    monkeypatch.setattr(torch, "arange", first_allocation)
    expected = TypeError if mode in ("none", "noncallable") else BaseException
    with pytest.raises(expected) as caught:
        pool.__init__(128, 7, 1, 8, device="cuda:0", metadata_ops=provider)
    if mode == "error":
        assert caught.value is initialize_error
        assert events == [("initialize", torch.device("cuda:0"))]
    elif mode in ("none", "noncallable"):
        assert str(caught.value) == "an explicit metadata initializer must be callable"
        assert events == []
    else:
        assert caught.value is storage_boundary
        assert events == (
            [("allocate",)]
            if mode == "missing"
            else [("initialize", torch.device("cuda:0")), ("allocate",)]
        )
    assert not any(
        hasattr(pool, name) for name in ("_free_pages", "layers", "_sessions", "free_slots")
    )


def test_cpu_pool_does_not_initialize_cuda_provider():
    calls = []
    pool = SharedSparseTokenPool(
        128, 7, 1, 8, device="cpu", metadata_ops=SimpleNamespace(initialize=calls.append)
    )
    try:
        assert calls == []
        assert pool.native_metadata is None
    finally:
        pool.close()


def _fixture(monkeypatch, stage=None, *, sessions=1, free=True, complete=True):
    failure = KeyboardInterrupt(f"sentinel {stage}")
    events, paths = [], []
    layer = SimpleNamespace(
        clock=100,
        clock_tensor=torch.tensor([100]),
        append_owner=object(),
        free=torch.ones(5, dtype=torch.bool),
    )
    pool = SimpleNamespace(
        _sessions=list(range(sessions)),
        union_bitmap=torch.empty(1),
        union_count=torch.empty(1),
        counter=torch.zeros(1, dtype=torch.uint32),
        free_slots=torch.ones(4, dtype=torch.int32),
        candidate_slots=0,
        allocation_log=torch.empty(4, dtype=torch.int64),
        miss_scratch=torch.empty(4, dtype=torch.int64),
        _active=object(),
    )
    owner = pool._active

    def mark(name):
        events.append((name, layer.clock))
        if stage == name:
            raise failure

    pool._clock_event = lambda _: layer
    pool.wait_host = lambda *_: mark("wait")
    pool.invalidate_residency = lambda _: mark("invalidate")
    cache = SimpleNamespace(
        _pool=pool,
        layer_id=0,
        host_written_end=4,
        written=4,
        slots=4,
        device="cpu",
        page_table=torch.empty(1),
        host_to_device=torch.empty(4),
        device_to_host=torch.empty(5),
        age=torch.arange(5),
        _native_totals=torch.empty(4),
        _native_recalled=torch.empty(1),
        session=object(),
        host=torch.empty((4, 7)),
        records=torch.empty((5, 7)),
        ensure=lambda ids: mark("reference"),
    )

    def complete_call(*args, **kwargs):
        paths.append(("complete", kwargs["free_only"]))
        for name in ("allocate", "copy", "publish"):
            mark(name)

    def allocate_call(kind):
        def allocate(*args, **kwargs):
            paths.append((kind,))
            mark("allocate")

        return allocate

    native = SimpleNamespace(
        sparse_selection_classify=lambda *a, **kw: mark("classify"),
        sparse_selection_allocate=allocate_call("general"),
        sparse_selection_compact=allocate_call("compact"),
        sparse_selection_publish=lambda *a, **kw: mark("publish"),
        sparse_selection_map=lambda *a, **kw: mark("map"),
    )
    if free:
        native.sparse_selection_allocate_free = allocate_call("free")
    if complete:
        native.sparse_selection_complete = complete_call
    monkeypatch.setattr(kv_transfer, "gather_host_records", lambda *a, **kw: mark("copy"))
    return cache, native, layer, events, paths, failure, owner


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize(
    "stage,clock,names",
    [
        ("classify", 100, ["classify"]),
        ("wait", 101, ["classify", "wait"]),
        ("invalidate", 101, ["classify", "wait", "invalidate"]),
        ("allocate", 101, ["classify", "wait", "invalidate", "allocate"]),
        ("copy", 101, ["classify", "wait", "invalidate", "allocate", "copy"]),
        ("publish", 101, ["classify", "wait", "invalidate", "allocate", "copy", "publish"]),
        ("map", 102, ["classify", "wait", "invalidate", "allocate", "copy", "publish", "map"]),
    ],
)
def test_cpu_clock_and_error_boundaries(monkeypatch, complete, stage, clock, names):
    cache, native, layer, events, _, failure, owner = _fixture(
        monkeypatch, stage, complete=complete
    )
    append_owner = layer.append_owner
    with pytest.raises(KeyboardInterrupt) as caught:
        SparseTokenCache._ensure_sparse_from_topk(
            cache, torch.ones((1, 2), dtype=torch.int32), native
        )
    assert caught.value is failure
    assert layer.clock == clock
    assert [name for name, _ in events] == names
    assert cache._pool._active is owner
    assert layer.append_owner is (append_owner if stage == "classify" else None)


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize(
    "sessions,free,expected", [(1, True, True), (2, True, False), (1, False, False)]
)
def test_sole_session_capability_dispatch(monkeypatch, complete, sessions, free, expected):
    cache, native, layer, events, paths, _, owner = _fixture(
        monkeypatch, sessions=sessions, free=free, complete=complete
    )
    SparseTokenCache._ensure_sparse_from_topk(cache, torch.ones((1, 2), dtype=torch.int32), native)
    expected_paths = (
        [("complete", expected)] if complete else [("free" if expected else "general",)]
    )
    assert paths == expected_paths
    assert layer.clock == 102 and cache._pool._active is owner
    assert events == [("classify", 100)] + [
        (name, 101) for name in ("wait", "invalidate", "allocate", "copy", "publish")
    ] + [("map", 102)]


@pytest.mark.parametrize("value", [None, 123])
def test_noncallable_complete_keeps_segmented_dispatch(monkeypatch, value):
    cache, native, layer, events, paths, _, _ = _fixture(monkeypatch)
    native.sparse_selection_complete = value
    SparseTokenCache._ensure_sparse_from_topk(cache, torch.ones((1, 2), dtype=torch.int32), native)
    assert paths == [("free",)]
    assert [name for name, _ in events].count("classify") == 1
    assert layer.clock == 102


@pytest.mark.parametrize("sessions,name", [(1, "free"), (2, "general")])
def test_none_allocator_keeps_original_compact_dispatch(monkeypatch, sessions, name):
    cache, native, layer, events, paths, _, _ = _fixture(monkeypatch, sessions=sessions)
    attribute = "sparse_selection_allocate" + ("_free" if name == "free" else "")
    setattr(native, attribute, None)
    SparseTokenCache._ensure_sparse_from_topk(cache, torch.ones((1, 2), dtype=torch.int32), native)
    assert paths == [("compact",)]
    assert [name for name, _ in events].count("classify") == 1
    assert layer.clock == 102


def test_rollover_uses_existing_reference_boundary(monkeypatch):
    cache, native, layer, events, paths, _, owner = _fixture(monkeypatch)
    layer.clock = PRIORITY_LIMIT - 1
    SparseTokenCache._ensure_sparse_from_topk(cache, torch.ones((1, 2), dtype=torch.int32), native)
    assert events == [("reference", PRIORITY_LIMIT - 1)]
    assert paths == [] and cache._pool._active is owner


def test_bound_method_provider_keeps_free_allocation_proof(monkeypatch):
    cache, native, _, _, paths, _, _ = _fixture(monkeypatch)

    class Provider:
        def __getattr__(self, name):
            return getattr(native, name)

        def sparse_selection_allocate_free(self, *args, **kwargs):
            return native.sparse_selection_allocate_free(*args, **kwargs)

    provider = Provider()
    SparseTokenCache._ensure_sparse_from_topk(
        cache, torch.ones((1, 2), dtype=torch.int32), provider
    )
    assert paths == [("complete", True)]


def test_aliased_allocator_provider_requires_sole_session_proof(monkeypatch):
    cache, native, _, _, paths, _, _ = _fixture(monkeypatch, sessions=2)
    native.sparse_selection_allocate_free = native.sparse_selection_allocate
    SparseTokenCache._ensure_sparse_from_topk(cache, torch.ones((1, 2), dtype=torch.int32), native)
    assert paths == [("complete", False)]
