"""Keep execution and cleanup failures together without discarding owned state."""

from types import SimpleNamespace

import pytest
import torch

from models.nosa.execution import compute_graphs
from models.nosa.tests.test_model import initialized_model
from models.nosa.tests.test_serving_resources import make_backend


def raise_error(error):
    raise error


def graph_fixture(monkeypatch, record_error):
    graph = object.__new__(compute_graphs.NosaComputeGraphs)
    storage = object()
    graph.model = SimpleNamespace(
        model=SimpleNamespace(layers=(), rotary=SimpleNamespace(cos_sin_cache=storage)),
        config={},
        named_parameters=lambda: (),
    )
    graph.layers, graph.weight_identity, graph.layer_policy = (), (), ()
    graph.config, graph.cos_sin = {}, storage
    graph.precision = compute_graphs._precision()
    graph.device = torch.device("cuda:0")
    graph._active = graph.failed = graph.closed = False
    graph.allocated = True
    graph._stream = None
    graph._last_stream = object()
    graph.pairs = {(0, 1): storage}
    calls = []
    stream = SimpleNamespace(wait_event=lambda event: calls.append(("wait", event)))

    def record(current):
        calls.append(("record", current))
        if record_error is not None:
            raise record_error

    graph._last_event = SimpleNamespace(record=record)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return graph, stream, calls


@pytest.mark.parametrize("body_type", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("record_fails", [False, True])
def test_graph_execution_keeps_both_errors_and_failed_storage(monkeypatch, body_type, record_fails):
    body = body_type("body failure")
    record = RuntimeError("record failure") if record_fails else None
    graph, stream, calls = graph_fixture(monkeypatch, record)
    storage, previous = graph.pairs, graph._last_stream
    with pytest.raises(BaseException) as caught, graph.execution():
        assert graph._active and graph._stream is stream
        raise body
    if record_fails:
        assert caught.value.exceptions == (body, record)
        assert graph.failed and graph._last_stream is previous
        with pytest.raises(RuntimeError, match="not available"), graph.execution():
            pytest.fail("poisoned graph was entered")
    else:
        assert caught.value is body
        assert not graph.failed and graph._last_stream is stream
    assert graph.pairs is storage and storage
    assert not graph._active and graph._stream is None
    assert [entry[0] for entry in calls] == ["wait", "record"]


def test_graph_record_only_failure_is_not_wrapped(monkeypatch):
    error = RuntimeError("record failure")
    graph, _, _ = graph_fixture(monkeypatch, error)
    with pytest.raises(RuntimeError) as caught, graph.execution():
        pass
    assert caught.value is error and graph.failed


@pytest.mark.parametrize("close_fails", [False, True])
def test_graph_setup_preserves_allocation_error_and_poisoned_graph(monkeypatch, close_fails):
    backend = make_backend("hbm")
    allocation, cleanup = ValueError("allocation failure"), RuntimeError("close failure")
    retained = object()
    calls = []

    def close():
        calls.append("close")
        if close_fails:
            raise cleanup

    graph = SimpleNamespace(
        allocate=lambda: raise_error(allocation), close=close, retained=retained
    )
    monkeypatch.setattr(compute_graphs, "NosaComputeGraphs", lambda *_a, **_k: graph)
    with pytest.raises(BaseException) as caught:
        backend.enable_compute_graphs((1,))
    assert calls == ["close"]
    if close_fails:
        assert caught.value.exceptions == (allocation, cleanup)
        assert backend.compute_graphs is graph and graph.retained is retained
        assert backend.resources.poisoned
        with pytest.raises(RuntimeError, match="poisoned"):
            backend.bind_owner(object())
    else:
        assert caught.value is allocation
        assert backend.compute_graphs is None and not backend.resources.poisoned


def test_model_preserves_compute_and_abort_failure(monkeypatch):
    model = initialized_model()
    cache = model.new_cache(16)
    body, abort = ValueError("layer failure"), RuntimeError("abort drain failure")
    calls = []
    monkeypatch.setattr(model.model.layers[0], "forward", lambda *_a, **_k: raise_error(body))

    def abort_step():
        calls.append("abort")
        raise abort

    monkeypatch.setattr(cache, "abort_step", abort_step)
    with pytest.raises(BaseExceptionGroup) as caught:
        model(torch.tensor([1, 2]), cache, return_hidden=True)
    assert caught.value.exceptions == (body, abort)
    assert calls == ["abort"] and cache.length == 0 and cache._pending_end == 2


def test_offload_begin_preserves_indexer_and_host_rollback_failure(monkeypatch):
    from cache.host_backing import HostBackingCache
    from models.nosa.tests.test_offload_cache import make_cache

    cache = make_cache(128)
    begin, abort = ValueError("indexer begin"), RuntimeError("host abort drain")
    monkeypatch.setattr(cache.indexer_cache, "begin_step", lambda *_: raise_error(begin))
    monkeypatch.setattr(HostBackingCache, "abort_step", lambda *_: raise_error(abort))
    with pytest.raises(BaseExceptionGroup) as caught:
        cache.begin_step(2)
    assert caught.value.exceptions == (begin, abort)
    assert cache.length == 0 and cache._pending_end == 2


def test_dense_begin_preserves_prefetch_and_abort_failure(monkeypatch):
    from models.nosa.cache.dense_prefetch import NosaDensePrefetchCache
    from models.nosa.tests.test_model import tiny_config

    cache = NosaDensePrefetchCache(tiny_config(), 32, device="cpu", dtype=torch.float32)
    prefetch, abort = ValueError("prefetch submission"), RuntimeError("abort drain")
    monkeypatch.setattr(cache, "_prefetch", lambda *_: raise_error(prefetch))
    monkeypatch.setattr(cache, "abort_step", lambda: raise_error(abort))
    with pytest.raises(BaseExceptionGroup) as caught:
        cache.begin_step(2)
    assert caught.value.exceptions == (prefetch, abort)
    assert cache.length == 0 and cache._pending_end == 2


def test_fixed_begin_preserves_prefetch_and_abort_failure(monkeypatch):
    from models.nosa.tests.test_fixed_serving import backend_for
    from models.nosa.tests.test_model import tiny_config
    from models.nosa.tests.test_sparse_model import initialized_sparse_model

    backend, limits = backend_for(
        initialized_sparse_model(tiny_config(max_position_embeddings=160)), "dense_prefetch"
    )
    backend.allocate_shared(backend.plan_resources(None, limits))
    cache = backend.create_session(128)
    prefetch, abort = ValueError("fixed prefetch"), RuntimeError("fixed abort drain")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(backend.resources, "prefetch", lambda *_: raise_error(prefetch))
            patch.setattr(cache, "abort_step", lambda: raise_error(abort))
            with backend.execution_lease(cache):
                with pytest.raises(BaseExceptionGroup) as caught:
                    cache.begin_step(64)
                assert caught.value.exceptions == (prefetch, abort)
                assert cache.length == 0 and cache._pending_end == 64
    finally:
        backend.release_session(cache)
        backend.close()


def test_generation_preserves_execution_and_release_errors(monkeypatch):
    from executor.model_executor import ModelExecutor
    from models.nosa.infer import generate
    from models.nosa.tests.test_infer import ScriptedModel

    model = ScriptedModel(2, [13])
    body, release = ValueError("prefill failure"), RuntimeError("release drain")
    calls = []
    monkeypatch.setattr(ModelExecutor, "prefill", lambda *_a, **_k: raise_error(body))

    def fail_release(executor, cache):
        calls.append(cache)
        raise release

    monkeypatch.setattr(ModelExecutor, "release", fail_release)
    with pytest.raises(BaseExceptionGroup) as caught:
        generate(model, torch.tensor([1, 2]), max_new_tokens=1)
    assert caught.value.exceptions == (body, release)
    assert calls == [model.cache]
