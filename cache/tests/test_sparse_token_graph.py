"""CPU tests for CUDA capture authorization and replay bookkeeping boundaries."""

from types import SimpleNamespace

import pytest
import torch

from cache import sparse_token_graph
from cache.sparse_token_pool import PRIORITY_LIMIT, SharedSparseTokenPool


@pytest.fixture
def graph_pool(monkeypatch):
    pool = SharedSparseTokenPool(64, 7, 2, 8, device="cpu", dtype=torch.float32)
    session = pool.allocate_session(8)
    for index in range(2):
        cache = session.layer(index)
        cache.length = cache.written = cache.indexer_visible_end = 4
        cache.begin_step(2)
        pool.layers[index].clock = 7
    calls = []

    class Stream:
        def synchronize(self):
            calls.append(("synchronize", self))

        def wait_stream(self, stream):
            calls.append(("wait_stream", self, stream))

    origin, previous, write = Stream(), Stream(), Stream()
    state = SimpleNamespace(capturing=False, stream=origin)
    pool.device = torch.device("cuda:0")
    pool._last_stream, pool._copy_stream = previous, write
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: state.stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.capturing)
    return pool, session, state, calls, origin, previous, write


def finish_capture(pool, session, state, *, source=None):
    with pool.graph_capture(session) as capture:
        state.capturing = True
        for index, cache in session._layers.items():
            with cache.operation():
                if source is not None:
                    capture.retain_source(session, source)
                cache.written = cache.indexer_visible_end = cache._step_end
                cache.stats.written_records += 2
                pool.layers[index].clock += 3
                pool.invalidate_residency(index)
                pool.layers[index].resident_owner = session.owner
                pool.layers[index].resident_end = 6
                if pool.dense_contiguous:
                    pool.certify_dense_history(cache, 6)
        capture.join()
        state.capturing = False
        recipe = capture.finish()
    return recipe


def test_capture_guard_requires_explicit_origin_stream_and_completed_recipe(graph_pool):
    pool, session, state, _, origin, _, _ = graph_pool
    state.capturing = True
    with pytest.raises(RuntimeError, match="explicit graph capture lease"):
        session._check()
    state.capturing = False
    with pool.graph_capture(session) as capture:
        state.capturing = True
        session._check()
        state.stream = object()
        with pytest.raises(RuntimeError, match="explicit graph capture lease"):
            session._check()
        state.stream = origin
        for cache in session._layers.values():
            cache.written = cache.indexer_visible_end = cache._step_end
        capture.join()
        with pytest.raises(RuntimeError, match="explicit graph capture lease"):
            session._check()
        state.capturing = False
        capture.finish()
    assert not pool.poisoned


def test_capture_retains_sources_joins_copy_and_restores_cpu_state(graph_pool):
    pool, session, state, calls, origin, previous, write = graph_pool
    source = torch.ones(2, 7)
    recipe = finish_capture(pool, session, state, source=source)
    assert calls == [
        ("synchronize", previous),
        ("synchronize", write),
        ("wait_stream", origin, write),
    ]
    assert recipe.sources == (source, source)
    assert recipe.source_bytes == {"hbm": 0, "dram": source.untyped_storage().nbytes()}
    assert pool._last_stream is previous and pool._graph_capture is None
    assert all(layer.clock == 7 for layer in pool.layers)
    assert all(cache.written == 4 for cache in session._layers.values())
    assert all(cache.stats.written_records == 0 for cache in session._layers.values())
    recipe.validate()
    recipe.apply()
    assert pool._last_stream is origin
    assert all(layer.clock == 10 for layer in pool.layers)
    assert all(cache.written == 6 and cache.length == 4 for cache in session._layers.values())
    assert all(cache.stats.written_records == 2 for cache in session._layers.values())
    with pytest.raises(RuntimeError, match="recipe does not match"):
        recipe.apply()
    recipe.close()
    with pytest.raises(RuntimeError, match="closed"):
        recipe.validate()


def test_replay_generation_is_relative_and_stats_are_deltas(graph_pool):
    pool, session, state, _, _, _, _ = graph_pool
    pool.dense_contiguous = True
    for layer in pool.layers:
        layer.map_generation = 3
        layer.dense_generation = 3
        layer.dense_owner = session.owner
        layer.dense_end = 4
    recipe = finish_capture(pool, session, state)
    # A legitimate restore may advance the map epoch while recertifying the
    # same contiguous prefix. Its literal generation must not be rewound.
    for layer in pool.layers:
        layer.map_generation = 20
        layer.dense_generation = 20
    for cache in session._layers.values():
        cache.stats.written_records = 11
    recipe.validate()
    recipe.apply()
    # First lease ownership and the explicit map update each advance the epoch.
    assert all(layer.map_generation == 22 for layer in pool.layers)
    assert all(layer.dense_generation == 22 and layer.dense_end == 6 for layer in pool.layers)
    assert all(cache.stats.written_records == 13 for cache in session._layers.values())


def test_replay_rejects_replaced_storage_and_clock_or_range_mismatch(graph_pool):
    pool, session, state, _, _, _, _ = graph_pool
    recipe = finish_capture(pool, session, state)
    pool.layers[0].clock += 1
    with pytest.raises(RuntimeError, match="clock or residency"):
        recipe.validate()
    pool.layers[0].clock -= 1
    session.layer(0)._step_end += 1
    with pytest.raises(RuntimeError, match="append range"):
        recipe.validate()
    session.layer(0)._step_end -= 1
    original = pool.layers[0].host_to_device
    pool.layers[0].host_to_device = original.clone()
    with pytest.raises(RuntimeError, match="ownership or storage"):
        recipe.validate()
    assert any(tensor is original for tensor in recipe.storage)


def test_recipe_validates_committed_prefix_before_begin_and_rejects_branch_change(graph_pool):
    pool, session, state, _, _, _, _ = graph_pool
    recipe = finish_capture(pool, session, state)
    for cache in session._layers.values():
        cache._step_end = None
    recipe.validate(pending=False)
    with pytest.raises(RuntimeError, match="append range"):
        recipe.validate()
    pool.layers[0].resident_owner = session.owner
    pool.layers[0].resident_end = 4
    with pytest.raises(RuntimeError, match="clock or residency"):
        recipe.validate(pending=False)


def test_capture_suppresses_event_polling_but_does_not_allow_drain(graph_pool):
    pool, session, state, _, _, _, _ = graph_pool

    class Event:
        def query(self):
            pytest.fail("captured event must never be polled")

        def synchronize(self):
            pytest.fail("captured event must never be synchronized")

    with pool.graph_capture(session) as capture:
        state.capturing = True
        pool._writes.append(SimpleNamespace(event=Event()))
        pool._reap_writes()
        pool._reap_writes(make_room=True)
        with pytest.raises(RuntimeError, match="stream joins"):
            pool.drain()
        for cache in session._layers.values():
            cache.written = cache.indexer_visible_end = cache._step_end
        capture.join()
        state.capturing = False
        capture.finish()
    assert not pool._writes


@pytest.mark.parametrize("invalid", ["range", "transient", "clock", "sessions"])
def test_capture_admission_rejects_unsupported_state_without_poison(graph_pool, invalid):
    pool, session, _, _, _, _, _ = graph_pool
    if invalid == "range":
        session.layer(0)._step_end = pool.slots + 1
    elif invalid == "transient":
        session.layer(0)._transient_start = 4
    elif invalid == "clock":
        pool.layers[0].clock = PRIORITY_LIMIT - 2
    else:
        pool._sessions[99] = object()
    with pytest.raises(RuntimeError), pool.graph_capture(session):
        pytest.fail("unsupported state must fail before entering capture")
    assert not pool.poisoned and pool._graph_capture is None


def test_capture_error_preserves_original_and_retains_storage(graph_pool):
    pool, session, state, _, _, previous, _ = graph_pool
    source = torch.ones(2, 7)
    failure = KeyboardInterrupt("capture failed")
    with pytest.raises(KeyboardInterrupt) as caught, pool.graph_capture(session) as capture:
        state.capturing = True
        capture.retain_source(session, source)
        session._layers[0].written = 6
        state.capturing = False
        raise failure
    assert caught.value is failure
    assert pool.poisoned and pool._failed_graph_capture is capture
    assert capture.sources == [source] and capture.storage
    assert session._layers[0].written == 4 and pool._last_stream is previous
    assert pool._graph_capture is None


def test_capture_cleanup_failure_preserves_both_exceptions(graph_pool, monkeypatch):
    pool, session, state, _, _, _, _ = graph_pool
    failure = KeyboardInterrupt("capture failed")
    cleanup = RuntimeError("CPU state restore failed")

    def fail_restore(*args):
        raise cleanup

    monkeypatch.setattr(sparse_token_graph, "_restore_cpu", fail_restore)
    with pytest.raises(BaseExceptionGroup) as caught, pool.graph_capture(session):
        state.capturing = False
        raise failure
    assert caught.value.exceptions == (failure, cleanup)
    assert pool.poisoned and pool._failed_graph_capture is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_graph_preserves_two_layer_cold_recall_writeback_and_bookkeeping():
    from cache.sparse_token_pool import MISSING
    from operators.deepseek_v32.indexer import cache_ops

    pool = SharedSparseTokenPool(64, 576, 2, 64, device="cuda", metadata_ops=cache_ops)
    session = pool.allocate_session(64)
    caches = [session.layer(index) for index in range(2)]
    sources = [
        (torch.arange(4 * 576, device="cuda").reshape(4, 576) % 31 + index).to(torch.bfloat16)
        for index in range(2)
    ]
    selections = torch.tensor(
        [[0, 1, 4, 6, 8, 15, 31, 32 + query] for query in range(4)],
        dtype=torch.int32,
        device="cuda",
    )
    graph = recipe = None

    def body():
        outputs = []
        for cache, source in zip(caches, sources, strict=True):
            with cache.operation():
                cache.append(source)
                physical = cache._ensure_from_topk(selections)
                outputs.append(cache.records[physical.long()].clone())
        return outputs

    def describe():
        result = []
        for cache, layer in zip(caches, pool.layers, strict=True):
            host = cache.host_records()
            forward = layer.host_to_device.cpu().long()
            reverse = layer.device_to_host.cpu().long()
            free = layer.free.cpu()
            logical = cache.logical_to_global(torch.arange(36, device="cuda")).cpu()
            slots = forward[logical]
            resident = slots != MISSING
            live = (reverse != MISSING).nonzero().flatten()
            assert torch.equal(forward[reverse[live]], live)
            assert torch.equal(free[1:], reverse[1:] == MISSING)
            assert not bool(free[0]) and int(reverse[0]) == MISSING
            torch.testing.assert_close(cache.records[slots[resident].cuda()].cpu(), host[resident])
            priorities = torch.full((36,), -1, dtype=torch.int64)
            priorities[resident] = layer.priority.cpu()[slots[resident]]
            result.append(
                {
                    "host": host,
                    "resident": resident,
                    "priority": priorities,
                    "clock": (layer.clock, int(layer.clock_tensor.item())),
                    "lengths": (cache.length, cache.written, cache.indexer_visible_end),
                    "metrics": cache.metrics(),
                }
            )
        return result

    try:
        for index, cache in enumerate(caches):
            prefix = (torch.arange(32 * 576, device="cuda").reshape(32, 576) % 37 + index).to(
                torch.bfloat16
            )
            cache.begin_step(32)
            cache.append(prefix)
            cache.commit()
            with cache.operation():
                cache._evict(torch.arange(1, 65, device="cuda"), count_eviction=False)
            cache.reset_stats()
        snapshot = pool.snapshot()
        for cache in caches:
            cache.begin_step(4)
        expected_output = [value.clone() for value in body()]
        for cache in caches:
            cache.commit()
        expected = describe()
        pool.restore(snapshot)
        for cache in caches:
            cache.begin_step(4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), pool.graph_capture(session) as capture:
            with torch.cuda.graph(graph, stream=stream):
                actual_output = body()
                capture.join()
            recipe = capture.finish()
        for cache in caches:
            cache._step_end = None
        for _ in range(2):
            pool.restore(snapshot)
            recipe.validate(pending=False)
            for cache in caches:
                cache.begin_step(4)
            recipe.validate()
            graph.replay()
            torch.cuda.synchronize()
            recipe.apply()
            for cache in caches:
                cache.commit()
            actual = describe()
            for output, expected_value in zip(actual_output, expected_output, strict=True):
                torch.testing.assert_close(output, expected_value, rtol=0, atol=0)
            for result, baseline in zip(actual, expected, strict=True):
                for name in ("host", "resident", "priority"):
                    torch.testing.assert_close(result[name], baseline[name], rtol=0, atol=0)
                for name in ("clock", "lengths", "metrics"):
                    assert result[name] == baseline[name]
                assert result["metrics"]["recalled_records"] == 7
                assert result["metrics"]["device_to_host_bytes"] == 4 * 576 * 2
        assert len(recipe.sources) == 2
        assert recipe.source_bytes["hbm"] == sum(source.nbytes for source in sources)
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        if recipe is not None:
            recipe.close()
        pool.close()
