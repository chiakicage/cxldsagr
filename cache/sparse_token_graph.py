"""Host bookkeeping and ownership for a fixed sparse-pool CUDA Graph recipe.

This module captures no model computation. The model owns CUDA capture/replay,
prefix restoration and synchronization; this lease authorizes existing cache
operations and retains every tensor used by their captured asynchronous writes.
"""

from copy import copy

import torch

from cache.sparse_token_pool import PRIORITY_LIMIT, _tensor_bytes

_LAYER_FIELDS = (
    "clock",
    "lease_owner",
    "map_generation",
    "resident_owner",
    "resident_end",
    "append_owner",
    "append_cursor",
    "dense_owner",
    "dense_end",
    "dense_generation",
)
_CACHE_FIELDS = ("length", "written", "indexer_visible_end", "_step_end", "_transient_start")


def _state(pool, session):
    return {
        "layers": [{name: getattr(layer, name) for name in _LAYER_FIELDS} for layer in pool.layers],
        "caches": {
            index: {
                **{name: getattr(cache, name) for name in _CACHE_FIELDS},
                "stats": copy(cache.stats),
            }
            for index, cache in session._layers.items()
        },
    }


def _restore_cpu(pool, session, state):
    for layer, saved in zip(pool.layers, state["layers"], strict=True):
        for name in _LAYER_FIELDS:
            setattr(layer, name, saved[name])
    for index, saved in state["caches"].items():
        cache = session._layers[index]
        for name in _CACHE_FIELDS:
            setattr(cache, name, saved[name])
        cache.stats = copy(saved["stats"])
        cache._prefetch = None


def _relative_layer(state):
    result = {name: value for name, value in state.items() if name != "map_generation"}
    result["dense_generation"] = (
        None
        if state["dense_generation"] == -1
        else state["dense_generation"] - state["map_generation"]
    )
    return result


def _storage_tensors(pool, session):
    tensors = [
        pool.free_slots,
        pool.allocation_log,
        pool.counter,
        pool.prefetch_stats,
        pool.miss_scratch,
        pool.union_bitmap,
        pool.union_count,
        session.page_table,
        session._counter_totals,
    ]
    for layer in pool.layers:
        tensors.extend(value for value in vars(layer).values() if isinstance(value, torch.Tensor))
    return tuple(tensors)


def _storage_identity(tensors):
    return tuple(
        (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )
        for tensor in tensors
    )


class SparseTokenGraphRecipe:
    """Fixed-prefix cache side effects, installed after successful replay completion.

    Native clock arguments are captured constants. Validation therefore requires
    matching clocks, pending append lengths, ownership and residency branches.
    Map generations are relative because a legitimate snapshot restore advances
    their host epochs without changing the restored GPU map contents.
    """

    def __init__(self, capture, post):
        self.pool, self.session = capture.pool, capture.session
        self.before, self.after = capture.before, post
        self.topology = self.pool._topology
        self.caches = tuple(self.session._layers.items())
        self.storage = capture.storage
        self.storage_identity = _storage_identity(self.storage)
        self.sources = tuple(capture.sources)
        self.source_bytes = _tensor_bytes(self.sources)
        self.closed = False

    def validate(self, *, pending=True):
        """Validate a committed prefix or its freshly started append transaction."""
        if self.closed:
            raise RuntimeError("sparse token graph recipe is closed")
        pool, session = self.pool, self.session
        session._check()
        if pool._graph_capture is not None or pool._active is not None or pool._writes:
            raise RuntimeError("graph replay requires a quiescent cache owner")
        if (
            pool._topology != self.topology
            or len(pool._sessions) != 1
            or tuple(session._layers.items()) != self.caches
            or _storage_identity(_storage_tensors(pool, session)) != self.storage_identity
        ):
            raise RuntimeError("sparse token graph cache ownership or storage changed")
        if pool._pending_prefetch is not None or pool._transient_owners:
            raise RuntimeError("graph replay cannot borrow an active prefetch or transient suffix")
        current = _state(pool, session)
        for expected, actual in zip(self.before["layers"], current["layers"], strict=True):
            if _relative_layer(expected) != _relative_layer(actual):
                raise RuntimeError("sparse token graph clock or residency recipe does not match")
        for index, expected in self.before["caches"].items():
            actual = current["caches"][index]
            expected_range = {name: expected[name] for name in _CACHE_FIELDS}
            if not pending:
                expected_range.update(
                    written=expected["length"],
                    indexer_visible_end=expected["length"],
                    _step_end=None,
                )
            if any(actual[name] != value for name, value in expected_range.items()):
                raise RuntimeError("sparse token graph pending append range does not match")
            if session._layers[index]._prefetch is not None:
                raise RuntimeError("sparse token graph cache has an active prefetch lease")

    def apply(self):
        """Install captured host effects after the caller synchronizes its replay.

        This does not commit the model step. Every layer must be applied before
        the model performs its usual all-layer commit. GPU counters are already
        updated by the graph and must not be added a second time here.
        """
        self.validate()
        pool, session = self.pool, self.session
        for layer, before, after in zip(
            pool.layers, self.before["layers"], self.after["layers"], strict=True
        ):
            generation = layer.map_generation + after["map_generation"] - before["map_generation"]
            for name in _LAYER_FIELDS:
                if name not in ("map_generation", "dense_generation"):
                    setattr(layer, name, after[name])
            layer.map_generation = generation
            layer.dense_generation = (
                -1
                if after["dense_generation"] == -1
                else generation + after["dense_generation"] - after["map_generation"]
            )
        for index, after in self.after["caches"].items():
            cache = session._layers[index]
            before = self.before["caches"][index]
            for name in _CACHE_FIELDS:
                setattr(cache, name, after[name])
            for name, value in vars(after["stats"]).items():
                current = getattr(cache.stats, name)
                setattr(
                    cache.stats,
                    name,
                    max(current, value)
                    if name == "max_working_set"
                    else current + value - getattr(before["stats"], name),
                )
        pool._last_stream = torch.cuda.current_stream(pool.device)

    def close(self):
        """Release recipe references after the model drains and destroys its graph."""
        self.storage = self.sources = self.caches = ()
        self.closed = True


class SparseTokenGraphCapture:
    """Explicit authorization for one bounded append capture, never general capture."""

    def __init__(self, pool, session):
        self.pool, self.session = pool, session
        self.sources = []
        self.storage = ()
        self.stream = self.before = self.recipe = None
        self.entered = self.joined = self.finished = False

    def __enter__(self):
        pool, session = self.pool, self.session
        session._check()
        if self.entered or pool._graph_capture is not None:
            raise RuntimeError("sparse token graph capture leases cannot be nested or reused")
        if pool.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("sparse token graph preparation must precede CUDA capture")
        if session.pool is not pool or len(pool._sessions) != 1 or not session._layers:
            raise RuntimeError("sparse token graph capture requires its sole live session")
        if pool._active is not None or pool._pending_prefetch is not None or pool._transient_owners:
            raise RuntimeError("sparse token graph capture requires quiescent cache leases")
        ranges = set()
        for cache in session._layers.values():
            if (
                cache._step_end is None
                or cache._transient_start is not None
                or cache.written != cache.length
                or cache.indexer_visible_end != cache.length
                or cache._prefetch is not None
                or not cache.length < cache._step_end <= min(cache.capacity, pool.slots)
            ):
                raise RuntimeError("graph capture requires a fresh bounded persistent append step")
            ranges.add((cache.length, cache._step_end))
        if len(ranges) != 1:
            raise RuntimeError("graph capture requires one append range across all layers")
        # Each layer runs a single query batch. Reserve ample room for its
        # append, preparation, selection/protection and publication events.
        if any(layer.clock >= PRIORITY_LIMIT - 16 for layer in pool.layers):
            raise RuntimeError("graph capture requires normalized FIFO clocks with event headroom")
        pool.drain()
        self.before = _state(pool, session)
        self.storage = _storage_tensors(pool, session)
        self.stream = torch.cuda.current_stream(pool.device)
        self.previous_stream = pool._last_stream
        # The drain established completion. Never record a pre-capture stream
        # event from inside capture merely to change the serial lease stream.
        pool._last_stream = None
        pool._graph_capture = self
        self.entered = True
        return self

    def authorizes_current(self):
        return (
            self.entered
            and not self.joined
            and self.pool._graph_capture is self
            and torch.cuda.current_stream(self.pool.device) == self.stream
        )

    def check_submission(self):
        if not self.authorizes_current() or not torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cache graph submissions require the authorized capture stream")

    def retain_source(self, session, source):
        self.check_submission()
        if session is not self.session:
            raise RuntimeError("captured writeback belongs to another session")
        # Retain before any copy/event submission can fail. The graph owner
        # accounts for all source storage, independently of eager ticket limits.
        self.sources.append(source)

    def join(self):
        """Join every captured writeback before ending the CUDA capture."""
        self.check_submission()
        if self.pool._active is not None or self.pool._pending_prefetch is not None:
            raise RuntimeError("finish layer and prefetch leases before joining the graph")
        if self.sources:
            self.stream.wait_stream(self.pool._copy_stream)
        self.joined = True

    def finish(self):
        """Build the replay recipe after CUDA capture, before this context exits."""
        if (
            not self.entered
            or self.finished
            or not self.joined
            or torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("finish requires a joined, completed CUDA capture")
        if any(cache.written != cache._step_end for cache in self.session._layers.values()):
            raise RuntimeError("captured graph did not finish every cache append")
        self.recipe = SparseTokenGraphRecipe(self, _state(self.pool, self.session))
        self.finished = True
        return self.recipe

    def __exit__(self, error_type, error, traceback):
        pool = self.pool
        cleanup = None
        try:
            _restore_cpu(pool, self.session, self.before)
            pool._writes.clear()
            pool._pending_prefetch = None
            pool._active = None
            pool._depth = 0
            pool._last_stream = self.previous_stream
        except BaseException as exc:  # noqa: BLE001 -- retain capture and cleanup errors
            cleanup = exc
        finally:
            pool._graph_capture = None
        if error is not None or cleanup is not None or not self.finished:
            pool.poisoned = True
            # Hold all owners even if capture failure leaves completion unknown.
            pool._failed_graph_capture = self
            if cleanup is not None:
                if error is not None:
                    raise BaseExceptionGroup(
                        "cache graph capture and host-state cleanup failed", [error, cleanup]
                    ) from None
                raise cleanup
            if error is None:
                raise RuntimeError("cache graph capture exited without a finished recipe")
        return False
