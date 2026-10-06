"""Shared fixed-width host arena and per-layer token pools.

The pool is deliberately serial. A layer lease covers indexer, append, recall,
and attention. A stream change records the last consumer dependency before
the next lease can reuse records or workspace. Sessions own host pages; device eviction does not
return those pages to the host allocator.
"""

import weakref
from bisect import bisect_right
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from cache.host_allocation import (
    allocate_host_tensor,
    pinned_allocation_bytes,
)

PAGE_SIZE = 64
MISSING = torch.iinfo(torch.int32).max
PRIORITY_LIMIT = 9_000_000


def _device(device):
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    if result.type not in ("cpu", "cuda"):
        raise ValueError("token pools support CPU reference or CUDA")
    return result


def _tensor_bytes(tensors):
    seen, result = set(), {"hbm": 0, "dram": 0}
    for tensor in tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        device = tensor.device
        key = (device, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            nbytes = storage.nbytes()
            if device.type == "cpu" and tensor.is_pinned():
                nbytes = pinned_allocation_bytes(nbytes)
            result["hbm" if device.type == "cuda" else "dram"] += nbytes
    return result


@dataclass
class _LayerPool:
    host: torch.Tensor
    records: torch.Tensor
    host_to_device: torch.Tensor
    device_to_host: torch.Tensor
    priority: torch.Tensor
    free: torch.Tensor
    clock_tensor: torch.Tensor
    append_order: torch.Tensor
    clock: int = 0
    lease_owner: int | None = None
    map_generation: int = 0
    resident_owner: int | None = None
    resident_end: int = 0
    append_owner: int | None = None
    append_cursor: int = 0
    dense_owner: int | None = None
    dense_end: int = 0
    dense_generation: int = -1


@dataclass
class _HostWrite:
    owner: int
    layer: int
    start: int
    end: int
    event: object
    source: torch.Tensor


@dataclass(eq=False)
class PoolSnapshot:
    """Diagnostic HBM snapshot; host/indexer prefix contents are not copied."""

    pool_id: int
    topology: int
    layers: list
    sessions: dict
    valid: bool = True
    invalid_reason: str = ""
    diagnostic_bytes: dict = field(default_factory=dict)


class SharedSparseTokenPool:
    """One host arena per layer and one shared, finite device pool per layer.

    ``slots`` is the usable capacity P. Row zero is an additional padding
    sentinel and is never allocated. ``host_capacity`` includes padded
    persistent session pages. Optional candidate rows follow each layer's
    history pool; they have no host pages or maps and are leased per layer.
    """

    @staticmethod
    def _validate(host_capacity, width, layers, slots, candidate_slots=0):
        if any(type(x) is not int or x < 1 for x in (host_capacity, width, layers, slots)):
            raise ValueError("host_capacity, width, layers and slots must be positive integers")
        if host_capacity % PAGE_SIZE:
            raise ValueError("host_capacity must be a multiple of 64 tokens")
        if type(candidate_slots) is not int or candidate_slots < 0:
            raise ValueError("candidate_slots must be a nonnegative integer")
        if host_capacity >= MISSING or slots + candidate_slots >= MISSING:
            raise ValueError("pool IDs must fit the int32 mapping domain")

    @classmethod
    def estimate_shared_bytes(
        cls,
        host_capacity,
        width,
        layers,
        slots,
        *,
        dtype=torch.bfloat16,
        device="cuda",
        candidate_slots=0,
    ):
        """Exact persistent tensor capacities; execution temporaries are separate.

        ``execution_workspace_bytes`` reports an additional temporary allocation
        allowance, which the backend must reserve before model execution.
        """
        cls._validate(host_capacity, width, layers, slots, candidate_slots)
        size = dtype.itemsize
        host_per_layer = host_capacity * width * size
        if torch.device(device).type == "cuda":
            host_per_layer = pinned_allocation_bytes(host_per_layer)
        host = layers * host_per_layer
        per_layer = (
            (slots + 1) * (width * size + 8 + 8 + 1)
            + candidate_slots * width * size
            + host_capacity * 4
            + 8
            + slots * 8
        )
        workspace = slots * (4 + 8) + (slots + 1) * 8 + 4 + 3 * 8
        workspace += ((slots + 1 + candidate_slots + 31) // 32) * 4 + 4
        result = {
            "hbm": layers * per_layer + workspace,
            "dram": host + host_capacity // PAGE_SIZE * 4,
        }
        if torch.device(device).type == "cpu":
            result = {"hbm": 0, "dram": sum(result.values())}
        return result

    @staticmethod
    def estimate_session_bytes(capacity, *, layers=1, device="cuda"):
        if type(capacity) is not int or capacity < 1 or layers < 1:
            raise ValueError("capacity and layers must be positive")
        page_bytes = ((capacity + PAGE_SIZE - 1) // PAGE_SIZE) * 4
        size = page_bytes + layers * 8 * 8
        return (
            {"hbm": size, "dram": page_bytes}
            if torch.device(device).type == "cuda"
            else {"hbm": 0, "dram": size}
        )

    def __init__(
        self,
        host_capacity,
        width,
        layers,
        slots,
        *,
        device="cuda",
        dtype=torch.bfloat16,
        max_inflight_writes=2,
        metadata_ops=None,
        candidate_slots=0,
        dense_contiguous=False,
    ):
        self._validate(host_capacity, width, layers, slots, candidate_slots)
        if type(dense_contiguous) is not bool:
            raise ValueError("dense_contiguous must be a boolean")
        if type(max_inflight_writes) is not int or max_inflight_writes < 1:
            raise ValueError("max_inflight_writes must be positive")
        self.device = _device(device)
        if self.device.type == "cuda" and metadata_ops is not None:
            absent = object()
            initialize = getattr(metadata_ops, "initialize", absent)
            if initialize is not absent:
                if not callable(initialize):
                    raise TypeError("an explicit metadata initializer must be callable")
                initialize(self.device)
        self.host_capacity, self.width, self.num_layers, self.slots = (
            host_capacity,
            width,
            layers,
            slots,
        )
        self.dtype = dtype
        self.candidate_slots = candidate_slots
        self.max_inflight_writes = max_inflight_writes
        self.metadata_ops = metadata_ops
        self.dense_contiguous = dense_contiguous
        self.host_page_capacity = host_capacity // PAGE_SIZE
        self._free_pages = torch.arange(self.host_page_capacity - 1, -1, -1, dtype=torch.int32)
        self._free_page_count = self.host_page_capacity
        self._dense_free_runs = ((0, self.host_page_capacity),) if dense_contiguous else ()
        self.layers = []
        for _ in range(layers):
            host = allocate_host_tensor(
                (host_capacity, width), dtype=dtype, pin_memory=self.device.type == "cuda"
            )
            records = torch.empty(
                (slots + 1 + candidate_slots, width), device=self.device, dtype=dtype
            )
            records.zero_()
            priority = torch.full((slots + 1,), -1, device=self.device, dtype=torch.int64)
            priority[0] = MISSING
            free = torch.ones(slots + 1, device=self.device, dtype=torch.bool)
            free[0] = False
            self.layers.append(
                _LayerPool(
                    host,
                    records,
                    torch.full((host_capacity,), MISSING, device=self.device, dtype=torch.int32),
                    torch.full((slots + 1,), MISSING, device=self.device, dtype=torch.int64),
                    priority,
                    free,
                    torch.zeros(1, device=self.device, dtype=torch.int64),
                    torch.arange(1, slots + 1, device=self.device, dtype=torch.int64)
                    if dense_contiguous
                    else torch.empty(slots, device=self.device, dtype=torch.int64),
                )
            )
        # One workspace is shared across serial layer operations and sessions.
        self.free_slots = torch.empty(slots, device=self.device, dtype=torch.int32)
        self.allocation_log = torch.full(
            (slots + 1,), MISSING, device=self.device, dtype=torch.int64
        )
        self.counter = torch.zeros(1, device=self.device, dtype=torch.uint32)
        self.prefetch_stats = torch.zeros(3, device=self.device, dtype=torch.int64)
        self.miss_scratch = torch.empty(slots, device=self.device, dtype=torch.int64)
        self.union_bitmap = torch.empty(
            (slots + 1 + candidate_slots + 31) // 32, device=self.device, dtype=torch.uint32
        )
        self.union_count = torch.empty(1, device=self.device, dtype=torch.uint32)
        self._sessions = {}
        self._next_owner = 0
        self._topology = 0
        self._snapshots = weakref.WeakSet()
        self._active = None
        self._pending_prefetch = None
        self._transient_owners = {}
        self._depth = 0
        self._last_stream = None
        self._writes = deque()
        self._graph_capture = None
        self._failed_graph_capture = None
        self._copy_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self.closed = False
        self.poisoned = False
        expected = self.estimate_shared_bytes(
            host_capacity,
            width,
            layers,
            slots,
            dtype=dtype,
            device=self.device,
            candidate_slots=candidate_slots,
        )
        if self.shared_bytes() != expected:
            raise RuntimeError("shared pool capacity differs from its pre-allocation reservation")

    @property
    def free_host_pages(self):
        return self._free_page_count

    @property
    def execution_workspace_bytes(self):
        # argsort/where/rank, evict IDs and boolean indexing temporaries. Exact
        # query union/remap is reserved by the model from C and top-k.
        return self.estimate_execution_workspace_bytes(self.host_capacity, self.slots)

    @staticmethod
    def estimate_execution_workspace_bytes(host_capacity, slots):
        return 64 * (slots + 1) + 64 * host_capacity

    @property
    def pending_source_bytes(self):
        return _tensor_bytes(ticket.source for ticket in self._writes)

    def shared_bytes(self):
        tensors = [
            self.free_slots,
            self.allocation_log,
            self.counter,
            self.prefetch_stats,
            self.miss_scratch,
            self.union_bitmap,
            self.union_count,
            self._free_pages,
        ]
        for layer in self.layers:
            tensors.extend(
                (
                    layer.host,
                    layer.records,
                    layer.host_to_device,
                    layer.device_to_host,
                    layer.priority,
                    layer.free,
                    layer.clock_tensor,
                    layer.append_order,
                )
            )
        return _tensor_bytes(tensors)

    def _check(self):
        if self.closed:
            raise RuntimeError("shared token pool is closed")
        if self.poisoned:
            raise RuntimeError("shared token pool is poisoned after a CUDA failure")
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            capture = self._graph_capture
            if capture is None or not capture.authorizes_current():
                raise RuntimeError(
                    "shared sparse token cache requires an explicit graph capture lease"
                )

    def graph_capture(self, session):
        """Authorize one fixed, sole-session append recipe on the current stream.

        Enter before CUDA capture, call ``join()`` inside it, then ``finish()``
        after capture ends. The returned recipe owns captured writeback sources
        and publishes host bookkeeping only after a synchronized graph replay.
        """
        from cache.sparse_token_graph import SparseTokenGraphCapture

        return SparseTokenGraphCapture(self, session)

    def allocate_session(self, capacity):
        self._check()
        if self._graph_capture is not None:
            raise RuntimeError("session allocation is forbidden during graph capture preparation")
        if self._active is not None:
            raise RuntimeError("session allocation requires a quiescent pool")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("session capacity must be positive")
        if self.dense_contiguous and capacity > self.slots:
            raise ValueError("dense contiguous session capacity must not exceed P")
        pages = (capacity + PAGE_SIZE - 1) // PAGE_SIZE
        if pages > self.free_host_pages:
            raise ValueError("insufficient host pages for session capacity")
        # Construct metadata before mutating the allocator: allocation failure
        # must leave the current session set and host page ownership unchanged.
        next_runs = self._dense_free_runs
        if self.dense_contiguous:
            for index, (first, stop) in enumerate(self._dense_free_runs):
                if stop - first >= pages:
                    selected = torch.arange(first, first + pages, dtype=torch.int32)
                    replacement = ((first + pages, stop),) if first + pages < stop else ()
                    next_runs = (
                        self._dense_free_runs[:index]
                        + replacement
                        + self._dense_free_runs[index + 1 :]
                    )
                    break
            else:
                raise ValueError("insufficient contiguous host pages for dense session capacity")
        else:
            selected = self._free_pages[self._free_page_count - pages : self._free_page_count].flip(
                0
            )
        owner = self._next_owner
        session = SparseTokenSession(self, owner, capacity, selected)
        self._free_page_count -= pages
        self._dense_free_runs = next_runs
        self._next_owner += 1
        self._sessions[owner] = session
        self._topology += 1
        return session

    acquire_session = allocate_session

    def _wait_previous(self):
        self._check()
        if self.device.type == "cuda":
            try:
                current = torch.cuda.current_stream(self.device)
                if current == self._last_stream:
                    return
                # Also cover callers using ensure() without an explicit operation
                # context: their consumer has been submitted on the recorded stream
                # before the next serial cache call arrives.
                if self._last_stream is not None:
                    event = torch.cuda.Event()
                    event.record(self._last_stream)
                    current.wait_event(event)
            except BaseException:
                self.poisoned = True
                raise

    @contextmanager
    def operation(self, session, layer):
        session._check()
        identity = (session.owner, layer)
        owner = self._transient_owners.get(layer)
        if owner is not None and owner != session.owner:
            raise RuntimeError("another session owns transient candidate records for this layer")
        if self._pending_prefetch is not None and self._pending_prefetch != identity:
            raise RuntimeError("another layer owns the pending fused prefetch workspace")
        if self._active is not None and self._active != identity:
            raise RuntimeError("shared cache workspace already has an exclusive layer lease")
        outer = self._active is None
        if outer:
            self._wait_previous()
            state = self.layers[layer]
            if state.lease_owner != session.owner:
                self.invalidate_residency(layer)
                state.lease_owner = session.owner
            self._active = identity
        self._depth += 1
        body_error = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            self._depth -= 1
            if outer:
                if self.device.type == "cuda":
                    try:
                        # Stream order is sufficient while a serial caller stays
                        # on this stream. _wait_previous records the dependency
                        # only when another stream actually acquires the pool;
                        # that later event also covers consumers submitted after
                        # a standalone ensure() lease returned.
                        self._last_stream = torch.cuda.current_stream(self.device)
                    except BaseException as completion_error:
                        self.poisoned = True
                        if body_error is not None:
                            raise BaseExceptionGroup(
                                "token-pool execution and stream tracking failed",
                                [body_error, completion_error],
                            ) from None
                        raise
                self._active = None

    def _reap_writes(self, *, make_room=False):
        if self._graph_capture is not None:
            self._graph_capture.check_submission()
            return
        while self._writes:
            ticket = self._writes[0]
            if make_room and len(self._writes) >= self.max_inflight_writes:
                ticket.event.synchronize()
            elif not ticket.event.query():
                break
            self._writes.popleft()

    def write_host(self, session, layer, start, source):
        """Keep each complete source storage alive until its D2H event finishes."""
        if self._graph_capture is not None:
            self._graph_capture.retain_source(session, source)
        self._reap_writes(make_room=True)
        target = self.layers[layer].host

        def copy_pages():
            position, consumed = start, 0
            run = bisect_right(session._host_run_ends, position)
            while consumed < len(source):
                logical_start, logical_end, host_start = session._host_runs[run]
                count = min(logical_end - position, len(source) - consumed)
                base = host_start + position - logical_start
                target[base : base + count].copy_(
                    source[consumed : consumed + count], non_blocking=self.device.type == "cuda"
                )
                position += count
                consumed += count
                run += 1

        if self.device.type == "cpu":
            copy_pages()
            return
        stream = torch.cuda.current_stream(self.device)
        self._copy_stream.wait_stream(stream)
        with torch.cuda.stream(self._copy_stream):
            copy_pages()
            event = torch.cuda.Event()
            event.record(self._copy_stream)
            source.record_stream(self._copy_stream)
        self._writes.append(
            _HostWrite(session.owner, layer, start, start + len(source), event, source)
        )

    def wait_host(self, session, layer):
        """Establish a dependency before *any* read, including current suffix."""
        self._reap_writes()
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            for ticket in self._writes:
                if ticket.owner == session.owner and ticket.layer == layer:
                    stream.wait_event(ticket.event)

    def drain(self):
        if self._graph_capture is not None:
            raise RuntimeError(
                "graph capture requires stream joins instead of host synchronization"
            )
        try:
            if self.device.type == "cuda":
                if self._last_stream is not None:
                    self._last_stream.synchronize()
                if self._active is not None:
                    torch.cuda.current_stream(self.device).synchronize()
                for ticket in self._writes:
                    ticket.event.synchronize()
                self._copy_stream.synchronize()
            self._writes.clear()
        except BaseException:
            self.poisoned = True
            raise

    def _clock_event(self, layer_id):
        layer = self.layers[layer_id]
        if layer.clock >= PRIORITY_LIMIT:
            # Preserve timestamp ordering and equal-priority groups at a global
            # quiescent point before the official helper's search limit.
            self.drain()
            live = layer.device_to_host[1:] != MISSING
            values = layer.priority[1:][live]
            unique = torch.unique(values, sorted=True)
            ranked = torch.searchsorted(unique, values)
            layer.priority[1:][live] = ranked
            layer.clock = unique.numel()
        return layer

    @property
    def native_metadata(self):
        return self.metadata_ops if self.device.type == "cuda" else None

    def invalidate_residency(self, layer_id):
        """Discard host-side proofs before an untracked map/ownership change."""
        layer = self.layers[layer_id]
        layer.map_generation += 1
        layer.resident_owner = layer.append_owner = None
        layer.resident_end = layer.append_cursor = 0
        layer.dense_owner = None
        layer.dense_end = 0
        layer.dense_generation = -1

    def dense_history_resident(self, cache, end=None):
        """Prove the complete prefix occupies logical row i at physical i + 1.

        A competing owner or untracked map change invalidates the whole proof.
        Remaining physical hits do not qualify as a partially reusable DMA span.
        """
        cache._check()
        if not self.dense_contiguous or cache._pool is not self:
            return False
        history = cache.host_written_end if end is None else end
        if type(history) is not int or not 0 <= history <= min(cache.capacity, self.slots):
            return False
        layer = self.layers[cache.layer_id]
        return history == 0 or (
            layer.dense_owner == cache.session.owner
            and layer.dense_end >= history
            and layer.dense_generation == layer.map_generation
        )

    def certify_dense_history(self, cache, end):
        """Publish a direct-layout proof after ordered append or DMA completion."""
        cache._check()
        if not self.dense_contiguous or cache._pool is not self:
            raise ValueError("dense history certification requires this dense contiguous pool")
        if type(end) is not int or not 0 <= end <= min(cache.capacity, self.slots):
            raise ValueError("dense history endpoint exceeds the session or device pool")
        layer = self.layers[cache.layer_id]
        layer.dense_owner = layer.resident_owner = cache.session.owner
        layer.dense_end = layer.resident_end = end
        layer.dense_generation = layer.map_generation

    def append_slots(self, cache, start, count):
        """Reuse stable FIFO order only during an uninterrupted cold build.

        New slots are strictly newer than all unconsumed entries. Selection
        protection can refresh only already-consumed entries while this proof
        holds, so it cannot change the next victim. No victim is evicted here.
        """
        layer = self.layers[cache.layer_id]
        if self.dense_contiguous:
            if start + count > self.slots:
                raise ValueError("dense contiguous append exceeds P")
            if start and not self.dense_history_resident(cache, end=start):
                raise RuntimeError("dense append requires a resident contiguous prefix")
            if start == 0:
                self.invalidate_residency(cache.layer_id)
            layer.append_owner = cache.session.owner
            layer.append_cursor = start
            return layer.append_order[start : start + count]
        if start == 0 and count <= self.slots:
            layer.append_order.copy_(torch.argsort(layer.priority[1:], stable=True) + 1)
            layer.append_owner = cache.session.owner
            layer.append_cursor = 0
        if (
            layer.append_owner != cache.session.owner
            or layer.append_cursor != start
            or start + count > self.slots
            or (start and not cache.all_history_resident)
        ):
            return None
        return layer.append_order[start : start + count]

    def certify_append(self, cache, end):
        layer = self.layers[cache.layer_id]
        layer.resident_owner = layer.append_owner = cache.session.owner
        layer.resident_end = layer.append_cursor = end
        if self.dense_contiguous:
            self.certify_dense_history(cache, end)

    def protect_resident_history(self, cache):
        """Run both all-hit dense FIFO events without materializing a range."""
        native = self.native_metadata
        if not cache.all_history_resident or not hasattr(native, "protect_resident_history"):
            return False
        layer = self._clock_event(cache.layer_id)
        # The checked path normalizes between events at this exact boundary.
        if layer.clock == PRIORITY_LIMIT - 1:
            return False
        native.protect_resident_history(
            cache.page_table,
            layer.host_to_device,
            layer.priority,
            layer.clock_tensor,
            history=cache.host_written_end,
            timestamp=layer.clock,
        )
        layer.clock += 2
        return True

    def stamp(self, layer_id, physical, *, preserve_append_plan=False):
        """Official FIFO event, including empty allocation/protection calls."""
        layer = self._clock_event(layer_id)
        if not preserve_append_plan:
            layer.append_owner = None
        native = self.native_metadata
        if physical.numel() == 0 and hasattr(native, "empty_event"):
            native.empty_event(layer.priority, layer.clock_tensor, timestamp=layer.clock)
            layer.clock += 1
            return
        layer.priority[physical] = layer.clock
        layer.clock += 1
        layer.clock_tensor.fill_(layer.clock)
        layer.priority[0] = MISSING

    def protect(self, layer_id, global_ids, *, preserve_append_plan=False):
        layer = self._clock_event(layer_id)
        if not preserve_append_plan:
            layer.append_owner = None
        if self.native_metadata is not None:
            layer.clock_tensor.fill_(layer.clock)
            self.native_metadata.protect(
                global_ids, layer.host_to_device, layer.priority, layer.clock_tensor
            )
            layer.clock += 1
        else:
            physical = layer.host_to_device[global_ids].long()
            self.stamp(
                layer_id,
                physical[physical != MISSING],
                preserve_append_plan=preserve_append_plan,
            )

    def finalize_prefetch(self, layer_id):
        self.invalidate_residency(layer_id)
        layer = self._clock_event(layer_id)
        if self.native_metadata is not None:
            layer.clock_tensor.fill_(layer.clock)
            self.native_metadata.finalize_prefetch(
                layer.priority, layer.free, self.allocation_log, layer.clock_tensor
            )
            layer.clock += 1
        else:
            physical = torch.where(self.allocation_log != MISSING)[0]
            if physical.numel() and bool((physical == 0).any()):
                raise RuntimeError("fused prefetch allocated padding sentinel slot zero")
            layer.free[physical] = False
            self.stamp(layer_id, physical)

    def release_ids(self, layer_id, global_ids):
        self.invalidate_residency(layer_id)
        layer = self.layers[layer_id]
        if self.native_metadata is not None:
            self.native_metadata.release_ids(
                global_ids, layer.host_to_device, layer.device_to_host, layer.priority, layer.free
            )
        else:
            physical = layer.host_to_device[global_ids].long()
            physical = physical[physical != MISSING]
            layer.host_to_device[global_ids] = MISSING
            layer.device_to_host[physical] = MISSING
            layer.priority[physical] = -1
            layer.free[physical] = True

    def invalidate_snapshots(self, session, length):
        for snapshot in self._snapshots:
            old = snapshot.sessions.get(session.owner)
            if old is not None and any(item["length"] > length for item in old["layers"]):
                snapshot.valid = False
                snapshot.invalid_reason = "host/indexer prefix was truncated below snapshot length"

    def release_session(self, session):
        session._check()
        if self._graph_capture is not None:
            raise RuntimeError("session release is forbidden during graph capture preparation")
        session._check_release()
        if self._active is not None:
            raise RuntimeError("session release requires a quiescent pool")
        self.drain()
        if self._pending_prefetch is not None:
            if self._pending_prefetch[0] != session.owner:
                raise RuntimeError("another session owns a pending fused prefetch")
            for cache in session._layers.values():
                if cache._prefetch is not None:
                    cache.finalize_prefetch()
        for cache in session._layers.values():
            if cache.transient_start is not None:
                cache.discard_transient()
        ids = session.global_ids()
        for layer_id in range(self.num_layers):
            self.release_ids(layer_id, ids)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        pages = len(session._pages)
        if self.dense_contiguous:
            first = session._host_runs[0][2] // PAGE_SIZE
            merged = []
            for begin, end in sorted((*self._dense_free_runs, (first, first + pages))):
                if merged and begin < merged[-1][1]:
                    raise RuntimeError("dense host page ownership overlaps an existing free run")
                if merged and begin == merged[-1][1]:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((begin, end))
            self._dense_free_runs = tuple(merged)
        else:
            self._free_pages[self._free_page_count : self._free_page_count + pages].copy_(
                session._pages
            )
        self._free_page_count += pages
        del self._sessions[session.owner]
        session.released = True
        # Released Python handles must not retain otherwise-dead cache storage.
        # Live sessions still reference the shared tensors through their views.
        for cache in session._layers.values():
            for name in (
                "records",
                "host",
                "host_to_device",
                "device_to_host",
                "age",
                "_prefetch_totals",
                "_counter_totals",
                "_native_totals",
                "_native_recalled",
            ):
                setattr(cache, name, None)
        session._layers.clear()
        session._release_guard = None
        session.page_table = session._pages = session._prefetch_totals = None
        session._counter_totals = None
        session._host_runs = session._host_run_ends = ()
        self._topology += 1

    def snapshot(self):
        self._check()
        if self._active is not None or any(
            cache._step_end is not None
            for session in self._sessions.values()
            for cache in session._layers.values()
        ):
            raise RuntimeError("snapshot requires quiescent, committed sessions")
        self.drain()
        layers, tensors = [], []
        for layer in self.layers:
            state = {
                name: getattr(layer, name).detach().cpu().clone()
                for name in (
                    "records",
                    "host_to_device",
                    "device_to_host",
                    "priority",
                    "free",
                    "clock_tensor",
                )
            }
            state["clock"] = layer.clock
            if self.dense_contiguous:
                state["dense_owner"] = (
                    layer.dense_owner if layer.dense_generation == layer.map_generation else None
                )
                state["dense_end"] = layer.dense_end
            tensors.extend(value for value in state.values() if isinstance(value, torch.Tensor))
            layers.append(state)
        sessions = {}
        for owner, session in self._sessions.items():
            sessions[owner] = {
                "pages": tuple(session._pages.tolist()),
                "layers": [session.layer(i)._snapshot_state() for i in range(self.num_layers)],
            }
        snapshot = PoolSnapshot(
            id(self), self._topology, layers, sessions, diagnostic_bytes=_tensor_bytes(tensors)
        )
        self._snapshots.add(snapshot)
        return snapshot

    snapshot_prefix = snapshot

    def restore(self, snapshot):
        self._check()
        if self._active is not None:
            raise RuntimeError("snapshot restore requires a quiescent pool")
        if not isinstance(snapshot, PoolSnapshot) or snapshot.pool_id != id(self):
            raise ValueError("snapshot belongs to another shared pool")
        if snapshot.topology != self._topology or snapshot.sessions.keys() != self._sessions.keys():
            raise ValueError("session set or host ID ownership changed since snapshot")
        if not snapshot.valid:
            raise ValueError(snapshot.invalid_reason)
        if any(
            cache._step_end is not None
            for s in self._sessions.values()
            for cache in s._layers.values()
        ):
            raise RuntimeError("abort or commit all steps before snapshot restore")
        self.drain()
        for owner, session in self._sessions.items():
            state = snapshot.sessions[owner]
            if tuple(session._pages.tolist()) != state["pages"]:
                raise ValueError("session host page allocation changed")
            for i, cache_state in enumerate(state["layers"]):
                if session.layer(i).length < cache_state["length"]:
                    raise ValueError("snapshot host prefix is no longer valid")
        for layer, state in zip(self.layers, snapshot.layers, strict=True):
            for name in (
                "records",
                "host_to_device",
                "device_to_host",
                "priority",
                "free",
                "clock_tensor",
            ):
                getattr(layer, name).copy_(state[name])
            layer.clock = state["clock"]
        for layer_id in range(self.num_layers):
            self.invalidate_residency(layer_id)
        for owner, session in self._sessions.items():
            for i, state in enumerate(snapshot.sessions[owner]["layers"]):
                session.layer(i)._restore_state(state)
        if self.dense_contiguous:
            for layer_id, state in enumerate(snapshot.layers):
                owner = state["dense_owner"]
                if owner is not None:
                    self.certify_dense_history(
                        self._sessions[owner].layer(layer_id), state["dense_end"]
                    )
                    self.layers[layer_id].lease_owner = owner
        self.allocation_log.fill_(MISSING)
        self.counter.zero_()
        self.prefetch_stats.zero_()
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    restore_prefix = restore

    def close(self):
        if self.closed:
            return
        if self._graph_capture is not None:
            raise RuntimeError("pool close is forbidden during graph capture preparation")
        if self._active is not None:
            raise RuntimeError("pool close requires a quiescent pool")
        self.drain()
        for session in list(self._sessions.values()):
            self.release_session(session)
        self.layers.clear()
        for name in (
            "free_slots",
            "allocation_log",
            "counter",
            "prefetch_stats",
            "miss_scratch",
            "union_bitmap",
            "union_count",
            "_free_pages",
        ):
            setattr(self, name, None)
        self._free_page_count = 0
        self._dense_free_runs = ()
        self.closed = True


class SparseTokenSession:
    """An owner-checked session page table and independent layer transactions."""

    def __init__(self, pool, owner, capacity, pages):
        self.pool, self.owner, self.capacity = pool, owner, capacity
        self._pages = pages
        runs = []
        for logical_page, host_page in enumerate(pages.tolist()):
            start = logical_page * PAGE_SIZE
            if runs and runs[-1][2] + (start - runs[-1][0]) == host_page * PAGE_SIZE:
                runs[-1] = (runs[-1][0], start + PAGE_SIZE, runs[-1][2])
            else:
                runs.append((start, start + PAGE_SIZE, host_page * PAGE_SIZE))
        self._host_runs = tuple(runs)
        self._host_run_ends = tuple(run[1] for run in runs)
        self.page_table = pages.to(device=pool.device, dtype=torch.int32)
        self._layers = {}
        # All per-layer counter storage is reserved at session admission.
        self._counter_totals = torch.zeros(
            (pool.num_layers, 8), device=pool.device, dtype=torch.int64
        )
        self._prefetch_totals = self._counter_totals[:, :3]
        self.released = False
        self._release_guard = None

    def bind_release_guard(self, guard):
        """Attach the owning provider's completion/lease check before release."""
        self._check()
        if self._release_guard is not None:
            raise RuntimeError("session already has a release guard")
        if not callable(guard):
            raise TypeError("release guard must be callable")
        self._release_guard = guard

    def _check_release(self):
        if self._release_guard is not None:
            self._release_guard()

    def _check(self):
        self.pool._check()
        if self.released or self.pool._sessions.get(self.owner) is not self:
            raise RuntimeError("session view was released or its owner epoch is stale")

    @property
    def host_pages(self):
        return len(self._pages)

    @property
    def host_tokens(self):
        return self.host_pages * PAGE_SIZE

    def layer(self, index):
        self._check()
        if not 0 <= index < self.pool.num_layers:
            raise ValueError("layer index outside shared pool")
        if index not in self._layers:
            from cache.sparse_token_cache import SparseTokenCache

            cache = object.__new__(SparseTokenCache)
            cache._bind_shared(self.pool, self, index)
            self._layers[index] = cache
        return self._layers[index]

    def global_ids(self):
        # Include padding when returning ownership to the allocator.
        local = torch.arange(self.host_tokens, device=self.pool.device, dtype=torch.int64)
        return self.page_table[local // PAGE_SIZE].long() * PAGE_SIZE + local % PAGE_SIZE

    def session_bytes(self):
        self._check()
        return _tensor_bytes((self._pages, self.page_table, self._prefetch_totals))

    def release(self):
        self.pool.release_session(self)

    def drain(self):
        self._check()
        self.pool.drain()
