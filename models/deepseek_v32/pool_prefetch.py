"""Whole-history lookahead into the existing finite, per-layer token pool.

Metadata preparation uses the caller's pool lease. Only record copies run on
the private stream, so they can overlap a different layer's attention without
using the pool's shared metadata workspace. The execution owner must wait for
each ticket before touching that layer, and drain before rollback or release.
"""

from dataclasses import dataclass, field

import torch

from cache.sparse_token_cache import MISSING, SparseTokenCache, WorkingSetTooLarge
from cache.sparse_token_pool import PRIORITY_LIMIT


@dataclass(eq=False)
class HistoryPrefetchTicket:
    requested_records: int
    resident_records: int
    fetched_records: int
    fetched_bytes: int
    _owner: object = field(repr=False)
    _cache: object = field(repr=False)
    _host: torch.Tensor = field(repr=False)
    _records: torch.Tensor = field(repr=False)
    _host_ids: torch.Tensor = field(repr=False)
    _slots: torch.Tensor = field(repr=False)
    _map_generation: int = field(repr=False)
    _caller_stream: object = field(default=None, repr=False)
    _ready: object = field(default=None, repr=False)
    _waited: bool = field(default=False, repr=False)
    _active: bool = field(default=True, repr=False)


class PoolHistoryPrefetch:
    """Prefetch every initialized history token, copying only HBM misses.

    The helper owns no KV storage and never touches candidate tail rows. Tickets
    retain copy inputs until drain; a failed submission terminates helper reuse.
    All tickets in an execution must use the same caller stream. Successful drain
    ends that execution and permits a later one to use another caller stream.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("history prefetch requires CPU or CUDA")
        self._copy_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self._caller_stream = None
        self._tickets = []
        self.failed = False
        self.closed = False

    def _check(self):
        if self.closed or self.failed:
            raise RuntimeError("history prefetch helper is closed or failed")

    def prefetch(self, cache):
        """Reserve the full host history on main, then launch independent copies."""
        self._check()
        cache._check()
        if not cache._shared or cache.device != self.device:
            raise ValueError("history prefetch requires a shared cache on the helper device")
        history = cache.host_written_end
        if history > cache.slots:
            raise WorkingSetTooLarge("whole-history prefetch requires H <= P")
        if cache._prefetch is not None:
            raise RuntimeError("finalize fused prefetch before whole-history prefetch")
        if any(ticket._cache is cache and not ticket._waited for ticket in self._tickets):
            raise RuntimeError("wait for this layer's prior history prefetch first")
        caller = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        if self._caller_stream is not None and caller != self._caller_stream:
            raise RuntimeError("one history prefetch execution requires one caller stream")
        self._caller_stream = caller
        try:
            return self._submit(cache, history, caller)
        except BaseException:
            # Metadata reservations precede copies. A failure at any later
            # submission boundary must disable reuse until execution cleanup.
            self.failed = True
            raise

    def _reserve_native(self, cache, layer, history):
        pool = cache._pool
        native = pool.native_metadata
        if type(cache) is not SparseTokenCache or not hasattr(native, "dense_history_classify"):
            return None
        layer = pool._clock_event(cache.layer_id)
        if layer.clock == PRIORITY_LIMIT - 1:
            return None
        timestamp = layer.clock
        native.dense_history_classify(
            cache.page_table,
            cache.host_to_device,
            cache.device_to_host,
            cache.age,
            layer.clock_tensor,
            pool.free_slots,
            pool.counter,
            history=history,
            timestamp=timestamp,
        )
        layer.clock += 1
        count = int(pool.counter.item())
        victims = misses = chosen = layer.append_order[:0]
        if count:
            layer.append_owner = None
            pool.wait_host(cache.session, cache.layer_id)
            victims = torch.argsort(cache.age[1:], stable=True)
            torch.cumsum(pool.free_slots, dim=0, dtype=torch.int64, out=pool.miss_scratch)
            # Both ticket arrays own M elements. No view of shared scratch or
            # the P-element sort may survive into the private copy stream.
            misses = torch.empty(count, device=self.device, dtype=torch.int64)
            chosen = torch.empty_like(misses)
            pool.invalidate_residency(cache.layer_id)
        native.dense_history_reserve(
            pool.miss_scratch,
            victims,
            cache.page_table,
            cache.host_to_device,
            cache.device_to_host,
            cache.age,
            layer.free,
            layer.clock_tensor,
            cache._native_totals[3:],
            misses,
            chosen,
            history=history,
            timestamp=timestamp,
        )
        layer.clock += 1
        return misses, chosen

    def _submit(self, cache, history, caller):
        with cache.operation():
            layer = cache._pool.layers[cache.layer_id]
            chosen = layer.append_order[:0]
            if cache._pool.protect_resident_history(cache):
                missing = chosen
            else:
                native_ids = self._reserve_native(cache, layer, history)
                if native_ids is not None:
                    missing, chosen = native_ids
                else:
                    global_ids = cache._global_range(0, history)
                    missing = (
                        layer.append_order[:0]
                        if cache.all_history_resident
                        else global_ids[cache.host_to_device[global_ids] == MISSING]
                    )
                    cache._pool.protect(
                        cache.layer_id, global_ids, preserve_append_plan=not missing.numel()
                    )
                    # Maps precede copies; wait(ticket) joins target readiness.
                    if missing.numel():
                        cache._pool.wait_host(cache.session, cache.layer_id)
                        chosen = cache._available_slots(global_ids, missing.numel())
                        cache.host_to_device[missing] = chosen.int()
                        cache.device_to_host[chosen] = missing
                        layer.free[chosen] = False
                    cache._pool.stamp(
                        cache.layer_id, chosen, preserve_append_plan=not missing.numel()
                    )
            metadata_ready = None
            if caller is not None and missing.numel():
                metadata_ready = torch.cuda.Event()
                metadata_ready.record(caller)
        ticket = HistoryPrefetchTicket(
            history,
            history - missing.numel(),
            missing.numel(),
            missing.numel() * cache.record_bytes,
            self,
            cache,
            cache.host,
            cache.records,
            missing,
            chosen,
            layer.map_generation,
            caller,
        )
        # Retain copy inputs even if submission fails after enqueueing work.
        self._tickets.append(ticket)
        if not missing.numel():
            pass
        elif self._copy_stream is None:
            cache.records[chosen] = cache.host[missing]
        else:
            from operators.common.kv_transfer import gather_host_records

            self._copy_stream.wait_event(metadata_ready)
            with torch.cuda.stream(self._copy_stream):
                gather_host_records(cache.host, cache.records, missing, chosen)
                missing.record_stream(self._copy_stream)
                chosen.record_stream(self._copy_stream)
                ticket._ready = torch.cuda.Event()
                ticket._ready.record(self._copy_stream)
        cache.stats.recalled_records += ticket.fetched_records
        return ticket

    def wait(self, ticket):
        """Order this layer's consumers after its whole-history copy completes."""
        self._check()
        if (
            not isinstance(ticket, HistoryPrefetchTicket)
            or ticket._owner is not self
            or not ticket._active
        ):
            raise ValueError("history prefetch ticket is foreign or expired")
        ticket._cache._check()
        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            if current != ticket._caller_stream:
                raise RuntimeError("wait must run on the ticket's caller stream")
            if ticket._ready is not None:
                current.wait_event(ticket._ready)
        cache = ticket._cache
        layer = cache._pool.layers[cache.layer_id]
        if layer.map_generation == ticket._map_generation:
            layer.resident_owner = cache.session.owner
            layer.resident_end = ticket.requested_records
        ticket._waited = True

    def drain(self):
        """Join all copies, including a speculative layer with no consumer."""
        if self._copy_stream is not None and (
            self.failed or any(ticket._ready is not None for ticket in self._tickets)
        ):
            try:
                self._copy_stream.synchronize()
            except BaseException:
                self.failed = True
                raise
        for ticket in self._tickets:
            ticket._active = False
            ticket._host = ticket._records = ticket._host_ids = ticket._slots = None
        self._tickets.clear()
        self._caller_stream = None

    def close(self):
        if self.closed:
            return
        self.drain()
        self._copy_stream = None
        self.closed = True
