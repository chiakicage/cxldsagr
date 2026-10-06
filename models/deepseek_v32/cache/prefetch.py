"""Contiguous whole-history DMA into a finite, per-layer token pool.

A dense pool maps logical history i to slot i+1. The caller clears displaced
maps before the private stream copies one pinned-host span with cudaMemcpyAsync.
Publication follows that copy; consumers join the ticket event before use.
"""

from dataclasses import dataclass, field

import torch

from cache.sparse_token_cache import MISSING, WorkingSetTooLarge


@dataclass(eq=False)
class HistoryPrefetchTicket:
    requested_records: int
    _resident_records: int
    _fetched_records: int
    _record_bytes: int
    _owner: object = field(repr=False)
    _cache: object = field(repr=False)
    _host: torch.Tensor = field(repr=False)
    _records: torch.Tensor = field(repr=False)
    _map_generation: int = field(repr=False)
    _caller_stream: object = field(default=None, repr=False)
    _ready: object = field(default=None, repr=False)
    _waited: bool = field(default=False, repr=False)
    _active: bool = field(default=True, repr=False)

    @property
    def fetched_records(self):
        return self._fetched_records

    @property
    def resident_records(self):
        return self._resident_records

    @property
    def fetched_bytes(self):
        return self.fetched_records * self._record_bytes

    def add_metrics(self, target):
        for name in ("requested_records", "resident_records", "fetched_records"):
            target["dense_" + name] += getattr(self, name)


class PoolHistoryPrefetch:
    """Copy full contiguous history unless its direct layout is certified.

    An unproven or partially resident history is copied in full, including any
    resident records. Metrics count the actual copied records. Tickets borrow
    existing host/device storage and allocate no ID/count buffers. Submission
    failure disables reuse and retains both owners until successful drain.
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

    @property
    def pending_bytes(self):
        # Tickets retain borrowed pool storage, already charged to that pool.
        return 0

    def prefetch(self, cache):
        """Clear displaced maps, then copy and publish on the private stream."""
        self._check()
        cache._check()
        if not cache._shared or cache.device != self.device:
            raise ValueError("history prefetch requires a shared cache on the helper device")
        if not cache._pool.dense_contiguous:
            raise ValueError("history DMA requires a dense_contiguous pool")
        history = cache.host_written_end
        if history > cache.slots:
            raise WorkingSetTooLarge("whole-history prefetch requires H <= P")
        runs = cache.session._host_runs
        if len(runs) != 1 or runs[0][0] != 0 or runs[0][1] < history:
            raise ValueError("history DMA requires one contiguous host run")
        if not cache.host.is_contiguous() or not cache.records.is_contiguous():
            raise ValueError("history DMA requires contiguous host and HBM record storage")
        if cache._prefetch is not None:
            raise RuntimeError("finalize fused prefetch before whole-history prefetch")
        if any(ticket._cache is cache and not ticket._waited for ticket in self._tickets):
            raise RuntimeError("wait for this layer's prior history prefetch first")
        caller = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        if self._caller_stream is not None and caller != self._caller_stream:
            raise RuntimeError("one history prefetch execution requires one caller stream")
        self._caller_stream = caller
        try:
            return self._submit(cache, history, runs[0][2], caller)
        except BaseException:
            self.failed = True
            raise

    @staticmethod
    def _native(cache):
        native = cache._pool.native_metadata
        if type(cache).__dict__.get("native_metadata_compatible") is not True or any(
            not hasattr(native, name) for name in ("dense_history_clear", "dense_history_publish")
        ):
            raise ValueError("CUDA history DMA requires native contiguous-map metadata")
        return native

    @staticmethod
    def _clear_reference(cache, history, host_start):
        layer = cache._pool.layers[cache.layer_id]
        for slot in range(1, cache.slots + 1):
            old = int(cache.device_to_host[slot])
            incoming = host_start <= old < host_start + history
            if slot > history and not incoming:
                continue
            if old != MISSING:
                if (
                    not 0 <= old < cache._pool.host_capacity
                    or int(cache.host_to_device[old]) != slot
                ):
                    raise RuntimeError("inconsistent dense source reverse map")
                cache.host_to_device[old] = MISSING
                cache.stats.evicted_records += int(not incoming)
            cache.device_to_host[slot] = MISSING
            cache.age[slot] = -1
            layer.free[slot] = True

    @staticmethod
    def _publish_reference(cache, history, host_start, timestamp):
        layer = cache._pool.layers[cache.layer_id]
        global_ids = torch.arange(host_start, host_start + history, dtype=torch.int64)
        slots = torch.arange(1, history + 1, dtype=torch.int32)
        cache.host_to_device[host_start : host_start + history] = slots
        cache.device_to_host[1 : history + 1] = global_ids
        cache.age[1 : history + 1] = timestamp
        layer.free[1 : history + 1] = False
        cache.age[0] = MISSING
        layer.clock_tensor.fill_(timestamp + 1)

    def _submit(self, cache, history, host_start, caller):
        pool = cache._pool
        with cache.operation():
            layer = pool.layers[cache.layer_id]
            resident = pool.dense_history_resident(cache)
            native = self._native(cache) if self.device.type == "cuda" else None
            ticket = HistoryPrefetchTicket(
                history,
                history if resident else 0,
                0 if resident else history,
                cache.record_bytes,
                self,
                cache,
                cache.host,
                cache.records,
                layer.map_generation,
                caller,
            )
            # Retain borrowed storage before any metadata or copy submission.
            self._tickets.append(ticket)
            if resident:
                if not pool.protect_resident_history(cache):
                    ids = cache._global_range(0, history)
                    pool.protect(cache.layer_id, ids, preserve_append_plan=True)
                    pool.stamp(cache.layer_id, layer.append_order[:0], preserve_append_plan=True)
                return ticket
            pool.wait_host(cache.session, cache.layer_id)
            layer = pool._clock_event(cache.layer_id)
            timestamp = layer.clock
            pool.invalidate_residency(cache.layer_id)
            ticket._map_generation = layer.map_generation
            if native is None:
                self._clear_reference(cache, history, host_start)
            else:
                native.dense_history_clear(
                    cache.page_table,
                    cache.host_to_device,
                    cache.device_to_host,
                    cache.age,
                    layer.free,
                    layer.clock_tensor,
                    cache._native_totals[3:],
                    history=history,
                    host_start=host_start,
                    timestamp=timestamp,
                )
            layer.clock += 1
            metadata_ready = None
            if caller is not None:
                metadata_ready = torch.cuda.Event()
                metadata_ready.record(caller)
        if self._copy_stream is None:
            cache.records[1 : history + 1].copy_(cache.host[host_start : host_start + history])
            self._publish_reference(cache, history, host_start, timestamp)
        else:
            from operators.common.kv_transfer import copy_host_records_async

            self._copy_stream.wait_event(metadata_ready)
            with torch.cuda.stream(self._copy_stream):
                copy_host_records_async(
                    cache.host, cache.records, host_start=host_start, device_start=1, count=history
                )
                native.dense_history_publish(
                    cache.page_table,
                    cache.host_to_device,
                    cache.device_to_host,
                    cache.age,
                    layer.free,
                    layer.clock_tensor,
                    history=history,
                    host_start=host_start,
                    timestamp=timestamp,
                )
                ticket._ready = torch.cuda.Event()
                ticket._ready.record(self._copy_stream)
        cache.stats.recalled_records += history
        return ticket

    def wait(self, ticket):
        """Order this layer's consumers after copy and map publication complete."""
        self._check()
        if (
            not isinstance(ticket, HistoryPrefetchTicket)
            or ticket._owner is not self
            or not ticket._active
        ):
            raise ValueError("history prefetch ticket is foreign or expired")
        ticket._cache._check()
        cache = ticket._cache
        if cache._pool.layers[cache.layer_id].map_generation != ticket._map_generation:
            self.failed = True
            raise RuntimeError("dense history maps changed while its ticket was pending")
        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            if current != ticket._caller_stream:
                raise RuntimeError("wait must run on the ticket's caller stream")
            if ticket._ready is not None:
                try:
                    current.wait_event(ticket._ready)
                except BaseException:
                    self.failed = True
                    raise
        cache._pool.certify_dense_history(cache, ticket.requested_records)
        ticket._waited = True

    def drain(self):
        """Join every copy, retaining owners if any completion check fails."""
        errors = []
        if self._copy_stream is not None and (
            self.failed or any(ticket._ready is not None for ticket in self._tickets)
        ):
            try:
                self._copy_stream.synchronize()
            except BaseException as exc:  # noqa: BLE001 - retain every cleanup exception
                errors.append(exc)
        if self.failed and self._caller_stream is not None:
            try:
                self._caller_stream.synchronize()
            except BaseException as exc:  # noqa: BLE001 - retain every cleanup exception
                errors.append(exc)
        if errors:
            self.failed = True
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("dense DMA completion checks failed", errors)
        for ticket in self._tickets:
            ticket._active = False
            ticket._host = ticket._records = None
        self._tickets.clear()
        self._caller_stream = None

    def close(self):
        if self.closed:
            return
        self.drain()
        self._copy_stream = None
        self.closed = True
