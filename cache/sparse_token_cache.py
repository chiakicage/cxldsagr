"""Model-independent sparse token cache views and a CPU reference path."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from cache.sparse_token_pool import (
    MISSING,
    PAGE_SIZE,
    PRIORITY_LIMIT,
    SharedSparseTokenPool,
    _device,
)


class WorkingSetTooLarge(ValueError):
    """The caller must split query consumption without changing its selection."""


@dataclass
class CacheStats:
    written_records: int = 0
    transient_written_records: int = 0
    recalled_records: int = 0
    evicted_records: int = 0
    max_working_set: int = 0
    capacity_splits: int = 0
    selection_records: int = 0
    resident_selection_records: int = 0


class SparseTokenCache:
    """Session/layer view, or a standalone resident compatibility cache.

    Shared views use global maps. ``logical_to_global`` translates logical IDs
    before direct map inspection. ``ensure`` accepts session-logical IDs. The
    ``operation()`` lease must include the attention consumer.
    """

    def __init__(
        self, capacity, width, *, device="cuda", dtype=torch.bfloat16, slots=None, candidate_slots=0
    ):
        if capacity < 1 or width < 1 or (slots is not None and not 1 <= slots <= capacity):
            raise ValueError("capacity, width and bounded slot count must be positive")
        if type(candidate_slots) is not int or candidate_slots < 0:
            raise ValueError("candidate_slots must be a nonnegative integer")
        if slots is not None:
            pool = SharedSparseTokenPool(
                ((capacity + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE,
                width,
                1,
                slots,
                device=device,
                dtype=dtype,
                candidate_slots=candidate_slots,
            )
            session = pool.allocate_session(capacity)
            self._bind_shared(pool, session, 0)
            session._layers[0] = self
            self._standalone = True
            return
        self.capacity, self.width = capacity, width
        self.offload = False
        self.device = _device(device)
        self.slots = capacity
        self._candidate_slots = candidate_slots
        self.records = torch.empty(
            (capacity + candidate_slots, width), device=self.device, dtype=dtype
        )
        self.host = None
        self.host_to_device = torch.full(
            (capacity,), MISSING, dtype=torch.int32, device=self.device
        )
        self.device_to_host = torch.full(
            (capacity,), MISSING, dtype=torch.int64, device=self.device
        )
        self.age = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        self.length = self.written = self.indexer_visible_end = 0
        self._step_end = None
        self._transient_start = None
        self._clock = 0
        self.stats = CacheStats()
        self.prefetch_counts = []
        self._pool = None

    def _bind_shared(self, pool, session, layer):
        self._pool, self.session, self.layer_id = pool, session, layer
        self.capacity, self.width, self.slots = session.capacity, pool.width, pool.slots
        self.device, self.offload = pool.device, True
        shared = pool.layers[layer]
        self.records, self.host = shared.records, shared.host
        self.host_to_device, self.device_to_host = shared.host_to_device, shared.device_to_host
        self.age = shared.priority
        self.length = self.written = self.indexer_visible_end = 0
        self._step_end = None
        self._transient_start = None
        self._prefetch = None
        self._counter_totals = session._counter_totals[layer]
        self._prefetch_totals = self._counter_totals[:3]
        self._native_totals = self._counter_totals[3:]
        self._standalone = False
        self.stats = CacheStats()
        # Compatibility only: no per-chunk counter tensor list is retained.
        self.prefetch_counts = []

    @property
    def _shared(self):
        return getattr(self, "_pool", None) is not None

    @property
    def _clock(self):
        return self._pool.layers[self.layer_id].clock if self._shared else self._resident_clock

    @_clock.setter
    def _clock(self, value):
        if self._shared:
            self._pool.layers[self.layer_id].clock = value
        else:
            self._resident_clock = value

    @property
    def record_bytes(self):
        return self.width * self.records.element_size()

    @property
    def committed_length(self):
        return self.length

    @property
    def kv_written_end(self):
        return self.written

    @property
    def transient_start(self):
        """First GPU-only logical token, or None outside a transient step."""
        return getattr(self, "_transient_start", None)

    @property
    def host_written_end(self):
        return self.written if self.transient_start is None else self.transient_start

    @property
    def all_history_resident(self):
        """Conservative host-side proof, never inferred from capacity alone."""
        self._check()
        history = self.host_written_end
        if not self._shared:
            return not self.offload
        layer = self._pool.layers[self.layer_id]
        return history == 0 or (
            history <= self.slots
            and layer.resident_owner == self.session.owner
            and layer.resident_end >= history
        )

    @property
    def page_table(self):
        return self.session.page_table if self._shared else None

    def _check(self):
        if self._shared:
            self.session._check()

    def operation(self):
        self._check()
        return self._pool.operation(self.session, self.layer_id) if self._shared else nullcontext()

    def drain(self):
        if self._shared:
            self._pool.drain()
        elif self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def reserve_append_source(self):
        """Bound retained writeback sources before the next projection allocates."""
        self._check()
        if self._shared and self.transient_start is None:
            with self.operation():
                self._pool._reap_writes(make_room=True)

    def host_records(self, length=None):
        """Copy logical host records for diagnostics outside measured execution."""
        self._check()
        length = self.host_written_end if length is None else length
        if not 0 <= length <= self.host_written_end:
            raise ValueError("diagnostic host range exceeds host-backed written records")
        self.drain()
        if self.host is None:
            return self.records[:length].detach().cpu().clone()
        if not self._shared:
            return self.host[:length].clone()
        ids = self.logical_to_global(torch.arange(length, device=self.device)).cpu()
        return self.host[ids].clone()

    def logical_to_global(self, indices):
        self._check()
        if indices.device != self.device or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be integer tensors on the cache device")
        if not self._shared:
            return indices.long()
        valid = indices >= 0
        safe = indices.clamp_min(0).long()
        limit = self.capacity if self.transient_start is None else self.transient_start
        if safe.numel() and bool((valid & (safe >= limit)).any()):
            raise ValueError("logical token ID exceeds host capacity or belongs to GPU-only suffix")
        result = self.page_table[safe // PAGE_SIZE].long() * PAGE_SIZE + safe % PAGE_SIZE
        return torch.where(valid, result, -1)

    def _global_range(self, start, end):
        """Map a host-validated contiguous logical range without scalar reads."""
        limit = self.capacity if self.transient_start is None else self.transient_start
        if not 0 <= start <= end <= limit:
            raise ValueError("logical range exceeds host capacity")
        logical = torch.arange(start, end, device=self.device, dtype=torch.int64)
        if not self._shared:
            return logical
        return self.page_table[logical // PAGE_SIZE].long() * PAGE_SIZE + logical % PAGE_SIZE

    def begin_step(self, count, *, transient=False):
        if transient:
            return self.begin_transient(count)
        self._check()
        if self._step_end is not None:
            raise RuntimeError("a cache step is already active")
        if self._shared:
            owner = self._pool._transient_owners.get(self.layer_id)
            if owner is not None and owner != self.session.owner:
                raise RuntimeError(
                    "another session owns transient candidate records for this layer"
                )
        if count < 1 or self.length + count > self.capacity:
            raise ValueError("cache step exceeds capacity")
        self._step_end = self.length + count
        self.written = self.indexer_visible_end = self.length

    def begin_transient(self, count):
        """Lease GPU-only suffix rows without extending persistent history.

        All suffix chunks belong to one step and remain available until
        discard_transient() or rollback(). The caller keeps its normal layer
        operation lease through each append, exact recall and MLA consumer.
        """
        self._check()
        if self._step_end is not None:
            raise RuntimeError("a cache step is already active")
        candidate_slots = self._pool.candidate_slots if self._shared else self._candidate_slots
        if type(count) is not int or not 1 <= count <= candidate_slots:
            raise ValueError("transient suffix exceeds candidate slot capacity")
        with self.operation():
            if self._shared:
                if self.layer_id in self._pool._transient_owners:
                    raise RuntimeError("candidate records already have a transient owner")
                self._pool._transient_owners[self.layer_id] = self.session.owner
            self._transient_start = self.length
            self._step_end = self.length + count
            self.written = self.indexer_visible_end = self.length

    def discard_transient(self):
        """Discard a complete or partial suffix after its consumers finish."""
        self._check()
        if self.transient_start is None:
            raise RuntimeError("no transient cache step is active")
        if self._shared and self._pool._transient_owners.get(self.layer_id) != self.session.owner:
            raise RuntimeError("transient suffix ownership was lost")
        self.drain()
        with self.operation():
            if self._shared and self._prefetch is not None:
                self.finalize_prefetch()
            self.written = self.indexer_visible_end = self.length
            self._step_end = self._transient_start = None
            if self._shared:
                del self._pool._transient_owners[self.layer_id]

    def declare_indexer_visible(self, end):
        self._check()
        if self._step_end is None or not self.written <= end <= self._step_end:
            raise ValueError("indexer range must belong to the pending cache step")
        if end < getattr(self, "indexer_visible_end", self.written):
            raise ValueError("indexer visible range cannot move backwards")
        self.indexer_visible_end = end

    def append(self, records):
        self._check()
        if self._step_end is None:
            raise RuntimeError("begin_step must precede append")
        if (
            records.ndim != 2
            or records.shape[1] != self.width
            or records.dtype != self.records.dtype
            or records.device != self.device
            or not records.is_contiguous()
            or not len(records)
            or self.written + len(records) > self._step_end
        ):
            raise ValueError("invalid appended token records")
        if self._shared and self._prefetch is not None:
            raise RuntimeError("finalize fused prefetch before appending main KV")
        if (
            self._shared
            and self.transient_start is None
            and len(records) > self.slots
            and not self._standalone
        ):
            raise WorkingSetTooLarge("new KV query batch exceeds the shared device pool")
        start, stop = self.written, self.written + len(records)
        if self.transient_start is not None:
            with self.operation():
                first = self.slots + 1 + start - self.transient_start if self._shared else start
                self.records[first : first + len(records)].copy_(records)
            self.stats.transient_written_records += len(records)
        elif self._shared:
            with self.operation():
                # Standalone legacy prefix appends retain bounded direct-write
                # batches. Shared model layers enforce Q <= P before launch.
                for offset in range(0, len(records), self.slots):
                    piece = records[offset : offset + self.slots]
                    chosen = self._pool.append_slots(self, start + offset, len(piece))
                    planned = chosen is not None
                    native = self._pool.native_metadata
                    if planned and hasattr(native, "planned_append"):
                        layer = self._pool._clock_event(self.layer_id)
                        native.planned_append(
                            piece,
                            self.records,
                            self.page_table,
                            chosen,
                            self.host_to_device,
                            self.device_to_host,
                            self.age,
                            layer.free,
                            layer.clock_tensor,
                            self._native_totals[3:],
                            start=start + offset,
                            timestamp=layer.clock,
                        )
                        layer.clock += 1
                        self._pool.certify_append(self, start + offset + len(piece))
                        continue
                    global_ids = self._global_range(start + offset, start + offset + len(piece))
                    if planned:
                        self._evict(chosen, preserve_residency=True)
                    else:
                        chosen = self._available_slots(global_ids, len(piece))
                    self.records.index_copy_(0, chosen, piece)
                    self.host_to_device[global_ids] = chosen.int()
                    self.device_to_host[chosen] = global_ids
                    self._pool.layers[self.layer_id].free[chosen] = False
                    self._pool.stamp(self.layer_id, chosen, preserve_append_plan=planned)
                    if planned:
                        self._pool.certify_append(self, start + offset + len(piece))
                self._pool.write_host(self.session, self.layer_id, start, records)
        elif self.offload:
            # Existing dense-prefetch subclass owns a separate full host buffer.
            self.host[start:stop].copy_(records, non_blocking=self.device.type == "cuda")
        else:
            self.records[start:stop].copy_(records)
            ids = torch.arange(start, stop, device=self.device)
            self.host_to_device[start:stop] = ids.int()
            self.device_to_host[start:stop] = ids
        self.written = stop
        self.indexer_visible_end = max(getattr(self, "indexer_visible_end", 0), stop)
        self.stats.written_records += len(records)
        return start

    def commit(self):
        self._check()
        if self.transient_start is not None:
            raise RuntimeError(
                "transient candidate KV cannot be committed; discard_transient instead"
            )
        if self._step_end is None or self.written != self._step_end:
            raise RuntimeError("cannot commit an incomplete cache step")
        self.drain()
        self.length = self.written
        self._step_end = None

    def rollback(self):
        self._check()
        if self.transient_start is not None:
            self.discard_transient()
            return
        if self._step_end is None:
            raise RuntimeError("no active cache step")
        self.drain()
        if self._shared and self._prefetch is not None:
            self.finalize_prefetch()
        self._step_end = None
        self.truncate(self.length)

    def truncate(self, length):
        self._check()
        if self._step_end is not None:
            raise RuntimeError("rollback active step before truncation")
        if not 0 <= length <= self.length:
            raise ValueError("truncate length must not extend committed history")
        if self._shared:
            self.drain()
            with self.operation():
                suffix = torch.arange(length, self.capacity, device=self.device, dtype=torch.int64)
                global_ids = self.logical_to_global(suffix)
                self._pool.release_ids(self.layer_id, global_ids)
            self._pool.invalidate_snapshots(self.session, length)
            self._prefetch = None
        else:
            invalid = self.device_to_host >= length
            self.device_to_host[invalid] = MISSING
            self.age[invalid] = 0
            self.host_to_device[length:] = MISSING
        self.length = self.written = self.indexer_visible_end = length

    def reset(self):
        self._check()
        if self._step_end is not None:
            self.rollback()
        self.truncate(0)
        self.reset_stats()

    def reset_stats(self):
        self.stats = CacheStats()
        self.prefetch_counts = []
        if self._shared:
            self._counter_totals.zero_()

    def _evict(self, slots, *, count_eviction=True, preserve_residency=False):
        if self._shared and not preserve_residency:
            self._pool.invalidate_residency(self.layer_id)
        old = self.device_to_host[slots]
        valid = old != MISSING
        self.host_to_device[old[valid]] = MISSING
        self.device_to_host[slots] = MISSING
        self.age[slots] = -1 if self._shared else 0
        if self._shared:
            self._pool.layers[self.layer_id].free[slots] = True
        if count_eviction:
            self.stats.evicted_records += old[valid].numel()

    def _available_slots(self, protected, count):
        # Empty slots first: partial-free recall evicts max(0, misses-free).
        if count == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        keep = torch.zeros(len(self.device_to_host), dtype=torch.bool, device=self.device)
        if self._shared:
            keep[0] = True
        mapped = self.host_to_device[protected].long()
        keep[mapped[mapped != MISSING]] = True
        available = torch.where(~keep)[0]
        if available.numel() < count:
            raise WorkingSetTooLarge("protected records exceed the device working set")
        rank = torch.where(self.device_to_host[available] == MISSING, -1, self.age[available])
        chosen = available[torch.argsort(rank, stable=True)[:count]]
        if chosen.numel() != count:
            raise RuntimeError("allocator returned fewer slots than requested")
        self._evict(chosen)
        return chosen

    def ensure(self, indices):
        """Recall the exact logical union, preserving negative padding IDs."""
        self._check()
        if indices.device != self.device or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be integer tensors on the cache device")
        if self._shared and self._prefetch is not None:
            raise RuntimeError("finalize fused prefetch before exact recall")
        unique = torch.unique(indices.long())
        unique = unique[unique >= 0]
        if unique.numel() and int(unique[-1]) >= self.written:
            raise ValueError("selection references unwritten cache records")
        history = unique if self.transient_start is None else unique[unique < self.transient_start]
        if history.numel() > self.slots:
            raise WorkingSetTooLarge(
                f"{history.numel()} selected history records exceed {self.slots} slots"
            )
        self.stats.max_working_set = max(self.stats.max_working_set, unique.numel())
        with self.operation():
            global_ids = self.logical_to_global(history)
            mapped = self.host_to_device[global_ids].long()
            if self._shared and self._pool.native_metadata is not None:
                scratch = self._pool.miss_scratch[: global_ids.numel()]
                self._pool.native_metadata.mark_misses(global_ids, self.host_to_device, scratch)
                missing = scratch[scratch != MISSING]
            else:
                missing = global_ids[mapped == MISSING]
            self.stats.selection_records += unique.numel()
            self.stats.resident_selection_records += unique.numel() - missing.numel()
            if self._shared:
                # Official protection and allocation events advance even on
                # all-hit and partial-free calls. Map-only queries never stamp.
                self._pool.protect(
                    self.layer_id, global_ids, preserve_append_plan=not missing.numel()
                )
            if self.offload and missing.numel():
                if self._shared:
                    self._pool.wait_host(self.session, self.layer_id)
                chosen = self._available_slots(global_ids, missing.numel())
                if self.device.type == "cpu":
                    self.records[chosen] = self.host[missing]
                else:
                    from operators.common.kv_transfer import gather_host_records

                    gather_host_records(self.host, self.records, missing, chosen)
                self.host_to_device[missing] = chosen.int()
                self.device_to_host[chosen] = missing
                if self._shared:
                    self._pool.layers[self.layer_id].free[chosen] = False
                self.stats.recalled_records += missing.numel()
            else:
                chosen = torch.empty(0, dtype=torch.int64, device=self.device)
            if self._shared:
                self._pool.stamp(self.layer_id, chosen, preserve_append_plan=not missing.numel())
            if self.transient_start is None:
                physical = self.host_to_device[self.logical_to_global(indices).clamp_min(0)]
            else:
                is_history = (indices >= 0) & (indices < self.transient_start)
                host_indices = torch.where(is_history, indices, -1)
                physical = self.host_to_device[
                    self.logical_to_global(host_indices).clamp_min(0)
                ].long()
                suffix_slots = (
                    self.slots + 1 + indices.long() - self.transient_start
                    if self._shared
                    else indices.long()
                )
                physical = torch.where(is_history, physical, suffix_slots)
            return torch.where(indices >= 0, physical, -1).int()

    def _ensure_from_topk(self, indices):
        """Private exact-top-k consumer; arbitrary public IDs still use ensure."""
        native = self._pool.native_metadata if self._shared else None
        if (
            type(self) is not SparseTokenCache
            or not hasattr(native, "resident_selection")
            or indices.ndim != 2
            or indices.dtype != torch.int32
            or indices.device != self.device
            or not indices.is_contiguous()
        ):
            return self.ensure(indices)
        with self.operation():
            if self._prefetch is not None:
                raise RuntimeError("finalize fused prefetch before exact recall")
            if not self.all_history_resident:
                if (
                    hasattr(native, "sparse_selection_classify")
                    and self.host_written_end <= self.slots
                ):
                    return self._ensure_sparse_from_topk(indices, native)
                return self.ensure(indices)
            layer = self._pool._clock_event(self.layer_id)
            if layer.clock == PRIORITY_LIMIT - 1:
                return self.ensure(indices)
            physical = torch.empty_like(indices)
            native.resident_selection(
                indices,
                physical,
                self.page_table,
                self.host_to_device,
                self.age,
                layer.clock_tensor,
                self._pool.union_bitmap,
                self._pool.union_count,
                self._native_totals[:3],
                written=self.written,
                history=self.host_written_end,
                transient=self.transient_start is not None,
                candidate_slots=self._pool.candidate_slots,
                timestamp=layer.clock,
            )
            layer.clock += 2
            return physical

    def _ensure_sparse_from_topk(self, indices, native):
        """Recall a bounded private union while its pool operation lease is held."""
        pool = self._pool
        layer = pool._clock_event(self.layer_id)
        if layer.clock == PRIORITY_LIMIT - 1:
            return self.ensure(indices)
        timestamp = layer.clock
        history = self.host_written_end
        physical = torch.empty_like(indices)
        native.sparse_selection_classify(
            indices,
            self.page_table,
            self.host_to_device,
            self.device_to_host,
            self.age,
            layer.clock_tensor,
            pool.union_bitmap,
            pool.union_count,
            pool.counter,
            pool.free_slots,
            self._native_totals[:3],
            written=self.written,
            history=history,
            candidate_slots=pool.candidate_slots,
            timestamp=timestamp,
        )
        # Each event commits before subsequent fallible work, as in ensure().
        layer.clock += 1
        missing_count = int(pool.counter.item())
        misses = pool.allocation_log[:missing_count]
        chosen = pool.miss_scratch[:0]
        if missing_count:
            layer.append_owner = None
            pool.wait_host(self.session, self.layer_id)
            chosen = torch.argsort(self.age[1:], stable=True)[:missing_count]
            torch.cumsum(pool.free_slots, dim=0, dtype=torch.int64, out=pool.miss_scratch)
            pool.invalidate_residency(self.layer_id)
            # Compaction follows logical order, not host page/global-ID order.
            # It also tombstones victims before the generic copy overwrites them.
            native.sparse_selection_compact(
                pool.miss_scratch,
                self.page_table,
                self.host_to_device,
                self.device_to_host,
                self.age,
                layer.free,
                self._native_totals[3:],
                misses,
                chosen,
                history=history,
                timestamp=timestamp,
            )
            from operators.common.kv_transfer import gather_host_records

            gather_host_records(self.host, self.records, misses, chosen)
        native.sparse_selection_publish(
            misses,
            chosen,
            self.host_to_device,
            self.device_to_host,
            self.age,
            layer.free,
            layer.clock_tensor,
            timestamp=timestamp,
        )
        layer.clock += 1
        self.stats.recalled_records += missing_count
        native.sparse_selection_map(
            indices,
            physical,
            self.page_table,
            self.host_to_device,
            self.age,
            layer.clock_tensor,
            history=history,
            written=self.written,
            candidate_slots=pool.candidate_slots,
        )
        return physical

    def prepare_prefetch(self, new_start, new_count, offset, *, limit=8192):
        """Sort all usable slots without eviction, before current main-KV write."""
        self._check()
        if not self._shared:
            return None
        if self._prefetch is not None:
            raise RuntimeError("a fused prefetch is already pending")
        if (
            self._step_end is None
            or new_start != self.written
            or not (0 < new_count and new_start + new_count <= self.indexer_visible_end)
        ):
            raise ValueError("prefetch requires declared indexer suffix before main-KV append")
        if new_count > self.slots and self.transient_start is None:
            raise WorkingSetTooLarge("query batch exceeds shared pool capacity (Q > P)")
        if limit < 0:
            raise ValueError("prefetch limit must be nonnegative")
        with self.operation():
            if self.all_history_resident:
                # A no-miss indexer needs no prefetch workspace. Retain the
                # allocation event that finalize would otherwise contribute.
                self._pool.stamp(
                    self.layer_id,
                    self._pool.layers[self.layer_id].append_order[:0],
                    preserve_append_plan=True,
                )
                return None
            self._pool.invalidate_residency(self.layer_id)
            self._pool.wait_host(self.session, self.layer_id)
            self._pool.free_slots.copy_((torch.argsort(self.age[1:], stable=True) + 1).int())
            self._pool.counter.zero_()
            self._pool.allocation_log.fill_(MISSING)
            self._pool.prefetch_stats.zero_()
            self._pool._pending_prefetch = (self.session.owner, self.layer_id)
            self._prefetch = {
                "host": self.host,
                "device": self.records[: self.slots + 1],
                "host_to_device": self.host_to_device,
                "device_to_host": self.device_to_host,
                "free_slots": self._pool.free_slots,
                "counter": self._pool.counter,
                "allocation_log": self._pool.allocation_log,
                "prefetch_stats": self._pool.prefetch_stats,
                "page_table": self.page_table,
                "offset": offset,
                "history_length": self.host_written_end,
                "transient_suffix": self.transient_start is not None,
                "max_prefetch": min(
                    limit, 8192, self.slots - (new_count if self.transient_start is None else 0)
                ),
            }
            return self._prefetch

    def finalize_prefetch(self, prefetch=None):
        self._check()
        if not self._shared or self._prefetch is None:
            raise RuntimeError("no fused prefetch is pending")
        if prefetch is not None and prefetch is not self._prefetch:
            raise ValueError("prefetch lease belongs to another operation")
        with self.operation():
            self._pool.finalize_prefetch(self.layer_id)
            self._prefetch_totals.add_(self._pool.prefetch_stats)
            self._prefetch = None
            self._pool._pending_prefetch = None

    def prefetch_reference(self, logical_ids):
        """Executable candidate-stream contract for CPU/GPU transport tests.

        This is not the fused indexer. Only live-map misses claim sorted victims.
        """
        if self._prefetch is None:
            raise RuntimeError("prepare_prefetch must precede reference candidates")
        state = self._prefetch
        with self.operation():
            rank = 0
            for logical in logical_ids:
                logical = int(logical)
                if not 0 <= logical < state["history_length"]:
                    continue
                global_id = (
                    self.session._pages[logical // PAGE_SIZE] * PAGE_SIZE + logical % PAGE_SIZE
                )
                if int(self.host_to_device[global_id]) != MISSING:
                    continue
                self._pool.counter.fill_(int(self._pool.counter.item()) + 1)
                if rank >= state["max_prefetch"]:
                    self._pool.prefetch_stats[2] += 1
                    continue
                slot = int(state["free_slots"][rank])
                old = int(self.device_to_host[slot])
                if old != MISSING:
                    self.host_to_device[old] = MISSING
                    self._pool.prefetch_stats[1] += 1
                if self.device.type == "cpu":
                    self.records[slot].copy_(self.host[global_id])
                else:
                    from operators.common.kv_transfer import gather_host_records

                    gather_host_records(
                        self.host,
                        self.records,
                        torch.tensor([global_id], device=self.device, dtype=torch.int64),
                        torch.tensor([slot], device=self.device, dtype=torch.int64),
                    )
                self.host_to_device[global_id] = slot
                self.device_to_host[slot] = global_id
                self._pool.allocation_log[slot] = global_id
                self._pool.prefetch_stats[0] += 1
                rank += 1

    def _snapshot_state(self):
        return {
            "length": self.length,
            "written": self.written,
            "indexer_visible_end": self.indexer_visible_end,
        }

    def _restore_state(self, state):
        self.length, self.written = state["length"], state["written"]
        self.indexer_visible_end = state["indexer_visible_end"]
        self._step_end = None
        self._transient_start = None
        self._prefetch = None
        self.reset_stats()

    def metrics(self):
        """Read bounded counters after measurement synchronization."""
        self._check()
        totals = self._counter_totals.tolist() if self._shared else [0] * 7
        prefetched, fused_evictions, failed_claims, selected, resident, maximum, evicted = totals
        return {
            **vars(self.stats),
            "host_written_records": self.stats.written_records
            - self.stats.transient_written_records
            if self.offload
            else 0,
            "selection_records": self.stats.selection_records + selected,
            "resident_selection_records": self.stats.resident_selection_records + resident,
            "max_working_set": max(self.stats.max_working_set, maximum),
            "evicted_records": self.stats.evicted_records + fused_evictions + evicted,
            "prefetched_records": prefetched,
            "prefetch_capacity_failures": failed_claims,
            "host_to_device_bytes": (prefetched + self.stats.recalled_records) * self.record_bytes,
            "device_to_host_bytes": (
                self.stats.written_records - self.stats.transient_written_records
            )
            * self.record_bytes
            if self.offload
            else 0,
            "device_record_bytes": self.records.numel() * self.records.element_size(),
            "host_record_bytes": self.host.numel() * self.host.element_size()
            if self.host is not None
            else 0,
            "record_bytes": self.record_bytes,
            "device_slots": self.slots,
            "pool_scope": "shared_per_layer" if self._shared else "session",
            "host_token_capacity": self._pool.host_capacity if self._shared else self.capacity,
            "session_host_tokens": self.session.host_tokens if self._shared else 0,
            "padding_slots": 1 if self._shared else 0,
            "candidate_slots": self._pool.candidate_slots
            if self._shared
            else self._candidate_slots,
        }
