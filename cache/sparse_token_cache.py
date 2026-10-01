"""Single-layer fixed-width token cache with resident or pinned-host backing.

The model supplies the record layout. This cache owns logical/physical maps,
bounded device slots, exact recalls, and step commit/rollback. It is independent
of the resident multi-layer CacheManager used by NOSA.
"""

from dataclasses import dataclass

import torch

MISSING = torch.iinfo(torch.int32).max


class WorkingSetTooLarge(ValueError):
    """The caller must split query consumption without changing its selection."""


@dataclass
class CacheStats:
    written_records: int = 0
    recalled_records: int = 0
    evicted_records: int = 0
    max_working_set: int = 0


class SparseTokenCache:
    def __init__(self, capacity, width, *, device="cuda", dtype=torch.bfloat16, slots=None):
        if capacity < 1 or width < 1 or (slots is not None and not 1 <= slots <= capacity):
            raise ValueError("capacity, width and bounded slot count must be positive")
        self.capacity, self.width = capacity, width
        self.offload = slots is not None
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("SparseTokenCache requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.slots = capacity if slots is None else slots
        self.records = torch.empty((self.slots, width), device=self.device, dtype=dtype)
        self.host = (
            torch.empty((capacity, width), dtype=dtype, pin_memory=True) if self.offload else None
        )
        self.host_to_device = torch.full(
            (capacity,), MISSING, dtype=torch.int32, device=self.device
        )
        self.device_to_host = torch.full(
            (self.slots,), MISSING, dtype=torch.int64, device=self.device
        )
        self.age = torch.zeros(self.slots, dtype=torch.int64, device=self.device)
        self.length = self.written = 0
        self._step_end = None
        self._clock = 0
        self.stats = CacheStats()
        self.prefetch_counts = []

    @property
    def record_bytes(self):
        return self.width * self.records.element_size()

    def begin_step(self, count):
        if self._step_end is not None:
            raise RuntimeError("a cache step is already active")
        if count < 1 or self.length + count > self.capacity:
            raise ValueError("cache step exceeds capacity")
        self._step_end = self.length + count
        self.written = self.length

    def append(self, records):
        if self._step_end is None:
            raise RuntimeError("begin_step must precede append")
        if (
            records.ndim != 2
            or records.shape[1] != self.width
            or records.dtype != self.records.dtype
            or records.device != self.device
            or self.written + len(records) > self._step_end
        ):
            raise ValueError("invalid appended token records")
        start, stop = self.written, self.written + len(records)
        if self.offload:
            # Ordered on the current stream before kernels dereference host memory.
            self.host[start:stop].copy_(records, non_blocking=True)
        else:
            self.records[start:stop].copy_(records)
            ids = torch.arange(start, stop, device=self.device)
            self.host_to_device[start:stop] = ids.int()
            self.device_to_host[start:stop] = ids
        self.written = stop
        self.stats.written_records += len(records)
        return start

    def commit(self):
        if self._step_end is None or self.written != self._step_end:
            raise RuntimeError("cannot commit an incomplete cache step")
        self.length = self.written
        self._step_end = None

    def rollback(self):
        if self._step_end is None:
            raise RuntimeError("no active cache step")
        self._step_end = None
        self.truncate(self.length)

    def truncate(self, length):
        if self._step_end is not None:
            raise RuntimeError("rollback active step before truncation")
        if not 0 <= length <= self.length:
            raise ValueError("truncate length must not extend committed history")
        invalid = self.device_to_host >= length
        self.device_to_host[invalid] = MISSING
        self.age[invalid] = 0
        self.host_to_device[length:] = MISSING
        self.length = self.written = length

    def reset(self):
        self._step_end = None
        self.length = self.written = 0
        self.host_to_device.fill_(MISSING)
        self.device_to_host.fill_(MISSING)
        self.age.zero_()
        self.reset_stats()

    def reset_stats(self):
        self.stats = CacheStats()
        self.prefetch_counts = []

    def _evict(self, slots):
        old = self.device_to_host[slots]
        valid = old != MISSING
        self.host_to_device[old[valid]] = MISSING
        self.device_to_host[slots] = MISSING
        self.age[slots] = 0
        self.stats.evicted_records += old[valid].numel()

    def _available_slots(self, protected, count):
        keep = torch.zeros(self.slots, dtype=torch.bool, device=self.device)
        mapped = self.host_to_device[protected].long()
        keep[mapped[mapped != MISSING]] = True
        available = torch.where(~keep)[0]
        if available.numel() < count:
            raise WorkingSetTooLarge("protected records exceed the device working set")
        # Empty slots first, then least recently used records. Never evict a
        # record consumed by the same attention launch.
        rank = torch.where(self.device_to_host[available] == MISSING, -1, self.age[available])
        chosen = available[torch.argsort(rank, stable=True)[:count]]
        self._evict(chosen)
        return chosen

    def ensure(self, indices):
        """Recall exact selected records and return physical IDs, preserving -1."""
        if indices.device != self.device or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be integer tensors on the cache device")
        unique = torch.unique(indices.long())
        unique = unique[unique >= 0]
        if unique.numel() and int(unique[-1]) >= self.written:
            raise ValueError("selection references unwritten cache records")
        if unique.numel() > self.slots:
            raise WorkingSetTooLarge(f"{unique.numel()} selected records exceed {self.slots} slots")
        self.stats.max_working_set = max(self.stats.max_working_set, unique.numel())
        if self.offload:
            missing = unique[self.host_to_device[unique] == MISSING]
            if missing.numel():
                from operators.sm90.kv_transfer import gather_host_records

                chosen = self._available_slots(unique, missing.numel())
                gather_host_records(self.host, self.records, missing, chosen)
                self.host_to_device[missing] = chosen.int()
                self.device_to_host[chosen] = missing
                self.stats.recalled_records += missing.numel()
        self._clock += 1
        self.age[self.host_to_device[unique].long()] = self._clock
        remap = self.host_to_device[indices.clamp_min(0).long()]
        return torch.where(indices >= 0, remap, -1).int()

    def prepare_prefetch(self, new_start, new_count, offset, *, limit=8192):
        """Offer empty slots to the fused indexer; current chunk stays protected.

        Eviction is completed before concurrent prefetch claims begin. This
        avoids races between invalidating an evicted record and re-prefetching
        that same logical record from another warp.
        """
        if not self.offload:
            return None
        if not 0 <= new_start < new_start + new_count <= self.written:
            raise ValueError("prefetch chunk is outside written records")
        current = torch.arange(new_start, new_start + new_count, device=self.device)
        self.ensure(current)
        count = min(limit, self.slots - new_count)
        slots = self._available_slots(current, count)
        counter = torch.zeros(1, dtype=torch.uint32, device=self.device)
        self.prefetch_counts.append((counter, count))
        return {
            "host": self.host,
            "device": self.records,
            "host_to_device": self.host_to_device,
            "device_to_host": self.device_to_host,
            "free_slots": slots.int(),
            "counter": counter,
            "offset": offset,
            "max_prefetch": count,
        }

    def metrics(self):
        """Read counters after measurement synchronization, outside timed scope."""
        prefetched = sum(min(int(counter.item()), cap) for counter, cap in self.prefetch_counts)
        return {
            **vars(self.stats),
            "prefetched_records": prefetched,
            "host_to_device_bytes": (prefetched + self.stats.recalled_records) * self.record_bytes,
            "device_to_host_bytes": self.stats.written_records * self.record_bytes
            if self.offload
            else 0,
            "device_record_bytes": self.records.numel() * self.records.element_size(),
            "host_record_bytes": self.capacity * self.record_bytes if self.offload else 0,
            "record_bytes": self.record_bytes,
            "device_slots": self.slots,
        }
