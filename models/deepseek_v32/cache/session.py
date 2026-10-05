"""DeepSeek private session state and borrowed dense cache views."""

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from cache.host_allocation import allocate_host_tensor, storage_allocation_bytes
from cache.sparse_token_cache import MISSING, CacheStats, SparseTokenCache


class ServingCacheGuard:
    """Authorize model-borrowed mutation without changing standalone cache behavior."""

    def bind_serving(self, session, lifecycle):
        self._serving_session = session
        self._serving_lifecycle = lifecycle

    def _check_serving_access(self, *, execution=False):
        session = getattr(self, "_serving_session", None)
        if session is None:
            return
        lifecycle = self._serving_lifecycle
        lifecycle.check_session(session, session.registration, released=session.released)
        if execution:
            lifecycle.check_execution(session)
        elif lifecycle.active_session is not session:
            raise RuntimeError(
                "borrowed DeepSeek cache mutation requires its active session operation"
            )
        else:
            lifecycle.check_mutation(session)

    def operation(self):
        self._check_serving_access()
        return super().operation()

    def begin_step(self, count):
        self._check_serving_access(execution=True)
        return super().begin_step(count)

    def begin_transient(self, count):
        self._check_serving_access(execution=True)
        return super().begin_transient(count)

    def append(self, records):
        self._check_serving_access(execution=True)
        return super().append(records)

    def declare_indexer_visible(self, end):
        self._check_serving_access(execution=True)
        return super().declare_indexer_visible(end)

    def reserve_append_source(self):
        self._check_serving_access(execution=True)
        return super().reserve_append_source()

    def prepare_prefetch(self, *args, **kwargs):
        self._check_serving_access(execution=True)
        return super().prepare_prefetch(*args, **kwargs)

    def finalize_prefetch(self, *args, **kwargs):
        self._check_serving_access()
        return super().finalize_prefetch(*args, **kwargs)

    def commit(self):
        self._check_serving_access(execution=True)
        return super().commit()

    def rollback(self):
        self._check_serving_access()
        return super().rollback()

    def discard_transient(self):
        self._check_serving_access()
        return super().discard_transient()

    def truncate(self, length):
        self._check_serving_access()
        return super().truncate(length)

    def reset(self):
        self._check_serving_access()
        return super().reset()


class ServingSparseTokenCache(ServingCacheGuard, SparseTokenCache):
    native_metadata_compatible = True

    @classmethod
    def for_layer(cls, session, layer):
        session._check()
        if not 0 <= layer < session.pool.num_layers:
            raise ValueError("layer index outside shared pool")
        existing = session._layers.get(layer)
        if existing is not None:
            if not isinstance(existing, cls):
                raise ValueError("a different cache factory already owns this layer")
            return existing
        cache = object.__new__(cls)
        cache._bind_shared(session.pool, session, layer)
        session._layers[layer] = cache
        return cache


@dataclass
class DenseWrite:
    source: torch.Tensor
    event: object = None


class DenseCache(ServingCacheGuard):
    """Pinned full-layer backing and a borrowed double-buffered HBM stage."""

    def operation(self):
        self._check_serving_access(execution=True)
        return nullcontext()

    def __init__(self, capacity, width, *, device, backend):
        self.capacity, self.width = capacity, width
        self.offload = True
        self.device = torch.device(device)
        self.slots = capacity
        self.backend = backend
        self.session = None
        self.records = None
        self.host = allocate_host_tensor(
            (capacity, width), dtype=torch.bfloat16, pin_memory=self.device.type == "cuda"
        )
        self.host_to_device = torch.full((capacity,), MISSING, dtype=torch.int32, device=device)
        self.device_to_host = torch.full((capacity,), MISSING, dtype=torch.int64, device=device)
        self.age = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.length = self.written = 0
        self._step_end = None
        self._clock = 0
        self.stats = CacheStats()
        self.prefetch_counts = []
        self.dense_fetched_records = 0

    @property
    def record_bytes(self):
        return self.width * self.host.element_size()

    def _check_execution(self):
        self.backend._check_dense_execution(self.session)

    def reserve_append_source(self):
        self._check_execution()
        self.backend._reserve_dense_source()

    def declare_indexer_visible(self, end):
        if self._step_end is None or not self.written <= end <= self._step_end:
            raise ValueError("indexer end must lie within the active step")

    def begin_step(self, count):
        self._check_execution()
        if self._step_end is not None:
            raise RuntimeError("a cache step is already active")
        if count < 1 or self.length + count > self.capacity:
            raise ValueError("cache step exceeds capacity")
        self._step_end = self.length + count

    def append(self, records):
        self._check_execution()
        if self.records is None:
            raise RuntimeError("dense append requires this layer's ready staging view")
        if self._step_end is None or self.written + len(records) > self._step_end:
            raise ValueError("append exceeds the active cache step")
        start, stop = self.written, self.written + len(records)
        self.records[start:stop].copy_(records)
        # Retain before enqueue: even an event-record failure must keep a source
        # alive until the enclosing execution lease confirms both streams drain.
        ticket = DenseWrite(records)
        self.backend._dense_sources.append(ticket)
        self.host[start:stop].copy_(records, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            ticket.event = torch.cuda.Event()
            ticket.event.record(torch.cuda.current_stream(self.device))
        self.written = stop
        self.stats.written_records += len(records)
        return start

    def commit(self):
        self._check_execution()
        if self._step_end is None or self.written != self._step_end:
            raise RuntimeError("cannot commit an incomplete cache step")
        self.length = self.written
        self._step_end = None

    def rollback(self):
        self._check_execution()
        self._step_end = None
        self.truncate(self.length)

    def truncate(self, length):
        self._check_serving_access()
        if self._step_end is not None or not 0 <= length <= self.length:
            raise ValueError("truncate requires an inactive committed prefix")
        self.length = self.written = length

    def reset_stats(self):
        self.stats = CacheStats()
        self.dense_fetched_records = 0

    def metrics(self):
        return {
            **vars(self.stats),
            "prefetched_records": 0,
            "dense_fetched_records": self.dense_fetched_records,
            "host_to_device_bytes": self.dense_fetched_records * self.record_bytes,
            "device_to_host_bytes": self.stats.written_records * self.record_bytes,
            "device_slots": self.slots,
            "device_record_bytes": 0,  # Staging belongs to the backend ledger.
            "host_record_bytes": self.host.numel() * self.host.element_size(),
            "host_allocation_bytes": storage_allocation_bytes(self.host),
            "record_bytes": self.record_bytes,
        }


@dataclass(eq=False)
class DeepSeekServingSession:
    capacity: int
    scheme: str
    runners: list
    owner: object = field(repr=False)
    registration: object = field(default=None, repr=False)
    storage_plan: object = field(default=None, repr=False)
    length: int = 0
    stages: list = field(default_factory=list)
    copy_stream: object = None
    stage_consumed: list = field(default_factory=list)
    released: bool = False
    sparse_session: object = None
    prefix_offsets: list = field(default_factory=list)
    prefix_length: int = 0
    dense_allocation_ready: object = None
    last_candidate_transient: bool = False

    def check_backing_release(self):
        lifecycle = self.owner.lifecycle
        lifecycle.check_session(self, self.registration, released=self.released)
        if lifecycle.active_session is not self or lifecycle.active_kind != "release":
            raise RuntimeError("borrowed DeepSeek session release requires its release lease")
        lifecycle.check_mutation(self, releasing=True)
