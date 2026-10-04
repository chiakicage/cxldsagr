"""Model-independent double buffering for serial, full-record prefetch.

The caller chooses the records and their logical extent. This module owns only
two fixed-capacity slots, their CUDA dependencies, and one serial execution
lease. Record views are borrowed and must not escape that lease. CPU operation
is a synchronous reference; its byte count describes the device-storage role,
not an observation of HBM allocation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from threading import Lock

import torch


def _validate_layout(capacity, record_shapes, dtype):
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("staging capacity must be a positive integer")
    if not isinstance(record_shapes, Mapping) or not record_shapes:
        raise ValueError("record_shapes must be a nonempty named mapping")
    shapes = {}
    for name, shape in record_shapes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("record names must be nonempty strings")
        if not isinstance(shape, (tuple, list)) or any(
            type(size) is not int or size <= 0 for size in shape
        ):
            raise ValueError("record shapes must contain positive integer dimensions")
        shapes[name] = tuple(shape)
    if not isinstance(dtype, torch.dtype):
        raise TypeError("staging dtype must be a torch.dtype")
    return shapes


@dataclass(frozen=True, eq=False)
class StagingIdentity:
    """Identity of one slot; session ownership uses object identity."""

    session: object
    generation: int
    layer: int
    cycle: int


class DoubleBufferStaging:
    """Two reusable record slots with a single caller and private copy stream.

    ``lease`` captures the current CUDA stream. All that lease's staging access
    and consumers must run on this same stream; a later lease may use another
    stream. Before a slot is rewritten, the copy stream waits for all work
    already queued by the caller, including the previous slot consumer. Waiting
    for copy-ready makes history visible before the caller writes a suffix.

    Successful lease return joins both streams, including speculative copies
    for a layer whose consumer was never reached. An unconfirmed join poisons
    the storage and preserves its ownership until a successful close drains it.
    """

    stage_count = 2

    def __init__(self, capacity, record_shapes, *, dtype, device):
        self.record_shapes = _validate_layout(capacity, record_shapes, dtype)
        self.capacity = capacity
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("staging supports CPU reference or CUDA")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._buffers = {
            name: torch.empty((2, capacity, *shape), dtype=dtype, device=self.device)
            for name, shape in self.record_shapes.items()
        }
        self._copy_stream = None
        self._allocation_stream = None
        self._ready_events = [None, None]
        self._consumer_events = [None, None]
        if self.device.type == "cuda":
            self._allocation_stream = torch.cuda.current_stream(self.device)
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._ready_events = [torch.cuda.Event(), torch.cuda.Event()]
            self._consumer_events = [torch.cuda.Event(), torch.cuda.Event()]
        self._slot_identities = [None, None]
        self._slot_ready = [False, False]
        self._active = None
        self._lock = Lock()
        self.closed = False
        self.poisoned = False
        self.poison_reason = None

    @staticmethod
    def estimate_bytes(capacity, record_shapes, *, dtype):
        """Return exact tensor bytes without allocating a tensor or touching CUDA."""
        shapes = _validate_layout(capacity, record_shapes, dtype)
        return 2 * capacity * sum(prod(shape) for shape in shapes.values()) * dtype.itemsize

    def storage_tensors(self):
        """Enumerate owned storage for an independent allocation audit."""
        return tuple(self._buffers.values())

    def shared_bytes(self):
        """Storage role ledger; CPU reference values are not measured HBM use."""
        seen, total = set(), 0
        for tensor in self.storage_tensors():
            storage = tensor.untyped_storage()
            identity = (tensor.device, storage.data_ptr())
            if identity not in seen:
                seen.add(identity)
                total += storage.nbytes()
        return {"hbm": total, "dram": 0}

    @property
    def active_lease(self):
        return self._active

    def _check_available(self):
        if self.closed:
            raise RuntimeError("staging is closed")
        if self.poisoned:
            raise RuntimeError(f"staging is poisoned: {self.poison_reason}")

    def lease(self, session, *, generation=0):
        """Acquire immediately; use as a context manager or explicitly close it."""
        if session is None:
            raise ValueError("staging requires a session identity")
        if type(generation) is not int or generation < 0:
            raise ValueError("staging generation must be a nonnegative integer")
        with self._lock:
            self._check_available()
            if self._active is not None:
                raise RuntimeError("staging already has an active execution lease")
            if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("staging does not support CUDA Graph capture")
            lease = StagingLease(self, session, generation)
            self._active = lease
            # The allocation stream can have a cached allocator's previous use
            # of this storage. Fence it once before writing on a private stream.
            if self._allocation_stream is not None:
                try:
                    self._copy_stream.wait_stream(self._allocation_stream)
                    self._allocation_stream = None
                except BaseException as exc:
                    self._poison(exc)
                    lease.closed = True
                    raise
            return lease

    def _poison(self, error):
        self.poisoned = True
        self.poison_reason = f"{type(error).__name__}: {error}"

    def _join(self, lease):
        errors = []
        # Attempt both joins even when one fails: a speculative next-layer copy
        # may have no dependency recorded on the caller stream yet.
        for stream in (lease.caller_stream, self._copy_stream):
            if stream is not None:
                try:
                    stream.synchronize()
                except BaseException as exc:  # noqa: BLE001 -- still join the other stream.
                    errors.append(exc)
        if errors:
            self._poison(errors[0])
            raise RuntimeError("unable to confirm completion of staging streams") from errors[0]
        lease._sources.clear()

    def _clear_slots(self):
        self._slot_identities[:] = [None, None]
        self._slot_ready[:] = [False, False]

    def close(self):
        """Drain and drop storage; a live execution lease must be returned first."""
        with self._lock:
            if self.closed:
                return
            if self._active is not None:
                if not self._active.closed:
                    raise RuntimeError("cannot close staging with a live execution lease")
                # A failed lease return keeps its streams and aliases alive.
                # A later successful join permits disposal, never another lease.
                self._join(self._active)
                self._active = None
            self._clear_slots()
            self._buffers.clear()
            self._copy_stream = self._allocation_stream = None
            self._ready_events[:] = [None, None]
            self._consumer_events[:] = [None, None]
            self.closed = True


class StagingLease:
    """Borrowed access to double staging during one session execution."""

    def __init__(self, staging, session, generation):
        self._staging = staging
        self.session = session
        self.generation = generation
        self.caller_stream = (
            torch.cuda.current_stream(staging.device) if staging.device.type == "cuda" else None
        )
        self.closed = False
        self._cycle = 0
        self._sources = []
        self._interrupted = False
        self.submitted_copy_bytes = 0

    def _check(self, *, check_stream=True, allow_interrupted=False):
        if self.closed or self._staging._active is not self:
            raise RuntimeError("staging lease is closed or stale")
        self._staging._check_available()
        if self._interrupted and not allow_interrupted:
            raise RuntimeError("staging submission failed; reset or close the lease before reuse")
        if (
            check_stream
            and self.caller_stream is not None
            and torch.cuda.current_stream(self._staging.device) != self.caller_stream
        ):
            raise RuntimeError("staging lease must use its captured caller stream")

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as join_error:
            if exc is not None:
                raise join_error from exc
            raise
        return False

    def _slot(self, layer_idx, *, require_owner):
        self._check()
        if type(layer_idx) is not int or layer_idx < 0:
            raise ValueError("layer index must be a nonnegative integer")
        slot = layer_idx % 2
        if require_owner:
            owner = self._staging._slot_identities[slot]
            if owner is None or (
                owner.session is not self.session
                or owner.generation != self.generation
                or owner.layer != layer_idx
                or owner.cycle != self._cycle
            ):
                raise RuntimeError("requested layer does not own its staging slot")
        return slot

    def slot_identity(self, slot):
        self._check()
        if type(slot) is not int or slot not in (0, 1):
            raise ValueError("slot must be zero or one")
        return self._staging._slot_identities[slot]

    def _validate_sources(self, sources):
        staging = self._staging
        if not isinstance(sources, Mapping) or sources.keys() != staging.record_shapes.keys():
            raise ValueError("prefetch record names must match staging record_shapes")
        length = None
        for name, shape in staging.record_shapes.items():
            source = sources[name]
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if source.ndim != len(shape) + 1 or tuple(source.shape[1:]) != shape:
                raise ValueError(f"{name} must have record shape {shape}")
            if length is None:
                length = len(source)
            if len(source) != length or length > staging.capacity:
                raise ValueError("prefetch records must have one length within staging capacity")
            if source.dtype != staging.dtype:
                raise ValueError(f"{name} must match staging dtype")
            if source.device != staging.device and source.device.type != "cpu":
                raise ValueError(f"{name} must be on CPU or the staging device")
            if staging.device.type == "cuda" and source.device.type == "cpu":
                if source.numel() and not source.is_pinned():
                    raise ValueError("CUDA prefetch requires pinned CPU source records")
            elif staging.device.type == "cpu" and source.device.type != "cpu":
                raise ValueError("CPU staging requires CPU source records")
        return length

    @torch.no_grad()
    def prefetch(self, layer_idx, sources):
        """Copy all supplied records; the model decides their content and length."""
        slot = self._slot(layer_idx, require_owner=False)
        length = self._validate_sources(sources)
        staging = self._staging
        old = staging._slot_identities[slot]
        if old is not None and not staging._slot_ready[slot]:
            raise RuntimeError("cannot replace a staging slot before its consumer waits ready")
        # Keep every submitted source alive through return, even if a copy or
        # event record below throws after earlier records were already queued.
        self._sources.extend(sources.values())
        staging._slot_identities[slot] = StagingIdentity(
            self.session, self.generation, layer_idx, self._cycle
        )
        staging._slot_ready[slot] = False

        def copy_records():
            for name, source in sources.items():
                staging._buffers[name][slot, :length].copy_(
                    source, non_blocking=staging.device.type == "cuda"
                )
                self.submitted_copy_bytes += source.numel() * source.element_size()

        try:
            if staging.device.type == "cuda":
                # This event contains the old slot's consumer, while compute queued
                # after this call can overlap the new copy in the other slot.
                staging._consumer_events[slot].record(self.caller_stream)
                staging._copy_stream.wait_event(staging._consumer_events[slot])
                with torch.cuda.stream(staging._copy_stream):
                    copy_records()
                    staging._ready_events[slot].record(staging._copy_stream)
            else:
                copy_records()
        except BaseException:
            self._interrupted = True
            raise

    def wait_ready(self, layer_idx):
        """Wait on the caller stream, then expose capacity views for suffix writes."""
        slot = self._slot(layer_idx, require_owner=True)
        staging = self._staging
        if self.caller_stream is not None:
            try:
                self.caller_stream.wait_event(staging._ready_events[slot])
            except BaseException:
                self._interrupted = True
                raise
        staging._slot_ready[slot] = True
        return self.view(layer_idx)

    def view(self, layer_idx, *, end=None):
        """Return borrowed views after copy-ready; their lifetime ends on reuse."""
        slot = self._slot(layer_idx, require_owner=True)
        staging = self._staging
        if not staging._slot_ready[slot]:
            raise RuntimeError("wait_ready must precede staging consumption")
        end = staging.capacity if end is None else end
        if type(end) is not int or not 0 <= end <= staging.capacity:
            raise ValueError("view end must lie within staging capacity")
        return {name: buffer[slot, :end] for name, buffer in staging._buffers.items()}

    def reset(self):
        """Drain the completed chunk and invalidate both layer ownerships."""
        self._check(allow_interrupted=True)
        self._staging._join(self)
        self._staging._clear_slots()
        self._cycle += 1
        self._interrupted = False

    def close(self):
        if self.closed:
            return
        staging = self._staging
        with staging._lock:
            if staging._active is not self:
                raise RuntimeError("staging lease ownership is stale")
            try:
                staging._join(self)
            finally:
                self.closed = True
            # These changes only occur once completion has been confirmed.
            staging._clear_slots()
            staging._active = None
