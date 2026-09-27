"""Model-declared derived records and shared scratch space for one request.

The owner supplies record lengths and validates their numerical meaning. This
module only manages allocation, append reservations, and transactional cursors.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class IndexerBufferSpec:
    capacity: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self):
        _integer(self.capacity, "Record capacity")
        shape = tuple(self.shape)
        for dimension in shape:
            _integer(dimension, "Record dimension", minimum=1)
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("Record dtype must be a torch.dtype")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class IndexerLayerState:
    lengths: Mapping[str, int]
    validated_tokens: int

    def __post_init__(self):
        object.__setattr__(self, "lengths", MappingProxyType(dict(self.lengths)))


@dataclass(frozen=True)
class IndexerReservation:
    """Borrowed buffers; only append ranges beyond ``previous`` may be written.

    ``views`` use the requested visible lengths, while ``target`` preserves any
    longer cached prefix. Callers must finish or abort the reservation before
    changing the request lifecycle, and discard views before releasing it.
    """

    previous: IndexerLayerState
    target: IndexerLayerState
    buffers: Mapping[str, torch.Tensor]
    views: Mapping[str, torch.Tensor]


class IndexerCache:
    """Lazily allocated per-layer records, with one request-wide byte workspace.

    Layers may remain unused while their owner's token length advances. A
    cache-only reservation can later catch up from resident records outside a
    pending append. Finished reservations inside a step publish only at commit.
    Sessions and their borrowed scratch space are used serially, on one stream.
    """

    def __init__(self, num_layers, max_seq_len, specs, *, device):
        self.num_layers = _integer(num_layers, "num_layers", minimum=1)
        self.max_seq_len = _integer(max_seq_len, "max_seq_len", minimum=1)
        if not specs or any(
            not isinstance(name, str) or not name or not isinstance(spec, IndexerBufferSpec)
            for name, spec in specs.items()
        ):
            raise ValueError("specs must map nonempty record names to IndexerBufferSpec")
        self.specs = MappingProxyType(dict(specs))
        self.device = torch.device(device)
        self._length = 0
        self._pending_end = None
        self._committed = [self._empty_state() for _ in range(num_layers)]
        self._pending_states = {}
        self._reservations = {}
        self._buffers = {}
        self._workspace = None
        self._workspace_capacity_bytes = 0
        self._released = False
        self._record_capacity_bytes = num_layers * sum(
            spec.capacity
            * self._elements(spec.shape)
            * torch.empty((), dtype=spec.dtype).element_size()
            for spec in self.specs.values()
        )

    @staticmethod
    def _elements(shape):
        count = 1
        for dimension in shape:
            count *= dimension
        return count

    def _empty_state(self):
        return IndexerLayerState(dict.fromkeys(self.specs, 0), 0)

    @property
    def length(self):
        return self._length

    @property
    def released(self):
        return self._released

    def _ensure_alive(self):
        if self._released:
            raise RuntimeError("Indexer cache has been released")

    def _validate_layer(self, layer_idx):
        self._ensure_alive()
        if type(layer_idx) is not int or not 0 <= layer_idx < self.num_layers:
            raise ValueError("Invalid indexer cache layer index")

    def _validate_lengths(self, lengths):
        if lengths.keys() != self.specs.keys():
            raise ValueError("Record lengths must match the declared specs")
        for name, length in lengths.items():
            _integer(length, f"{name} length")
            if length > self.specs[name].capacity:
                raise ValueError(f"{name} length exceeds its capacity")

    def layer_state(self, layer_idx):
        self._validate_layer(layer_idx)
        return self._pending_states.get(layer_idx, self._committed[layer_idx])

    def _allocate_layer(self, layer_idx):
        if layer_idx not in self._buffers:
            self._buffers[layer_idx] = {
                name: torch.empty(
                    (spec.capacity, *spec.shape), device=self.device, dtype=spec.dtype
                )
                for name, spec in self.specs.items()
            }
        return self._buffers[layer_idx]

    def reserve_layer(self, layer_idx, token_length, lengths):
        self._validate_layer(layer_idx)
        if layer_idx in self._reservations:
            raise RuntimeError("This layer already has a pending reservation")
        _integer(token_length, "token_length")
        visible_end = self.length if self._pending_end is None else self._pending_end
        if token_length > visible_end:
            raise ValueError("Cannot validate tokens beyond the owner's visible length")
        self._validate_lengths(lengths)
        previous = self.layer_state(layer_idx)
        target = IndexerLayerState(
            {name: max(previous.lengths[name], length) for name, length in lengths.items()},
            max(previous.validated_tokens, token_length),
        )
        buffers = self._allocate_layer(layer_idx)
        reservation = IndexerReservation(
            previous,
            target,
            MappingProxyType(buffers),
            MappingProxyType({name: buffers[name][:length] for name, length in lengths.items()}),
        )
        self._reservations[layer_idx] = reservation
        return reservation

    def finish_layer(self, layer_idx):
        self._validate_layer(layer_idx)
        if layer_idx not in self._reservations:
            raise RuntimeError("No indexer layer reservation is pending")
        state = self._reservations.pop(layer_idx).target
        if self._pending_end is None:
            self._committed[layer_idx] = state
        else:
            self._pending_states[layer_idx] = state

    def abort_layer(self, layer_idx):
        self._validate_layer(layer_idx)
        self._reservations.pop(layer_idx, None)

    def layer_view(self, layer_idx, *, lengths=None):
        state = self.layer_state(layer_idx)
        lengths = state.lengths if lengths is None else lengths
        self._validate_lengths(lengths)
        if any(length > state.lengths[name] for name, length in lengths.items()):
            raise ValueError("Requested records have not been materialized")
        if layer_idx not in self._buffers:
            return {
                name: torch.empty((0, *spec.shape), dtype=spec.dtype, device=self.device)
                for name, spec in self.specs.items()
            }
        return {name: self._buffers[layer_idx][name][:length] for name, length in lengths.items()}

    def begin_step(self, end):
        self._ensure_alive()
        if self._pending_end is not None or self._reservations:
            raise RuntimeError("An indexer cache step or reservation is already pending")
        _integer(end, "Step end", minimum=1)
        if not self.length < end <= self.max_seq_len:
            raise ValueError("Step end must append within indexer cache capacity")
        self._pending_end = end

    def validate_commit(self):
        self._ensure_alive()
        if self._pending_end is None:
            raise RuntimeError("No indexer cache step is pending")
        if self._reservations:
            raise RuntimeError("Every pending indexer reservation must finish before commit")

    def commit_step(self):
        self.validate_commit()
        for layer_idx, state in self._pending_states.items():
            self._committed[layer_idx] = state
        self._length = self._pending_end
        self._clear_pending()

    def _clear_pending(self):
        self._pending_end = None
        self._pending_states.clear()
        self._reservations.clear()

    def abort_step(self):
        self._ensure_alive()
        self._clear_pending()

    def truncate(self, token_length, lengths):
        self._ensure_alive()
        if self._pending_end is not None or self._reservations:
            raise RuntimeError("Cannot truncate while an indexer cache operation is pending")
        _integer(token_length, "token_length")
        if token_length > self.length:
            raise ValueError("Cannot advance indexer cache length through truncate")
        self._validate_lengths(lengths)
        self._committed = [
            IndexerLayerState(
                {name: min(state.lengths[name], limit) for name, limit in lengths.items()},
                min(state.validated_tokens, token_length),
            )
            for state in self._committed
        ]
        self._length = token_length

    def workspace(self, numel, dtype):
        """Borrow flat scratch space; the next caller may overwrite every byte."""
        self._ensure_alive()
        _integer(numel, "Workspace element count")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("Workspace dtype must be a torch.dtype")
        element_size = torch.empty((), dtype=dtype).element_size()
        required = numel * element_size
        if self._workspace is None or required > self._workspace_capacity_bytes:
            capacity = (required + 255) // 256 * 256
            self._workspace = torch.empty(capacity, dtype=torch.uint8, device=self.device)
            self._workspace_capacity_bytes = capacity
        return self._workspace.view(dtype)[:numel]

    def reset(self):
        self._ensure_alive()
        self._length = 0
        self._committed = [self._empty_state() for _ in range(self.num_layers)]
        self._clear_pending()

    def release(self):
        if self.released:
            return
        self.reset()
        self._buffers.clear()
        self._workspace = None
        self._released = True

    def stats(self):
        resident = sum(
            tensor.numel() * tensor.element_size()
            for layer in self._buffers.values()
            for tensor in layer.values()
        )
        workspace = 0 if self._workspace is None else self._workspace.numel()
        return {
            "capacity_bytes": self._record_capacity_bytes + self._workspace_capacity_bytes,
            "resident_bytes": resident + workspace,
            "workspace_bytes": workspace,
            "host_bytes": 0,
            "length": self.length,
            "max_seq_len": self.max_seq_len,
        }
