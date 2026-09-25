"""Request cache ownership and the existing fully resident inference backend.

The resident backend allocates every layer on one device. CPU placement is
useful for correctness tests; it does not implement DRAM backing or GPU fetches.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from weakref import WeakSet

import torch

from cache.contracts import CacheSession


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CacheSpec:
    """Model-declared record layout, excluding the layer and token dimensions.

    ``compatibility_key`` is opaque to the cache manager. The model adapter
    decides whether a session's representation is compatible with its model.
    """

    num_layers: int
    max_position_embeddings: int
    record_shapes: Mapping[str, tuple[int, ...]]
    compatibility_key: object = None

    def __post_init__(self) -> None:
        _positive_integer(self.num_layers, "num_layers")
        _positive_integer(self.max_position_embeddings, "max_position_embeddings")
        if not self.record_shapes:
            raise ValueError("record_shapes must contain at least one record")
        shapes = {}
        for name, dimensions in self.record_shapes.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Record names must be nonempty strings")
            shape = tuple(dimensions)
            for dimension in shape:
                _positive_integer(dimension, f"{name} record dimension")
            shapes[name] = shape
        object.__setattr__(self, "record_shapes", MappingProxyType(shapes))


class ResidentCache:
    """Preallocated records and transactional appends for one request.

    Each buffer has shape ``[layer, capacity, *record_shape]``. ``length`` only
    advances after all layers have written and the model commits the step.
    Aborting leaves uncommitted suffix bytes available for the next overwrite.
    Sessions are used by one executor at a time; they are not thread safe.
    """

    def __init__(self, spec: CacheSpec, max_seq_len: int, *, device, dtype):
        _positive_integer(max_seq_len, "Cache capacity")
        if max_seq_len > spec.max_position_embeddings:
            raise ValueError("Cache capacity must be within max_position_embeddings")
        self.spec = spec
        self.max_seq_len = max_seq_len
        self.buffers = {
            name: torch.empty((spec.num_layers, max_seq_len, *shape), device=device, dtype=dtype)
            for name, shape in spec.record_shapes.items()
        }
        first = next(iter(self.buffers.values()))
        self.device = first.device
        self.dtype = first.dtype
        self._capacity_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in self.buffers.values()
        )
        self._length = 0
        self._pending_end: int | None = None
        self._written_layers: set[int] = set()
        self._layer_states: dict[int, object] = {}
        self._pending_layer_states: dict[int, object] | None = None
        self._released = False

    @property
    def length(self) -> int:
        return self._length

    @property
    def released(self) -> bool:
        return self._released

    def _ensure_alive(self) -> None:
        if self._released:
            raise RuntimeError("Cache session has been released")

    def _validate_layer(self, layer_idx: int) -> None:
        self._ensure_alive()
        if (
            not isinstance(layer_idx, int)
            or isinstance(layer_idx, bool)
            or not 0 <= layer_idx < self.spec.num_layers
        ):
            raise ValueError(f"Layer index must be within [0, {self.spec.num_layers})")

    def begin_step(self, token_count: int) -> tuple[int, int]:
        """Start an append and return its half-open token range."""
        self._ensure_alive()
        if self._pending_end is not None:
            raise RuntimeError("A cache step is already pending")
        _positive_integer(token_count, "token_count")
        end = self.length + token_count
        if end > self.max_seq_len:
            raise ValueError("Cache capacity would be exceeded by this step")
        self._pending_end = end
        self._written_layers.clear()
        self._pending_layer_states = dict(self._layer_states)
        return self.length, end

    def write_layer(self, layer_idx: int, **records: torch.Tensor) -> None:
        """Write all declared records for a layer's pending token range."""
        self._validate_layer(layer_idx)
        if self._pending_end is None:
            raise RuntimeError("begin_step must precede a layer write")
        if layer_idx in self._written_layers:
            raise RuntimeError(f"Layer {layer_idx} has already written this step")
        if records.keys() != self.spec.record_shapes.keys():
            raise ValueError("Layer records must match the names declared by CacheSpec")
        token_count = self._pending_end - self.length
        for name, record in records.items():
            if not isinstance(record, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            expected = (token_count, *self.spec.record_shapes[name])
            if tuple(record.shape) != expected:
                raise ValueError(f"{name} has shape {tuple(record.shape)}; expected {expected}")
            if record.device != self.device or record.dtype != self.dtype:
                raise ValueError(f"{name} must match the cache device and dtype")
        with torch.no_grad():
            for name, record in records.items():
                self.buffers[name][layer_idx, self.length : self._pending_end].copy_(record)
        self._written_layers.add(layer_idx)

    def layer_view(self, layer_idx: int) -> Mapping[str, torch.Tensor]:
        """Expose this resident layer's committed prefix plus its written step.

        A layer must write before seeing a pending suffix, so attention cannot
        read another layer's uninitialized records during the same forward.
        Callers must discard these borrowed tensor views before releasing the
        session; external views can otherwise keep the allocation alive.
        """
        self._validate_layer(layer_idx)
        end = self.length
        if self._pending_end is not None:
            if layer_idx not in self._written_layers:
                raise RuntimeError(f"Layer {layer_idx} has not written the pending step")
            end = self._pending_end
        return {name: tensor[layer_idx, :end] for name, tensor in self.buffers.items()}

    def commit_step(self) -> None:
        """Publish a successful model forward only after all layer writes."""
        self._ensure_alive()
        if self._pending_end is None:
            raise RuntimeError("No cache step is pending")
        if len(self._written_layers) != self.spec.num_layers:
            raise RuntimeError("Every layer must write before committing a cache step")
        self._length = self._pending_end
        self._layer_states = self._pending_layer_states
        self._clear_pending()

    def _clear_pending(self) -> None:
        self._pending_end = None
        self._written_layers.clear()
        self._pending_layer_states = None

    def abort_step(self) -> None:
        """Discard pending metadata without changing the committed prefix."""
        self._ensure_alive()
        self._clear_pending()

    def get_layer_state(self, layer_idx: int) -> object | None:
        """Return opaque model state owned by this request, if present."""
        self._validate_layer(layer_idx)
        states = (
            self._pending_layer_states
            if self._pending_layer_states is not None
            else self._layer_states
        )
        return states.get(layer_idx)

    def set_layer_state(self, layer_idx: int, state: object) -> None:
        """Replace model state, publishing it with the current step's commit.

        State algorithms are model-owned. During a step they must replace
        objects instead of mutating committed objects in place if abort needs
        to preserve the previous state; opaque objects are not deep-copied.
        """
        self._validate_layer(layer_idx)
        states = (
            self._pending_layer_states
            if self._pending_layer_states is not None
            else self._layer_states
        )
        states[layer_idx] = state

    def reset(self) -> None:
        """Discard request state and reuse the resident allocation."""
        self._ensure_alive()
        self._length = 0
        self._layer_states.clear()
        self._clear_pending()

    def release(self) -> None:
        """Drop owned tensors and state; repeated releases are harmless."""
        if self._released:
            return
        self.reset()
        self.buffers.clear()
        self._released = True

    def stats(self) -> dict[str, int]:
        """Report allocation bytes, including unused capacity, and token use.

        ``capacity_bytes`` describes the configured layout, while
        ``resident_bytes`` becomes zero after release. ``host_bytes`` denotes
        offload backing and is zero even for the CPU correctness backend.
        """
        return {
            "capacity_bytes": self._capacity_bytes,
            "resident_bytes": sum(
                tensor.numel() * tensor.element_size() for tensor in self.buffers.values()
            ),
            "host_bytes": 0,
            "length": self.length,
            "max_seq_len": self.max_seq_len,
        }


class CacheManager[SessionT: CacheSession]:
    """Allocate and release independent request sessions via a model factory.

    Weak references track ownership without retaining sessions created through
    legacy ``model.new_cache`` calls that do not explicitly release them.
    """

    def __init__(self, factory: Callable[[int], SessionT]):
        self.factory = factory
        self._sessions: WeakSet[SessionT] = WeakSet()

    def allocate(self, max_seq_len: int) -> SessionT:
        session = self.factory(max_seq_len)
        if session.released or session in self._sessions:
            raise ValueError("Cache factory must return a fresh session")
        self._sessions.add(session)
        return session

    def release(self, session: SessionT) -> None:
        if session not in self._sessions:
            raise ValueError("Cache session does not belong to this manager")
        session.release()
