"""Transactional pinned-host records with device-resident append staging.

Record layouts belong to the model. This backend deliberately provides no
resident layer view: consumers fetch historical records through their own
device implementation while the uncommitted append remains on the device.
"""

import torch

from cache.manager import CacheSpec, _positive_integer


class HostBackingCache:
    """One request's host backing and owned device append tensors.

    CUDA writes run on a private transfer stream. Commit and destructive
    lifecycle operations synchronize the device before publishing metadata or
    releasing buffers also used by an attention backend's asynchronous streams.
    CPU placement provides the same transaction semantics for reference tests.
    """

    def __init__(self, spec: CacheSpec, max_seq_len: int, *, device, dtype):
        _positive_integer(max_seq_len, "Cache capacity")
        if max_seq_len > spec.max_position_embeddings:
            raise ValueError("Cache capacity must be within max_position_embeddings")
        self.spec = spec
        self.max_seq_len = max_seq_len
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Host backing supports CPU reference or CUDA computation")
        self.buffers = {
            name: torch.empty(
                (spec.num_layers, max_seq_len, *shape),
                device="cpu",
                dtype=dtype,
                pin_memory=self.device.type == "cuda",
            )
            for name, shape in spec.record_shapes.items()
        }
        self._capacity_bytes = sum(self._bytes(tensor) for tensor in self.buffers.values())
        self._length = 0
        self._pending_end = None
        self._written_layers = set()
        self._suffixes = {}
        self._layer_states = {}
        self._pending_layer_states = None
        self._released = False
        self._write_stream = None

    @staticmethod
    def _bytes(tensor):
        return tensor.numel() * tensor.element_size()

    @property
    def length(self):
        return self._length

    @property
    def released(self):
        return self._released

    def _ensure_alive(self):
        if self.released:
            raise RuntimeError("Cache session has been released")

    def _validate_layer(self, layer_idx):
        self._ensure_alive()
        if type(layer_idx) is not int or not 0 <= layer_idx < self.spec.num_layers:
            raise ValueError(f"Layer index must be within [0, {self.spec.num_layers})")

    def begin_step(self, token_count):
        self._ensure_alive()
        if self._pending_end is not None:
            raise RuntimeError("A cache step is already pending")
        _positive_integer(token_count, "token_count")
        end = self.length + token_count
        if end > self.max_seq_len:
            raise ValueError("Cache capacity would be exceeded by this step")
        self._pending_end = end
        self._written_layers.clear()
        self._suffixes.clear()
        self._pending_layer_states = dict(self._layer_states)
        return self.length, end

    def _validate_write(self, layer_idx, records):
        self._validate_layer(layer_idx)
        if self._pending_end is None:
            raise RuntimeError("begin_step must precede a layer write")
        if layer_idx in self._written_layers:
            raise RuntimeError(f"Layer {layer_idx} has already written this step")
        if records.keys() != self.spec.record_shapes.keys():
            raise ValueError("Layer records must match the names declared by CacheSpec")
        count = self._pending_end - self.length
        for name, record in records.items():
            if not isinstance(record, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            expected = (count, *self.spec.record_shapes[name])
            if tuple(record.shape) != expected:
                raise ValueError(f"{name} has shape {tuple(record.shape)}; expected {expected}")
            if record.device != self.device or record.dtype != self.dtype:
                raise ValueError(f"{name} must match the cache device and dtype")

    @torch.no_grad()
    def write_layer(self, layer_idx, **records):
        self._validate_write(layer_idx, records)
        # Own the append: the caller may reuse its projection output once this
        # call returns, and a background host write must not read that storage.
        suffix = {
            name: record.detach().clone(memory_format=torch.contiguous_format)
            for name, record in records.items()
        }
        self._suffixes[layer_idx] = suffix
        if self.device.type == "cuda":
            if self._write_stream is None:
                self._write_stream = torch.cuda.Stream(device=self.device)
            self._write_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._write_stream):
                for name, tensor in suffix.items():
                    self.buffers[name][layer_idx, self.length : self._pending_end].copy_(
                        tensor, non_blocking=True
                    )
                    tensor.record_stream(self._write_stream)
        else:
            for name, tensor in suffix.items():
                self.buffers[name][layer_idx, self.length : self._pending_end].copy_(tensor)
        self._written_layers.add(layer_idx)

    def visible_length(self, layer_idx):
        self._validate_layer(layer_idx)
        if self._pending_end is not None:
            if layer_idx not in self._written_layers:
                raise RuntimeError(f"Layer {layer_idx} has not written the pending step")
            return self._pending_end
        return self.length

    def host_layer_view(self, layer_idx):
        """Full-capacity host tensors; only the committed prefix is readable."""
        self._validate_layer(layer_idx)
        return {name: tensor[layer_idx] for name, tensor in self.buffers.items()}

    def suffix_layer_view(self, layer_idx):
        """Owned resident append, or empty device tensors outside a step."""
        self.visible_length(layer_idx)
        if self._pending_end is None:
            return {
                name: torch.empty((0, *shape), device=self.device, dtype=self.dtype)
                for name, shape in self.spec.record_shapes.items()
            }
        return self._suffixes[layer_idx]

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def validate_commit(self):
        self._ensure_alive()
        if self._pending_end is None:
            raise RuntimeError("No cache step is pending")
        if len(self._written_layers) != self.spec.num_layers:
            raise RuntimeError("Every layer must write before committing a cache step")

    def commit_step(self):
        self.validate_commit()
        self.synchronize()
        self._length = self._pending_end
        self._layer_states = self._pending_layer_states
        self._clear_pending()

    def _clear_pending(self):
        self._pending_end = None
        self._written_layers.clear()
        self._suffixes.clear()
        self._pending_layer_states = None

    def abort_step(self):
        self._ensure_alive()
        self.synchronize()
        self._clear_pending()

    def get_layer_state(self, layer_idx):
        self._validate_layer(layer_idx)
        states = self._pending_layer_states
        return (self._layer_states if states is None else states).get(layer_idx)

    def set_layer_state(self, layer_idx, state):
        self._validate_layer(layer_idx)
        states = self._pending_layer_states
        (self._layer_states if states is None else states)[layer_idx] = state

    def _validate_truncate(self, length):
        self._ensure_alive()
        if self._pending_end is not None:
            raise RuntimeError("Cannot truncate while a cache step is pending")
        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError("Invalid cache length: truncate only accepts a committed prefix")
        if 0 < length < self.length and self._layer_states:
            raise ValueError("Cannot truncate opaque layer state")

    def truncate(self, length):
        self._validate_truncate(length)
        self.synchronize()
        self._length = length
        if length == 0:
            self._layer_states.clear()

    def reset(self):
        self._ensure_alive()
        self.synchronize()
        self._length = 0
        self._layer_states.clear()
        self._clear_pending()

    def release(self):
        if self.released:
            return
        self.reset()
        self.buffers.clear()
        self._write_stream = None
        self._released = True

    def stats(self):
        return {
            "capacity_bytes": self._capacity_bytes,
            "resident_bytes": sum(
                self._bytes(tensor)
                for layer in self._suffixes.values()
                for tensor in layer.values()
            ),
            "host_bytes": sum(self._bytes(tensor) for tensor in self.buffers.values()),
            "length": self.length,
            "max_seq_len": self.max_seq_len,
        }
