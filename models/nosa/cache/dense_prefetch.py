"""NOSA host history with a fenced double-buffer prefetch cache."""

import torch

from models.nosa.cache.offload import NosaOffloadCache


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


class NosaDensePrefetchCache(NosaOffloadCache):
    """Pinned historical K/V with two alternating, full-layer HBM buffers.

    Layer zero is queued at the start of each append. Once the current layer has
    written its suffix, its stream waits for that layer's history and queues the
    next layer's complete history on a copy stream. The copy stream waits for all
    previous consumers before reusing a slot. Thus next-layer copies can overlap
    the current layer's indexer, attention, and MLP. No sparse fetch is performed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._execution_resources is not None:
            if self._execution_resources.scheme != "dense_prefetch":
                raise ValueError("Dense prefetch cache requires matching execution resources")
            self._stage_keys = self._stage_values = None
            self._prefetch_stream = None
            self._ready = [None, None]
            self._staged_layers = [None, None]
            self.last_prefetch_bytes = 0
            return
        shape = (2, self.max_seq_len, self.config.num_key_value_heads, self.config.head_dim)
        self._stage_keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        self._stage_values = torch.empty_like(self._stage_keys)
        self._prefetch_stream = None
        self._ready = [None, None]
        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
            self._ready = [torch.cuda.Event(), torch.cuda.Event()]
        self._staged_layers = [None, None]
        self.last_prefetch_bytes = 0

    @property
    def attention_workspace(self):
        raise RuntimeError("Dense prefetch uses double staging, not sparse fetch workspace")

    def begin_step(self, token_count):
        result = super().begin_step(token_count)
        self._staged_layers = [None, None]
        self.last_prefetch_bytes = 0
        try:
            if self._execution_resources is not None:
                self._execution_resources.staging_lease.reset()
            self._prefetch(0)
        except BaseException as error:
            try:
                self.abort_step()
            except BaseException as rollback_error:  # noqa: BLE001 -- retain both failures.
                raise BaseExceptionGroup(
                    "NOSA cache begin and rollback both failed", [error, rollback_error]
                ) from None
            raise
        return result

    @torch.no_grad()
    def _prefetch(self, layer_idx):
        if layer_idx >= self.spec.num_layers:
            return
        slot = layer_idx % 2
        host = self.host_layer_view(layer_idx)
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self)
            lease = self._execution_resources.staging_lease
            previous = lease.submitted_copy_bytes
            try:
                lease.prefetch(
                    layer_idx,
                    sources={name: host[name][: self.length] for name in ("keys", "values")},
                )
            finally:
                self._transfer_metrics.add_payload(
                    "main_kv_host_to_device_bytes", lease.submitted_copy_bytes - previous
                )
            self.last_prefetch_bytes += (
                2
                * self.length
                * self.config.num_key_value_heads
                * self.config.head_dim
                * self.dtype.itemsize
            )
            self._staged_layers[slot] = layer_idx
            return

        def copy_history():
            self._stage_keys[slot, : self.length].copy_(
                host["keys"][: self.length], non_blocking=self.device.type == "cuda"
            )
            self._stage_values[slot, : self.length].copy_(
                host["values"][: self.length], non_blocking=self.device.type == "cuda"
            )

        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            # The current stream includes every earlier attention consumer.
            # This fences slot reuse, while later current-layer compute can
            # overlap the next-layer history copy queued below.
            self._prefetch_stream.wait_stream(current)
            with torch.cuda.stream(self._prefetch_stream):
                copy_history()
                self._ready[slot].record(self._prefetch_stream)
        else:
            copy_history()
        self.last_prefetch_bytes += (
            2
            * self.length
            * self.config.num_key_value_heads
            * self.config.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )
        self._staged_layers[slot] = layer_idx

    @torch.no_grad()
    def write_layer(self, layer_idx, **records):
        if self._staged_layers[layer_idx % 2] != layer_idx:
            raise RuntimeError("Dense prefetch requires layers in increasing order")
        super().write_layer(layer_idx, **records)
        slot = layer_idx % 2
        if self._execution_resources is not None:
            stage = self._execution_resources.staging_lease.wait_ready(layer_idx)
            for name in ("keys", "values"):
                stage[name][self.length : self._pending_end].copy_(records[name])
            self._prefetch(layer_idx + 1)
            return
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_event(self._ready[slot])
        self._stage_keys[slot, self.length : self._pending_end].copy_(records["keys"])
        self._stage_values[slot, self.length : self._pending_end].copy_(records["values"])
        self._prefetch(layer_idx + 1)

    def dense_layer_view(self, layer_idx):
        end = self.visible_length(layer_idx)
        slot = layer_idx % 2
        if self._staged_layers[slot] != layer_idx:
            raise RuntimeError("Requested layer no longer owns its dense staging slot")
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self)
            return {
                **self._execution_resources.staging_lease.view(layer_idx, end=end),
                "cis_scores": self.cis_scores[layer_idx, :end],
            }
        return {
            "keys": self._stage_keys[slot, :end],
            "values": self._stage_values[slot, :end],
            "cis_scores": self.cis_scores[layer_idx, :end],
        }

    def stats(self):
        result = super().stats()
        staging = sum(_bytes(t) for t in (self._stage_keys, self._stage_values) if t is not None)
        result["resident_bytes"] += staging
        result["capacity_bytes"] += staging
        return result

    def release(self):
        if self.released:
            return
        super().release()
        self._stage_keys = self._stage_values = None
        self._prefetch_stream = None
        self._ready = [None, None]
        self._staged_layers = [None, None]
