"""NOSA's post-RoPE K / unmodified V layout on the shared resident backend."""

from copy import deepcopy

import torch

from cache.indexer_cache import IndexerBufferSpec, IndexerCache
from cache.manager import CacheSpec, ResidentCache
from models.nosa.config import NosaConfig


class _NosaTransferMetrics:
    """Last request's submitted KV payload, separated into prefix and candidate.

    These are tensor payload bytes, not physical PCIe traffic. Device scalar
    observation, metadata, allocator traffic and ordinary activations are not
    included. Failed requests retain observed/submitted attempts and are marked
    failed; they must never become successful measurement rows.
    """

    _fields = (
        "main_kv_host_to_device_bytes",
        "main_kv_device_to_host_bytes",
        "indexer_host_to_device_bytes",
    )

    def __init__(self, resources, *, device_copy_counter=False):
        self.is_cuda = resources.device.type == "cuda"
        self._payload = [{name: 0 for name in self._fields} for _ in range(2)]
        self._phase = None
        self.status = "idle"
        self._sparse_fetch_bytes = None
        self._allocation_stream = None
        if self.is_cuda and (
            resources.scheme in ("serial_sparse", "overlap") or device_copy_counter
        ):
            self._sparse_fetch_bytes = torch.empty(2, dtype=torch.int64, device=resources.device)
            self._allocation_stream = torch.cuda.current_stream(resources.device)
        self._initialized = False

    @property
    def storage_bytes(self):
        if self._sparse_fetch_bytes is None:
            return 0
        return self._sparse_fetch_bytes.untyped_storage().nbytes()

    def begin(self, phase):
        if phase not in ("prefix", "candidate"):
            raise ValueError("Transfer phase must be prefix or candidate")
        if self._phase is not None:
            raise RuntimeError("A transfer measurement is already active")
        preserve_prefix = phase == "candidate" and self.status == "prefill_ready"
        self._phase = int(phase == "candidate")
        self.status = f"running_{phase}"
        for index in (1,) if preserve_prefix else (0, 1):
            self._payload[index] = {name: 0 for name in self._fields}
        if self._sparse_fetch_bytes is not None:
            if self._allocation_stream is not None:
                torch.cuda.current_stream(self._sparse_fetch_bytes.device).wait_stream(
                    self._allocation_stream
                )
                self._allocation_stream = None
            target = self._sparse_fetch_bytes[1:] if preserve_prefix else self._sparse_fetch_bytes
            target.zero_()
        self._initialized = True

    def finish(self, *, success):
        if self._phase is None:
            raise RuntimeError("No transfer measurement is active")
        self.status = ("prefill_ready" if self._phase == 0 else "complete") if success else "failed"
        self._phase = None

    def add_payload(self, field, size):
        if self.is_cuda and self._phase is not None:
            self._payload[self._phase][field] += size

    def add_sparse_fetch(self, source):
        if self._sparse_fetch_bytes is not None and self._phase is not None:
            # The source belongs to a reusable workspace. Consume it on the
            # same stream before the next layer resets it; retain no source view.
            self._sparse_fetch_bytes[self._phase : self._phase + 1].add_(source)

    def snapshot(self):
        if self._phase is not None:
            raise RuntimeError("Read transfer metrics after the execution lease finishes")
        fetch = (
            self._sparse_fetch_bytes.tolist()
            if self._sparse_fetch_bytes is not None and self._initialized
            else [0, 0]
        )
        phases = [dict(values) for values in self._payload]
        for index, values in enumerate(phases):
            values["main_kv_host_to_device_bytes"] += fetch[index]
            values["host_to_device_bytes"] = (
                values["main_kv_host_to_device_bytes"] + values["indexer_host_to_device_bytes"]
            )
            values["device_to_host_bytes"] = values["main_kv_device_to_host_bytes"]
        result = {key: phases[0][key] + phases[1][key] for key in phases[0]}
        for label, values in zip(("prefix", "candidate"), phases, strict=True):
            result.update({f"{label}_{name}": value for name, value in values.items()})
        result["transfer_metrics_status"] = self.status
        result["transfer_metrics_scope"] = "last_request"
        result["transfer_bytes_semantics"] = (
            "KV and indexer-boundary payload, not physical link traffic"
        )
        return result

    def release(self):
        self._sparse_fetch_bytes = self._allocation_stream = None


class NosaKVCache(ResidentCache):
    """One unpadded request, with [layer, capacity, kv_head, head_dim] tensors.

    The original constructor and ``config``, ``keys``, ``values`` attributes are
    kept for callers. Model layers write through the session operations rather
    than managing the physical buffers themselves.
    """

    def __init__(
        self,
        config: NosaConfig,
        max_seq_len: int,
        *,
        device,
        dtype,
        with_cis=False,
        execution_resources=None,
    ):
        self._execution_resources = execution_resources
        if execution_resources is not None:
            execution_resources.validate_cache(
                capacity=max_seq_len,
                device=device,
                dtype=dtype,
                kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
            )
            if execution_resources.scheme != "hbm" or not with_cis:
                raise ValueError("Shared resident cache requires the hbm full-NOSA scheme")
        if with_cis and isinstance(max_seq_len, int) and max_seq_len > 262144:
            raise ValueError("Full NOSA indexer cache supports at most 262144 tokens")
        # LongRoPE factors live in a nested mutable dict even though NosaConfig
        # is frozen. A snapshot makes the model reject changed rotation settings
        # instead of appending differently rotated keys to an existing cache.
        self.config = deepcopy(config)
        self.with_cis = with_cis
        shape = (config.num_key_value_heads, config.head_dim)
        records = {"keys": shape, "values": shape}
        if with_cis:
            records["cis_scores"] = (config.num_key_value_heads,)
        spec = CacheSpec(
            num_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            record_shapes=records,
            compatibility_key=(self.config, with_cis),
        )
        super().__init__(spec, max_seq_len, device=device, dtype=dtype)
        self.indexer_cache = None
        self._native_indexer_host_flag = None
        self._transfer_metrics = (
            _NosaTransferMetrics(execution_resources) if execution_resources is not None else None
        )
        if with_cis:
            compressed, pooled = self._indexer_lengths(max_seq_len)
            self.indexer_cache = IndexerCache(
                config.num_hidden_layers,
                max_seq_len,
                {
                    "compressed_keys": IndexerBufferSpec(compressed, shape, dtype),
                    "compressed_cis": IndexerBufferSpec(compressed, (shape[0],), dtype),
                    "pooled_cis": IndexerBufferSpec(pooled, (shape[0],), dtype),
                },
                device=self.device,
            )

    def native_indexer_host_flag(self):
        """One checked native flag for this request's serial CUDA stream."""
        import torch

        self._ensure_alive()
        if self._native_indexer_host_flag is None:
            self._native_indexer_host_flag = torch.empty(
                (), dtype=torch.bool, device="cpu", pin_memory=True
            )
        return self._native_indexer_host_flag

    @staticmethod
    def _indexer_lengths(length):
        return max(0, length // 16 - 1), max(0, (length - 16) // 64)

    def begin_step(self, token_count):
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self, token_count)
        start, end = super().begin_step(token_count)
        try:
            if self.indexer_cache is not None:
                self.indexer_cache.begin_step(end)
        except BaseException:
            super().abort_step()
            raise
        return start, end

    def write_layer(self, layer_idx, **records):
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self)
        return super().write_layer(layer_idx, **records)

    def validate_commit(self):
        self._ensure_alive()
        if self._pending_end is None:
            raise RuntimeError("No cache step is pending")
        if len(self._written_layers) != self.spec.num_layers:
            raise RuntimeError("Every layer must write before committing a cache step")
        if self.indexer_cache is not None:
            self.indexer_cache.validate_commit()

    def commit_step(self):
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self)
        # Validate both sides before either committed cursor advances. The
        # following commits only publish metadata and cannot launch work.
        self.validate_commit()
        super().commit_step()
        if self.indexer_cache is not None:
            self.indexer_cache.commit_step()

    def abort_step(self):
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self)
        super().abort_step()
        if self.indexer_cache is not None:
            self.indexer_cache.abort_step()

    def truncate(self, length):
        if self._execution_resources is not None:
            self._execution_resources.check_mutation(self)
        self._ensure_alive()
        if self._pending_end is not None:
            raise RuntimeError("Cannot truncate while a cache step is pending")
        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError("Invalid cache length: truncate only accepts a committed prefix")
        if 0 < length < self.length and self._layer_states:
            raise ValueError("Cannot truncate opaque layer state")
        if self.indexer_cache is not None:
            compressed, pooled = self._indexer_lengths(length)
            self.indexer_cache.truncate(
                length,
                {
                    "compressed_keys": compressed,
                    "compressed_cis": compressed,
                    "pooled_cis": pooled,
                },
            )
        self._length = length
        if length == 0:
            self._layer_states.clear()

    def reset(self):
        if self._execution_resources is not None:
            self._execution_resources.check_mutation(self)
        super().reset()
        if self.indexer_cache is not None:
            self.indexer_cache.reset()

    def release(self):
        if self.released:
            return
        resources = self._execution_resources
        if resources is not None:
            resources.check_mutation(self, releasing=True)
        super().release()
        self._native_indexer_host_flag = None
        if self._transfer_metrics is not None:
            self._transfer_metrics.release()
            self._transfer_metrics = None
        if self.indexer_cache is not None:
            self.indexer_cache.release()
        if resources is not None:
            resources.detach(self)
            self._execution_resources = None

    def stats(self):
        result = super().stats()
        if self.indexer_cache is not None:
            derived = self.indexer_cache.stats()
            result["capacity_bytes"] += derived["capacity_bytes"]
            result["resident_bytes"] += derived["resident_bytes"]
        if self._execution_resources is not None and self._native_indexer_host_flag is not None:
            result["host_bytes"] += self._native_indexer_host_flag.untyped_storage().nbytes()
        return result

    @property
    def keys(self):
        return self.buffers["keys"]

    @property
    def values(self):
        return self.buffers["values"]

    @property
    def cis_scores(self):
        return self.buffers["cis_scores"]

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        """Keep legacy rewinds without exposing uncommitted or stale records."""
        self.truncate(value)
