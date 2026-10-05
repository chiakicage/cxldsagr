"""NOSA host K/V backing with resident CIS and incremental indexer records."""

from copy import deepcopy

import torch

from cache.host_backing import HostBackingCache
from cache.indexer_cache import IndexerBufferSpec, IndexerCache
from cache.manager import CacheSpec
from models.nosa.cache.resident import _NosaTransferMetrics


class NosaOffloadCache(HostBackingCache):
    """Host history, owned resident append, and one shared attention workspace.

    Only full NOSA is supported. Historical K is never reloaded in full for
    indexing: each append reads at most 31 committed boundary tokens to finish
    newly complete 32-token, stride-16 compression windows. Main attention
    borrows a single layer's logical-address staging allocation across layers.
    """

    def __init__(
        self,
        config,
        max_seq_len,
        *,
        device,
        dtype,
        with_cis=True,
        query_tile_size=128,
        fetch_ctas=96,
        overlap=True,
        execution_resources=None,
        host_capacity=None,
    ):
        self._execution_resources = execution_resources
        if execution_resources is not None:
            execution_resources.validate_cache(
                capacity=max_seq_len,
                device=device,
                dtype=dtype,
                kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                query_tile_size=query_tile_size,
                fetch_ctas=fetch_ctas,
                overlap=overlap,
            )
            if execution_resources.scheme == "hbm":
                raise ValueError("Offload cache requires an offload execution scheme")
        if not with_cis:
            raise ValueError("NOSA offload requires full NOSA with CIS")
        if type(query_tile_size) is not int or query_tile_size <= 0:
            raise ValueError("query_tile_size must be a positive integer")
        if type(fetch_ctas) is not int or fetch_ctas <= 0:
            raise ValueError("fetch_ctas must be a positive integer")
        if type(overlap) is not bool:
            raise ValueError("overlap must be a boolean")
        if type(max_seq_len) is int and max_seq_len > 262144:
            raise ValueError("Full NOSA indexer cache supports at most 262144 tokens")
        if torch.device(device).type == "cuda":
            if dtype != torch.bfloat16 or config.head_dim != 128:
                raise ValueError("CUDA NOSA offload requires BF16 and head_dim=128")
            if config.num_attention_heads != config.num_key_value_heads * 16:
                raise ValueError("CUDA NOSA offload requires 16 query heads per KV head")
            if torch.cuda.get_device_capability(device)[0] != 9:
                raise RuntimeError("CUDA NOSA offload requires a Hopper GPU")
        self.config = deepcopy(config)
        self.with_cis = True
        self.query_tile_size = query_tile_size
        self.fetch_ctas = fetch_ctas
        self.overlap = overlap
        heads, dim = config.num_key_value_heads, config.head_dim
        spec = CacheSpec(
            config.num_hidden_layers,
            config.max_position_embeddings,
            {"keys": (heads, dim), "values": (heads, dim)},
            compatibility_key=(self.config, True, "offload"),
        )
        if host_capacity is None:
            host_capacity = max_seq_len
        if type(host_capacity) is not int or not 0 < host_capacity <= max_seq_len:
            raise ValueError("Host capacity must be positive and within execution capacity")
        super().__init__(spec, host_capacity, device=device, dtype=dtype)
        # Fixed-history serving keeps candidate execution capacity on device;
        # only retained history owns pinned backing. Ordinary caches use equal
        # capacities and retain their original persistent-append semantics.
        self.max_seq_len = max_seq_len
        self.cis_scores = torch.empty(
            (config.num_hidden_layers, max_seq_len, heads), device=self.device, dtype=dtype
        )
        self._cis_capacity_bytes = self._bytes(self.cis_scores)
        compressed, pooled = self._indexer_lengths(max_seq_len)
        self.indexer_cache = IndexerCache(
            config.num_hidden_layers,
            max_seq_len,
            {
                "compressed_keys": IndexerBufferSpec(compressed, (heads, dim), dtype),
                "compressed_cis": IndexerBufferSpec(compressed, (heads,), dtype),
                "pooled_cis": IndexerBufferSpec(pooled, (heads,), dtype),
            },
            device=self.device,
        )
        self._attention_workspace = None
        self._transfer_metrics = (
            _NosaTransferMetrics(execution_resources) if execution_resources is not None else None
        )

    @staticmethod
    def _indexer_lengths(length):
        return max(0, length // 16 - 1), max(0, (length - 16) // 64)

    @property
    def attention_workspace(self):
        self._ensure_alive()
        if self._execution_resources is not None:
            return self._execution_resources.fetch(self)
        if self._attention_workspace is None:
            from operators.nosa.attention.offload.api import NosaFetchWorkspace

            self._attention_workspace = NosaFetchWorkspace(
                self.max_seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim,
                device=self.device,
                dtype=self.dtype,
                query_tile_size=self.query_tile_size,
                fetch_ctas=self.fetch_ctas,
                overlap=self.overlap,
            )
        return self._attention_workspace

    def begin_step(self, token_count):
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self, token_count)
        start, end = super().begin_step(token_count)
        try:
            self.indexer_cache.begin_step(end)
        except BaseException as error:
            try:
                super().abort_step()
            except BaseException as rollback_error:  # noqa: BLE001 -- retain both failures.
                raise BaseExceptionGroup(
                    "NOSA cache begin and rollback both failed", [error, rollback_error]
                ) from None
            raise
        return start, end

    @torch.no_grad()
    def write_layer(self, layer_idx, **records):
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self)
        if records.keys() != {"keys", "values", "cis_scores"}:
            raise ValueError("NOSA offload writes require keys, values and cis_scores")
        kv = {name: records[name] for name in ("keys", "values")}
        self._validate_write(layer_idx, kv)
        cis = records["cis_scores"]
        shape = (self._pending_end - self.length, self.config.num_key_value_heads)
        if not isinstance(cis, torch.Tensor):
            raise TypeError("cis_scores must be a tensor")
        if cis.shape != shape or cis.device != self.device or cis.dtype != self.dtype:
            raise ValueError("cis_scores must match the append shape and cache device/dtype")
        if self._transfer_metrics is None:
            super().write_layer(layer_idx, **kv)
        else:
            self._write_shared_layer(layer_idx, kv)
        self.cis_scores[layer_idx, self.length : self._pending_end].copy_(cis)

    def _write_shared_layer(self, layer_idx, records):
        """HostBackingCache's owned append, counted after each successful copy API.

        Shared serving retains the same independent suffix clones, transfer
        stream dependency and lifetime as the default HostBackingCache path.
        Per-record observation avoids counting a second copy if it throws.
        """
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
                    self._transfer_metrics.add_payload(
                        "main_kv_device_to_host_bytes", tensor.numel() * tensor.element_size()
                    )
                    tensor.record_stream(self._write_stream)
        else:
            for name, tensor in suffix.items():
                self.buffers[name][layer_idx, self.length : self._pending_end].copy_(tensor)
        self._written_layers.add(layer_idx)

    def offload_layer_view(self, layer_idx):
        end = self.visible_length(layer_idx)
        suffix = self.suffix_layer_view(layer_idx)
        host = self.host_layer_view(layer_idx)
        return {
            "host_keys": host["keys"],
            "host_values": host["values"],
            "suffix_keys": suffix["keys"],
            "suffix_values": suffix["values"],
            "cis_scores": self.cis_scores[layer_idx, :end],
            "query_start": self.length,
        }

    @torch.no_grad()
    def prepare_indexer_inputs(self, layer_idx, *, query=None):
        """Append derived records without transferring the historical K prefix.

        Ordinary calls observe validation before derived writes. Deferred model
        calls retain the device flag until the pre-commit decision and may write
        only unpublished append records. Previously committed records remain
        immutable; rollback leaves pending bytes to overwrite.
        Calling twice in one step reuses prepared records, but a supplied Q is
        always checked again. Direct cache commit supplies no Q and still
        validates every newly appended K/CIS value.
        """
        end = self.visible_length(layer_idx)
        state = self.indexer_cache.layer_state(layer_idx)
        count, stable = self._indexer_lengths(end)
        lengths = {"compressed_keys": count, "compressed_cis": count, "pooled_cis": stable}
        from models.nosa.execution.deferred_validation import finite_flag

        deferred = finite_flag(self, layer_idx) if query is not None else None
        if state.validated_tokens == end and dict(state.lengths) == lengths:
            if query is not None:
                if query.is_cuda:
                    from operators.nosa.indexer.validation import all_finite

                    finite = all_finite(
                        query,
                        query[:0, : self.config.num_key_value_heads],
                        self.cis_scores[layer_idx, :0],
                        out=deferred,
                    )
                else:
                    finite = torch.isfinite(query).all()
                if deferred is None and not finite:
                    raise ValueError("NOSA indexer requires finite Q, K and CIS scores")
            return self.indexer_cache.layer_view(layer_idx)
        if state.validated_tokens != self.length:
            raise RuntimeError("Offload indexer must materialize every committed append")
        suffix = self.suffix_layer_view(layer_idx)["keys"]
        cis = self.cis_scores[layer_idx, :end]
        if suffix.is_cuda:
            from operators.nosa.indexer.validation import all_finite

            # all_finite requires a nonempty Q. A direct cache commit can use
            # one K row as that input while still checking the complete append.
            finite = all_finite(
                suffix[:1] if query is None else query, suffix, cis[self.length :], out=deferred
            )
        else:
            finite = torch.isfinite(suffix).all() & torch.isfinite(cis[self.length :]).all()
            if query is not None:
                finite = finite & torch.isfinite(query).all()
        if deferred is None and not finite:
            raise ValueError("NOSA indexer requires finite Q, K and CIS scores")
        reservation = self.indexer_cache.reserve_layer(layer_idx, end, lengths)
        try:
            old_count = state.lengths["compressed_keys"]
            old_stable = state.lengths["pooled_cis"]
            keys = suffix[:0]
            if old_count < count:
                first_token = old_count * 16
                boundary_length = self.length - first_token
                if not 0 <= boundary_length <= 31:
                    raise RuntimeError("Offload compression boundary exceeds 31 historical tokens")
                boundary = self.buffers["keys"][layer_idx, first_token : self.length].to(
                    device=self.device, non_blocking=self.device.type == "cuda"
                )
                if self._transfer_metrics is not None:
                    self._transfer_metrics.add_payload(
                        "indexer_host_to_device_bytes", boundary.numel() * boundary.element_size()
                    )
                keys = torch.cat((boundary, suffix), dim=0)
            if self.device.type == "cuda":
                from operators.nosa.indexer.offload_compression import append_compressed

                append_compressed(
                    keys,
                    cis,
                    **reservation.buffers,
                    compressed_start=old_count,
                    pooled_start=old_stable,
                )
            elif old_count < count:
                reservation.buffers["compressed_keys"][old_count:count].copy_(
                    keys.float().unfold(0, 32, 16).mean(-1).to(self.dtype)
                )
                reservation.buffers["compressed_cis"][old_count:count].copy_(
                    cis[first_token:].float().unfold(0, 32, 16).mean(-1).to(self.dtype)
                )
            if self.device.type != "cuda" and old_stable < stable:
                blocks = torch.arange(old_stable, stable, device=self.device)
                windows = blocks[:, None] * 4 + torch.arange(-1, 4, device=self.device)
                compressed_cis = reservation.buffers["compressed_cis"]
                pools = compressed_cis[windows.clamp_min(0)]
                pools = pools.masked_fill((windows < 0)[..., None], -torch.inf).amax(1)
                reservation.buffers["pooled_cis"][old_stable:stable].copy_(pools)
            self.indexer_cache.finish_layer(layer_idx)
        except BaseException:
            self.indexer_cache.abort_layer(layer_idx)
            raise
        return dict(reservation.views)

    def validate_commit(self):
        super().validate_commit()
        self.indexer_cache.validate_commit()

    def commit_step(self):
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self)
        self.validate_commit()
        # Direct cache initialization also preserves the invariant that every
        # committed append has resident compressed records and a checked prefix.
        for layer in range(self.spec.num_layers):
            self.prepare_indexer_inputs(layer)
        self.validate_commit()
        super().commit_step()
        self.indexer_cache.commit_step()

    def abort_step(self):
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self)
        super().abort_step()
        self.indexer_cache.abort_step()

    def truncate(self, length):
        if self._execution_resources is not None:
            self._execution_resources.check_mutation(self)
        # Validate and synchronize before either side publishes a shorter view.
        self._validate_truncate(length)
        self.synchronize()
        compressed, pooled = self._indexer_lengths(length)
        self.indexer_cache.truncate(
            length,
            {"compressed_keys": compressed, "compressed_cis": compressed, "pooled_cis": pooled},
        )
        self._length = length
        if length == 0:
            self._layer_states.clear()

    def reset(self):
        if self._execution_resources is not None:
            self._execution_resources.check_mutation(self)
        super().reset()
        self.indexer_cache.reset()

    def release(self):
        if self.released:
            return
        resources = self._execution_resources
        if resources is not None:
            resources.check_mutation(self, releasing=True)
        super().release()
        self.indexer_cache.release()
        self.cis_scores = None
        self._attention_workspace = None
        if self._transfer_metrics is not None:
            self._transfer_metrics.release()
            self._transfer_metrics = None
        if resources is not None:
            resources.detach(self)
            self._execution_resources = None

    def stats(self):
        result = super().stats()
        derived = self.indexer_cache.stats()
        result["capacity_bytes"] += self._cis_capacity_bytes + derived["capacity_bytes"]
        result["resident_bytes"] += derived["resident_bytes"]
        if self.cis_scores is not None:
            result["resident_bytes"] += self._bytes(self.cis_scores)
        if self._attention_workspace is not None:
            workspace_bytes = self._attention_workspace.capacity_bytes
            result["capacity_bytes"] += workspace_bytes
            result["resident_bytes"] += workspace_bytes
        if self._transfer_metrics is not None:
            result["capacity_bytes"] += self._transfer_metrics.storage_bytes
            result["resident_bytes"] += self._transfer_metrics.storage_bytes
        return result

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self.truncate(value)
