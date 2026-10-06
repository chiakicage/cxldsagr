"""Standalone complete DeepSeek V3.2 with layer placement across CUDA devices."""

from __future__ import annotations

import json
from contextlib import ExitStack, nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from models.deepseek_v32.checkpoint import CheckpointReader
from models.deepseek_v32.config import Config
from models.deepseek_v32.inputs import prepare_token_ids
from models.deepseek_v32.nonmatrix import residual_rms_norm


def layer_parameter_bytes(reader, layers):
    sizes = [0] * layers
    for name, metadata in reader.tensor_metadata.items():
        parts = name.split(".")
        if len(parts) > 3 and parts[:2] == ["model", "layers"]:
            layer = int(parts[2])
            if layer < layers:
                a, b = metadata["data_offsets"]
                sizes[layer] += b - a
    if not all(sizes):
        raise ValueError("checkpoint is missing one or more transformer layers")
    return sizes


def plan_layer_devices(sizes, devices, budgets):
    """Contiguous partitions minimizing the largest resident weight allocation."""
    n, groups = len(sizes), len(devices)
    if not 1 <= groups <= n or len(budgets) != groups:
        raise ValueError("one memory budget per nonempty device partition is required")
    prefix = [0]
    for size in sizes:
        prefix.append(prefix[-1] + size)
    dp = {(0, 0): (0, [])}
    for g in range(1, groups + 1):
        for end in range(g, n + 1):
            options = []
            for start in range(g - 1, end):
                previous = dp.get((g - 1, start))
                weight = prefix[end] - prefix[start]
                if previous is not None and weight <= budgets[g - 1]:
                    options.append((max(previous[0], weight), previous[1] + [(start, end)]))
            if options:
                dp[g, end] = min(options, key=lambda item: item[0])
    if (groups, n) not in dp:
        raise RuntimeError(
            "available GPU memory cannot hold the complete FP8 checkpoint plus reserve"
        )
    mapping = []
    for device, (start, stop) in zip(devices, dp[groups, n][1]):
        mapping.extend([device] * (stop - start))
    return mapping


class DeepSeekEchoModel:
    """Checkpoint transformer layers, embedding, final norm, and LM head.

    Weights remain on their assigned devices. Hidden states cross NVLink at
    placement boundaries; neither weights nor historical main KV are implicitly
    copied wholesale during an extend step. This is sequential layer placement,
    not tensor-parallel serving or continuous batching. By default all checkpoint
    layers execute; ``num_layers`` explicitly selects a contiguous prefix for
    diagnostics, with each layer consuming the preceding layer's real output.
    """

    def __init__(
        self,
        model_path,
        *,
        devices=(0, 1, 2, 6, 7),
        capacity=66560,
        offload=False,
        slots=16384,
        chunk_size=2048,
        extend_chunk_size=None,
        host_arena_tokens=None,
        workspace_query_tokens=None,
        hbm_cache_budget_bytes=None,
        dram_cache_budget_bytes=64 * 2**30,
        reserve_gib=5,
        num_layers=None,
    ):
        from models.deepseek_v32.layers import CheckpointBlock

        self.path = Path(model_path)
        self.cfg = Config.from_checkpoint(self.path)
        if offload and slots < min(self.cfg.index_topk, capacity):
            raise ValueError("sparse pool must fit one query's full exact selection")
        self.num_layers = self.cfg.num_hidden_layers if num_layers is None else num_layers
        if (
            not isinstance(self.num_layers, int)
            or isinstance(self.num_layers, bool)
            or not 1 <= self.num_layers <= self.cfg.num_hidden_layers
        ):
            raise ValueError("num_layers must select 1 through all checkpoint transformer layers")
        if capacity > self.cfg.max_seq_len:
            raise ValueError("requested context exceeds checkpoint max_position_embeddings")
        self.devices = [torch.device("cuda", int(index)) for index in devices]
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("devices must be distinct")
        self.reader = CheckpointReader(self.path)
        index = self.path / "model.safetensors.index.json"
        if index.is_file():
            expected = json.loads(index.read_text())["weight_map"]
            if self.num_layers < self.cfg.num_hidden_layers:
                prefixes = tuple(f"model.layers.{layer}." for layer in range(self.num_layers))
                expected = {
                    name: shard
                    for name, shard in expected.items()
                    if name.startswith(prefixes)
                    or name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
                }
            missing = [name for name in expected if name not in self.reader.tensor_files]
            if missing:
                raise ValueError(
                    f"incomplete checkpoint: {len(missing)} tensors missing; first={missing[0]}"
                )
        # Fail before allocating hundreds of GB if required endpoint tensors are absent.
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            if name not in self.reader.tensor_files:
                raise ValueError(f"incomplete checkpoint: missing {name}")
        sizes = layer_parameter_bytes(self.reader, self.num_layers)
        budgets = []
        for device in self.devices:
            with torch.cuda.device(device):
                if torch.cuda.get_device_capability(device)[0] != 9:
                    raise ValueError("this implementation requires Hopper SM90 GPUs")
                free, _ = torch.cuda.mem_get_info(device)
                budgets.append(free - max(int(reserve_gib * 2**30), hbm_cache_budget_bytes or 0))
        # Embedding and LM head remain BF16 on endpoint devices.
        for index, name in ((0, "model.embed_tokens.weight"), (-1, "lm_head.weight")):
            offsets = self.reader.tensor_metadata[name]["data_offsets"]
            budgets[index] -= offsets[1] - offsets[0]
        self.placement = plan_layer_devices(sizes, self.devices, budgets)
        if chunk_size < 1 or extend_chunk_size is not None and extend_chunk_size < 1:
            raise ValueError("chunk sizes must be positive")
        if offload and max(chunk_size, extend_chunk_size or 0) > slots:
            raise ValueError("query batch must fit the usable sparse pool")
        from models.deepseek_v32.execution.cache_resources import (
            execution_reservation,
            padded_tokens,
        )

        minimum_host = padded_tokens(capacity)
        if host_arena_tokens is not None and (
            type(host_arena_tokens) is not int
            or host_arena_tokens < minimum_host
            or host_arena_tokens % 64
        ):
            raise ValueError(
                "host_arena_tokens must be a 64-token multiple covering session capacity"
            )
        self.host_arena_tokens = host_arena_tokens or minimum_host
        required_queries = max(chunk_size, extend_chunk_size or 0)
        if workspace_query_tokens is not None and (
            type(workspace_query_tokens) is not int or workspace_query_tokens < required_queries
        ):
            raise ValueError(
                "workspace_query_tokens must cover configured prefill and extend chunks"
            )
        self.workspace_query_tokens = workspace_query_tokens or required_queries
        self.execution_reservation = execution_reservation(
            self.workspace_query_tokens,
            capacity,
            topk=self.cfg.index_topk,
            width=self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim,
        )
        self.extend_chunk_size = extend_chunk_size
        self.capacity, self.slots, self.chunk_size = capacity, slots, chunk_size
        self._shared_pools, self._shared_sessions = {}, {}
        self._pool_prefetch = {}
        self._compute_graphs = None
        self._graph_reservation_bytes = 0
        self.cache_method = "echo" if offload else "hbm"
        self._cache_generation = 0
        self._poisoned = False
        self._closed = False
        self.hbm_cache_budget_bytes = (
            int(reserve_gib * 2**30) if hbm_cache_budget_bytes is None else hbm_cache_budget_bytes
        )
        self.dram_cache_budget_bytes = dram_cache_budget_bytes
        if any(
            type(value) is not int or value < 1
            for value in (self.hbm_cache_budget_bytes, self.dram_cache_budget_bytes)
        ):
            raise ValueError("cache HBM and DRAM budgets must be positive bytes")
        self._cache_resource_plan = self._plan_cache_resources(offload)
        caches = self._allocate_shared_caches() if offload else {}
        self.blocks = []
        self.embedding_weight = self.reader.get_tensor("model.embed_tokens.weight").to(
            self.devices[0]
        )
        self.final_norm = self.reader.get_tensor("model.norm.weight").to(self.devices[-1])
        self.head_weight = self.reader.get_tensor("lm_head.weight").to(self.devices[-1])
        for layer, device in enumerate(self.placement):
            with torch.cuda.device(device):
                block = CheckpointBlock(
                    self.path,
                    layer,
                    device,
                    reader=self.reader,
                    capacity=capacity,
                    offload=offload,
                    slots=slots,
                    chunk_size=chunk_size,
                    cache=caches.get(layer),
                )
                self.blocks.append(block)
            print(f"loaded layer {layer + 1}/{self.num_layers} on {device}", flush=True)
        self.length = 0
        self.offload = offload
        self.synchronize()

    def _plan_cache_resources(self, offload):
        """Check each device ledger before allocating any model cache tensor."""
        from cache.prefix_pool import CacheBudgetExceeded
        from cache.sparse_token_pool import SharedSparseTokenPool
        from models.deepseek_v32.execution.cache_resources import dense_ticket_reservation

        if offload and self.slots < min(self.cfg.index_topk, self.capacity):
            raise ValueError("sparse pool must fit one query's full exact selection")
        if (
            offload
            and max(
                self.execution_reservation.query_tokens,
                getattr(self, "chunk_size", 0),
                getattr(self, "extend_chunk_size", None) or 0,
            )
            > self.slots
        ):
            raise ValueError("query batch and workspace must fit the usable sparse pool")
        width = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim
        by_device, total_dram = {}, 0
        for device in self.devices:
            layers = self.placement.count(device)
            if not layers:
                continue
            index = layers * (self.capacity * (self.cfg.index_head_dim + 4) + 3 * 64)
            if offload:
                shared = SharedSparseTokenPool.estimate_shared_bytes(
                    self.host_arena_tokens, width, layers, self.slots, device=device
                )
                session = SharedSparseTokenPool.estimate_session_bytes(
                    self.capacity, layers=layers, device=device
                )
                metadata_workspace = SharedSparseTokenPool.estimate_execution_workspace_bytes(
                    self.host_arena_tokens, self.slots
                )
                # Cache methods can switch after allocation planning. Reserve
                # the dense lookahead bound for every offload method here.
                ticket_workspace = dense_ticket_reservation(
                    min(self.capacity, self.slots), layers, device
                )
                hbm = shared["hbm"] + session["hbm"] + metadata_workspace + ticket_workspace + index
                dram = shared["dram"] + session["dram"]
            else:
                hbm = layers * self.capacity * (width * 2 + 20) + index
                dram, metadata_workspace, ticket_workspace = 0, 0, 0
            hbm += self.execution_reservation.hbm
            hbm += getattr(self, "_graph_reservation_bytes", 0)
            dram += self.execution_reservation.dram
            if hbm > self.hbm_cache_budget_bytes:
                raise CacheBudgetExceeded(
                    f"{device} cache needs {hbm} bytes including execution reservation; "
                    f"budget is {self.hbm_cache_budget_bytes}"
                )
            by_device[str(device)] = {
                "hbm": hbm,
                "dram": dram,
                "layers": layers,
                "metadata_workspace_bytes": metadata_workspace,
                "dense_ticket_workspace_bytes": ticket_workspace,
                "indexer_workspace_bytes": self.execution_reservation.indexer_bytes,
                "copy_source_bytes": self.execution_reservation.copy_source_bytes,
                "compute_graph_reservation_bytes": getattr(self, "_graph_reservation_bytes", 0),
                **self.execution_reservation.cpu_workspace_metadata,
            }
            total_dram += dram
        if total_dram > self.dram_cache_budget_bytes:
            raise CacheBudgetExceeded(
                f"cache arenas and host metadata need {total_dram} DRAM bytes; "
                f"budget is {self.dram_cache_budget_bytes}"
            )
        return {
            "devices": by_device,
            "dram_bytes": total_dram,
            "hbm_budget_per_device": self.hbm_cache_budget_bytes,
            "dram_budget": self.dram_cache_budget_bytes,
        }

    def _allocate_shared_caches(self, *, dense_contiguous=False):
        from cache.sparse_token_pool import SharedSparseTokenPool
        from operators.deepseek_v32.indexer import cache_ops

        self._check_usable()
        caches = {}
        try:
            for device in self.devices:
                layers = [layer for layer, placed in enumerate(self.placement) if placed == device]
                if not layers:
                    continue
                pool = SharedSparseTokenPool(
                    self.host_arena_tokens,
                    self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim,
                    len(layers),
                    self.slots,
                    device=device,
                    metadata_ops=cache_ops,
                    dense_contiguous=dense_contiguous,
                )
                self._shared_pools[device] = pool
                session = pool.allocate_session(self.capacity)
                self._shared_sessions[device] = session
                for local, layer in enumerate(layers):
                    caches[layer] = session.layer(local)
        except BaseException as error:
            try:
                self._release_shared_caches()
            except BaseException as cleanup_error:  # noqa: BLE001 -- preserve allocation and cleanup
                self._poisoned = True
                raise BaseExceptionGroup(
                    "DeepSeek cache allocation and cleanup both failed", [error, cleanup_error]
                ) from None
            raise
        return caches

    def _release_shared_caches(self):
        errors = []
        for device, helper in tuple(getattr(self, "_pool_prefetch", {}).items()):
            try:
                helper.close()
            except BaseException as error:  # noqa: BLE001 -- retain pending copy owners
                errors.append(error)
            else:
                del self._pool_prefetch[device]
        if errors:
            self._poisoned = True
            raise BaseExceptionGroup("DeepSeek history prefetch cleanup failed", errors)
        for device, session in tuple(self._shared_sessions.items()):
            try:
                session.release()
            except BaseException as error:  # noqa: BLE001 -- release independent devices
                errors.append(error)
            else:
                del self._shared_sessions[device]
        for device, pool in tuple(self._shared_pools.items()):
            if device in self._shared_sessions:
                continue  # A failed session release still owns this pool.
            try:
                pool.close()
            except BaseException as error:  # noqa: BLE001 -- retain every failed owner
                errors.append(error)
            else:
                del self._shared_pools[device]
        if errors:
            self._poisoned = True
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("DeepSeek shared cache cleanup failed", errors)

    def synchronize(self):
        errors = []
        for device in self.devices:
            try:
                torch.cuda.synchronize(device)
            except BaseException as error:  # noqa: BLE001 -- drain every independent device
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("DeepSeek device synchronization failed", errors)

    def _check_usable(self):
        if getattr(self, "_poisoned", False):
            raise RuntimeError("model is poisoned after an asynchronous CUDA failure")
        if getattr(self, "_closed", False):
            raise RuntimeError("model is closed")

    def set_cache_mode(self, offload, *, dense_contiguous=False):
        from models.deepseek_v32.attention import EchoAttentionRunner

        self._check_usable()
        if dense_contiguous and not offload:
            raise ValueError("contiguous dense history requires offloaded cache storage")
        planned = self._plan_cache_resources(offload)
        try:
            self.synchronize()
        except BaseException:
            self._poisoned = True
            raise
        attentions = [block.attention.attention for block in self.blocks]
        self._release_shared_caches()
        # Drop every old cache before the new plan's buffers are allocated.
        for block in self.blocks:
            block.attention = block.cache = None
        self._cache_resource_plan = planned
        self._cache_generation += 1
        try:
            caches = (
                self._allocate_shared_caches(dense_contiguous=True)
                if dense_contiguous
                else self._allocate_shared_caches()
                if offload
                else {}
            )
        except BaseException:
            # Allocation already attempted its own cleanup; never retry it here.
            self._poisoned = True
            raise
        try:
            for layer, block in enumerate(self.blocks):
                attention = attentions[layer]
                device = attention.device
                with torch.cuda.device(device):
                    block.attention = EchoAttentionRunner(
                        attention,
                        self.capacity,
                        offload=offload,
                        slots=self.slots,
                        chunk_size=self.chunk_size,
                        cache=caches.get(layer),
                    )
                    block.cache = block.attention.cache
        except BaseException as error:
            self._poisoned = True
            try:
                self._release_shared_caches()
            except BaseException as cleanup_error:  # noqa: BLE001 -- preserve replacement failure
                raise BaseExceptionGroup(
                    "DeepSeek cache replacement and cleanup both failed", [error, cleanup_error]
                ) from None
            raise
        self.length, self.offload = 0, offload
        self.cache_method = "echo" if offload else "hbm"

    def set_cache_method(self, method):
        """Create an independent empty cache for one of the four local methods."""
        if method not in ("hbm", "echo", "serial_sparse", "dense_prefetch"):
            raise ValueError("unsupported DeepSeek cache method")
        if method == "dense_prefetch" and self.slots < self.capacity:
            raise ValueError("persistent dense prefetch requires P >= session capacity")
        if method == "dense_prefetch":
            self.set_cache_mode(True, dense_contiguous=True)
        else:
            self.set_cache_mode(method != "hbm")
        try:
            for block in self.blocks:
                block.attention.fused_prefetch = method == "echo"
            if method == "dense_prefetch":
                from models.deepseek_v32.cache.prefetch import PoolHistoryPrefetch

                for device in self._shared_pools:
                    self._pool_prefetch[device] = PoolHistoryPrefetch(device)
        except BaseException:
            self._poisoned = True
            raise
        self.cache_method = method

    def prepare_compute_graphs(self, query_sizes):
        """Capture dense three-layer compute once, independently of cache policy.

        This bank uses the same pure compute islands as the serving backend.
        Cache, exact selection, IO and transaction handling remain eager.
        """
        self._check_usable()
        from models.deepseek_v32.execution.compute_graphs import (
            DEFAULT_PRIVATE_LIMIT_BYTES,
            DeepSeekComputeGraphs,
            plan_compute_graphs,
        )

        if self._compute_graphs is not None:
            raise RuntimeError("compute graphs require a healthy model without an existing bank")
        if len(self.devices) != 1 or self.num_layers > 3 or any(b.is_moe for b in self.blocks):
            raise ValueError("model compute graphs support a single-device dense layer prefix 0-2")
        queries = sorted(set(query_sizes))
        if not queries or any(type(q) is not int or not 1 <= q <= self.capacity for q in queries):
            raise ValueError("graph query sizes must be positive integers within capacity")
        plans = [
            plan_compute_graphs(self.cfg, len(self.blocks), q, q, q, DEFAULT_PRIVATE_LIMIT_BYTES)
            for q in queries
        ]
        metadata = dict(plans[0])
        metadata["compute_graph_query_sizes"] = queries
        for name in (
            "compute_graph_count",
            "compute_graph_static_storage_bytes",
            "compute_graph_static_allocation_limit_bytes",
        ):
            metadata[name] = sum(plan[name] for plan in plans)
        metadata["compute_graph_reserved_limit_bytes"] = (
            metadata["compute_graph_static_allocation_limit_bytes"] + DEFAULT_PRIVATE_LIMIT_BYTES
        )
        self._graph_reservation_bytes = metadata["compute_graph_reserved_limit_bytes"]
        try:
            planned = self._plan_cache_resources(self.offload)
        except BaseException:
            self._graph_reservation_bytes = 0
            raise
        self._cache_resource_plan = planned
        bank = DeepSeekComputeGraphs(
            [block.attention.attention for block in self.blocks],
            self.blocks,
            self.devices[0],
            metadata,
        )
        # Retain partial captures if initialization or its asynchronous work fails.
        self._compute_graphs = bank
        try:
            with torch.cuda.device(self.devices[0]):
                bank.allocate()
        except BaseException:
            self._poisoned = True
            raise

    def evict_prefix_residency(self):
        """Reset only main-KV residency for an explicitly cold extend experiment."""
        self._check_usable()
        if not self.offload:
            raise RuntimeError("cold residency requires a healthy offload model")
        if any(block.cache._step_end is not None for block in self.blocks):
            raise RuntimeError("cannot evict residency during an active cache step")
        try:
            self.synchronize()
            for block in self.blocks:
                cache = block.cache
                with torch.cuda.device(cache.device), cache.operation():
                    slots = torch.arange(1, cache.slots + 1, device=cache.device)
                    cache._evict(slots, count_eviction=False)
                    cache.reset_stats()
            self.synchronize()
        except BaseException:
            self._poisoned = True
            raise

    def close(self):
        """Drain graph/IO owners before releasing cache storage; retain on failure."""
        if getattr(self, "_closed", False):
            return
        try:
            self.synchronize()
            bank = getattr(self, "_compute_graphs", None)
            if bank is not None:
                bank.close()
                self._compute_graphs = None
            self._release_shared_caches()
        except BaseException:
            self._poisoned = True
            raise
        for block in self.blocks:
            block.cache = block.attention = None
        self._closed = True

    @torch.inference_mode()
    def forward(self, token_ids, *, scope=None, all_logits=False, return_hidden=False):
        self._check_usable()
        bank = getattr(self, "_compute_graphs", None)
        try:
            with bank.execution() if bank is not None else nullcontext(), ExitStack() as hints:
                for block in self.blocks:
                    defer = getattr(
                        getattr(block, "attention", None), "defer_unused_prefill_hints", None
                    )
                    if defer is not None:
                        hints.enter_context(defer())
                return self._forward(
                    token_ids, scope=scope, all_logits=all_logits, return_hidden=return_hidden
                )
        finally:
            if bank is not None and bank.failed:
                self._poisoned = True

    def _forward(self, token_ids, *, scope=None, all_logits=False, return_hidden=False):
        """Execute a cache step, optionally returning every token's normalized hidden.

        The default returns last-token logits, or all logits with ``all_logits``.
        ``return_hidden=True`` returns ``{"hidden": ..., "logits": ...}``, where
        hidden includes all input tokens after the final RMSNorm. For a truncated
        diagnostic model these are outputs after its selected transformer prefix,
        not the full checkpoint's hidden states or language-model predictions.
        """
        self._check_usable()
        scope = scope or (lambda _: nullcontext())
        ids = prepare_token_ids(
            token_ids,
            device=self.devices[0],
            vocab_size=self.cfg.vocab_size,
            max_tokens=self.capacity - self.length,
        )
        query_chunk = (
            self.chunk_size if not self.length else getattr(self, "extend_chunk_size", None)
        ) or len(ids)
        if query_chunk > getattr(self, "workspace_query_tokens", query_chunk):
            raise ValueError("query batch exceeds reserved workspace_query_tokens")
        if getattr(self, "offload", False) and query_chunk > self.slots:
            raise ValueError("query batch exceeds usable pool; configure extend_chunk_size")
        started = []
        offsets = []
        try:
            for block in self.blocks:
                offsets.append(
                    block.attention.offset.clone()
                    if hasattr(block, "attention")
                    and getattr(block.attention, "mutates_prefetch_hint", True)
                    else None
                )
            for block in self.blocks:
                block.cache.begin_step(ids.numel())
                started.append(block.cache)
            logits_parts = []
            hidden_parts = []
            for chunk_start in range(0, ids.numel(), query_chunk):
                chunk_stop = min(ids.numel(), chunk_start + query_chunk)
                with torch.cuda.device(self.devices[0]), scope("embedding"):
                    hidden = F.embedding(ids[chunk_start:chunk_stop], self.embedding_weight)
                    residual = None
                tickets = {}
                helpers = getattr(self, "_pool_prefetch", {})
                if helpers:
                    for device, helper in helpers.items():
                        first = self.placement.index(device)
                        with (
                            torch.cuda.device(device),
                            scope(f"dense_history_prefetch_layer_{first}"),
                        ):
                            tickets[first] = helper.prefetch(self.blocks[first].cache)
                for layer, (block, device) in enumerate(zip(self.blocks, self.placement)):
                    with torch.cuda.device(device):
                        helper = helpers.get(device)
                        if helper is not None:
                            with scope(f"dense_history_wait_layer_{layer}"):
                                helper.wait(tickets.pop(layer))
                            next_layer = next(
                                (
                                    i
                                    for i in range(layer + 1, len(self.blocks))
                                    if self.placement[i] == device
                                ),
                                None,
                            )
                            if next_layer is not None:
                                with scope(f"dense_history_prefetch_layer_{next_layer}"):
                                    tickets[next_layer] = helper.prefetch(
                                        self.blocks[next_layer].cache
                                    )
                        with scope("hidden_transfer"):
                            hidden = hidden.to(device, non_blocking=True)
                            if residual is not None:
                                residual = residual.to(device, non_blocking=True)
                        # Outer scheduling owns the full layer batch, including
                        # norm/MLP geometry for independently configured extend.
                        block.chunk_size = len(hidden)
                        with scope(f"layer_{layer}"):
                            bank = getattr(self, "_compute_graphs", None)
                            if bank is not None and bank.supports(layer, hidden, residual):
                                hidden, residual = bank.forward_block(
                                    layer, block.attention, hidden, residual, scope=scope
                                )
                            else:
                                if bank is not None:
                                    bank.eager_fallbacks += 1
                                hidden, residual = block.forward(hidden, residual, scope=scope)
                if return_hidden or all_logits or chunk_stop == ids.numel():
                    with torch.cuda.device(self.devices[-1]), scope("final_norm_lm_head"):
                        hidden = hidden.to(self.devices[-1], non_blocking=True)
                        residual = residual.to(self.devices[-1], non_blocking=True)
                        all_hidden = return_hidden or all_logits
                        selected = hidden if all_hidden else hidden[-1:]
                        selected_residual = residual if all_hidden else residual[-1:]
                        normalized, _ = residual_rms_norm(
                            selected,
                            selected_residual,
                            self.final_norm,
                            self.cfg.norm_eps,
                        )
                        if return_hidden:
                            hidden_parts.append(normalized)
                        if all_logits or chunk_stop == ids.numel():
                            head_input = normalized if all_logits else normalized[-1:]
                            logits_parts.append(F.linear(head_input, self.head_weight).float())
                for helper in helpers.values():
                    helper.drain()
            output = logits_parts[0] if len(logits_parts) == 1 else torch.cat(logits_parts)
            if return_hidden:
                output = {
                    "hidden": hidden_parts[0]
                    if len(hidden_parts) == 1
                    else torch.cat(hidden_parts),
                    "logits": output,
                }
            # Synchronization surfaces asynchronous execution failures before
            # advancing valid length for any layer.
            self.synchronize()
            if any(
                cache._step_end is None or cache.written != cache._step_end for cache in started
            ):
                raise RuntimeError("one or more model layers did not finish the cache step")
            for block in self.blocks:
                block.cache.commit()
            self.length += ids.numel()
            return output
        except BaseException as error:
            try:
                self.synchronize()
            except BaseException as synchronization_error:  # noqa: BLE001 -- retain unsafe backing
                self._poisoned = True
                raise BaseExceptionGroup(
                    "DeepSeek model execution and synchronization both failed",
                    [error, synchronization_error],
                ) from None
            cleanup_errors = []
            for helper in getattr(self, "_pool_prefetch", {}).values():
                try:
                    helper.drain()
                except BaseException as drain_error:  # noqa: BLE001 -- preserve all cleanup errors
                    cleanup_errors.append(drain_error)
            for cache in started:
                if cache._step_end is not None:
                    try:
                        cache.rollback()
                    except BaseException as rollback_error:  # noqa: BLE001 -- attempt every layer
                        cleanup_errors.append(rollback_error)
            for block, offset in zip(self.blocks, offsets):
                if offset is not None:
                    try:
                        block.attention.offset.copy_(offset)
                    except BaseException as restore_error:  # noqa: BLE001 -- preserve body failure
                        cleanup_errors.append(restore_error)
            if cleanup_errors:
                self._poisoned = True
                raise BaseExceptionGroup(
                    "DeepSeek model execution and rollback both failed", [error, *cleanup_errors]
                ) from None
            raise

    def snapshot_prefix(self):
        """Snapshot shared pools once; diagnostic CPU storage is not serving capacity."""
        self._check_usable()
        self.synchronize()
        return {
            "schema": "echo-shared-prefix-v1",
            "generation": self._cache_generation,
            "length": self.length,
            "offload": self.offload,
            "pools": {str(device): pool.snapshot() for device, pool in self._shared_pools.items()},
            "offsets": [block.attention.offset.cpu().clone() for block in self.blocks],
            # Resident records retain their unchanged prefix; only lengths rewind.
            "resident_lengths": [block.cache.length for block in self.blocks],
        }

    def restore_prefix(self, state):
        self._check_usable()
        if (
            state.get("schema") != "echo-shared-prefix-v1"
            or state["generation"] != self._cache_generation
            or state["offload"] != self.offload
            or len(state["offsets"]) != len(self.blocks)
        ):
            raise ValueError("prefix snapshot does not match the current model cache")
        self.synchronize()
        for device, pool in self._shared_pools.items():
            pool.restore(state["pools"][str(device)])
        for block, length, offset in zip(
            self.blocks, state["resident_lengths"], state["offsets"], strict=True
        ):
            if not self.offload:
                block.cache.truncate(length)
            block.cache.reset_stats()
            block.attention.offset.copy_(offset)
        self.length = state["length"]
        self.synchronize()
