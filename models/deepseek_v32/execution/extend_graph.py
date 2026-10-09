"""One complete CUDA Graph for a fixed-prefix DeepSeek extend.

The graph owns all GPU computation and IO. Validation, cache transaction start,
synchronization and host commit remain in the model. Its captured scalar clocks
and shapes require restoration of the matching prefix before every replay.
"""

from contextlib import ExitStack, nullcontext
from copy import copy
from time import perf_counter

import torch

from cache.prefix_pool import CacheBudgetExceeded
from models.deepseek_v32.execution.compute_graphs import _precision_policy

EXTEND_GRAPH_POLICY_REVISION = "deepseek-full-extend-graph-v2-dense-late-wait"
DEFAULT_PRIVATE_LIMIT_BYTES = 2 * 2**30
_CACHE_FIELDS = ("length", "written", "indexer_visible_end", "_step_end", "_transient_start")


def _tensor_identity(tensor):
    # Inference tensors have no version counter; storage identity still binds them.
    version = None if torch.is_inference(tensor) else tensor._version
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        version,
    )


def _weight_identity(model):
    """Bind only immutable compute objects, excluding cache and graph state."""
    seen = set()
    result = []

    def visit(value, path):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            result.append((path, _tensor_identity(value)))
        elif isinstance(value, dict):
            for key, child in sorted(value.items(), key=lambda item: str(item[0])):
                visit(child, f"{path}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{path}.{index}")
        elif hasattr(value, "__dict__"):
            for name, child in sorted(vars(value).items()):
                if name in ("reader", "cfg", "config", "precision"):
                    continue
                visit(child, f"{path}.{name}")

    for name in ("embedding_weight", "final_norm", "head_weight"):
        visit(getattr(model, name), name)
    for index, block in enumerate(model.blocks):
        visit(block.attention.attention, f"layer.{index}.attention")
        stream = getattr(block.attention.attention, "_q1_projection_stream", None)
        if stream is not None:
            result.append((f"layer.{index}.projection_stream", stream.cuda_stream))
        visit(block.post_norm_weight, f"layer.{index}.post_norm")
        visit(block.mlp, f"layer.{index}.mlp")
    return tuple(result)


def _cache_identity(model):
    return tuple(
        (
            id(block),
            id(block.cache),
            id(block.attention),
            tuple(
                _tensor_identity(tensor)[:-1]
                for tensor in (
                    block.cache.records,
                    block.cache.host_to_device,
                    block.cache.device_to_host,
                    block.attention.index_keys,
                    block.attention.index_scales,
                    block.attention.offset,
                )
            ),
            block.attention.fused_prefetch,
        )
        for block in model.blocks
    )


class _ResidentRecipe:
    """Host-only append bookkeeping for a non-offloaded cache."""

    def __init__(self, cache):
        self.cache = cache
        self.before = {name: getattr(cache, name) for name in _CACHE_FIELDS}
        self.before_stats = copy(cache.stats)

    def finish(self):
        self.after = {name: getattr(self.cache, name) for name in _CACHE_FIELDS}
        self.after_stats = copy(self.cache.stats)
        self.restore()
        return self

    def restore(self):
        for name, value in self.before.items():
            setattr(self.cache, name, value)
        self.cache.stats = copy(self.before_stats)

    def validate(self, *, pending=True):
        expected = dict(self.before)
        if not pending:
            expected["_step_end"] = None
        if any(getattr(self.cache, name) != value for name, value in expected.items()):
            raise RuntimeError("resident extend graph pending append range changed")

    def apply(self):
        self.validate()
        for name, value in self.after.items():
            setattr(self.cache, name, value)
        for name, value in vars(self.after_stats).items():
            current = getattr(self.cache.stats, name)
            setattr(
                self.cache.stats,
                name,
                max(current, value)
                if name == "max_working_set"
                else current + value - getattr(self.before_stats, name),
            )

    def close(self):
        self.cache = None


class DeepSeekExtendGraph:
    """Complete fixed-prefix graph with borrowed outputs and explicit host effects."""

    def __init__(self, model, queries, all_logits=False, return_hidden=False):
        self.model = model
        self.device = model.devices[0]
        self.queries = queries
        self.all_logits, self.return_hidden = bool(all_logits), bool(return_hidden)
        self.history = model.length
        self.generation = model._cache_generation
        self.method = model.cache_method
        self.dense_history_after_selection = (
            self.method == "dense_prefetch" and len(model.blocks) == 3
        )
        self.geometry = (
            model.capacity,
            model.slots,
            model.workspace_query_tokens,
            model.extend_chunk_size,
            tuple(model.placement),
        )
        self.private_limit_bytes = DEFAULT_PRIVATE_LIMIT_BYTES
        self.static_limit_bytes = ((queries * 8 + 511) // 512) * 512
        self.reservation_bytes = self.private_limit_bytes + self.static_limit_bytes
        self.graph = self.static_ids = self.output = None
        self.last_hidden = self.last_residual = None
        self.saved_offsets = []
        self.recipes = []
        self.capture_owners = []
        self.stream = None
        self.allocated = self.closed = self.failed = self.pending = False
        self.replays = 0
        self.private_reserved_bytes = self.static_allocated_bytes = 0
        self.setup_seconds = None
        self.memory_at_allocation = None
        self.precision = self.weights = self.cache_identity = None

    def _check(self):
        if self.closed or self.failed:
            raise RuntimeError("complete extend graph is closed or failed")
        self.model._check_usable()

    def validate(self):
        self._check()
        model = self.model
        if not self.allocated or self.pending:
            raise RuntimeError("complete extend graph is unprepared or has an unfinished replay")
        if (
            model.length != self.history
            or model._cache_generation != self.generation
            or model.cache_method != self.method
            or (
                model.capacity,
                model.slots,
                model.workspace_query_tokens,
                model.extend_chunk_size,
                tuple(model.placement),
            )
            != self.geometry
            or _cache_identity(model) != self.cache_identity
        ):
            raise ValueError("complete extend graph prefix, geometry or cache storage changed")
        if _precision_policy() != self.precision or _weight_identity(model) != self.weights:
            raise ValueError("complete extend graph precision or weights changed")
        if any(block.cache._step_end is not None for block in model.blocks):
            raise RuntimeError("complete extend graph requires a committed prefix before begin")
        for recipe in self.recipes:
            recipe.validate(pending=False)

    @torch.inference_mode()
    def allocate(self, ids, *, capture_scope=None):
        self._check()
        if self.allocated or self.graph is not None:
            raise RuntimeError("complete extend graph has already been prepared")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("extend graph preparation cannot be nested inside CUDA capture")
        started = perf_counter()
        model = self.model
        scope = capture_scope or (lambda _: nullcontext())
        snapshot = model.snapshot_prefix()
        try:
            # Compile and warm the same direct computation that capture will use.
            for _ in range(3):
                model.restore_prefix(snapshot)
                model._forward(
                    ids,
                    all_logits=self.all_logits,
                    return_hidden=self.return_hidden,
                    use_extend_graph=False,
                    use_compute_graphs=False,
                    dense_history_after_selection=self.dense_history_after_selection,
                )
            model.restore_prefix(snapshot)
            free, _ = torch.cuda.mem_get_info(self.device)
            if self.reservation_bytes > free:
                raise CacheBudgetExceeded("complete extend graph allocation bound exceeds free HBM")
            self.static_ids = ids.clone()
            self.stream = torch.cuda.Stream(device=self.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            self.graph = torch.cuda.CUDAGraph()
            resident = []
            for block in model.blocks:
                block.cache.begin_step(self.queries)
                if not block.cache.offload:
                    resident.append(_ResidentRecipe(block.cache))
            retained = {}
            with torch.cuda.stream(self.stream), ExitStack() as owners:
                pool_scopes = [
                    owners.enter_context(pool.graph_capture(model._shared_sessions[device]))
                    for device, pool in model._shared_pools.items()
                ]
                dense_scopes = [
                    owners.enter_context(helper.graph_capture())
                    for helper in model._pool_prefetch.values()
                ]
                self.capture_owners.extend((*pool_scopes, *dense_scopes))
                with torch.cuda.graph(self.graph, stream=self.stream), scope("extend_graph_body"):
                    with scope("extend_graph_rollback_backup"):
                        self.saved_offsets = [
                            block.attention.offset.clone()
                            if block.attention.mutates_prefetch_hint
                            else None
                            for block in model.blocks
                        ]
                    self.output = model._execute_gpu_body(
                        self.static_ids,
                        self.queries,
                        scope=scope,
                        all_logits=self.all_logits,
                        return_hidden=self.return_hidden,
                        use_compute_graphs=False,
                        dense_history_after_selection=self.dense_history_after_selection,
                        retained=retained,
                    )
                    for owner in pool_scopes:
                        owner.join()
                for owner in pool_scopes:
                    self.recipes.append(owner.finish())
                for owner in dense_scopes:
                    owner.finish()
                self.recipes.extend(recipe.finish() for recipe in resident)
            self.last_hidden, self.last_residual = retained["hidden"], retained["residual"]
            # Capture submits no kernels. Only its temporary Python step state
            # needs cancellation before the normal snapshot restore can run.
            for block in model.blocks:
                cache = block.cache
                cache._step_end = None
                cache.written = cache.indexer_visible_end = cache.length
            model.restore_prefix(snapshot)
            self.precision = _precision_policy()
            self.weights = _weight_identity(model)
            self.cache_identity = _cache_identity(model)
            self._audit_capacity()
            self.allocated = True
        except BaseException:
            self.failed = True
            raise
        finally:
            self.setup_seconds = perf_counter() - started

    def replay(self, ids, *, scope=None):
        self._check()
        if not self.allocated or self.pending:
            raise RuntimeError("complete extend graph cannot replay in its current state")
        if (
            ids.shape != self.static_ids.shape
            or ids.dtype != torch.long
            or ids.device != self.device
        ):
            raise ValueError("complete extend graph token input shape, dtype or device changed")
        for recipe in self.recipes:
            recipe.validate()
        scope = scope or (lambda _: nullcontext())
        try:
            with scope("extend_graph_inputs"):
                self.static_ids.copy_(ids)
            self.pending = True
            with scope(f"extend_graph_replay_q_{len(ids)}"):
                self.graph.replay()
            self.replays += 1
            return self.output
        except BaseException:
            self.failed = True
            raise

    def apply(self):
        """Called only after the model has synchronized the complete replay."""
        self._check()
        if not self.pending:
            raise RuntimeError("complete extend graph has no replay to finish")
        try:
            for recipe in self.recipes:
                recipe.validate()
            for recipe in self.recipes:
                recipe.apply()
        except BaseException:
            self.failed = True
            raise
        self.pending = False

    def _audit_capacity(self):
        pool = tuple(self.graph.pool())
        segments = [
            item for item in torch.cuda.memory_snapshot() if item["device"] == self.device.index
        ]
        graph_segments = [item for item in segments if tuple(item["segment_pool_id"]) == pool]
        if not graph_segments:
            raise RuntimeError("complete extend graph private allocation pool was not found")
        self.private_reserved_bytes = sum(item["total_size"] for item in graph_segments)
        address = self.static_ids.untyped_storage().data_ptr()
        allocations = [
            block["size"]
            for segment in segments
            for block in segment["blocks"]
            if block["address"] == address and block["state"] == "active_allocated"
        ]
        if len(allocations) != 1:
            raise RuntimeError("complete extend graph static input allocation was not found")
        self.static_allocated_bytes = allocations[0]
        if (
            self.private_reserved_bytes > self.private_limit_bytes
            or self.static_allocated_bytes > self.static_limit_bytes
        ):
            raise CacheBudgetExceeded("complete extend graph exceeded its allocation bound")
        free, total = torch.cuda.mem_get_info(self.device)
        self.memory_at_allocation = {
            "pytorch_allocated": torch.cuda.memory_allocated(self.device),
            "pytorch_reserved": torch.cuda.memory_reserved(self.device),
            "device_used": total - free,
            "device_total": total,
        }

    def describe(self):
        sources = {name: 0 for name in ("hbm", "dram")}
        for recipe in self.recipes:
            for name, value in getattr(recipe, "source_bytes", {}).items():
                sources[name] += value
        return {
            "enabled": True,
            "policy_revision": EXTEND_GRAPH_POLICY_REVISION,
            "history_tokens": self.history,
            "query_tokens": self.queries,
            "cache_method": self.method,
            "q1_projection_schedule": (
                "query-and-kv-index-key-branches-v1"
                if self.queries == 1
                and all(
                    getattr(block.attention.attention, "_q1_projection_stream", None) is not None
                    for block in self.model.blocks
                )
                else None
            ),
            "dense_history_wait": (
                "after_indexer_topk_before_main_kv_append"
                if self.dense_history_after_selection
                else "before_layer"
                if self.method == "dense_prefetch"
                else None
            ),
            "cache_generation": self.generation,
            "all_logits": self.all_logits,
            "return_hidden": self.return_hidden,
            "allocated": self.allocated,
            "replays": self.replays,
            "graph_count": 1,
            "setup_seconds": self.setup_seconds,
            "static_allocated_bytes": self.static_allocated_bytes,
            "private_reserved_bytes": self.private_reserved_bytes,
            "chosen_private_limit_bytes": self.private_limit_bytes,
            "reservation_bytes": self.reservation_bytes,
            "retained_writeback_source_bytes": sources,
            "writeback_source_accounting": "included in the graph private pool, not added twice",
            "memory_at_allocation": self.memory_at_allocation,
            "captured_precision_policy": self.precision,
            "scope": "embedding, all selected layers including cache and IO, final norm and LM head",
            "replay_contract": "restore matching fixed prefix before each replay; outputs borrowed",
        }

    def close(self):
        if self.closed:
            return
        try:
            torch.cuda.synchronize(self.device)
        except BaseException:
            self.failed = True
            raise
        # The graph must disappear before its captured writeback owners do.
        if self.graph is not None:
            self.graph.reset()
        self.graph = None
        self.output = self.last_hidden = self.last_residual = self.static_ids = None
        self.saved_offsets.clear()
        for recipe in self.recipes:
            recipe.close()
        self.recipes.clear()
        self.capture_owners.clear()
        self.stream = None
        self.pending = False
        self.closed = True
