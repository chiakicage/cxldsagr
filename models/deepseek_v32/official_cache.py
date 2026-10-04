"""Execute pinned ECHO cache methods against the motivation storage contract.

The selected upstream definitions are compiled without rewriting their bodies.
SGLang's scheduler and model imports are unnecessary for this serial adapter.
History host IDs use the existing zero-based arena; negative selections are
filtered before upstream recall, and transient candidates never receive IDs.
"""

import ast
import hashlib
import importlib.util
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Optional

import torch

from cache.sparse_token_cache import MISSING, SparseTokenCache, WorkingSetTooLarge
from cache.sparse_token_pool import PAGE_SIZE, _tensor_bytes

_ROOT = Path(__file__).resolve().parents[2] / "3rdparty" / "ECHO"
_MODULES = {}


def _selected_module(path, names, namespace, *, class_name=None):
    """Load exact AST nodes while retaining upstream filenames and line numbers."""
    source = path.read_text()
    parsed = ast.parse(source, filename=str(path))
    nodes = parsed.body
    if class_name is not None:
        nodes = next(
            node.body
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    selected = [node for node in nodes if getattr(node, "name", None) in names]
    found = {node.name for node in selected}
    if found != set(names):
        raise RuntimeError(f"missing pinned ECHO definitions: {set(names) - found}")
    name = (
        "_echo_official_" + hashlib.sha256((str(path) + str(class_name)).encode()).hexdigest()[:16]
    )
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(
        {key: value for key, value in namespace.items() if not key.startswith("__")}
    )
    sys.modules[name] = module
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102 -- pinned source definitions
    module._source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    module._selected_names = sorted(names)
    return module


def official_cache_modules(root=_ROOT):
    """Import executable upstream cache code without importing the SGLang server."""
    root = Path(root).resolve()
    if root in _MODULES:
        return _MODULES[root]
    import triton
    import triton.language as tl

    from operators.deepseek_v32.indexer.official import topk_module

    directory = root / "sglang/python/sglang/srt/mem_cache"
    path = directory / "recall_ops.py"
    name = "_echo_official_recall_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    recall = importlib.util.module_from_spec(spec)
    sys.modules[name] = recall
    spec.loader.exec_module(recall)
    # This is the exact extension entry point used by the original wrapper.
    # The isolated module avoids installing another SGLang or replacing DeepGEMM.
    recall._fast_argmin_bounded = topk_module().fast_argmin_bounded
    recall._source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    common = {
        "torch": torch,
        "triton": triton,
        "tl": tl,
        "os": os,
        "Optional": Optional,
        "Sequence": Sequence,
        "Set": set,
        "Tuple": tuple,
    }
    allocator_path = directory / "allocator.py"
    tree = ast.parse(allocator_path.read_text())
    allocator_names = {"CudaGraphTokenToKVPoolAllocator"} | {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_cuda_graph_allocator_")
    }
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith("_CUDA_GRAPH_ALLOCATOR_"):
                constants[target.id] = ast.literal_eval(node.value)
    allocator = _selected_module(allocator_path, allocator_names, {**common, **constants})
    memory = _selected_module(
        directory / "memory_pool.py",
        {
            "set_mla_kv_buffer_kernel",
            "set_mla_kv_buffer_triton",
            "_view_kv_buffer_as_logical_dtype",
        },
        common,
    )
    host = _selected_module(
        directory / "memory_pool_host.py",
        {
            "free_device_pool_cuda_graph",
            "recall_miss_tokens_extend_cuda_graph",
            "update_priority_when_use",
            "free_req_device_pool",
            "set_mla_kv_buffer",
            "get_host_transfer_stream",
        },
        {
            **common,
            **vars(recall),
            "NSA_USE_CUDA_GRAPH": True,
            "set_mla_kv_buffer_triton": memory.set_mla_kv_buffer_triton,
            "_view_kv_buffer_as_logical_dtype": memory._view_kv_buffer_as_logical_dtype,
        },
        class_name="NSATokenToKVPoolHost",
    )
    result = SimpleNamespace(recall=recall, allocator=allocator, memory=memory, host=host)
    _MODULES[root] = result
    return result


class _LayerViews:
    """Support upstream layer/advanced indexing without duplicating NH mappings."""

    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if isinstance(key, tuple):
            layer, indices = key
            return self.values[layer][indices]
        return self.values[key]

    def __setitem__(self, key, value):
        layer, indices = key
        self.values[layer][indices] = value


class OfficialCacheState:
    """Shared official FIFO allocator, prefetch, writeback and recall state."""

    @staticmethod
    def estimate_extra_bytes(host_capacity, slots, layers):
        # Per layer: priorities, allocator buffers/scalars, FIFO and 16 hints.
        # Shared: free/allocated locations, recall counter, NH+1 recall flags.
        return {"hbm": layers * (12 * slots + 92) + 12 * slots + host_capacity + 9, "dram": 0}

    def __init__(self, pool, *, official_root=_ROOT):
        if pool.device.type != "cuda" or pool.dtype != torch.bfloat16 or pool.width != 576:
            raise ValueError("official ECHO requires CUDA BF16 512+64 MLA records")
        if pool._sessions:
            raise RuntimeError("bind the official cache before creating sessions")
        self.pool = pool
        self.modules = official_cache_modules(official_root)
        self.layer_num = pool.num_layers
        self.start_layer = 0
        self.size = pool.host_capacity
        self.kv_lora_rank, self.qk_rope_head_dim = 512, 64
        self.use_nsa, self.nsa_kv_cache_store_fp8 = True, False
        self.device_pool = SimpleNamespace(
            size=pool.slots,
            dtype=pool.dtype,
            device=pool.device,
            kv_buffer=[layer.records[: pool.slots + 1].unsqueeze(1) for layer in pool.layers],
        )
        self.device_pool.get_key_buffer = lambda layer: self.device_pool.kv_buffer[layer]
        self.kv_buffer = [layer.host.unsqueeze(1) for layer in pool.layers]
        self.host_token_to_device = _LayerViews([layer.host_to_device for layer in pool.layers])
        self.device_token_to_host = _LayerViews([layer.device_to_host for layer in pool.layers])
        self.device_pool_priority = torch.full(
            (self.layer_num, pool.slots + 1), -1, device=pool.device, dtype=torch.int32
        )
        self.device_pool_priority[:, 0] = MISSING
        self.fifo_counter = torch.zeros(self.layer_num, device=pool.device, dtype=torch.int32)
        self.extend_logits_offsets = torch.zeros(
            (self.layer_num, 16), device=pool.device, dtype=torch.float32
        )
        self.device_pool_loc_small_priority = torch.zeros(
            pool.slots + 1, device=pool.device, dtype=torch.int32
        )
        self.free_index_device_buf = torch.zeros(pool.slots, device=pool.device, dtype=torch.int32)
        self.device_pool_loc_alloc_buf = torch.zeros(
            pool.slots, device=pool.device, dtype=torch.int32
        )
        self.recall_counter = torch.zeros(1, device=pool.device, dtype=torch.uint32)
        self.host_need_recall = torch.zeros(
            pool.host_capacity + 1, device=pool.device, dtype=torch.bool
        )
        self.device_pool_allocator = [
            self.modules.allocator.CudaGraphTokenToKVPoolAllocator(
                pool.slots, pool.dtype, pool.device, self.device_pool, need_sort=False
            )
            for _ in pool.layers
        ]
        # The artifact defaults to four per-layer host-transfer streams.
        stream_count = max(1, int(os.getenv("SGLANG_NSA_HOST_TRANSFER_STREAMS", "4")))
        self.host_transfer_streams = [
            [torch.cuda.Stream(device=pool.device) for _ in range(stream_count)]
            for _ in pool.layers
        ]
        self.host_transfer_next_lane = [0] * self.layer_num
        self.host_transfer_req_events = [{} for _ in pool.layers]
        for name in self.modules.host._selected_names:
            setattr(self, name, MethodType(getattr(self.modules.host, name), self))
        self._original_drain = pool.drain
        self._original_release_ids = pool.release_ids
        pool.drain = self.drain
        pool.release_ids = self.release_ids
        if self.shared_bytes() != self.estimate_extra_bytes(
            pool.host_capacity, pool.slots, pool.num_layers
        ):
            raise RuntimeError("official metadata allocation does not match its capacity plan")

    def shared_bytes(self):
        arrays = [
            self.device_pool_priority,
            self.fifo_counter,
            self.extend_logits_offsets,
            self.device_pool_loc_small_priority,
            self.free_index_device_buf,
            self.device_pool_loc_alloc_buf,
            self.recall_counter,
            self.host_need_recall,
        ]
        for allocator in self.device_pool_allocator:
            arrays.extend(
                value for value in vars(allocator).values() if isinstance(value, torch.Tensor)
            )
        return _tensor_bytes(arrays)

    def provenance(self):
        return [
            {
                "source": module.__file__,
                "sha256": module._source_sha256,
                "definitions": getattr(module, "_selected_names", "entire module"),
            }
            for module in vars(self.modules).values()
        ]

    def layer_cache(self, cache):
        if cache._pool is not self.pool:
            raise ValueError("official cache views must share the bound pool")
        if isinstance(cache, OfficialSparseTokenCache):
            return cache
        result = OfficialSparseTokenCache.__new__(OfficialSparseTokenCache)
        result.__dict__ = cache.__dict__
        result._official = self
        cache.session._layers[cache.layer_id] = result
        return result

    def wait_host(self, cache):
        event = self.host_transfer_req_events[cache.layer_id].get(cache.session.owner)
        if event is not None:
            torch.cuda.current_stream(self.pool.device).wait_event(event)

    def drain(self):
        try:
            self._original_drain()
            for streams in self.host_transfer_streams:
                for stream in streams:
                    stream.synchronize()
        except RuntimeError:
            self.pool.poisoned = True
            raise

    def release_ids(self, layer_id, global_ids):
        """Invoke the artifact's session release on a view of one layer."""
        self.drain()
        layer = SimpleNamespace(
            layer_num=1,
            device_pool=self.device_pool,
            host_token_to_device=_LayerViews([self.host_token_to_device[layer_id]]),
            device_token_to_host=_LayerViews([self.device_token_to_host[layer_id]]),
            device_pool_priority=self.device_pool_priority[layer_id : layer_id + 1],
            device_pool_allocator=[self.device_pool_allocator[layer_id]],
        )
        self.modules.host.free_req_device_pool(layer, global_ids)

    def close(self):
        """Detach lifecycle hooks after the backend releases all sessions."""
        if getattr(self, "closed", False):
            return
        self.drain()
        self.pool.drain = self._original_drain
        self.pool.release_ids = self._original_release_ids
        self.device_pool.kv_buffer.clear()
        self.kv_buffer.clear()
        self.host_token_to_device.values.clear()
        self.device_token_to_host.values.clear()
        self.device_pool_allocator.clear()
        self.host_transfer_streams.clear()
        self.host_transfer_req_events.clear()
        for name in (
            "device_pool_priority",
            "fifo_counter",
            "extend_logits_offsets",
            "device_pool_loc_small_priority",
            "free_index_device_buf",
            "device_pool_loc_alloc_buf",
            "recall_counter",
            "host_need_recall",
        ):
            setattr(self, name, None)
        self.closed = True


class OfficialSparseTokenCache(SparseTokenCache):
    """Keep transactions while replacing every measured cache algorithm."""

    def _global_history_ids(self, indices):
        # Bounds are established by the model's current step and exact top-k.
        safe = indices.clamp_min(0).long()
        ids = self.page_table[safe // PAGE_SIZE].long() * PAGE_SIZE + safe % PAGE_SIZE
        return torch.where(indices >= 0, ids, -1)

    def reserve_append_source(self):
        self._check()
        # Upstream writeback owns tensor lifetimes through record_stream.

    def prepare_prefetch(self, new_start, new_count, offset=None, *, limit=8192):
        self._check()
        if self._prefetch is not None:
            raise RuntimeError("an official prefetch is already pending")
        if self._step_end is None or new_start != self.written or new_count < 1:
            raise ValueError("official prefetch requires an active declared append")
        if new_start + new_count > self.indexer_visible_end:
            raise ValueError("official prefetch exceeds declared indexer storage")
        if self.host_written_end > self.slots:
            raise WorkingSetTooLarge("official fixed-history adapter requires H <= P")
        state = self._official
        state.wait_host(self)
        hints = state.extend_logits_offsets[self.layer_id]
        if new_start == 0:
            hints[0].fill_(0)
        state.device_pool_loc_alloc_buf.fill_(0)
        priority = state.device_pool_priority[self.layer_id]
        ordered = torch.argsort(priority, descending=False).to(torch.int32)
        state.recall_counter.fill_(0)
        # The official kernel reads only KV entries before the current extend.
        # Its zero tail is never used as a candidate host ID.
        page_table = torch.zeros((1, new_start + new_count), device=self.device, dtype=torch.int32)
        if new_start:
            logical = torch.arange(new_start, device=self.device, dtype=torch.int64)
            page_table[0, :new_start] = self._global_history_ids(logical).int()
        arguments = {
            "page_table_1": page_table,
            "extend_seq_lens": torch.tensor([new_count], device=self.device, dtype=torch.int32),
            "extend_seq_to_req": torch.zeros(new_count, device=self.device, dtype=torch.int32),
            "device_pool_buf": state.device_pool.get_key_buffer(self.layer_id),
            "host_pool_buf": state.kv_buffer[self.layer_id],
            "device_pool_loc_alloc_buf": state.device_pool_loc_alloc_buf,
            "device_pool_priority": priority,
            "device_pool_loc_small_priority": ordered,
            "device_token_to_host": self.device_to_host,
            "host_token_to_device": self.host_to_device,
            "recall_counter": state.recall_counter,
            "extend_logits_offsets": hints,
        }
        self._prefetch = {"arguments": arguments, "new_count": new_count}
        self._pool._pending_prefetch = (self.session.owner, self.layer_id)
        return self._prefetch

    def finalize_prefetch(self, prefetch=None, *, logits=None):
        if self._prefetch is None or prefetch is not None and prefetch is not self._prefetch:
            raise RuntimeError("no matching official fused prefetch")
        state = self._official
        if logits is not None:
            hints = state.extend_logits_offsets[self.layer_id]
            hints[0].fill_(torch.mean(logits[-4:]))
        allocated = state.device_pool_loc_alloc_buf
        live = allocated > 0
        # Three existing per-session counters record prefetch, recall, eviction.
        self._prefetch_totals[0].add_(live.sum())
        self._prefetch_totals[2].add_(
            (live & (state.device_pool_priority[self.layer_id, allocated] >= 0)).sum()
        )
        state.device_pool_allocator[self.layer_id].post_alloc(allocated)
        state.update_priority_when_use(self.layer_id, allocated)
        self._prefetch = None
        self._pool._pending_prefetch = None

    def append(self, records):
        if self.transient_start is not None:
            # Only the explicit GPU suffix copy and transaction bookkeeping.
            return super().append(records)
        self._check()
        if self._step_end is None or self._prefetch is not None:
            raise RuntimeError("complete official prefetch before main-KV append")
        if (
            records.ndim != 2
            or records.shape[1] != self.width
            or records.dtype != self.records.dtype
            or records.device != self.device
            or not records.is_contiguous()
            or not len(records)
            or self.written + len(records) > self._step_end
        ):
            raise ValueError("invalid appended official MLA records")
        if len(records) > self.slots:
            raise WorkingSetTooLarge("official query batch exceeds P")
        state = self._official
        start = self.written
        logical = torch.arange(start, start + len(records), device=self.device, dtype=torch.int64)
        host_ids = self._global_history_ids(logical).int()
        need = torch.tensor([len(records)], device=self.device, dtype=torch.int32)
        available = state.device_pool_allocator[self.layer_id].available_size()
        self._prefetch_totals[2].add_(torch.clamp_min(need[0] - available, 0))
        state.set_mla_kv_buffer(
            SimpleNamespace(layer_id=self.layer_id),
            host_ids,
            records[:, None, :512],
            records[:, None, 512:],
            metadata=SimpleNamespace(nsa_non_padded=need),
            host_transfer_req_pool_indices=[self.session.owner],
        )
        self.written += len(records)
        self.indexer_visible_end = max(self.indexer_visible_end, self.written)
        self.stats.written_records += len(records)
        return start

    def ensure(self, indices):
        self._check()
        if self._prefetch is not None:
            raise RuntimeError("finalize official prefetch before exact recall")
        if self.host_written_end > self.slots:
            raise WorkingSetTooLarge("official fixed-history adapter requires H <= P")
        if (
            indices.ndim != 2
            or indices.device != self.device
            or indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("official exact recall expects a 2-D integer GPU selection")
        state = self._official
        history = indices >= 0
        if self.transient_start is not None:
            history = history & (indices < self.transient_start)
        host_ids = self._global_history_ids(torch.where(history, indices, -1))
        mapped = self.host_to_device[host_ids.clamp_min(0)]
        protected = torch.where(history, mapped, -1).int()
        # The upstream protect kernel assumes its 128-wide final tile is padded.
        # Production top-2048 already satisfies this; small correctness probes
        # need explicit negative padding without changing the actual selection.
        if protected.shape[1] % 128:
            protected = torch.nn.functional.pad(
                protected, (0, 128 - protected.shape[1] % 128), value=-1
            )
        missing = torch.where(history & (mapped == MISSING), host_ids, -1).int()
        state.wait_host(self)
        state.recall_miss_tokens_extend_cuda_graph(missing, protected, self.layer_id)
        recalled = state.recall_counter[0].to(torch.int64)
        self._prefetch_totals[1].add_(recalled)
        self._prefetch_totals[2].add_(recalled)
        physical = self.host_to_device[host_ids.clamp_min(0)].long()
        if self.transient_start is not None:
            suffix = self.slots + 1 + indices.long() - self.transient_start
            physical = torch.where(history, physical, suffix)
        return torch.where(indices >= 0, physical, -1).int()

    def metrics(self):
        result = super().metrics()
        prefetched, recalled, evicted = [int(value) for value in self._prefetch_totals.tolist()]
        result.update(
            {
                "prefetched_records": prefetched,
                "recalled_records": recalled,
                "evicted_records": evicted,
                "host_to_device_bytes": (prefetched + recalled) * self.record_bytes,
                "prefetch_capacity_failures": None,
                "selection_records": None,
                "resident_selection_records": None,
                "max_working_set": None,
                "implementation": "official_echo_bc1b75c",
                "transport_counter_boundary": "original allocator/recall counters plus included GPU reductions",
            }
        )
        return result
