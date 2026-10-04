"""Single-GPU, checkpoint-backed dense-layer replay model for GR serving.

This is a workload surrogate, not a trained smaller DeepSeek model. Every
physical layer owns an independent copy of one of checkpoint layers 0, 1, 2.
Its input is copied from that source layer's input in the first three layers,
including the separate residual stream. Copies therefore do not invent a
deeper residual trajectory. Their KV/indexer states are independent.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

from cache.host_allocation import (
    allocate_host_tensor,
    pinned_allocation_bytes,
    storage_allocation_bytes,
)
from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint
from cache.sparse_token_cache import MISSING, CacheStats, SparseTokenCache
from cache.staging import DoubleBufferStaging
from executor.serving_backend import SharedCachePlan
from models.deepseek_v32.cache_resources import (
    CACHE_POLICY_REVISION,
    check_dense_staging_allocation,
    dense_staging_allocation_bytes,
    execution_reservation,
    padded_tokens,
)
from models.deepseek_v32.echo_attention import EchoAttentionRunner
from models.deepseek_v32.echo_model import CheckpointReader, Config, rms_norm

SCHEMES = ("hbm", "echo", "serial_sparse", "dense_prefetch")


def replay_parameter_count(reader, num_layers):
    """Count model parameters, excluding FP8 scaling metadata."""
    if type(num_layers) is not int or num_layers < 3:
        raise ValueError("the replay model requires at least three physical layers")
    counts = []
    for layer in range(3):
        prefix = f"model.layers.{layer}."
        names = [name for name in reader.tensor_metadata if name.startswith(prefix)]
        if not names:
            raise ValueError(f"checkpoint is missing source layer {layer}")
        counts.append(
            sum(
                math.prod(reader.tensor_metadata[name]["shape"])
                for name in names
                if not name.endswith(".weight_scale_inv")
            )
        )
    endpoints = sum(
        math.prod(reader.tensor_metadata[name]["shape"])
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
    )
    return {
        "source_layer_parameters": counts,
        "backbone_parameters": sum(counts[layer % 3] for layer in range(num_layers)),
        "endpoint_parameters": endpoints,
        "total_parameters": endpoints + sum(counts[layer % 3] for layer in range(num_layers)),
    }


def _storage_bytes(tensors):
    seen = set()
    result = {"hbm": 0, "dram": 0}
    for tensor in tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        device = tensor.device
        identity = (device, storage.data_ptr())
        if identity not in seen:
            seen.add(identity)
            nbytes = storage.nbytes()
            if device.type == "cpu" and tensor.is_pinned():
                nbytes = pinned_allocation_bytes(nbytes)
            result["hbm" if device.type == "cuda" else "dram"] += nbytes
    return result


@dataclass
class _DenseWrite:
    source: torch.Tensor
    event: object = None


class _DenseCache:
    """Pinned full-layer backing and a borrowed double-buffered HBM stage."""

    def __init__(self, capacity, width, *, device, backend):
        self.capacity, self.width = capacity, width
        self.offload = True
        self.device = torch.device(device)
        self.slots = capacity
        self.backend = backend
        self.session = None
        self.records = None
        self.host = allocate_host_tensor(
            (capacity, width), dtype=torch.bfloat16, pin_memory=self.device.type == "cuda"
        )
        self.host_to_device = torch.full((capacity,), MISSING, dtype=torch.int32, device=device)
        self.device_to_host = torch.full((capacity,), MISSING, dtype=torch.int64, device=device)
        self.age = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.length = self.written = 0
        self._step_end = None
        self._clock = 0
        self.stats = CacheStats()
        self.prefetch_counts = []
        self.dense_fetched_records = 0

    @property
    def record_bytes(self):
        return self.width * self.host.element_size()

    def _check_execution(self):
        self.backend._check_dense_execution(self.session)

    def reserve_append_source(self):
        self._check_execution()
        self.backend._reserve_dense_source()

    def declare_indexer_visible(self, end):
        if self._step_end is None or not self.written <= end <= self._step_end:
            raise ValueError("indexer end must lie within the active step")

    def begin_step(self, count):
        self._check_execution()
        if self._step_end is not None:
            raise RuntimeError("a cache step is already active")
        if count < 1 or self.length + count > self.capacity:
            raise ValueError("cache step exceeds capacity")
        self._step_end = self.length + count

    def append(self, records):
        self._check_execution()
        if self.records is None:
            raise RuntimeError("dense append requires this layer's ready staging view")
        if self._step_end is None or self.written + len(records) > self._step_end:
            raise ValueError("append exceeds the active cache step")
        start, stop = self.written, self.written + len(records)
        self.records[start:stop].copy_(records)
        # Retain before enqueue: even an event-record failure must keep a source
        # alive until the enclosing execution lease confirms both streams drain.
        ticket = _DenseWrite(records)
        self.backend._dense_sources.append(ticket)
        self.host[start:stop].copy_(records, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            ticket.event = torch.cuda.Event()
            ticket.event.record(torch.cuda.current_stream(self.device))
        self.written = stop
        self.stats.written_records += len(records)
        return start

    def commit(self):
        self._check_execution()
        if self._step_end is None or self.written != self._step_end:
            raise RuntimeError("cannot commit an incomplete cache step")
        self.length = self.written
        self._step_end = None

    def rollback(self):
        self._check_execution()
        self._step_end = None
        self.truncate(self.length)

    def truncate(self, length):
        if self._step_end is not None or not 0 <= length <= self.length:
            raise ValueError("truncate requires an inactive committed prefix")
        self.length = self.written = length

    def reset_stats(self):
        self.stats = CacheStats()
        self.dense_fetched_records = 0

    def metrics(self):
        return {
            **vars(self.stats),
            "prefetched_records": 0,
            "dense_fetched_records": self.dense_fetched_records,
            "host_to_device_bytes": self.dense_fetched_records * self.record_bytes,
            "device_to_host_bytes": self.stats.written_records * self.record_bytes,
            "device_slots": self.slots,
            "device_record_bytes": 0,  # Staging belongs to the backend ledger.
            "host_record_bytes": self.host.numel() * self.host.element_size(),
            "host_allocation_bytes": storage_allocation_bytes(self.host),
            "record_bytes": self.record_bytes,
        }


class _ServingAttention(EchoAttentionRunner):
    def __init__(
        self, attention, capacity, *, scheme, slots, chunk_size, dense_backend=None, cache=None
    ):
        self.scheme = scheme
        if scheme == "dense_prefetch" and cache is None:
            cache = _DenseCache(
                capacity,
                attention.cfg.kv_lora_rank + attention.cfg.qk_rope_head_dim,
                device=attention.device,
                backend=dense_backend,
            )
        super().__init__(
            attention,
            capacity,
            offload=scheme != "hbm",
            slots=slots,
            chunk_size=chunk_size,
            cache=cache,
            fused_prefetch=scheme == "echo",
        )

    def _consume(self, q, indices, scope):
        if isinstance(self.cache, _DenseCache):
            from operators.deepseek_v32.attention.device_only.mla import sparse_mla

            with scope("sparse_mla"):
                return sparse_mla(q, self.cache.records, indices, self.cfg.attention_scale)
        return super()._consume(q, indices, scope)


@dataclass
class DeepSeekServingSession:
    capacity: int
    scheme: str
    runners: list
    owner: object = field(repr=False)
    length: int = 0
    stages: list = field(default_factory=list)
    copy_stream: object = None
    stage_consumed: list = field(default_factory=list)
    released: bool = False
    sparse_session: object = None
    prefix_offsets: list = field(default_factory=list)
    prefix_length: int = 0
    dense_allocation_ready: object = None
    last_candidate_transient: bool = False


class DeepSeekServingBackend:
    """Serve retained user sessions using one physical single-GPU replay model.

    Cache allocations are fixed at session creation. The serving cache manager
    can therefore reserve their upper bound before admitting a user. Temporary
    activations, selections and GEMM workspace are model execution memory and
    are reported separately from retained cache memory.
    """

    def __init__(
        self,
        model_path,
        *,
        scheme="hbm",
        device="cuda:0",
        num_layers=10,
        chunk_size=2048,
        extend_chunk_size=None,
        slots=None,
        sparse_pool_tokens=32768,
        host_arena_tokens=None,
        workspace_query_tokens=None,
        linear_backend="fp8",
        enable_compute_graphs=False,
        compute_graph_private_limit_bytes=12 * 2**30,
    ):
        from models.deepseek_v32.echo_block import CheckpointBlock

        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        if slots is not None:
            raise ValueError("slots was per-session; use sparse_pool_tokens for the backend pool")
        if chunk_size < 1 or sparse_pool_tokens < chunk_size:
            raise ValueError("chunk_size must be positive and fit in sparse_pool_tokens")
        slots = sparse_pool_tokens
        if host_arena_tokens is not None and (host_arena_tokens < 64 or host_arena_tokens % 64):
            raise ValueError("host_arena_tokens must be a positive multiple of 64")
        if workspace_query_tokens is not None and workspace_query_tokens < chunk_size:
            raise ValueError("workspace query bound cannot be smaller than prefill chunk")
        self.host_arena_tokens = host_arena_tokens
        self.workspace_query_tokens = workspace_query_tokens or max(
            chunk_size, extend_chunk_size or 0
        )
        self._shared_pool = None
        self._merged_index_keys = self._merged_index_scales = None
        self._dense_staging = None
        self._pool_prefetch = None
        self._dense_staging_allocated_bytes = 0
        self._dense_lease = None
        self._dense_sources = []
        self._resource_plan = None
        if type(enable_compute_graphs) is not bool:
            raise ValueError("enable_compute_graphs must be a bool")
        if (
            type(compute_graph_private_limit_bytes) is not int
            or compute_graph_private_limit_bytes < 1
        ):
            raise ValueError("compute graph private limit must be a positive integer")
        self.enable_compute_graphs = enable_compute_graphs
        self.compute_graph_private_limit_bytes = compute_graph_private_limit_bytes
        self._compute_graphs = None
        self._allocation_failure = None
        self._admission_owner = None
        self._admission_initial_plan = None
        self._active_session = None
        self._sessions = []
        self.path = Path(model_path)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DeepSeek serving requires one SM90 CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (9, 0):
            raise ValueError("DeepSeek serving requires one SM90 CUDA device")
        self.scheme = scheme
        self.cfg = Config.from_checkpoint(model_path)
        if self.cfg.first_k_dense_replace < 3:
            raise ValueError("checkpoint layers 0, 1, 2 must all have dense MLPs")
        self.max_seq_len = self.cfg.max_seq_len
        self.reader = CheckpointReader(model_path)
        self.parameter_counts = replay_parameter_count(self.reader, num_layers)
        if extend_chunk_size is not None and extend_chunk_size < 1:
            raise ValueError("extend_chunk_size must be positive or None for a whole batch")
        self.extend_chunk_size = extend_chunk_size
        self.num_layers, self.chunk_size, self.slots = num_layers, chunk_size, slots
        if scheme in ("echo", "serial_sparse") and slots < self.cfg.index_topk:
            raise ValueError("the sparse pool must fit one query's exact top-k selection")
        self.linear_backend = linear_backend
        self.embedding_weight = self.reader.get_tensor("model.embed_tokens.weight").to(self.device)
        self.final_norm = self.reader.get_tensor("model.norm.weight").to(self.device)
        self.head_weight = self.reader.get_tensor("lm_head.weight").to(self.device)
        self.blocks, self.attentions = [], []
        # Load every copy independently. No weight aliases or shared KV rows are
        # introduced between copies, even when the source layer is the same.
        for layer in range(num_layers):
            block = CheckpointBlock(
                model_path,
                layer % 3,
                self.device,
                self.reader,
                capacity=1,
                chunk_size=chunk_size,
                linear_backend=linear_backend,
            )
            self.attentions.append(block.attention.attention)
            block.attention = block.cache = None
            self.blocks.append(block)
        self.last_logits = None
        self.capture_hook = None
        self._busy = False
        self._poisoned = False
        self.synchronize()

    def plan_resources(self, budgets, limits):
        """Plan fixed allocations and workspace; optional budgets limit admission."""
        from cache.sparse_token_pool import SharedSparseTokenPool

        if budgets is None and self.scheme != "hbm" and self.host_arena_tokens is None:
            raise ValueError("fixed-pools admission requires explicit host_arena_tokens")
        shared_pool = self.scheme in ("echo", "serial_sparse") or (
            self.scheme == "dense_prefetch" and budgets is None
        )
        transient = shared_pool or budgets is None
        capacity = limits["max_session_capacity"]
        if capacity > self.max_seq_len:
            raise ValueError("planned session exceeds model context limit")
        if shared_pool and self.slots < min(self.cfg.index_topk, capacity):
            raise CacheBudgetExceeded("sparse pool must fit one query's full exact selection")
        queries = self.workspace_query_tokens
        candidate = limits.get(
            "max_candidate_tokens",
            capacity if self.extend_chunk_size and not transient else min(capacity, queries),
        )
        if type(candidate) is not int or not 0 < candidate <= capacity:
            raise ValueError("candidate limit must satisfy 0 < A <= C")
        history = limits.get("max_history_tokens", capacity)
        if type(history) is not int or not 0 < history <= capacity:
            raise ValueError("history limit must satisfy 0 < H <= context capacity")
        if candidate > queries and (transient or self.extend_chunk_size is None):
            raise CacheBudgetExceeded("candidate limit exceeds whole-batch query workspace")
        if self.extend_chunk_size is not None and self.extend_chunk_size > queries:
            raise CacheBudgetExceeded("extend chunk exceeds query workspace")
        if shared_pool and queries > self.slots:
            raise CacheBudgetExceeded("query workspace exceeds the per-layer device pool")
        if budgets is None and self.scheme in ("hbm", "dense_prefetch") and history > self.slots:
            raise CacheBudgetExceeded("fixed HBM/dense pools must fit one complete history")
        width = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim
        execution = execution_reservation(queries, capacity, topk=self.cfg.index_topk, width=width)
        graph_plan = {}
        if getattr(self, "enable_compute_graphs", False):
            from models.deepseek_v32.compute_graphs import plan_compute_graphs

            graph_plan = plan_compute_graphs(
                self.cfg,
                self.num_layers,
                self.chunk_size,
                history,
                candidate,
                self.compute_graph_private_limit_bytes,
            )
        graph_reservation = graph_plan.get("compute_graph_reserved_limit_bytes", 0)
        merged_indexer_bytes = capacity * (self.cfg.index_head_dim + 4) if transient else 0
        if not shared_pool:
            staging_bytes = (
                DoubleBufferStaging.estimate_bytes(
                    capacity, {"records": (width,)}, dtype=torch.bfloat16
                )
                if self.scheme == "dense_prefetch"
                else 0
            )
            staging_allocation = (
                dense_staging_allocation_bytes(staging_bytes, self.device) if staging_bytes else 0
            )
            return SharedCachePlan(
                shared=CacheFootprint(
                    execution.hbm + staging_allocation + merged_indexer_bytes + graph_reservation,
                    execution.dram,
                ),
                hbm_tokens=self.slots if budgets is None else 0,
                metadata={
                    **graph_plan,
                    **({"resource_mode": "fixed_pools"} if budgets is None else {}),
                    "cache_policy_revision": CACHE_POLICY_REVISION,
                    "pool_scope": self.scheme,
                    "max_session_capacity": capacity,
                    "max_history_tokens": history,
                    "candidate_persistence": "gpu_transient" if transient else "committed",
                    "candidate_slots": candidate if transient else 0,
                    "merged_indexer_bytes": merged_indexer_bytes,
                    "sparse_pool_tokens": self.slots,
                    "workspace_query_tokens": queries,
                    "max_candidate_tokens": candidate,
                    "workspace_indexer_bytes": execution.indexer_bytes,
                    "workspace_copy_source_bytes": execution.copy_source_bytes,
                    "dense_staging_bytes": staging_bytes,
                    "dense_staging_allocation_bytes": staging_allocation,
                    "dense_staging_scope": "backend" if staging_bytes else None,
                    "max_inflight_writes": 2,
                    **execution.cpu_workspace_metadata,
                },
            )
        minimum = padded_tokens(history)
        # Raw bytes give only an upper bound. The search below checks each
        # layer's independent pinned bin and metadata at every candidate NH.
        upper = self.host_arena_tokens
        if upper is None:
            upper = budgets.dram // (self.num_layers * width * 2) // 64 * 64
        private = SharedSparseTokenPool.estimate_session_bytes(
            history, layers=self.num_layers, device=self.device
        )
        session = CacheFootprint(
            private["hbm"] + self.num_layers * (history * (self.cfg.index_head_dim + 4) + 3 * 64),
            private["dram"],
        )
        transient_indexer_bytes = merged_indexer_bytes

        def estimate(tokens):
            result = SharedSparseTokenPool.estimate_shared_bytes(
                tokens,
                width,
                self.num_layers,
                self.slots,
                dtype=torch.bfloat16,
                device=self.device,
                candidate_slots=candidate,
            )
            metadata_workspace = SharedSparseTokenPool.estimate_execution_workspace_bytes(
                tokens, self.slots
            )
            return CacheFootprint(
                result["hbm"]
                + execution.hbm
                + graph_reservation
                + metadata_workspace
                + (transient_indexer_bytes if self.device.type == "cuda" else 0),
                result["dram"]
                + execution.dram
                + (transient_indexer_bytes if self.device.type != "cuda" else 0),
            )

        def fits(tokens):
            shared = estimate(tokens)
            # Every page can move from the shared free stack to a session CPU
            # ownership table. Leave that DRAM available for private admission.
            private_page_tables = tokens // 64 * 4
            return (
                shared.hbm + session.hbm <= budgets.hbm
                and shared.dram + max(session.dram, private_page_tables) <= budgets.dram
            )

        # NH is constrained by global maps in HBM as well as allocated host DRAM.
        # Require room for one maximum-size session before allocating or evicting.
        if budgets is None:
            if upper < minimum:
                raise CacheBudgetExceeded("fixed host arena cannot hold one maximum-size session")
            tokens = upper
        elif upper < minimum or not fits(minimum):
            raise CacheBudgetExceeded("shared pool, workspace and one session do not fit budgets")
        elif self.host_arena_tokens is None:
            low, high = minimum // 64, upper // 64
            while low < high:
                middle = (low + high + 1) // 2
                if fits(middle * 64):
                    low = middle
                else:
                    high = middle - 1
            tokens = low * 64
        else:
            tokens = upper
            if not fits(tokens):
                raise CacheBudgetExceeded("configured host arena plus one session exceeds budgets")
        return SharedCachePlan(
            shared=estimate(tokens),
            host_pages=tokens // 64,
            metadata={
                **graph_plan,
                "cache_policy_revision": CACHE_POLICY_REVISION,
                **({"resource_mode": "fixed_pools"} if budgets is None else {}),
                "pool_scope": "backend_per_layer",
                "shared_token_pool": True,
                "dense_fetch_policy": "all_history_cache_misses_next_layer"
                if self.scheme == "dense_prefetch"
                else None,
                "host_arena_tokens": tokens,
                "sparse_pool_tokens": self.slots,
                "max_session_capacity": capacity,
                "max_history_tokens": history,
                "candidate_persistence": "gpu_transient",
                "candidate_slots": candidate,
                "merged_indexer_bytes": merged_indexer_bytes,
                "workspace_query_tokens": queries,
                "max_candidate_tokens": candidate,
                "workspace_indexer_bytes": execution.indexer_bytes,
                "workspace_copy_source_bytes": execution.copy_source_bytes,
                **execution.cpu_workspace_metadata,
                "workspace_metadata_bytes": SharedSparseTokenPool.estimate_execution_workspace_bytes(
                    tokens, self.slots
                ),
                "max_inflight_writes": 2,
            },
        )

    def allocate_shared(self, plan):
        if getattr(self, "_poisoned", False):
            raise RuntimeError("backend is poisoned after an asynchronous CUDA failure")
        if not isinstance(plan, SharedCachePlan):
            raise TypeError("plan must be a SharedCachePlan")
        if self.scheme == "dense_prefetch" and not plan.metadata.get("shared_token_pool"):
            expected = self.plan_resources(plan.shared, plan.metadata)
            if plan != expected:
                raise ValueError("shared dense plan does not match this backend's reservation")
        if self._resource_plan == plan:
            return
        if self._resource_plan is not None or self._shared_pool is not None:
            raise RuntimeError("close the prior shared plan before changing its plan")
        if self._sessions or self._execution_active():
            raise RuntimeError("cannot allocate shared resources with live sessions or execution")
        try:
            metadata = plan.metadata
            if metadata.get("shared_token_pool"):
                from cache.sparse_token_pool import SharedSparseTokenPool
                from operators.deepseek_v32.indexer import cache_ops

                self._shared_pool = SharedSparseTokenPool(
                    metadata["host_arena_tokens"],
                    self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim,
                    self.num_layers,
                    self.slots,
                    device=self.device,
                    dtype=torch.bfloat16,
                    max_inflight_writes=metadata["max_inflight_writes"],
                    metadata_ops=cache_ops,
                    candidate_slots=metadata["candidate_slots"],
                )
                if self.scheme == "dense_prefetch":
                    from models.deepseek_v32.pool_prefetch import PoolHistoryPrefetch

                    self._pool_prefetch = PoolHistoryPrefetch(self.device)
            elif self.scheme == "dense_prefetch":
                self._dense_staging = DoubleBufferStaging(
                    metadata["max_session_capacity"],
                    {"records": (self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim,)},
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                (stage_tensor,) = self._dense_staging.storage_tensors()
                self._dense_staging_allocated_bytes = check_dense_staging_allocation(
                    stage_tensor, metadata["dense_staging_allocation_bytes"]
                )
            if metadata.get("candidate_persistence") == "gpu_transient":
                self._merged_index_keys = torch.empty(
                    (metadata["max_session_capacity"], self.cfg.index_head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=self.device,
                )
                self._merged_index_scales = torch.empty(
                    metadata["max_session_capacity"], dtype=torch.float32, device=self.device
                )
            if metadata.get("compute_graphs_enabled", False):
                from models.deepseek_v32.compute_graphs import (
                    DeepSeekComputeGraphs,
                    plan_compute_graphs,
                )

                expected_graph = plan_compute_graphs(
                    self.cfg,
                    self.num_layers,
                    self.chunk_size,
                    metadata["max_history_tokens"],
                    metadata["max_candidate_tokens"],
                    self.compute_graph_private_limit_bytes,
                )
                if not self.enable_compute_graphs or any(
                    metadata.get(name) != value for name, value in expected_graph.items()
                ):
                    raise ValueError("compute graph plan does not match this backend")
                self._compute_graphs = DeepSeekComputeGraphs(
                    self.attentions, self.blocks, self.device, metadata
                )
                self._compute_graphs.allocate()
            elif getattr(self, "enable_compute_graphs", False):
                raise ValueError("enabled compute graphs are absent from the resource plan")
            self._resource_plan = plan
        except BaseException as error:
            # The traceback retains a partially constructed pool/staging object
            # if its constructor failed after enqueueing allocation work.
            self._allocation_failure = error
            self._release_shared()
            raise

    def shared_bytes(self):
        result = {"hbm": 0, "dram": 0}
        if self._shared_pool is not None:
            fixed = self._shared_pool.shared_bytes()
            pending = self._shared_pool.pending_source_bytes
            result = {name: fixed[name] + pending[name] for name in fixed}
        staging = getattr(self, "_dense_staging", None)
        if staging is not None:
            fixed = staging.shared_bytes()
            result = {name: result[name] + fixed[name] for name in result}
        pending = _storage_bytes(ticket.source for ticket in getattr(self, "_dense_sources", []))
        transient = _storage_bytes(
            getattr(self, name, None)
            for name in (
                "_merged_index_keys",
                "_merged_index_scales",
            )
        )
        graphs = getattr(self, "_compute_graphs", None)
        compute = graphs.shared_bytes() if graphs is not None else {"hbm": 0, "dram": 0}
        return {
            name: result[name] + pending[name] + transient[name] + compute[name] for name in result
        }

    def retained_session_capacity(self, capacity, prefix_tokens):
        """GR admission owns history only; standalone extend retains its old API."""
        return prefix_tokens if self._transient_candidates_enabled() else capacity

    def _uses_shared_token_pool(self):
        plan = getattr(self, "_resource_plan", None)
        return self.scheme in ("echo", "serial_sparse") or (
            plan is not None and plan.metadata.get("shared_token_pool", False)
        )

    def _transient_candidates_enabled(self):
        plan = getattr(self, "_resource_plan", None)
        return self.scheme in ("echo", "serial_sparse") or (
            plan is not None and plan.metadata.get("candidate_persistence") == "gpu_transient"
        )

    def estimate_session_host_pages(self, capacity):
        shared = self._uses_shared_token_pool() or (
            self.scheme == "dense_prefetch" and getattr(self, "_resource_plan", None) is None
        )
        return padded_tokens(capacity) // 64 if shared else 0

    def estimate_session_hbm_tokens(self, capacity):
        return capacity if self.scheme == "hbm" else 0

    def session_hbm_tokens(self, session):
        self._check_session(session)
        return session.capacity if self.scheme == "hbm" else 0

    def session_host_pages(self, session):
        self._check_session(session)
        return session.sparse_session.host_pages if session.sparse_session is not None else 0

    def estimate_session_bytes(self, capacity, prefix_tokens=0):
        if type(capacity) is not int or not 1 <= capacity <= self.max_seq_len:
            raise ValueError("session capacity exceeds the model context limit")
        if not 0 <= prefix_tokens <= capacity:
            raise ValueError("prefix length must fit session capacity")
        width = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim
        record = width * 2
        # Current hint, committed-prefix hint and transaction backup are distinct.
        index = capacity * (self.cfg.index_head_dim + 4) + 3 * 64
        if self._uses_shared_token_pool():
            from cache.sparse_token_pool import SharedSparseTokenPool

            local = SharedSparseTokenPool.estimate_session_bytes(
                capacity, layers=self.num_layers, device=self.device
            )
            return {"hbm": local["hbm"] + self.num_layers * index, "dram": local["dram"]}
        maps = capacity * 4 + capacity * 16
        candidate = (
            self._resource_plan.metadata["candidate_slots"]
            if self.scheme == "hbm" and self._transient_candidates_enabled()
            else 0
        )
        records = (capacity + candidate) * record if self.scheme == "hbm" else 0
        return {
            "hbm": self.num_layers * (index + maps + records),
            "dram": self.num_layers * pinned_allocation_bytes(capacity * record)
            if self.scheme == "dense_prefetch"
            else 0,
        }

    def create_session(self, capacity):
        if getattr(self, "_poisoned", False) or self._execution_active():
            raise RuntimeError("cannot create a session during execution or after a CUDA failure")
        self.estimate_session_bytes(capacity)
        plan = getattr(self, "_resource_plan", None)
        if plan is not None and capacity > plan.metadata["max_session_capacity"]:
            raise CacheBudgetExceeded("session exceeds the reserved workspace context bound")
        if (
            self.scheme == "dense_prefetch"
            and not self._uses_shared_token_pool()
            and getattr(self, "_dense_staging", None) is None
        ):
            raise RuntimeError("plan_resources and allocate_shared must precede create_session")
        sparse_session = None
        if self._uses_shared_token_pool():
            if self._shared_pool is None:
                raise RuntimeError("plan_resources and allocate_shared must precede create_session")
            if capacity > self._resource_plan.metadata["max_session_capacity"]:
                raise CacheBudgetExceeded("session exceeds the reserved workspace context bound")
            sparse_session = self._shared_pool.allocate_session(capacity)
        try:
            runners = [
                _ServingAttention(
                    attention,
                    capacity,
                    scheme=self.scheme,
                    slots=self.slots,
                    chunk_size=self.chunk_size,
                    dense_backend=self if self.scheme == "dense_prefetch" else None,
                    cache=(
                        sparse_session.layer(layer)
                        if sparse_session is not None
                        else SparseTokenCache(
                            capacity,
                            self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim,
                            device=self.device,
                            candidate_slots=plan.metadata["candidate_slots"],
                        )
                        if self.scheme == "hbm" and self._transient_candidates_enabled()
                        else None
                    ),
                )
                for layer, attention in enumerate(self.attentions)
            ]
            session = DeepSeekServingSession(
                capacity,
                self.scheme,
                runners,
                self,
                sparse_session=sparse_session,
            )
            if self.scheme == "dense_prefetch" and sparse_session is None:
                for runner in runners:
                    runner.cache.session = session
                if self.device.type == "cuda":
                    session.dense_allocation_ready = torch.cuda.Event()
                    session.dense_allocation_ready.record(torch.cuda.current_stream(self.device))
        except BaseException as error:
            if self.scheme == "dense_prefetch" and not self._uses_shared_token_pool():
                try:
                    self.synchronize()
                except BaseException as drain_error:
                    self._allocation_failure = error
                    self._poisoned = True
                    raise RuntimeError(
                        "unable to drain partial dense session allocation"
                    ) from drain_error
            if sparse_session is not None:
                sparse_session.release()
            raise
        self._sessions.append(session)
        return session

    def _check_session(self, session):
        if getattr(self, "_poisoned", False):
            raise RuntimeError("backend is poisoned after an asynchronous CUDA failure")
        if session.owner is not self or session.released:
            raise ValueError("session belongs to another backend or has been released")
        if session.scheme != self.scheme:
            raise ValueError("session cache policy differs from the backend")

    def session_bytes(self, session):
        self._check_session(session)
        tensors = list(session.prefix_offsets)
        for runner in session.runners:
            tensors.extend((runner.index_keys, runner.index_scales, runner.offset))
            if session.sparse_session is None:
                cache = runner.cache
                tensors.extend(
                    (
                        None if session.scheme == "dense_prefetch" else cache.records,
                        cache.host,
                        cache.host_to_device,
                        cache.device_to_host,
                        cache.age,
                    )
                )
        result = _storage_bytes(tensors)
        if session.sparse_session is not None:
            local = session.sparse_session.session_bytes()
            result = {name: result[name] + local[name] for name in result}
        return result

    def _check_dense_execution(self, session):
        if session is None or getattr(self, "_active_session", None) is not session:
            raise RuntimeError("dense cache requires its session's execution lease")
        lease = getattr(self, "_dense_lease", None)
        if lease is None or lease.closed:
            raise RuntimeError("dense staging execution lease is unavailable")
        lease._check()

    def _execution_active(self):
        staging = getattr(self, "_dense_staging", None)
        return (
            getattr(self, "_busy", False)
            or getattr(self, "_active_session", None) is not None
            or getattr(self, "_dense_lease", None) is not None
            or staging is not None
            and staging.active_lease is not None
        )

    def _reserve_dense_source(self):
        self._check_dense_execution(self._active_session)
        pending = self._dense_sources
        while pending and (self.device.type == "cpu" or pending[0].event.query()):
            pending.pop(0)
        limit = self._resource_plan.metadata["max_inflight_writes"]
        if len(pending) >= limit:
            pending[0].event.synchronize()
            pending.pop(0)

    def _prefetch_dense_layer(self, session, layer):
        self._check_dense_execution(session)
        cache = session.runners[layer].cache
        lease = self._dense_lease
        before = lease.submitted_copy_bytes
        try:
            lease.prefetch(layer, {"records": cache.host[: cache.written]})
        finally:
            cache.dense_fetched_records += (
                lease.submitted_copy_bytes - before
            ) // cache.record_bytes

    @contextmanager
    def _execution(self, session):
        self._check_session(session)
        if self._execution_active():
            raise RuntimeError("the single-GPU backend executes one request at a time")
        self._busy = True
        self._active_session = session
        try:
            if self.scheme == "dense_prefetch" and not self._uses_shared_token_pool():
                staging = getattr(self, "_dense_staging", None)
                if staging is None:
                    raise RuntimeError("dense execution requires allocated shared staging")
                if session.dense_allocation_ready is not None:
                    try:
                        torch.cuda.current_stream(self.device).wait_event(
                            session.dense_allocation_ready
                        )
                    except BaseException:
                        self._poisoned = True
                        raise
                try:
                    self._dense_lease = staging.lease(session)
                except BaseException:
                    if staging.poisoned:
                        self._poisoned = True
                        self._dense_lease = staging.active_lease
                    raise
            graphs = getattr(self, "_compute_graphs", None)
            with graphs.execution() if graphs is not None else nullcontext():
                yield
        finally:
            try:
                graphs = getattr(self, "_compute_graphs", None)
                if graphs is not None and graphs.failed:
                    # A failed completion event leaves borrowed graph outputs
                    # live. Retain the execution/admission owner and storage.
                    self._poisoned = True
                prefetch = getattr(self, "_pool_prefetch", None)
                if prefetch is not None:
                    prefetch.drain()
                lease = getattr(self, "_dense_lease", None)
                if lease is not None:
                    try:
                        lease.close()
                    except BaseException:
                        self._poisoned = True
                        raise
                staging = getattr(self, "_dense_staging", None)
                if staging is not None and staging.poisoned:
                    self._poisoned = True
                if not getattr(self, "_poisoned", False):
                    for runner in session.runners:
                        if isinstance(runner.cache, _DenseCache):
                            runner.cache.records = None
                    self._dense_lease = None
                    self._dense_sources = []
                    self._active_session = None
            finally:
                self._busy = False

    @torch.inference_mode()
    def _forward(
        self,
        session,
        token_ids,
        *,
        scope=None,
        chunk_size=None,
        _capture_prefix=False,
        _transient_candidate=False,
    ):
        # This lease covers all prefill chunks, commit/abort, final LM head, and
        # the retained prefix hint snapshot, including direct profiler calls.
        self._check_session(session)
        if not _capture_prefix:
            self._check_candidate_limit(token_ids)
        with self._execution(session):
            result = self._forward_leased(
                session,
                token_ids,
                scope=scope,
                chunk_size=chunk_size,
                transient_candidate=_transient_candidate,
            )
            if _capture_prefix:
                session.prefix_length = session.length
                session.prefix_offsets = [
                    runner.offset.clone() if hasattr(runner, "offset") else None
                    for runner in session.runners
                ]
            return result

    @contextmanager
    def _candidate_indexer_view(self, session, runner):
        """Borrow one combined indexer view while retaining only private history."""
        history = session.length
        if runner.cache.written != history:
            raise RuntimeError("candidates must execute as one whole batch")
        old_keys, old_scales = runner.index_keys, runner.index_scales
        keys, scales = self._merged_index_keys, self._merged_index_scales
        keys[:history].copy_(old_keys[:history])
        scales[:history].copy_(old_scales[:history])
        runner.index_keys, runner.index_scales = keys, scales
        try:
            yield
        finally:
            runner.index_keys, runner.index_scales = old_keys, old_scales

    def _prefetch_pool_history(self, cache):
        ticket = self._pool_prefetch.prefetch(cache)
        for name in ("requested_records", "resident_records", "fetched_records"):
            key = "dense_" + name
            cache.dense_history_metrics[key] += getattr(ticket, name)
        return ticket

    def _forward_leased(
        self,
        session,
        token_ids,
        *,
        scope=None,
        chunk_size=None,
        transient_candidate=False,
    ):
        self._check_session(session)
        session.last_candidate_transient = False
        if transient_candidate and chunk_size is not None:
            raise ValueError("transient candidates execute in one batch")
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        planned = getattr(self, "_resource_plan", None)
        limit = (
            planned.metadata["max_session_capacity"] if transient_candidate else session.capacity
        )
        if ids.ndim != 1 or not len(ids) or session.length + len(ids) > limit:
            raise ValueError("nonempty token IDs must fit the session cache")
        if int(ids.min()) < 0 or int(ids.max()) >= self.cfg.vocab_size:
            raise ValueError("token ID is outside the checkpoint vocabulary")
        query_chunk = len(ids) if chunk_size is None else chunk_size
        if planned is not None and query_chunk > planned.metadata["workspace_query_tokens"]:
            raise CacheBudgetExceeded("query batch exceeds reserved cache execution workspace")
        scope = scope or (lambda _: nullcontext())
        started = []
        offsets = []
        try:
            for runner in session.runners:
                offsets.append(runner.offset.clone() if hasattr(runner, "offset") else None)
            for runner in session.runners:
                runner.cache.reset_stats()
                runner.cache.dense_history_metrics = {
                    "dense_requested_records": 0,
                    "dense_resident_records": 0,
                    "dense_fetched_records": 0,
                }
                if transient_candidate:
                    runner.cache.begin_transient(len(ids))
                else:
                    runner.cache.begin_step(len(ids))
                started.append(runner.cache)
            query_chunk = len(ids) if chunk_size is None else chunk_size
            output_parts = []
            for start in range(0, len(ids), query_chunk):
                stop = min(len(ids), start + query_chunk)
                with scope("embedding"):
                    hidden = F.embedding(ids[start:stop], self.embedding_weight)
                    residual = None
                sources = []
                pool_dense = getattr(self, "_pool_prefetch", None)
                dense = self.scheme == "dense_prefetch" and pool_dense is None
                if dense:
                    self._prefetch_dense_layer(session, 0)
                ticket = (
                    self._prefetch_pool_history(session.runners[0].cache)
                    if pool_dense is not None
                    else None
                )
                for layer, (block, runner) in enumerate(
                    zip(self.blocks, session.runners, strict=True)
                ):
                    if dense:
                        runner.cache.records = self._dense_lease.wait_ready(layer)["records"][
                            : session.capacity
                        ]
                        if layer + 1 < self.num_layers:
                            self._prefetch_dense_layer(session, layer + 1)
                    if pool_dense is not None:
                        pool_dense.wait(ticket)
                        if layer + 1 < self.num_layers:
                            ticket = self._prefetch_pool_history(session.runners[layer + 1].cache)
                    if layer < 3:
                        sources.append((hidden, residual))
                    else:
                        with scope("replay_input_copy"):
                            source_hidden, source_residual = sources[layer % 3]
                            hidden = source_hidden.clone()
                            residual = (
                                source_residual.clone() if source_residual is not None else None
                            )
                    block.attention, block.cache = runner, runner.cache
                    block.chunk_size = len(hidden)
                    indexer_view = (
                        self._candidate_indexer_view(session, runner)
                        if transient_candidate
                        else nullcontext()
                    )
                    candidate_lease = (
                        runner.cache.operation() if transient_candidate else nullcontext()
                    )
                    with scope(f"layer_{layer}_source_{layer % 3}"), candidate_lease, indexer_view:
                        graphs = getattr(self, "_compute_graphs", None)
                        graph_used = graphs is not None and graphs.supports(layer, hidden, residual)
                        if graph_used:
                            hidden, residual = graphs.forward_block(
                                layer, runner, hidden, residual, scope=scope
                            )
                        else:
                            if graphs is not None:
                                graphs.eager_fallbacks += 1
                            hidden, residual = block.forward(hidden, residual, scope=scope)
                    if self.capture_hook is not None:
                        if graph_used:
                            # A retained diagnostic reference must survive later replay.
                            self.capture_hook(layer, hidden.clone(), residual.clone())
                        else:
                            self.capture_hook(layer, hidden, residual)
                    if dense:
                        runner.cache.records = None
                with scope("final_norm_lm_head"):
                    output = rms_norm(
                        hidden.float() + residual.float(), self.final_norm, self.cfg.norm_eps
                    )
                    output = output.bfloat16()
                    output_parts.append(output)
                    if stop == len(ids):
                        self.last_logits = F.linear(output[-1:], self.head_weight).float()
                if dense and stop < len(ids):
                    self._dense_lease.reset()
                    self._dense_sources.clear()
                if pool_dense is not None:
                    pool_dense.drain()
            output = torch.cat(output_parts)
            self.synchronize()
            if any(cache.written != cache._step_end for cache in started):
                raise RuntimeError("one or more layers did not complete their cache step")
            for cache in started:
                if transient_candidate:
                    cache.discard_transient()
                else:
                    cache.commit()
            if transient_candidate:
                for runner, offset in zip(session.runners, offsets, strict=True):
                    if offset is not None:
                        runner.offset.copy_(offset)
                self.synchronize()
                session.last_candidate_transient = True
            else:
                session.length += len(ids)
            return output
        except BaseException as error:
            # Drain valid in-flight work before making append slots reusable.
            try:
                prefetch = getattr(self, "_pool_prefetch", None)
                if prefetch is not None:
                    prefetch.drain()
                self.synchronize()
            except RuntimeError as synchronization_error:
                self._poisoned = True
                error.add_note(
                    f"backend poisoned; draining GPU work failed: {synchronization_error}"
                )
                raise error from synchronization_error
            for cache in started:
                if cache._step_end is not None:
                    cache.rollback()
            if not transient_candidate:
                for runner, offset in zip(session.runners, offsets):
                    if offset is not None:
                        runner.offset.copy_(offset)
            raise
        finally:
            for block in self.blocks:
                block.attention = block.cache = None

    def prefill(self, session, token_ids):
        self._check_session(session)
        if session.length:
            raise ValueError("prefill requires an empty user session")
        return self._forward(session, token_ids, chunk_size=self.chunk_size, _capture_prefix=True)

    def extend(self, session, token_ids):
        self._check_session(session)
        self._check_candidate_limit(token_ids)
        return self._forward(
            session, token_ids, chunk_size=getattr(self, "extend_chunk_size", None)
        )

    def extend_candidate(self, session, token_ids):
        """Execute a GR suffix without committing KV or indexer state to history."""
        self._check_session(session)
        self._check_candidate_limit(token_ids)
        if not self._transient_candidates_enabled():
            # Resident/dense adapters keep their existing storage policy.
            return self.extend(session, token_ids)
        if (
            self._resource_plan is None
            or session.length != session.prefix_length
            or not session.length
        ):
            raise ValueError("transient candidates require a planned, committed fixed history")
        return self._forward(
            session,
            token_ids,
            _transient_candidate=True,
        )

    def _check_candidate_limit(self, token_ids):
        planned = getattr(self, "_resource_plan", None)
        limit = None if planned is None else planned.metadata.get("max_candidate_tokens")
        if limit is not None and len(token_ids) > limit:
            raise CacheBudgetExceeded("candidate suffix exceeds the planned candidate limit")

    def truncate(self, session, prefix_tokens):
        self._check_session(session)
        if self._execution_active():
            raise RuntimeError("cannot truncate a session during execution")
        if not 0 <= prefix_tokens <= session.length:
            raise ValueError("truncation cannot extend a prefix")
        if (
            session.last_candidate_transient
            and prefix_tokens == session.length == session.prefix_length
            and all(
                cache.length == cache.written == cache.indexer_visible_end == prefix_tokens
                and cache._step_end is None
                and cache.transient_start is None
                and getattr(cache, "_prefetch", None) is None
                for runner in session.runners
                for cache in (runner.cache,)
            )
        ):
            # Successful transient execution has already drained its consumers,
            # discarded every suffix and restored the retained offset tensors.
            return
        self.synchronize()
        for runner in session.runners:
            runner.cache.truncate(prefix_tokens)
        if prefix_tokens == session.prefix_length:
            for runner, offset in zip(session.runners, session.prefix_offsets, strict=True):
                if offset is not None:
                    runner.offset.copy_(offset)
        else:
            for runner in session.runners:
                if hasattr(runner, "offset"):
                    runner.offset.zero_()
        session.length = prefix_tokens

    def session_metrics(self, session):
        self._check_session(session)
        metrics = [runner.cache.metrics() for runner in session.runners]
        selected = sum(item.get("selection_records", 0) for item in metrics)
        resident = sum(item.get("resident_selection_records", 0) for item in metrics)
        return {
            **{
                key: sum(
                    getattr(runner.cache, "dense_history_metrics", {}).get(key, 0)
                    for runner in session.runners
                )
                for key in (
                    "dense_requested_records",
                    "dense_resident_records",
                    "dense_fetched_records",
                )
            },
            "candidate_persistence": "gpu_transient"
            if session.last_candidate_transient
            else "committed",
            "retained_length": session.length,
            "candidate_device_to_host_bytes": (
                sum(item["device_to_host_bytes"] for item in metrics)
                if session.last_candidate_transient
                else None
            ),
            "host_to_device_bytes": sum(item["host_to_device_bytes"] for item in metrics),
            "device_to_host_bytes": sum(item["device_to_host_bytes"] for item in metrics),
            "prefetched_records": sum(item["prefetched_records"] for item in metrics),
            "recalled_records": sum(item["recalled_records"] for item in metrics),
            "capacity_splits": sum(item.get("capacity_splits", 0) for item in metrics),
            "evicted_records": sum(item["evicted_records"] for item in metrics),
            "selection_records": selected,
            "resident_selection_records": resident,
            "hbm_token_hit_ratio_before_recall": resident / selected if selected else None,
            "hit_ratio_stage": "exact_consumer_union_after_fused_prefetch_and_append",
            "layer_diagnostics": [
                getattr(runner, "cache_diagnostics", None) for runner in session.runners
            ],
            "layers": metrics,
        }

    def release_session(self, session):
        self._check_session(session)
        if self._execution_active():
            raise RuntimeError("cannot release a session during execution")
        self.synchronize()
        if session.sparse_session is not None:
            session.sparse_session.release()
        if hasattr(self, "_sessions"):
            self._sessions = [item for item in self._sessions if item is not session]
        session.prefix_offsets.clear()
        session.runners.clear()
        session.stages.clear()
        session.stage_consumed.clear()
        session.copy_stream = None
        session.dense_allocation_ready = None
        session.released = True

    def bind_owner(self, owner):
        if owner is None:
            raise ValueError("admission requires a non-None owner identity")
        if getattr(self, "_poisoned", False):
            raise RuntimeError("cannot bind a poisoned backend")
        if getattr(self, "_admission_owner", None) is not None:
            raise RuntimeError("backend already has an admission owner")
        if getattr(self, "_sessions", []) or self._execution_active():
            raise RuntimeError("cannot bind admission with live direct sessions or execution")
        self._admission_owner = owner
        self._admission_initial_plan = self._resource_plan

    def unbind_owner(self, owner, *, rollback=False):
        if owner is None or getattr(self, "_admission_owner", None) is not owner:
            raise ValueError("foreign admission owner")
        if getattr(self, "_sessions", []) or self._execution_active():
            raise RuntimeError("release sessions and execution before unbinding admission")
        if getattr(self, "_poisoned", False):
            raise RuntimeError("cannot unbind a poisoned backend")
        if rollback and self._admission_initial_plan is None:
            self._release_shared()
        self._admission_owner = self._admission_initial_plan = None

    def _release_shared(self):
        if getattr(self, "_poisoned", False):
            raise RuntimeError("cannot release a poisoned backend's shared storage")
        pool = getattr(self, "_shared_pool", None)
        staging = getattr(self, "_dense_staging", None)
        graphs = getattr(self, "_compute_graphs", None)
        try:
            prefetch = getattr(self, "_pool_prefetch", None)
            if prefetch is not None:
                prefetch.close()
            if (
                pool is not None
                or staging is not None
                or graphs is not None
                or self._resource_plan is not None
                or getattr(self, "_allocation_failure", None) is not None
            ):
                self.synchronize()
            if pool is not None:
                pool.close()
            if staging is not None:
                staging.close()
            if graphs is not None:
                graphs.close()
        except BaseException as error:
            self._poisoned = True
            raise RuntimeError("unable to drain the backend's shared storage") from error
        self._shared_pool = self._dense_staging = self._resource_plan = None
        self._pool_prefetch = None
        self._compute_graphs = None
        self._merged_index_keys = self._merged_index_scales = None
        self._dense_staging_allocated_bytes = 0
        self._allocation_failure = None
        self._dense_sources = []

    def close(self):
        if (
            getattr(self, "_sessions", [])
            or self._execution_active()
            or getattr(self, "_admission_owner", None) is not None
        ):
            raise RuntimeError("release all sessions, execution and admission owner before close")
        self._release_shared()

    def configure_scheme(self, scheme):
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        self.close()
        self.scheme = scheme

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def describe(self):
        tensors = [self.embedding_weight, self.head_weight, self.final_norm]
        for attention, block in zip(self.attentions, self.blocks, strict=True):
            for item in (attention, block, block.mlp):
                for value in vars(item).values():
                    if isinstance(value, torch.Tensor):
                        tensors.append(value)
                    elif hasattr(value, "weight") and isinstance(value.weight, torch.Tensor):
                        tensors.extend((value.weight, getattr(value, "scales", None)))
        return {
            "model": "DeepSeek-V3.2-dense-layer-input-replay",
            "checkpoint": str(self.path),
            "scheme": self.scheme,
            "device": str(self.device),
            "physical_layers": self.num_layers,
            "source_layers": [layer % 3 for layer in range(self.num_layers)],
            "input_semantics": "copy_source_layer_hidden_and_residual_for_each_physical_copy",
            "model_scope": "checkpoint-backed workload surrogate; not a trained 8B model",
            "linear_backend": self.linear_backend,
            "compute_graphs": self._compute_graphs.describe()
            if getattr(self, "_compute_graphs", None) is not None
            else {"enabled": getattr(self, "enable_compute_graphs", False), "allocated": False},
            "chunk_size": self.chunk_size,
            "sparse_pool_tokens": self.slots,
            "pool_scope": "backend_per_layer" if self._uses_shared_token_pool() else self.scheme,
            "cache_policy_revision": CACHE_POLICY_REVISION,
            "echo_flags": {
                "fused_logits_recall_extend": self.scheme == "echo",
                "early_evict": False,
                "offset_policy": "mean_of_last_up_to_four_rows_with_finite_guard",
                "exact_union_overflow": "local_query_consumption_split_without_selection_clipping",
            },
            "extend_chunk_size": self.extend_chunk_size,
            "resource_plan": dict(self._resource_plan.metadata) if self._resource_plan else None,
            "dense_staging_allocated_bytes": getattr(self, "_dense_staging_allocated_bytes", 0),
            "weights": _storage_bytes(tensors),
            "output": "all_candidate_normalized_hidden_and_last_token_lm_head",
            "cache_budget_scope": "shared pools, global maps, host arenas, indexer/logits/topk, page tables, staging and pending copies",
            "cache_budget_excludes": "weights, ordinary model activations and GEMM workspace",
            **self.parameter_counts,
        }
