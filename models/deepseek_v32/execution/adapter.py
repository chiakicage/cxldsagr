"""Single-GPU, checkpoint-backed dense-layer replay model for GR serving.

This is a workload surrogate, not a trained smaller DeepSeek model. Every
physical layer owns an independent copy of one of checkpoint layers 0, 1, 2.
Its input is copied from that source layer's input in the first three layers,
including the separate residual stream. Copies therefore do not invent a
deeper residual trajectory. Their KV/indexer states are independent.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from cache.capacity import ResourcePlan, SessionPlan
from cache.host_allocation import (
    pinned_allocation_bytes,
)
from cache.lifecycle import ResourceLifecycle
from cache.prefix_pool import CacheBudgetExceeded
from cache.staging import DoubleBufferStaging
from models.deepseek_v32.cache.session import (
    DeepSeekServingSession,
    DenseCache,
    ServingSparseTokenCache,
)
from models.deepseek_v32.checkpoint import CheckpointReader
from models.deepseek_v32.config import Config
from models.deepseek_v32.execution.cache_resources import (
    CACHE_POLICY_REVISION,
    check_dense_staging_allocation,
    padded_tokens,
)
from models.deepseek_v32.execution.pipeline import LocalPipeline
from models.deepseek_v32.execution.planning import (
    ServingResourceConfig,
    SessionStoragePlan,
    plan_serving_resources,
    plan_session_storage,
)
from models.deepseek_v32.inputs import prepare_token_ids
from models.deepseek_v32.nonmatrix import residual_rms_norm
from models.deepseek_v32.replay import ReplayLayout, replay_parameter_count

SCHEMES = ("hbm", "echo", "serial_sparse", "dense_prefetch")


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
        num_layers,
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
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        self.pipeline = LocalPipeline()
        self.pipeline.validate_scheme(scheme)
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
        self.lifecycle = ResourceLifecycle("DeepSeek")
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
        self.replay_layout = ReplayLayout.repeated_sources(num_layers)
        self.blocks = self.replay_layout.load_blocks(
            model_path,
            self.device,
            self.reader,
            chunk_size=chunk_size,
            linear_backend=linear_backend,
        )
        self.attentions = [block.attention_layer for block in self.blocks]
        self.last_logits = None
        self.capture_hook = None
        self.synchronize()

    def runtime_driver(self, policy):
        from executor.adapters import BackendAdapter

        return BackendAdapter(
            self,
            shared=True,
            owner_aware=True,
            candidate_mode=(
                "gpu_transient"
                if self.scheme in ("echo", "serial_sparse") or policy.mode == "fixed_pools"
                else "append_truncate"
            ),
            session_length=lambda session: session.length,
            chunk_size=self.chunk_size,
            run_lm_head=True,
            last_logits=lambda: self.last_logits,
            diagnostics=lambda session: self.session_metrics(session),
            session_planner=self.plan_runtime_session,
            session_factory=self.create_planned_session,
        )

    def _planning_config(self):
        return ServingResourceConfig(
            scheme=self.scheme,
            host_arena_tokens=self.host_arena_tokens,
            max_seq_len=self.max_seq_len,
            slots=self.slots,
            cfg=self.cfg,
            workspace_query_tokens=self.workspace_query_tokens,
            extend_chunk_size=self.extend_chunk_size,
            chunk_size=self.chunk_size,
            num_layers=self.num_layers,
            device=self.device,
            enable_compute_graphs=getattr(self, "enable_compute_graphs", False),
            compute_graph_private_limit_bytes=getattr(
                self, "compute_graph_private_limit_bytes", 12 * 2**30
            ),
        )

    def plan_resources(self, budgets, limits):
        config = self._planning_config()
        return plan_serving_resources(config, budgets, limits)

    def allocate_shared(self, plan, *, owner=None):
        self.lifecycle.check_access(owner, allow_closed=True)
        if not isinstance(plan, ResourcePlan):
            raise TypeError("plan must be a ResourcePlan")
        if self.scheme == "dense_prefetch" and not plan.metadata.get("shared_token_pool"):
            expected = self.plan_resources(plan.shared, plan.metadata)
            if plan != expected:
                raise ValueError("shared dense plan does not match this backend's reservation")
        if self.lifecycle.plan == plan:
            return
        if self.lifecycle.plan is not None or self._shared_pool is not None:
            raise RuntimeError("close the prior shared plan before changing its plan")
        if self.lifecycle.sessions or self._execution_active():
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
                    dense_contiguous=self.scheme == "dense_prefetch",
                )
                if self.scheme == "dense_prefetch":
                    from models.deepseek_v32.cache.prefetch import PoolHistoryPrefetch

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
                from models.deepseek_v32.execution.compute_graphs import (
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
            self.lifecycle.allocated(plan, owner=owner)
        except BaseException as error:
            # The traceback retains a partially constructed pool/staging object
            # if its constructor failed after enqueueing allocation work.
            self._allocation_failure = error
            try:
                self._release_shared()
            except BaseException as cleanup_error:  # noqa: BLE001 -- preserve both failures
                raise BaseExceptionGroup(
                    "DeepSeek allocation and cleanup both failed", [error, cleanup_error]
                ) from None
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
        prefetch = getattr(self, "_pool_prefetch", None)
        if prefetch is not None:
            pending["hbm" if self.device.type == "cuda" else "dram"] += prefetch.pending_bytes
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
        plan = self.lifecycle.plan
        return self.scheme in ("echo", "serial_sparse") or (
            plan is not None and plan.metadata.get("shared_token_pool", False)
        )

    def _transient_candidates_enabled(self):
        plan = self.lifecycle.plan
        return self.scheme in ("echo", "serial_sparse") or (
            plan is not None and plan.metadata.get("candidate_persistence") == "gpu_transient"
        )

    def estimate_session_host_pages(self, capacity):
        shared = self._uses_shared_token_pool() or (
            self.scheme == "dense_prefetch" and self.lifecycle.plan is None
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

    def _plan_session_storage(self, capacity, prefix_tokens=0):
        if not 0 <= prefix_tokens <= capacity:
            raise ValueError("prefix length must fit session capacity")
        candidate_slots = (
            self.lifecycle.plan.metadata["candidate_slots"]
            if self.scheme == "hbm" and self._transient_candidates_enabled()
            else 0
        )
        return plan_session_storage(
            self._planning_config(),
            capacity,
            shared_pool=self._uses_shared_token_pool(),
            candidate_slots=candidate_slots,
            resource_identity=self.lifecycle.plan,
            generation=self.lifecycle.generation,
            history_tokens=prefix_tokens,
        )

    def estimate_session_bytes(self, capacity, prefix_tokens=0):
        plan = self._plan_session_storage(capacity, prefix_tokens)
        return {"hbm": plan.reservation.hbm, "dram": plan.reservation.dram}

    def plan_runtime_session(self, resources, shape, identity):
        resources.limits.validate(shape)
        transient = (
            self.scheme in ("echo", "serial_sparse") or resources.policy.mode == "fixed_pools"
        )
        capacity = shape.history_tokens if transient else shape.total_tokens
        storage = self._plan_session_storage(capacity, shape.history_tokens)
        # Bind the admitted quotas without recalculating the retained storage.
        storage = replace(
            storage,
            host_pages=storage.host_pages if resources.host_pages else 0,
            hbm_tokens=storage.hbm_tokens if resources.hbm_tokens else 0,
        )
        return SessionPlan(
            resources,
            identity,
            shape.history_tokens,
            capacity,
            storage.reservation,
            storage.host_pages,
            storage.hbm_tokens,
            storage.allocations,
            storage,
        )

    def create_planned_session(self, plan, *, owner=None):
        if not isinstance(plan, SessionPlan) or not isinstance(plan.model_plan, SessionStoragePlan):
            raise TypeError("DeepSeek session allocation requires its native SessionPlan")
        storage = plan.model_plan
        if (
            plan.retained_capacity != storage.capacity
            or plan.reservation != storage.reservation
            or plan.allocations != storage.allocations
            or plan.history_tokens != storage.history_tokens
            or plan.host_pages != storage.host_pages
            or plan.hbm_tokens != storage.hbm_tokens
        ):
            raise ValueError("session reservation differs from its native storage plan")
        return self._create_session_from_plan(storage, owner=owner)

    def create_session(self, capacity, *, owner=None):
        return self._create_session_from_plan(self._plan_session_storage(capacity), owner=owner)

    def _create_session_from_plan(self, storage, *, owner=None):
        self.lifecycle.check_access(owner, allow_closed=self.scheme == "hbm")
        if (
            storage.resource_identity is not self.lifecycle.plan
            or storage.generation != self.lifecycle.generation
            or storage.scheme != self.scheme
        ):
            raise ValueError("session storage plan belongs to a different resource generation")
        capacity = storage.capacity
        if self.lifecycle.poisoned or self._execution_active():
            raise RuntimeError("cannot create a session during execution or after a CUDA failure")
        plan = self.lifecycle.plan
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
            if capacity > self.lifecycle.plan.metadata["max_session_capacity"]:
                raise CacheBudgetExceeded("session exceeds the reserved workspace context bound")
            sparse_session = self._shared_pool.allocate_session(capacity)
        try:
            runners = [
                self.pipeline.create_runner(
                    attention,
                    capacity,
                    scheme=self.scheme,
                    slots=self.slots,
                    chunk_size=self.chunk_size,
                    dense_backend=self if self.scheme == "dense_prefetch" else None,
                    cache=(
                        self.pipeline.layer_cache(sparse_session, layer)
                        if sparse_session is not None
                        else ServingSparseTokenCache(
                            capacity,
                            self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim,
                            device=self.device,
                            candidate_slots=storage.candidate_slots,
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
            for runner in runners:
                runner.cache.bind_serving(session, self.lifecycle)
            if self.scheme == "dense_prefetch" and sparse_session is None:
                for runner in runners:
                    runner.cache.session = session
                if self.device.type == "cuda":
                    session.dense_allocation_ready = torch.cuda.Event()
                    session.dense_allocation_ready.record(torch.cuda.current_stream(self.device))
            session.storage_plan = storage
            session.registration = self.lifecycle.register(
                session,
                owner=owner,
                allow_unplanned=self.scheme == "hbm" and self.lifecycle.plan is None,
            )
            if sparse_session is not None:
                sparse_session.bind_release_guard(session.check_backing_release)
        except BaseException as error:
            self._allocation_failure = error
            if self.scheme == "dense_prefetch" and not self._uses_shared_token_pool():
                try:
                    self.synchronize()
                except BaseException as drain_error:  # noqa: BLE001 -- retain failed allocation
                    self.lifecycle.poison(drain_error)
                    raise BaseExceptionGroup(
                        "DeepSeek session allocation and drain both failed", [error, drain_error]
                    ) from None
            if sparse_session is not None:
                try:
                    sparse_session.release()
                except BaseException as release_error:  # noqa: BLE001 -- preserve both failures
                    self.lifecycle.poison(release_error)
                    raise BaseExceptionGroup(
                        "DeepSeek session allocation and release both failed",
                        [error, release_error],
                    ) from None
            self._allocation_failure = None
            raise
        return session

    def _check_session(self, session):
        if self.lifecycle.poisoned:
            raise RuntimeError("backend is poisoned after an asynchronous CUDA failure")
        if session.owner is not self:
            raise ValueError("session belongs to another backend")
        self.lifecycle.check_session(session, session.registration, released=session.released)
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
        if session is None or self.lifecycle.active_session is not session:
            raise RuntimeError("dense cache requires its session's execution lease")
        lease = getattr(self, "_dense_lease", None)
        if lease is None or lease.closed:
            raise RuntimeError("dense staging execution lease is unavailable")
        lease._check()

    def _execution_active(self):
        staging = getattr(self, "_dense_staging", None)
        return (
            self.lifecycle.active_session is not None
            or getattr(self, "_dense_lease", None) is not None
            or staging is not None
            and staging.active_lease is not None
        )

    def _reserve_dense_source(self):
        self._check_dense_execution(self.lifecycle.active_session)
        pending = self._dense_sources
        while pending and (self.device.type == "cpu" or pending[0].event.query()):
            pending.pop(0)
        limit = self.lifecycle.plan.metadata["max_inflight_writes"]
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

    def _prepare_execution(self, session):
        if self.scheme == "dense_prefetch" and not self._uses_shared_token_pool():
            staging = getattr(self, "_dense_staging", None)
            if staging is None:
                raise RuntimeError("dense execution requires allocated shared staging")
            if session.dense_allocation_ready is not None:
                try:
                    torch.cuda.current_stream(self.device).wait_event(
                        session.dense_allocation_ready
                    )
                except BaseException as error:
                    self.lifecycle.poison(error)
                    raise
            try:
                self._dense_lease = staging.lease(session)
            except BaseException as error:
                if staging.poisoned:
                    self.lifecycle.poison(error)
                    self._dense_lease = staging.active_lease
                raise

    def _drain_execution(self, session):
        graphs = getattr(self, "_compute_graphs", None)
        if graphs is not None and graphs.failed:
            self.lifecycle.poison()
        errors = []
        prefetch = getattr(self, "_pool_prefetch", None)
        if prefetch is not None:
            try:
                prefetch.drain()
            except BaseException as error:  # noqa: BLE001 -- attempt every required drain
                errors.append(error)
        lease = getattr(self, "_dense_lease", None)
        if lease is not None:
            try:
                lease.close()
            except BaseException as error:  # noqa: BLE001 -- retain every completion failure
                errors.append(error)
        staging = getattr(self, "_dense_staging", None)
        if staging is not None and staging.poisoned:
            self.lifecycle.poison()
        if errors:
            self.lifecycle.poison(errors[0])
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("DeepSeek providers failed to drain", errors)
        if not self.lifecycle.poisoned:
            for runner in session.runners:
                if isinstance(runner.cache, DenseCache):
                    runner.cache.records = None
            self._dense_lease = None
            self._dense_sources = []

    @contextmanager
    def _execution(self, session, *, owner=None):
        self._check_session(session)
        if self._execution_active():
            raise RuntimeError("the single-GPU backend executes one request at a time")
        with self.lifecycle.execution(
            session,
            session.registration,
            owner=owner,
            prepare=lambda: self._prepare_execution(session),
            drain=lambda: self._drain_execution(session),
        ):
            graphs = getattr(self, "_compute_graphs", None)
            with graphs.execution() if graphs is not None else nullcontext():
                yield

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
        owner=None,
    ):
        # This lease covers all prefill chunks, commit/abort, final LM head, and
        # the retained prefix hint snapshot, including direct profiler calls.
        self._check_session(session)
        if not _capture_prefix:
            self._check_candidate_limit(token_ids)
        with self._execution(session, owner=owner), ExitStack() as hints:
            for runner in session.runners:
                defer = getattr(runner, "defer_unused_prefill_hints", None)
                if defer is not None:
                    hints.enter_context(defer())
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
        ticket.add_metrics(cache.dense_history_metrics)
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
        planned = self.lifecycle.plan
        limit = (
            planned.metadata["max_session_capacity"] if transient_candidate else session.capacity
        )
        ids = prepare_token_ids(
            token_ids,
            device=self.device,
            vocab_size=self.cfg.vocab_size,
            max_tokens=limit - session.length,
        )
        query_chunk = len(ids) if chunk_size is None else chunk_size
        if planned is not None and query_chunk > planned.metadata["workspace_query_tokens"]:
            raise CacheBudgetExceeded("query batch exceeds reserved cache execution workspace")
        scope = scope or (lambda _: nullcontext())
        started = []
        offsets = []
        try:
            for runner in session.runners:
                offsets.append(
                    runner.offset.clone()
                    if hasattr(runner, "offset") and getattr(runner, "mutates_prefetch_hint", True)
                    else None
                )
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
                replay_inputs = self.replay_layout.begin_chunk()
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
                    graphs = getattr(self, "_compute_graphs", None)
                    source_inputs = (
                        (hidden, residual) if layer < 3 else replay_inputs.sources[layer % 3]
                    )
                    graph_used = graphs is not None and graphs.supports(layer, *source_inputs)
                    hidden, residual = replay_inputs.for_layer(
                        layer, hidden, residual, scope=scope, copy_source=not graph_used
                    )
                    indexer_view = (
                        self._candidate_indexer_view(session, runner)
                        if transient_candidate
                        else nullcontext()
                    )
                    candidate_lease = (
                        runner.cache.operation() if transient_candidate else nullcontext()
                    )
                    with scope(f"layer_{layer}_source_{layer % 3}"), candidate_lease, indexer_view:
                        if graph_used:
                            hidden, residual = graphs.forward_block(
                                layer, runner, hidden, residual, scope=scope
                            )
                        else:
                            if graphs is not None:
                                graphs.eager_fallbacks += 1
                            hidden, residual = block.forward(
                                hidden,
                                residual,
                                scope=scope,
                                attention=runner,
                                chunk_size=len(hidden),
                            )
                    if self.capture_hook is not None:
                        if graph_used:
                            # A retained diagnostic reference must survive later replay.
                            self.capture_hook(layer, hidden.clone(), residual.clone())
                        else:
                            self.capture_hook(layer, hidden, residual)
                    if dense:
                        runner.cache.records = None
                with scope("final_norm_lm_head"):
                    output, _ = residual_rms_norm(
                        hidden, residual, self.final_norm, self.cfg.norm_eps
                    )
                    output_parts.append(output)
                    if stop == len(ids):
                        self.last_logits = F.linear(output[-1:], self.head_weight).float()
                if dense and stop < len(ids):
                    self._dense_lease.reset()
                    self._dense_sources.clear()
                if pool_dense is not None:
                    pool_dense.drain()
            output = output_parts[0] if len(output_parts) == 1 else torch.cat(output_parts)
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
            except BaseException as synchronization_error:  # noqa: BLE001 -- preserve both failures
                self.lifecycle.poison(synchronization_error)
                raise BaseExceptionGroup(
                    "DeepSeek execution and synchronization both failed",
                    [error, synchronization_error],
                ) from None
            rollback_errors = []
            for cache in started:
                if cache._step_end is not None:
                    try:
                        cache.rollback()
                    except BaseException as rollback_error:  # noqa: BLE001 -- preserve body failure
                        rollback_errors.append(rollback_error)
            if not transient_candidate:
                for runner, offset in zip(session.runners, offsets):
                    if offset is not None:
                        try:
                            runner.offset.copy_(offset)
                        except BaseException as restore_error:  # noqa: BLE001 -- preserve all failures
                            rollback_errors.append(restore_error)
            if rollback_errors:
                self.lifecycle.poison(rollback_errors[0])
                raise BaseExceptionGroup(
                    "DeepSeek execution and rollback both failed", [error, *rollback_errors]
                ) from None
            raise

    def prefill(self, session, token_ids, *, owner=None):
        self._check_session(session)
        if session.length:
            raise ValueError("prefill requires an empty user session")
        return self._forward(
            session, token_ids, chunk_size=self.chunk_size, _capture_prefix=True, owner=owner
        )

    def extend(self, session, token_ids, *, owner=None):
        self._check_session(session)
        self._check_candidate_limit(token_ids)
        return self._forward(
            session, token_ids, chunk_size=getattr(self, "extend_chunk_size", None), owner=owner
        )

    def extend_candidate(self, session, token_ids, *, owner=None):
        """Execute a GR suffix without committing KV or indexer state to history."""
        self._check_session(session)
        self._check_candidate_limit(token_ids)
        if not self._transient_candidates_enabled():
            # Resident/dense adapters keep their existing storage policy.
            return self.extend(session, token_ids, owner=owner)
        if (
            self.lifecycle.plan is None
            or session.length != session.prefix_length
            or not session.length
        ):
            raise ValueError("transient candidates require a planned, committed fixed history")
        return self._forward(
            session,
            token_ids,
            _transient_candidate=True,
            owner=owner,
        )

    def _check_candidate_limit(self, token_ids):
        planned = self.lifecycle.plan
        limit = None if planned is None else planned.metadata.get("max_candidate_tokens")
        if limit is not None and len(token_ids) > limit:
            raise CacheBudgetExceeded("candidate suffix exceeds the planned candidate limit")

    def truncate(self, session, prefix_tokens, *, owner=None):
        self._check_session(session)
        if self._execution_active():
            raise RuntimeError("cannot truncate a session during execution")
        with self.lifecycle.mutation(session, session.registration, owner=owner):
            self._truncate_leased(session, prefix_tokens)

    def _truncate_leased(self, session, prefix_tokens):
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
        try:
            self.synchronize()
            for runner in session.runners:
                runner.cache.truncate(prefix_tokens)
        except BaseException as error:
            self.lifecycle.poison(error)
            raise
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
        return self.pipeline.session_metrics(session)

    def release_session(self, session, *, owner=None):
        self._check_session(session)
        if self._execution_active():
            raise RuntimeError("cannot release a session during execution")
        with self.lifecycle.mutation(session, session.registration, owner=owner, releasing=True):
            try:
                self.synchronize()
                if session.sparse_session is not None:
                    session.sparse_session.release()
            except BaseException as error:
                self.lifecycle.poison(error)
                raise
            session.prefix_offsets.clear()
            session.runners.clear()
            session.stages.clear()
            session.stage_consumed.clear()
            session.copy_stream = None
            session.dense_allocation_ready = None
            session.released = True
            self.lifecycle.detach(session)

    def bind_owner(self, owner):
        if self._execution_active():
            raise RuntimeError("cannot bind admission with active execution")
        self.lifecycle.bind_owner(owner)

    def unbind_owner(self, owner, *, rollback=False):
        if self._execution_active():
            raise RuntimeError("release execution before unbinding admission")
        self.lifecycle.unbind_owner(owner, rollback=rollback, release_new=self._release_shared)

    def _release_shared(self):
        if self.lifecycle.poisoned:
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
                or self.lifecycle.plan is not None
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
            self.lifecycle.poison(error)
            raise
        self.lifecycle.drop_plan()
        self._shared_pool = self._dense_staging = None
        self._pool_prefetch = None
        self._compute_graphs = None
        self._merged_index_keys = self._merged_index_scales = None
        self._dense_staging_allocated_bytes = 0
        self._allocation_failure = None
        self._dense_sources = []

    def close(self):
        if self._execution_active():
            raise RuntimeError("release all sessions and execution before close")
        self.lifecycle.close(self._release_shared)

    def configure_scheme(self, scheme):
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        self.pipeline.validate_scheme(scheme)
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
        result = {
            "model": "DeepSeek-V3.2-dense-layer-input-replay",
            "checkpoint": str(self.path),
            "scheme": self.scheme,
            "device": str(self.device),
            "physical_layers": self.num_layers,
            "source_layers": list(self.replay_layout.source_layers),
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
            "resource_plan": dict(self.lifecycle.plan.metadata) if self.lifecycle.plan else None,
            "dense_staging_allocated_bytes": getattr(self, "_dense_staging_allocated_bytes", 0),
            "weights": _storage_bytes(tensors),
            "output": "all_candidate_normalized_hidden_and_last_token_lm_head",
            "cache_budget_scope": "shared pools, global maps, host arenas, indexer/logits/topk, page tables, staging and pending copies",
            "cache_budget_excludes": "weights, ordinary model activations and GEMM workspace",
            **self.parameter_counts,
        }
        return result
