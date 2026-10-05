"""Persistent NOSA sessions for the serial multi-user GR serving runner.

The shared serving cache manager owns admission and eviction. This adapter
owns model-specific cache layout and execution; it never reads GR requests.
All schemes execute complete NOSA sparse attention, including CIS. ``dense``
describes the volume transferred, not a different attention mask.
"""

from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType

import torch

from models.nosa.attention import NosaDensePrefetchAttention
from models.nosa.cache.dense_prefetch import NosaDensePrefetchCache
from models.nosa.cache.offload import NosaOffloadCache
from models.nosa.cache.resident import NosaKVCache
from models.nosa.execution.resources import POLICY_REVISION, NosaExecutionResources
from models.nosa.model import NosaForCausalLM


class NosaServingBackend:
    """Full NOSA model with persistent, independently evictable user sessions.

    ``estimate_session_bytes`` reserves cache allocation blocks, including owned
    pending appends, indexer validation/selection and CIS helper temporaries.
    Model weights, ordinary QKV activations and attention outputs are separate.
    There is no per-request weight loading, CPU snapshotting, or prefix replay on
    a hit. Caller-owned LRU eviction discards both tiers of one user's session.
    """

    schemes = ("hbm", "serial_sparse", "dense_prefetch", "overlap")

    def __init__(self, model, scheme, *, chunk_size=1024):
        if scheme not in self.schemes:
            raise ValueError(f"NOSA serving scheme must be one of {self.schemes}")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if model.attention_mode != "sparse":
            raise ValueError("All NOSA serving schemes require the complete sparse NOSA policy")
        self.model = model
        self.scheme = scheme
        self.chunk_size = chunk_size
        self.config = model.config
        self.max_seq_len = self.config.max_position_embeddings
        self.device = model.model.embed_tokens.weight.device
        self.dtype = model.model.embed_tokens.weight.dtype
        self.compute_graphs = None
        self._session_layout_cache = None
        self._dense_attention = NosaDensePrefetchAttention()
        cfg = self.config
        self.resources = NosaExecutionResources(
            scheme=scheme,
            layers=cfg.num_hidden_layers,
            kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            query_heads=cfg.num_attention_heads,
            max_seq_len=self.max_seq_len,
            chunk_size=chunk_size,
            dtype=self.dtype,
            device=self.device,
        )

    def runtime_driver(self, policy):
        from executor.adapters import BackendAdapter

        if policy.mode != "budget":
            raise ValueError("Ordinary NOSA serving requires byte-budget capacity")
        return BackendAdapter(
            self,
            shared=True,
            candidate_mode="append_truncate",
            session_length=lambda session: session.length,
            chunk_size=self.chunk_size,
            diagnostics=lambda session: self.session_metrics(session),
            owner_aware=True,
            session_planner=self.plan_runtime_session,
            session_factory=self.create_planned_session,
        )

    @classmethod
    def from_pretrained(
        cls, checkpoint, *, scheme, device="cuda:0", chunk_size=1024, max_seq_len=None
    ):
        model = NosaForCausalLM.from_pretrained(
            checkpoint,
            device=device,
            dtype=torch.bfloat16,
            attention_mode="sparse",
            sparse_backend="auto",
        )
        if (model.config.num_hidden_layers, len(model.model.layers)) != (32, 32):
            raise ValueError("NOSA-8B serving requires all 32 checkpoint layers")
        if max_seq_len is not None:
            if type(max_seq_len) is not int or not 0 < max_seq_len <= 262144:
                raise ValueError("max_seq_len must be an integer in [1,262144]")
            model.config = replace(model.config, max_position_embeddings=max_seq_len)
        backend = cls(model, scheme, chunk_size=chunk_size)
        backend.checkpoint = str(checkpoint)
        return backend

    def _validate_capacity(self, capacity):
        self.resources.validate_capacity(capacity)

    def plan_resources(self, budgets, limits):
        if (
            self.device.type == "cuda"
            and getattr(self.model.main_attention, "backend", None) == "reference"
        ):
            raise NotImplementedError("Shared CUDA serving requires native attention")
        plan = self.resources.plan_resources(budgets, limits)
        if self.compute_graphs is None:
            return plan
        if budgets is not None:
            raise NotImplementedError("NOSA compute graphs currently require fixed P/NH mode")
        from cache.prefix_pool import CacheFootprint

        self._graph_base_plan = plan
        shared = self.compute_graphs.shared_bytes()["hbm"]
        description = {
            key: value
            for key, value in self.compute_graphs.describe().items()
            if key not in ("project_replays", "finish_replays", "eager_fallbacks")
        }
        self._graph_plan = replace(
            plan,
            shared=CacheFootprint(plan.shared.hbm + shared, plan.shared.dram),
            metadata=MappingProxyType({**plan.metadata, "compute_graphs": description}),
        )
        return self._graph_plan

    def allocate_shared(self, plan, *, owner=None):
        if self.compute_graphs is not None:
            base = getattr(self, "_graph_base_plan", None)
            if base is None or plan != getattr(self, "_graph_plan", None):
                raise ValueError("Compute graph allocation requires its matching resource plan")
            self.resources.allocate_shared(base, owner=owner)
        else:
            self.resources.allocate_shared(plan, owner=owner)

    def shared_bytes(self):
        result = self.resources.shared_bytes()
        if self.compute_graphs is not None:
            result["hbm"] += self.compute_graphs.shared_bytes()["hbm"]
        return result

    def enable_compute_graphs(self, query_sizes, *, private_limit_bytes=8 * 2**30):
        """Capture pure compute before runner admission or session allocation."""
        if (
            self.compute_graphs is not None
            or self.resources.plan is not None
            or self.resources._sessions
        ):
            raise RuntimeError("Enable compute graphs before planning or allocating sessions")
        from models.nosa.execution.compute_graphs import NosaComputeGraphs

        graphs = NosaComputeGraphs(self.model, query_sizes, private_limit_bytes=private_limit_bytes)
        self.compute_graphs = graphs
        try:
            graphs.allocate()
            self.resources.allowed_graph_pool_ids = graphs.pool_ids
        except BaseException as error:
            import traceback

            # Setup tracebacks can retain the last graph pair and private output
            # tensors after capture has unwound. Release those local aliases
            # before checking that close returned its allocator segments.
            traceback.clear_frames(error.__traceback__)
            self.resources.lifecycle.cleanup_after_failure(error, graphs.close)
            self.compute_graphs = None
            raise

    def bind_owner(self, owner):
        self.resources.bind_owner(owner)

    def unbind_owner(self, owner, *, rollback=False):
        self.resources.unbind_owner(owner, rollback=rollback)

    def estimate_session_host_pages(self, capacity):
        self._validate_capacity(capacity)
        return 0

    def session_host_pages(self, session):
        self.resources.check_session(session)
        return 0

    def estimate_session_bytes(self, capacity, prefix_tokens):
        self._validate_capacity(capacity)
        if type(prefix_tokens) is not int or not 0 < prefix_tokens < capacity:
            raise ValueError("Prefix must leave a nonempty candidate suffix")
        cfg = self.config
        queries = max(min(self.chunk_size, prefix_tokens), capacity - prefix_tokens)
        if self.resources.plan is not None:
            metadata = self.resources.plan.metadata
            if capacity - prefix_tokens > metadata["max_candidate_tokens"]:
                raise ValueError("Candidate suffix exceeds planned execution capacity")
            if queries > metadata["max_query_tokens"]:
                raise ValueError("Query length exceeds planned execution capacity")
        from models.nosa.execution.session_budget import estimate_session_bytes

        return estimate_session_bytes(
            capacity=capacity,
            queries=queries,
            layers=cfg.num_hidden_layers,
            kv_heads=cfg.num_key_value_heads,
            query_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
            dtype=self.dtype,
            device=self.device,
            scheme=self.scheme,
        )

    def _session_geometry(self, resource_plan, shape):
        from models.nosa.execution.session_budget import NosaSessionGeometry

        capacity = shape.total_tokens
        self._validate_capacity(capacity)
        return NosaSessionGeometry(
            provider_identity=self.resources.lifecycle.identity,
            generation=self.resources.generation,
            history_tokens=shape.history_tokens,
            capacity=capacity,
            execution_capacity=capacity,
            host_capacity=capacity,
            queries=max(min(self.chunk_size, shape.history_tokens), shape.candidate_tokens),
            scheme=self.scheme,
        )

    def _session_layout_options(self, geometry):
        return {"host_capacity": geometry.host_capacity}

    def plan_runtime_session(self, resource_plan, shape, identity):
        from cache.capacity import CacheFootprint, SessionPlan
        from models.nosa.execution.session_budget import session_allocation_layout

        resource_plan.limits.validate(shape)
        geometry = self._session_geometry(resource_plan, shape)
        if geometry.queries > resource_plan.metadata["max_query_tokens"]:
            raise ValueError("Query length exceeds planned execution capacity")
        cfg = self.config
        layout_options = {
            "capacity": geometry.execution_capacity,
            "queries": geometry.queries,
            "layers": cfg.num_hidden_layers,
            "kv_heads": cfg.num_key_value_heads,
            "query_heads": cfg.num_attention_heads,
            "head_dim": cfg.head_dim,
            "dtype": self.dtype,
            "device": self.device,
            "scheme": self.scheme,
            **self._session_layout_options(geometry),
        }
        layout_key = (
            self.resources.lifecycle.identity,
            self.resources.generation,
            tuple((name, type(value), value) for name, value in layout_options.items()),
        )
        layout = self._session_layout_cache
        if layout is None or layout[0] != layout_key:
            allocations, breakdown = session_allocation_layout(**layout_options)
            # Retain only one pure declaration, never a request or audit result.
            layout = (layout_key, allocations, CacheFootprint.from_mapping(breakdown))
            self._session_layout_cache = layout
        _, allocations, reservation = layout
        geometry = replace(
            geometry,
            allocations=allocations,
            reservation=reservation,
            host_pages=self.estimate_session_host_pages(geometry.capacity)
            if resource_plan.host_pages
            else 0,
            hbm_tokens=self.estimate_session_hbm_tokens(geometry.capacity)
            if resource_plan.hbm_tokens
            else 0,
        )
        return SessionPlan(
            resource_identity=resource_plan,
            history_identity=identity,
            history_tokens=shape.history_tokens,
            retained_capacity=geometry.capacity,
            reservation=geometry.reservation,
            host_pages=geometry.host_pages,
            hbm_tokens=geometry.hbm_tokens,
            allocations=allocations,
            model_plan=geometry,
        )

    def create_planned_session(self, plan, *, owner=None):
        from models.nosa.execution.session_budget import NosaSessionGeometry

        geometry = plan.model_plan
        if (
            not isinstance(geometry, NosaSessionGeometry)
            or geometry.provider_identity is not self.resources.lifecycle.identity
            or geometry.generation != self.resources.generation
            or geometry.scheme != self.scheme
        ):
            raise ValueError("Session plan belongs to foreign or stale NOSA resources")
        if (
            plan.history_tokens != geometry.history_tokens
            or plan.retained_capacity != geometry.capacity
            or plan.allocations != geometry.allocations
            or plan.reservation != geometry.reservation
            or plan.host_pages != geometry.host_pages
            or plan.hbm_tokens != geometry.hbm_tokens
        ):
            raise ValueError("Session declaration differs from its frozen NOSA allocation plan")
        declared = {item.name: item for item in plan.allocations}
        if declared["cis_scores"].shape[1] != geometry.execution_capacity:
            raise ValueError("Session declaration and constructor geometry differ")
        session = self._allocate_session(
            geometry.capacity, owner=owner, execution_capacity=geometry.execution_capacity
        )
        session._allocation_plan = plan
        return session

    def create_session(self, capacity, *, owner=None):
        return self._allocate_session(capacity, owner=owner, execution_capacity=capacity)

    def _allocate_session(self, capacity, *, owner=None, execution_capacity):
        self.resources.lifecycle.check_access(owner)
        self._validate_capacity(capacity)
        if execution_capacity != capacity:
            raise ValueError("Ordinary NOSA sessions retain their full execution capacity")
        if self.resources.plan is None:
            raise RuntimeError("Plan and allocate shared resources before creating a session")
        options = {
            "device": self.device,
            "dtype": self.dtype,
            "with_cis": True,
            "execution_resources": self.resources,
        }
        if self.scheme == "hbm":
            session = NosaKVCache(self.config, capacity, **options)
        else:
            cache_type = (
                NosaDensePrefetchCache if self.scheme == "dense_prefetch" else NosaOffloadCache
            )
            session = cache_type(
                self.config,
                capacity,
                overlap=self.scheme == "overlap",
                query_tile_size=128,
                **options,
            )
        self.resources.attach(session, owner=owner)
        return session

    @contextmanager
    def execution_lease(self, session, *, phase="candidate", owner=None):
        metrics, started = None, False
        try:
            with self.resources.lease(session, owner=owner):
                metrics = session._transfer_metrics
                started = True
                metrics.begin(phase)
                yield self.resources
        except BaseException:
            if started and metrics._phase is not None:
                metrics.finish(success=False)
            raise
        else:
            metrics.finish(success=True)

    def _validate_ids(self, session, ids, *, prefill):
        self.resources.check_session(session)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or not len(ids):
            raise ValueError("Tokens must be a nonempty one-dimensional tensor")
        if ids.dtype not in (torch.int32, torch.int64) or ids.device != self.device:
            raise ValueError("Tokens must be integral and on the backend device")
        if session.length + len(ids) > session.max_seq_len:
            raise ValueError("Input exceeds session capacity")
        if not prefill and len(ids) > self.resources.plan.metadata["max_candidate_tokens"]:
            raise ValueError("Candidate suffix exceeds planned execution capacity")
        queries = min(self.chunk_size, len(ids)) if prefill else len(ids)
        if queries > self.resources.plan.metadata["max_query_tokens"]:
            raise ValueError("Query length exceeds planned execution capacity")

    def _execution_attention(self):
        from models.nosa.attention import NosaSparseAttention

        return (
            self._dense_attention
            if self.scheme == "dense_prefetch"
            else NosaSparseAttention(workspace=self.resources.attention_workspace)
        )

    def _forward(self, session, ids, *, main_attention=None):
        if main_attention is None:
            main_attention = self._execution_attention()
        if self.compute_graphs is None:
            return self.model(ids, session, return_hidden=True, main_attention=main_attention)
        with self.compute_graphs.execution():
            return self.model(
                ids,
                session,
                return_hidden=True,
                compute_graphs=self.compute_graphs,
                main_attention=main_attention,
            )

    @torch.inference_mode()
    def prefill(self, session, ids, *, owner=None):
        self._validate_ids(session, ids, prefill=True)
        if session.length != 0:
            raise ValueError("Prefix construction requires an empty session")
        if ids.ndim != 1 or not len(ids):
            raise ValueError("Prefix must be a nonempty one-dimensional token tensor")
        with self.execution_lease(session, phase="prefix", owner=owner):
            attention = self._execution_attention()
            for start in range(0, len(ids), self.chunk_size):
                self._forward(
                    session, ids[start : start + self.chunk_size], main_attention=attention
                )

    @torch.inference_mode()
    def extend(self, session, ids, *, owner=None):
        self._validate_ids(session, ids, prefill=False)
        with self.execution_lease(session, owner=owner):
            return self._forward(session, ids)

    def truncate(self, session, prefix, *, owner=None):
        # Resident truncate publishes metadata only, so synchronize before the
        # next user or a possible eviction may reuse/free its tensors.
        with self.resources.mutation(session, owner=owner):
            self._synchronize_mutation()
            session.truncate(prefix)

    def session_storage_bytes(self, session):
        """Logical owning storage bytes, separate from allocator capacity."""
        self.resources.check_session(session, allow_released=True)
        stats = session.stats()
        return {"hbm": stats["resident_bytes"], "dram": stats["host_bytes"]}

    def session_bytes(self, session):
        """HBM tensor storage and owned DRAM capacity for admission checks.

        CUDA allocator padding is covered by the reservation and intrusive
        allocation audit. Pinned host bins have an exact power-of-two capacity,
        which is reported here even when a tensor exposes a shorter storage.
        """
        from cache.allocator.budget import pinned_allocation_bytes

        result = self.session_storage_bytes(session)
        tensors = [*session.buffers.values(), getattr(session, "_native_indexer_host_flag", None)]
        seen = set()
        for tensor in tensors:
            if tensor is None or tensor.device.type != "cpu" or not tensor.is_pinned():
                continue
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in seen:
                seen.add(storage.data_ptr())
                result["dram"] += (
                    pinned_allocation_bytes(storage.nbytes(), "cuda") - storage.nbytes()
                )
        return result

    def session_metrics(self, session):
        """Observe last-request KV payload after execution, outside its timing.

        Prefix construction is retained for its immediately following candidate;
        a later candidate starts a revisit with zero prefix bytes. ``truncate``
        retains these counters. CUDA scalar observation is a diagnostic D2H and
        is excluded from the KV payload fields, as are metadata and link overhead.
        """
        self.resources.check_session(session)
        return session._transfer_metrics.snapshot()

    def release_session(self, session, *, owner=None):
        self.resources.check_session(session, allow_released=True)
        if session.released:
            self.resources.lifecycle.check_access(owner, allow_closed=True)
            return
        self.resources.lifecycle.check_access(owner)
        with self.resources.mutation(session, owner=owner, releasing=True):
            self._synchronize_mutation()
            session.release()

    def _synchronize_mutation(self):
        try:
            self.synchronize()
        except BaseException as error:
            self.resources.lifecycle.poison(error)
            raise

    def close(self):
        self.resources.close()
        if self.compute_graphs is not None:
            try:
                self.compute_graphs.close()
            except BaseException as error:
                self.resources.lifecycle.poison(error)
                raise
            self.compute_graphs = None
            self.resources.allowed_graph_pool_ids = set()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def describe(self):
        return {
            "model": "nosa",
            "checkpoint": getattr(self, "checkpoint", None),
            "scheme": self.scheme,
            "layers": len(self.model.model.layers),
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "dtype": str(self.dtype),
            "device": str(self.device),
            "max_seq_len": self.max_seq_len,
            "prefix_chunk_size": self.chunk_size,
            "policy_revision": POLICY_REVISION,
            "workspace_scope": "backend",
            "host_scope": "session",
            "resource_plan": None
            if self.resources.plan is None
            else dict(self.resources.plan.metadata),
            "shared_cache_bytes": self.shared_bytes(),
            "attention_policy": "full NOSA: 33 QA incl sink/local, then CIS to 64 blocks",
            "output": "all candidate normalized hidden states; no LM head or decode",
            "cache_eviction": "whole-user session; both HBM and DRAM released",
            "cache_budget_scope": "tensor capacities, derived records, staging, owned append, scratch",
            "dense_prefetch": "full next-layer historical K/V on copy stream, two fenced HBM slots",
            "sparse_offload": "native full-batch unique sparse union; overlap uses cooperative FA3",
            "compute_graphs": {"enabled": False}
            if self.compute_graphs is None
            else self.compute_graphs.describe(),
        }
