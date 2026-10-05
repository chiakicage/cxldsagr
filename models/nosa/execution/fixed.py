"""Four-scheme NOSA serving under fixed P and NH history quotas.

P is the retained HBM token quota for HBM-only and the number of historical
token slots in every offload layer. NH is a global retained host-token quota;
host backing is allocated lazily for admitted sessions. Complete histories
must be page aligned and fit P. Candidates execute once and discard their
pending K/V, CIS, and indexer state without consuming host pages.
"""

from dataclasses import replace

import torch

from models.nosa.attention import NosaFixedAttention
from models.nosa.cache.fixed import (
    NosaFixedOffloadCache,
    NosaFixedResidentCache,
)
from models.nosa.execution.adapter import NosaServingBackend
from models.nosa.execution.fixed_resources import FIXED_POLICY_REVISION, NosaFixedResources
from models.nosa.model import NosaForCausalLM


class NosaFixedServingBackend(NosaServingBackend):
    def __init__(
        self,
        model,
        scheme,
        *,
        sparse_pool_tokens,
        host_arena_tokens,
        chunk_size=1024,
    ):
        scheme = {"sync_sparse": "serial_sparse", "async_sparse": "overlap"}.get(scheme, scheme)
        super().__init__(model, scheme, chunk_size=chunk_size)
        cfg = self.config
        self.resources = NosaFixedResources(
            scheme=scheme,
            layers=cfg.num_hidden_layers,
            kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            query_heads=cfg.num_attention_heads,
            max_seq_len=self.max_seq_len,
            chunk_size=chunk_size,
            dtype=self.dtype,
            device=self.device,
            sparse_pool_tokens=sparse_pool_tokens,
            host_arena_tokens=host_arena_tokens,
        )
        self._fixed_attention = NosaFixedAttention()

    def runtime_driver(self, policy):
        from executor.adapters import BackendAdapter

        if policy.mode != "fixed_pools":
            raise ValueError("Fixed NOSA serving requires P/NH capacity quotas")
        return BackendAdapter(
            self,
            shared=True,
            candidate_mode="gpu_transient",
            session_length=lambda session: session.length,
            chunk_size=self.chunk_size,
            diagnostics=lambda session: self.session_metrics(session),
            owner_aware=True,
            session_planner=self.plan_runtime_session,
            session_factory=self.create_planned_session,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint,
        *,
        scheme,
        sparse_pool_tokens,
        host_arena_tokens,
        device="cuda:0",
        chunk_size=1024,
        max_seq_len=None,
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
        result = cls(
            model,
            scheme,
            sparse_pool_tokens=sparse_pool_tokens,
            host_arena_tokens=host_arena_tokens,
            chunk_size=chunk_size,
        )
        result.checkpoint = str(checkpoint)
        return result

    def retained_session_capacity(self, capacity, prefix):
        return prefix

    def estimate_session_host_pages(self, capacity):
        self._validate_capacity(capacity)
        return 0 if self.scheme == "hbm" else (capacity + 63) // 64

    def session_host_pages(self, session):
        self.resources.check_session(session)
        return self.estimate_session_host_pages(session.history_capacity)

    def estimate_session_hbm_tokens(self, capacity):
        self._validate_capacity(capacity)
        return capacity if self.scheme == "hbm" else 0

    def session_hbm_tokens(self, session):
        self.resources.check_session(session)
        return self.estimate_session_hbm_tokens(session.history_capacity)

    def estimate_session_bytes(self, capacity, prefix_tokens):
        if capacity != prefix_tokens or capacity % 64:
            raise ValueError("Fixed NOSA retains only page-aligned history")
        candidate = self.resources.plan.metadata["max_candidate_tokens"]
        # Include session CIS/derived candidate tails and every existing helper
        # allocation. The host term is conservatively bounded at H+A while the
        # actual pinned allocation contains only H tokens.
        result = super().estimate_session_bytes(capacity + candidate, prefix_tokens)
        if self.scheme == "dense_prefetch" and self.device.type == "cuda":
            from cache.allocator.budget import allocation_bytes

            result["hbm"] += allocation_bytes(2 * torch.int64.itemsize, self.device)
        return result

    def create_session(self, capacity, *, owner=None):
        self.resources.lifecycle.check_access(owner)
        if self.resources.plan is None:
            raise RuntimeError("Plan and allocate shared resources before creating a session")
        return self._allocate_session(
            capacity,
            owner=owner,
            execution_capacity=capacity + self.resources.plan.metadata["max_candidate_tokens"],
        )

    def _session_geometry(self, resource_plan, shape):
        from models.nosa.execution.session_budget import NosaSessionGeometry

        capacity = shape.history_tokens
        if capacity % 64:
            raise ValueError("Fixed NOSA retains only page-aligned history")
        execution_capacity = capacity + resource_plan.metadata["max_candidate_tokens"]
        self._validate_capacity(execution_capacity)
        return NosaSessionGeometry(
            provider_identity=self.resources.lifecycle.identity,
            generation=self.resources.generation,
            history_tokens=shape.history_tokens,
            capacity=capacity,
            execution_capacity=execution_capacity,
            host_capacity=capacity,
            queries=max(
                min(self.chunk_size, capacity), resource_plan.metadata["max_candidate_tokens"]
            ),
            scheme=self.scheme,
        )

    def _session_layout_options(self, geometry):
        return {"host_capacity": geometry.host_capacity, "dense_counter": True}

    def _allocate_session(self, capacity, *, owner=None, execution_capacity):
        self.resources.lifecycle.check_access(owner)
        self.resources.check_quota(capacity)
        options = {
            "device": self.device,
            "dtype": self.dtype,
            "with_cis": True,
            "execution_resources": self.resources,
            "execution_capacity": execution_capacity,
        }
        if self.scheme == "hbm":
            session = NosaFixedResidentCache(self.config, capacity, **options)
        else:
            session = NosaFixedOffloadCache(
                self.config, capacity, overlap=self.scheme == "overlap", **options
            )
        self.resources.attach(session, owner=owner)
        return session

    def _execution_attention(self):
        return self._fixed_attention

    @torch.inference_mode()
    def extend_candidate(self, session, ids, *, owner=None):
        self._validate_ids(session, ids, prefill=False)
        if session.length != session.history_capacity or session.transient_candidate:
            raise RuntimeError("Candidate execution requires one complete, idle history")
        with self.execution_lease(session, owner=owner):
            session.transient_candidate = True
            try:
                result = self._forward(session, ids)
                if session.length != session.history_capacity or session._pending_end is not None:
                    raise RuntimeError("Candidate execution changed retained history")
                return result
            finally:
                session.transient_candidate = False

    def session_metrics(self, session):
        return {
            **super().session_metrics(session),
            "candidate_persistence": "gpu_transient",
            "retained_length": session.length,
            "retained_capacity": session.history_capacity,
            "session_host_history_capacity": 0
            if self.scheme == "hbm"
            else session.buffers["keys"].shape[1],
        }

    def describe(self):
        result = super().describe()
        result.update(
            policy_revision=FIXED_POLICY_REVISION,
            sparse_pool_tokens=self.resources.sparse_pool_tokens,
            host_arena_tokens=self.resources.host_arena_tokens,
            host_scope="global history quota; lazy per-session pinned allocation",
            cache_eviction="session LRU admission; direct-mapped historical HBM pages",
            dense_prefetch="next-layer historical misses into that layer's P slots",
            sparse_offload="exact full-batch sparse union; fetch only session-tag misses",
            fixed_scope="page-aligned H <= P; page-aligned history chunks",
            candidate_persistence="GPU-only execution followed by discard",
        )
        return result
