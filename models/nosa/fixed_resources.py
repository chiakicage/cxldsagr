"""Per-layer direct-mapped NOSA cache and bounded execution storage."""

from types import MappingProxyType

import torch

from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint
from executor.serving_backend import SharedCachePlan
from models.nosa.allocation_budget import allocation_bytes
from models.nosa.serving_resources import NosaExecutionResources, _positive

FIXED_POLICY_REVISION = "nosa_fixed_page_cache_v1"


class NosaFixedResources(NosaExecutionResources):
    def __init__(self, *, sparse_pool_tokens, host_arena_tokens, **kwargs):
        super().__init__(**kwargs)
        for name, value in (
            ("sparse_pool_tokens", sparse_pool_tokens),
            ("host_arena_tokens", host_arena_tokens),
        ):
            _positive(name, value)
            if value % 64:
                raise ValueError(f"{name} must be a multiple of 64")
        if self.chunk_size % 64:
            raise ValueError("Fixed NOSA history chunks must be multiples of 64")
        self.sparse_pool_tokens = sparse_pool_tokens
        self.host_arena_tokens = host_arena_tokens
        self.keys = self.values = self.tags = None
        self._prefetch_bytes = None
        self._prefetch_stream = None
        self._prefetch_ready = []
        self._next_owner = 1

    def next_cache_owner(self):
        value = self._next_owner
        if value >= 2**63:
            raise RuntimeError("NOSA cache identity space exhausted")
        self._next_owner += 1
        return value

    def plan_resources(self, budgets, limits):
        if budgets is not None:
            raise ValueError("Fixed NOSA uses P/NH quotas; omit cache byte budgets")
        # Reuse the established native geometry and allocator validation.
        base = super().plan_resources(CacheFootprint(2**63 - 1, 2**63 - 1), limits)
        metadata = dict(base.metadata)
        capacity = metadata["max_session_capacity"]
        candidate = metadata["max_candidate_tokens"]
        history = limits.get("max_history_tokens", capacity - candidate)
        _positive("max_history_tokens", history)
        if history % 64 or history > self.sparse_pool_tokens or history + candidate > capacity:
            raise ValueError("Fixed NOSA requires page-aligned H <= P and H + A <= C")
        queries = metadata["max_query_tokens"]
        from operators.nosa.attention.workspace import NosaAttentionWorkspace

        if self.scheme in ("serial_sparse", "overlap"):
            from operators.nosa.attention.offload.api import NosaFetchWorkspace

            # The first two allocations are replaced by the per-layer pool.
            allocations = NosaFetchWorkspace.allocation_sizes(
                capacity,
                self.kv_heads,
                self.head_dim,
                dtype=self.dtype,
                query_tile_size=self.query_tile_size,
                max_queries=queries,
                trace_capacity=metadata["trace_capacity"],
            )[2:]
        else:
            allocations = NosaAttentionWorkspace.allocation_sizes(
                queries, self.kv_heads, self.head_dim, dtype=self.dtype
            )
        if self.scheme != "hbm":
            pool_tokens = self.sparse_pool_tokens + candidate
            kv = self.layers * pool_tokens * self.kv_heads * self.head_dim * self.dtype.itemsize
            tags = self.layers * ((capacity + 63) // 64) * self.kv_heads * torch.int64.itemsize
            allocations = (*allocations, kv, kv, tags)
        if self.scheme == "dense_prefetch":
            allocations = (*allocations, torch.int64.itemsize)
        logical = sum(allocations)
        shared = sum(allocation_bytes(size, self.device) for size in allocations)
        metadata.update(
            policy_revision=FIXED_POLICY_REVISION,
            logical_shared_hbm_bytes=logical,
            max_history_tokens=history,
            sparse_pool_tokens=self.sparse_pool_tokens,
            host_arena_tokens=self.host_arena_tokens,
            pool_scope="backend_per_layer_direct_mapped_pages",
            host_scope="global_history_page_quota_with_lazy_session_backing",
            host_allocation="lazy_pinned_per_session_history_only",
            candidate_persistence="gpu_transient",
            candidate_capacity_scope="backend_main_kv_tail; session_cis_and_derived_tail",
            hbm_candidate_capacity_scope="session_resident_tail",
            cache_mapping="logical_page_offset_with_session_tag",
            cache_page_tokens=64,
            stage_count=0,
        )
        return SharedCachePlan(
            shared=CacheFootprint(hbm=shared),
            host_pages=0 if self.scheme == "hbm" else self.host_arena_tokens // 64,
            hbm_tokens=self.sparse_pool_tokens if self.scheme == "hbm" else 0,
            page_size=64,
            metadata=MappingProxyType(metadata),
        )

    def allocate_shared(self, plan):
        if self.poisoned:
            raise RuntimeError("NOSA execution resources are poisoned")
        if (
            not isinstance(plan, SharedCachePlan)
            or self.plan_resources(None, plan.metadata) != plan
        ):
            raise ValueError("Shared plan does not match fixed NOSA resources")
        if self.plan == plan:
            return
        if self._sessions or self._active_session is not None or self.plan is not None:
            raise RuntimeError("Cannot replace allocated fixed NOSA resources")
        metadata = plan.metadata
        capacity, queries = metadata["max_session_capacity"], metadata["max_query_tokens"]
        try:
            if self.scheme in ("serial_sparse", "overlap"):
                from operators.nosa.attention.offload.api import NosaFetchWorkspace

                self.fetch_workspace = NosaFetchWorkspace(
                    capacity,
                    self.kv_heads,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                    query_tile_size=self.query_tile_size,
                    fetch_ctas=self.fetch_ctas,
                    overlap=self.scheme == "overlap",
                    max_queries=queries,
                    bounded=True,
                    trace_capacity=metadata["trace_capacity"],
                )
            else:
                from operators.nosa.attention.workspace import NosaAttentionWorkspace

                self.attention_workspace = NosaAttentionWorkspace(
                    queries, self.kv_heads, self.head_dim, device=self.device, dtype=self.dtype
                )
            if self.scheme != "hbm":
                shape = (
                    self.layers,
                    self.sparse_pool_tokens + metadata["max_candidate_tokens"],
                    self.kv_heads,
                    self.head_dim,
                )
                self.keys = torch.empty(shape, device=self.device, dtype=self.dtype)
                self.values = torch.empty_like(self.keys)
                self.tags = torch.zeros(
                    (self.layers, (capacity + 63) // 64, self.kv_heads),
                    device=self.device,
                    dtype=torch.int64,
                )
                if self.fetch_workspace is not None:
                    self.fetch_workspace.keys = self.keys[0, :capacity]
                    self.fetch_workspace.values = self.values[0, :capacity]
                if self.device.type == "cuda" and self.scheme == "dense_prefetch":
                    self._prefetch_stream = torch.cuda.Stream(device=self.device)
                    self._prefetch_ready = [torch.cuda.Event() for _ in range(self.layers)]
                if self.scheme == "dense_prefetch":
                    self._prefetch_bytes = torch.empty((), device=self.device, dtype=torch.int64)
            self.plan = plan
            actual = self.shared_bytes()
            if actual != {"hbm": metadata["logical_shared_hbm_bytes"], "dram": 0}:
                raise RuntimeError(f"Fixed NOSA storage disagrees with plan: {actual}")
            self._allocator_key = self._validate_allocator()["configuration_key"]
            if self.device.type == "cuda":
                self._allocation_ready = torch.cuda.Event()
                self._allocation_ready.record(torch.cuda.current_stream(self.device))
        except BaseException as error:
            self._allocation_failure = error
            self._drain_and_drop_storage()
            raise
        self.generation += 1

    def storage_tensors(self):
        return (
            *super().storage_tensors(),
            *(
                tensor
                for tensor in (self.keys, self.values, self.tags, self._prefetch_bytes)
                if tensor is not None
            ),
        )

    def _drop_storage(self):
        super()._drop_storage()
        self.keys = self.values = self.tags = None
        self._prefetch_bytes = None
        self._prefetch_stream = None
        self._prefetch_ready = []

    def check_quota(self, history_capacity):
        _positive("history capacity", history_capacity)
        if self.plan is None:
            raise RuntimeError("Plan resources before creating sessions")
        if history_capacity % 64 or history_capacity > self.plan.metadata["max_history_tokens"]:
            raise ValueError("Retained history exceeds the planned page-aligned limit")
        used = sum(session.history_capacity for session in self._sessions)
        quota = self.sparse_pool_tokens if self.scheme == "hbm" else self.host_arena_tokens
        if used + history_capacity > quota:
            raise CacheBudgetExceeded("Fixed NOSA history quota is full; evict a session first")

    def commit_history(self, session, start, end):
        self.check_execution(session)
        if start % 64 or end % 64:
            raise ValueError("Fixed history commits must contain complete pages")
        self.tags[:, start // 64 : end // 64].fill_(session.cache_owner)

    def invalidate_suffix(self, layer, start, end):
        # A shorter session's candidate can occupy historical slots belonging
        # to a longer session. The overwritten bytes cease to be cache hits
        # even though the candidate itself will never publish a page identity.
        self.tags[layer, start // 64 : (end + 63) // 64].zero_()

    def prefetch(self, session, layer):
        self.check_execution(session)
        if self.scheme != "dense_prefetch":
            raise RuntimeError("Only dense prefetch can queue whole historical layers")
        if self.device.type == "cpu":
            self._copy_reference(session, layer, range(session.length // 64))
            return
        from operators.nosa.attention.offload.api import prefetch_cached_history

        current = torch.cuda.current_stream(self.device)
        self._prefetch_stream.wait_stream(current)
        with torch.cuda.stream(self._prefetch_stream):
            copied = prefetch_cached_history(
                self.keys[layer],
                self.values[layer],
                session.buffers["keys"][layer],
                session.buffers["values"][layer],
                self.tags[layer],
                session.cache_owner,
                session.length,
                out=self._prefetch_bytes,
            )
            session._transfer_metrics.add_sparse_fetch(copied)
            self._prefetch_ready[layer].record(self._prefetch_stream)

    def wait_prefetch(self, session, layer):
        self.check_execution(session)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_event(self._prefetch_ready[layer])

    def _copy_reference(self, session, layer, blocks_by_head):
        if isinstance(blocks_by_head, range):
            blocks_by_head = [blocks_by_head] * self.kv_heads
        for head, blocks in enumerate(blocks_by_head):
            for block in blocks:
                if block < 0 or block * 64 >= session.length:
                    continue
                if self.tags[layer, block, head] == session.cache_owner:
                    continue
                for name in ("keys", "values"):
                    getattr(self, name)[layer, block * 64 : (block + 1) * 64, head].copy_(
                        session.buffers[name][layer, block * 64 : (block + 1) * 64, head]
                    )
                self.tags[layer, block, head] = session.cache_owner

    def fetch_reference(self, session, layer, selection):
        self.check_execution(session)
        ids = selection.block_ids.expand(-1, self.kv_heads, -1)
        valid = selection.valid_mask
        if valid is not None:
            valid = valid.expand_as(ids)
        blocks = []
        for head in range(self.kv_heads):
            selected = ids[:, head]
            if valid is not None:
                selected = selected[valid[:, head]]
            blocks.append(selected.unique().tolist())
        self._copy_reference(session, layer, blocks)
        suffix = session.suffix_layer_view(layer)
        end = session.visible_length(layer)
        self.invalidate_suffix(layer, session.length, end)
        for name in ("keys", "values"):
            getattr(self, name)[layer, session.length : end].copy_(suffix[name])
        return {
            "keys": self.keys[layer, :end],
            "values": self.values[layer, :end],
            "cis_scores": session.cis_scores[layer, :end],
        }
