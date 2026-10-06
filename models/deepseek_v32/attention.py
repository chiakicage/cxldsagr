"""Chunked DeepSeek indexer → ECHO prefetch → exact recall → sparse MLA."""

from contextlib import contextmanager, nullcontext

import torch

from cache.sparse_token_cache import MISSING, SparseTokenCache, WorkingSetTooLarge
from models.attention_contracts import TokenSelection
from operators.deepseek_v32.attention.device_only.mla import sparse_mla
from operators.deepseek_v32.attention.offload.mla import sparse_mla_from_pool
from operators.deepseek_v32.indexer.prefetch_hint import update_prefetch_hint
from operators.deepseek_v32.indexer.selection import exact_topk


class EchoAttentionRunner:
    """One layer's attention execution; the model owns multi-layer commit.

    Indexer keys remain on the model GPU. Only the main MLA token records use
    offload. A too-large exact union splits attention consumption, preserving
    every query's original top-k and the full indexer search domain.
    """

    def __init__(
        self,
        attention,
        capacity,
        *,
        offload=False,
        slots=16384,
        chunk_size=1024,
        cache=None,
        fused_prefetch=True,
        collect_cache_diagnostics=False,
    ):
        if chunk_size < 1 or offload and slots < chunk_size:
            raise ValueError("chunk_size must be positive and fit in offload slots")
        self.attention = attention
        self.cfg = attention.cfg
        self.chunk_size = chunk_size
        self.fused_prefetch = fused_prefetch
        self.collect_cache_diagnostics = collect_cache_diagnostics
        self.cache_diagnostics = None
        self._diagnostic_state = None
        self.cache = (
            cache
            if cache is not None
            else SparseTokenCache(
                capacity,
                self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim,
                device=attention.device,
                slots=min(slots, capacity) if offload else None,
            )
        )
        self.index_keys = torch.empty(
            (capacity, self.cfg.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device=attention.device,
        )
        self.index_scales = torch.empty(capacity, dtype=torch.float32, device=attention.device)
        self.offset = torch.zeros(16, device=attention.device, dtype=torch.float32)
        self.last_indices = None
        self.capture_hook = None

    @property
    def mutates_prefetch_hint(self):
        """Whether a successful layer call can change its retained hint."""
        return self.fused_prefetch and self.cache.offload

    @contextmanager
    def defer_unused_prefill_hints(self):
        """An exclusive model step may defer hints unused by its later chunks.

        Standalone layer calls retain immediate updates. The per-call proof
        below additionally requires the complete pending step to fit in P.
        """
        previous = getattr(self, "_defer_prefill_hints", False)
        self._defer_prefill_hints = True
        try:
            yield
        finally:
            self._defer_prefill_hints = previous

    def _needs_prefetch_hint(self, end):
        step_end = getattr(self.cache, "_step_end", None)
        return not (
            getattr(self, "_defer_prefill_hints", False)
            and self.capture_hook is None
            and not self.collect_cache_diagnostics
            and self.cache.transient_start is None
            and step_end is not None
            and end < step_end <= self.cache.slots
            and self.cache.all_history_resident
        )

    def _start_cache_diagnostics(self, position, queries):
        """Intrusive opt-in observation, kept out of formal latency runs.

        The map snapshot and consumed bitmap require 5*NH bytes. They share the
        declared metadata temporary allowance (64*NH plus per-slot scratch),
        and are dropped before this layer call returns, including failures.
        """
        self._diagnostic_state = {
            "initial_map": self.cache.host_to_device.clone(),
            "seen_consumed": torch.zeros_like(self.cache.host_to_device, dtype=torch.bool),
            "before": self.cache.metrics(),
            "counts": {
                "schema": "echo-layer-cache-diagnostics-v1",
                "query_start": position,
                "query_tokens": queries,
                "consumer_groups": 0,
                "group_selection_records": 0,
                "group_recalled_records": 0,
                "cross_group_reread_records": 0,
                "cross_group_reread_definition": (
                    "actual recall of a record selected by an earlier consumption group"
                ),
            },
        }

    def _diagnose_selection(self, indices, position):
        state = self._diagnostic_state
        logical = torch.unique(indices[indices >= 0].long())
        transient_start = getattr(self.cache, "transient_start", None)
        host_mask = (
            logical < transient_start
            if transient_start is not None
            else torch.ones_like(logical, dtype=torch.bool)
        )
        global_ids = self.cache.logical_to_global(logical[host_mask])
        resident = logical < position  # Earlier candidate chunks already occupy GPU tail slots.
        resident[host_mask] = state.pop("initial_map")[global_ids] != MISSING
        history = logical < position
        counts = state["counts"]
        counts.update(
            unique_selection_records=logical.numel(),
            historical_selection_records=int(history.sum()),
            fused_before_resident_selection_records=int(resident.sum()),
            fused_before_resident_historical_records=int((resident & history).sum()),
        )
        counts["fused_before_hbm_token_hit_ratio"] = (
            counts["fused_before_resident_selection_records"] / logical.numel()
            if logical.numel()
            else None
        )
        state["global_selection"] = global_ids

    def _diagnose_after_append(self):
        state = self._diagnostic_state
        global_ids = state.pop("global_selection")
        state["counts"]["remaining_miss_after_append"] = int(
            (self.cache.host_to_device[global_ids] == MISSING).sum()
        )

    def _diagnose_group_before_recall(self, indices):
        logical = torch.unique(indices[indices >= 0].long())
        transient_start = getattr(self.cache, "transient_start", None)
        host_logical = (
            logical[logical < transient_start] if transient_start is not None else logical
        )
        if host_logical.numel() > self.cache.slots:
            return None  # A failed oversized probe did not consume or copy records.
        global_ids = self.cache.logical_to_global(host_logical)
        missing = global_ids[self.cache.host_to_device[global_ids] == MISSING]
        reread = int(self._diagnostic_state["seen_consumed"][missing].sum())
        return global_ids, missing.numel(), reread, logical.numel()

    def _diagnose_group_after_recall(self, group):
        global_ids, misses, reread, selected = group
        state = self._diagnostic_state
        state["seen_consumed"][global_ids] = True
        counts = state["counts"]
        counts["consumer_groups"] += 1
        counts["group_selection_records"] += selected
        counts["group_recalled_records"] += misses
        counts["cross_group_reread_records"] += reread

    def _finish_cache_diagnostics(self):
        state = self._diagnostic_state
        before, after = state["before"], self.cache.metrics()
        counts = state["counts"]
        for name in (
            "prefetched_records",
            "recalled_records",
            "evicted_records",
            "written_records",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "capacity_splits",
        ):
            counts[name] = after[name] - before[name]
        if not self.cache.offload:
            counts["consumer_groups"] = 1
            counts["group_selection_records"] = counts["unique_selection_records"]
        counts["cross_group_reread_bytes"] = (
            counts["cross_group_reread_records"] * self.cache.record_bytes
        )
        self.cache_diagnostics = counts

    def _consume(self, q, selection: TokenSelection, scope):
        indices = selection.token_ids
        if not self.cache.offload:
            with scope("sparse_mla"):
                return sparse_mla(q, self.cache.records, indices, self.cfg.attention_scale)
        diagnostics = getattr(self, "_diagnostic_state", None)
        group = None
        if diagnostics is not None:
            with scope("cache_diagnostic_group"):
                group = self._diagnose_group_before_recall(indices)
        try:
            with scope("offload_exact_recall"):
                # Exact top-k is the trusted producer of bounded logical IDs.
                # The internal path may fuse a proven all-resident mapping;
                # arbitrary external selections still use checked ensure().
                ensure = getattr(self.cache, "_ensure_from_topk", self.cache.ensure)
                physical = ensure(indices)
        except WorkingSetTooLarge:
            if len(q) == 1:
                raise ValueError(
                    "offload slots must fit at least one query's exact selection"
                ) from None
            self.cache.stats.capacity_splits += 1
            middle = len(q) // 2
            left = self._consume(q[:middle], TokenSelection(indices[:middle]), scope)
            right = self._consume(q[middle:], TokenSelection(indices[middle:]), scope)
            return torch.cat((left, right))
        if diagnostics is not None:
            with scope("cache_diagnostic_group"):
                self._diagnose_group_after_recall(group)
        with scope("sparse_mla"):
            return sparse_mla_from_pool(q, self.cache.records, physical, self.cfg.attention_scale)

    def forward(
        self,
        hidden,
        *,
        scope=None,
        capture_indices=False,
        normalized=False,
        project_callback=None,
        output_callback=None,
        indexer_bounds=None,
    ):
        self.cache_diagnostics = None
        self._diagnostic_state = None
        try:
            callbacks = {}
            if project_callback is not None:
                callbacks["project_callback"] = project_callback
            if output_callback is not None:
                callbacks["output_callback"] = output_callback
            if indexer_bounds is not None:
                callbacks["indexer_bounds"] = indexer_bounds
            result = self._forward(
                hidden,
                scope=scope,
                capture_indices=capture_indices,
                normalized=normalized,
                **callbacks,
            )
            if self._diagnostic_state is not None:
                self._finish_cache_diagnostics()
            return result
        finally:
            # No map, bitmap, temporary selection, or failed-call counters may
            # survive into a later request or inflate retained session bytes.
            self._diagnostic_state = None

    def _forward(
        self,
        hidden,
        *,
        scope=None,
        capture_indices=False,
        normalized=False,
        project_callback=None,
        output_callback=None,
        indexer_bounds=None,
    ):
        from operators.deepseek_v32.indexer.echo import logits as index_logits

        scope = scope or (lambda _: nullcontext())
        project = self.attention.project if project_callback is None else project_callback
        if not callable(project) or output_callback is not None and not callable(output_callback):
            raise TypeError("attention compute callbacks must be callable")
        if hidden.ndim != 2 or not len(hidden):
            raise ValueError("attention requires a nonempty token batch")
        # The model owns prefill chunking. A layer receives exactly one indexer
        # query batch; only exact-union capacity can split MLA consumption.
        position = self.cache.written
        end = position + len(hidden)
        operation = getattr(self.cache, "operation", nullcontext)
        with operation():
            reserve_source = getattr(self.cache, "reserve_append_source", None)
            if reserve_source is not None:
                with scope("offload_source_reservation"):
                    reserve_source()
            with scope("attention_projection"):
                p = project(hidden, position, normalized=normalized)
            with scope("index_cache_write"):
                self.index_keys[position:end] = p.index_k
                self.index_scales[position:end] = p.index_scale
                self.cache.declare_indexer_visible(end)
            if getattr(self, "collect_cache_diagnostics", False):
                if not hasattr(self.cache, "logical_to_global"):
                    raise ValueError("cache diagnostics require a logical-to-global token map")
                with scope("cache_diagnostic_initial_state"):
                    self._start_cache_diagnostics(position, len(hidden))
            with scope("offload_prepare"):
                prefetch = (
                    self.cache.prepare_prefetch(position, len(hidden), self.offset)
                    if self.fused_prefetch
                    else None
                )
            with scope("indexer_prefetch" if prefetch is not None else "indexer"):
                indexer_options = {"_bounds": indexer_bounds} if indexer_bounds is not None else {}
                if p.index_q.is_cuda and prefetch is None and end >= self.cfg.index_topk:
                    # The official score allocation already has an aligned
                    # stride. Expose it to top-k without repacking Q x N;
                    # end >= k keeps the logical selection capacity unchanged.
                    indexer_options["_pad_to_stride"] = True
                scores = index_logits(
                    p.index_q,
                    self.index_keys[:end],
                    p.index_weights,
                    self.index_scales[:end],
                    position,
                    prefetch=prefetch,
                    **indexer_options,
                )
            if prefetch is not None:
                with scope("offload_finalize"):
                    self.cache.finalize_prefetch(prefetch)
            with scope("exact_topk"):
                if scores.is_cuda:
                    # Both indexer backends return causal logits, including
                    # initialized -inf tails. Avoid a second full Q x N mask
                    # and score copy before the exact selection API.
                    values, indices = exact_topk(scores, self.cfg.index_topk)
                else:
                    # Small CPU cache/reference tests preserve the PyTorch oracle.
                    ends = torch.arange(position + 1, end + 1, device=hidden.device)
                    valid = torch.arange(end, device=hidden.device)[None, :] < ends[:, None]
                    scores = scores[:, :end].masked_fill(~valid, -torch.inf)
                    values, indices = torch.topk(scores, min(self.cfg.index_topk, end), dim=-1)
                    indices = torch.where(torch.isfinite(values), indices, -1).int()
            if self.fused_prefetch and self.cache.offload and self._needs_prefetch_hint(end):
                # Cache prediction is separate from the model's exact selection.
                # A later visit can use this hint after another user's eviction.
                with scope("prefetch_hint"):
                    # Keep the original reduction shape/order even when
                    # selection consumed an aligned score view.
                    update_prefetch_hint(scores[:, :end], self.offset)
            del scores, values
            if self._diagnostic_state is not None:
                with scope("cache_diagnostic_selection"):
                    self._diagnose_selection(indices, position)
            with scope("cache_write"):
                self.cache.append(p.kv)
            if self._diagnostic_state is not None:
                with scope("cache_diagnostic_after_append"):
                    self._diagnose_after_append()
            if self.capture_hook is not None:
                self.capture_hook(
                    p, self.index_keys[:end], self.index_scales[:end], indices, self.cache, position
                )
            attn = self._consume(p.q, TokenSelection(indices), scope)
            with scope("attention_output"):
                result = (
                    self.attention.output(attn)
                    if output_callback is None
                    else output_callback(attn)
                )
        self.last_indices = [indices] if capture_indices else None
        return result

    def reset(self):
        self.cache.reset()
        self.offset.zero_()
