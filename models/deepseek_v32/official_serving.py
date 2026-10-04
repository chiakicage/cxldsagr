"""Pinned ECHO offload implementation under the fixed-pool replay workload.

The checkpoint projections, independent block weights and FlashMLA consumer
are the same as the motivation experiment. Official indexer, fused top-k,
allocator, prefetch and residual recall execute directly from the pinned
artifact. This adapter is not the upstream TP=8/AWQ SGLang deployment.
"""

from contextlib import nullcontext
from dataclasses import replace

import torch

from cache.prefix_pool import CacheFootprint
from models.deepseek_v32.echo_attention import EchoAttentionRunner
from models.deepseek_v32.serving_backend import DeepSeekServingBackend


class OfficialAttentionRunner(EchoAttentionRunner):
    @classmethod
    def from_existing(cls, runner, cache):
        """Reuse the already allocated model and private indexer tensors."""
        result = cls.__new__(cls)
        result.__dict__.update(vars(runner))
        result.cache = cache
        return result

    def _forward(
        self,
        hidden,
        *,
        scope=None,
        capture_indices=False,
        normalized=False,
        project_callback=None,
        output_callback=None,
    ):
        from operators.deepseek_v32.indexer.official import module, topk_module

        project = self.attention.project if project_callback is None else project_callback
        if not callable(project) or output_callback is not None and not callable(output_callback):
            raise TypeError("attention compute callbacks must be callable")
        if self.collect_cache_diagnostics:
            raise ValueError("local ECHO diagnostic hooks do not describe the official pipeline")
        if hidden.ndim != 2 or not len(hidden):
            raise ValueError("attention requires a nonempty token batch")
        scope = scope or (lambda _: nullcontext())
        position = self.cache.written
        end = position + len(hidden)
        with self.cache.operation():
            with scope("offload_source_reservation"):
                self.cache.reserve_append_source()
            with scope("attention_projection"):
                projection = project(hidden, position, normalized=normalized)
            with scope("index_cache_write"):
                self.index_keys[position:end] = projection.index_k
                self.index_scales[position:end] = projection.index_scale
                self.cache.declare_indexer_visible(end)
            with scope("offload_prepare"):
                prefetch = (
                    self.cache.prepare_prefetch(position, len(hidden), self.offset)
                    if self.cache.offload
                    else None
                )
                starts = torch.zeros(len(hidden), dtype=torch.int32, device=hidden.device)
                ends = torch.arange(position + 1, end + 1, dtype=torch.int32, device=hidden.device)
            with scope("indexer_prefetch" if prefetch is not None else "indexer"):
                arguments = (
                    projection.index_q,
                    (self.index_keys[:end], self.index_scales[:end]),
                    projection.index_weights,
                    starts,
                    ends,
                )
                scores = (
                    module().fp8_mqa_logits_fuse_prefetch(
                        *arguments, **prefetch["arguments"], clean_logits=False
                    )
                    if prefetch is not None
                    else module().fp8_mqa_logits(*arguments, clean_logits=False)
                )
            if prefetch is not None:
                with scope("offload_finalize"):
                    # Preserve the artifact hint, including unfilled causal
                    # logits. Official top-k consumes valid lengths directly.
                    self.cache.finalize_prefetch(prefetch, logits=scores)
            with scope("exact_topk"):
                if self.cfg.index_topk != 2048:
                    raise ValueError("official DeepSeek top-k requires k=2048")
                indices = torch.empty((len(hidden), 2048), dtype=torch.int32, device=hidden.device)
                # Keep logical IDs at the common model/cache boundary. The
                # original fused transform still executes its page-table load.
                logical_table = torch.arange(end, dtype=torch.int32, device=hidden.device)[None, :]
                cumulative = torch.tensor([0, len(hidden)], dtype=torch.int32, device=hidden.device)
                topk_module().fast_topk_transform(
                    scores, ends, indices, logical_table, cumulative, None
                )
            del scores, logical_table, cumulative
            with scope("cache_write"):
                self.cache.append(projection.kv)
            if self.capture_hook is not None:
                self.capture_hook(
                    projection,
                    self.index_keys[:end],
                    self.index_scales[:end],
                    indices,
                    self.cache,
                    position,
                )
            result = self._consume(projection.q, indices, scope)
            with scope("attention_output"):
                output = (
                    self.attention.output(result)
                    if output_callback is None
                    else output_callback(result)
                )
        self.last_indices = [indices] if capture_indices else None
        return output


class OfficialDeepSeekServingBackend(DeepSeekServingBackend):
    """The motivation model with the pinned upstream ECHO offload pipeline."""

    def plan_resources(self, budgets, limits):
        plan = super().plan_resources(budgets, limits)
        if self.scheme != "echo":
            return plan
        if budgets is not None:
            raise ValueError("official ECHO experiment requires explicit fixed P/NH capacities")
        from models.deepseek_v32.official_cache import OfficialCacheState

        extra = OfficialCacheState.estimate_extra_bytes(
            plan.metadata["host_arena_tokens"], self.slots, self.num_layers
        )
        return replace(
            plan,
            shared=CacheFootprint(plan.shared.hbm + extra["hbm"], plan.shared.dram + extra["dram"]),
            metadata={
                **plan.metadata,
                "official_extra_shared_bytes": extra,
                "offload_implementation": "pinned_official_echo",
            },
        )

    def allocate_shared(self, plan):
        if self._resource_plan == plan:
            return
        super().allocate_shared(plan)
        if self.scheme == "echo":
            from models.deepseek_v32.official_cache import OfficialCacheState

            try:
                self._official_state = OfficialCacheState(self._shared_pool)
                if (
                    self._official_state.shared_bytes()
                    != plan.metadata["official_extra_shared_bytes"]
                ):
                    raise RuntimeError("official cache allocation differs from its reservation")
            except BaseException:
                self._release_shared()
                raise

    def shared_bytes(self):
        result = super().shared_bytes()
        state = getattr(self, "_official_state", None)
        if state is not None:
            extra = state.shared_bytes()
            result = {key: result[key] + extra[key] for key in result}
        return result

    def create_session(self, capacity):
        session = super().create_session(capacity)
        if self.scheme in ("hbm", "echo"):
            try:
                session.runners = [
                    OfficialAttentionRunner.from_existing(
                        runner,
                        self._official_state.layer_cache(runner.cache)
                        if self.scheme == "echo"
                        else runner.cache,
                    )
                    for runner in session.runners
                ]
            except BaseException:
                self.release_session(session)
                raise
        return session

    def _release_shared(self):
        state = getattr(self, "_official_state", None)
        super()._release_shared()
        if state is not None:
            state.close()
        self._official_state = None

    def official_provenance(self):
        from operators.deepseek_v32.indexer.official import module, provenance, topk_module

        module()
        topk_module()
        return provenance()

    def session_metrics(self, session):
        if self.scheme != "echo":
            return super().session_metrics(session)
        self._check_session(session)
        layers = [runner.cache.metrics() for runner in session.runners]
        measured = (
            "host_to_device_bytes",
            "device_to_host_bytes",
            "prefetched_records",
            "recalled_records",
            "capacity_splits",
            "evicted_records",
        )
        return {
            **{key: sum(layer.get(key, 0) for layer in layers) for key in measured},
            "candidate_persistence": "gpu_transient"
            if session.last_candidate_transient
            else "committed",
            "retained_length": session.length,
            "candidate_device_to_host_bytes": sum(layer["device_to_host_bytes"] for layer in layers)
            if session.last_candidate_transient
            else None,
            "selection_records": None,
            "resident_selection_records": None,
            "hbm_token_hit_ratio_before_recall": None,
            "hit_ratio_stage": "not_instrumented_in_official_implementation",
            "counter_boundary": "official_allocated_prefetch_and_residual_recall_records",
            "layer_diagnostics": [None] * len(layers),
            "layers": layers,
        }

    def describe(self):
        result = super().describe()
        result["indexer_backend"] = (
            "official_echo_fused_logits"
            if self.scheme == "echo"
            else "official_echo_resident_logits"
        )
        result["selection_backend"] = "official_echo_fast_topk_transform_fused_logical_ids"
        if self.scheme == "echo":
            result["cache_policy_revision"] = "official-echo-bc1b75c-fixed-history-adapter-v1"
            result["offload_implementation"] = "pinned_official_echo"
            result["echo_flags"] = {
                "fused_logits_recall_extend": True,
                "early_evict": False,
                "offset_policy": "official_mean_last_four_rows_before_causal_mask",
                "exact_union_overflow": "rejected_by_fixed_history_fits_P_contract",
            }
            result["official_adaptation"] = {
                "model": "same motivation checkpoint replay; no indexer Hadamard",
                "candidate": "shared GPU tail, discarded; no host IDs or writes",
                "indexer_storage": "private history plus one shared combined workspace",
                "consumer": "same FlashMLA as motivation; official top-k with logical page table",
                "deployment": "single GPU serial GR runner, not upstream SGLang server",
            }
            state = getattr(self, "_official_state", None)
            if state is not None:
                result["official_cache_sources"] = state.provenance()
        return result
