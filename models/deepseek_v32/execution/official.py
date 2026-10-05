"""Pinned ECHO offload implementation under the fixed-pool replay workload.

The checkpoint projections, independent block weights and FlashMLA consumer
are the same as the motivation experiment. Official indexer, fused top-k,
allocator, prefetch and residual recall execute directly from the pinned
artifact. This adapter is not the upstream TP=8/AWQ SGLang deployment.
"""

from contextlib import nullcontext
from dataclasses import replace

import torch

from cache.capacity import CacheFootprint, allocation_footprint
from models.attention_contracts import TokenSelection
from models.deepseek_v32.attention import EchoAttentionRunner
from models.deepseek_v32.cache.session import ServingSparseTokenCache
from models.deepseek_v32.execution.planning import _allocation


class OfficialAttentionRunner(EchoAttentionRunner):
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
            result = self._consume(projection.q, TokenSelection(indices), scope)
            with scope("attention_output"):
                output = (
                    self.attention.output(result)
                    if output_callback is None
                    else output_callback(result)
                )
        self.last_indices = [indices] if capture_indices else None
        return output


class OfficialPipeline:
    """Pinned official attention, cache adapter, and extra-resource factories."""

    def validate_scheme(self, scheme):
        if scheme not in ("hbm", "echo"):
            raise ValueError("official ECHO supports only hbm and echo pipelines")

    def plan_resources(self, config, plan, budgets):
        if config.scheme != "echo":
            return plan
        if budgets is not None:
            raise ValueError("official ECHO experiment requires explicit fixed P/NH capacities")
        from models.deepseek_v32.cache.official import OfficialCacheState

        extra = OfficialCacheState.estimate_extra_bytes(
            plan.metadata["host_arena_tokens"], config.slots, config.num_layers
        )
        extra_allocations = []
        for name, dtype, shape in (
            ("priorities", "int32", (config.num_layers, config.slots + 1)),
            ("fifo_counter", "int32", (config.num_layers,)),
            ("extend_logits_offsets", "float32", (config.num_layers, 16)),
            ("small_priority_locations", "int32", (config.slots + 1,)),
            ("free_locations", "int32", (config.slots,)),
            ("allocated_locations", "int32", (config.slots,)),
            ("recall_counter", "uint32", (1,)),
            ("host_recall_flags", "bool", (plan.metadata["host_arena_tokens"] + 1,)),
        ):
            extra_allocations.append(_allocation("official." + name, dtype, shape, config.device))
        for layer in range(config.num_layers):
            for name, shape in (
                ("size", (1,)),
                ("free_pages", (config.slots + 1,)),
                ("free_stack", (config.slots,)),
                ("available_size", (1,)),
                ("last_alloc_size", (1,)),
                ("alloc_counter", (1,)),
            ):
                extra_allocations.append(
                    _allocation(
                        f"official.layer_{layer}.allocator.{name}",
                        "int32",
                        shape,
                        config.device,
                    )
                )
        extra_allocations = tuple(extra_allocations)
        if allocation_footprint(extra_allocations) != CacheFootprint.from_mapping(extra):
            raise RuntimeError("official named allocations differ from the reservation")
        return replace(
            plan,
            allocations=plan.allocations + extra_allocations,
            shared=CacheFootprint(plan.shared.hbm + extra["hbm"], plan.shared.dram + extra["dram"]),
            metadata={
                **plan.metadata,
                "official_extra_shared_bytes": extra,
                "offload_implementation": "pinned_official_echo",
            },
        )

    def allocate_resources(self, pool, plan, scheme):
        if scheme != "echo":
            return None
        from models.deepseek_v32.cache.official import OfficialCacheState

        return OfficialCacheState(pool)

    def audit_resources(self, resources, plan):
        if resources is not None and (
            resources.shared_bytes() != plan.metadata["official_extra_shared_bytes"]
        ):
            raise RuntimeError("official cache allocation differs from its reservation")

    def layer_cache(self, session, layer, resources):
        if resources is None:
            raise RuntimeError("official ECHO resources must be allocated before sessions")
        return resources.layer_cache(session, layer)

    def create_runner(
        self, attention, capacity, *, resources, scheme, slots, chunk_size, cache, dense_backend
    ):
        if scheme == "echo" and (resources is None or cache is None):
            raise RuntimeError("official ECHO resources must be allocated before sessions")
        if scheme == "hbm" and cache is None:
            cache = ServingSparseTokenCache(
                capacity,
                attention.cfg.kv_lora_rank + attention.cfg.qk_rope_head_dim,
                device=attention.device,
            )
        return OfficialAttentionRunner(
            attention,
            capacity,
            offload=scheme == "echo",
            slots=slots,
            chunk_size=chunk_size,
            cache=cache,
            fused_prefetch=scheme == "echo",
        )

    def provenance(self):
        from operators.deepseek_v32.indexer.official import module, provenance, topk_module

        module()
        topk_module()
        return provenance()

    def session_metrics(self, session):
        if session.scheme != "echo":
            from models.deepseek_v32.execution.pipeline import LocalPipeline

            return LocalPipeline().session_metrics(session)
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

    def describe(self, metadata, resources):
        result = dict(metadata)
        result["indexer_backend"] = (
            "official_echo_fused_logits"
            if result["scheme"] == "echo"
            else "official_echo_resident_logits"
        )
        result["selection_backend"] = "official_echo_fast_topk_transform_fused_logical_ids"
        if result["scheme"] == "echo":
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
            if resources is not None:
                result["official_cache_sources"] = resources.provenance()
        return result


def build_official_backend(model_path, **options):
    """Assemble the common serving backend with the pinned official factories."""
    from models.deepseek_v32.execution.adapter import DeepSeekServingBackend

    return DeepSeekServingBackend(model_path, pipeline=OfficialPipeline(), **options)
