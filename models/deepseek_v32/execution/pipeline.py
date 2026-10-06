"""Model-specific attention and cache factories for serving assembly."""

from models.attention_contracts import TokenSelection
from models.deepseek_v32.attention import EchoAttentionRunner
from models.deepseek_v32.cache.session import DenseCache, ServingSparseTokenCache


class ServingAttention(EchoAttentionRunner):
    def __init__(
        self, attention, capacity, *, scheme, slots, chunk_size, dense_backend=None, cache=None
    ):
        self.scheme = scheme
        if scheme == "dense_prefetch" and cache is None:
            cache = DenseCache(
                capacity,
                attention.cfg.kv_lora_rank + attention.cfg.qk_rope_head_dim,
                device=attention.device,
                backend=dense_backend,
            )
        if scheme == "hbm" and cache is None:
            cache = ServingSparseTokenCache(
                capacity,
                attention.cfg.kv_lora_rank + attention.cfg.qk_rope_head_dim,
                device=attention.device,
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

    def _consume(self, q, selection: TokenSelection, scope):
        if isinstance(self.cache, DenseCache):
            from operators.deepseek_v32.attention.device_only.mla import sparse_mla

            with scope("sparse_mla"):
                return sparse_mla(
                    q, self.cache.records, selection.token_ids, self.cfg.attention_scale
                )
        return super()._consume(q, selection, scope)


class LocalPipeline:
    """Local fused ECHO, serial sparse, resident, and dense cache pipelines."""

    def validate_scheme(self, scheme):
        if scheme not in ("hbm", "echo", "serial_sparse", "dense_prefetch"):
            raise ValueError("unsupported DeepSeek serving pipeline")

    def layer_cache(self, session, layer):
        return ServingSparseTokenCache.for_layer(session, layer)

    def create_runner(self, attention, capacity, **kwargs):
        return ServingAttention(attention, capacity, **kwargs)

    def session_metrics(self, session):
        metrics = [runner.cache.metrics() for runner in session.runners]
        selected = sum(item.get("selection_records", 0) for item in metrics)
        resident = sum(item.get("resident_selection_records", 0) for item in metrics)
        result = {
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

        return result
