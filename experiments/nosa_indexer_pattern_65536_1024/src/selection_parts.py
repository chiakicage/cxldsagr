"""Separate fixed and query-aware block unions and estimate their fetch time.

Classification happens per query before taking unions. A block that belongs to
an early query's local window may be selected by a later query's query-aware
branch, so the resulting fixed and query-aware unions can overlap.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

from experiments.nosa_indexer_pattern_65536_1024.src.analyze import _integer, _stats, summarize


def _selection_union(block_ids, mask, shape):
    union = np.zeros(shape, dtype=np.bool_)
    for layer in range(shape[0]):
        for head in range(shape[1]):
            selected = block_ids[layer, :, head, :][mask[layer, :, head, :]]
            union[layer, head, np.unique(selected)] = True
    return union


def decompose_selections(
    block_ids,
    valid_mask,
    *,
    prefix_tokens,
    total_tokens,
    head_dim,
    element_size,
    block_size=64,
    sink_blocks=1,
    local_blocks=16,
    bandwidth_gbps=50.0,
):
    """Return ``(component_unions, report)`` for saved block selections.

    Inputs are ``[layer, query, kv_head, selection]``. Queries form the suffix
    ``[prefix_tokens, total_tokens)``. The existing ``summarize`` validation
    applies, including block-aligned lengths, causal IDs and per-query/head
    uniqueness. Every query/head must contain its complete causal sink/local
    set. Invalid padding is ignored, and input arrays are never modified.

    Each output union is boolean ``[layer, kv_head, logical_block]``. Components
    are sink, local, fixed, query_aware, overlap, query_aware_additional and
    combined. ``fixed = sink | local``; ``combined = fixed | query_aware``.
    Only ``fixed + query_aware_additional`` is an additive byte partition.

    Report rows contain component-name-to-statistics mappings. Fetch estimates
    use decimal GB/s: ``bytes / (bandwidth_gbps * 1e9) * 1000`` milliseconds.
    They assume each component's union is fetched once, with no startup cost,
    cache hits or overlap with computation; they are not measured traffic.
    """
    combined, base_report = summarize(
        block_ids,
        valid_mask,
        prefix_tokens=prefix_tokens,
        total_tokens=total_tokens,
        head_dim=head_dim,
        element_size=element_size,
        block_size=block_size,
    )
    sink_blocks = _integer("sink_blocks", sink_blocks, minimum=0)
    local_blocks = _integer("local_blocks", local_blocks, minimum=0)
    if (
        isinstance(bandwidth_gbps, bool)
        or not isinstance(bandwidth_gbps, Real)
        or not np.isfinite(bandwidth_gbps)
        or bandwidth_gbps <= 0
    ):
        raise ValueError("bandwidth_gbps must be a positive finite number")
    bandwidth_gbps = float(bandwidth_gbps)
    bandwidth_bytes_per_second = bandwidth_gbps * 1e9
    if not np.isfinite(bandwidth_bytes_per_second):
        raise ValueError("bandwidth_gbps must give a finite bandwidth in bytes per second")

    block_ids = np.asarray(block_ids)
    valid_mask = np.asarray(valid_mask)
    params = base_report["parameters"]
    prefix_blocks = params["prefix_tokens"] // params["block_size"]
    positions = params["prefix_tokens"] + np.arange(params["num_queries"])
    query_blocks = positions // params["block_size"]
    sink_mask = valid_mask & (block_ids < sink_blocks)
    local_mask = (
        valid_mask
        & (block_ids >= query_blocks[None, :, None, None] - local_blocks + 1)
        & (block_ids <= query_blocks[None, :, None, None])
    )
    fixed_mask = sink_mask | local_mask

    # The selections are already causal and duplicate-free. Their fixed subset
    # is therefore complete exactly when its size equals the required interval
    # union. Clip to the causal prefix and subtract sink/local overlap.
    visible_blocks = query_blocks + 1
    sink_counts = np.minimum(sink_blocks, visible_blocks)
    local_counts = np.minimum(local_blocks, visible_blocks)
    local_starts = np.maximum(0, visible_blocks - local_blocks)
    shared_counts = np.maximum(0, sink_counts - local_starts)
    required_counts = sink_counts + local_counts - shared_counts
    if np.any(fixed_mask.sum(axis=-1) != required_counts[None, :, None]):
        raise ValueError("Each query/head must include every causal sink/local block")

    sink = _selection_union(block_ids, sink_mask, combined.shape)
    local = _selection_union(block_ids, local_mask, combined.shape)
    fixed = sink | local
    query_aware = _selection_union(block_ids, valid_mask & ~fixed_mask, combined.shape)
    unions = {
        "sink": sink,
        "local": local,
        "fixed": fixed,
        "query_aware": query_aware,
        "overlap": query_aware & fixed,
        "query_aware_additional": query_aware & ~fixed,
        "combined": combined,
    }

    block_bytes = params["block_bytes"]
    head_prefix_capacity = prefix_blocks * block_bytes
    head_capacity = params["blocks_per_head"] * block_bytes
    counts = {
        name: (
            union[..., :prefix_blocks].sum(axis=-1),
            union[..., prefix_blocks:].sum(axis=-1),
        )
        for name, union in unions.items()
    }

    def component_stats(prefix_count, candidate_count, instances):
        stats = _stats(
            prefix_count,
            candidate_count,
            block_bytes=block_bytes,
            prefix_capacity=head_prefix_capacity * instances,
            total_capacity=head_capacity * instances,
        )
        for field, byte_field in (
            ("fetch_ms", "union_bytes"),
            ("prefix_fetch_ms", "prefix_union_bytes"),
            ("candidate_fetch_ms", "candidate_union_bytes"),
        ):
            stats[field] = stats[byte_field] / bandwidth_bytes_per_second * 1000
        return stats

    layers, heads, _ = combined.shape
    head_stats = [
        {
            "layer": layer,
            "kv_head": head,
            **{
                name: component_stats(prefix[layer, head], candidate[layer, head], 1)
                for name, (prefix, candidate) in counts.items()
            },
        }
        for layer in range(layers)
        for head in range(heads)
    ]
    layer_stats = [
        {
            "layer": layer,
            **{
                name: component_stats(prefix[layer].sum(), candidate[layer].sum(), heads)
                for name, (prefix, candidate) in counts.items()
            },
        }
        for layer in range(layers)
    ]
    report = {
        "schema_version": 1,
        "metric": "fixed and query-aware unique full-block K+V payload across candidate queries",
        "deduplication_axes": base_report["deduplication_axes"],
        "byte_accounting": base_report["byte_accounting"],
        "fetch_estimate": "union bytes / decimal bandwidth, without latency or compute overlap",
        "parameters": {
            **params,
            "sink_blocks": sink_blocks,
            "local_blocks": local_blocks,
            "bandwidth_gbps": bandwidth_gbps,
            "bandwidth_bytes_per_second": bandwidth_bytes_per_second,
        },
        "head_stats": head_stats,
        "layer_stats": layer_stats,
        "summary": {
            name: component_stats(prefix.sum(), candidate.sum(), layers * heads)
            for name, (prefix, candidate) in counts.items()
        },
    }
    return unions, report
