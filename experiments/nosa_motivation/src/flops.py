"""Useful forward matrix work for full NOSA, with observed selection validation.

No LM head runs in NOSA serving. Every multiply-add counts as two FLOPs.
The indexer owns one logical causal compressed-key QK; recomputation, padding,
pooling, sorting and numerical repair do not inflate the useful work numerator.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass


def model_dimensions(config):
    return asdict(config) if is_dataclass(config) else dict(config)


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def work_counts(query_start, query_tokens, chunk_size):
    """Exact causal pairs, per query head and layer, for the complete policy.

    Each query retains its current block and up to 63 complete historical
    blocks. The indexer bypasses QK when an entire invocation fits in 64 blocks.
    Candidate execution must pass its full batch as chunk_size.
    """
    if type(query_start) is not int or query_start < 0:
        raise ValueError("query_start must be a nonnegative integer")
    _positive(query_tokens, "query_tokens")
    _positive(chunk_size, "chunk_size")
    end = query_start + query_tokens
    attention = sum(
        64 * min(position // 64, 63) + position % 64 + 1 for position in range(query_start, end)
    )
    compressed = scored = 0
    for start in range(query_start, end, chunk_size):
        stop = min(start + chunk_size, end)
        if (stop + 63) // 64 <= 64:
            continue
        scored += stop - start
        compressed += sum(max(0, (position - 31) // 16 + 1) for position in range(start, stop))
    return {
        "attention_pairs_per_query_head_layer": attention,
        "compressed_pairs_per_query_head_layer": compressed,
        "indexer_scored_queries": scored,
        "indexer_bypassed_queries": query_tokens - scored,
    }


def matrix_flops(config, query_start, query_tokens, chunk_size):
    """All checkpoint linears, CIS delta, indexer QK, and sparse QK/PV."""
    cfg = model_dimensions(config)
    layers, hidden, intermediate, heads, kv_heads = (
        _positive(cfg[name], name)
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
        )
    )
    if heads % kv_heads:
        raise ValueError("Q heads must be a multiple of KV heads")
    dim = _positive(cfg.get("head_dim", hidden // heads), "head_dim")
    q_width, kv_width = heads * dim, kv_heads * dim
    projection = 2 * query_tokens * layers * hidden
    pairs = work_counts(query_start, query_tokens, chunk_size)
    return {
        "qkv_proj": projection * (q_width + 2 * kv_width),
        "o_proj": projection * q_width,
        "gate_up_proj": projection * 2 * intermediate,
        "down_proj": projection * intermediate,
        "cis_projection": 2 * layers * query_tokens * kv_width * kv_heads,
        "indexer_qk": 2 * layers * heads * dim * pairs["compressed_pairs_per_query_head_layer"],
        "block_sparse_attention": (
            4 * layers * heads * dim * pairs["attention_pairs_per_query_head_layer"]
        ),
    }


def mfu_percent(flops, duration_ms, peak_tflops=989.5):
    if duration_ms <= 0 or peak_tflops <= 0:
        raise ValueError("duration and reference peak must be positive")
    return 100 * sum(flops.values()) / (duration_ms * peak_tflops * 1e9)


def observe_selection(selection, *, query_start, query_heads):
    """Validate actual consumed IDs and count causal pairs outside timed work.

    Copies to CPU and sorting are deliberately intrusive. Use this on a separate
    replay. Counts summed across KV heads are expanded by the actual GQA ratio.
    """
    import torch

    ids = selection.block_ids.detach().cpu().to(torch.int64)
    if ids.ndim != 3 or selection.block_size != 64 or ids.shape[-1] != 64:
        raise ValueError("Expected full NOSA [query, KV head, 64] selection")
    valid = (ids >= 0) if selection.valid_mask is None else selection.valid_mask.detach().cpu()
    if valid.shape != ids.shape or valid.dtype != torch.bool:
        raise ValueError("Selection validity must be boolean and match IDs")
    queries, kv_heads, _ = ids.shape
    if query_heads % kv_heads:
        raise ValueError("Observed query heads must divide into KV groups")
    positions = torch.arange(query_start, query_start + queries)[:, None, None]
    blocks = positions // 64
    if ((ids < 0) & valid).any() or ((ids > blocks) & valid).any():
        raise ValueError("Selection contains negative or future valid blocks")
    expected_count = (blocks + 1).clamp_max(64).expand(queries, kv_heads, 1).squeeze(-1)
    if not torch.equal(valid.sum(-1), expected_count):
        raise ValueError("Selection does not retain the full policy block count")
    if not ((ids == blocks) & valid).any(-1).all():
        raise ValueError("Selection omitted mandatory current block")
    ordered = ids.masked_fill(~valid, -1).sort(-1).values
    if ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any():
        raise ValueError("Selection contains duplicate valid blocks")
    pairs = ((positions - ids * 64 + 1).clamp(0, 64) * valid).sum().item()
    expected = work_counts(query_start, queries, queries)["attention_pairs_per_query_head_layer"]
    if pairs != expected * kv_heads:
        raise ValueError("Observed causal pairs disagree with the full policy")
    return {
        "query_start": query_start,
        "query_tokens": queries,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "valid_blocks": int(valid.sum().item()),
        "causal_pairs_all_query_heads": pairs * (query_heads // kv_heads),
        "selection_verified": True,
    }
