"""Independent shape-weighted GEMM/BMM API reference, separate from serving.

Uses all checkpoint layers' actual projection weights. The indexer reference is
a rectangular BF16 BMM with physically expanded GQA keys and materialized QK
scores. It excludes normalization/selection and is not the native fused indexer.
"""

from __future__ import annotations

import statistics
from collections import Counter

from experiments.nosa_motivation.src.flops import matrix_flops


def invocation_geometry(history_tokens, candidate_tokens, chunk_size):
    if any(
        type(value) is not int or value <= 0
        for value in (history_tokens, candidate_tokens, chunk_size)
    ):
        raise ValueError("history, candidate and chunk lengths must be positive integers")
    return [
        (start, min(chunk_size, history_tokens - start))
        for start in range(0, history_tokens, chunk_size)
    ] + [(history_tokens, candidate_tokens)]


def _graph_samples(function, repeats):
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        function()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def benchmark_matrix_apis(
    model,
    history_tokens,
    candidate_tokens,
    chunk_size,
    *,
    linear_repeats=7,
    bmm_repeats=5,
    peak_tflops=989.5,
):
    """Benchmark one complete layer set per shape; return weighted API medians.

    All operations execute. CUDA Graph merely removes per-call Python launch
    gaps. Every linear reads its own checkpoint weights and its own input
    allocation, so a single repeatedly cache-hot weight is not the reference.
    Cache execution must finish before calling this: these temporary graphs own
    private allocator pools outside the serving resource/admission contract.
    """
    import torch
    import torch.nn.functional as F

    if any(type(value) is not int or value <= 0 for value in (linear_repeats, bmm_repeats)):
        raise ValueError("API repeat counts must be positive integers")
    if peak_tflops <= 0:
        raise ValueError("reference dense peak must be positive")
    geometry = invocation_geometry(history_tokens, candidate_tokens, chunk_size)
    multiplicities = Counter(queries for _, queries in geometry)
    layers, cfg = list(model.model.layers), model.config
    if len(layers) != cfg.num_hidden_layers:
        raise ValueError("API reference requires the complete configured layer set")
    device, dtype = model.model.embed_tokens.weight.device, model.model.embed_tokens.weight.dtype
    if device.type != "cuda" or dtype != torch.bfloat16:
        raise ValueError("API reference requires CUDA BF16 checkpoint weights")
    rows = []
    getters = {
        "qkv_proj": lambda layer: layer.self_attn.qkv_proj,
        "o_proj": lambda layer: layer.self_attn.o_proj,
        "gate_up_proj": lambda layer: layer.mlp.gate_up_proj,
        "down_proj": lambda layer: layer.mlp.down_proj,
        "cis_projection": lambda layer: layer.self_attn.delta,
    }
    with torch.inference_mode(), torch.cuda.device(device):
        for queries, count in sorted(multiplicities.items(), reverse=True):
            for name, getter in getters.items():
                weights = [getter(layer).weight for layer in layers]
                biases = [getter(layer).bias for layer in layers]
                if name == "cis_projection":
                    q_width = cfg.num_attention_heads * cfg.head_dim
                    kv_width = cfg.num_key_value_heads * cfg.head_dim
                    inputs = [
                        torch.randn(queries, q_width + 2 * kv_width, device=device, dtype=dtype)[
                            :, q_width + kv_width :
                        ]
                        for _ in layers
                    ]
                else:
                    inputs = [
                        torch.randn(queries, weight.shape[1], device=device, dtype=dtype)
                        for weight in weights
                    ]

                def operation(values=inputs, matrices=weights, offsets=biases):
                    for value, weight, bias in zip(values, matrices, offsets, strict=True):
                        F.linear(value, weight, bias)

                samples = _graph_samples(operation, linear_repeats)
                flops = sum(2 * queries * weight.numel() for weight in weights)
                rows.append(
                    {
                        "operation": name,
                        "queries": queries,
                        "layers": len(layers),
                        "input_stride": list(inputs[0].stride()),
                        "weight_shape": list(weights[0].shape),
                        "useful_flops": flops,
                        "dense_executed_flops": flops,
                        "samples_ms": samples,
                        "median_ms": statistics.median(samples),
                        "request_repetitions": count,
                    }
                )
                del inputs, operation
        score_geometry = [
            (start, queries) for start, queries in geometry if (start + queries + 63) // 64 > 64
        ]
        if score_geometry:
            max_queries = max(queries for _, queries in score_geometry)
            max_keys = max((start + queries) // 16 - 1 for start, queries in score_geometry)
            query = [
                torch.randn(
                    cfg.num_attention_heads, max_queries, cfg.head_dim, device=device, dtype=dtype
                )
                for _ in layers
            ]
            keys = [
                torch.randn(
                    cfg.num_key_value_heads, cfg.head_dim, max_keys, device=device, dtype=dtype
                ).repeat_interleave(cfg.num_attention_heads // cfg.num_key_value_heads, dim=0)
                for _ in layers
            ]
            for start, queries in score_geometry:
                count = (start + queries) // 16 - 1
                query_views = [value[:, :queries] for value in query]
                key_views = [value[:, :, :count] for value in keys]

                def operation(values=query_views, matrices=key_views):
                    for value, key in zip(values, matrices, strict=True):
                        torch.bmm(value, key)

                samples = _graph_samples(operation, bmm_repeats)
                rows.append(
                    {
                        "operation": "indexer_qk_bmm",
                        "query_start": start,
                        "queries": queries,
                        "compressed_keys": count,
                        "layers": len(layers),
                        "input_stride": list(query_views[0].stride()),
                        "key_stride": list(key_views[0].stride()),
                        "useful_flops": matrix_flops(cfg, start, queries, queries)["indexer_qk"],
                        "dense_executed_flops": 2
                        * len(layers)
                        * cfg.num_attention_heads
                        * queries
                        * cfg.head_dim
                        * count,
                        "samples_ms": samples,
                        "median_ms": statistics.median(samples),
                        "request_repetitions": 1,
                    }
                )
    total_ms = sum(row["median_ms"] * row["request_repetitions"] for row in rows)
    useful = sum(row["useful_flops"] * row["request_repetitions"] for row in rows)
    expected = matrix_flops(cfg, 0, history_tokens, chunk_size)
    candidate = matrix_flops(cfg, history_tokens, candidate_tokens, candidate_tokens)
    expected_total = sum(
        expected[name] + candidate[name] for name in expected if name != "block_sparse_attention"
    )
    if useful != expected_total:
        raise AssertionError("independent API shapes do not conserve complete request matrix work")
    return {
        "rows": rows,
        "equivalent_full_request_gemm_bmm_ms": total_ms,
        "useful_gemm_bmm_flops": useful,
        "useful_mfu_pct": 100 * useful / (total_ms * peak_tflops * 1e9),
        "reference_dense_bf16_tflops": peak_tflops,
        "linear_repeats": linear_repeats,
        "bmm_repeats": bmm_repeats,
        "warmup": 3,
        "boundary": "Independent graph-submitted F.linear on every checkpoint layer and "
        "BF16 BMM at every compressed-key geometry. CUDA events include graph launch. "
        "BMM uses GQA-expanded keys and materializes rectangular scores; causal useful "
        "and dense executed FLOPs differ. Excludes attention QK/PV, softmax, selection, "
        "compression, normalization, RoPE, activation, cache and host validation. "
        "A sum of shape-weighted independent API medians, not an observed full request.",
    }
