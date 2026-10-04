"""Audit matching matrix work and compare it with accepted runner latency.

Raw GEMM/BMM omits sparse attention. Its numerator therefore remains a named
subset; adding a separately profiled attention API yields only a composition
estimate, never an observed standalone matrix reference for the full request.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter

from experiments.nosa_motivation.src.flops import matrix_flops, model_dimensions
from experiments.nosa_motivation.src.matrix_baseline import invocation_geometry

LINEARS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "cis_projection")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _close(actual, expected):
    return type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-12)


def _work(model_config, config, *, cached):
    history, candidate, chunk = (
        config[name] for name in ("history_tokens", "candidate_tokens", "chunk_size")
    )
    work = matrix_flops(model_config, history, candidate, candidate)
    if not cached:
        prefix = matrix_flops(model_config, 0, history, chunk)
        work = {name: value + prefix[name] for name, value in work.items()}
    return work


def audit_matrix_native_provenance(raw_native, replay_native):
    """Allow only explicitly identified additions after serving replay closes."""
    build = replay_native["build_after"]
    _require(
        bool(build)
        and build == replay_native["build_before"]
        and raw_native["build_before"] == build
        and raw_native["build_after"] == build,
        "raw matrix native build identity differs from replay",
    )
    before, after = raw_native["artifacts_before"], raw_native["artifacts_after"]
    _require(
        bool(before) and before == replay_native["artifacts_final"],
        "raw matrix initial native artifacts differ from replay final",
    )
    _require(
        all(after.get(path) == identity for path, identity in before.items()),
        "raw matrix changed or unloaded a preexisting native artifact",
    )
    return {
        "build_sha256": build["sha256"],
        "replay_artifact_count": len(before),
        "raw_only_artifacts": {path: after[path] for path in sorted(set(after).difference(before))},
        "boundary": "All libraries mapped before the independent raw matrix benchmark match the replay's final inventory and retain identical recorded file identities afterward. Newly mapped libraries are classified only as raw-benchmark additions; they are not evidence that replay used those libraries. Native build/dependency identity is unchanged throughout.",
    }


def audit_matrix_reference(reference, model_config, config):
    """Recompute shape coverage, FLOPs and timing totals from every raw row."""
    cfg = model_dimensions(model_config)
    history, candidate, chunk = (
        config[name] for name in ("history_tokens", "candidate_tokens", "chunk_size")
    )
    peak = config["peak_bf16_tflops"]
    _require(_positive(peak), "invalid declared dense BF16 peak")
    _require(reference["reference_dense_bf16_tflops"] == peak, "raw API peak differs")
    geometry = invocation_geometry(history, candidate, chunk)
    multiplicities = Counter(queries for _, queries in geometry)
    q_width = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_width = cfg["num_key_value_heads"] * cfg["head_dim"]
    shapes = {
        "qkv_proj": [q_width + 2 * kv_width, cfg["hidden_size"]],
        "o_proj": [cfg["hidden_size"], q_width],
        "gate_up_proj": [2 * cfg["intermediate_size"], cfg["hidden_size"]],
        "down_proj": [cfg["hidden_size"], cfg["intermediate_size"]],
        "cis_projection": [cfg["num_key_value_heads"], kv_width],
    }
    expected = {
        (name, queries): count for queries, count in multiplicities.items() for name in LINEARS
    }
    expected.update(
        {
            ("indexer_qk_bmm", start, queries): 1
            for start, queries in geometry
            if (start + queries + 63) // 64 > 64
        }
    )
    indexed = {}
    for row in reference["rows"]:
        operation, queries = row["operation"], row["queries"]
        _require(operation in (*LINEARS, "indexer_qk_bmm"), "unknown raw API operation")
        key = (
            (operation, row["query_start"], queries)
            if operation == "indexer_qk_bmm"
            else (operation, queries)
        )
        _require(
            key in expected and key not in indexed, "raw API geometry is missing or duplicated"
        )
        _require(row["request_repetitions"] == expected[key], "raw API repetition count differs")
        _require(row["layers"] == cfg["num_hidden_layers"], "raw API omits checkpoint layers")
        repeats = reference["bmm_repeats" if operation == "indexer_qk_bmm" else "linear_repeats"]
        samples = row["samples_ms"]
        _require(
            type(repeats) is int and repeats > 0 and len(samples) == repeats,
            "raw API sample count differs",
        )
        _require(all(_positive(value) for value in samples), "invalid raw API duration")
        _require(
            _close(row["median_ms"], statistics.median(samples)),
            "raw API median differs from samples",
        )
        if operation == "indexer_qk_bmm":
            start = row["query_start"]
            keys = (start + queries) // 16 - 1
            _require(row["compressed_keys"] == keys, "raw BMM compressed-key count differs")
            useful = matrix_flops(cfg, start, queries, queries)["indexer_qk"]
            dense = (
                2
                * cfg["num_hidden_layers"]
                * cfg["num_attention_heads"]
                * queries
                * cfg["head_dim"]
                * keys
            )
        else:
            _require(
                row["weight_shape"] == shapes[operation], "raw linear checkpoint geometry differs"
            )
            useful = dense = matrix_flops(cfg, 0, queries, queries)[operation]
        _require(
            row["useful_flops"] == useful and row["dense_executed_flops"] == dense,
            "raw API FLOPs differ from geometry",
        )
        indexed[key] = row
    _require(set(indexed) == set(expected), "raw API geometry coverage is incomplete")
    results = {}
    for cached, name in ((False, "full_request"), (True, "candidate_only")):
        weights = {
            key: (
                int(key == ("indexer_qk_bmm", history, candidate))
                if key[0] == "indexer_qk_bmm"
                else int(key[1] == candidate)
            )
            if cached
            else row["request_repetitions"]
            for key, row in indexed.items()
        }
        elapsed = sum(indexed[key]["median_ms"] * count for key, count in weights.items())
        useful = sum(indexed[key]["useful_flops"] * count for key, count in weights.items())
        work = _work(cfg, config, cached=cached)
        _require(
            useful == sum(value for key, value in work.items() if key != "block_sparse_attention"),
            "raw matrix work does not conserve request subset",
        )
        results[name] = {
            "equivalent_gemm_bmm_ms": elapsed,
            "useful_gemm_bmm_flops": useful,
            "dense_executed_gemm_bmm_flops": sum(
                indexed[key]["dense_executed_flops"] * count for key, count in weights.items()
            ),
            "useful_mfu_pct": 100 * useful / (elapsed * peak * 1e9),
        }
    full = results["full_request"]
    _require(
        _close(reference["equivalent_full_request_gemm_bmm_ms"], full["equivalent_gemm_bmm_ms"]),
        "raw API total duration differs",
    )
    _require(
        reference["useful_gemm_bmm_flops"] == full["useful_gemm_bmm_flops"]
        and _close(reference["useful_mfu_pct"], full["useful_mfu_pct"]),
        "raw API total FLOPs or MFU differs",
    )
    return results


def compare_request_apis(accepted_row, model_config, config, api_timeline, raw_reference):
    """Use the accepted runner timer, including admission, for every wall metric."""
    cached = accepted_row["prefix_cache_hit"]
    _require(type(cached) is bool, "request cache-hit state must be boolean")
    batches = (
        1
        if cached
        else 1 + (config["history_tokens"] + config["chunk_size"] - 1) // config["chunk_size"]
    )
    calls = config["layers"] * batches
    expected_scopes = {"matrix_api/indexer": calls, "matrix_api/attention": calls}
    if config["enable_compute_graphs"]:
        expected_scopes.update(
            {"matrix_api/graph_project": calls, "matrix_api/graph_finish": calls}
        )
    else:
        expected_scopes.update(
            {
                "matrix_api/linear": 4 * calls,
                "matrix_api/cis_projection": calls,
                "matrix_api/norm_helper": 2 * calls,
                "matrix_api/mlp_helper": calls,
                "matrix_api/project_helper": calls,
            }
        )
    _require(
        api_timeline["api_scope_counts"] == expected_scopes
        and api_timeline["api_scopes_with_activity"] == expected_scopes
        and api_timeline["uncorrelated_activities"] == 0,
        "incomplete request compute API attribution",
    )
    api_ms = api_timeline["api_execution_span_union_ms"]
    activity_ms = api_timeline["api_gpu_union_ms"]
    wall_ms = accepted_row.get("latency_ms")
    attention_ms = api_timeline["api_execution_span_ms"]["matrix_api/attention"]
    _require(
        all(_positive(value) for value in (api_ms, activity_ms, attention_ms))
        and (wall_ms is None or _positive(wall_ms))
        and activity_ms <= api_ms + 1e-9
        and attention_ms <= api_ms + 1e-9,
        "invalid request/API duration",
    )
    work = _work(model_config, config, cached=cached)
    _require(
        wall_ms is None or accepted_row["effective_work"]["request_flops"] == work,
        "accepted request matrix work differs",
    )
    full_work = sum(work.values())
    peak = config["peak_bf16_tflops"]
    raw = audit_matrix_reference(raw_reference, model_config, config)[
        "candidate_only" if cached else "full_request"
    ]
    matrix_work = raw["useful_gemm_bmm_flops"]
    raw_ms = raw["equivalent_gemm_bmm_ms"]
    composition_ms = raw_ms + attention_ms
    return {
        "method": accepted_row["method"],
        "request_id": accepted_row["request_id"],
        "prefix_cache_hit": cached,
        "accepted_runner_latency_ms": wall_ms,
        "useful_full_model_flops": full_work,
        "request_wall_mfu_pct": None
        if wall_ms is None
        else 100 * full_work / (wall_ms * peak * 1e9),
        "complete_compute_api_gpu_span_union_ms": api_ms,
        "compute_api_gpu_activity_union_ms": activity_ms,
        "compute_api_gpu_activity_only_mfu_pct": 100 * full_work / (activity_ms * peak * 1e9),
        "complete_compute_api_mfu_pct": 100 * full_work / (api_ms * peak * 1e9),
        "wall_to_complete_api_mfu_ratio": None if wall_ms is None else api_ms / wall_ms,
        "independent_gemm_bmm": {
            **raw,
            "same_subset_over_request_wall_mfu_pct": None
            if wall_ms is None
            else 100 * matrix_work / (wall_ms * peak * 1e9),
            "wall_to_raw_same_subset_mfu_ratio": None if wall_ms is None else raw_ms / wall_ms,
            "excluded_useful_attention_flops": work["block_sparse_attention"],
        },
        "composition_estimate": {
            "profiled_attention_api_ms": attention_ms,
            "raw_matrix_plus_attention_ms": composition_ms,
            "full_work_mfu_pct": 100 * full_work / (composition_ms * peak * 1e9),
            "wall_to_composition_mfu_ratio": None if wall_ms is None else composition_ms / wall_ms,
            "is_observed_request": False,
        },
        "boundary": (
            "Independent check has no benchmark wall latency. API spans and API MFUs are diagnostic; "
            "all runner wall metrics/ratios are null. A separate clean bench is required for "
            "end-to-end performance or API-proximity conclusions."
            if wall_ms is None
            else "Wall latency is the accepted uninstrumented runner request including token validation/upload, admission/eviction, required history construction, candidate, and cleanup. Complete API GPU execution-span unions come from a separate exact-output profile: first correlated device activity to last completion per invocation, including intra-API gaps but excluding earlier host work. They are not API CPU/wall latency. Activity-only unions are also retained. Raw F.linear/BMM medians cover the matching GEMM/BMM subset and omit sparse QK/PV and helpers; BMM expands GQA and materializes rectangular scores. Full-model and raw-subset MFUs have different numerators. Their same-subset ratio uses GEMM/BMM FLOPs over full runner wall time. Raw matrix plus profiled attention is only a composition estimate; it does not establish an independent full-model matrix-API baseline or a passed proximity gate."
        ),
    }
