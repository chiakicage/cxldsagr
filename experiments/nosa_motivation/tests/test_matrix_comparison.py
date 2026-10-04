"""Keep request admission, raw subset work and composition estimates distinct."""

from collections import Counter
from copy import deepcopy

import pytest

from experiments.nosa_motivation.src.flops import matrix_flops
from experiments.nosa_motivation.src.matrix_baseline import invocation_geometry
from experiments.nosa_motivation.src.matrix_comparison import (
    audit_matrix_native_provenance,
    audit_matrix_reference,
    compare_request_apis,
)


def evidence():
    model = {
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
    }
    config = {
        "history_tokens": 4224,
        "candidate_tokens": 128,
        "chunk_size": 1024,
        "layers": 2,
        "enable_compute_graphs": True,
        "peak_bf16_tflops": 100,
    }
    geometry = invocation_geometry(4224, 128, 1024)
    rows = []
    shapes = {
        "qkv_proj": [16, 8],
        "o_proj": [8, 8],
        "gate_up_proj": [32, 8],
        "down_proj": [8, 16],
        "cis_projection": [2, 4],
    }
    for queries, count in Counter(queries for _, queries in geometry).items():
        flops = matrix_flops(model, 0, queries, queries)
        for operation, shape in shapes.items():
            rows.append(
                {
                    "operation": operation,
                    "queries": queries,
                    "layers": 2,
                    "weight_shape": shape,
                    "useful_flops": flops[operation],
                    "dense_executed_flops": flops[operation],
                    "samples_ms": [1.0, 3.0, 2.0],
                    "median_ms": 2.0,
                    "request_repetitions": count,
                }
            )
    for start, queries in geometry:
        if (start + queries + 63) // 64 <= 64:
            continue
        keys = (start + queries) // 16 - 1
        rows.append(
            {
                "operation": "indexer_qk_bmm",
                "queries": queries,
                "query_start": start,
                "compressed_keys": keys,
                "layers": 2,
                "useful_flops": matrix_flops(model, start, queries, queries)["indexer_qk"],
                "dense_executed_flops": 2 * 2 * 4 * queries * 2 * keys,
                "samples_ms": [5.0, 3.0, 4.0],
                "median_ms": 4.0,
                "request_repetitions": 1,
            }
        )
    useful = sum(row["useful_flops"] * row["request_repetitions"] for row in rows)
    reference = {
        "rows": rows,
        "reference_dense_bf16_tflops": 100,
        "linear_repeats": 3,
        "bmm_repeats": 3,
        "equivalent_full_request_gemm_bmm_ms": 68.0,
        "useful_gemm_bmm_flops": useful,
        "useful_mfu_pct": 100 * useful / (68.0 * 100 * 1e9),
    }
    return model, config, reference


def request_evidence(model, config, *, cached):
    work = matrix_flops(model, 4224, 128, 128)
    if not cached:
        prefix = matrix_flops(model, 0, 4224, 1024)
        work = {key: value + prefix[key] for key, value in work.items()}
    calls = 2 if cached else 12
    row = {
        "method": "hbm",
        "request_id": 16 if cached else 0,
        "prefix_cache_hit": cached,
        "latency_ms": 25.0 if cached else 100.0,
        "extend_ms": 10.0,
        "effective_work": {"request_flops": work},
    }
    timeline = {
        "api_scope_counts": {
            f"matrix_api/{name}": calls
            for name in ("indexer", "attention", "graph_project", "graph_finish")
        },
        "uncorrelated_activities": 0,
        "api_gpu_union_ms": 16.0 if cached else 70.0,
        "api_execution_span_union_ms": 18.0 if cached else 80.0,
        "api_execution_span_ms": {"matrix_api/attention": 4.0 if cached else 20.0},
    }
    timeline["api_scopes_with_activity"] = dict(timeline["api_scope_counts"])
    return row, timeline


def test_candidate_uses_one_call_when_history_tail_has_the_same_query_shape():
    model, config, raw = evidence()
    result = audit_matrix_reference(raw, model, config)
    assert result["full_request"]["equivalent_gemm_bmm_ms"] == 68.0
    assert result["candidate_only"]["equivalent_gemm_bmm_ms"] == 14.0
    candidate = matrix_flops(model, 4224, 128, 128)
    assert result["candidate_only"]["useful_gemm_bmm_flops"] == sum(
        value for key, value in candidate.items() if key != "block_sparse_attention"
    )


def test_independent_check_profile_has_no_runner_wall_claims():
    model, config, raw = evidence()
    row, timeline = request_evidence(model, config, cached=False)
    for key in ("latency_ms", "extend_ms", "effective_work"):
        row.pop(key)
    result = compare_request_apis(row, model, config, timeline, raw)
    assert result["accepted_runner_latency_ms"] is None
    assert result["request_wall_mfu_pct"] is None
    assert result["wall_to_complete_api_mfu_ratio"] is None
    assert result["complete_compute_api_mfu_pct"] > 0


@pytest.mark.parametrize("cached", [False, True])
def test_comparison_keeps_admission_in_wall_and_attention_out_of_raw_subset(cached):
    model, config, raw = evidence()
    row, timeline = request_evidence(model, config, cached=cached)
    result = compare_request_apis(row, model, config, timeline, raw)
    assert result["accepted_runner_latency_ms"] == row["latency_ms"]
    assert result["wall_to_complete_api_mfu_ratio"] == (0.72 if cached else 0.8)
    assert result["compute_api_gpu_activity_only_mfu_pct"] > result["complete_compute_api_mfu_pct"]
    subset = result["independent_gemm_bmm"]
    assert subset["wall_to_raw_same_subset_mfu_ratio"] == (0.56 if cached else 0.68)
    assert result["useful_full_model_flops"] == (
        subset["useful_gemm_bmm_flops"] + subset["excluded_useful_attention_flops"]
    )
    assert result["composition_estimate"]["is_observed_request"] is False
    assert result["composition_estimate"]["raw_matrix_plus_attention_ms"] == (
        18.0 if cached else 88.0
    )


@pytest.mark.parametrize("change", ["missing", "duplicate", "median", "flops", "keys"])
def test_raw_reference_rejects_incomplete_or_inconsistent_evidence(change):
    model, config, raw = evidence()
    raw = deepcopy(raw)
    if change == "missing":
        raw["rows"].pop()
    elif change == "duplicate":
        raw["rows"].append(raw["rows"][0])
    elif change == "median":
        raw["rows"][0]["median_ms"] = 0.5
    elif change == "flops":
        raw["rows"][0]["useful_flops"] += 1
    else:
        raw["rows"][-1]["compressed_keys"] -= 1
    with pytest.raises(ValueError):
        audit_matrix_reference(raw, model, config)


def test_missing_compute_scope_cannot_produce_api_mfu():
    model, config, raw = evidence()
    row, timeline = request_evidence(model, config, cached=False)
    timeline["api_scope_counts"]["matrix_api/attention"] -= 1
    with pytest.raises(ValueError, match="incomplete request compute API"):
        compare_request_apis(row, model, config, timeline, raw)


def native_evidence():
    build = {"sha256": "build", "dependencies": {"cuda": "headers"}}
    before = {"/runtime.so": {"sha256": "runtime", "size": 30}}
    replay = {"build_before": build, "build_after": build, "artifacts_final": before}
    raw = {
        "build_before": deepcopy(build),
        "build_after": deepcopy(build),
        "artifacts_before": deepcopy(before),
        "artifacts_after": {**deepcopy(before), "/matrix.so": {"sha256": "matrix", "size": 50}},
    }
    return raw, replay


def test_native_reference_additions_are_raw_only_and_do_not_change_replay_inventory():
    raw, replay = native_evidence()
    result = audit_matrix_native_provenance(raw, replay)
    assert result["replay_artifact_count"] == 1
    assert result["raw_only_artifacts"] == {"/matrix.so": {"sha256": "matrix", "size": 50}}
    assert set(replay["artifacts_final"]) == {"/runtime.so"}


@pytest.mark.parametrize("change", ["build", "before", "changed", "unloaded"])
def test_raw_matrix_cannot_replace_build_or_preexisting_artifacts(change):
    raw, replay = native_evidence()
    if change == "build":
        raw["build_after"]["dependencies"]["cuda"] = "changed headers"
    elif change == "before":
        raw["artifacts_before"]["/extra.so"] = {"sha256": "unexpected"}
    elif change == "changed":
        raw["artifacts_after"]["/runtime.so"]["sha256"] = "replaced"
    else:
        del raw["artifacts_after"]["/runtime.so"]
    with pytest.raises(ValueError):
        audit_matrix_native_provenance(raw, replay)
