"""Composition checks for replay work and cache-hit-sensitive wall normalization."""

import pytest

from experiments.deepseek_v32_motivation.src.flops import (
    LINEARS,
    aggregate_operators,
    phase_ledger,
    precision_evidence,
    request_mfu,
    summarize_mfu,
    work_summary,
)
from experiments.deepseek_v32_motivation.src.measure import INDEXER_DISPATCH_POLICY

MODEL = {
    "hidden_size": 7168,
    "num_attention_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 2048,
    "intermediate_size": 18432,
    "vocab_size": 129280,
}
PRECISIONS = {source: dict.fromkeys(LINEARS, "fp8") for source in range(3)}
PEAKS = {"fp8": 1979.0, "bf16": 989.5, "fp32": 67.0}


def test_verified_disabled_policy_removes_historical_uncertainty_only_with_evidence():
    old = precision_evidence({"run_id": "historical"})
    assert not old["verified_tf32_disabled"]
    assert "did not record TF32 policy" in old["boundary"]
    assert old["fp32_normalization"] == "assumed_fp32_policy"
    new = {
        "precision_settings": {"cuda_matmul_allow_tf32": False},
        "precision_settings_verified_after_execution": True,
    }
    evidence = precision_evidence(new)
    assert evidence["verified_tf32_disabled"]
    assert evidence["fp32_normalization"] == "recorded_fp32_policy"
    assert "counterfactual" in evidence["boundary"]
    assert not precision_evidence({"precision_settings": new["precision_settings"]})[
        "verified_tf32_disabled"
    ]
    new["precision_settings"]["cuda_matmul_allow_tf32"] = True
    assert not precision_evidence(new)["verified_tf32_disabled"]


def ledger(*, phase="history", scheme="echo", history=2050):
    return phase_ledger(
        MODEL,
        {"history_tokens": history, "candidate_tokens": 128, "chunk_size": 1024, "layers": 10},
        PRECISIONS,
        scheme=scheme,
        phase=phase,
    )


def test_chunking_and_source_input_replay_preserve_all_ten_block_computations():
    rows = ledger()
    projections = [row for row in rows if row["name"] == "q_a_proj"]
    assert {(row["source_layer"], row["layer_copies"]) for row in projections} == {
        (0, 4),
        (1, 3),
        (2, 3),
    }
    assert sum(row["layer_copies"] for row in projections) == 3 * 10
    assert sum(row["useful_flops"] for row in projections) == 2 * 2050 * 7168 * 1536 * 10
    indexer = [row for row in rows if row["name"] == "indexer_prefetch"]
    assert sum(row["useful_flops"] for row in indexer) == 2 * 64 * 128 * (2050 * 2051 // 2) * 10


@pytest.mark.parametrize("phase", ["history", "candidate"])
def test_each_forward_has_one_last_token_lm_head_outside_replayed_blocks(phase):
    heads = [row for row in ledger(phase=phase) if row["name"] == "lm_head"]
    assert len(heads) == heads[0]["layer_copies"] == 1
    assert heads[0]["useful_flops"] == 2 * 7168 * 129280
    assert heads[0]["precision"] == "bf16"


def test_resident_and_echo_have_same_useful_work_but_distinct_known_indexer_padding():
    summaries = []
    padding = []
    for scheme in ("hbm", "echo"):
        operators = aggregate_operators(ledger(phase="candidate", scheme=scheme, history=65536))
        summaries.append(work_summary(operators, PEAKS))
        padding.append(
            next(
                row["executed_matmul_flops"]
                for row in operators
                if row["operator"].startswith("indexer")
            )
        )
    assert summaries[0]["useful_flops_by_precision"] == summaries[1]["useful_flops_by_precision"]
    assert padding[0] > padding[1]
    assert summaries[0]["executed_matmul_flops"] is None
    assert "q_a_proj" in summaries[0]["padding_unknown_operators"]


@pytest.mark.parametrize("phase", ["history", "candidate"])
def test_dynamic_echo_dispatch_preserves_work_without_claiming_fused_padding(phase):
    rows = phase_ledger(
        MODEL,
        {
            "history_tokens": 2050,
            "candidate_tokens": 128,
            "chunk_size": 1024,
            "layers": 10,
            "indexer_dispatch_policy": INDEXER_DISPATCH_POLICY,
        },
        PRECISIONS,
        scheme="echo",
        phase=phase,
    )
    indexer = [row for row in rows if row["name"] == "indexer"]
    assert indexer and not any(row["name"] == "indexer_prefetch" for row in rows)
    assert all(row["executed_matmul_flops"] is None for row in indexer)
    assert all(row["executed_formula"] is None for row in indexer)
    assert all("BLOCK_KV" not in row["dimensions"] for row in indexer)
    assert all("padded_pairs" not in row["dimensions"] for row in indexer)
    assert sum(row["useful_flops"] for row in rows) == sum(
        row["useful_flops"] for row in ledger(phase=phase)
    )
    summary = work_summary(aggregate_operators(rows), PEAKS)
    assert "indexer" in summary["padding_unknown_operators"]


def test_mfu_omits_history_work_on_hits_and_weights_by_total_wall_time():
    work = {
        "echo": {
            "history": {"useful_flops_by_precision": {"fp8": 1979_000_000_000}},
            "candidate": {"useful_flops_by_precision": {"bf16": 989_500_000_000}},
        }
    }
    rows = [
        {
            "scheme": "echo",
            "request_id": index,
            "is_revisit": True,
            "prefix_cache_hit": True,
            "prefix_ms": 0.01,
            "extend_ms": duration,
            "latency_ms": duration + 0.01,
        }
        for index, duration in enumerate((1, 9))
    ]
    requests = request_mfu(rows, work, PEAKS)
    summary = summarize_mfu(requests)
    prefix = next(row for row in summary if row["stage"] == "prefix")
    extend = next(row for row in summary if row["stage"] == "extend")
    assert prefix["history_builds"] == 0 and prefix["effective_mfu_pct"] is None
    assert extend["effective_mfu_pct"] == pytest.approx(20)
    rows[0]["prefix_cache_hit"] = False
    result = request_mfu(rows[:1], work, PEAKS)
    e2e = next(row for row in result if row["stage"] == "e2e")
    assert e2e["ideal_compute_ms"] == 2
    assert e2e["useful_fp8_flops"] == 1979_000_000_000
