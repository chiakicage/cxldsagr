"""Weighted aggregate MFU with mixed indexer paths and complete-work checks."""

import pytest

from experiments.deepseek_v32_motivation.src.aggregate_mfu import aggregate


def operator(*, segment="candidate", stage="indexer_qk", ideal=1, active=4, **updates):
    return {
        "capture_index": "1",
        "scheme": "echo",
        "phase": "cold",
        "segment": segment,
        "stage": stage,
        "precision": "FP8",
        "calls": 1,
        "useful_flops": 1000 * ideal,
        "ideal_ms": ideal,
        "operator_gpu_active_ms": active,
        **updates,
    }


def formal(stage, ideal, wall, *, visit="first"):
    return {
        "scheme": "echo",
        "visit_kind": visit,
        "stage": stage,
        "requests": 16,
        "ideal_compute_mean_ms": ideal,
        "wall_mean_ms": wall,
        "effective_mfu_pct": 100 * ideal / wall,
    }


def test_mixed_resident_fused_indexer_is_preserved_and_request_totals_are_weighted():
    ops = [
        operator(segment="history", active=2),
        operator(segment="history", stage="indexer_fused", ideal=3, active=12),
        operator(),
    ]
    measured = [formal("prefix", 4, 28), formal("extend", 1, 8), formal("e2e", 5, 40)]
    comparison, stages = aggregate(ops, measured)
    request = next(row for row in comparison if row["segment"] == "request")
    assert request["operator_gpu_active_ms"] == 18
    assert request["operator_gpu_active_mfu_pct"] == pytest.approx(100 * 5 / 18)
    assert request["formal_wall_to_operator_api_ratio"] == pytest.approx(40 / 18)
    assert request["formal_mfu_to_operator_mfu_ratio"] == pytest.approx(18 / 40)
    assert request["calls"] == 3
    assert {row["stage"] for row in stages} == {"indexer_qk", "indexer_fused"}
    assert len(stages) == 3


def test_multiple_captures_use_mean_duration_and_sum_work_not_mean_percentages():
    ops = [
        operator(phase="revisit", active=5),
        operator(phase="revisit", capture_index="2", active=15),
    ]
    measured = [formal("extend", 1, 20, visit="revisit"), formal("e2e", 1, 24, visit="revisit")]
    comparison, _ = aggregate(ops, measured)
    candidate = next(row for row in comparison if row["segment"] == "candidate")
    assert candidate["captures"] == 2
    assert candidate["operator_gpu_active_mean_ms"] == 10
    assert candidate["operator_gpu_active_mfu_pct"] == 10
    assert candidate["formal_wall_to_operator_api_ratio"] == 2


def test_missing_history_work_cannot_be_reported_as_a_full_request_api_total():
    measured = [formal("extend", 1, 8), formal("e2e", 5, 40)]
    with pytest.raises(ValueError, match="matrix work differs"):
        aggregate([operator()], measured)


def test_duplicate_groups_and_mismatched_formal_mfu_fail():
    measured = [formal("extend", 1, 8), formal("e2e", 1, 10)]
    with pytest.raises(ValueError, match="duplicate operator"):
        aggregate([operator(), operator()], measured)
    measured[0]["effective_mfu_pct"] = 999
    with pytest.raises(ValueError, match="arithmetic differs"):
        aggregate([operator()], measured)
