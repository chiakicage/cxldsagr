"""Independent review must detect radix ordering and memory-accounting mistakes."""

from copy import deepcopy

import pytest
import torch

from experiments.deepseek_v32_echo_official.src import review_q1_fused_prepare_model as review


def test_full_score_radix_order_preserves_signed_zero_and_ascending_id_ties():
    scores = torch.full((1, 65537), -1.0)
    scores[0, :2046] = 1.0
    scores[0, 2046] = -0.0
    scores[0, 60000] = 0.0
    indices = torch.tensor([list(range(2046)) + [60000, 2046]], dtype=torch.int32)
    review.exact_radix_topk(scores, indices)
    wrong = indices.clone()
    wrong[0, -2:] = wrong[0, -2:].flip(0)
    with pytest.raises(ValueError, match="radix top-k ordering"):
        review.exact_radix_topk(scores, wrong)


def test_raw_bit_comparison_rejects_numerically_equal_signed_zero():
    with pytest.raises(ValueError, match="tensor bits differ"):
        review.exact(torch.tensor([0.0]), torch.tensor([-0.0]), "signed zero")


@pytest.fixture
def memory():
    arm = {
        "allocated": 100,
        "reserved": 200,
        "device_used": 300,
        "graph": {
            "private_reserved_bytes": 64,
            "chosen_private_limit_bytes": 128,
            "static_allocated_bytes": 8,
            "reservation_bytes": 136,
            "memory_at_allocation": {
                "pytorch_allocated": 80,
                "pytorch_reserved": 160,
                "device_used": 240,
                "device_total": 1000,
            },
        },
    }
    return {name: deepcopy(arm) for name in review.ARMS}


def test_memory_reports_private_delta_without_treating_process_snapshots_as_model_footprints(
    memory,
):
    memory["candidate"]["graph"]["private_reserved_bytes"] = 80
    result = review.memory_audit(memory)
    assert result["candidate_minus_baseline_private_reserved_bytes"] == 16
    assert "not isolated model footprints" in result["boundary"]


@pytest.mark.parametrize(
    "defect", ["boolean", "negative", "hierarchy", "private_limit", "reservation"]
)
def test_memory_rejects_invalid_or_under_accounted_counters(memory, defect):
    row = memory["candidate"]
    if defect == "boolean":
        row["allocated"] = True
    elif defect == "negative":
        row["allocated"] = -1
    elif defect == "hierarchy":
        row["allocated"] = row["reserved"] + 1
    elif defect == "private_limit":
        row["graph"]["private_reserved_bytes"] = row["graph"]["chosen_private_limit_bytes"] + 1
    else:
        row["graph"]["reservation_bytes"] -= 1
    with pytest.raises(ValueError):
        review.memory_audit(memory)
