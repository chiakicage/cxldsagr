"""Display fused capability independently of observed transfer counts."""

from copy import deepcopy

import pytest

from experiments.deepseek_v32_echo_official.src import decode_display as display


@pytest.mark.parametrize(
    ("potential_io", "actual_io"),
    [
        (True, None),
        (False, {"status": "phase_zero", "records": 0, "bytes": 0}),
        (True, {"status": "unique_positive", "records": 64, "bytes": 73728}),
    ],
)
def test_same_fused_hatch_preserves_unknown_zero_and_positive_io_evidence(potential_io, actual_io):
    row = {
        "kind": "kernel",
        "name": "void sm90_fp8_paged_mqa_logits_fused_v2<128>()",
        "lane": "Compute",
        "category": "Indexer / top-k",
        "potential_io": potential_io,
        "actual_io": actual_io,
        "start_ns": 100,
        "end_ns": 200,
    }
    before = deepcopy(row)
    assert display.activity_style(row) == ("Compute", "Indexer / top-k", True)
    assert row == before


def test_resident_scoring_is_not_marked_as_fused_prefetch():
    row = {
        "kind": "kernel",
        "name": "void sm90_fp8_paged_mqa_logits<128>()",
        "lane": "Compute",
        "category": "Indexer / top-k",
        "potential_io": False,
    }
    assert display.activity_style(row) == ("Compute", "Indexer / top-k", False)


def test_recall_display_keeps_its_separate_unknown_bytes_boundary():
    row = {
        "kind": "kernel",
        "name": "recall_update_with_prefetch",
        "lane": "GPU control",
        "category": display.RECALL_CATEGORY,
        "potential_io": True,
    }
    assert display.activity_style(row) == ("H2D", "H2D", False)
