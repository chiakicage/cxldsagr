import pytest

from experiments.nosa_mfu.src.sparse.launch_analysis import timeline, union_ns


def activity(start, end, launch, kind="kernel"):
    return {"start": start, "end": end, "launch_start": launch, "kind": kind}


def test_union_handles_overlap_nesting_and_unsorted_intervals():
    assert union_ns([(10, 20), (0, 15), (2, 4), (30, 40)]) == 30
    assert union_ns([]) == 0
    with pytest.raises(ValueError):
        union_ns([(4, 4)])


def test_gap_separates_late_submission_from_queued_work_and_copies():
    result = timeline(
        [
            activity(0, 10, 0),
            activity(20, 30, 15, "memcpy"),
            activity(35, 45, 18),
        ]
    )
    assert result["hull_ms"] == 45 / 1e6
    assert result["gpu_active_union_ms"] == 30 / 1e6
    assert result["kernel_union_ms"] == 20 / 1e6
    assert result["idle_ms"] == 15 / 1e6
    assert result["before_next_submission_ms"] == 5 / 1e6
    assert result["remaining_gap_ms"] == 10 / 1e6
    assert result["copy_memset_count"] == 1


def test_timestamp_overlap_is_not_negative_idle_or_double_counted():
    result = timeline([activity(10, 30, 0), activity(29, 40, 1), activity(50, 60, 45)])
    assert result["gpu_active_union_ms"] == 40 / 1e6
    assert result["idle_ms"] == 10 / 1e6
    assert result["before_next_submission_ms"] == 5 / 1e6


def test_device_work_cannot_precede_submission():
    with pytest.raises(ValueError):
        timeline([activity(10, 20, 11)])
