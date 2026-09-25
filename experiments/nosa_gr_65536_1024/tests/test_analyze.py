"""Nsight GPU interval and host submission accounting."""

import pytest

from experiments.nosa_gr_65536_1024.src.analyze import activity_summary


def test_nsys_busy_union_excludes_overlap_and_counts_host_late_gap():
    activities = [
        {"start": 0, "end": 10000, "correlationId": 1},
        {"start": 5000, "end": 12000, "correlationId": 2},
        {"start": 20000, "end": 30000, "correlationId": 3},
    ]
    result = activity_summary(activities, {3: {"start": 16000}})
    assert result["gpu_busy_ms"] == 0.022
    assert result["gpu_gap_ms"] == 0.008
    assert result["gap_before_next_cuda_api_ms"] == 0.004
    assert result["gpu_busy_pct"] == pytest.approx(100 * 22 / 30)
