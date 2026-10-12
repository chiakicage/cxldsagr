"""Protect physical-traffic units, profile denominator, and missing coverage."""

import pytest

from experiments.deepseek_v32_observation.src.bandwidth import from_metrics, validate_bandwidth


def measured(**changes):
    metrics = {
        "dram__bytes_read.sum": 6_000_000,
        "dram__bytes_write.sum": 2_000_000,
        "gpu__time_duration.sum": 2_000,
        "dram__bytes.sum.peak_sustained_elapsed.per_second": 5_000_000_000_000,
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": 81.0,
        "profiler__replayer_passes": 1,
    }
    return from_metrics(
        {**metrics, **changes},
        inner=20,
        profile_path="profile.ncu-rep",
        profile_sha256="a" * 64,
        replay_mode="kernel-graph",
        time_boundary="NCU graph GPU duration; separate from clean API events",
    )


def test_total_actual_bytes_use_range_time_not_clean_time_or_average_pct():
    row = measured()
    assert row["hbm_read_bytes_per_call"] == 300000
    assert row["hbm_write_bytes_per_call"] == 100000
    assert row["hbm_profile_time_us"] == 0.1
    assert row["hbm_bandwidth_gbps"] == 4000
    assert row["hbm_util_pct"] == 80
    assert row["hbm_ncu_throughput_pct"] == 81
    assert validate_bandwidth({**row, "time_us": 999}) == row


def test_l2_hit_can_have_zero_actual_hbm_bytes():
    row = measured(**{"dram__bytes_read.sum": 0, "dram__bytes_write.sum": 0})
    assert row["hbm_bandwidth_gbps"] == row["hbm_util_pct"] == 0


@pytest.mark.parametrize(
    "change", ["missing", "bad_total", "bad_duration", "bad_util", "cache_flush"]
)
def test_missing_or_mislabeled_profile_rejected(change):
    row = measured()
    if change == "missing":
        del row["hbm_profile_sha256"]
    elif change == "bad_total":
        row["hbm_bytes_per_call"] += 1
    elif change == "bad_duration":
        row["hbm_profile_time_us"] *= 2
    elif change == "bad_util":
        row["hbm_util_pct"] += 1
    else:
        row["hbm_cache_control"] = "all"
    with pytest.raises(ValueError):
        validate_bandwidth(row)


def test_multiple_passes_rejected_for_warm_traffic():
    with pytest.raises(ValueError, match="one profiling pass"):
        measured(profiler__replayer_passes=2)
