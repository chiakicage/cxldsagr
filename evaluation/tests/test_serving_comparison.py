"""Reject incomparable controls and expose stable latency/allocation changes."""

from copy import deepcopy

import pytest

from evaluation.serving_comparison import (
    ACCOUNTING,
    MEMORY,
    TIMINGS,
    assess_repeats,
    compare_versions,
    memory_value,
    summarize,
    trace_memory,
)


def run(name, *, role, latency=10.0, source="frozen", uuid="gpu0", charge=100):
    return {
        "run_id": name,
        "role": role,
        "source_sha256": source,
        "validation_identity_sha256": source,
        "contract": {"hardware": {"uuid": uuid}, "workload_sha256": "same-input"},
        "total_trace_latency_ms": 2 * latency,
        "groups": {
            "hbm/revisit": {
                "timings": {name: summarize([latency - 1, latency + 1]) for name in TIMINGS},
                "accounting": dict.fromkeys(ACCOUNTING, charge),
                "requests": 2,
                "prefix_hits": 0,
            }
        },
    }


def versions():
    return (
        run("published", role="published", source="historical", uuid="gpu5"),
        [run(f"baseline{i}", role="baseline", latency=10 + i / 10) for i in range(3)],
        [run(f"current{i}", role="current", latency=12 + i / 10, source="new") for i in range(3)],
    )


def test_stable_small_slowdown_is_not_hidden_by_a_percentage_allowance():
    result = assess_repeats([100, 100.01, 100.02], [100.04, 100.05, 100.06])
    assert result["delta_pct"] < 0.1
    assert result["status"] == "investigate_slowdown"
    assert result["delta"] > result["baseline"]["range"]


def test_small_increase_within_observed_noise_is_not_called_a_pass():
    result = assess_repeats([9, 10, 11], [9.1, 10.1, 11.1])
    assert result["status"] == "increase_within_observed_repeat_range"
    assert result["baseline"]["mad"] == 1
    assert result["baseline"]["values"] == [9, 10, 11]


def test_three_identities_and_hbm_revisit_misses_remain_distinct():
    published, baseline, current = versions()
    result = compare_versions(published, baseline, current)
    assert result["published_contract_differences"] == ["hardware"]
    assert len(result["runs"]) == 7
    assert result["runs"][1]["groups"]["hbm/revisit"]["prefix_hits"] == 0
    assert all(item["status"] == "investigate_slowdown" for item in result["comparisons"])


@pytest.mark.parametrize("changed", ["gpu", "workload", "source", "run_id"])
def test_mixed_or_nonindependent_controls_are_rejected(changed):
    published, baseline, current = versions()
    if changed == "gpu":
        current[1]["contract"]["hardware"]["uuid"] = "other"
    elif changed == "workload":
        current[1]["contract"]["workload_sha256"] = "other"
    elif changed == "source":
        current[1]["source_sha256"] = "mixed"
    else:
        current[1]["run_id"] = baseline[0]["run_id"]
    with pytest.raises(ValueError):
        compare_versions(published, baseline, current)


def test_allocation_growth_is_reported_separately_from_latency_improvement():
    published, baseline, current = versions()
    for item in current:
        item["groups"] = deepcopy(baseline[0]["groups"])
        item["groups"]["hbm/revisit"]["accounting"]["cache_hbm_bytes"] += 512
    result = compare_versions(published, baseline, current)
    growth = [item for item in result["accounting_comparisons"] if item["increased"]]
    assert len(growth) == 1 and growth[0]["delta_bytes"] == 512


def test_a_relabelled_baseline_is_not_a_refactored_version():
    published, baseline, current = versions()
    for item in current:
        item["source_sha256"] = baseline[0]["source_sha256"]
    with pytest.raises(ValueError, match="distinct identities"):
        compare_versions(published, baseline, current)


def test_postrequest_memory_samples_do_not_replace_allocator_peaks():
    item = {
        "groups": {
            "hbm/first": {"sampled_memory": dict(zip(MEMORY, (100, 200, 300), strict=True))},
            "hbm/revisit": {"sampled_memory": dict(zip(MEMORY, (150, 200, 250), strict=True))},
        },
        "allocator_peaks": {
            "hbm": {"torch_peak_allocated_bytes": 180, "torch_peak_reserved_bytes": 240}
        },
    }
    assert trace_memory(item) == {
        "torch_allocated_bytes": 150,
        "torch_reserved_bytes": 200,
        "device_used_bytes": 300,
        "torch_peak_allocated_bytes": 180,
        "torch_peak_reserved_bytes": 240,
    }


def test_deepseek_device_usage_uses_its_recorded_free_and_total_bytes():
    assert (
        memory_value({"cuda_total_bytes": 1000, "cuda_free_bytes": 300}, "device_used_bytes") == 700
    )
    assert memory_value({"device_used_bytes": 400}, "device_used_bytes") == 400


@pytest.mark.parametrize("invalid", [[1, 2], [1, 2, float("nan")], [1, 2, -1]])
def test_incomplete_or_invalid_repeat_evidence_cannot_establish_a_comparison(invalid):
    with pytest.raises(ValueError):
        assess_repeats([1, 2, 3], invalid)
