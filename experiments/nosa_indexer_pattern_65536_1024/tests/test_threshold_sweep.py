"""Independent interval and hand-calculated objective checks for threshold sweeps."""

import json
import math
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from experiments.nosa_indexer_pattern_65536_1024.src.threshold_sweep import (
    sweep_thresholds,
    threshold_regions,
)
from experiments.nosa_indexer_pattern_65536_1024.tests.test_overlap import source_report


def contains(region, threshold):
    lower = region["lower_pct"]
    return (threshold > lower or (threshold == lower and region["lower_inclusive"])) and (
        threshold <= region["upper_pct"]
    )


def conflicting_objectives_source():
    # At 50 GB/s and MFU scale .5, layer 1 has a .1 ms attention window.
    # Its dense fetch is 2 ms, hiding 1.5 ms in layer 0's other work (75%).
    # Its sparse fetch is .4 ms, hiding .1 ms (25%) but exposing only .3 ms.
    return source_report([[0], [20_000_000]], attention=[0.1, 0.05], other=[1.5, 0.25])


@pytest.mark.parametrize(
    ("coverages", "bounds"),
    [
        ([20, 20, 50, 100], [(0, 20), (20, 50), (50, 100)]),
        ([0, 0, 50, 100], [(0, 0), (0, 50), (50, 100)]),
        ([100], [(0, 100)]),
        ([0], [(0, 0), (0, 100)]),
        ([20], [(0, 20), (20, 100)]),
    ],
)
def test_regions_partition_zero_to_100_with_dense_equality_and_deduplicated_breaks(
    coverages, bounds
):
    regions = threshold_regions(coverages)
    assert [(row["lower_pct"], row["upper_pct"]) for row in regions] == bounds
    assert len({row["region_id"] for row in regions}) == len(regions)
    assert [row["lower_inclusive"] for row in regions] == [True] + [False] * (len(regions) - 1)
    assert all(row["upper_inclusive"] for row in regions)
    assert sum(contains(region, 0) for region in regions) == 1
    assert sum(contains(region, 100) for region in regions) == 1
    for region in regions:
        representative = region["representative_threshold_pct"]
        assert contains(region, representative)
        assert tuple(value < representative for value in coverages) == tuple(
            value < region["upper_pct"] for value in coverages
        )


def test_interval_count_and_classifications_match_brute_force_dense_grid():
    coverages = [0, 13.37, 13.37, 50, 99.9, 100]
    regions = threshold_regions(coverages)
    grid = set(np.linspace(0, 100, 10001).tolist()) | set(coverages)
    for value in coverages:
        for direction in (-math.inf, math.inf):
            adjacent = math.nextafter(value, direction)
            if 0 <= adjacent <= 100:
                grid.add(adjacent)
    observed = set()
    for threshold in grid:
        matches = [region for region in regions if contains(region, threshold)]
        assert len(matches) == 1
        signature = tuple(value < threshold for value in coverages)
        observed.add(signature)
        assert signature == tuple(
            value < matches[0]["representative_threshold_pct"] for value in coverages
        )
    assert len(regions) == len(observed)
    assert len(observed) == len(
        {
            tuple(value < row["representative_threshold_pct"] for value in coverages)
            for row in regions
        }
    )


def test_representative_remains_inside_a_region_between_adjacent_floats():
    first = 20.0
    second = math.nextafter(first, math.inf)
    regions = threshold_regions([first, second])
    for region in regions:
        assert contains(region, region["representative_threshold_pct"])
    middle = next(row for row in regions if row["lower_pct"] == first)
    representative = middle["representative_threshold_pct"]
    assert first < representative <= second


def test_same_coverage_heads_change_class_together_and_100_percent_never_becomes_sparse():
    source = source_report([[20_000_000, 20_000_000, 100_000_000]], attention=[0.1], other=[1.0])
    report = sweep_thresholds(source)
    assert len(report["regions"]) == 2
    assert [row["sparse_head_count"] for row in report["regions"]] == [0, 2]
    samples = {row["threshold_pct"]: row for row in report["samples"]}
    assert samples[20]["sparse_head_count"] == 0
    assert samples[21]["sparse_head_count"] == 2
    assert samples[100]["sparse_head_count"] == 2
    assert samples[100]["dense_head_count"] == 1


def test_noninteger_percentage_boundary_is_dense_in_regions_and_baseline():
    source = source_report([[1]], attention=[0.1], other=[0.2], full_per_head=3)
    boundary = 100 * 1 / 3
    report = sweep_thresholds(source, baseline_threshold_pct=boundary)
    baseline = next(row for row in report["selected_reports"] if row["objective"] == "baseline")
    assert baseline["heads"][0]["coverage_pct"] == boundary
    assert baseline["heads"][0]["head_class"] == "dense"
    matching = [row for row in report["regions"] if contains(row, boundary)]
    assert len(matching) == 1
    assert matching[0]["dense_head_count"] == 1
    samples = {row["threshold_pct"]: row for row in report["samples"]}
    assert samples[boundary]["dense_head_count"] == 1
    assert samples[34]["sparse_head_count"] == 1


def test_minimum_exposed_time_and_maximum_efficiency_can_choose_different_regions():
    source = conflicting_objectives_source()
    original = deepcopy(source)
    report = sweep_thresholds(source)
    assert source == original
    regions = report["regions"]
    assert [(row["lower_pct"], row["upper_pct"]) for row in regions] == [(0, 0), (0, 20), (20, 100)]
    assert [row["fetch_ms"] for row in regions] == pytest.approx([4.0, 2.0, 0.4])
    assert [row["hidden_ms"] for row in regions] == pytest.approx([1.5, 1.5, 0.1])
    assert [row["unhidden_ms"] for row in regions] == pytest.approx([2.5, 0.5, 0.3])
    assert [row["overlap_efficiency"] for row in regions] == pytest.approx([0.375, 0.75, 0.25])
    assert report["best_unhidden_region_ids"] == [regions[2]["region_id"]]
    assert report["best_efficiency_region_ids"] == [regions[1]["region_id"]]
    assert [row["compute_ms"] for row in regions] == pytest.approx([2.05] * 3)
    assert [row["compute_plus_unhidden_ms"] for row in regions] == pytest.approx([4.55, 2.55, 2.35])
    selected = {row["objective"]: row for row in report["selected_reports"]}
    assert set(selected) == {"baseline", "min_unhidden", "max_efficiency"}
    assert selected["baseline"]["parameters"]["threshold_pct"] == 30.0
    assert selected["min_unhidden"]["totals"]["unhidden_ms"] == pytest.approx(0.3)
    assert selected["max_efficiency"]["totals"]["overlap_efficiency"] == pytest.approx(0.75)
    for row in selected.values():
        assert set(row) >= {"objective", "parameters", "heads", "layers", "totals"}
    assert len(report["samples"]) == 101
    assert {row["threshold_pct"] for row in report["samples"]} == set(range(101))


def test_all_tied_optimum_regions_have_their_own_selected_report():
    source = source_report(
        [[0, 0], [20_000_000, 50_000_000]], attention=[0.1, 1.0], other=[4.0, 0.1]
    )
    report = sweep_thresholds(source)
    expected = {
        row["region_id"]
        for row in report["regions"]
        if row["lower_pct"] >= 0 and not row["lower_inclusive"]
    }
    assert len(expected) == 3
    assert set(report["best_unhidden_region_ids"]) == expected
    assert set(report["best_efficiency_region_ids"]) == expected
    assert Counter(row["objective"] for row in report["selected_reports"]) == {
        "baseline": 1,
        "min_unhidden": 3,
        "max_efficiency": 3,
    }
    for objective in ("min_unhidden", "max_efficiency"):
        selected = [row for row in report["selected_reports"] if row["objective"] == objective]
        assert {row["region_id"] for row in selected} == expected
        for row in selected:
            assert row["totals"]["unhidden_ms"] == 0
            assert row["totals"]["overlap_efficiency"] == 1


def test_bandwidth_mfu_and_fractional_baseline_are_forwarded_to_all_evaluations():
    source = conflicting_objectives_source()
    source["parameters"].update(bandwidth_gbps=50.0, bytes_per_second=50e9)
    report = sweep_thresholds(
        source,
        bandwidth_gbps=25.0,
        attention_mfu_scale=0.25,
        baseline_threshold_pct=30.5,
    )
    assert report["parameters"]["bandwidth_gbps"] == 25.0
    assert report["parameters"]["bytes_per_second"] == 25e9
    assert source["parameters"]["bytes_per_second"] == 50e9
    assert report["parameters"]["attention_mfu_scale"] == 0.25
    assert report["parameters"]["baseline_threshold_pct"] == 30.5
    for selected in report["selected_reports"]:
        assert selected["parameters"]["bandwidth_gbps"] == 25.0
        assert selected["parameters"]["bytes_per_second"] == 25e9
        assert selected["parameters"]["attention_mfu_scale"] == 0.25
    baseline = next(row for row in report["selected_reports"] if row["objective"] == "baseline")
    assert baseline["layers"][1]["attention_window_ms"] == pytest.approx(0.2)
    assert baseline["layers"][1]["sparse_fetch_ms"] == pytest.approx(0.8)
    assert baseline["totals"]["unhidden_ms"] == pytest.approx(0.6)
    samples = {row["threshold_pct"]: row for row in report["samples"]}
    assert len(samples) == 102
    assert set(samples) == set(range(101)) | {30.5}
    assert samples[30.5]["unhidden_ms"] == pytest.approx(0.6)
    assert next(row for row in report["regions"] if contains(row, 30.5))[
        "unhidden_ms"
    ] == pytest.approx(0.6)


def test_zero_fetch_efficiency_is_none_and_is_not_a_numeric_optimum():
    report = sweep_thresholds(source_report([[0]], attention=[0.1], other=[0.2]))
    cold, no_fetch = report["regions"]
    assert cold["fetch_ms"] == pytest.approx(2.0)
    assert cold["overlap_efficiency"] == 0
    assert no_fetch["fetch_ms"] == no_fetch["hidden_ms"] == no_fetch["unhidden_ms"] == 0
    assert no_fetch["overlap_efficiency"] is None
    assert report["best_unhidden_region_ids"] == [no_fetch["region_id"]]
    assert report["best_efficiency_region_ids"] == [cold["region_id"]]
    for row in report["samples"]:
        assert row["overlap_efficiency"] == (0 if row["threshold_pct"] == 0 else None)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("coverage", [-0.1, 100.1, math.nan, math.inf, -math.inf])
def test_invalid_coverage_percentages_are_rejected(coverage):
    with pytest.raises(ValueError):
        threshold_regions([20, coverage])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_threshold_pct": -0.1},
        {"baseline_threshold_pct": 100.1},
        {"baseline_threshold_pct": math.nan},
        {"baseline_threshold_pct": math.inf},
        {"bandwidth_gbps": 0},
        {"bandwidth_gbps": math.nan},
        {"bandwidth_gbps": math.inf},
        {"attention_mfu_scale": 0},
        {"attention_mfu_scale": -1},
        {"attention_mfu_scale": math.nan},
        {"attention_mfu_scale": math.inf},
    ],
)
def test_invalid_sweep_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        sweep_thresholds(conflicting_objectives_source(), **kwargs)
