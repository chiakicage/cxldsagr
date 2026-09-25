"""Hand-calculated checks for shared-link layer overlap window estimates."""

from copy import deepcopy

import pytest

from experiments.nosa_indexer_pattern_65536_1024.src.overlap import estimate_overlap


def source_report(unions, *, attention, other, selection_slots=64, full_per_head=100_000_000):
    """Build source estimates; a full head costs 2 ms at the default 50 GB/s."""
    layers = len(unions)
    heads = len(unions[0])
    assert len(attention) == len(other) == layers
    assert all(len(row) == heads for row in unions)
    return {
        "parameters": {
            "num_layers": layers,
            "num_kv_heads": heads,
            "selection_slots": selection_slots,
        },
        "layers": [
            {
                "layer": layer,
                "sparse_attention_gpu_ms": attention[layer],
                "other_gpu_ms": other[layer],
                "sparse_union_bytes": sum(unions[layer]),
                "full_kv_bytes": full_per_head * heads,
            }
            for layer in range(layers)
        ],
        "heads": [
            {
                "layer": layer,
                "kv_head": head,
                "sparse_union_bytes": unions[layer][head],
                "full_kv_bytes": full_per_head,
            }
            for layer in range(layers)
            for head in range(heads)
        ],
    }


@pytest.mark.parametrize("selection_slots", [32, 64])
def test_threshold_is_sparse_below_30_and_full_fetch_at_exactly_30(selection_slots):
    source = source_report(
        [[29_000_000, 30_000_000]],
        attention=[0.25],
        other=[10.0],
        selection_slots=selection_slots,
    )
    original = deepcopy(source)
    report = estimate_overlap(source)
    row = report["layers"][0]
    assert row["sparse_head_count"] == row["dense_head_count"] == 1
    assert row["sparse_fetch_ms"] == pytest.approx(0.58)
    assert row["dense_fetch_ms"] == pytest.approx(2.0)
    # Fetch classification does not replace the source's sparse attention work
    # with a larger dense calculation or remove the full-fetch head's work.
    assert row["attention_window_ms"] == pytest.approx(0.25)
    assert row["sparse_hidden_ms"] == pytest.approx(0.25)
    assert row["dense_hidden_ms"] == 0
    assert row["fetch_ms"] == pytest.approx(2.58)
    assert row["unhidden_ms"] == pytest.approx(2.33)
    assert source == original


def test_multiple_heads_share_each_layer_attention_and_prefetch_window_once():
    report = estimate_overlap(
        source_report(
            [[29_000_000, 29_000_000], [40_000_000, 40_000_000]],
            attention=[0.75, 0.25],
            other=[3.0, 20.0],
        )
    )
    sparse, dense = report["layers"]
    assert sparse["sparse_head_count"] == 2
    assert sparse["dense_head_count"] == 0
    assert sparse["sparse_fetch_ms"] == pytest.approx(1.16)
    # Per-head min(0.58, 0.75) would incorrectly hide all 1.16 ms.
    assert sparse["sparse_hidden_ms"] == pytest.approx(0.75)
    assert dense["sparse_head_count"] == 0
    assert dense["dense_head_count"] == 2
    assert dense["dense_fetch_ms"] == pytest.approx(4.0)
    assert dense["prefetch_window_ms"] == pytest.approx(3.0)
    # Per-head min(2, 3) would incorrectly reuse the previous layer's window.
    assert dense["dense_hidden_ms"] == pytest.approx(3.0)
    assert report["totals"]["hidden_ms"] == pytest.approx(3.75)
    assert report["totals"]["fetch_ms"] == pytest.approx(5.16)


def test_sparse_attention_capacity_cannot_be_pooled_across_layers():
    report = estimate_overlap(
        source_report([[20_000_000], [20_000_000]], attention=[0.0, 2.0], other=[0.0, 0.0])
    )
    assert [row["sparse_hidden_ms"] for row in report["layers"]] == pytest.approx([0, 0.4])
    assert report["totals"]["hidden_ms"] == pytest.approx(0.4)
    assert report["totals"]["unhidden_ms"] == pytest.approx(0.4)
    assert report["totals"]["overlap_efficiency"] == pytest.approx(0.5)
    # min(sum(fetch)=0.8, sum(attention)=2) would incorrectly hide everything.
    assert report["totals"]["hidden_ms"] < min(0.8, 2.0)


def test_dense_fetch_uses_only_previous_layer_other_time_in_forward_order():
    report = estimate_overlap(
        source_report(
            [[30_000_000], [30_000_000], [30_000_000]],
            attention=[9.0, 9.0, 9.0],
            other=[0.25, 0.75, 100.0],
        )
    )
    assert [row["layer"] for row in report["layers"]] == [0, 1, 2]
    assert [row["prefetch_window_ms"] for row in report["layers"]] == pytest.approx([0, 0.25, 0.75])
    assert [row["dense_hidden_ms"] for row in report["layers"]] == pytest.approx([0, 0.25, 0.75])
    assert report["totals"]["dense_fetch_ms"] == pytest.approx(6.0)
    assert report["totals"]["dense_hidden_ms"] == pytest.approx(1.0)
    # The final layer's 100 ms cannot help a current or already executed layer.
    assert sum(row["prefetch_window_ms"] for row in report["layers"]) == pytest.approx(1.0)
    assert [row["prefetch_source_layer"] for row in report["layers"]] == [None, 0, 1]


def test_dense_prefetch_capacity_cannot_be_pooled_between_different_target_layers():
    report = estimate_overlap(
        source_report(
            [[0], [30_000_000], [30_000_000]],
            attention=[0.0, 0.0, 0.0],
            other=[0.0, 4.0, 0.0],
        )
    )
    assert [row["dense_hidden_ms"] for row in report["layers"]] == pytest.approx([0, 0, 2.0])
    assert report["totals"]["dense_fetch_ms"] == pytest.approx(4.0)
    assert report["totals"]["dense_hidden_ms"] == pytest.approx(2.0)
    assert report["totals"]["unhidden_ms"] == pytest.approx(2.0)


def test_first_layer_dense_fetch_is_fully_exposed_even_with_large_compute_windows():
    report = estimate_overlap(
        source_report([[30_000_000, 100_000_000]], attention=[100.0], other=[100.0])
    )
    row = report["layers"][0]
    assert row["prefetch_window_ms"] == 0
    assert row["dense_fetch_ms"] == pytest.approx(4.0)
    assert row["hidden_ms"] == 0
    assert row["unhidden_ms"] == pytest.approx(4.0)
    assert row["overlap_efficiency"] == 0


def test_totals_sum_times_and_counts_and_use_fetch_weighted_efficiency():
    report = estimate_overlap(
        source_report(
            [[29_000_000, 30_000_000], [0, 30_000_000]],
            attention=[0.25, 0.5],
            other=[1.0, 10.0],
        )
    )
    rows, totals = report["layers"], report["totals"]
    fields = (
        "sparse_head_count",
        "dense_head_count",
        "sparse_fetch_bytes",
        "dense_fetch_bytes",
        "sparse_fetch_ms",
        "dense_fetch_ms",
        "base_attention_ms",
        "other_gpu_ms",
        "attention_window_ms",
        "compute_ms",
        "compute_plus_unhidden_ms",
        "sparse_hidden_ms",
        "dense_hidden_ms",
        "sparse_unhidden_ms",
        "dense_unhidden_ms",
        "hidden_ms",
        "unhidden_ms",
        "fetch_ms",
    )
    for field in fields:
        assert totals[field] == pytest.approx(sum(row[field] for row in rows))
    for row in [*rows, totals]:
        assert row["fetch_ms"] == pytest.approx(row["sparse_fetch_ms"] + row["dense_fetch_ms"])
        assert row["hidden_ms"] == pytest.approx(row["sparse_hidden_ms"] + row["dense_hidden_ms"])
        assert row["fetch_ms"] == pytest.approx(row["hidden_ms"] + row["unhidden_ms"])
        assert row["overlap_efficiency"] == pytest.approx(row["hidden_ms"] / row["fetch_ms"])
        assert row["compute_ms"] == pytest.approx(row["attention_window_ms"] + row["other_gpu_ms"])
        assert row["compute_plus_unhidden_ms"] == pytest.approx(
            row["compute_ms"] + row["unhidden_ms"]
        )
    assert totals["hidden_ms"] == pytest.approx(1.25)
    assert totals["fetch_ms"] == pytest.approx(4.58)
    assert totals["overlap_efficiency"] == pytest.approx(1.25 / 4.58)
    assert totals["sparse_overlap_efficiency"] == pytest.approx(0.25 / 0.58)
    assert totals["dense_overlap_efficiency"] == pytest.approx(1.0 / 4.0)
    assert totals["overlap_efficiency"] != pytest.approx(
        sum(row["overlap_efficiency"] for row in rows) / len(rows)
    )


def test_bandwidth_is_decimal_gbps_and_does_not_change_compute_windows():
    report = estimate_overlap(
        source_report([[20_000_000]], attention=[0.3], other=[4.0]), bandwidth_gbps=25.0
    )
    row = report["layers"][0]
    assert row["sparse_fetch_ms"] == pytest.approx(0.8)
    assert row["attention_window_ms"] == pytest.approx(0.3)
    assert row["sparse_hidden_ms"] == pytest.approx(0.3)
    assert row["unhidden_ms"] == pytest.approx(0.5)


def test_zero_fetch_has_no_defined_overlap_efficiency():
    report = estimate_overlap(
        source_report([[0, 0], [0, 0]], attention=[2.0, 3.0], other=[4.0, 5.0])
    )
    for row in [*report["layers"], report["totals"]]:
        assert row["fetch_ms"] == 0
        assert row["hidden_ms"] == 0
        assert row["unhidden_ms"] == 0
        assert row["overlap_efficiency"] is None
    assert report["totals"]["sparse_overlap_efficiency"] is None
    assert report["totals"]["dense_overlap_efficiency"] is None


def test_default_attention_mfu_scale_equals_explicit_one_and_preserves_base_windows():
    source = source_report(
        [[20_000_000, 30_000_000], [20_000_000, 30_000_000]],
        attention=[0.1, 0.3],
        other=[0.75, 1.0],
    )
    default = estimate_overlap(source)
    assert default == estimate_overlap(source, attention_mfu_scale=1.0)
    assert default["parameters"]["attention_mfu_scale"] == 1.0
    assert [row["attention_window_ms"] for row in default["layers"]] == pytest.approx([0.1, 0.3])
    assert default["totals"]["hidden_ms"] == pytest.approx(1.15)
    assert default["totals"]["compute_plus_unhidden_ms"] == pytest.approx(5.8)


def test_halving_attention_mfu_doubles_only_attention_and_hidden_time_remains_capped():
    source = source_report(
        [[20_000_000, 30_000_000], [20_000_000, 30_000_000]],
        attention=[0.1, 0.3],
        other=[0.75, 1.0],
    )
    original = deepcopy(source)
    base = estimate_overlap(source)
    scaled = estimate_overlap(source, attention_mfu_scale=0.5)
    assert scaled["parameters"]["attention_mfu_scale"] == 0.5
    assert scaled["heads"] == base["heads"]
    assert source == original
    unchanged = (
        "sparse_head_count",
        "dense_head_count",
        "sparse_fetch_bytes",
        "dense_fetch_bytes",
        "sparse_fetch_ms",
        "dense_fetch_ms",
        "fetch_ms",
        "prefetch_source_layer",
        "prefetch_window_ms",
        "dense_hidden_ms",
        "other_gpu_ms",
        "base_attention_ms",
    )
    for base_row, scaled_row in zip(base["layers"], scaled["layers"], strict=True):
        for field in unchanged:
            assert scaled_row[field] == base_row[field]
        assert scaled_row["attention_window_ms"] == pytest.approx(
            2 * base_row["attention_window_ms"]
        )
    rows = scaled["layers"]
    assert [row["base_attention_ms"] for row in rows] == pytest.approx([0.1, 0.3])
    assert [row["attention_window_ms"] for row in rows] == pytest.approx([0.2, 0.6])
    assert [row["sparse_hidden_ms"] for row in rows] == pytest.approx([0.2, 0.4])
    assert [row["dense_hidden_ms"] for row in rows] == pytest.approx([0, 0.75])
    assert [row["compute_ms"] for row in rows] == pytest.approx([0.95, 1.6])
    assert [row["compute_plus_unhidden_ms"] for row in rows] == pytest.approx([3.15, 2.85])
    # A larger hidden fraction does not imply a faster execution: the work
    # creating the additional attention window itself takes longer.
    assert scaled["totals"]["hidden_ms"] == pytest.approx(1.35)
    assert scaled["totals"]["compute_plus_unhidden_ms"] == pytest.approx(6.0)
    assert scaled["totals"]["compute_plus_unhidden_ms"] > base["totals"]["compute_plus_unhidden_ms"]


@pytest.mark.parametrize("scale", [0, -0.5, float("nan"), float("inf"), -float("inf")])
def test_attention_mfu_scale_must_be_positive_and_finite(scale):
    with pytest.raises(ValueError):
        estimate_overlap(
            source_report([[20_000_000]], attention=[0.1], other=[0.2]),
            attention_mfu_scale=scale,
        )


@pytest.mark.parametrize(("attention", "scaled_total"), [(0.1, 0.6), (0.3, 0.8)])
def test_lower_attention_mfu_cannot_reduce_compute_plus_unhidden(attention, scaled_total):
    # Sparse fetch is 0.4 ms. If A < fetch, increasing A just replaces exposed
    # transfer time; once A reaches fetch, additional compute time is exposed.
    source = source_report([[20_000_000]], attention=[attention], other=[0.2])
    base = estimate_overlap(source)
    scaled = estimate_overlap(source, attention_mfu_scale=0.5)
    assert base["totals"]["compute_plus_unhidden_ms"] == pytest.approx(0.6)
    assert scaled["totals"]["compute_plus_unhidden_ms"] == pytest.approx(scaled_total)
    assert (
        scaled["totals"]["compute_plus_unhidden_ms"]
        >= base["totals"]["compute_plus_unhidden_ms"] - 1e-12
    )
