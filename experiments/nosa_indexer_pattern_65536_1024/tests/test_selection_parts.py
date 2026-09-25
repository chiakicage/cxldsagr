"""Set-based fixed/query-aware decomposition and decimal-bandwidth checks."""

import json

import numpy as np
import pytest

from experiments.nosa_indexer_pattern_65536_1024.src.analyze import summarize
from experiments.nosa_indexer_pattern_65536_1024.src.selection_parts import decompose_selections


def selections_with_cross_query_overlap():
    # Queries 4 and 5, local width 2. Block 3 is local at query 4 and QA at
    # query 5. Only head 0 also selects block 1, which is outside fixed's union.
    ids = np.array(
        [[[[4, 1, 0, 3, -1], [0, 3, 4, -1, -1]], [[3, 0, 5, 4, -1], [0, 3, 4, 5, -1]]]],
        dtype=np.int32,
    )
    ids = np.repeat(ids, 2, axis=0)
    return ids, ids >= 0


def decompose(ids, mask, **kwargs):
    params = {
        "prefix_tokens": 4,
        "total_tokens": 6,
        "head_dim": 3,
        "element_size": 2,
        "block_size": 1,
        "sink_blocks": 1,
        "local_blocks": 2,
        "bandwidth_gbps": 50.0,
    }
    params.update(kwargs)
    return decompose_selections(ids, mask, **params)


def selected_ids(union, layer=0, head=0):
    return set(np.flatnonzero(union[layer, head]).tolist())


def test_cross_query_overlap_requires_additional_qa_for_an_additive_partition():
    unions, report = decompose(*selections_with_cross_query_overlap())
    assert selected_ids(unions["sink"]) == {0}
    assert selected_ids(unions["local"]) == {3, 4, 5}
    assert selected_ids(unions["fixed"]) == {0, 3, 4, 5}
    assert selected_ids(unions["query_aware"]) == {1, 3}
    assert selected_ids(unions["overlap"]) == {3}
    assert selected_ids(unions["query_aware_additional"]) == {1}
    assert selected_ids(unions["combined"]) == {0, 1, 3, 4, 5}
    assert selected_ids(unions["query_aware"], head=1) == {3}
    assert selected_ids(unions["query_aware_additional"], head=1) == set()
    for name, union in unions.items():
        assert union.shape == (2, 2, 6), name
        assert union.dtype == np.bool_
    np.testing.assert_array_equal(unions["combined"], unions["fixed"] | unions["query_aware"])
    assert not (unions["fixed"] & unions["query_aware_additional"]).any()
    for row in [*report["head_stats"], *report["layer_stats"], report["summary"]]:
        for field in ("union_blocks", "union_bytes", "prefix_union_bytes", "candidate_union_bytes"):
            assert (
                row["combined"][field] == row["fixed"][field] + row["query_aware_additional"][field]
            )
            assert row["combined"][field] == (
                row["fixed"][field] + row["query_aware"][field] - row["overlap"][field]
            )
        assert (
            row["fixed"]["union_bytes"] + row["query_aware"]["union_bytes"]
            > row["combined"]["union_bytes"]
        )


def test_query_aware_and_fixed_are_disjoint_within_each_query():
    ids, mask = selections_with_cross_query_overlap()
    for query in range(2):
        unions, _ = decompose(
            ids[:, query : query + 1],
            mask[:, query : query + 1],
            prefix_tokens=4 + query,
            total_tokens=5 + query,
        )
        assert not unions["overlap"].any()
        np.testing.assert_array_equal(unions["query_aware"], unions["query_aware_additional"])


def test_heads_layers_and_prefix_candidate_bytes_are_counted_independently():
    _, report = decompose(*selections_with_cross_query_overlap())
    # A block/head costs 1 token * 3 dimensions * K,V * 2 B = 12 B.
    assert report["parameters"]["block_bytes"] == 12
    assert [(row["layer"], row["kv_head"]) for row in report["head_stats"]] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [row["combined"]["union_bytes"] for row in report["head_stats"]] == [60, 48, 60, 48]
    assert [row["combined"]["union_bytes"] for row in report["layer_stats"]] == [108, 108]
    summary = report["summary"]
    assert summary["combined"]["union_blocks"] == 18
    assert summary["combined"]["union_bytes"] == 216
    assert summary["combined"]["kv_cache_bytes"] == 288
    assert summary["combined"]["fraction"] == pytest.approx(0.75)
    assert summary["combined"]["prefix_union_bytes"] == 120
    assert summary["combined"]["candidate_union_bytes"] == 96
    assert summary["query_aware"]["candidate_union_bytes"] == 0
    assert summary["fixed"]["prefix_union_bytes"] == 96
    assert summary["fixed"]["candidate_union_bytes"] == 96
    json.dumps(report, allow_nan=False)


def test_combined_stats_match_existing_summary_and_fetch_uses_decimal_gbps():
    ids, mask = selections_with_cross_query_overlap()
    unions, report = decompose(ids, mask, bandwidth_gbps=2.5)
    expected_union, expected = summarize(
        ids, mask, prefix_tokens=4, total_tokens=6, head_dim=3, element_size=2, block_size=1
    )
    np.testing.assert_array_equal(unions["combined"], expected_union)
    for key, value in expected["summary"].items():
        assert report["summary"]["combined"][key] == value
    assert report["parameters"]["bandwidth_bytes_per_second"] == 2_500_000_000
    assert report["parameters"]["bandwidth_gbps"] == 2.5
    for row in [*report["head_stats"], *report["layer_stats"], report["summary"]]:
        for component in unions:
            stats = row[component]
            assert stats["fetch_ms"] == pytest.approx(stats["union_bytes"] / 2_500_000)
            assert stats["prefix_fetch_ms"] == pytest.approx(
                stats["prefix_union_bytes"] / 2_500_000
            )
            assert stats["candidate_fetch_ms"] == pytest.approx(
                stats["candidate_union_bytes"] / 2_500_000
            )
            assert stats["fetch_ms"] == pytest.approx(
                stats["prefix_fetch_ms"] + stats["candidate_fetch_ms"]
            )


def test_invalid_padding_is_ignored_and_inputs_are_not_modified():
    ids, mask = selections_with_cross_query_overlap()
    expected, _ = decompose(ids, mask)
    ids[~mask] = 1000000
    saved_ids, saved_mask = ids.copy(), mask.copy()
    actual, _ = decompose(ids, mask)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    np.testing.assert_array_equal(ids, saved_ids)
    np.testing.assert_array_equal(mask, saved_mask)


def test_short_prefix_allows_overlapping_sink_and_local_without_double_counting():
    ids = np.full((1, 8, 1, 4), -1, dtype=np.int32)
    for query in range(8):
        ids[0, query, 0, : query // 4 + 1] = np.arange(query // 4 + 1)
    unions, report = decompose(
        ids, ids >= 0, prefix_tokens=0, total_tokens=8, block_size=4, local_blocks=16
    )
    assert selected_ids(unions["sink"]) == {0}
    assert selected_ids(unions["local"]) == {0, 1}
    assert selected_ids(unions["fixed"]) == {0, 1}
    assert not unions["query_aware"].any()
    assert report["summary"]["fixed"]["union_blocks"] == 2
    assert report["summary"]["fixed"]["prefix_fraction"] is None
    assert report["summary"]["fixed"]["candidate_fraction"] == 1
    json.dumps(report, allow_nan=False)


def test_formal_64k_plus_1k_fixed_union_has_32_blocks_per_head():
    query_blocks = (65536 + np.arange(1024)) // 64
    fixed_ids = np.concatenate(
        [np.zeros((1024, 1), dtype=np.int32), query_blocks[:, None] + np.arange(-15, 1)], axis=1
    )
    ids = np.broadcast_to(fixed_ids[None, :, None, :], (2, 1024, 2, 17))
    unions, report = decompose_selections(
        ids,
        np.ones_like(ids, dtype=np.bool_),
        prefix_tokens=65536,
        total_tokens=66560,
        head_dim=128,
        element_size=2,
    )
    assert selected_ids(unions["local"]) == set(range(1009, 1040))
    assert selected_ids(unions["fixed"]) == {0, *range(1009, 1040)}
    assert not unions["query_aware"].any()
    for row in report["head_stats"]:
        fixed = row["fixed"]
        assert fixed["union_blocks"] == 32
        assert fixed["prefix_union_blocks"] == fixed["candidate_union_blocks"] == 16
        assert fixed["union_bytes"] == 2**20
        assert fixed["fetch_ms"] == pytest.approx(0.02097152)
    for row in report["layer_stats"]:
        assert row["fixed"]["union_bytes"] == 2 * 2**20
        assert row["fixed"]["fraction"] == pytest.approx(2 / 65)
        assert row["fixed"]["fetch_ms"] == pytest.approx(0.04194304)


@pytest.mark.parametrize("missing_block", [0, 3, 4])
def test_missing_required_fixed_blocks_are_rejected(missing_block):
    ids, mask = selections_with_cross_query_overlap()
    mask[0, 0, 0] &= ids[0, 0, 0] != missing_block
    with pytest.raises(ValueError, match="sink/local"):
        decompose(ids, mask)


@pytest.mark.parametrize(("bad_id", "message"), [(4, "duplicate"), (5, "causal"), (6, "range")])
def test_existing_selection_validation_is_preserved(bad_id, message):
    ids, mask = selections_with_cross_query_overlap()
    ids[0, 0, 0, 1] = bad_id
    with pytest.raises(ValueError, match=message):
        decompose(ids, mask)


@pytest.mark.parametrize("bandwidth", [0, -1, np.nan, np.inf, True, "50"])
def test_bandwidth_must_be_positive_finite_and_numeric(bandwidth):
    with pytest.raises(ValueError, match="bandwidth_gbps"):
        decompose(*selections_with_cross_query_overlap(), bandwidth_gbps=bandwidth)


@pytest.mark.parametrize(
    "kwargs", [{"sink_blocks": -1}, {"local_blocks": 1.5}, {"sink_blocks": True}]
)
def test_policy_counts_must_be_nonnegative_integers(kwargs):
    with pytest.raises(ValueError, match="blocks"):
        decompose(*selections_with_cross_query_overlap(), **kwargs)
