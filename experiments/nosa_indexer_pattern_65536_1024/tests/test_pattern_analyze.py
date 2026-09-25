"""Set-based checks for independent layer/head block unions and byte accounting."""

import json

import numpy as np
import pytest

from experiments.nosa_indexer_pattern_65536_1024.src import analyze


def selection_fixture():
    # Two prefix blocks, one candidate block; four queries at positions 8..11.
    ids = np.full((2, 4, 2, 2), -1, dtype=np.int32)
    selections = (
        (((0, 2), (0,)), ((0,), (1,)), ((2,), (0,)), ((0, 2), (1,))),
        (((1,), (1, 2)), ((1,), (1,)), ((1,), (2,)), ((1,), (1, 2))),
    )
    for layer, queries in enumerate(selections):
        for query, heads in enumerate(queries):
            for head, blocks in enumerate(heads):
                ids[layer, query, head, : len(blocks)] = blocks
    return ids, ids >= 0


def summarize(ids, mask, **kwargs):
    params = {
        "prefix_tokens": 8,
        "total_tokens": 12,
        "head_dim": 3,
        "element_size": 2,
        "block_size": 4,
    }
    params.update(kwargs)
    return analyze.summarize(ids, mask, **params)


def test_hand_sets_keep_heads_and_layers_independent_and_partition_bytes():
    ids, mask = selection_fixture()
    unions, report = summarize(ids, mask)
    assert unions.dtype == np.bool_
    np.testing.assert_array_equal(
        unions,
        [[[True, False, True], [True, True, False]], [[False, True, False], [False, True, True]]],
    )
    # 4 tokens * 3 dimensions * 2 (K,V) * 2 bytes = 48 bytes per block/head.
    assert report["parameters"]["block_bytes"] == 48
    rows = report["head_stats"]
    assert [row["union_blocks"] for row in rows] == [2, 2, 1, 2]
    assert [row["union_bytes"] for row in rows] == [96, 96, 48, 96]
    assert [row["kv_cache_bytes"] for row in rows] == [144] * 4
    assert [row["prefix_union_blocks"] for row in rows] == [1, 2, 1, 1]
    assert [row["candidate_union_blocks"] for row in rows] == [1, 0, 0, 1]
    assert [row["fraction"] for row in rows] == pytest.approx([2 / 3, 2 / 3, 1 / 3, 2 / 3])
    assert [row["prefix_fraction"] for row in rows] == pytest.approx([0.5, 1, 0.5, 0.5])
    assert [row["candidate_fraction"] for row in rows] == [1, 0, 0, 1]
    assert [row["union_bytes"] for row in report["layer_stats"]] == [192, 144]
    assert [row["kv_cache_bytes"] for row in report["layer_stats"]] == [288, 288]
    assert report["summary"]["union_blocks"] == 7
    assert report["summary"]["union_bytes"] == 336
    assert report["summary"]["kv_cache_bytes"] == 576
    assert report["summary"]["fraction"] == pytest.approx(7 / 12)
    assert report["summary"]["prefix_union_bytes"] == 240
    assert report["summary"]["candidate_union_bytes"] == 96
    for row in [*rows, *report["layer_stats"], report["summary"]]:
        assert row["union_bytes"] == row["prefix_union_bytes"] + row["candidate_union_bytes"]
    json.dumps(report, allow_nan=False)


def test_invalid_padding_is_ignored_and_input_is_not_modified():
    ids, mask = selection_fixture()
    expected, _ = summarize(ids, mask)
    ids[~mask] = 1000000
    original = ids.copy()
    actual, _ = summarize(ids, mask)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(ids, original)


def test_duplicates_within_one_selection_are_rejected():
    ids, mask = selection_fixture()
    ids[0, 0, 0] = [0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        summarize(ids, mask)


@pytest.mark.parametrize("bad_id", [-1, 3])
def test_valid_ids_outside_cache_are_rejected(bad_id):
    ids, mask = selection_fixture()
    ids[0, 0, 0, 0] = bad_id
    with pytest.raises(ValueError, match="range"):
        summarize(ids, mask)


def test_future_block_is_rejected_but_current_partial_block_is_allowed():
    ids = np.full((1, 8, 1, 1), 2, dtype=np.int32)
    mask = np.ones_like(ids, dtype=np.bool_)
    # The first candidate at position 8 can select its block 2 despite token causal masking.
    summarize(ids, mask, total_tokens=16)
    ids[0, 0, 0, 0] = 3
    with pytest.raises(ValueError, match="causal"):
        summarize(ids, mask, total_tokens=16)


@pytest.mark.parametrize(
    "transform,error",
    [
        (lambda ids, mask: (ids[0], mask[0]), "shape"),
        (lambda ids, mask: (ids[:, :0], mask[:, :0]), "nonempty"),
        (lambda ids, mask: (ids.astype(float), mask), "integer dtype"),
        (lambda ids, mask: (ids, mask.astype(np.uint8)), "boolean"),
        (lambda ids, mask: (ids, mask[..., :1]), "same shape"),
        (lambda ids, mask: (ids[:, :3], mask[:, :3]), "query dimension"),
    ],
)
def test_bad_shapes_and_dtypes_are_rejected(transform, error):
    ids, mask = transform(*selection_fixture())
    with pytest.raises(ValueError, match=error):
        summarize(ids, mask)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"prefix_tokens": 9}, "block aligned"),
        ({"total_tokens": 13}, "block aligned"),
        ({"prefix_tokens": 12}, "exceed"),
        ({"head_dim": 0}, "head_dim"),
        ({"element_size": True}, "element_size"),
        ({"block_size": 0}, "block_size"),
    ],
)
def test_bad_parameters_are_rejected(kwargs, error):
    with pytest.raises(ValueError, match=error):
        summarize(*selection_fixture(), **kwargs)


def test_empty_selection_and_zero_prefix_have_finite_defined_metrics():
    ids = np.full((1, 4, 1, 2), -1, dtype=np.int32)
    unions, report = summarize(ids, ids >= 0, prefix_tokens=0, total_tokens=4)
    assert not unions.any()
    row = report["summary"]
    assert row["union_bytes"] == 0
    assert row["fraction"] == 0
    assert row["prefix_fraction"] is None
    assert row["candidate_fraction"] == 0
    json.dumps(report, allow_nan=False)


def test_full_nosa_cache_byte_denominator():
    # All candidate queries select one historical block, independently for both heads.
    ids = np.zeros((1, 1024, 2, 1), dtype=np.int32)
    _, report = analyze.summarize(
        ids,
        np.ones_like(ids, dtype=np.bool_),
        prefix_tokens=65536,
        total_tokens=66560,
        head_dim=128,
        element_size=2,
    )
    assert report["parameters"]["block_bytes"] == 32768
    assert report["parameters"]["blocks_per_head"] == 1040
    assert report["head_stats"][0]["kv_cache_bytes"] == 32.5 * 2**20
    assert report["summary"]["kv_cache_bytes"] == 65 * 2**20
    assert report["summary"]["union_bytes"] == 65536
    assert report["summary"]["fraction"] == pytest.approx(1 / 1040)


def test_analyze_writes_tables_and_portable_union_artifact(tmp_path, monkeypatch):
    ids, mask = selection_fixture()
    np.save(tmp_path / "block_ids.npy", ids)
    np.save(tmp_path / "valid_mask.npy", mask)
    metadata = {
        "execution": {"prefix_tokens": 8, "total_tokens": 12},
        "model_config": {"head_dim": 3},
        "element_size": 2,
        "block_size": 4,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    plotted = []
    monkeypatch.setattr(analyze, "_plot", lambda path, unions, report: plotted.append(unions))
    report = analyze.analyze(tmp_path)
    assert len(plotted) == 1
    np.testing.assert_array_equal(np.load(tmp_path / "union_mask.npy"), plotted[0])
    assert json.loads((tmp_path / "summary.json").read_text()) == report
    assert len((tmp_path / "head_stats.csv").read_text().splitlines()) == 5
    assert len((tmp_path / "layer_stats.csv").read_text().splitlines()) == 3
    assert "different heads and layers" in (tmp_path / "report.md").read_text()
