"""Independent set oracles for dense/sparse selection and physical KV unions."""

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from experiments.nosa_indexer_pattern_65536_1024.src import sparse_compare as compare


def _tiny_choices():
    left = np.array(
        [
            [[[0, 1, 2, 3], [0, 1, 4, 5]], [[1, 2, 4, 5], [1, 2, 3, 6]]],
            [[[0, 1, 2, 3], [4, 5, 6, 7]], [[1, 3, 4, 6], [0, 2, 4, 6]]],
        ],
        dtype=np.int32,
    )
    right = np.array(
        [
            [[[3, 2, 1, 0], [0, 2, 4, 6]], [[1, 3, 4, 6], [1, 2, 3, 7]]],
            [[[4, 5, 6, 7], [7, 6, 5, 4]], [[0, 2, 5, 7], [0, 2, 3, 6]]],
        ],
        dtype=np.int32,
    )
    return left, np.ones_like(left, dtype=np.bool_), right, np.ones_like(right, dtype=np.bool_)


def _oracle_union(ids, masks, count=8):
    result = np.zeros((ids.shape[0], ids.shape[2], count), dtype=np.bool_)
    for layer in range(ids.shape[0]):
        for head in range(ids.shape[2]):
            blocks = set(ids[layer, :, head][masks[layer, :, head]].tolist())
            for block in blocks:
                result[layer, head, block] = True
    return result


def test_first_layer_equivalence_compares_sets_and_rejects_different_blocks():
    left, _, _, _ = _tiny_choices()
    right = left[..., ::-1].copy()
    right[1] = 7 - right[1]
    compare._validate_first_layer_choices(left, right)
    right[0, 0, 0, 0] = 7
    with pytest.raises(ValueError, match="identical first-layer block sets"):
        compare._validate_first_layer_choices(left, right)


def test_union_capacity_counts_layers_and_heads_separately():
    left, lm, right, rm = _tiny_choices()
    lu, ru = _oracle_union(left, lm), _oracle_union(right, rm)
    report = compare.compare_pair(lu, ru, block_bytes=32)
    physical_left = set(zip(*np.nonzero(lu), strict=True))
    physical_right = set(zip(*np.nonzero(ru), strict=True))
    row = report["summary"]
    assert row["left_union_mib"] == len(physical_left) * 32 / 2**20
    assert row["right_union_mib"] == len(physical_right) * 32 / 2**20
    assert (
        row["coverage_delta_right_minus_left"]
        == (len(physical_right) - len(physical_left)) / lu.size
    )
    assert sum(r["left_union_blocks"] for r in report["head_stats"]) == len(physical_left)
    assert sum(r["right_union_blocks"] for r in report["layer_stats"]) == len(physical_right)


@pytest.mark.parametrize("problem", ["duplicate", "range", "mask_dtype", "shape", "float"])
def test_validation_rejects_malformed_choices(problem):
    left, lm, _, _ = _tiny_choices()
    if problem == "duplicate":
        left[0, 0, 0, 1] = left[0, 0, 0, 0]
    elif problem == "range":
        left[0, 0, 0, 0] = 8
    elif problem == "mask_dtype":
        lm = lm.astype(np.uint8)
    elif problem == "shape":
        lm = lm[:, :1]
    else:
        left = left.astype(np.float32)
    with pytest.raises(ValueError):
        compare._choice_arrays(left, lm, 8)


def _mandatory_fixture():
    row = [0, *range(1, 48), *range(65, 81)]
    ids = np.broadcast_to(np.array(row, dtype=np.int32), (1, 64, 2, 64)).copy()
    return ids, np.ones_like(ids, dtype=np.bool_)


def test_mandatory_validation_distinguishes_16_and_17_local_policies():
    ids, valid = _mandatory_fixture()
    options = {"prefix_tokens": 5120, "total_tokens": 5184}
    compare.validate_choices(ids, valid, local_blocks=16, **options)
    with pytest.raises(ValueError, match="offset 16"):
        compare.validate_choices(ids, valid, local_blocks=17, **options)
    ids[..., 47] = 64
    compare.validate_choices(ids, valid, local_blocks=17, **options)


@pytest.mark.parametrize("problem", ["sink", "local", "future", "incomplete"])
def test_mandatory_validation_rejects_illegal_selections(problem):
    ids, valid = _mandatory_fixture()
    if problem == "sink":
        ids[..., 0] = 48
    elif problem == "local":
        ids[..., -1] = 48
    elif problem == "future":
        ids[..., 1] = 81
    else:
        valid[..., 1] = False
    with pytest.raises(ValueError):
        compare.validate_choices(ids, valid, prefix_tokens=5120, total_tokens=5184, local_blocks=16)


def _metadata():
    arms = {}
    for arm in compare.ARMS:
        qa = arm == "dense_qa64"
        propagation = "sparse" if arm == "sparse_nosa64" else "dense"
        arms[arm] = {
            "label": arm,
            "activation_source": "dense_full_attention_post_rope"
            if propagation == "dense"
            else "resident_block_sparse_attention_with_cis_post_rope",
            "prefix_propagation": propagation,
            "candidate_propagation": propagation,
            "selection_source": "sidecar_same_dense_activations"
            if propagation == "dense"
            else "actual_attention_input_without_reselection",
            "indexer_mode": "query_aware" if qa else "nosa",
            "indexer_backend": "reference" if qa else "triton",
            "query_stage_blocks": None if qa else 33,
            "indexer_policy": {
                "block_size": 64,
                "block_budget": 64,
                "sink_blocks": 1,
                "local_blocks": 16 if qa else 17,
                "topk_blocks": 47 if qa else 15,
                "compression_kernel_size": 32,
                "compression_stride": 16,
            },
        }
    return {
        "schema_version": 1,
        "run_id": "fixture-comparison",
        "execution": dict(compare.EXECUTION),
        "model_config": {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 43,
        },
        "dtype": "bfloat16",
        "element_size": 2,
        "block_size": 64,
        "indexer_query_chunk_size": 64,
        "request_sha256": "request",
        "input_ids_sha256": "tokens",
        "checkpoint_sha256": {"model.safetensors": "weights", "config.json": "config"},
        "source_sha256": {"old_capture.py": "captured-version"},
        "arms": arms,
    }


def _baseline_metadata(metadata):
    return metadata | {
        "run_id": "historical",
        "activation_source": "dense_full_attention_post_rope",
        "indexer_compute_dtype": "float32",
        "indexer_policy": metadata["arms"]["dense_qa64"]["indexer_policy"],
    }


@pytest.mark.parametrize(
    "field", ["request_sha256", "input_ids_sha256", "model_config", "checkpoint_sha256"]
)
def test_baseline_refuses_request_model_or_checkpoint_mismatch(field):
    metadata = _metadata()
    baseline = deepcopy(_baseline_metadata(metadata))
    compare.baseline_compatibility(metadata, baseline)
    baseline[field] = "different"
    with pytest.raises(ValueError, match=field):
        compare.baseline_compatibility(metadata, baseline)


def test_baseline_refuses_different_execution_or_policy():
    metadata = _metadata()
    baseline = deepcopy(_baseline_metadata(metadata))
    baseline["execution"]["prefix_chunk_size"] = 512
    with pytest.raises(ValueError, match="prefix_chunk_size"):
        compare.baseline_compatibility(metadata, baseline)
    baseline = deepcopy(_baseline_metadata(metadata))
    baseline["indexer_policy"]["local_blocks"] = 17
    with pytest.raises(ValueError, match="QA-only policy"):
        compare.baseline_compatibility(metadata, baseline)


@pytest.mark.parametrize(
    "problem", ["execution", "dimensions", "budget", "policy", "backend", "arms"]
)
def test_three_arm_metadata_rejects_incompatible_configuration(problem):
    metadata = _metadata()
    compare._validate_metadata(metadata)
    if problem == "execution":
        metadata["execution"]["new_tokens"] = 512
    elif problem == "dimensions":
        metadata["model_config"]["num_hidden_layers"] = 31
    elif problem == "budget":
        metadata["arms"]["dense_qa64"]["indexer_policy"]["block_budget"] = 32
    elif problem == "policy":
        metadata["arms"]["dense_nosa64"]["indexer_policy"]["local_blocks"] = 16
    elif problem == "backend":
        metadata["arms"]["sparse_nosa64"]["indexer_backend"] = "reference"
    else:
        del metadata["arms"]["sparse_nosa64"]
    with pytest.raises(ValueError):
        compare._validate_metadata(metadata)


def test_three_arm_metadata_accepts_and_checks_per_arm_batching():
    metadata = _metadata()
    del metadata["indexer_query_chunk_size"]
    metadata["indexer_query_chunk_sizes"] = {
        "dense_qa64": 64,
        "dense_nosa64": None,
        "sparse_nosa64": None,
    }
    compare._validate_metadata(metadata)
    metadata["indexer_query_chunk_sizes"]["dense_qa64"] = None
    with pytest.raises(ValueError, match="reference chunks"):
        compare._validate_metadata(metadata)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dtype",), "float16"),
        (("indexer_query_chunk_size",), 32),
        (("model_config", "num_attention_heads"), 16),
        *[
            (("arms", arm, field), "incorrect")
            for arm in compare.ARMS
            for field in (
                "prefix_propagation",
                "candidate_propagation",
                "activation_source",
                "selection_source",
            )
        ],
    ],
)
def test_metadata_rejects_wrong_precision_or_capture_semantics(path, value):
    metadata = _metadata()
    target = metadata
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        compare._validate_metadata(metadata)


def test_complete_capture_exports_metrics_and_final_analysis_source_snapshot(tmp_path):
    metadata = _metadata()
    tokens = np.arange(66560, dtype="<i8") % 43
    request = json.dumps({"input_ids": tokens.tolist()}).encode()
    metadata["request_sha256"] = hashlib.sha256(request).hexdigest()
    metadata["input_ids_sha256"] = hashlib.sha256(tokens.tobytes()).hexdigest()
    (tmp_path / "request.json").write_bytes(request)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    for arm in compare.ARMS:
        rows = []
        for query in range(1024):
            block = 1024 + query // 64
            if arm == "dense_qa64":
                row = [0, *range(1, 48), *range(block - 15, block + 1)]
            else:
                first = 2 if arm == "dense_nosa64" else 3
                row = [0, *range(first, first + 46), *range(block - 16, block + 1)]
            rows.append(row)
        ids = np.broadcast_to(
            np.array(rows, dtype=np.int32)[None, :, None, :], (32, 1024, 2, 64)
        ).copy()
        if arm == "sparse_nosa64":
            ids[0] = np.load(tmp_path / "arms" / "dense_nosa64" / "block_ids.npy")[0]
        destination = tmp_path / "arms" / arm
        destination.mkdir(parents=True)
        np.save(destination / "block_ids.npy", ids, allow_pickle=False)
        np.save(
            destination / "valid_mask.npy", np.ones(ids.shape, dtype=np.bool_), allow_pickle=False
        )
    report = compare.analyze(tmp_path, baseline_data_dir=None, plots=False)
    assert report["historical_baseline"]["status"] == "disabled"
    summary = report["pairs"]["sparse_propagation"]["summary"]
    assert summary["left_union_blocks"] == summary["right_union_blocks"] == 79 * 32 * 2
    assert summary["union_delta_mib_right_minus_left"] == 0
    for arm in compare.ARMS:
        destination = tmp_path / "arms" / arm
        assert np.load(destination / "union_mask.npy").shape == (32, 2, 1040)
        assert len((destination / "head_stats.csv").read_text().splitlines()) == 65
        assert len((destination / "layer_stats.csv").read_text().splitlines()) == 33
    assert len((tmp_path / "head_comparison.csv").read_text().splitlines()) == 129
    assert len((tmp_path / "layer_comparison.csv").read_text().splitlines()) == 65
    for name, expected in report["provenance"]["analysis_source_sha256"].items():
        assert compare._sha256(tmp_path / "analysis_sources" / name) == expected
    assert report["provenance"]["capture_source_sha256"] == metadata["source_sha256"]
    assert json.loads((tmp_path / "comparison.json").read_text()) == report
    assert "no timing" in (tmp_path / "comparison.md").read_text()
    (tmp_path / "request.json").write_bytes(request + b"\n")
    with pytest.raises(ValueError, match="request.json SHA256"):
        compare.analyze(tmp_path, baseline_data_dir=None, plots=False)
