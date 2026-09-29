"""Keep the published inference tree analyzable without accepting unknown code."""

import hashlib
from pathlib import Path

import pytest

from experiments.indexer_block_sparse_profile.src.module_mfu import (
    _BF16_PAIR_FORMATTED_GRAPH,
    _BF16_PAIR_GRAPH,
    _validate_source_graph,
)

ROOT = Path(__file__).resolve().parents[3]


def test_published_checkpoint_matches_actual_inference_sources():
    actual = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in _BF16_PAIR_FORMATTED_GRAPH
    }
    assert actual == _BF16_PAIR_FORMATTED_GRAPH
    assert _validate_source_graph(actual) == 4


def test_original_measured_checkpoint_remains_analyzable():
    assert _validate_source_graph(_BF16_PAIR_GRAPH) == 4
    changed = {
        name
        for name in _BF16_PAIR_GRAPH
        if _BF16_PAIR_GRAPH[name] != _BF16_PAIR_FORMATTED_GRAPH[name]
    }
    assert changed == {"operators/sm90/_nosa_attention_fa3.py"}


@pytest.mark.parametrize("name", sorted(_BF16_PAIR_FORMATTED_GRAPH))
@pytest.mark.parametrize("missing", [False, True])
def test_checkpoint_cannot_fall_back_to_legacy_attribution_when_source_changes(name, missing):
    hashes = dict(_BF16_PAIR_FORMATTED_GRAPH)
    if missing:
        hashes.pop(name)
    else:
        hashes[name] = "unknown_source_revision"
    with pytest.raises(ValueError, match="Unreviewed inference graph"):
        _validate_source_graph(hashes)
