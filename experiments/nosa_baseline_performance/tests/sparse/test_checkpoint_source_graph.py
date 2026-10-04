"""Keep the published inference tree analyzable without accepting unknown code."""

import json
from pathlib import Path

import pytest

from experiments.nosa_baseline_performance.src.sparse.module_mfu import (
    _BASELINE_RESIDENT_GRAPH,
    _BF16_PAIR_FORMATTED_GRAPH,
    _BF16_PAIR_GRAPH,
    _MODEL_RESIDENT_GRAPH,
    _validate_source_graph,
)

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_published_checkpoint_matches_its_recorded_inference_sources(backend):
    # Published measurements belong to their captured source tree. Additive
    # offload development must not silently relabel them as current-code runs.
    # The runtime source-graph gate below still rejects every unknown revision.
    metadata = json.loads(
        (
            ROOT / "experiments/nosa_baseline_performance/report/sparse" / backend / "metadata.json"
        ).read_text()
    )
    recorded = {name: metadata["source_sha256"][name] for name in _MODEL_RESIDENT_GRAPH}
    assert recorded == _MODEL_RESIDENT_GRAPH
    assert _validate_source_graph(recorded) == 4


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


def test_current_baseline_capture_matches_reviewed_resident_attribution():
    from experiments.nosa_baseline_performance.src.sparse.capture import fingerprint_sources

    assert _validate_source_graph(fingerprint_sources()) == 4


@pytest.mark.parametrize("name", sorted(_BASELINE_RESIDENT_GRAPH))
def test_baseline_graph_rejects_changed_or_missing_anchor(name):
    changed = dict(_BASELINE_RESIDENT_GRAPH)
    changed[name] = "unknown_source_revision"
    with pytest.raises(ValueError, match="Unreviewed inference graph"):
        _validate_source_graph(changed)
