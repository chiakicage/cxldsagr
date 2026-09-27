"""Benchmark/profile metadata must survive JSON serialization without weakening checks."""

import json
from dataclasses import asdict

import pytest

from experiments.indexer_block_sparse_profile.src.capture import validate_profile_metadata
from models.nosa.tests.test_model import tiny_config


def metadata():
    return {
        "model_config": asdict(tiny_config()),
        "checkpoint_path": "/weights/nosa",
        "checkpoint_files": {"weights.safetensors": {"size": 123, "mtime_ns": 456}},
        "checkpoint_config_sha256": "config",
        "request_sha256": "request",
        "source_sha256": {"model.py": "source"},
        "torch": "torch-version",
        "triton": "triton-version",
        "flashinfer": "flashinfer-version",
        "gpu": {"uuid": "gpu-uuid", "capability": [9, 0]},
    }


def test_serialized_config_tuples_are_compatible():
    current = metadata()
    saved = json.loads(json.dumps(current))
    assert current["model_config"]["eos_token_id"] != saved["model_config"]["eos_token_id"]
    validate_profile_metadata(current, saved)


@pytest.mark.parametrize("key", ["model_config", "source_sha256", "gpu", "checkpoint_files"])
def test_profile_rejects_changed_runtime_or_source(key):
    current = metadata()
    saved = json.loads(json.dumps(current))
    saved[key]["changed"] = True
    with pytest.raises(ValueError, match=key):
        validate_profile_metadata(current, saved)
