"""Benchmark/profile metadata must survive JSON serialization without weakening checks."""

import json
from dataclasses import asdict

import pytest

from experiments.nosa_baseline_performance.src.sparse.capture import validate_profile_metadata
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
        "cuda": "cuda-version",
        "triton": "triton-version",
        "flashinfer": "flashinfer-version",
        "tvm_ffi": "ffi-version",
        "native_build": {"compiler": {"version": "nvcc-version"}},
        "gpu": {"uuid": "gpu-uuid", "capability": [9, 0]},
    }


def test_serialized_config_tuples_are_compatible():
    current = metadata()
    saved = json.loads(json.dumps(current))
    assert current["model_config"]["eos_token_id"] != saved["model_config"]["eos_token_id"]
    validate_profile_metadata(current, saved)


@pytest.mark.parametrize(
    "key", ["model_config", "source_sha256", "gpu", "checkpoint_files", "native_build"]
)
def test_profile_rejects_changed_runtime_or_source(key):
    current = metadata()
    saved = json.loads(json.dumps(current))
    saved[key]["changed"] = True
    with pytest.raises(ValueError, match=key):
        validate_profile_metadata(current, saved)


@pytest.mark.parametrize("key", ["cuda", "tvm_ffi"])
def test_profile_rejects_changed_runtime_version(key):
    current = metadata()
    saved = json.loads(json.dumps(current))
    saved[key] = "different-version"
    with pytest.raises(ValueError, match=key):
        validate_profile_metadata(current, saved)


def test_kernel_backend_cli_defaults_to_native_and_accepts_triton_control():
    from experiments.nosa_baseline_performance.src.sparse.capture import _build_parser

    required = ["--run-id", "fixture", "--output-dir", "/tmp/unused"]
    parser = _build_parser()
    assert parser.parse_args(required).kernel_backend == "native"
    assert parser.parse_args([*required, "--kernel-backend", "triton"]).kernel_backend == "triton"
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--kernel-backend", "auto"])
