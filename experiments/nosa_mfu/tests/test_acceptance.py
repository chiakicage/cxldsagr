"""Independent acceptance survives layout moves and rejects changed execution."""

import copy
import hashlib
import json

import pytest

from experiments.nosa_mfu.src import acceptance
from experiments.nosa_mfu.src.acceptance import (
    attach_acceptance,
    execution_identity,
    publish_check,
    verify_acceptance,
)


def metadata():
    return {
        "args": {
            "prefix_tokens": 64,
            "new_tokens": 64,
            "chunk_size": 64,
            "kernel_backend": "native",
            "repeats": 1,
        },
        "model_config": {"num_hidden_layers": 2},
        "checkpoint_path": "/model",
        "checkpoint_files": {"weights": {"size": 12, "mtime_ns": 1}},
        "checkpoint_config_sha256": "model-config",
        "request_sha256": "input",
        "source_sha256": {"model.py": hashlib.sha256(b"model").hexdigest()},
        "validation": {"finite": True},
        "runtime_settings": {
            "matmul_allow_tf32": False,
            "matmul_allow_bf16_reduced_precision_reduction": True,
            "environment": {"CXLDSAGR_SM90_BACKEND": "native"},
        },
        "native_artifacts": {"/library.so": {"sha256": "library", "size": 12}},
        "torch": "torch",
        "cuda": "cuda",
        "flashinfer": "flashinfer",
        "gpu": {"capability": [9, 0]},
    }


@pytest.fixture(autouse=True)
def source_repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "model.py").write_bytes(b"model")
    monkeypatch.setattr(acceptance, "SOURCE_ROOT", root)


def completed_check(data_dir, checked):
    (data_dir / "sources").mkdir()
    (data_dir / "sources/model.py").write_bytes(b"model")
    (data_dir / "metadata.json").write_text(json.dumps(checked))


def test_check_reused_for_repetitions_and_portable_after_copy(tmp_path):
    checked = metadata()
    check_dir = tmp_path / "check"
    check_dir.mkdir()
    (check_dir / "attention_audit.json").write_text(json.dumps([{"layer": 0}]))
    completed_check(check_dir, checked)
    publish_check(check_dir, checked, "sparse", {"finite": True}, "attention_audit.json")
    benchmark = copy.deepcopy(checked)
    benchmark["args"].update(mode="benchmark", repeats=25, warmup=4)
    assert execution_identity(checked, "sparse") == execution_identity(benchmark, "sparse")
    target = tmp_path / "benchmark"
    target.mkdir()
    attached = attach_acceptance(check_dir / "receipt.json", target, benchmark, "sparse")
    assert (
        verify_acceptance(target, benchmark, "sparse")["receipt_sha256"]
        == attached["receipt_sha256"]
    )
    (check_dir / "attention_audit.json").unlink()
    assert verify_acceptance(target, benchmark, "sparse")["checks"]["passed"]
    (target / "numerical_acceptance/attention_audit.json").write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        verify_acceptance(target, benchmark, "sparse")


@pytest.mark.parametrize(
    "field",
    ["source_sha256", "request_sha256", "checkpoint_files", "runtime_settings", "native_artifacts"],
)
def test_acceptance_rejects_changed_runtime_input_or_checkpoint(tmp_path, field):
    checked = metadata()
    (tmp_path / "audit.json").write_text("[]")
    completed_check(tmp_path, checked)
    publish_check(tmp_path, checked, "sparse", {"finite": True}, "audit.json")
    changed = copy.deepcopy(checked)
    changed[field] = {"changed": True} if isinstance(changed[field], dict) else "changed"
    with pytest.raises(ValueError, match="execution identity"):
        attach_acceptance(tmp_path / "receipt.json", tmp_path / "bench", changed, "sparse")


@pytest.mark.parametrize("failure", ["live_source", "snapshot", "metadata", "missing_metadata"])
def test_final_verification_failure_leaves_no_successful_receipt(tmp_path, failure):
    checked = metadata()
    (tmp_path / "audit.json").write_text("[]")
    completed_check(tmp_path, checked)
    if failure == "live_source":
        (acceptance.SOURCE_ROOT / "model.py").write_bytes(b"changed")
    elif failure == "snapshot":
        (tmp_path / "sources/model.py").write_bytes(b"changed")
    elif failure == "metadata":
        (tmp_path / "metadata.json").write_text("{}")
    else:
        (tmp_path / "metadata.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        publish_check(tmp_path, checked, "sparse", {"finite": True}, "audit.json")
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize(
    "field", ["matmul_allow_tf32", "matmul_allow_bf16_reduced_precision_reduction", "environment"]
)
def test_receipt_rejects_each_precision_or_dispatch_change(tmp_path, field):
    checked = metadata()
    (tmp_path / "audit.json").write_text("[]")
    completed_check(tmp_path, checked)
    publish_check(tmp_path, checked, "sparse", {"finite": True}, "audit.json")
    changed = copy.deepcopy(checked)
    changed["runtime_settings"][field] = (
        {"CXLDSAGR_SM90_BACKEND": "triton"}
        if field == "environment"
        else not changed["runtime_settings"][field]
    )
    with pytest.raises(ValueError, match="execution identity"):
        attach_acceptance(tmp_path / "receipt.json", tmp_path / "bench", changed, "sparse")
