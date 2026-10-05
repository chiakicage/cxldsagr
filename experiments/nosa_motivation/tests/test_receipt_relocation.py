"""Explicit receipt relocation preserves the signed execution and evidence."""

import hashlib
import json
import shutil
from copy import deepcopy

import pytest

from evaluation.validation import write_receipt
from experiments.nosa_motivation.src.validation import (
    audit_receipt,
    base_identity,
    reference_directory,
)


@pytest.fixture
def relocated_receipt(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "source_manifest.json").write_text(json.dumps({"executor/runtime.py": "source"}))
    metadata = {
        "schema": "test-bench-v1",
        "config": {"methods": ["hbm", "offload"]},
        "workload_sha256": "same input",
        "checkpoint": {"identity": "checkpoint"},
        "hardware": {"uuid": "gpu"},
        "precision_settings": {"dtype": "bfloat16"},
        "execution_environment": {},
    }
    identity = {
        "base": base_identity(metadata, bench),
        "methods": {"hbm": {"runtime": "resident"}, "offload": {"runtime": "sparse"}},
    }
    original = tmp_path / "original"
    original.mkdir()
    (original / "metadata.json").write_text('{"status": "accepted"}\n')
    (original / "hidden.bin").write_bytes(b"all checked hidden values")
    receipt = original / "receipt.json"
    write_receipt(
        receipt,
        kind="test-model",
        identity=identity,
        checks={"passed": True},
        artifacts={"metadata": original / "metadata.json", "hidden": original / "hidden.bin"},
    )
    metadata["validation_identity"] = deepcopy(identity)
    metadata["correctness_receipt"] = {
        "path": str(receipt),
        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "identity": deepcopy(identity),
    }
    (bench / "metadata.json").write_text(json.dumps(metadata))
    relocated = tmp_path / "retained"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    return bench, metadata, relocated / "receipt.json"


def test_explicit_copy_reaudits_without_rewriting_metadata(relocated_receipt):
    bench, metadata, receipt = relocated_receipt
    before = deepcopy(metadata)
    on_disk = (bench / "metadata.json").read_bytes()
    with pytest.raises(FileNotFoundError):
        audit_receipt(metadata, bench, "test-model")
    audit = audit_receipt(metadata, bench, "test-model", receipt_override=receipt)
    assert audit["path"] == str(receipt)
    assert audit["recorded_path"] == before["correctness_receipt"]["path"]
    assert metadata == before and (bench / "metadata.json").read_bytes() == on_disk
    assert (
        reference_directory(
            bench, bench_schema="test-bench-v1", kind="test-model", receipt_override=receipt
        )
        == receipt.parent
    )


def test_override_rejects_even_semantically_identical_receipt_bytes(relocated_receipt):
    bench, metadata, receipt = relocated_receipt
    receipt.write_bytes(receipt.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="external correctness receipt changed"):
        audit_receipt(metadata, bench, "test-model", receipt_override=receipt)


def test_override_still_checks_benchmark_identity(relocated_receipt):
    bench, metadata, receipt = relocated_receipt
    metadata["checkpoint"]["identity"] = "other checkpoint"
    with pytest.raises(ValueError, match="bench identity differs"):
        audit_receipt(metadata, bench, "test-model", receipt_override=receipt)


def test_override_still_checks_every_numerical_artifact(relocated_receipt):
    bench, metadata, receipt = relocated_receipt
    (receipt.parent / "hidden.bin").write_bytes(b"corrupted values")
    with pytest.raises(ValueError, match="evidence changed: hidden"):
        audit_receipt(metadata, bench, "test-model", receipt_override=receipt)


def test_profile_override_is_not_ignored_for_nonbench_input(relocated_receipt):
    bench, _, receipt = relocated_receipt
    with pytest.raises(ValueError, match="requires benchmark"):
        reference_directory(
            bench, bench_schema="other-schema", kind="test-model", receipt_override=receipt
        )
