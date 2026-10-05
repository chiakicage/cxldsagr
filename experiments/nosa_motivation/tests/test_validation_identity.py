"""New receipts bind executed helpers while old evidence keeps its old identity."""

import json
from copy import deepcopy

import pytest

from evaluation.validation import write_receipt
from experiments.nosa_motivation.src.validation import base_identity, begin_validation

HELPERS = (
    "experiments/nosa_motivation/src/provenance.py",
    "experiments/nosa_motivation/src/cpu_environment.py",
    "experiments/nosa_motivation/src/flops.py",
    "experiments/nosa_motivation/src/report.py",
    "experiments/nosa_motivation/src/sources.py",
    "experiments/cache_management/src/capacity_probe.py",
    "experiments/deepseek_v32_echo_prefill/src/backend_provenance.py",
    "experiments/deepseek_v32_motivation/src/report.py",
    "experiments/deepseek_v32_echo_official/src/report.py",
)


@pytest.fixture
def metadata(tmp_path):
    manifest = {
        "executor/runtime.py": "runtime",
        "experiments/nosa_motivation/src/measure.py": "measure",
        **dict.fromkeys(HELPERS, "helper"),
        "experiments/nosa_motivation/src/profile.py": "profile",
        "experiments/nosa_motivation/src/plot.py": "plot",
        "experiments/unrelated/src/measure.py": "unrelated",
    }
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest))
    return {
        "mode": "check",
        "config": {"methods": ["hbm"]},
        "workload_sha256": "tokens",
        "checkpoint": {"source": "checkpoint"},
        "hardware": {"uuid": "gpu"},
        "precision_settings": {"dtype": "bfloat16"},
        "execution_environment": {},
    }


def test_missing_revision_preserves_exact_legacy_source_filter(metadata, tmp_path):
    identity = base_identity(metadata, tmp_path)
    assert identity["schema"] == "motivation-validation-identity-v1"
    assert identity["sources"] == {
        "executor/runtime.py": "runtime",
        "experiments/nosa_motivation/src/measure.py": "measure",
        "experiments/unrelated/src/measure.py": "unrelated",
    }


def test_new_run_declares_revision_and_excludes_unexecuted_profiles(metadata, tmp_path):
    begin_validation(metadata, tmp_path, None, "model")
    assert metadata["validation_identity_revision"] == 2
    identity = metadata["validation_identity"]["base"]
    assert identity["schema"] == "motivation-validation-identity-v2"
    assert identity["sources"] == {
        "executor/runtime.py": "runtime",
        "experiments/nosa_motivation/src/measure.py": "measure",
        **dict.fromkeys(HELPERS, "helper"),
    }


@pytest.mark.parametrize("helper", HELPERS)
def test_changed_executed_helper_rejects_receipt_reuse(metadata, tmp_path, helper):
    begin_validation(metadata, tmp_path, None, "model")
    metadata["validation_identity"]["methods"]["hbm"] = {"backend": "native"}
    receipt = tmp_path / "receipt.json"
    write_receipt(
        receipt, kind="model", identity=metadata["validation_identity"], checks={"passed": True}
    )
    bench = deepcopy(metadata)
    bench["mode"] = "bench"
    begin_validation(bench, tmp_path, receipt, "model")
    manifest_path = tmp_path / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[helper] = "modified helper"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="execution identity"):
        begin_validation(bench, tmp_path, receipt, "model")


@pytest.mark.parametrize("revision", [0, 3, True, "2"])
def test_unknown_identity_revision_fails(metadata, tmp_path, revision):
    metadata["validation_identity_revision"] = revision
    with pytest.raises(ValueError, match="unsupported validation identity"):
        base_identity(metadata, tmp_path)
