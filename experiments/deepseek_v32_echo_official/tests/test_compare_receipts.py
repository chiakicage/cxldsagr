import hashlib
import shutil
from copy import deepcopy

import pytest

from evaluation.validation import write_receipt
from experiments.deepseek_v32_echo_official.src.compare_existing import numerical_evidence


@pytest.fixture(params=["motivation", "official"])
def checked_report(tmp_path, request):
    kind = request.param
    stem = "motivation" if kind == "motivation" else "echo-official"
    receipt_kind = f"deepseek-v32-{stem}-full-trace-v1"
    numerical_key = (
        "all_candidate_hidden_and_logits_exact"
        if kind == "motivation"
        else "all_candidate_hidden_and_logits_numerical_pass"
    )
    checked = {
        "status": "passed",
        "checked_requests": 128 if kind == "motivation" else 64,
        numerical_key: True,
        "numerical_summary": {"verified": True},
    }
    identity = {"base": {"source": "fixture"}, "methods": {"hbm": {}, "echo": {}}}
    artifact = tmp_path / "saved_output.bin"
    artifact.write_bytes(b"checked output")
    path = tmp_path / "receipt.json"
    write_receipt(
        path,
        kind=receipt_kind,
        identity=identity,
        checks={"passed": True, "numerical_and_lifecycle": checked},
        artifacts={"output": artifact},
    )
    evidence = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": receipt_kind,
        "identity": identity,
    }
    metadata = {
        "schema": f"deepseek-v32-{stem}-bench-v1",
        "mode": "bench",
        "validation_identity": identity,
        "correctness_receipt": evidence,
    }
    audit = {numerical_key: kind == "official", "numerical_summary": checked["numerical_summary"]}
    report_evidence = {key: evidence[key] for key in ("path", "sha256", "kind")}
    if kind == "motivation":
        report_evidence["checks"] = {"passed": True, "numerical_and_lifecycle": checked}
        audit["independent_correctness_receipt"] = report_evidence
    else:
        audit["correctness_receipt"] = report_evidence
    return metadata, audit, kind, path, artifact


def test_comparison_reverifies_independent_receipt(checked_report):
    metadata, audit, kind, _, _ = checked_report
    result = numerical_evidence(metadata, audit, kind)
    assert result["mode"] == "independent_check_receipt"
    assert result["checked_requests"] == (128 if kind == "motivation" else 64)


def test_comparison_rejects_changed_numerical_artifact(checked_report):
    metadata, audit, kind, _, artifact = checked_report
    artifact.write_bytes(b"changed output")
    with pytest.raises(ValueError, match="evidence changed"):
        numerical_evidence(metadata, audit, kind)


def test_comparison_rejects_mismatching_execution_identity(checked_report):
    metadata, audit, kind, _, _ = checked_report
    metadata = deepcopy(metadata)
    metadata["correctness_receipt"]["identity"] = deepcopy(metadata["validation_identity"])
    metadata["correctness_receipt"]["identity"]["base"]["source"] = "other source"
    with pytest.raises(ValueError, match="identity differs"):
        numerical_evidence(metadata, audit, kind)


def test_comparison_accepts_exact_relocated_receipt(checked_report, tmp_path):
    metadata, audit, kind, path, artifact = checked_report
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copyfile(artifact, relocated / artifact.name)
    replacement = relocated / "renamed.json"
    shutil.copyfile(path, replacement)
    path.unlink()
    result = numerical_evidence(metadata, audit, kind, receipt_override=replacement)
    assert result["sha256"] == metadata["correctness_receipt"]["sha256"]


def test_legacy_comparison_retains_its_original_numerical_boundary(checked_report):
    metadata, audit, kind, _, _ = checked_report
    metadata["schema"] = metadata["schema"].replace("-bench-v1", "-v1")
    numerical_key = (
        "all_candidate_hidden_and_logits_exact"
        if kind == "motivation"
        else "all_candidate_hidden_and_logits_numerical_pass"
    )
    audit[numerical_key] = True
    assert numerical_evidence(metadata, audit, kind) == {"mode": "combined_measurement_and_check"}
