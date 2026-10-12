"""Warm cohort evidence cannot borrow cold results or alias child roles."""

import json
from copy import deepcopy

import pytest

from experiments.deepseek_v32_mfu.src import q1_warm_report as report


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(
        report,
        "validate_completed_result",
        lambda path, **kwargs: json.loads(path.read_text()),
    )
    manifest = {
        "schema_version": 1,
        "kind": report.KIND,
        "completed_at_utc": "2026-10-11T10:00:00+00:00",
        "methods": {"hbm": {}, "echo": {}},
        "execution_order": [],
    }
    evidence = report.audit.Evidence()
    for index, (method, phase) in enumerate(report.ORDER):
        run_id = f"warm_{method}_{phase}"
        directory = tmp_path / run_id
        directory.mkdir()
        (directory / "result.json").write_text(
            json.dumps({"schema_version": 4, "methods": [method], "run_id": run_id})
        )
        (directory / "runner.json").write_text("{}")
        manifest["methods"][method][phase] = {
            "directory": str(directory),
            "run_id": run_id,
            "order": index,
            "result_sha256": evidence.digest(directory / "result.json"),
            "runner_record_sha256": evidence.digest(directory / "runner.json"),
        }
        manifest["execution_order"].append({"method": method, "phase": phase, "run_id": run_id})
    return manifest


def test_four_child_roles_are_read(cohort):
    directories, results = report.manifest_children(cohort, report.audit.Evidence())
    assert set(directories["hbm"]) == {"check"}
    assert set(results["echo"]) == {"check", "bench", "profile"}


@pytest.mark.parametrize("defect", ("kind", "missing", "alias", "changed", "order", "method"))
def test_incomplete_or_mixed_cohort_rejected(cohort, defect):
    manifest = deepcopy(cohort)
    child = manifest["methods"]["echo"]["profile"]
    if defect == "kind":
        manifest["kind"] = "deepseek-v32-method-isolation"
    elif defect == "missing":
        del manifest["methods"]["echo"]["bench"]
    elif defect == "alias":
        child.update(manifest["methods"]["echo"]["check"])
    elif defect == "changed":
        (report.Path(child["directory"]) / "runner.json").write_text('{"changed":true}')
    elif defect == "order":
        manifest["execution_order"].reverse()
    else:
        path = report.Path(child["directory"]) / "result.json"
        value = json.loads(path.read_text())
        value["methods"] = ["hbm"]
        path.write_text(json.dumps(value))
        child["result_sha256"] = report.audit.Evidence().digest(path)
    with pytest.raises(ValueError):
        report.manifest_children(manifest, report.audit.Evidence())


def test_saved_outputs_are_reread_on_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    directories = {"hbm": {}, "echo": {}}
    for method, paths in directories.items():
        check = tmp_path / method
        check.mkdir()
        paths["check"] = check
        baseline = {
            "hidden": torch.zeros(1, 7168, dtype=torch.bfloat16),
            "logits": torch.zeros(1, 129280),
        }
        changed = {key: value + 1 for key, value in baseline.items()}
        torch.save(baseline, check / f"{method}_control.pt")
        torch.save(baseline["logits"], check / f"{method}_prefix_logits.pt")
        torch.save(
            {"baseline": baseline, "changed_baseline": changed, "changed_graph": changed},
            check / f"{method}_extend_graph_check.pt",
        )
    profile = tmp_path / "profile"
    profile.mkdir()
    directories["echo"]["profile"] = profile
    torch.save(baseline, profile / "echo_profile_output.pt")
    output = report.saved_outputs(directories, report.audit.Evidence())
    assert len(output["inventory"]) == 20
    assert all(row["passed"] for row in output["comparisons"])
    baseline["hidden"][0, 0] = 10
    torch.save(baseline, profile / "echo_profile_output.pt")
    with pytest.raises(ValueError, match="Numerical mismatch"):
        report.saved_outputs(directories, report.audit.Evidence())
