import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluation import provenance as measure


def test_numerical_gate_checks_every_token_and_rejects_nonfinite():
    reference = torch.ones((4, 8), dtype=torch.bfloat16)
    actual = reference.clone()
    actual[1, 2] = 2
    with pytest.raises(AssertionError):
        measure.numerical_comparison(actual, reference, atol=0, rtol=0)
    actual[1, 2] = torch.nan
    with pytest.raises(AssertionError, match="nonfinite"):
        measure.numerical_comparison(actual, reference, atol=0, rtol=0)


def test_source_gate_rejects_drift_after_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "source_root"
    source.mkdir()
    code = source / "operator.py"
    code.write_text("old source\n")
    monkeypatch.setattr(measure, "ROOT", source)
    manifest = {"operator.py": hashlib.sha256(code.read_bytes()).hexdigest()}
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest))
    measure.verify_source_snapshot(tmp_path)
    code.write_text("changed while measuring\n")
    with pytest.raises(RuntimeError, match="implementation changed"):
        measure.verify_source_snapshot(tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        "models/attention_contracts.py",
        "evaluation/provenance.py",
        "experiments/cache_management/src/artifact_locations.py",
        "experiments/cache_management/scripts/run.sh",
    ],
)
def test_snapshot_includes_shared_model_layers_and_experiment_entrypoints(
    tmp_path, monkeypatch, relative
):
    root = tmp_path / "repo"
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_text("shared = 1\n")
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(measure, "ROOT", root)
    measure.source_snapshot(output)
    manifest = json.loads((output / "source_manifest.json").read_text())
    assert relative in manifest
    assert (output / "source" / relative).read_text() == "shared = 1\n"


def test_report_helpers_snapshot_loaded_files_and_environment_with_explicit_boundary(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    source = root / "experiments/example/src/report.py"
    source.parent.mkdir(parents=True)
    content = b"report = 'current disk source'\r\n"
    source.write_bytes(content)
    (source.parent / "not_loaded.py").write_text("not imported\n")
    for name in ("pyproject.toml", "uv.lock"):
        (root / name).write_text(f"# {name}\n")
    monkeypatch.setattr(measure, "ROOT", root)
    monkeypatch.setitem(sys.modules, "fixture_report", SimpleNamespace(__file__=str(source)))
    monkeypatch.setitem(sys.modules, "fixture_alias", SimpleNamespace(__file__=str(source)))
    monkeypatch.setattr(sys, "orig_argv", ["python", "-m", "fixture_report"])

    output = tmp_path / "report"
    manifest = measure.snapshot_report_helpers(output)

    assert json.loads((output / "report_helper_sources.json").read_text()) == manifest
    assert set(manifest["files"]) == {
        "experiments/example/src/report.py",
        "pyproject.toml",
        "uv.lock",
    }
    entry = manifest["files"]["experiments/example/src/report.py"]
    assert entry["modules"] == ["fixture_alias", "fixture_report"]
    for name, saved in manifest["files"].items():
        original = (root / name).read_bytes()
        assert (output / saved["snapshot_path"]).read_bytes() == original
        assert saved["sha256"] == hashlib.sha256(original).hexdigest()
        assert saved["size_bytes"] == len(original)
    assert manifest["invocation"] == {
        "orig_argv": ["python", "-m", "fixture_report"],
        "cwd": str(Path.cwd()),
        "python": {"executable": sys.executable, "version": sys.version},
    }
    assert "Current loaded repository Python modules only" in manifest["boundary"]
    assert "not captured runtime/native identity" in manifest["boundary"]
    assert "not every possible source" in manifest["boundary"]


def test_report_helpers_exclude_external_generated_and_non_python_files(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        (root / name).write_text("environment\n")
    relative_paths = [
        ".venv/lib/dependency.py",
        "3rdparty/library/helper.py",
        "experiments/example/output/source/helper.py",
        "experiments/example/report/source/helper.py",
        "operators/example/build/helper.py",
        "GR/generated/helper.py",
        "other_directory/helper.py",
        "operators/example/native.so",
    ]
    paths = [root / relative for relative in relative_paths]
    paths.append(tmp_path / "external/helper.py")
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded\n")
        monkeypatch.setitem(
            sys.modules, f"fixture_excluded_{index}", SimpleNamespace(__file__=path)
        )
    monkeypatch.setattr(measure, "ROOT", root)
    manifest = measure.snapshot_report_helpers(tmp_path / "report")
    assert set(manifest["files"]) == {"pyproject.toml", "uv.lock"}


def test_report_helpers_preserve_source_error_and_existing_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    source = root / "evaluation/helper.py"
    source.parent.mkdir(parents=True)
    source.write_text("helper\n")
    for name in ("pyproject.toml", "uv.lock"):
        (root / name).write_text("environment\n")
    monkeypatch.setattr(measure, "ROOT", root)
    monkeypatch.setitem(sys.modules, "fixture_report_error", SimpleNamespace(__file__=source))
    failure = OSError("source read failed")
    original_read = Path.read_bytes

    def fail_source_read(path):
        if path == source:
            raise failure
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", fail_source_read)
    output = tmp_path / "report"
    with pytest.raises(OSError) as caught:
        measure.snapshot_report_helpers(output)
    assert caught.value is failure
    assert not (output / "report_helper_sources.json").exists()
    snapshot = output / "source/report_helpers"
    (snapshot / "keep").write_text("existing snapshot\n")
    with pytest.raises(FileExistsError):
        measure.snapshot_report_helpers(output)
    assert (snapshot / "keep").read_text() == "existing snapshot\n"


def test_access_trace_gate_rejects_changed_draw_or_truncated_workload(tmp_path):
    trace = tmp_path / "requests.csv"
    trace.write_text(
        "request_id,user_id,visit_index,is_revisit,previous_request_id,synthetic_timestamp\n"
        "0,17,0,0,,0.0\n"
        "1,17,1,1,0,1.0\n"
    )
    rows = [
        {
            "request_id": 0,
            "user_id": 17,
            "visit_index": 0,
            "is_revisit": False,
            "previous_request_id": None,
            "timestamp": 0.0,
        },
        {
            "request_id": 1,
            "user_id": 17,
            "visit_index": 1,
            "is_revisit": True,
            "previous_request_id": 0,
            "timestamp": 1.0,
        },
    ]
    workload = SimpleNamespace(requests=rows)
    assert (
        measure.verify_access_trace(workload, trace)
        == hashlib.sha256(trace.read_bytes()).hexdigest()
    )
    rows[1]["user_id"] = 18
    with pytest.raises(ValueError, match="differs from saved access trace"):
        measure.verify_access_trace(workload, trace)
    rows.pop()
    with pytest.raises(ValueError, match="different lengths"):
        measure.verify_access_trace(workload, trace)
