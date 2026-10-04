import hashlib
import json
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
        "layers/attention.py",
        "evaluation/provenance.py",
        "experiments/deepseek_v32_echo_cache/src/artifact_locations.py",
        "experiments/deepseek_v32_echo_cache/scripts/run.sh",
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
