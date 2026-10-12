import json
from pathlib import Path

import pytest

from experiments.deepseek_v32_echo_official.src import report_decode_recall_observation as report
from experiments.deepseek_v32_echo_official.src.decode_recall_observer import classify_recall


def test_publish_preserves_saved_input_evidence_and_refuses_overwrite(tmp_path, monkeypatch):
    layers = [
        {
            "layer": layer,
            **classify_recall([[5, -1]], [[100, 0]], [[5]], 100, 1),
            "input_snapshot": {
                "host_ids": [[5, -1]],
                "old_slots": [[100, 0]],
                "stage_ids": [[5]],
                "counter": [1],
            },
        }
        for layer in range(3)
    ]
    observed = {"layers": layers, "totals": {"prefetch_d2d_bytes": 3456}}
    record = {"run_id": "fixture", "workers": [{"hooks": {"fixture.json": "digest"}}]}
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text("{}")
    monkeypatch.setattr(report, "audit", lambda _: (record, observed))
    output, selected = tmp_path / "output", tmp_path / "selected"
    result = report.publish(run, output, selected)
    assert result["totals"] == {"prefetch_d2d_bytes": 3456}
    assert json.loads((selected / "recall_observation.json").read_text()) == observed
    assert json.loads((selected / "provenance.json").read_text())["formal_timing"] is False
    assert (selected / "layers.csv").read_text().count("1152") == 6
    with pytest.raises(FileExistsError):
        report.publish(run, output)


def test_audit_rejects_bench_or_incomplete_runs_before_reading_artifacts(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="completed independent"):
        report.audit(tmp_path)


def test_published_input_rows_reconstruct_exact_record_counts():
    path = Path(__file__).parents[1] / "report/decode_recall_observation/recall_observation.json"
    observed = json.loads(path.read_text())
    for layer in observed["layers"]:
        raw = layer["input_snapshot"]
        expected = classify_recall(
            raw["host_ids"],
            raw["old_slots"],
            raw["stage_ids"],
            layer["prefetch_device_start"],
            raw["counter"][0],
        )
        assert all(layer[key] == value for key, value in expected.items())
