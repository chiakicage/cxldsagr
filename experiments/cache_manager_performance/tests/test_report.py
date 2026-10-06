"""Explicit observer reconciliation must bind every retained source and race."""

import copy
import json
from pathlib import Path

import pytest

from experiments.cache_manager_performance.src import report


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def observation(tmp_path):
    directory = tmp_path / "observer"
    directory.mkdir()
    (directory / "driver.py").write_text("# fixed observer source\n")
    (directory / "stdout.log").write_text("successful measured run\n")
    (directory / "stderr.log").write_text("")
    run_id = "cache_manager_bench_test"
    write_json(
        directory / "run.json",
        {
            "command": ["python", "measure.py", "--run-id", run_id],
            "driver_sha256": report.file_sha256(directory / "driver.py"),
            "exit_code": 0,
        },
    )
    samples = [
        {
            "returncode": 0,
            "gpu_utilization_returncode": 0,
            "gpu_utilization": "3, GPU-example, 100",
            "time_utc": f"2026-10-06T00:00:0{index}+00:00",
        }
        for index in range(2)
    ]
    (directory / "gpu_monitor.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples)
    )
    audit = {
        "exit_code": 0,
        "samples": 2,
        "foreign_processes": [],
        "monitor_errors": [],
        "exit_races": [
            {
                "pid": 123,
                "time_utc": samples[1]["time_utc"],
                "prior_observation": {"pid": 123, "start_ticks": 42},
                "reason": "Previously confirmed owned process exited during observation.",
            }
        ],
    }
    write_json(directory / "monitor_audit.json", audit)
    result_path = tmp_path / "result.json"
    write_json(
        result_path,
        {"passed": True, "run_id": run_id, "identity_sha256": "1" * 64},
    )
    entry = {
        "accepted": True,
        "classification": "owned_process_exit_race",
        "run_id": run_id,
        "observer_directory": str(directory),
        "observer_files": {
            name: report.file_sha256(directory / name) for name in report.OBSERVER_FILES
        },
        "result_path": str(result_path),
        "result_sha256": report.file_sha256(result_path),
        "identity_sha256": "1" * 64,
        "reconciled_exit_races": copy.deepcopy(audit["exit_races"]),
    }
    return directory, run_id, result_path, entry


def read(observation, **overrides):
    directory, run_id, result_path, entry = observation
    return report.read_observer(
        directory,
        run_id,
        "GPU-example",
        **{"reconciliation": entry, "result_path": result_path, **overrides},
    )


def test_exit_race_remains_strict_without_explicit_reconciliation(observation):
    directory, run_id, _, _ = observation
    with pytest.raises(ValueError, match="explicit independent reconciliation"):
        report.read_observer(directory, run_id, "GPU-example")


def test_reconciled_exit_race_retains_original_audit_and_all_six_sources(observation):
    directory, _, _, entry = observation
    original = (directory / "monitor_audit.json").read_bytes()
    accepted = read(observation)
    assert accepted["audit"]["exit_races"] == entry["reconciled_exit_races"]
    assert accepted["exit_race_reconciliation"] == entry
    assert set(accepted["files"]) == set(report.OBSERVER_FILES)
    assert (directory / "monitor_audit.json").read_bytes() == original


def test_clean_observer_needs_no_reconciliation(observation):
    directory, run_id, _, _ = observation
    audit = json.loads((directory / "monitor_audit.json").read_text())
    audit["exit_races"] = []
    write_json(directory / "monitor_audit.json", audit)
    accepted = report.read_observer(directory, run_id, "example")
    assert "exit_race_reconciliation" not in accepted


@pytest.mark.parametrize("name", report.OBSERVER_FILES)
def test_every_observer_source_hash_is_required(observation, name):
    observation[3]["observer_files"][name] = "0" * 64
    with pytest.raises(ValueError, match="source file hashes"):
        read(observation)


@pytest.mark.parametrize(
    "fault",
    ["unaccepted", "classification", "run", "missing_race", "extra_race", "changed_race"],
)
def test_reconciliation_cannot_change_or_partially_cover_races(observation, fault):
    entry = observation[3]
    if fault == "unaccepted":
        entry["accepted"] = False
    elif fault == "classification":
        entry["classification"] = "unverified"
    elif fault == "run":
        entry["run_id"] = "different_run"
    elif fault == "missing_race":
        entry["reconciled_exit_races"] = []
    elif fault == "extra_race":
        entry["reconciled_exit_races"] *= 2
    else:
        entry["reconciled_exit_races"][0]["prior_observation"]["start_ticks"] += 1
    with pytest.raises(ValueError, match="exactly cover"):
        read(observation)


@pytest.mark.parametrize("key", ["result_sha256", "identity_sha256"])
def test_reconciliation_binds_result_bytes_and_execution_identity(observation, key):
    observation[3][key] = "0" * 64
    with pytest.raises(ValueError, match="result or execution identity"):
        read(observation)


def test_result_argument_is_required_for_reconciliation(observation):
    with pytest.raises(ValueError, match="measured paths"):
        read(observation, result_path=None)


@pytest.mark.parametrize("key", ["observer_directory", "result_path"])
def test_reconciliation_cannot_bind_other_paths(observation, key, tmp_path):
    other = tmp_path / "other"
    if key == "observer_directory":
        other.mkdir()
    else:
        other.write_bytes(observation[2].read_bytes())
    observation[3][key] = str(other)
    with pytest.raises(ValueError):
        read(observation)


@pytest.mark.parametrize("key", ["foreign_processes", "monitor_errors"])
def test_reconciliation_never_waives_foreign_process_or_monitor_error(observation, key):
    directory, _, _, entry = observation
    path = directory / "monitor_audit.json"
    audit = json.loads(path.read_text())
    audit[key] = [{"reason": "must reject even with matching reconciliation hashes"}]
    write_json(path, audit)
    entry["observer_files"][path.name] = report.file_sha256(path)
    with pytest.raises(ValueError, match="complete measured process"):
        read(observation)


def test_reconciliation_does_not_waive_raw_monitor_failure(observation):
    directory, _, _, entry = observation
    path = directory / "gpu_monitor.jsonl"
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    samples[0]["gpu_utilization_returncode"] = 1
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    entry["observer_files"][path.name] = report.file_sha256(path)
    with pytest.raises(ValueError, match="raw samples"):
        read(observation)


@pytest.mark.parametrize("fault", ["schema", "accepted", "missing_runs", "unknown_mode"])
def test_joint_reconciliation_document_is_explicit_and_scoped(observation, tmp_path, fault):
    document = {
        "schema": "cache-manager-observer-reconciliation-v1",
        "accepted": True,
        "runs": {"bench": observation[3]},
    }
    if fault == "schema":
        document["schema"] = "unreviewed"
    elif fault == "accepted":
        document["accepted"] = False
    elif fault == "missing_runs":
        document["runs"] = {}
    else:
        document["runs"]["different_mode"] = observation[3]
    path = tmp_path / "reconciliation.json"
    write_json(path, document)
    with pytest.raises(ValueError, match="accepted observer reconciliation document"):
        report._read_observer_reconciliation(path)


def test_cli_forwards_explicit_reconciliation(monkeypatch, tmp_path):
    captured = {}

    def generate(*args, **kwargs):
        captured.update(kwargs)
        return {"checks": {"ok": True}}

    monkeypatch.setattr(report, "generate", generate)
    path = tmp_path / "reconciliation.json"
    report.main(
        [
            "--profile-run",
            "profile",
            "--validation-receipt",
            "receipt",
            "--output-dir",
            "output",
            "--check-observer",
            "check",
            "--profile-observer",
            "profile_observer",
            "--observer-reconciliation",
            str(path),
        ]
    )
    assert captured["observer_reconciliation"] == Path(path)
