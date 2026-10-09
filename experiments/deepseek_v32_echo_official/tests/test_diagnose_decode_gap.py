"""The gap audit resolves each isolated profile's own result, capture and process."""

import json
import sqlite3
from pathlib import Path

import pytest

from experiments.deepseek_v32_echo_official.src import diagnose_decode_gap as diagnose


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def isolated_profiles(tmp_path):
    panels, hashes = [], {}
    for index, method in enumerate(("hbm", "echo")):
        directory = tmp_path / f"cohort_{method}_profile"
        directory.mkdir()
        result = {
            "schema_version": 4,
            "run_id": directory.name,
            "selected_method": method,
            "methods": [method],
            "method_isolation": "fresh-process-one-method-v1",
            "benchmark": {"run_id": f"cohort_{method}_bench"},
            "validation_receipt": {"receipt_path": str(tmp_path / f"{method}_receipt.json")},
            "nsys_capture_order": [
                "graph_setup",
                f"{method}/prefill_annotated",
                f"{method}/extend_graph_setup",
                f"{method}/extend_annotated",
            ],
            "measurement_identity": {"kind": "minimal-node-profile", "source_sha256": {}},
            "process_provenance": {"pid": 200 + index},
        }
        write_json(directory / "result.json", result)
        capture = directory / "capture_4.sqlite"
        capture.write_bytes(f"test {method} capture".encode())
        for path in (directory / "result.json", capture):
            hashes[str(path)] = diagnose.sha256(path)
        panels.append(
            {
                "method": method,
                "profile_run_id": directory.name,
                "provenance": {
                    "profile_directory": str(directory),
                    "profile_run_id": directory.name,
                    "result_sha256": hashes[str(directory / "result.json")],
                    "sqlite": str(capture),
                    "sqlite_sha256": hashes[str(capture)],
                    "benchmark": result["benchmark"],
                    "validation_receipt": result["validation_receipt"],
                },
            }
        )
    return panels, {"format": "isolated-method-artifacts-v1", "input_hashes": hashes}


def test_resolves_hbm_and_echo_from_different_profile_results(isolated_profiles):
    panels, context = isolated_profiles
    sources = {}
    profiles = [diagnose.local_profile_input(panel, context, sources) for panel in panels]
    assert [profile["profile_run_id"] for profile in profiles] == [
        "cohort_hbm_profile",
        "cohort_echo_profile",
    ]
    assert [profile["target_pid"] for profile in profiles] == [200, 201]
    assert len({profile["sqlite"] for profile in profiles}) == 2
    assert len(sources) == 4
    assert all("source_sha256" not in profile["measurement"] for profile in profiles)


@pytest.mark.parametrize("defect", ["capture_order", "method", "benchmark", "receipt", "digest"])
def test_rejects_misbound_isolated_raw_profile(isolated_profiles, defect):
    panels, context = isolated_profiles
    panel = panels[1]
    provenance = panel["provenance"]
    result_path = Path(provenance["profile_directory"]) / "result.json"
    result = diagnose.read(result_path)
    if defect == "capture_order":
        result["nsys_capture_order"].reverse()
    elif defect == "method":
        result["selected_method"] = "hbm"
    elif defect == "benchmark":
        result["benchmark"] = panels[0]["provenance"]["benchmark"]
    elif defect == "receipt":
        result["validation_receipt"] = panels[0]["provenance"]["validation_receipt"]
    elif defect == "digest":
        result["measurement_identity"]["kind"] = "changed"
    write_json(result_path, result)
    if defect != "digest":
        provenance["result_sha256"] = diagnose.sha256(result_path)
        context["input_hashes"][str(result_path)] = diagnose.sha256(result_path)
    with pytest.raises(ValueError):
        diagnose.local_profile_input(panel, context, {})


def test_legacy_result_uses_the_original_mixed_capture_order(tmp_path):
    result_path = tmp_path / "result.json"
    write_json(
        result_path,
        {
            "run_id": "mixed_profile",
            "nsys_capture_order": ["hbm/extend_annotated", "echo/extend_annotated"],
            "measurement_identity": {"kind": "mixed-profile", "source_sha256": {}},
        },
    )
    hashes = {str(result_path): diagnose.sha256(result_path)}
    for index in (1, 2):
        path = tmp_path / f"capture_{index}.sqlite"
        path.write_text(f"legacy capture {index}")
        hashes[str(path)] = diagnose.sha256(path)
    context = {
        "format": "mixed-method-compact-receipt",
        "receipt": {"inputs_and_sources_sha256": hashes},
    }
    profiles = [
        diagnose.local_profile_input({"method": method}, context, {}) for method in ("hbm", "echo")
    ]
    assert [Path(profile["sqlite"]).name for profile in profiles] == [
        "capture_1.sqlite",
        "capture_2.sqlite",
    ]
    assert all(profile["profile_run_id"] == "mixed_profile" for profile in profiles)
    assert all(profile["target_pid"] is None for profile in profiles)


def test_raw_capture_rejects_a_different_os_process_before_consuming_activities(tmp_path):
    path = tmp_path / "capture.sqlite"
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        database.execute("CREATE TABLE PROCESSES (globalPid INTEGER, pid INTEGER)")
        database.execute("INSERT INTO PROCESSES VALUES (16777216, 42)")
    panel = {"rows": [{"correlation": 1, "process": 16777216}]}
    with pytest.raises(ValueError, match="Raw capture process differs"):
        diagnose.raw_capture(panel, path, expected_pid=43)


def test_main_retains_per_method_measurements_without_a_shared_isolated_identity(
    isolated_profiles, tmp_path, monkeypatch
):
    local, context = isolated_profiles
    experiment = tmp_path / "experiment"
    monkeypatch.setattr(diagnose, "EXPERIMENT", experiment)
    official = []
    official_report = {"sources": {}}
    for method in ("hbm", "echo"):
        directory = experiment / "output/data" / f"official_{method}"
        directory.mkdir(parents=True)
        capture = directory / "decode.sqlite"
        capture.write_text(f"official {method} capture")
        run_path = directory / "run.json"
        write_json(
            run_path, {"workers": [{"nsys": {"path": str(capture.with_suffix(".nsys-rep"))}}]}
        )
        official_report["sources"][f"{method}_profile"] = {
            "run_id": directory.name,
            "run_sha256": diagnose.sha256(run_path),
        }
        official_report["sources"][method] = {"sqlite_sha256": diagnose.sha256(capture)}
        official.append({"method": method, "source": "Official SGLang"})
    for current in official + local:
        current.setdefault("source", "Ours (local MFU)")
        current.update(
            condition="synthetic fixture",
            window={"start_ns": 100, "end_ns": 200, "gpu_idle_ms": 0.0},
            rows=[
                {
                    "kind": "kernel",
                    "category": "Projection / RoPE",
                    "lane": "Compute",
                    "name": "projection",
                    "layer": 0,
                    "stage": "attention_projection",
                    "start_ns": 100,
                    "end_ns": 200,
                    "stream": 1,
                    "graph_node_id": 1,
                    "potential_io": False,
                }
            ],
        )
    panels = official + local + [{"method": "serial_sparse"}, {"method": "dense_prefetch"}]
    monkeypatch.setattr(diagnose, "load_panels", lambda *args: (panels, official_report, context))
    observed_pids = []

    def raw_capture(panel, path, *, expected_pid=None):
        observed_pids.append(expected_pid)
        return {"sqlite": str(path)}, {("kernel", 1): {}}

    monkeypatch.setattr(diagnose, "raw_capture", raw_capture)
    output = experiment / "output/data/gap_audit"
    monkeypatch.setattr("sys.argv", ["diagnose_decode_gap", "--output-dir", str(output)])
    diagnose.main()
    report = diagnose.read(output / "report.json")
    assert "local_profile_measurement" not in report
    assert set(report["local_profile_measurements"]) == {"hbm", "echo"}
    assert report["numerical_acceptance_between_implementations"] is False
    assert observed_pids == [None, 200, None, 201]
    assert report["local_profiles"]["hbm"]["profile_run_id"] == "cohort_hbm_profile"
    assert report["local_profiles"]["echo"]["profile_run_id"] == "cohort_echo_profile"
