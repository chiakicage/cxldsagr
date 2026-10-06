"""Publication must bind original local DSO records, including dispatch bridges."""

import copy
import hashlib
import json

import pytest

from experiments.deepseek_v32_mfu.src import report_mfu as report

RUNS = ("check", "bench", "profile")
SNAPSHOTS = ("execution_runtime_artifacts", "flashinfer_runtime_artifacts")


def native_entry(tmp_path, name, category):
    path = tmp_path / name
    path.write_bytes(name.encode())
    return {
        "name": name,
        "category": category,
        "library": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        },
    }


@pytest.fixture
def records(tmp_path):
    entries = [
        native_entry(tmp_path, "cxldsagr_echo_indexer_a.so", "echo_indexer"),
        native_entry(tmp_path, "cxldsagr_kv_transfer.so", "record_transfer"),
        native_entry(tmp_path, "libcxldsagr_recall_bridge.so", "other_local_native"),
    ]
    return {
        run: {field: {"local_native_jit": copy.deepcopy(entries)} for field in SNAPSHOTS}
        for run in RUNS
    }


def replace_entries(records, entries, *, runs=RUNS):
    for run in runs:
        for field in SNAPSHOTS:
            records[run][field]["local_native_jit"] = copy.deepcopy(entries)


def test_all_local_libraries_are_verified_and_returned(records):
    files = report.audit_local_native_artifacts(records)
    assert len(files) == 3
    assert {item["category"] for item in files} == {
        "echo_indexer",
        "record_transfer",
        "other_local_native",
    }
    assert all(item["source"] == "local_native_jit" and item["kind"] == "library" for item in files)
    assert [item["path"] for item in files] == [
        entry["library"]["path"]
        for entry in records["profile"]["execution_runtime_artifacts"]["local_native_jit"]
    ]


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("field", SNAPSHOTS)
def test_older_runs_cannot_gain_original_mapping_evidence(records, run, field):
    del records[run][field]["local_native_jit"]
    with pytest.raises(ValueError, match=rf"{run} lacks {field}.*retrospectively"):
        report.audit_local_native_artifacts(records)


@pytest.mark.parametrize("missing", ["echo_indexer", "record_transfer"])
def test_both_required_categories_must_be_present(records, missing):
    entries = records["check"]["execution_runtime_artifacts"]["local_native_jit"]
    replace_entries(records, [entry for entry in entries if entry["category"] != missing])
    with pytest.raises(ValueError, match="exactly one echo_indexer and one record_transfer"):
        report.audit_local_native_artifacts(records)


def test_duplicate_required_library_is_rejected(records, tmp_path):
    entries = records["check"]["execution_runtime_artifacts"]["local_native_jit"]
    entries.append(native_entry(tmp_path, "cxldsagr_echo_indexer_b.so", "echo_indexer"))
    replace_entries(records, entries)
    with pytest.raises(ValueError, match="exactly one echo_indexer and one record_transfer"):
        report.audit_local_native_artifacts(records)


def test_duplicate_bridge_path_is_rejected(records):
    entries = records["check"]["execution_runtime_artifacts"]["local_native_jit"]
    entries.append(copy.deepcopy(entries[-1]))
    replace_entries(records, entries)
    with pytest.raises(ValueError, match="repeats a local native library path"):
        report.audit_local_native_artifacts(records)


@pytest.mark.parametrize("run", RUNS)
def test_bridge_loaded_during_execution_is_rejected(records, run):
    records[run]["flashinfer_runtime_artifacts"]["local_native_jit"].pop()
    with pytest.raises(ValueError, match=rf"{run} local native libraries changed.*before/after"):
        report.audit_local_native_artifacts(records)


@pytest.mark.parametrize("run", RUNS)
def test_different_bridge_between_runs_is_rejected(records, run):
    entries = records[run]["execution_runtime_artifacts"]["local_native_jit"][:-1]
    replace_entries(records, entries, runs=(run,))
    with pytest.raises(ValueError, match="check, bench and profile have different"):
        report.audit_local_native_artifacts(records)


@pytest.mark.parametrize("field", ["sha256", "bytes"])
def test_current_file_hash_and_byte_count_must_match(records, field):
    entries = records["check"]["execution_runtime_artifacts"]["local_native_jit"]
    entries[-1]["library"][field] = "0" * 64 if field == "sha256" else 1
    replace_entries(records, entries)
    with pytest.raises(ValueError, match="recorded loaded native artifact changed"):
        report.audit_local_native_artifacts(records)


def test_category_cannot_certify_a_different_named_library(records):
    entries = records["check"]["execution_runtime_artifacts"]["local_native_jit"]
    entries[0]["category"], entries[-1]["category"] = (
        entries[-1]["category"],
        entries[0]["category"],
    )
    replace_entries(records, entries)
    with pytest.raises(ValueError, match="incorrect local native library category"):
        report.audit_local_native_artifacts(records)


@pytest.mark.parametrize("invalid", [None, True, 0, "32"])
def test_invalid_byte_identity_is_rejected(records, invalid):
    entries = records["check"]["execution_runtime_artifacts"]["local_native_jit"]
    entries[0]["library"]["bytes"] = invalid
    replace_entries(records, entries)
    with pytest.raises(ValueError, match="valid path/hash/byte identity"):
        report.audit_local_native_artifacts(records)


def test_file_replacement_during_publication_hash_is_rejected(records, tmp_path, monkeypatch):
    original = report.hashlib.file_digest

    def replacing(stream, algorithm):
        result = original(stream, algorithm)
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"replaced after descriptor hashing")
        replacement.replace(stream.name)
        return result

    monkeypatch.setattr(report.hashlib, "file_digest", replacing)
    with pytest.raises(ValueError, match="recorded loaded native artifact changed"):
        report.audit_local_native_artifacts(records)


def test_direct_publish_rejects_missing_native_evidence_before_writing(
    records, tmp_path, monkeypatch
):
    profile = {
        "schema_version": 3,
        "mode": "profile",
        "accepted": True,
        "num_layers": 3,
        "methods": list(report.METHODS),
    }
    (tmp_path / "result.json").write_text(json.dumps(profile))
    del records["profile"]["execution_runtime_artifacts"]["local_native_jit"]
    monkeypatch.setattr(report, "_independent_runs", lambda *args: (None, None, records))
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="formal native-bound publication"):
        report.publish(tmp_path, output)
    assert not output.exists()


def test_acceptance_includes_bridge_and_required_local_files(records, tmp_path, monkeypatch):
    import torch

    directories = {}
    for run, result in records.items():
        directory = directories[run] = tmp_path / run
        directory.mkdir()
        result.update(
            accepted=True,
            run_id=run,
            correctness={},
            validation_receipt={},
            **{
                field: {}
                for field in (
                    "execution_identity",
                    "source_sha256",
                    "backend_provenance",
                    "indexer_build",
                    "request_sha256",
                    "checkpoint_identity",
                )
            },
        )
        result["execution_runtime_artifacts"]["native_jit"] = []
        (directory / "result.json").write_text(json.dumps(result))
    receipt = {
        "artifact_paths": {
            method + "_control.pt": str(tmp_path / (method + "_control.pt"))
            for method in report.METHODS
        }
    }
    monkeypatch.setattr(report, "_independent_runs", lambda *args: (receipt, directories, records))
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"hidden": torch.ones(2, 2), "logits": torch.ones(1, 2)},
    )
    output = tmp_path / "report"
    output.mkdir()
    acceptance = report.attach_run_acceptance(output, directories["profile"], {}, None)
    files = acceptance["current_recorded_native_artifacts"]
    assert len(files) == 3
    assert {item["category"] for item in files} == {
        "echo_indexer",
        "record_transfer",
        "other_local_native",
    }
    assert (
        "Missing original mapped-library records cannot be supplied retrospectively"
        in (acceptance["native_boundary"])
    )
