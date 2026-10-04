import json
from pathlib import Path

import pytest

from experiments.gr_serving.src import retain_nosa_subset as subset


def test_jsonl_selection_preserves_original_bytes_and_drops_full_acceptance():
    original = (
        b'{"model":"deepseek_v32","latency_ms":19}\r\n'
        b'{ "model": "nosa", "latency_ms": 1.2500 }\r\n'
        b'{"event":"accepted","cases":56}\n'
    )
    assert subset.select_lines(original, log=True) == (
        b'{ "model": "nosa", "latency_ms": 1.2500 }\r\n'
    )
    with pytest.raises(ValueError, match="unknown model"):
        subset.select_lines(original)
    with pytest.raises(ValueError, match="unknown model"):
        subset.select_lines(b'{"model":"another_model"}\n')


def test_metadata_keeps_old_identity_and_marks_derivation():
    original = {
        "status": "accepted",
        "run_id": "old-run",
        "source_sha256": "original-source",
        "started_unix": 1,
        "completed_unix": 2,
        "parameters": {"models": ["deepseek_v32", "nosa"], "requests": 32},
        "models": {"deepseek_v32": {"weights": "d"}, "nosa": {"weights": "n"}},
        "cases": [
            {"model": "deepseek_v32", "requests": 32},
            {"model": "nosa", "requests": 32},
        ],
        "measured_requests": 64,
    }
    retained = subset.metadata_subset(original, "old-metadata-hash")
    assert original["status"] == "accepted" and original["measured_requests"] == 64
    assert retained["status"] == "retained_subset"
    assert retained["run_id"] == "old-run"
    assert retained["source_sha256"] == "original-source"
    assert retained["completed_unix"] == 2
    assert retained["parameters"]["models"] == ["nosa"]
    assert retained["measured_requests"] == 32
    assert retained["subset_retention"]["original_metadata_sha256"] == "old-metadata-hash"


def test_original_report_execution_does_not_write_bytecode(tmp_path):
    name = "experiments/gr_serving/src/report.py"
    source = tmp_path / "source" / name
    source.parent.mkdir(parents=True)
    source.write_text("def summarize(rows):\n    return len(rows)\n")
    manifest = {name: subset.digest(source)}
    before = sorted(tmp_path.rglob("*"))
    assert subset.load_original_report(tmp_path, manifest).summarize([1, 2]) == 2
    assert sorted(tmp_path.rglob("*")) == before
    source.write_text("raise RuntimeError('changed')\n")
    with pytest.raises(ValueError, match="source changed"):
        subset.load_original_report(tmp_path, manifest)


def test_safe_path_rejects_escape_and_symlink(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "link").symlink_to(tmp_path, target_is_directory=True)
    for relative in ("../other", "/absolute", "link/file"):
        with pytest.raises(ValueError):
            subset.safe_path(repo, relative)


def make_preparation(tmp_path):
    repo, stage = tmp_path / "repo", tmp_path / "stage"
    run = subset.EXPERIMENT / "output/data/gr_serving_h200_20261002_h4k_01"
    before_files = {
        str(run / "measurements.jsonl"): b"mixed original\n",
        str(run / "audit.json"): b"old audit\n",
        str(run / "reference/deepseek_v32/0/000000.pt"): b"retired tensor",
        str(run / "reference/nosa/0/000000.pt"): b"nosa tensor",
        str(run / "source/models/nosa/model.py"): b"original source",
    }
    for name, blob in before_files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    writes = {}
    for name, blob in {
        str(run / "measurements.jsonl"): b"retained nosa\n",
        str(run / "audit.json"): b"subset audit\n",
    }.items():
        target = stage / "files" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        writes[name] = subset.digest(target)
    plan = {
        "schema": subset.SCHEMA,
        "mode": "prepared",
        "repository": str(repo.resolve()),
        "tool_sha256": subset.digest(Path(subset.__file__)),
        "roots": [str(run)],
        "before": subset.inventory(repo, run),
        "writes": writes,
        "remove": [str(run / "reference/deepseek_v32/0/000000.pt")],
    }
    (stage / "plan.json").write_bytes(subset.json_bytes(plan))
    published = repo / subset.EXPERIMENT / "report/echo_chunks/summary.json"
    published.parent.mkdir(parents=True)
    published.write_text('{"status":"accepted"}\n')
    ledger = {
        "schema": "echo-cache-publication-v1",
        "status": "published",
        "retired_model": "deepseek_v32",
        "retired_runs": [f"gr_serving_h200_20261002_{label}_01" for label in subset.SPECS],
        "prepared_visual_review": "passed",
        "accepted_gates": [
            "layers3_numeric",
            "memory",
            "chunk_screen",
            "formal_traces",
            "diagnostics",
            "default_review",
        ],
        "published_files_sha256": {
            str(published.relative_to(repo)): subset.digest(published),
        },
    }
    publication = tmp_path / "publication.json"
    publication.write_bytes(subset.json_bytes(ledger))
    return repo, stage, plan, publication, before_files


def test_verify_rejects_stale_originals_and_changed_payload(tmp_path):
    repo, stage, plan, _, _ = make_preparation(tmp_path)
    assert subset.verify_prepared(repo, stage) == plan
    target = repo / next(iter(plan["writes"]))
    old = target.read_bytes()
    target.write_bytes(b"concurrent change")
    with pytest.raises(ValueError, match="original products changed"):
        subset.verify_prepared(repo, stage)
    target.write_bytes(old)
    staged = stage / "files" / next(iter(plan["writes"]))
    staged.write_bytes(b"edited prepared content")
    with pytest.raises(ValueError, match="replacement files changed"):
        subset.verify_prepared(repo, stage)


@pytest.mark.parametrize(
    "bad",
    [
        "experiments/deepseek_v32_echo_prefill/report/layers3/summary.json",
        "experiments/gr_serving/output/data/gr_serving_h200_20261002_h4k_01/source/model.py",
        "experiments/gr_serving/output/data/gr_serving_h200_20261002_h4k_01/reference/nosa/0/000000.pt",
    ],
)
def test_verify_forbids_unrelated_or_protected_operations(tmp_path, bad):
    repo, stage, plan, _, _ = make_preparation(tmp_path)
    plan["remove"].append(bad)
    (stage / "plan.json").write_bytes(subset.json_bytes(plan))
    with pytest.raises(ValueError):
        subset.verify_prepared(repo, stage)


def test_publication_is_required_and_hash_bound(tmp_path):
    repo, stage, _plan, publication, before = make_preparation(tmp_path)
    ledger = json.loads(publication.read_text())
    ledger["status"] = "pending"
    publication.write_bytes(subset.json_bytes(ledger))
    with pytest.raises(ValueError, match="not complete"):
        subset.apply_prepared(repo, stage, publication)
    assert {name: (repo / name).read_bytes() for name in before} == before
    ledger["status"] = "published"
    publication.write_bytes(subset.json_bytes(ledger))
    (repo / next(iter(ledger["published_files_sha256"]))).write_text("changed")
    with pytest.raises(ValueError, match="published replacement changed"):
        subset.apply_prepared(repo, stage, publication)
    assert not (stage / "rollback").exists()


def test_apply_preserves_source_and_nosa_and_removes_only_declared_files(tmp_path):
    repo, stage, plan, publication, before = make_preparation(tmp_path)
    result = subset.apply_prepared(repo, stage, publication)
    assert result["status"] == "applied"
    assert not (stage / "rollback").exists()
    for name in before:
        if name in plan["writes"]:
            assert subset.digest(repo / name) == plan["writes"][name]
        elif name in plan["remove"]:
            assert not (repo / name).exists()
        else:
            assert (repo / name).read_bytes() == before[name]


def test_apply_failure_rolls_back_all_mutations(tmp_path, monkeypatch):
    repo, stage, _, publication, before = make_preparation(tmp_path)
    original_copy = subset.shutil.copyfile

    def fail_second_payload(source, destination, *args, **kwargs):
        if str(source).startswith(str(stage / "files")) and Path(source).name == "audit.json":
            raise OSError("simulated disk failure")
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(subset.shutil, "copyfile", fail_second_payload)
    with pytest.raises(OSError, match="disk failure"):
        subset.apply_prepared(repo, stage, publication)
    assert {name: (repo / name).read_bytes() for name in before} == before
    assert not list(repo.rglob("*.subset-tmp"))


def test_preexisting_temp_is_not_deleted_on_failure(tmp_path):
    repo, stage, plan, publication, before = make_preparation(tmp_path)
    target = repo / next(iter(plan["writes"]))
    temporary = target.with_name(target.name + ".subset-tmp")
    # Add to the protected original inventory so failure occurs in the install,
    # rather than at the earlier concurrent-file check.
    temporary.write_bytes(b"unrelated preexisting data")
    plan["before"][str(temporary.relative_to(repo))] = subset.digest(temporary)
    (stage / "plan.json").write_bytes(subset.json_bytes(plan))
    with pytest.raises(ValueError, match="already exists"):
        subset.apply_prepared(repo, stage, publication)
    assert temporary.read_bytes() == b"unrelated preexisting data"
    assert {name: (repo / name).read_bytes() for name in before} == before
