import json
import shutil
import sqlite3
from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_echo_official.src import measure
from experiments.deepseek_v32_echo_official.src.profile import (
    SCHEMA,
    finish_evidence,
    instrument_stages,
    main,
    profile_environment,
)
from experiments.deepseek_v32_echo_official.src.profile_report import analyze
from experiments.deepseek_v32_echo_official.tests.test_report import (
    _independent_check,
)
from experiments.deepseek_v32_echo_official.tests.test_report import (
    saved_run as saved_run,  # noqa: PLC0414 -- expose the shared pytest fixture
)
from experiments.deepseek_v32_echo_prefill.tests.test_analyze_nsys import make_capture
from experiments.deepseek_v32_motivation.src.profile import Scopes
from experiments.nosa_motivation.src.validation import begin_validation


@pytest.mark.parametrize("fail", [False, True])
def test_stage_instrumentation_preserves_owner_arguments_and_closes_scopes(fail):
    owner, output = object(), object()
    session = SimpleNamespace(length=0)

    def forward(actual, ids, *, scope=None, **kwargs):
        assert actual is session and ids == [1, 2]
        assert kwargs == {"owner": owner, "chunk_size": 2}
        with scope("embedding"):
            pass
        with scope("layer_0_source_0"), scope("attention_projection"):
            if fail:
                raise ValueError("injected model failure")
        return output

    backend = SimpleNamespace(_forward=forward)
    scopes = Scopes("echo", "cold", 1, nvtx=False)
    with instrument_stages(backend, scopes):
        if fail:
            with pytest.raises(ValueError, match="injected model failure"):
                backend._forward(session, [1, 2], owner=owner, chunk_size=2)
        else:
            assert backend._forward(session, [1, 2], owner=owner, chunk_size=2) is output
    assert backend._forward is forward
    assert not scopes.active and scopes._chunk_scope is None
    assert {row["stage"] for row in scopes.calls} == {
        "forward",
        "chunk",
        "embedding",
        "layer",
        "attention_projection",
    }


def test_profile_requires_explicit_capture_before_loading_reference(tmp_path):
    with pytest.raises(ValueError, match="requires --nsys"):
        main(["--run-id", "test", "--reference-run", str(tmp_path / "absent")])


def _official_capture(path, phase):
    make_capture(path)
    prefix = f"motivation/echo/{phase}/history/chunk_0/"

    def convert(label):
        tail = label.split("extend_annotated/", 1)[1]
        if "/" not in tail:
            tail += "/layer"
        if tail.startswith("shared/"):
            tail = "layer_" + tail
        return prefix + tail

    with sqlite3.connect(path) as db:
        for rowid, label in db.execute(
            "SELECT rowid, text FROM NVTX_EVENTS WHERE text IS NOT NULL"
        ):
            db.execute("UPDATE NVTX_EVENTS SET text=? WHERE rowid=?", (convert(label), rowid))
        label = db.execute("SELECT value FROM StringIds WHERE id=20").fetchone()[0]
        db.execute("UPDATE StringIds SET value=? WHERE id=20", (convert(label),))


@pytest.fixture
def profile_run(saved_run):
    rows = _independent_check(saved_run)
    metadata = json.loads((saved_run / "metadata.json").read_text())
    methods = metadata["validation_identity"]["methods"]
    directory = saved_run / "profile"
    directory.mkdir()
    for name in ("source", "workload"):
        shutil.copytree(saved_run / name, directory / name)
    shutil.copyfile(saved_run / "source_manifest.json", directory / "source_manifest.json")
    metadata.update(
        schema=SCHEMA,
        mode="bench",
        run_id="profile_fixture",
        cases=[{"scheme": "echo", "requests": 3}],
        profiling_environment={},
        nsys_injection=None,
        reference_receipt_sha256=measure.digest(saved_run / "receipt.json"),
    )
    begin_validation(metadata, directory, saved_run / "receipt.json", measure.RECEIPT_KIND)
    metadata["validation_identity"]["methods"] = {"echo": methods["echo"]}
    metadata["captures"] = [
        {
            "capture_index": index,
            "scheme": "echo",
            "phase": phase,
            "request_id": request,
            "sqlite": f"capture_{index}.sqlite",
        }
        for index, (phase, request) in enumerate((("cold", 0), ("revisit", 2)), 1)
    ]
    calls = [dict(item, stage="request") for item in metadata["captures"]]
    comparisons = [
        {key: row[key] for key in ("scheme", "request_id", "numerical")}
        for row in rows
        if row["scheme"] == "echo" and row["request_id"] <= 2
    ]
    finish_evidence(directory, metadata, calls, comparisons)
    for index, phase in enumerate(("cold", "revisit"), 1):
        _official_capture(directory / f"capture_{index}.sqlite", phase)
    return directory


def test_profile_audit_conserves_real_kernel_time_and_retains_unmatched_activity(profile_run):
    result = analyze(profile_run, profile_run / "report")
    assert result["status"] == "passed"
    for capture in result["captures"]:
        assert capture["kernel_count"] == 4
        assert capture["kernel_duration_sum_ms"] == pytest.approx(0.44)
        assert capture["kernel_count_conserved"] and capture["kernel_duration_conserved"]
        assert capture["unattributed_activity_count"] == 1
        assert capture["unattributed_activity_duration_sum_ms"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "damage", ["source", "correctness", "receipt", "empty_cases", "capture_request", "calls"]
)
def test_profile_rejects_changed_or_missing_saved_evidence(profile_run, damage):
    metadata = json.loads((profile_run / "metadata.json").read_text())
    if damage == "source":
        (profile_run / "source/example.py").write_text("changed")
    elif damage == "correctness":
        (profile_run / "correctness.json").unlink()
    elif damage == "receipt":
        (profile_run.parent / "receipt.json").write_text("changed")
    elif damage == "empty_cases":
        metadata["cases"] = []
        measure.write_json(profile_run / "metadata.json", metadata)
    elif damage == "capture_request":
        metadata["captures"][1]["request_id"] = 1
        measure.write_json(profile_run / "metadata.json", metadata)
    else:
        measure.write_json(profile_run / "calls.json", [])
    with pytest.raises((ValueError, FileNotFoundError)):
        analyze(profile_run, profile_run / "report")


def test_nsight_injection_is_recorded_separately_from_model_environment(tmp_path, monkeypatch):
    from experiments.deepseek_v32_echo_official.src import profile

    library = tmp_path / "libToolsInjection64.so"
    library.write_bytes(b"nsight injection fixture")
    observed = {"CUDA_VISIBLE_DEVICES": "3", "CUDA_INJECTION64_PATH": str(library)}
    monkeypatch.setattr(profile, "execution_environment", lambda: observed)
    model, effective, injection = profile_environment()
    assert model == {"CUDA_VISIBLE_DEVICES": "3"}
    assert effective == observed
    assert injection == {"path": str(library), "sha256": measure.digest(library)}
