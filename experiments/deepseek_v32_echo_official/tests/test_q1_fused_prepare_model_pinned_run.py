"""Pre-gate evidence and immutable native archival must survive failures."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_model_pinned_run as run


@pytest.fixture
def launch(tmp_path, monkeypatch):
    args = SimpleNamespace(mode="bench", pairs=500, output_dir=tmp_path)
    monkeypatch.setattr(run.model, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(run.model.component.raw, "environment", lambda: None)
    monkeypatch.setattr(run.flashinfer, "pinned", nullcontext)
    return tmp_path


def test_rejected_model_gate_saves_actual_identity_and_propagates(launch, monkeypatch):
    identity = {"source": {"checked": True}, "timing_runtime": {"actual": "native"}}
    failure = ValueError("identity differs")
    writes = []
    original = run.model.require_receipt
    monkeypatch.setattr(run, "_write", lambda path, value: writes.append((path, value)))

    def require(_path, *, kind, identity):
        if kind == run.model.KIND:
            raise failure
        return identity

    def main():
        assert run.model.require_receipt(
            launch / "component.json", kind=run.model.component.KIND, identity={"component": True}
        ) == {"component": True}
        assert writes == []
        run.model.require_receipt(launch / "receipt.json", kind=run.model.KIND, identity=identity)

    monkeypatch.setattr(run, "_require_receipt", require)
    monkeypatch.setattr(run.model, "main", main)
    with pytest.raises(ValueError) as error:
        run.main()
    assert error.value is failure
    assert run.model.require_receipt is original
    assert len(writes) == 1 and writes[0][0] == launch / "pre_gate_identity.json"
    assert writes[0][1]["identity"] is identity and writes[0][1]["pairs"] == 500


@pytest.mark.parametrize("fail", [False, True])
def test_artifacts_finish_before_result_declaration(launch, monkeypatch, fail):
    events = []
    original = run.model.write
    failure = OSError("archive failed")

    def archive(path):
        events.append(("archive", path))
        if fail:
            raise failure

    monkeypatch.setattr(run.flashinfer, "archive", archive)
    monkeypatch.setattr(run, "_write", lambda path, value: events.append(("write", path)))
    monkeypatch.setattr(
        run.model, "main", lambda: run.model.write(launch / "result.json", {"completed": True})
    )
    if fail:
        with pytest.raises(OSError) as error:
            run.main()
        assert error.value is failure
        assert events == [("archive", launch)]
    else:
        run.main()
        assert events == [("archive", launch), ("write", launch / "result.json")]
    assert run.model.write is original


def test_runtime_requires_all_three_pinned_libraries(monkeypatch):
    records = {
        name: {
            "library": {"path": name + ".so", "sha256": name},
            "build_metadata": {"path": name + ".ninja", "sha256": name},
            "build_identity": {
                "sources": [name + ".cu"],
                "request": {"extra_cuda_cflags": ["-O3"], "extra_cflags": ["-O3"]},
            },
        }
        for name in run.MODULES
    }
    rows = [
        {
            "name": name,
            "loaded_in_this_process": True,
            "library": record["library"],
            "build_metadata": record["build_metadata"],
            "sources": record["build_identity"]["sources"],
            "cuda_flags": ["-O3"],
            "cxx_flags": ["-O3"],
        }
        for name, record in records.items()
    ]
    original = {"model": {"loaded_jit": {"native_jit": rows}}}
    monkeypatch.setattr(run, "_runtime", lambda: dict(original))
    monkeypatch.setattr(run.flashinfer, "native_info", lambda: records)
    assert run.runtime() == {
        **original,
        "private_flashinfer_native": records,
    }
    rows[0]["loaded_in_this_process"] = False
    with pytest.raises(RuntimeError, match="collector"):
        run.runtime()
    rows[0]["loaded_in_this_process"] = True
    records.pop("topk")
    with pytest.raises(RuntimeError, match="Incomplete"):
        run.runtime()
