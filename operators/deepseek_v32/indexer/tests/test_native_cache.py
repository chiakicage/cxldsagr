"""Exact native artifact reuse and failed-build publication boundaries."""

import hashlib
from pathlib import Path

import pytest

from operators.deepseek_v32.indexer import _native_cache as cache


@pytest.fixture
def build_case(tmp_path, monkeypatch):
    import tvm_ffi
    import tvm_ffi.cpp

    source = tmp_path / "input.cu"
    source.write_bytes(b"source-v1")
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_environment_identity", lambda: {"compiler": "fixture"})
    builds, loads = [], []

    def build(**kwargs):
        builds.append(kwargs)
        Path(kwargs["output"]).write_bytes(b"ELF-fixture")

    def load(path):
        loads.append(path)
        return path

    monkeypatch.setattr(tvm_ffi.cpp, "build", build)
    monkeypatch.setattr(tvm_ffi, "load_module", load)
    arguments = {
        "name": "test_native_cache",
        "sources": [str(source)],
        "extra_include_paths": [],
        "extra_cuda_cflags": ["-O3"],
        "extra_ldflags": [],
        "source_identity": {
            "source_sha256": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
        },
    }
    return arguments, builds, loads, source


def test_second_load_reuses_exact_artifact_without_building(build_case):
    arguments, builds, loads, _ = build_case
    first = cache.load(**arguments)
    before = cache.native_info(arguments["name"])
    second = cache.load(**arguments)
    assert first == second and loads == [first, first] and len(builds) == 1
    assert cache.native_info(arguments["name"]) == before


@pytest.mark.parametrize("corruption", ["artifact", "record", "missing"])
def test_invalid_existing_entry_fails_without_rebuilding(build_case, corruption):
    arguments, builds, _, _ = build_case
    path = Path(cache.load(**arguments))
    if corruption == "artifact":
        path.write_bytes(b"different ELF")
    elif corruption == "record":
        (path.parent / "record.json").write_text("{}")
    else:
        path.unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        cache.load(**arguments)
    assert len(builds) == 1


def test_source_change_during_build_is_never_published(build_case, monkeypatch):
    import tvm_ffi.cpp

    arguments, _, _, source = build_case

    def changed(**kwargs):
        Path(kwargs["output"]).write_bytes(b"unknown-source ELF")
        source.write_bytes(b"source-v2")

    monkeypatch.setattr(tvm_ffi.cpp, "build", changed)
    with pytest.raises(RuntimeError, match="source changed during build"):
        cache.load(**arguments)
    assert not list(source.parent.rglob("record.json"))


def test_build_and_cleanup_failures_preserve_both_exception_objects(build_case, monkeypatch):
    import tvm_ffi.cpp

    arguments, _, _, _ = build_case
    primary, cleanup = RuntimeError("build failed"), OSError("cleanup failed")

    def fail_build(**kwargs):
        raise primary

    def fail_cleanup(*args):
        raise cleanup

    monkeypatch.setattr(tvm_ffi.cpp, "build", fail_build)
    monkeypatch.setattr(cache.shutil, "rmtree", fail_cleanup)
    with pytest.raises(BaseExceptionGroup) as result:
        cache.load(**arguments)
    assert result.value.exceptions == (primary, cleanup)
