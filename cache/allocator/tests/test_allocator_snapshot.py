"""CPU loader/guard contracts; no native compilation or CUDA calls in this file."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cache.allocator import budget
from cache.allocator import snapshot as adapter


@pytest.fixture(autouse=True)
def reset_provider(monkeypatch):
    monkeypatch.setattr(adapter, "_provider", None)
    monkeypatch.setattr(adapter, "_runtime", None)


def test_cpu_validation_does_not_prepare_or_inspect_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU must not prepare native snapshot or query CUDA")

    monkeypatch.setattr(adapter, "_load_native", forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", forbidden)
    monkeypatch.setattr(torch._C, "_cuda_memorySnapshot", forbidden)
    assert budget.validate_allocator("cpu")["configuration_key"] == budget.CPU_POLICY
    assert adapter.runtime_info() is None


def test_provider_is_prepared_once_but_snapshots_remain_fresh_without_global_patch(monkeypatch):
    official = torch._C._cuda_memorySnapshot
    calls, setups = [], []

    def snapshot():
        calls.append(len(calls))
        return {"allocator_settings": {"generation": len(calls)}, "segments": []}

    def load():
        setups.append(True)
        return snapshot, {"backend": "private_cpp", "fingerprint": "fixture"}

    monkeypatch.setattr(adapter, "_load_native", load)
    with ThreadPoolExecutor(max_workers=8) as threads:
        results = list(threads.map(lambda _: adapter.snapshot(), range(32)))
    assert len(setups) == 1 and len(calls) == 32
    assert sorted(row[0]["allocator_settings"]["generation"] for row in results) == list(
        range(1, 33)
    )
    assert torch._C._cuda_memorySnapshot is official
    results[0][1]["backend"] = "changed"
    assert adapter.runtime_info()["backend"] == "private_cpp"


def test_unavailable_build_propagates_without_publishing_or_calling_official(monkeypatch):
    setups = []
    failure = FileNotFoundError("compiler unavailable")

    def unavailable():
        setups.append(True)
        raise failure

    def official(arguments):
        pytest.fail("failed native setup must not select another provider")

    monkeypatch.setattr(adapter, "_load_native", unavailable)
    monkeypatch.setattr(torch._C, "_cuda_memorySnapshot", official)
    with pytest.raises(FileNotFoundError) as caught:
        adapter.snapshot()
    assert caught.value is failure
    assert len(setups) == 1
    assert adapter.runtime_info() is None and adapter._provider is None


def test_runtime_snapshot_errors_propagate_without_silent_official_retry(monkeypatch):
    def fail():
        raise RuntimeError("snapshot state failed")

    monkeypatch.setattr(adapter, "_load_native", lambda: (fail, {"backend": "private_cpp"}))
    monkeypatch.setattr(
        torch._C,
        "_cuda_memorySnapshot",
        lambda *args: pytest.fail("must not hide snapshot runtime failure"),
    )
    with pytest.raises(RuntimeError, match="snapshot state"):
        adapter.snapshot()


@pytest.fixture
def mock_build(tmp_path, monkeypatch):
    header = tmp_path / "included.h"
    header.write_text("original dependency")
    identity = {
        "extension_suffix": ".so",
        "source_and_dependency_sha256": {str(header): adapter._digest(header)},
    }
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def fresh_identity():
        current = deepcopy(identity)
        current["source_and_dependency_sha256"][str(header)] = adapter._digest(header)
        return current, ["fake-compiler"], {}

    monkeypatch.setattr(adapter, "_build_identity", fresh_identity)
    builds, imports = [], []

    def compile_(command, environment):
        assert environment == {} and command[-2] == "-o"
        builds.append(command)
        Path(command[-1]).write_bytes(b"verified fake extension")
        return ""

    class Loader:
        def exec_module(self, module):
            path = Path(module.__file__)
            manifest = json.loads((path.parent / "manifest.json").read_text())
            assert manifest["binary_sha256"] == adapter._digest(path)
            imports.append(path)

    monkeypatch.setattr(adapter, "_run", compile_)
    monkeypatch.setattr(
        adapter.importlib.util,
        "spec_from_file_location",
        lambda name, path: SimpleNamespace(loader=Loader(), path=path),
    )
    monkeypatch.setattr(
        adapter.importlib.util,
        "module_from_spec",
        lambda spec: SimpleNamespace(__file__=str(spec.path), snapshot=lambda: {"segments": []}),
    )
    return identity, header, builds, imports, tmp_path


def test_locked_cache_publishes_complete_artifact_once_for_concurrent_loads(mock_build):
    identity, _, builds, imports, _ = mock_build
    with ThreadPoolExecutor(max_workers=4) as threads:
        results = list(threads.map(lambda _: adapter._load_native(), range(4)))
    assert len(builds) == 1 and len(imports) == 4
    assert len(set(imports)) == 1
    for provider, runtime in results:
        assert provider() == {"segments": []}
        assert runtime["fingerprint"] == adapter._fingerprint(identity)
        assert runtime["loaded_binary_sha256"] == adapter._digest(runtime["loaded_binary_path"])


def test_compiler_failure_does_not_publish_partial_cache(mock_build, monkeypatch):
    identity, _, _, imports, root = mock_build

    def fail(command, environment):
        Path(command[-1]).write_bytes(b"partial")
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(adapter, "_run", fail)
    with pytest.raises(RuntimeError, match="compiler failed"):
        adapter._load_native()
    cache = root / "cache/cxldsagr/allocator-snapshot"
    assert not (cache / adapter._fingerprint(identity)).exists()
    assert not imports
    assert list(cache.glob(".*")) == []


def test_changed_dependency_rejects_build_before_import(mock_build, monkeypatch):
    identity, header, _, imports, root = mock_build

    def changing(command, environment):
        Path(command[-1]).write_bytes(b"extension")
        header.write_text("changed during compile")
        return ""

    monkeypatch.setattr(adapter, "_run", changing)
    with pytest.raises(RuntimeError, match="dependency changed"):
        adapter._load_native()
    assert not imports
    assert not (
        root / "cache/cxldsagr/allocator-snapshot" / adapter._fingerprint(identity)
    ).exists()


def test_new_dependency_rejects_publication_even_when_known_file_hashes_match(
    mock_build, monkeypatch
):
    identity, _, _, imports, root = mock_build
    calls = []

    def changed_closure():
        current = deepcopy(identity)
        if calls:
            current["source_and_dependency_sha256"]["newly_included.h"] = "f" * 64
        calls.append(True)
        return current, ["fake-compiler"], {}

    monkeypatch.setattr(adapter, "_build_identity", changed_closure)
    with pytest.raises(RuntimeError, match="dependency changed"):
        adapter._load_native()
    assert len(calls) == 2 and not imports
    assert not (
        root / "cache/cxldsagr/allocator-snapshot" / adapter._fingerprint(identity)
    ).exists()


def test_cached_artifact_rechecks_full_identity_after_waiting_for_lock(mock_build, monkeypatch):
    identity, _, builds, imports, _ = mock_build
    _, runtime = adapter._load_native()
    binary = Path(runtime["loaded_binary_path"])
    original = binary.read_bytes()
    calls = []

    def changed_toolchain():
        current = deepcopy(identity)
        if calls:
            current["compiler_tools"] = {"ld": "/different/linker"}
        calls.append(True)
        return current, ["fake-compiler"], {}

    monkeypatch.setattr(adapter, "_build_identity", changed_toolchain)
    with pytest.raises(RuntimeError, match="dependency changed"):
        adapter._load_native()
    assert len(calls) == 2 and len(builds) == len(imports) == 1
    assert binary.read_bytes() == original


def test_corrupt_existing_artifact_is_preserved_and_failure_propagates(mock_build, monkeypatch):
    identity, _, builds, _, _ = mock_build
    _, runtime = adapter._load_native()
    binary = Path(runtime["loaded_binary_path"])
    binary.write_bytes(b"corrupt artifact must not be replaced")
    monkeypatch.setattr(torch._C, "_cuda_memorySnapshot", lambda args: pytest.fail("fallback"))
    with pytest.raises(RuntimeError, match="cache integrity failed"):
        adapter.snapshot()
    assert adapter.runtime_info() is None
    assert binary.read_bytes() == b"corrupt artifact must not be replaced"
    assert len(builds) == 1
    assert (binary.parent / "manifest.json").exists()
    assert adapter._fingerprint(identity) == binary.parent.name


def test_build_identity_covers_local_torch_cuda_headers_compiler_tools_abis_and_libc10(
    tmp_path, monkeypatch
):
    root = tmp_path / "model"
    root.mkdir()
    loader = root / "snapshot.py"
    loader.write_text("loader")
    source = root / "csrc/allocator_snapshot.cpp"
    source.parent.mkdir()
    source.write_text("source")
    torch_root = tmp_path / "torch"
    (torch_root / "lib").mkdir(parents=True)
    (torch_root / "include").mkdir()
    paths = [
        root / "local.h",
        torch_root / "include/torch.h",
        tmp_path / "cuda/include/cuda.h",
        torch_root / "lib/libc10.so",
        torch_root / "lib/libc10_cuda.so",
    ]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    compiler = bin_dir / "g++"
    tools = {name: bin_dir / name for name in ("cc1plus", "as", "ld", "collect2")}
    runtimes = {
        name: bin_dir / name
        for name in ("libstdc++.so", "libgcc_s.so.1", "crtbeginS.o", "crtendS.o")
    }
    for path in [compiler, *paths, *tools.values(), *runtimes.values()]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
        path.chmod(0o755)
    monkeypatch.setattr(adapter, "__file__", str(loader))
    monkeypatch.setattr(torch, "__file__", str(torch_root / "__init__.py"))
    monkeypatch.setattr(adapter, "_unsupported_reason", lambda: None)
    monkeypatch.setenv("CXX", str(compiler))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setattr(
        torch.cuda, "is_initialized", lambda: pytest.fail("identity discovery must not query CUDA")
    )

    def inspect(command, environment):
        assert set(environment) == {"PATH", "LC_ALL"}
        if "-M" in command:
            return "dependencies: " + " ".join(str(p) for p in paths[:3])
        flag = command[-1]
        if flag.startswith("-print-prog-name="):
            return str(tools[flag.partition("=")[2]])
        if flag.startswith("-print-file-name="):
            return str(runtimes[flag.partition("=")[2]])
        assert flag == "--version"
        return "fixture GNU compiler"

    monkeypatch.setattr(adapter, "_run", inspect)
    first, command, _ = adapter._build_identity()
    hashed = first["source_and_dependency_sha256"]
    assert all(
        str(p.resolve()) in hashed
        for p in [loader, source, compiler, *paths, *tools.values(), *runtimes.values()]
    )
    assert "torch_cxx11_abi" in first and first["extension_suffix"]
    assert str(torch_root / "lib/libc10_cuda.so") in command
    for path in (paths[0], paths[1], paths[2], paths[-1], tools["ld"]):
        previous = adapter._fingerprint(first)
        path.write_text(path.read_text() + " changed")
        first, _, _ = adapter._build_identity()
        assert adapter._fingerprint(first) != previous
