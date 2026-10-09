"""Private immutable build interception must preserve exact requests and errors."""

from __future__ import annotations

import pytest

from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_model_native as transfer
from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_native as native


@pytest.fixture
def loader(monkeypatch):
    import tvm_ffi.cpp

    request = {
        "name": "cxldsagr_echo_indexer_test",
        "sources": ["unchanged.cu"],
        "extra_include_paths": ["unchanged/include"],
        "extra_cuda_cflags": ["-lineinfo"],
        "extra_ldflags": ["-lcuda"],
    }
    identity = {"source_sha256": {"unchanged.cu": "bound"}}
    calls = []
    result = object()

    def original(*args, **kwargs):
        calls.append(("original", args, kwargs))
        return result

    def immutable(**kwargs):
        calls.append(("immutable", kwargs))
        return result

    monkeypatch.setattr(tvm_ffi.cpp, "load", original)
    monkeypatch.setattr(native, "build_spec", lambda: (request, identity))
    monkeypatch.setattr(native._native_cache, "load", immutable)
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(native.RUNTIME))
    monkeypatch.setenv("TVM_FFI_CUDA_ARCH_LIST", "9.0a")
    return tvm_ffi.cpp, request, identity, calls, result, original


def test_exact_request_and_unrelated_delegation(loader):
    cpp, request, identity, calls, result, original = loader
    unrelated = object()
    with native.pinned():
        assert cpp.load(**request) is result
        assert cpp.load("other_module", sources=unrelated) is result
    assert cpp.load is original
    assert calls == [
        ("immutable", {**request, "source_identity": identity}),
        ("original", ("other_module",), {"sources": unrelated}),
    ]
    assert calls[1][2]["sources"] is unrelated


@pytest.mark.parametrize("change", ["source", "name", "positional", "runtime", "arch"])
def test_unexpected_generic_request_fails_before_build(loader, monkeypatch, change):
    cpp, request, _identity, calls, _result, original = loader
    modified = dict(request)
    args = ()
    if change == "source":
        modified["sources"] = ["different.cu"]
    elif change == "name":
        modified["name"] = "cxldsagr_echo_indexer_different"
    elif change == "positional":
        args = (modified.pop("name"),)
    elif change == "runtime":
        monkeypatch.setenv("TVM_FFI_CACHE_DIR", "/root/.cache/tvm-ffi")
    else:
        monkeypatch.setenv("TVM_FFI_CUDA_ARCH_LIST", "9.0")
    with pytest.raises(RuntimeError), native.pinned():
        cpp.load(*args, **modified)
    assert cpp.load is original
    assert calls == []


def test_original_build_failure_propagates_and_restores(loader, monkeypatch):
    cpp, request, _identity, _calls, _result, original = loader
    failure = RuntimeError("build failed")

    def fail(**_kwargs):
        raise failure

    monkeypatch.setattr(native._native_cache, "load", fail)
    with pytest.raises(RuntimeError) as error, native.pinned():
        cpp.load(**request)
    assert error.value is failure
    assert cpp.load is original


def test_model_transfer_exact_binding_and_generic_delegation(loader, monkeypatch):
    cpp, _request, _identity, calls, result, original = loader
    request = {
        "name": transfer.NAME,
        "sources": ["unchanged-transfer.cu"],
        "extra_cuda_cflags": ["-lineinfo"],
    }
    identity = {"source_sha256": {"unchanged-transfer.cu": "bound"}}
    monkeypatch.setattr(transfer, "build_spec", lambda: (request, identity))
    with transfer.pinned():
        assert cpp.load(**request) is result
        assert cpp.load("cxldsagr_echo_indexer_example", sources=request["sources"]) is result
    assert cpp.load is original
    assert calls[0] == (
        "immutable",
        {**request, "extra_include_paths": [], "extra_ldflags": [], "source_identity": identity},
    )
    assert calls[1][0] == "original"
    assert calls[1][2]["sources"] is request["sources"]


@pytest.mark.parametrize("change", ["source", "runtime"])
def test_model_transfer_rejects_unbound_build(loader, monkeypatch, change):
    cpp, _request, _identity, calls, _result, original = loader
    request, identity = transfer.build_spec()
    monkeypatch.setattr(transfer, "build_spec", lambda: (request, identity))
    modified = dict(request)
    if change == "source":
        modified["sources"] = ["different.cu"]
    else:
        monkeypatch.setenv("TVM_FFI_CACHE_DIR", "/root/.cache/tvm-ffi")
    with pytest.raises(RuntimeError), transfer.pinned():
        cpp.load(**modified)
    assert calls == []
    assert cpp.load is original
