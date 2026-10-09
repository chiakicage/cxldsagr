"""Private immutable loading of the unchanged generic ECHO translation unit."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from operators.deepseek_v32.indexer import _native_cache, echo

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "experiments/deepseek_v32_echo_official/output/runtime/q1-official-prefetch/tvm"


def build_spec():
    info = echo.build_info()
    name = "cxldsagr_echo_indexer_" + hashlib.sha256(repr(info).encode()).hexdigest()[:16]
    source = ROOT / "operators/deepseek_v32/indexer/csrc"
    request = {
        "name": name,
        "sources": [str(source / "echo_indexer.cu")],
        "extra_include_paths": [str(ROOT / "3rdparty/cutlass/include"), str(source)],
        "extra_cuda_cflags": info["cuda_flags"],
        "extra_ldflags": ["-lcuda"],
    }
    identity = {
        **info,
        "private_loader": "unchanged-generic-echo-immutable-v1",
        "source_sha256": {
            **info["source_sha256"],
            **info["shared_header_sha256"],
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    return request, identity


def native_info():
    return _native_cache.native_info(build_spec()[0]["name"])


@contextmanager
def pinned():
    """Intercept only the exact existing generic build call, preserving its API.

    Keeping the original echo._module function also preserves cache_ops' imported
    alias. No model, computation, cleanup, dispatch or production source changes.
    The immutable loader verifies source, headers, toolchain, ABI and ELF bytes.
    """
    import tvm_ffi.cpp

    original = tvm_ffi.cpp.load
    request, identity = build_spec()

    def load(*args, **kwargs):
        name = args[0] if args else kwargs.get("name", "")
        if not str(name).startswith("cxldsagr_echo_indexer_"):
            return original(*args, **kwargs)
        if args or kwargs != request:
            raise RuntimeError("Unexpected generic ECHO build request")
        if Path(os.environ.get("TVM_FFI_CACHE_DIR", "")).resolve() != RUNTIME:
            raise RuntimeError("Private generic ECHO build escaped experiment runtime")
        if os.environ.get("TVM_FFI_CUDA_ARCH_LIST") != "9.0a":
            raise RuntimeError("Generic ECHO build must preserve SM90a target")
        return _native_cache.load(**request, source_identity=identity)

    with patch.object(tvm_ffi.cpp, "load", load):
        yield
