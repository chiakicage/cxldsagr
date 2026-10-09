"""Pin the unchanged common CUDA transfer for the private full-model harness."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_native as generic
from operators.common import kv_transfer
from operators.deepseek_v32.indexer import _native_cache

NAME = "cxldsagr_kv_transfer"


def build_spec():
    module = Path(kv_transfer.__file__).resolve()
    source = module.parent / "csrc/kv_transfer.cu"
    request = {
        "name": NAME,
        "sources": [str(source)],
        "extra_cuda_cflags": ["-O3", "-std=c++17", "-lineinfo"],
    }
    identity = {
        "private_loader": "unchanged-common-transfer-immutable-v1",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (module, source, Path(__file__).resolve())
        },
    }
    return request, identity


def native_info():
    return _native_cache.native_info(NAME)


@contextmanager
def pinned():
    """Intercept the exact common transfer build without changing its functions."""
    import tvm_ffi.cpp

    original = tvm_ffi.cpp.load
    request, identity = build_spec()

    def load(*args, **kwargs):
        name = args[0] if args else kwargs.get("name", "")
        if name != NAME:
            return original(*args, **kwargs)
        if args or kwargs != request:
            raise RuntimeError("Unexpected common transfer build request")
        if Path(os.environ.get("TVM_FFI_CACHE_DIR", "")).resolve() != generic.RUNTIME:
            raise RuntimeError("Private transfer build escaped experiment runtime")
        return _native_cache.load(
            **request, extra_include_paths=[], extra_ldflags=[], source_identity=identity
        )

    with patch.object(tvm_ffi.cpp, "load", load):
        yield
