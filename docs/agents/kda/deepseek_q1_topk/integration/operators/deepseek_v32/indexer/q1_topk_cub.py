"""CUB ordering and invalid-index masking for Q1 FlashInfer SMALL selection."""

import hashlib
import os
from functools import cache
from pathlib import Path

from operators.deepseek_v32.indexer import _native_cache

SOURCE = Path(__file__).with_name("csrc") / "q1_topk_sort.cu"
NAME = "cxldsagr_q1_topk_sort_mask"
FLAGS = ["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"]


@cache
def build_info():
    """Bind the wrapper, loader and complete installed CCCL include closure."""
    import flashinfer

    package = Path(flashinfer.__file__).resolve().parent
    includes = [
        package / "data/cccl" / suffix for suffix in ("cub", "libcudacxx/include", "thrust")
    ]
    files = {Path(__file__), SOURCE, Path(_native_cache.__file__)}
    for root in includes:
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.update(path for path in root.rglob("*") if path.is_file())
    return {
        "policy": "q1-cub-uint64-composite-sort-mask-256x8-v1",
        "include_paths": [str(path) for path in includes],
        "source_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)
        },
    }


@cache
def _module():
    info = build_info()
    previous = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    try:
        return _native_cache.load(
            name=NAME,
            sources=[str(SOURCE)],
            extra_include_paths=info["include_paths"],
            extra_cuda_cflags=FLAGS,
            extra_ldflags=[],
            source_identity=info,
        )
    finally:
        if previous is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous


def runtime_info():
    """Inspect only this wrapper's already-loaded immutable native module."""
    return _native_cache.native_info(NAME) if _module.cache_info().currsize else None


def sort_mask_(values, indices):
    """Order fresh contiguous native outputs in place on the caller's stream."""
    import tvm_ffi

    module = _module()
    with tvm_ffi.use_torch_stream():
        module.q1_topk_sort_mask(values, indices)
