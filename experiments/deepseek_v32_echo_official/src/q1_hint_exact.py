"""Private fixed-shape mean candidate; the official ECHO and EMA stay unchanged."""

import hashlib
import os
from functools import cache
from pathlib import Path

import torch

from operators.deepseek_v32.indexer import _native_cache, prefetch_hint

SOURCE = Path(__file__).with_suffix(".cu")
NAME = "cxldsagr_q1_hint_exact_mean"
FLAGS = [
    "-O3",
    "-std=c++20",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-lineinfo",
    "--ftz=false",
    "--fmad=false",
]


def build_info():
    from tvm_ffi.cpp.extension import _find_cuda_home

    headers = Path(_find_cuda_home()).resolve() / "include"
    if not headers.is_dir():
        raise FileNotFoundError(headers)
    paths = {Path(__file__), SOURCE, Path(_native_cache.__file__)}
    paths.update(path for path in headers.rglob("*") if path.is_file())
    return {
        "policy": "private-q1-exact-torch-tree-65537-512x4-v1",
        "source_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
        },
        "flags": FLAGS,
    }


@cache
def module():
    before = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    try:
        return _native_cache.load(
            name=NAME,
            sources=[str(SOURCE)],
            extra_include_paths=[],
            extra_cuda_cflags=FLAGS,
            extra_ldflags=[],
            source_identity=build_info(),
        )
    finally:
        if before is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = before


def runtime_info():
    return _native_cache.native_info(NAME) if module.cache_info().currsize else None


def supported(scores, offset):
    return (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.shape == (1, 65537)
        and scores.stride(1) == 1
        and scores.data_ptr() % 16 == 0
        and offset.shape == (16,)
        and offset.dtype == torch.float32
        and offset.device == scores.device
        and offset.is_contiguous()
        and not torch.is_grad_enabled()
        and torch.cuda.get_device_capability(scores.device) == (9, 0)
    )


def update(scores, offset):
    if not supported(scores, offset):
        return prefetch_hint.update_prefetch_hint(scores, offset)
    import tvm_ffi

    native = module()
    with torch.cuda.device(scores.device), tvm_ffi.use_torch_stream():
        native.q1_hint_mean(scores, offset)
