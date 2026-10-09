"""Private CUB composite-key sort/mask after unchanged FlashInfer SMALL selection."""

import hashlib
import os
from functools import cache
from pathlib import Path

import torch

from operators.deepseek_v32.indexer import _native_cache
from operators.deepseek_v32.indexer.selection import _topk_backend

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("q1_topk_sort.cu")
NAME = "cxldsagr_q1_cub_sort_mask"
FLAGS = ["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"]


@cache
def build_info():
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
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)
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
    return _native_cache.native_info(NAME) if _module.cache_info().currsize else None


def exact_topk_cub(scores, k):
    """Match production values, logical IDs and nonfinite mask without tie relaxation."""
    import tvm_ffi

    if (
        not isinstance(scores, torch.Tensor)
        or scores.ndim != 2
        or scores.shape[1] < 1
        or scores.dtype != torch.float32
        or scores.requires_grad
    ):
        raise ValueError("scores must be inference FP32 [query, nonempty visible_tokens]")
    if type(k) is not int or not 1 <= k <= 2048:
        raise ValueError("k must be an integer in [1, 2048]")
    if not scores.is_cuda or torch.cuda.get_device_capability(scores.device) != (9, 0):
        raise NotImplementedError("DeepSeek FlashInfer selection requires Hopper SM90")
    count = min(k, scores.shape[1])
    if scores.shape[0] == 0:
        return scores.new_empty((0, count)), torch.empty(
            (0, count), device=scores.device, dtype=torch.int32
        )
    module = _module()
    with torch.cuda.device(scores.device):
        backend = _topk_backend()
        states = backend._get_cache_buf(
            f"radix_topk_row_states_{scores.device}", 1024 * 1024, scores.device, zero_init=True
        )
        values = torch.empty(scores.shape[0], count, dtype=scores.dtype, device=scores.device)
        indices = backend.get_topk_module().radix_topk(
            scores.contiguous(),
            count,
            False,
            False,
            backend.TopKTieBreak.SMALL,
            states,
            values,
            False,
        )
        with tvm_ffi.use_torch_stream():
            module.q1_topk_sort_mask(values, indices)
    return values, indices
