"""Independent source bridge to pinned official ECHO's Q1 decode prefetch.

This API owns no cache policy. The caller owns the temporary 64-record staging
buffer and must consume or clear its encoded mappings before another call.
It is not a replacement for the local prefill pool/journal ABI.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from functools import cache
from pathlib import Path

import torch

REVISION = "bc1b75c1000010d0ac6f032ebaac283255c050b1"
PREFETCH_CAP = 64
ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "3rdparty/ECHO"
INCLUDE = UPSTREAM / "DeepGEMM/deep_gemm/include"
SOURCE = Path(__file__).with_name("csrc") / "official_decode.cu"
CLEAN_SOURCE = SOURCE.with_name("official_decode_clean.cu")
FLAGS = [
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--ptxas-options=--register-usage-level=10",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-lineinfo",
]


def build_info():
    """Bind the unchanged upstream source and actual local include closure."""
    revision = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Official ECHO checkout must be pinned to {REVISION}")
    subprocess.run(
        ["git", "-C", str(UPSTREAM), "diff", "--exit-code", "HEAD", "--", "DeepGEMM"],
        check=True,
        capture_output=True,
    )
    files = {Path(__file__), Path(__file__).with_name("_native_cache.py"), SOURCE, CLEAN_SOURCE}
    pending = [
        INCLUDE / "deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh",
        INCLUDE / "deep_gemm/impls/smxx_clean_logits.cuh",
    ]
    while pending:
        path = pending.pop()
        if path in files:
            continue
        files.add(path)
        for delim, include in re.findall(r'#include\s*([<"])([^>"]+)[>"]', path.read_text()):
            if include.startswith("deep_gemm/"):
                pending.append(INCLUDE / include)
            elif delim == '"':
                pending.append(path.parent / include)
    cutlass = ROOT / "3rdparty/cutlass/include"
    files.update(path for path in cutlass.rglob("*") if path.is_file())
    # The host-side launch formula is transcribed from these original files.
    files.update(
        UPSTREAM / relative
        for relative in (
            "DeepGEMM/csrc/apis/attention.hpp",
            "DeepGEMM/csrc/jit_kernels/impls/smxx_fp8_paged_mqa_logits.hpp",
            "DeepGEMM/csrc/jit_kernels/impls/smxx_clean_logits.hpp",
            "DeepGEMM/csrc/jit_kernels/impls/runtime_utils.hpp",
            "DeepGEMM/csrc/jit_kernels/heuristics/sm90.hpp",
            "DeepGEMM/csrc/jit_kernels/heuristics/sm100.hpp",
        )
    )
    return {
        "upstream_revision": revision,
        "cutlass_revision": subprocess.check_output(
            ["git", "-C", str(ROOT / "3rdparty/cutlass"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cuda_flags": FLAGS,
        "source_sha256": {
            str(path.resolve().relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)
        },
        "policy": "Official strict FP32 predictive threshold; 64 temporary records per query",
        "scope": "Independent Q1/H64/D128/DMLA576/page64 SM90 source bridge",
    }


@cache
def _module(num_sms):
    from operators.deepseek_v32.indexer import _native_cache

    info = build_info()
    digest = hashlib.sha256(repr(info).encode()).hexdigest()[:16]
    previous_arch = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    try:
        return _native_cache.load(
            name=f"cxldsagr_official_echo_decode_{num_sms}_{digest}",
            sources=[str(SOURCE), str(CLEAN_SOURCE)],
            extra_include_paths=[str(INCLUDE), str(ROOT / "3rdparty/cutlass/include")],
            extra_cuda_cflags=[*FLAGS, f"-DCXL_ECHO_NUM_SMS={num_sms}"],
            extra_ldflags=["-lcuda"],
            source_identity=info,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous_arch


def native_info(num_sms):
    from operators.deepseek_v32.indexer import _native_cache

    digest = hashlib.sha256(repr(build_info()).encode()).hexdigest()[:16]
    return _native_cache.native_info(f"cxldsagr_official_echo_decode_{num_sms}_{digest}")


def _check(tensor, shape, dtype, device, name):
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != tuple(shape)
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} must be contiguous {dtype} {tuple(shape)} on {device}")


def _device_module(tensor):
    if tensor.device.type != "cuda" or torch.cuda.get_device_capability(tensor.device) != (9, 0):
        raise ValueError("The official source bridge requires SM90")
    sms = torch.cuda.get_device_properties(tensor.device).multi_processor_count
    return _module(sms), sms


def metadata(context_lens):
    """Run the official one-query split-KV scheduler on current lengths."""
    import tvm_ffi

    _check(context_lens, (1,), torch.int32, context_lens.device, "context_lens")
    module, sms = _device_module(context_lens)
    result = torch.empty((sms + 1, 2), device=context_lens.device, dtype=torch.int32)
    with torch.cuda.device(context_lens.device), tvm_ffi.use_torch_stream():
        module.official_decode_metadata(context_lens, result)
    return result


def logits(
    q,
    packed,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    page_table_1,
    device_pool,
    host_pool,
    prefetch_host_loc,
    prefetch_kv,
    host_to_device,
    query_counter,
    decode_topk_logits,
):
    """Invoke the unchanged official kernel and causal cleaning kernel.

    Caller preconditions: page_table_1 contains unique valid host IDs for all
    initialized history positions [0, context_len-1); context_len is positive
    and fits packed/max_context_len; block_table references allocated pages.
    Missing host mappings equal INT32_MAX. query_counter starts at zero, and
    staging IDs start at -1. Host IDs may include zero in this isolated API.
    The final history-excluded token must be the only pending query token.

    Mapping values >= len(device_pool) encode temporary slots. They are not
    persistent pool indices or readiness signals. Stream completion and owner
    lifetime are the caller's responsibility.
    """
    import tvm_ffi

    device = q.device
    module, sms = _device_module(q)
    if type(max_context_len) is not int or not 0 < max_context_len <= packed.shape[0] * 64:
        raise ValueError("max_context_len must fit the allocated packed pages")
    if packed.ndim != 2 or packed.shape[1] != 64 * 132:
        raise ValueError("packed must use the official page64 K-then-scales ABI")
    if host_pool.ndim != 2 or host_pool.shape[1] != 576 or not host_pool.is_pinned():
        raise ValueError("host_pool must contain pinned BF16 MLA records")
    if device_pool.ndim != 2 or device_pool.shape[1] != 576 or len(device_pool) < 2:
        raise ValueError("device_pool must contain a sentinel and allocatable MLA records")
    if len(host_pool) >= 2**31 - 1 or len(device_pool) + PREFETCH_CAP >= 2**31 - 1:
        raise ValueError("Host IDs and encoded temporary mappings must fit signed int32")
    if host_to_device.ndim != 1 or len(host_to_device) not in (len(host_pool), len(host_pool) + 1):
        raise ValueError("host_to_device must cover valid host IDs, with an optional sentinel")
    for name, tensor, shape, dtype in (
        ("q", q, (1, 1, 64, 128), torch.float8_e4m3fn),
        ("packed", packed, tuple(packed.shape), torch.uint8),
        ("weights", weights, (1, 64), torch.float32),
        ("context_lens", context_lens, (1,), torch.int32),
        ("block_table", block_table, (1, len(packed)), torch.int32),
        ("schedule_meta", schedule_meta, (sms + 1, 2), torch.int32),
        ("page_table_1", page_table_1, (1, max_context_len), torch.int32),
        ("device_pool", device_pool, tuple(device_pool.shape), torch.bfloat16),
        ("prefetch_host_loc", prefetch_host_loc, (1, PREFETCH_CAP), torch.int32),
        ("prefetch_kv", prefetch_kv, (1, PREFETCH_CAP, 576), torch.bfloat16),
        ("host_to_device", host_to_device, tuple(host_to_device.shape), torch.int32),
        ("query_counter", query_counter, (1,), torch.uint32),
        ("decode_topk_logits", decode_topk_logits, (1,), torch.float32),
    ):
        _check(tensor, shape, dtype, device, name)
    _check(host_pool, tuple(host_pool.shape), torch.bfloat16, torch.device("cpu"), "host_pool")
    width = (max_context_len + 255) // 256 * 256
    output = torch.empty((1, width), dtype=torch.float32, device=device)
    with torch.cuda.device(device), tvm_ffi.use_torch_stream():
        module.official_decode_forward(
            q,
            packed,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            output,
            page_table_1,
            device_pool,
            host_pool,
            prefetch_host_loc,
            prefetch_kv,
            host_to_device,
            query_counter,
            decode_topk_logits,
            max_context_len,
        )
    return output[:, :max_context_len]
