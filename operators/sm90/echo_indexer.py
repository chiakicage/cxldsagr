"""Standalone ECHO inter-query prefetch on Hopper, without SGLang.

The index keys remain in HBM. Main-attention BF16 latent/RoPE records live in
mapped pinned host memory. While computing later query rows, the fused kernel
prefetches a provable subset of earlier rows' exact top-2048 into eligible HBM
slots. The caller still performs exact top-k and recalls every remaining miss.
"""

import hashlib
import os
import subprocess
from functools import cache
from pathlib import Path

import torch

UPSTREAM_REVISION = "bc1b75c1000010d0ac6f032ebaac283255c050b1"
_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = Path(__file__).resolve().parent / "csrc"
_FLAGS = [
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-lineinfo",
]


def build_info():
    """Record source and shared-header versions without initializing CUDA."""
    files = [Path(__file__), *_SOURCE.glob("echo_*.cu*")]
    return {
        "upstream": "https://github.com/sjtu-zhao-lab/ECHO",
        "upstream_revision": UPSTREAM_REVISION,
        "cutlass_revision": subprocess.check_output(
            ["git", "-C", str(_ROOT / "3rdparty/cutlass"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cuda_flags": _FLAGS,
        "source_sha256": {
            str(path.relative_to(_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)
        },
    }


@cache
def _module():
    import tvm_ffi.cpp

    include = _ROOT / "3rdparty/cutlass/include"
    if not (include / "cute/tensor.hpp").is_file():
        raise RuntimeError("Prepare shared CUTLASS with python scripts/prepare_3rdparty.py --init")
    info = build_info()
    digest = hashlib.sha256(repr(info).encode()).hexdigest()[:16]
    # TVM FFI otherwise adds a generic sm_90 target, which rejects WGMMA PTX.
    previous_arch = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    try:
        return tvm_ffi.cpp.load(
            name=f"cxldsagr_echo_indexer_{digest}",
            sources=[str(_SOURCE / "echo_indexer.cu")],
            extra_include_paths=[str(include), str(_SOURCE)],
            extra_cuda_cflags=_FLAGS,
            extra_ldflags=["-lcuda"],
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous_arch


def _check_tensor(tensor, shape, dtype, device, name):
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != tuple(shape)
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} must be contiguous {dtype} {tuple(shape)} on {device}")


def logits(q, k, weights, kscale, query_start, prefetch=None):
    """Return causal FP32 ``[Q,N]`` logits with a padded physical row stride.

    Q/K are E4M3 indexer values, ``weights`` includes the query quantization
    scales and model head weights, and ``kscale`` is the per-key FP32 scale.
    This model-specific implementation accepts 64 index heads of dimension 128.

    ``prefetch`` contains ``host`` pinned BF16 ``[capacity,576]``, ``device``
    BF16 ``[pool,576]``, ``host_to_device`` int32 ``[capacity]`` (INT_MAX misses),
    ``device_to_host`` int64 ``[pool]``, ``free_slots`` int32, ``counter`` uint32
    ``[1]``, and ``offset`` FP32 ``[16]``. The first ``max_prefetch`` free slots
    must be unique valid slot IDs, excluding all currently protected records.
    The optional limit defaults to ``min(8192, pool-Q, len(free_slots))``.

    The counter is reset here and counts reservation attempts; clamp it to the
    limit to obtain the copied-record count. No slot zero is reserved. The
    caller must serialize cache mutations and complete residual recall before
    attention consumes the logical-to-physical mapping.
    """
    if q.ndim != 3 or q.shape[1:] != (64, 128) or not q.is_cuda or len(q) < 1:
        raise ValueError("ECHO requires nonempty CUDA Q[query,64,128]")
    if k.ndim != 2 or k.shape[1] != 128 or len(k) < 1:
        raise ValueError("ECHO requires nonempty K[token,128]")
    rows, columns, device = len(q), len(k), q.device
    if type(query_start) is not int or not 0 <= query_start <= columns - rows:
        raise ValueError("query_start must place all query rows inside the visible KV sequence")
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise ValueError("ECHO's WGMMA prefetch kernel requires Hopper SM90")
    for tensor, shape, dtype, name in (
        (q, (rows, 64, 128), torch.float8_e4m3fn, "q"),
        (k, (columns, 128), torch.float8_e4m3fn, "k"),
        (weights, (rows, 64), torch.float32, "weights"),
        (kscale, (columns,), torch.float32, "kscale"),
    ):
        _check_tensor(tensor, shape, dtype, device, name)
    module = _module()
    starts = torch.zeros(rows, dtype=torch.int32, device=device)
    ends = torch.arange(query_start + 1, query_start + rows + 1, dtype=torch.int32, device=device)
    if columns % 4:
        scales = torch.nn.functional.pad(kscale, (0, 4 - columns % 4))
    else:
        scales = kscale
    output = torch.empty(((rows + 1) // 2 * 2, (columns + 127) // 128 * 128), device=device)
    placeholder = torch.empty(0, dtype=torch.int32, device=device)
    if prefetch is None:
        host = torch.empty(0)
        pool = host_to_device = device_to_host = slots = counter = offsets = placeholder
        page_table = torch.empty((0, 0), dtype=torch.int32, device=device)
        extend_lengths = query_requests = allocations = placeholder
        limit = 0
    else:
        host, pool = prefetch["host"], prefetch["device"]
        if host.ndim != 2 or host.shape[1] != 576 or len(host) < columns:
            raise ValueError("host must cover all visible 576-element MLA records")
        if pool.ndim != 2 or pool.shape[1] != 576 or len(pool) < rows:
            raise ValueError("device pool must have room for the current query chunk")
        _check_tensor(host, host.shape, torch.bfloat16, torch.device("cpu"), "host")
        if not host.is_pinned():
            raise ValueError("host MLA records must use pinned, CUDA-mapped storage")
        _check_tensor(pool, pool.shape, torch.bfloat16, device, "device")
        host_to_device, device_to_host = prefetch["host_to_device"], prefetch["device_to_host"]
        slots, counter, offsets = prefetch["free_slots"], prefetch["counter"], prefetch["offset"]
        for tensor, shape, dtype, name in (
            (host_to_device, (len(host),), torch.int32, "host_to_device"),
            (device_to_host, (len(pool),), torch.int64, "device_to_host"),
            (slots, (slots.numel(),), torch.int32, "free_slots"),
            (counter, (1,), torch.uint32, "counter"),
            (offsets, (16,), torch.float32, "offset"),
        ):
            _check_tensor(tensor, shape, dtype, device, name)
        limit = prefetch.get("max_prefetch", min(8192, len(pool) - rows, len(slots)))
        if type(limit) is not int or not 0 <= limit <= min(8192, len(slots)):
            raise ValueError("max_prefetch must fit the unique eligible free_slots prefix")
        counter.zero_()
        allocations = torch.empty_like(slots)
        page_table = torch.arange(columns, device=device, dtype=torch.int32).unsqueeze(0)
        extend_lengths = torch.tensor([rows], dtype=torch.int32, device=device)
        query_requests = torch.zeros(rows, dtype=torch.int32, device=device)
    import tvm_ffi

    with torch.cuda.device(device), tvm_ffi.use_torch_stream():
        module.echo_logits(
            q.view(torch.uint8),
            k.view(torch.uint8),
            weights,
            scales,
            starts,
            ends,
            output,
            page_table,
            extend_lengths,
            query_requests,
            host,
            pool,
            host_to_device,
            device_to_host,
            slots,
            allocations,
            counter,
            offsets,
            query_start,
            limit,
            prefetch is not None,
        )
    return output[:rows, :columns]
