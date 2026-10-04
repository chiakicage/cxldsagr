"""Official resident indexer and ECHO inter-query prefetch on Hopper.

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
_ROOT = Path(__file__).resolve().parents[3]
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
    files = [Path(__file__), Path(__file__).with_name("cache_ops.py"), *_SOURCE.glob("echo_*.cu*")]
    include = _ROOT / "3rdparty/cutlass/include"
    headers = sorted(path for path in include.rglob("*") if path.is_file())
    return {
        "upstream": "https://github.com/sjtu-zhao-lab/ECHO",
        "upstream_revision": UPSTREAM_REVISION,
        "cutlass_revision": subprocess.check_output(
            ["git", "-C", str(_ROOT / "3rdparty/cutlass"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cuda_flags": _FLAGS,
        "shared_header_sha256": {
            str(path.relative_to(_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in headers
        },
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


def _validate_prefetch(prefetch, rows, columns, device):
    """Validate lease tensor ABI without synchronizing tensor values to CPU."""
    host, pool = prefetch["host"], prefetch["device"]
    if host.ndim != 2 or host.shape[1] != 576 or len(host) % 64:
        raise ValueError("host must contain whole 64-token pages of 576-element MLA records")
    if pool.ndim != 2 or pool.shape[1] != 576 or len(pool) <= rows:
        raise ValueError("device pool needs Q <= P allocatable slots plus sentinel slot zero")
    _check_tensor(host, host.shape, torch.bfloat16, torch.device("cpu"), "host")
    if not host.is_pinned():
        raise ValueError("host MLA records must use pinned, CUDA-mapped storage")
    _check_tensor(pool, pool.shape, torch.bfloat16, device, "device")
    if len(host) >= 2**31 - 1:
        raise ValueError("global host IDs must fit signed int32 below the MISSING sentinel")
    page_table = prefetch["page_table"]
    history = prefetch.get("history_length", columns)
    if type(history) is not int or not 0 <= history <= columns:
        raise ValueError("prefetch history must lie within the indexer-visible context")
    if page_table.ndim != 1 or page_table.numel() < (history + 63) // 64:
        raise ValueError("page_table must cover initialized host history pages")
    for key, shape, dtype in (
        ("host_to_device", (len(host),), torch.int32),
        ("device_to_host", (len(pool),), torch.int64),
        ("free_slots", (len(pool) - 1,), torch.int32),
        ("allocation_log", (len(pool),), torch.int64),
        ("prefetch_stats", (3,), torch.int64),
        ("counter", (1,), torch.uint32),
        ("offset", (16,), torch.float32),
        ("page_table", tuple(page_table.shape), torch.int32),
    ):
        _check_tensor(prefetch[key], shape, dtype, device, key)
    headroom = len(pool) - 1 - (0 if prefetch.get("transient_suffix", False) else rows)
    limit = prefetch.get("max_prefetch", min(8192, headroom))
    if type(limit) is not int or not 0 <= limit <= min(8192, headroom):
        raise ValueError("max_prefetch must satisfy min(8192, P-Q), excluding sentinel slot zero")


def logits(q, k, weights, kscale, query_start, prefetch=None):
    """Return causal FP32 ``[Q,N]`` logits with a padded physical row stride.

    Q/K are E4M3 indexer values, ``weights`` includes the query quantization
    scales and model head weights, and ``kscale`` is the per-key FP32 scale.
    This model-specific implementation accepts 64 index heads of dimension 128.
    Resident execution calls official ``deep_gemm.fp8_fp4_mqa_logits``; only
    the offload path loads the local fused ECHO prefetch extension.

    ``prefetch`` is an exclusive shared-pool lease. It supplies pinned host
    records, physical ``device[P+1,576]`` (padding slot 0), live global maps,
    ``page_table`` of global 64-token page numbers, a priority-sorted
    ``free_slots[P]`` permutation, reusable ``allocation_log[P+1]`` int64,
    ``counter[1]`` uint32, ``prefetch_stats[3]`` int64, and ``offset[16]`` FP32.
    Log entries are global host IDs indexed by physical slot; MISSING is INT_MAX.

    Preparation must wait for all host writes read by this invocation. The
    initialized ``history_length`` excludes the pending main-KV suffix.
    ``max_prefetch`` is capped at ``min(8192,P-Q)``; Q>P is rejected. Allocation
    claims evict only actual victims. Stats contain actual copied records,
    actual evictions and rejected reservations; the attempt counter may
    overshoot. Finalize and all consumers must follow kernel completion on
    the owning stream. The caller supplies and accounts for reusable scratch.
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
    starts = torch.zeros(rows, dtype=torch.int32, device=device)
    ends = torch.arange(query_start + 1, query_start + rows + 1, dtype=torch.int32, device=device)
    if columns % 4:
        scales = torch.nn.functional.pad(kscale, (0, 4 - columns % 4))
    else:
        scales = kscale
    if prefetch is None:
        import deep_gemm

        # The official API requires the logical N shape. Its SM90 TMA scale
        # descriptor rounds N to four floats, so retain padded backing storage.
        with torch.cuda.device(device):
            result = deep_gemm.fp8_fp4_mqa_logits(
                (q, None),
                (k, scales[:columns]),
                weights,
                starts,
                ends,
                max_seqlen_k=columns,
            )
            # Mainline exposes compressed logits without initialized tails.
            # All row starts are zero, so these are logical token positions;
            # explicitly mask each exclusive endpoint before exact top-k.
            if query_start + 1 < columns:
                positions = torch.arange(query_start, columns, device=device)
                tail = result if query_start == 0 else result[:, query_start:]
                tail.masked_fill_(positions[None, :] >= ends[:, None], -torch.inf)
            return result

    _validate_prefetch(prefetch, rows, columns, device)
    module = _module()
    output = torch.empty(
        ((rows + 1) // 2 * 2, (columns + 127) // 128 * 128),
        device=device,
        dtype=torch.float32,
    )
    host, pool = prefetch["host"], prefetch["device"]
    host_to_device, device_to_host = prefetch["host_to_device"], prefetch["device_to_host"]
    slots, counter, offsets = prefetch["free_slots"], prefetch["counter"], prefetch["offset"]
    allocations, stats = prefetch["allocation_log"], prefetch["prefetch_stats"]
    headroom = len(pool) - 1 - (0 if prefetch.get("transient_suffix", False) else rows)
    limit = prefetch.get("max_prefetch", min(8192, headroom))
    history_length = prefetch.get("history_length", query_start)
    if type(history_length) is not int or not 0 <= history_length <= query_start:
        raise ValueError("history_length must cover only initialized history before query_start")
    counter.zero_()
    allocations.fill_(2**31 - 1)
    stats.zero_()
    page_table = prefetch["page_table"]
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
            stats,
            offsets,
            query_start,
            history_length,
            limit,
        )
    return output[:rows, :columns]
