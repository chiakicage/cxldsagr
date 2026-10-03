"""Fused finite-input checks for resident CUDA NOSA Q/K/CIS tensors."""

import torch
import triton
import triton.language as tl


@triton.jit
def _finite_partials(
    Q,
    K,
    CIS,
    PARTIAL,
    Q_NUMEL,
    K_NUMEL,
    C_NUMEL,
    Q_BLOCKS,
    K_BLOCKS,
    C_BLOCKS,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    B: tl.constexpr,
):
    program = tl.program_id(0)
    lane = tl.arange(0, B).to(tl.int64)
    passed = tl.full((), True, tl.int1)
    # Every CTA scans one tensor. Logical indexing skips storage holes and
    # supports strided projection views without materializing a contiguous copy.
    if program < Q_BLOCKS:
        for tile in range(tl.cdiv(Q_NUMEL, tl.maximum(Q_BLOCKS, 1) * B)):
            index = (tile * Q_BLOCKS + program) * B + lane
            offset = (
                index // (Q_HEADS * Q_DIM) * QS0
                + index // Q_DIM % Q_HEADS * QS1
                + index % Q_DIM * QS2
            )
            values = tl.load(Q + offset, index < Q_NUMEL, other=0).to(tl.float32)
            passed &= tl.min((tl.abs(values) < float("inf")).to(tl.int32), 0) != 0
    elif program < Q_BLOCKS + K_BLOCKS:
        for tile in range(tl.cdiv(K_NUMEL, tl.maximum(K_BLOCKS, 1) * B)):
            index = (tile * K_BLOCKS + program - Q_BLOCKS) * B + lane
            offset = (
                index // (K_HEADS * K_DIM) * KS0
                + index // K_DIM % K_HEADS * KS1
                + index % K_DIM * KS2
            )
            values = tl.load(K + offset, index < K_NUMEL, other=0).to(tl.float32)
            passed &= tl.min((tl.abs(values) < float("inf")).to(tl.int32), 0) != 0
    else:
        for tile in range(tl.cdiv(C_NUMEL, tl.maximum(C_BLOCKS, 1) * B)):
            index = (tile * C_BLOCKS + program - Q_BLOCKS - K_BLOCKS) * B + lane
            offset = index // K_HEADS * CS0 + index % K_HEADS * CS1
            values = tl.load(CIS + offset, index < C_NUMEL, other=0).to(tl.float32)
            passed &= tl.min((tl.abs(values) < float("inf")).to(tl.int32), 0) != 0
    # Ordered comparison rejects both infinities and every NaN bit pattern;
    # finite extrema and subnormals remain accepted after an exact FP32 cast.
    tl.store(PARTIAL + program, passed)


@triton.jit
def _finite_reduce(PARTIAL, OUT, COUNT, B: tl.constexpr):
    index = tl.arange(0, B)
    values = tl.load(PARTIAL + index, index < COUNT, other=1).to(tl.int32)
    tl.store(OUT, tl.min(values, 0))


def all_finite(q, keys, cis):
    """Return a CUDA scalar bool after scanning Q, resident K, and resident CIS.

    This leaves host synchronization to the caller. Two kernels avoid
    materializing elementwise masks and preserve all input tensors and strides.
    Q is nonempty; K/CIS may be empty when their managed prefix was already
    validated. K is ``[token, head, dimension]`` and CIS is ``[token, KV head]``.
    """
    if q.ndim != 3 or keys.ndim != 3 or cis.shape != keys.shape[:2]:
        raise ValueError("Expected Q/K [token, head, dimension] and CIS [token, KV head]")
    if q.numel() == 0 or any(size <= 0 for size in keys.shape[1:]):
        raise ValueError("NOSA finite checks require nonempty Q and positive K head dimensions")
    if any(not t.is_cuda or t.device != q.device for t in (q, keys, cis)):
        raise ValueError("NOSA finite checks require CUDA tensors on one device")
    if any(t.dtype not in (torch.float16, torch.bfloat16, torch.float32) for t in (q, keys, cis)):
        raise ValueError("NOSA finite checks support BF16, FP16 and FP32 tensors")
    block = 4096
    # Bound the final reduction even for a full 256K-token query batch.
    counts = [min(1024, triton.cdiv(t.numel(), block)) for t in (q, keys, cis)]
    count = sum(counts)
    partial = torch.empty((count,), dtype=torch.uint8, device=q.device)
    result = torch.empty((), dtype=torch.bool, device=q.device)
    with torch.cuda.device(q.device):
        _finite_partials[(count,)](
            q,
            keys,
            cis,
            partial,
            q.numel(),
            keys.numel(),
            cis.numel(),
            counts[0],
            counts[1],
            counts[2],
            q.shape[1],
            keys.shape[1],
            q.shape[2],
            keys.shape[2],
            *q.stride(),
            *keys.stride(),
            *cis.stride(),
            block,
            num_warps=4,
        )
        _finite_reduce[(1,)](partial, result, count, triton.next_power_of_2(count), num_warps=4)
    return result
