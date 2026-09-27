"""Append complete NOSA compression windows and immutable CIS pools on Hopper."""

import torch
import triton
import triton.language as tl


@triton.jit
def _append_compressed(
    K,
    CIS,
    CK,
    CC,
    POOL,
    FIRST_BLOCK,
    C_OLD,
    C_NEW,
    S_OLD,
    S_NEW,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    BD: tl.constexpr,
):
    block = FIRST_BLOCK + tl.program_id(0)
    head = tl.program_id(1)
    token = tl.arange(0, 32)
    feature = tl.arange(0, BD)
    make_pool = (block >= S_OLD) & (block < S_NEW)
    pooled = tl.full((), -float("inf"), tl.float32)
    # The owner of block b writes compressed windows 4b .. 4b+3 exactly once.
    # A new left CIS halo is recomputed from raw input, never read from another
    # CTA's pending output. Previously committed windows are immutable.
    for within in tl.static_range(4):
        window = block * 4 + within
        if (window >= C_OLD) & (window < C_NEW):
            row = window * 16 + token
            values = tl.load(
                K + row[:, None] * KS0 + head * KS1 + feature[None, :],
                feature[None, :] < DIM,
                other=0,
            ).to(tl.float32)
            mean = (tl.sum(values, 0) * (1.0 / 32)).to(CK.dtype.element_ty)
            tl.store(CK + (window * HEADS + head) * DIM + feature, mean, feature < DIM)
            values_cis = tl.load(CIS + row * CS0 + head * CS1).to(tl.float32)
            mean_cis = (tl.sum(values_cis, 0) * (1.0 / 32)).to(CC.dtype.element_ty)
            tl.store(CC + window * HEADS + head, mean_cis)
            pooled = tl.maximum(pooled, mean_cis.to(tl.float32))
        elif make_pool & (window < C_NEW):
            mean_cis = tl.load(CC + window * HEADS + head).to(tl.float32)
            pooled = tl.maximum(pooled, mean_cis)
    if make_pool:
        left = block * 4 - 1
        if left >= C_OLD:
            row = left * 16 + token
            values_cis = tl.load(CIS + row * CS0 + head * CS1).to(tl.float32)
            mean_cis = (tl.sum(values_cis, 0) * (1.0 / 32)).to(CC.dtype.element_ty)
            pooled = tl.maximum(pooled, mean_cis.to(tl.float32))
        elif left >= 0:
            pooled = tl.maximum(pooled, tl.load(CC + left * HEADS + head).to(tl.float32))
        tl.store(POOL + block * HEADS + head, pooled)


def update_compressed_cache(
    keys,
    cis,
    compressed_keys,
    compressed_cis,
    pooled_cis,
    *,
    compressed_start,
    pooled_start,
):
    """Write only newly complete windows and newly stable five-window pools.

    Output tensors contain capacity, not just the currently valid prefix.
    Inputs cover the visible token prefix. A pool for logical block b becomes
    stable at length 64*b+80. The caller publishes valid lengths only after
    successful request commit; suffix bytes can be overwritten after abort.
    """
    if keys.ndim != 3 or cis.shape != keys.shape[:2]:
        raise ValueError("Expected K [token, head, dimension] and CIS [token, head]")
    length, heads, dim = keys.shape
    if not 0 <= length <= 262144 or heads <= 0 or not 1 <= dim <= 256:
        raise ValueError("Unsupported NOSA compression shape or length (maximum 262144)")
    tensors = (keys, cis, compressed_keys, compressed_cis, pooled_cis)
    if any(not t.is_cuda or t.device != keys.device or t.requires_grad for t in tensors):
        raise ValueError("NOSA compression requires CUDA inference tensors on one device")
    if torch.cuda.get_device_capability(keys.device)[0] != 9:
        raise RuntimeError("SM90 NOSA compression requires a Hopper GPU")
    if keys.dtype not in (torch.bfloat16, torch.float16, torch.float32) or any(
        t.dtype != keys.dtype for t in tensors
    ):
        raise ValueError("NOSA compression requires matching BF16/FP16/FP32 dtypes")
    if keys.stride(-1) != 1 or any(not t.is_contiguous() for t in tensors[2:]):
        raise ValueError("NOSA compression requires contiguous features and output buffers")
    count = max(0, length // 16 - 1)
    stable = max(0, (length - 16) // 64)
    if (
        compressed_keys.ndim != 3
        or compressed_keys.shape[1:] != (heads, dim)
        or compressed_keys.shape[0] < count
        or compressed_cis.ndim != 2
        or compressed_cis.shape[1] != heads
        or compressed_cis.shape[0] < count
        or pooled_cis.ndim != 2
        or pooled_cis.shape[1] != heads
        or pooled_cis.shape[0] < stable
    ):
        raise ValueError("NOSA compression output capacity or shape does not match inputs")
    for name, old, end in (
        ("compressed_start", compressed_start, count),
        ("pooled_start", pooled_start, stable),
    ):
        if type(old) is not int or not 0 <= old <= end:
            raise ValueError(f"{name} must be within the visible complete prefix")
    if compressed_start == count and pooled_start == stable:
        return
    first = min(compressed_start // 4, pooled_start)
    stop = max(triton.cdiv(count, 4), stable)
    with torch.cuda.device(keys.device):
        _append_compressed[(stop - first, heads)](
            keys,
            cis,
            compressed_keys,
            compressed_cis,
            pooled_cis,
            first,
            compressed_start,
            count,
            pooled_start,
            stable,
            heads,
            dim,
            *keys.stride()[:2],
            *cis.stride(),
            triton.next_power_of_2(dim),
            num_warps=4,
        )
