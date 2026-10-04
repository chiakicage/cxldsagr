"""Exact finite-mean hint for the next ECHO indexer/prefetch call."""

from functools import cache

import torch


def _reference_update(scores, offset):
    tail = scores[-4:]
    finite = torch.isfinite(tail)
    offset[0] = tail.masked_fill(~finite, 0).sum() / finite.sum().clamp_min(1)


@cache
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_count(Scores, Masked, Counts, n, stride, total, BLOCK: tl.constexpr):
        pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(Scores + (pos // n) * stride + pos % n, pos < total, 0)
        magnitude = value.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        finite = (pos < total) & (magnitude < 0x7F800000)
        tl.store(Masked + pos, tl.where(finite, value, 0.0), pos < total)
        tl.store(Counts + tl.program_id(0), tl.sum(finite.to(tl.int32), 0))

    @triton.jit
    def publish(Total, Counts, Offset, count, BLOCK: tl.constexpr):
        pos = tl.arange(0, BLOCK)
        finite = tl.sum(tl.load(Counts + pos, pos < count, 0), 0)
        denominator = tl.maximum(finite, 1).to(tl.float32)
        tl.store(Offset, tl.div_rn(tl.load(Total), denominator))

    return mask_count, publish


def update_prefetch_hint(scores, offset):
    """Write only offset[0], preserving the checked FP32 reduction order.

    The hint is the mean of finite scores in the last min(4,Q) query rows.
    A contiguous masked tensor retains the exact shape and torch.sum call of
    the checked expression. Integer partial counts are exact; publication uses
    round-to-nearest FP32 division. Inputs and offset[1:] remain unchanged.
    Scratch is temporary on the calling stream, with no retained allocation.
    """
    if (
        scores.ndim != 2
        or not all(scores.shape)
        or scores.dtype != torch.float32
        or offset.shape != (16,)
        or offset.dtype != torch.float32
        or offset.device != scores.device
        or not offset.is_contiguous()
    ):
        raise ValueError("expected nonempty FP32 scores[Q,N] and matching FP32 offset[16]")
    if (
        scores.device.type != "cuda"
        or torch.is_grad_enabled()
        or scores.stride(1) != 1
        or min(4, len(scores)) * scores.shape[1] >= 2**31
        or torch.cuda.get_device_capability(scores.device) != (9, 0)
    ):
        _reference_update(scores, offset)
        return
    tail = scores[-4:]
    count = (tail.numel() + 1023) // 1024
    mask_count, publish = _kernels()
    with torch.cuda.device(scores.device):
        masked = torch.empty(tail.shape, device=tail.device, dtype=tail.dtype)
        counts = torch.empty(count, device=tail.device, dtype=torch.int32)
        mask_count[(count,)](
            tail, masked, counts, tail.shape[1], tail.stride(0), tail.numel(), BLOCK=1024
        )
        total = masked.sum()
        publish[(1,)](
            total, counts, offset, count, BLOCK=1 << (count - 1).bit_length(), num_warps=4
        )
