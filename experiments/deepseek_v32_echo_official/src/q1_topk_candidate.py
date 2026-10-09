"""Private Q1 candidate: unchanged SMALL selection and fused exact ordering/mask."""

from functools import cache

import torch

from operators.deepseek_v32.indexer.selection import _topk_backend


@cache
def _sort_mask_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sort_mask(Values, Indices, K: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        x = tl.arange(0, BLOCK)
        value = tl.load(Values + row * K + x, x < K, -float("inf"))
        index = tl.load(Indices + row * K + x, x < K, -1).to(tl.uint32)
        bits = value.to(tl.uint32, bitcast=True)
        ordered = bits ^ tl.where(bits & 0x80000000 != 0, 0xFFFFFFFF, 0x80000000)
        key = ((~ordered).to(tl.uint64) << 32) | index.to(tl.uint64)
        key = tl.where(x < K, key, 0xFFFFFFFFFFFFFFFF)
        key = tl.sort(key, descending=False)
        ordered = ~(key >> 32).to(tl.uint32)
        bits = ordered ^ tl.where(ordered & 0x80000000 != 0, 0x80000000, 0xFFFFFFFF)
        value = bits.to(tl.float32, bitcast=True)
        index = key.to(tl.uint32).to(tl.int32)
        index = tl.where((bits & 0x7FFFFFFF) >= 0x7F800000, -1, index)
        tl.store(Values + row * K + x, value, x < K)
        tl.store(Indices + row * K + x, index, x < K)

    return sort_mask


def exact_topk_candidate(scores, k):
    """Match production exact_topk, retaining its deterministic SMALL membership."""
    import triton

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
    with torch.cuda.device(scores.device):
        backend = _topk_backend()
        states = backend._get_cache_buf(
            f"radix_topk_row_states_{scores.device}",
            1024 * 1024,
            scores.device,
            zero_init=True,
        )
        values = torch.empty(scores.shape[0], count, dtype=scores.dtype, device=scores.device)
        # SMALL keeps deterministic_selection=true in the vendor dispatcher.
        # These flags skip only the separate index and value sorting launches.
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
        block = triton.next_power_of_2(count)
        _sort_mask_kernel()[(len(scores),)](
            values, indices, count, block, num_warps=8 if block >= 512 else 4
        )
    return values, indices
