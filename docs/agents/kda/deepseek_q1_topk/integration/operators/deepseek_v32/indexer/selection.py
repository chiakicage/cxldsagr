"""Exact selection from DeepSeek FP32 indexer logits using official FlashInfer."""

import importlib
from functools import cache

import torch


@cache
def _topk_backend():
    return importlib.import_module("flashinfer.topk")


def _official_topk_int32(scores, count, *, sorted_output=True, deterministic=True):
    """Call the same official sorted operation before its public int64 conversion."""
    backend = _topk_backend()
    row_states = backend._get_cache_buf(
        f"radix_topk_row_states_{scores.device}",
        1024 * 1024,
        scores.device,
        zero_init=True,
    )
    values = torch.empty(scores.shape[0], count, dtype=scores.dtype, device=scores.device)
    indices = backend.get_topk_module().radix_topk(
        scores,
        count,
        sorted_output,
        deterministic,
        backend.TopKTieBreak.SMALL,
        row_states,
        values,
        False,
    )
    return values, indices


@cache
def _nonfinite_index_mask_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_kernel(Values, Indices, count, BLOCK: tl.constexpr):
        position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(Values + position, position < count, 0)
        magnitude = value.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        # Preserve every value bit and every finite index. Only invalid slots
        # need a write; no temporary boolean tensor or index read is required.
        tl.store(Indices + position, -1, (position < count) & (magnitude >= 0x7F800000))

    return mask_kernel


def _mask_nonfinite_indices_(values, indices):
    count = values.numel()
    if count:
        _nonfinite_index_mask_kernel()[((count + 1023) // 1024,)](
            values, indices, count, BLOCK=1024, num_warps=4
        )


def exact_topk(scores, k):
    """Return descending values and int32 token IDs; nonfinite slots use -1.

    ``scores`` is FP32 ``[query, visible_tokens]`` on Hopper. The caller must
    have applied its causal mask, using -inf for invalid tokens. Model logits
    must be finite before masking; arbitrary NaN payloads are outside this
    contract. This function does not change logits or causal visibility.

    FlashInfer performs exact selection with deterministic ordering and its
    SMALL tie mode (smaller index among equal ordered floating-point keys).
    Signed zeros follow the upstream radix ordering. PyTorch does not promise
    a stable tie-index order, so the selected IDs/order may differ on ties;
    this is an explicit policy, not approximate or truncated selection.

    Q1 with 2048 selected entries and at least 32768 visible columns keeps the
    official SMALL selection and fuses ordering/masking with official CUB.
    Other shapes use FlashInfer's original sorted operation. Capacity is
    ``min(k, visible_tokens)``; layout conversion, ordering and invalid-slot
    masking are part of this operator call.
    """
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
        scores = scores.contiguous()
        if scores.shape[0] == 1 and count == 2048 and scores.shape[1] >= 32768:
            from operators.deepseek_v32.indexer.q1_topk_cub import sort_mask_

            # SMALL still forces deterministic Filtered selection. The flags
            # disable only the two upstream post-selection ordering kernels.
            values, indices = _official_topk_int32(
                scores, count, sorted_output=False, deterministic=False
            )
            sort_mask_(values, indices)
        else:
            values, indices = _official_topk_int32(scores, count)
            _mask_nonfinite_indices_(values, indices)
    return values, indices
