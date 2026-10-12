"""Independent-sequence resident indexer adapters to official DeepGEMM.

Hopper's paged API supports one or two query tokens per sequence. This
adapter deliberately exposes Q1 decode; longer single-sequence extend uses
the official nonpaged API. Preparation and causal cleanup are separate calls
so an observation experiment can assign their time without double counting.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from operators.deepseek_v32.indexer.page64 import pack_q1_keys

_PACKED_WORDS_PER_PAGE = 2112
_SAFE_PACK_PAGES = 2**31 // _PACKED_WORDS_PER_PAGE


def _packing_page_ranges(total_pages, pages_per_sequence):
    """Bound existing page64 kernel's signed int32 local word offsets.

    The unchanged kernel uses page*2112 for its output and page*2048 for its
    key input. Rebase each contiguous slice before launching. Whole-sequence
    slices up to 256 sequences make boundaries easy to audit; exceptionally
    large single sequences may be divided at a page boundary instead.
    """
    pages_per_launch = min(_SAFE_PACK_PAGES, 256 * pages_per_sequence)
    if pages_per_sequence <= _SAFE_PACK_PAGES:
        pages_per_launch = pages_per_launch // pages_per_sequence * pages_per_sequence
    return tuple(
        (start, min(total_pages, start + pages_per_launch))
        for start in range(0, total_pages, pages_per_launch)
    )


def _pack_independent_keys(k, scales, pages_per_sequence):
    total_pages = k.shape[0] * pages_per_sequence
    if total_pages <= _SAFE_PACK_PAGES:
        return pack_q1_keys(k.view(-1, 128), scales.view(-1))
    from operators.deepseek_v32.indexer.page64 import _kernel

    # Allocate final storage once. Each existing kernel writes directly into
    # a disjoint view; there is no intermediate packing or concatenation copy.
    packed = torch.empty((total_pages, 8448), dtype=torch.uint8, device=k.device)
    flat_keys, flat_scales = k.view(-1, 128), scales.view(-1)
    kernel = _kernel()
    for start, stop in _packing_page_ranges(total_pages, pages_per_sequence):
        first_token, end_token = start * 64, stop * 64
        kernel[(stop - start,)](
            flat_keys[first_token:end_token].view(torch.int32),
            flat_scales[first_token:end_token].view(torch.int32),
            packed[start:stop].view(torch.int32),
            TOKENS=end_token - first_token,
            num_warps=4,
        )
    return packed


@dataclass(frozen=True)
class PreparedKeys:
    packed: torch.Tensor
    block_table: torch.Tensor
    batch_size: int
    logical_tokens: int


def _check_device(tensor):
    if not tensor.is_cuda or torch.cuda.get_device_capability(tensor.device) != (9, 0):
        raise NotImplementedError("Resident batched DeepSeek indexer requires Hopper SM90")


def prepare_keys(k, scales, *, logical_tokens=None):
    """Pack independent sequences into page64 and construct their page table.

    Inputs are FP8 ``[B,N,128]`` and FP32 ``[B,N]``. The optional logical
    length permits an already padded contiguous backing; page padding must
    be initialized by the caller and is excluded by each context length.
    Without that option this call zero pads each sequence independently.
    Packing, padding copies and the page table belong to this operation.
    """
    _check_device(k)
    if k.ndim != 3 or k.shape[2] != 128 or min(k.shape[:2]) < 1:
        raise ValueError("keys must be nonempty [B,N,128]")
    if (
        k.dtype != torch.float8_e4m3fn
        or scales.dtype != torch.float32
        or scales.device != k.device
        or tuple(scales.shape) != tuple(k.shape[:2])
        or k.requires_grad
        or scales.requires_grad
    ):
        raise ValueError("keys/scales must be inference FP8/FP32 tensors on the same device")
    batch, capacity, _ = k.shape
    logical = capacity if logical_tokens is None else logical_tokens
    if type(logical) is not int or not 1 <= logical <= capacity:
        raise ValueError("logical_tokens must lie within each sequence's initialized storage")
    pages = (logical + 63) // 64
    padded_tokens = pages * 64
    if logical_tokens is None and padded_tokens != capacity:
        # FP8 padding is expressed through its byte representation because
        # some Torch versions do not implement constant_pad_nd for FP8.
        k = F.pad(k.view(torch.uint8), (0, 0, 0, padded_tokens - capacity)).view(k.dtype)
        scales = F.pad(scales, (0, padded_tokens - capacity))
    elif capacity != padded_tokens:
        raise ValueError("explicit padded capacity must equal round_up(logical_tokens,64)")
    k, scales = k.contiguous(), scales.contiguous()
    with torch.cuda.device(k.device):
        packed = _pack_independent_keys(k, scales, pages)
        block_table = torch.arange(batch * pages, device=k.device, dtype=torch.int32).view(
            batch, pages
        )
    return PreparedKeys(packed.view(batch * pages, 64, 1, 132), block_table, batch, logical)


def schedule(context_lens):
    """Build official paged scheduler metadata for inference int32 [B,1]."""
    _check_device(context_lens)
    if (
        context_lens.ndim != 2
        or context_lens.shape[1] != 1
        or context_lens.shape[0] < 1
        or context_lens.dtype != torch.int32
        or not context_lens.is_contiguous()
    ):
        raise ValueError("context_lens must be contiguous int32 [B,1]")
    import deep_gemm

    with torch.cuda.device(context_lens.device):
        return deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, 64, deep_gemm.get_num_sms(), indices=None
        )


def paged_logits(q, prepared, weights, context_lens, metadata, *, score_width):
    """Score independent Q1 sequences; caller separately masks invalid tails.

    FP8 Q has shape ``[B,1,64,128]`` and weights are FP32 ``[B,64]``.
    Output ``[B,score_width]`` uses local token positions in each sequence.
    Native compressed output can contain uninitialized causal/page tails.
    It must pass through ``mask_causal_`` before exact selection.
    """
    _check_device(q)
    if (
        not isinstance(prepared, PreparedKeys)
        or tuple(q.shape) != (prepared.batch_size, 1, 64, 128)
        or q.dtype != torch.float8_e4m3fn
        or not q.is_contiguous()
        or tuple(weights.shape) != (prepared.batch_size, 64)
        or weights.dtype != torch.float32
        or tuple(context_lens.shape) != (prepared.batch_size, 1)
        or context_lens.dtype != torch.int32
    ):
        raise ValueError("paged Q1 shapes and FP8/FP32/int32 dtypes do not agree")
    if type(score_width) is not int or score_width < prepared.logical_tokens:
        raise ValueError("score_width must cover logical tokens")
    tensors = (prepared.packed, prepared.block_table, weights, context_lens, metadata)
    if any(t.device != q.device for t in tensors):
        raise ValueError("all paged indexer tensors must use the same device")
    import deep_gemm

    with torch.cuda.device(q.device):
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            q=(q, None),
            kv_cache=prepared.packed,
            weights=weights,
            context_lens=context_lens,
            block_table=prepared.block_table,
            schedule_meta=metadata,
            max_context_len=score_width,
            indices=None,
        )


def single_sequence_logits(q, k, weights, scales, starts, ends, *, score_width):
    """Call the official nonpaged score API; cleanup is a separate operation."""
    _check_device(q)
    if q.ndim != 3 or q.shape[1:] != (64, 128) or k.ndim != 2 or k.shape[1] != 128:
        raise ValueError("expected Q[Q,64,128] and K[N,128]")
    if len(k) % 4:
        scales = F.pad(scales, (0, 4 - len(k) % 4))
    import deep_gemm

    with torch.cuda.device(q.device):
        return deep_gemm.fp8_fp4_mqa_logits(
            (q, None),
            (k, scales[: len(k)]),
            weights,
            starts,
            ends,
            max_seqlen_k=score_width,
        )


def mask_causal_(scores, ends, *, history_length):
    """Initialize only causal and physical tails, preserving valid score bits."""
    from operators.deepseek_v32.indexer.echo import _resident_tail_mask_kernel

    rows, width = scores.shape
    if ends.shape != (rows,) or ends.dtype != torch.int32 or ends.device != scores.device:
        raise ValueError("ends must match score rows as int32 exclusive local causal endpoints")
    if type(history_length) is not int or not 0 <= history_length < width:
        raise ValueError("history_length must lie within score columns")
    tail = width - history_length
    _resident_tail_mask_kernel()[((rows * tail + 255) // 256,)](
        scores,
        ends,
        rows,
        tail,
        history_length,
        scores.stride(0),
        scores.stride(1),
        BLOCK=256,
        num_warps=4,
    )
    return scores
