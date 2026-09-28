"""Resident NOSA block attention and its explicit PyTorch FP32 reference.

Logical selections are consumed directly; CUDA loads the selected 64-token
tiles from NHD K/V without constructing query-sized gathered K/V tensors.
This resident operator does not implement offloaded cache access or fetching.
"""

import torch

from layers.attention import BlockSelection


def _validate_inputs(q, keys, values, selection, query_start, cis_bias):
    if q.ndim != 3 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("Expected Q=[queries,Q heads,dim] and K/V=[tokens,KV heads,dim]")
    queries, query_heads, dimension = q.shape
    tokens, kv_heads, key_dimension = keys.shape
    if (
        dimension == 0
        or key_dimension != dimension
        or kv_heads == 0
        or query_heads == 0
        or query_heads % kv_heads
    ):
        raise ValueError("K/V dimensions and GQA head grouping must match Q")
    if isinstance(query_start, bool) or not isinstance(query_start, int) or query_start < 0:
        raise ValueError("query_start must be a nonnegative integer")
    if query_start + queries > tokens:
        raise ValueError("Resident K/V must include every query position")
    if not isinstance(selection, BlockSelection) or selection.block_size != 64:
        raise ValueError("NOSA attention requires a 64-token BlockSelection")
    ids = selection.block_ids
    if (
        ids.ndim != 3
        or ids.shape[0] not in (1, queries)
        or ids.shape[1] not in (1, kv_heads)
        or not 1 <= ids.shape[2] <= 64
        or ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Block IDs must be integral [queries or 1,KV heads or 1,1..64]")
    mask = selection.valid_mask
    if mask is not None and (mask.shape != ids.shape or mask.dtype != torch.bool):
        raise ValueError("Selection validity must be boolean with the same shape as block IDs")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("NOSA attention requires FP16, BF16 or FP32 Q/K/V")
    if keys.dtype != q.dtype or values.dtype != q.dtype:
        raise ValueError("Q, K and V must share a dtype")
    if cis_bias is not None and (
        cis_bias.shape != (tokens, kv_heads)
        or cis_bias.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError("CIS bias must be floating point [tokens,KV heads]")
    tensors = [q, keys, values, ids]
    if mask is not None:
        tensors.append(mask)
    if cis_bias is not None:
        tensors.append(cis_bias)
    if any(t.device != q.device for t in tensors):
        raise ValueError("NOSA attention tensors must share one device")
    if any(t.requires_grad for t in tensors):
        raise ValueError("NOSA block attention is inference-only")


@torch.inference_mode()
def reference_nosa_block_sparse_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    selection: BlockSelection,
    query_start: int = 0,
    cis_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP32 reference for softmax(QK/sqrt(D) + CIS) V over logical blocks.

    CIS is an already computed additive logit bias, shared by a KV head's GQA
    query heads. False validity, negative IDs, out-of-range blocks and future
    tokens are masked. Empty rows return zero. Valid block IDs must be unique
    per selection row, as guaranteed by the NOSA indexer.

    Supports CPU and CUDA for correctness checks; this is not the fused GPU
    backend. The result has Q's shape and dtype.
    """
    _validate_inputs(q, keys, values, selection, query_start, cis_bias)
    old_precision = None
    if q.is_cuda:
        old_precision = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        with torch.autocast(device_type=q.device.type, enabled=False):
            return _reference_attention(q, keys, values, selection, query_start, cis_bias)
    finally:
        if old_precision is not None:
            torch.backends.cuda.matmul.fp32_precision = old_precision


def _reference_attention(q, keys, values, selection, query_start, cis_bias):
    queries, query_heads, dimension = q.shape
    tokens, kv_heads, _ = keys.shape
    groups = query_heads // kv_heads
    output = torch.zeros_like(q)
    offsets = torch.arange(64, device=q.device)
    for row in range(queries):
        selection_row = 0 if selection.block_ids.shape[0] == 1 else row
        for head in range(kv_heads):
            selection_head = 0 if selection.block_ids.shape[1] == 1 else head
            ids = selection.block_ids[selection_row, selection_head]
            valid = (ids >= 0) & (ids < (tokens + 63) // 64)
            if selection.valid_mask is not None:
                valid &= selection.valid_mask[selection_row, selection_head]
            selected = ids[valid].long()[:, None] * 64 + offsets
            selected = selected[(selected < tokens) & (selected <= query_start + row)]
            if selected.numel() == 0:
                continue
            query = q[row, head * groups : (head + 1) * groups].float()
            logits = query @ keys[selected, head].float().T * (dimension**-0.5)
            if cis_bias is not None:
                logits += cis_bias[selected, head].float()[None, :]
            # Bias can deliberately suppress all selected tokens with -inf.
            nonempty = ~torch.isneginf(logits).all(dim=-1, keepdim=True)
            probabilities = torch.where(nonempty, logits, 0.0).softmax(dim=-1)
            probabilities = torch.where(nonempty, probabilities, 0.0)
            result = probabilities @ values[selected, head].float()
            output[row, head * groups : (head + 1) * groups] = result.to(q.dtype)
    return output


def nosa_block_sparse_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    selection: BlockSelection,
    query_start: int = 0,
    cis_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused inference-only SM90 implementation of the reference operation.

    Q/K/V use BF16 or FP16 with D=64/128 and at most 32 Q heads per KV head.
    Tensor Cores compute QK and AV; online softmax and accumulators are FP32.
    Probabilities are rounded to Q's dtype for AV, so results are not bitwise
    identical to the FP32 reference. K/V remain resident and are read in place.

    The native CuTe implementation specializes D=128 and GQA=16 with TMA
    aligned pointers and row/head strides. Four adjacent queries share the
    union of their selected blocks while retaining independent causal masks,
    selection membership and softmax state. Packages with little overlap or
    nonfinite values requiring separate masking use the per-query kernel.
    Other supported shapes/layouts use Triton. CXLDSAGR_SM90_BACKEND=triton
    selects that implementation explicitly.
    """
    _validate_inputs(q, keys, values, selection, query_start, cis_bias)
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise NotImplementedError("The fused NOSA block attention backend requires SM90/Hopper")
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or q.shape[-1] not in (64, 128)
        or q.shape[1] // keys.shape[1] > 32
    ):
        raise ValueError("Fused NOSA requires BF16/FP16, D=64/128 and GQA groups <=32")
    if any(t.stride(-1) != 1 for t in (q, keys, values)):
        raise ValueError("Fused NOSA requires contiguous Q/K/V innermost dimensions")
    from operators.sm90._native import native_enabled

    if native_enabled():
        from operators.sm90 import _nosa_attention_cuda as native_attention

        if native_attention.supports_native_attention(q, keys, values):
            return native_attention.launch_nosa_block_attention(
                q, keys, values, selection, query_start, cis_bias
            )
    # CPU-only users of the mathematical reference do not import Triton.
    from operators.sm90._nosa_attention_triton import launch_nosa_block_attention

    return launch_nosa_block_attention(q, keys, values, selection, query_start, cis_bias)
