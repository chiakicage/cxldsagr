"""Explicit PyTorch FP32 reference for NOSA block sparse attention."""

import torch

from layers.attention import BlockSelection
from operators.nosa.attention.common import _validate_inputs


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
