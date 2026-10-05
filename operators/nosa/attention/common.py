"""Tensor contracts shared by NOSA attention implementations."""

import torch

from models.attention_contracts import BlockSelection


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
