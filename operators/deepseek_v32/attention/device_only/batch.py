"""BF16 resident sparse MLA for independent sequences without offload."""

import torch

from operators.deepseek_v32.attention.device_only.mla import sparse_mla, sparse_mla_decode


def global_indices(local_indices, tokens_per_sequence):
    """Translate local selected slots, preserving padding and invalid int64 IDs."""
    if local_indices.ndim != 3 or local_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("local selection must be int32/int64 [B,Q,K]")
    if type(tokens_per_sequence) is not int or tokens_per_sequence < 1:
        raise ValueError("tokens_per_sequence must be positive")
    batch = local_indices.shape[0]
    if batch * tokens_per_sequence > torch.iinfo(torch.int32).max:
        raise ValueError("flattened resident KV rows must fit int32")
    valid = (local_indices >= 0) & (local_indices < tokens_per_sequence)
    offsets = torch.arange(batch, device=local_indices.device, dtype=torch.int64)
    translated = local_indices.to(torch.int64) + offsets[:, None, None] * tokens_per_sequence
    return torch.where(valid, translated, -1).to(torch.int32)


def sparse_mla_batch(q, kv, local_indices, scale, value_dim=512):
    """Consume BF16 Q[B,Q,heads,576], KV[B,N,576], and local IDs[B,Q,K].

    Each sequence has independent full resident KV storage. Local IDs are
    translated to disjoint flattened spans before one official sparse-prefill
    invocation. Original B1/Q1 uses the existing explicit split16 decode path
    when selection width is 2048. ID conversion and contiguous copies belong
    to this adapter's measured cost. This is a single-layer microbenchmark
    primitive, not a model serving lifecycle or full-model capacity claim.
    """
    if q.ndim != 4 or kv.ndim != 3 or local_indices.ndim != 3:
        raise ValueError("expected Q[B,Q,H,D], KV[B,N,D], and IDs[B,Q,K]")
    batch, queries = q.shape[:2]
    if (
        batch < 1
        or queries < 1
        or kv.shape[0] != batch
        or tuple(local_indices.shape[:2]) != (batch, queries)
    ):
        raise ValueError("batch/query dimensions must be positive and agree")
    if q.device != kv.device or q.device != local_indices.device:
        raise ValueError("queries, resident KV and local IDs must share a device")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("resident batch MLA uses BF16 query and main cache")
    q, kv = q.contiguous(), kv.contiguous()
    flat_q = q.view(batch * queries, *q.shape[2:])
    flat_kv = kv.view(batch * kv.shape[1], kv.shape[2])
    if batch == queries == 1 and local_indices.shape[2] == 2048:
        output = sparse_mla_decode(flat_q, flat_kv, local_indices[0], scale, value_dim)
    else:
        translated = global_indices(local_indices, kv.shape[1])
        output = sparse_mla(flat_q, flat_kv, translated.view(batch * queries, -1), scale, value_dim)
    return output.view(batch, queries, q.shape[2], value_dim)
