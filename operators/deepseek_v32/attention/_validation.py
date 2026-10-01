"""Input contract shared by DeepSeek sparse MLA implementations."""

import math

import torch


def _validate(q, kv, indices, scale, value_dim):
    if q.ndim != 3 or kv.ndim != 2 or indices.ndim != 2:
        raise ValueError("Expected q [Q,H,D], kv [N,D], and indices [Q,K]")
    if q.shape[0] != indices.shape[0] or q.shape[-1] != kv.shape[-1]:
        raise ValueError("Query, cache, and selection dimensions do not agree")
    if not isinstance(value_dim, int) or not 1 <= value_dim <= q.shape[-1]:
        raise ValueError("value_dim must be an integer in [1, query dimension]")
    if value_dim > 512 or q.shape[-1] - value_dim > 128:
        raise ValueError("Supported dimensions are value_dim <= 512 and position dimension <= 128")
    if q.device != kv.device or q.device != indices.device:
        raise ValueError("q, kv, and indices must be on the same device")
    if q.dtype != kv.dtype or not q.is_floating_point():
        raise ValueError("q and kv must share a floating-point dtype")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be int32 or int64")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
