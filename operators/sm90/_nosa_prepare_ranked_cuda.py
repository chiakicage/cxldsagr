"""Explicit append preparation with a separately owned stable CIS ranking."""

import torch

from operators.sm90._native import load_module
from operators.sm90._nosa_prepare_cuda import _validate, supports


def supports_ranked(query, keys, cis, *, query_start, validated_start, pooled_start):
    """The original v4 causal guard plus at most one new prefix pool.

    Only contiguous appends with a validated prefix enter this path. Read-only
    short prefixes and a cold derived cache use the ordinary preparation path.
    """
    return supports(query, keys, cis) and _geometry_supported(
        query,
        keys,
        query_start=query_start,
        validated_start=validated_start,
        pooled_start=pooled_start,
    )


def _geometry_supported(query, keys, *, query_start, validated_start, pooled_start):
    if any(type(value) is not int for value in (query_start, validated_start, pooled_start)):
        return False
    first = query_start // 64
    blocks = (len(keys) + 63) // 64
    stable = max(0, (len(keys) - 16) // 64)
    return (
        query.dtype == torch.bfloat16
        and keys.shape[2] == 128
        and len(query) >= 128
        and query_start == validated_start
        and query_start + len(query) <= len(keys)
        and first >= 64
        and (query_start + len(query) - 1) // 64 - first <= 16
        and blocks <= 1056
        and stable >= max(first, blocks - 2)
        and 0 <= pooled_start <= stable
        and first - pooled_start <= 1
    )


def prepare_ranked_out(
    query,
    keys,
    cis,
    compressed_keys,
    compressed_cis,
    pooled_cis,
    ranking,
    *,
    query_start,
    validated_start,
    compressed_start,
    pooled_start,
    scratch,
):
    """Return a CUDA finite flag and write packed prefix candidates on success.

    ``ranking`` is int32 [KV head, 64] storage that must remain alive until the
    selector consumes it. No cache metadata is changed. On false, all derived
    buffers and ranking retain their previous bytes. This call is asynchronous
    and CUDA Graph compatible; the owner checks the flag before publishing.
    """
    import tvm_ffi

    _validate(
        query,
        keys,
        cis,
        compressed_keys,
        compressed_cis,
        pooled_cis,
        scratch,
        validated_start,
        compressed_start,
        pooled_start,
    )
    # _validate has already checked generic preparation support. Repeating the
    # tensor/device/stride predicate here needlessly lengthens the host path.
    if not _geometry_supported(
        query,
        keys,
        query_start=query_start,
        validated_start=validated_start,
        pooled_start=pooled_start,
    ):
        raise ValueError("Unsupported native ranked preparation; use ordinary preparation")
    if (
        ranking.shape != (keys.shape[1], 64)
        or ranking.dtype != torch.int32
        or ranking.device != query.device
        or not ranking.is_contiguous()
        or ranking.requires_grad
        or ranking.data_ptr() % 4
    ):
        raise ValueError("Ranked preparation requires separate contiguous int32 [head, 64] storage")
    module = load_module("nosa_prepare_ranked")
    with torch.cuda.device(query.device), tvm_ffi.use_torch_stream():
        module.prepare_ranked_out(
            query,
            keys,
            cis,
            compressed_keys,
            compressed_cis,
            pooled_cis,
            ranking,
            scratch.partial,
            scratch.finite,
            validated_start,
            compressed_start,
            pooled_start,
            query_start,
        )
    return scratch.finite
