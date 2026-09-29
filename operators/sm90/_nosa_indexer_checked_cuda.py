"""Checked ranked prepare/select with explicit request-owned outputs and scratch."""

import torch

from operators.sm90._native import load_module
from operators.sm90._nosa_prepare_cuda import PreparationScratch


def select_prepared_out(
    query,
    keys,
    cis,
    compressed_keys,
    compressed_cis,
    pooled_cis,
    workspace,
    normalizers,
    block_ids,
    valid_mask,
    ranking,
    *,
    query_start,
    validated_start,
    compressed_start,
    pooled_start,
    scratch,
):
    """Return selected outputs after a successful finite check on this stream.

    This endpoint accepts the owned BF16/D128/GQA16 ranked native joint path.
    Other geometry retains the original preparation and selection calls.
    Native code validates every buffer before launching. Validation scratch
    may change; invalid Q/new K/new CIS leaves all other output bytes intact.
    A pinned host bool is mandatory, and CUDA Graph capture is rejected before
    launching any work. Async prepare APIs retain their capture semantics.
    """
    import tvm_ffi

    if not isinstance(scratch, PreparationScratch) or (
        scratch.host_finite is None
        or scratch.host_finite.shape != ()
        or scratch.host_finite.dtype != torch.bool
        or not scratch.host_finite.is_pinned()
    ):
        raise ValueError("Checked ranked indexer requires a reusable pinned host bool")
    if any(
        type(value) is not int
        for value in (query_start, validated_start, compressed_start, pooled_start)
    ):
        raise ValueError("Checked ranked indexer bounds must be integers")
    module = load_module("nosa_indexer_checked")
    with torch.cuda.device(query.device), tvm_ffi.use_torch_stream():
        finite = module.checked_ranked_indexer_out(
            query,
            keys,
            cis,
            compressed_keys,
            compressed_cis,
            pooled_cis,
            workspace,
            normalizers,
            block_ids,
            valid_mask,
            ranking,
            scratch.partial,
            scratch.finite,
            scratch.host_finite,
            validated_start,
            compressed_start,
            pooled_start,
            query_start,
        )
    if not finite:
        raise ValueError("NOSA indexer requires finite Q, K and CIS scores")
    return block_ids, valid_mask
