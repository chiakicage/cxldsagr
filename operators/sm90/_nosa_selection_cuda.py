"""Native stable 33/64 selection over the model's rounded pooled scores."""

import torch

from operators.sm90._native import load_module


def _uses_shared_ranking(workspace, positions, query_start, rows, pooled_cis):
    first = query_start // 64
    return (
        positions is None
        and rows >= 128
        and workspace.dtype == torch.bfloat16
        and pooled_cis is not None
        and pooled_cis.dtype == torch.bfloat16
        and workspace.shape[1] <= 1056
        and len(pooled_cis) >= max(first, workspace.shape[1] - 2)
        and first >= 64
        and (query_start + rows - 1) // 64 - first <= 16
    )


def _ranking_workspace(workspace, positions, query_start, rows, heads, pooled_cis):
    """Allocate shared CIS candidates only for the proven causal prefix path."""
    if _uses_shared_ranking(workspace, positions, query_start, rows, pooled_cis):
        return torch.empty((heads, 64), dtype=torch.int32, device=workspace.device)
    return workspace


def _validate_prepared_ranking(ranking, workspace, positions, query_start, rows, heads, pool):
    if not _uses_shared_ranking(workspace, positions, query_start, rows, pool):
        raise ValueError("Prepared CIS ranking requires the native contiguous prefix path")
    if (
        not isinstance(ranking, torch.Tensor)
        or ranking.shape != (heads, 64)
        or ranking.dtype != torch.int32
        or ranking.device != workspace.device
        or not ranking.is_contiguous()
        or ranking.requires_grad
        or ranking.data_ptr() % 4
    ):
        raise ValueError("Prepared CIS ranking requires contiguous int32 [head, 64] storage")
    left, right = ranking.data_ptr(), workspace.data_ptr()
    if (
        left < right + workspace.numel() * workspace.element_size()
        and right < left + ranking.numel() * 4
    ):
        raise ValueError("Prepared CIS ranking must not overlap the score workspace")


def select_pooled_blocks(
    workspace,
    cis,
    positions,
    query_start,
    rows,
    heads,
    *,
    pooled_cis=None,
    return_valid_mask=False,
    prepared_ranking=None,
):
    import tvm_ffi

    ids = torch.empty((rows, heads, 64), dtype=torch.int64, device=workspace.device)
    valid = torch.empty_like(ids, dtype=torch.bool) if return_valid_mask else None
    if rows:
        ranking = prepared_ranking
        if ranking is None:
            ranking = _ranking_workspace(workspace, positions, query_start, rows, heads, pooled_cis)
        else:
            _validate_prepared_ranking(
                ranking, workspace, positions, query_start, rows, heads, pooled_cis
            )
        module = load_module("nosa_selection")
        with torch.cuda.device(workspace.device), tvm_ffi.use_torch_stream():
            if prepared_ranking is not None:
                module.select_ranked(
                    workspace,
                    cis,
                    pooled_cis,
                    workspace,
                    ids,
                    valid if valid is not None else workspace,
                    ranking,
                    query_start,
                    return_valid_mask,
                )
                return (ids, valid) if return_valid_mask else ids
            module.select(
                workspace,
                cis,
                pooled_cis if pooled_cis is not None else workspace,
                positions if positions is not None else workspace,
                ids,
                valid if valid is not None else workspace,
                ranking,
                query_start,
                pooled_cis is not None,
                positions is None,
                return_valid_mask,
            )
    return (ids, valid) if return_valid_mask else ids
