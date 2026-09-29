"""Joint native submission of NOSA scoring and exact two-stage selection."""

import torch

from operators.sm90._native import load_module


def select(query, keys, cis, pooled_cis, workspace, query_start, *, prepared_ranking=None):
    import tvm_ffi

    from operators.sm90._nosa_selection_cuda import _ranking_workspace

    ids = torch.empty((*query.shape[:2], 64), dtype=torch.int64, device=query.device)
    valid = torch.empty_like(ids, dtype=torch.bool)
    normalizers = torch.empty((1, *query.shape[:3], 2), dtype=torch.float32, device=query.device)
    ranking = prepared_ranking
    if ranking is None:
        ranking = _ranking_workspace(
            workspace, None, query_start, len(query), query.shape[1], pooled_cis
        )
    module = load_module("nosa_indexer")
    with torch.cuda.device(query.device), tvm_ffi.use_torch_stream():
        submit = module.indexer_out if prepared_ranking is None else module.indexer_ranked_out
        submit(
            query, keys, cis, pooled_cis, workspace, normalizers, ids, valid, ranking, query_start
        )
    return ids, valid
