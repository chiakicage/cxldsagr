"""Owned Hopper WGMMA scoring kernels, loaded through tvm-ffi."""

import torch

from operators.sm90._native import load_module


def supports(query, keys, output):
    """The initial WGMMA tile maps sixteen GQA heads to each warp."""
    return (
        query.shape[2:] == (16, 128)
        and query.dtype in (torch.bfloat16, torch.float16)
        and output.dtype in (query.dtype, torch.float32)
        and all(
            stride > 0 and stride % 8 == 0 for stride in (*query.stride()[:3], *keys.stride()[:2])
        )
        and query.data_ptr() % 16 == 0
        and keys.data_ptr() % 16 == 0
    )


def scores_out(query, keys, positions, output, query_start, blocks, *, pool_output):
    import tvm_ffi

    module = load_module("nosa_scores")
    splits = 1 if len(query) >= 128 else min(4, (len(keys) + 127) // 128)
    normalizers = torch.empty(
        (splits, *query.shape[:3], 2), device=query.device, dtype=torch.float32
    )
    with torch.cuda.device(query.device), tvm_ffi.use_torch_stream():
        module.scores_out(
            query,
            keys,
            query if positions is None else positions,
            output,
            normalizers,
            query_start,
            blocks,
            positions is None,
            pool_output,
        )
