"""Local Hopper WGMMA/TMA attention, bound through TVM FFI."""

import torch
import tvm_ffi

from operators.sm90._native import load_module


def supports_native_attention(q, keys, values):
    """TMA specialization for NOSA-8B; general layouts retain the Triton path."""
    return (
        q.shape[-1] == 128
        and q.shape[1] == keys.shape[1] * 16
        and all(t.data_ptr() % 16 == 0 for t in (q, keys, values))
        and all(
            t.stride(0) > 0 and t.stride(1) > 0 and t.stride(0) % 8 == t.stride(1) % 8 == 0
            for t in (q, keys, values)
        )
    )


def launch_nosa_block_attention(q, keys, values, selection, query_start, cis_bias):
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if not len(q):
        return output
    module = load_module("nosa_attention")
    empty = torch.empty(0, dtype=torch.bool, device=q.device)
    with torch.cuda.device(q.device), tvm_ffi.use_torch_stream():
        module.forward(
            q,
            keys,
            values,
            selection.block_ids,
            selection.valid_mask if selection.valid_mask is not None else empty,
            cis_bias if cis_bias is not None else empty,
            output,
            query_start,
        )
    return output
