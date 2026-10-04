"""Local Hopper WGMMA/TMA attention, bound through TVM FFI."""


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


def launch_nosa_block_attention(
    q, keys, values, selection, query_start, cis_bias, *, workspace=None
):
    from operators.nosa.attention.device_only._fa3 import launch_nosa_fa3_attention

    if workspace is not None:
        return launch_nosa_fa3_attention(
            q, keys, values, selection, query_start, cis_bias, workspace=workspace
        )
    return launch_nosa_fa3_attention(q, keys, values, selection, query_start, cis_bias)
