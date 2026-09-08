"""Fused FP8 quantization and append for the V3.2 extend cache layouts."""

import triton
import triton.language as tl


@triton.jit
def _append(
    KV,
    IK,
    CACHE,
    KEYS,
    SCALES,
    offset,
    kv_stride: tl.constexpr,
    ik_stride: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.arange(0, 4)
    channel = tl.arange(0, 128)
    dims = group[:, None] * 128 + channel[None, :]
    x = tl.load(KV + row * kv_stride + dims).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    fp8 = (x / scale[:, None]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    record = CACHE + (offset + row) * 656
    tl.store(record + dims, fp8)
    tl.store((record + 512).to(tl.pointer_type(tl.float32)) + group, scale)
    rope_dim = tl.arange(0, 64)
    rope = tl.load(KV + row * kv_stride + 512 + rope_dim)
    tl.store((record + 528).to(tl.pointer_type(tl.bfloat16)) + rope_dim, rope)
    ik = tl.load(IK + row * ik_stride + channel).to(tl.float32)
    ik_scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(ik), 0), 1e-4) / 448.0)))
    tl.store(KEYS + (offset + row) * 128 + channel, (ik / ik_scale).to(tl.float8e4nv))
    tl.store(SCALES + offset + row, ik_scale)


def append_cache(kv, index_k, cache, keys, scales, offset):
    _append[(kv.shape[0],)](
        kv,
        index_k,
        cache,
        keys,
        scales,
        offset,
        kv.stride(0),
        index_k.stride(0),
        num_warps=4,
    )


@triton.jit
def _quantize_activation(
    X, Y, SF, M: tl.constexpr, N: tl.constexpr, STRIDE: tl.constexpr, GROUPS: tl.constexpr
):
    block = tl.program_id(0) * 4 + tl.arange(0, 4)
    row, group = block // GROUPS, block % GROUPS
    col = group[:, None] * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(X + row[:, None] * STRIDE + col, (row[:, None] < M) & (col < N), other=0).to(
        tl.float32
    )
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(x), 1), 1e-4), 448.0)
    bits = scale.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (tl.minimum(tl.maximum(exponent, 1), 254) << 23).to(tl.float32, bitcast=True)
    q = (x * (1.0 / scale[:, None])).to(tl.float8e4nv)
    tl.store(Y + row[:, None] * N + col, q, (row[:, None] < M) & (col < N))
    tl.store(SF + block, scale, row < M)


def quantize_activation(x, use_ue8m0=True, gran_k=128):
    import torch

    if x.ndim != 2 or gran_k != 128 or not use_ue8m0 or x.stride(1) != 1:
        raise ValueError("Expected 2D, last-dimension contiguous, 1x128 UE8M0 quantization")
    m, n = x.shape
    groups = triton.cdiv(n, 128)
    data = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((m, groups), dtype=torch.float32, device=x.device)
    _quantize_activation[(triton.cdiv(m * groups, 4),)](
        x,
        data,
        scales,
        m,
        n,
        x.stride(0),
        groups,
        num_warps=4,
    )
    return data, scales
