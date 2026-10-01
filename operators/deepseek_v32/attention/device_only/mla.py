"""Standalone sparse MLA for DeepSeek V3.2 prefill/extend on Hopper.

The cache stores one shared ``[latent, rotated_position]`` vector per token.
Queries have already absorbed the per-head key projection into the latent part.
The first ``value_dim`` cache elements are also the values; output projection is
the caller's responsibility. Selection is per query and shared by all heads.
Indices address rows of the supplied cache, so the same kernel accepts logical
resident IDs and remapped physical IDs after sparse prefetch. Causality must be
enforced by the caller when constructing the selection, before remapping IDs.

No SGLang, DeepGEMM, FlashMLA, or SM120 extension is imported. The online softmax
uses Triton tensor-core dots for both QK and PV without materializing gathered
``[query, selected_token, dimension]`` cache tensors or attention matrices.
"""

import torch
import triton
import triton.language as tl

from operators.deepseek_v32.attention._validation import _validate


@triton.jit
def _sparse_mla_kernel(
    Q,
    KV,
    IDS,
    OUT,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    Q_DIM: tl.constexpr,
    KV_ROW: tl.constexpr,
    KV_DIM: tl.constexpr,
    ID_ROW: tl.constexpr,
    ID_COLUMN: tl.constexpr,
    TOKENS,
    HEADS: tl.constexpr,
    DIMENSION: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    SELECTED: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    values = tl.arange(0, BLOCK_V)
    positions = tl.arange(0, BLOCK_R)
    slots = tl.arange(0, BLOCK_K)
    q_latent = tl.load(
        Q + row * Q_ROW + heads[:, None] * Q_HEAD + values[None, :] * Q_DIM,
        (heads[:, None] < HEADS) & (values[None, :] < VALUE_DIM),
        other=0,
    )
    if DIMENSION > VALUE_DIM:
        q_rope = tl.load(
            Q + row * Q_ROW + heads[:, None] * Q_HEAD + (VALUE_DIM + positions[None, :]) * Q_DIM,
            (heads[:, None] < HEADS) & (positions[None, :] < DIMENSION - VALUE_DIM),
            other=0,
        )
    maximum = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_H,), tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_V), tl.float32)
    for block in range(tl.cdiv(SELECTED, BLOCK_K)):
        selected_slot = block * BLOCK_K + slots
        token = tl.load(
            IDS + row * ID_ROW + selected_slot * ID_COLUMN,
            selected_slot < SELECTED,
            other=-1,
        )
        valid = (selected_slot < SELECTED) & (token >= 0) & (token < TOKENS)
        latent = tl.load(
            KV + token[:, None].to(tl.int64) * KV_ROW + values[None, :] * KV_DIM,
            valid[:, None] & (values[None, :] < VALUE_DIM),
            other=0,
        )
        logits = tl.dot(q_latent, tl.trans(latent))
        if DIMENSION > VALUE_DIM:
            rope = tl.load(
                KV
                + token[:, None].to(tl.int64) * KV_ROW
                + (VALUE_DIM + positions[None, :]) * KV_DIM,
                valid[:, None] & (positions[None, :] < DIMENSION - VALUE_DIM),
                other=0,
            )
            logits = tl.dot(q_rope, tl.trans(rope), logits)
        # exp2 avoids an extra multiply inside each online-softmax update.
        logits *= SCALE * 1.4426950408889634
        logits = tl.where(valid[None, :], logits, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, axis=1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        probability = tl.exp2(logits - safe_maximum[:, None])
        accumulator *= correction[:, None]
        accumulator = tl.dot(probability.to(latent.dtype), latent, accumulator)
        denominator = denominator * correction + tl.sum(probability, axis=1)
        maximum = next_maximum
    result = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        OUT + row * HEADS * VALUE_DIM + heads[:, None] * VALUE_DIM + values[None, :],
        result,
        (heads[:, None] < HEADS) & (values[None, :] < VALUE_DIM),
    )


def sparse_mla(q, kv, indices, scale, value_dim=512):
    """Apply sparse MLA to BF16/FP16 queries and a shared latent KV cache.

    Returns ``[Q, H, value_dim]`` in the query dtype. Each selected slot
    participates once, including repeated IDs. Negative and out-of-range IDs
    are padding and never read cache memory; all-padding rows return zeros.
    Strided views are supported. This is an inference-only SM90 operator;
    use ``attention.reference.torch.reference_sparse_mla`` as the FP32 oracle.
    """
    _validate(q, kv, indices, scale, value_dim)
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise NotImplementedError("sparse_mla requires an SM90/Hopper CUDA device")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("The SM90 sparse MLA kernel requires BF16 or FP16 inputs")
    output = torch.empty((*q.shape[:2], value_dim), dtype=q.dtype, device=q.device)
    if not q.shape[0] or not q.shape[1]:
        return output
    if not kv.shape[0] or not indices.shape[1]:
        return output.zero_()
    with torch.cuda.device(q.device):
        _sparse_mla_kernel[(q.shape[0], triton.cdiv(q.shape[1], 16))](
            q,
            kv,
            indices,
            output,
            *q.stride(),
            *kv.stride(),
            *indices.stride(),
            kv.shape[0],
            q.shape[1],
            q.shape[2],
            value_dim,
            indices.shape[1],
            float(scale),
            16,
            64,
            max(16, triton.next_power_of_2(value_dim)),
            max(16, triton.next_power_of_2(q.shape[2] - value_dim)),
            num_warps=8 if value_dim > 128 else 4,
            num_stages=1,
        )
    return output
