"""Triton implementation, imported only when the SM90 backend is selected."""

import torch
import triton
import triton.language as tl


@triton.jit
def _nosa_block_attention(
    Q,
    K,
    V,
    IDS,
    VALID,
    BIAS,
    OUT,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_ROW: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_ROW: tl.constexpr,
    V_HEAD: tl.constexpr,
    ID_ROW: tl.constexpr,
    ID_HEAD: tl.constexpr,
    ID_BLOCK: tl.constexpr,
    VALID_ROW: tl.constexpr,
    VALID_HEAD: tl.constexpr,
    VALID_BLOCK: tl.constexpr,
    BIAS_ROW: tl.constexpr,
    BIAS_HEAD: tl.constexpr,
    TOKENS,
    QUERY_START,
    QUERY_HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    SELECTED_BLOCKS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    g = tl.arange(0, BLOCK_G)
    d = tl.arange(0, DIM)
    n = tl.arange(0, 64)
    q = tl.load(
        Q + row * Q_ROW + (head * GROUPS + g[:, None]) * Q_HEAD + d[None, :],
        g[:, None] < GROUPS,
        other=0,
    )
    maximum = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_G,), tl.float32)
    accumulator = tl.zeros((BLOCK_G, DIM), tl.float32)
    for block in range(SELECTED_BLOCKS):
        block_id = tl.load(IDS + row * ID_ROW + head * ID_HEAD + block * ID_BLOCK)
        selected_valid = (block_id >= 0) & (block_id < tl.cdiv(TOKENS, 64))
        if HAS_VALID:
            selected_valid &= tl.load(
                VALID + row * VALID_ROW + head * VALID_HEAD + block * VALID_BLOCK
            )
        logical = block_id.to(tl.int64) * 64 + n
        valid = selected_valid & (logical < TOKENS) & (logical <= QUERY_START + row)
        # Masked padding never loads K/V or CIS, including record zero.
        k = tl.load(
            K + logical[None, :] * K_ROW + head * K_HEAD + d[:, None],
            valid[None, :],
            other=0,
        )
        v = tl.load(
            V + logical[:, None] * V_ROW + head * V_HEAD + d[None, :],
            valid[:, None],
            other=0,
        )
        logits = tl.dot(q, k) * (DIM**-0.5)
        if HAS_BIAS:
            bias = tl.load(BIAS + logical * BIAS_ROW + head * BIAS_HEAD, valid, other=0)
            logits += bias.to(tl.float32)[None, :]
        logits = tl.where(valid[None, :], logits, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        # Empty tiles leave state unchanged; entirely masked rows yield zero.
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp(maximum - safe_maximum)
        probability = tl.exp(logits - safe_maximum[:, None])
        accumulator *= correction[:, None]
        accumulator = tl.dot(probability.to(v.dtype), v, accumulator)
        denominator = denominator * correction + tl.sum(probability, 1)
        maximum = next_maximum
    result = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        OUT + row * QUERY_HEADS * DIM + (head * GROUPS + g[:, None]) * DIM + d[None, :],
        result,
        g[:, None] < GROUPS,
    )


def _selection_strides(tensor):
    return (
        0 if tensor.shape[0] == 1 else tensor.stride(0),
        0 if tensor.shape[1] == 1 else tensor.stride(1),
        tensor.stride(2),
    )


def launch_nosa_block_attention(q, keys, values, selection, query_start, cis_bias):
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if not len(q):
        return output
    ids, valid = selection.block_ids, selection.valid_mask
    groups = q.shape[1] // keys.shape[1]
    with torch.cuda.device(q.device):
        _nosa_block_attention[(len(q), keys.shape[1])](
            q,
            keys,
            values,
            ids,
            valid if valid is not None else ids,
            cis_bias if cis_bias is not None else q,
            output,
            q.stride(0),
            q.stride(1),
            keys.stride(0),
            keys.stride(1),
            values.stride(0),
            values.stride(1),
            *_selection_strides(ids),
            *((0, 0, 0) if valid is None else _selection_strides(valid)),
            0 if cis_bias is None else cis_bias.stride(0),
            0 if cis_bias is None else cis_bias.stride(1),
            len(keys),
            query_start,
            q.shape[1],
            groups,
            q.shape[-1],
            ids.shape[2],
            max(16, triton.next_power_of_2(groups)),
            valid is not None,
            cis_bias is not None,
            num_warps=4,
            num_stages=2,
        )
    return output
