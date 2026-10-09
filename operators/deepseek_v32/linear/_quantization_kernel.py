"""Handwritten SM90 activation quantization; no TorchInductor dependency."""

import triton
import triton.language as tl


@triton.jit
def _maximum_nan(left, right):
    return tl.maximum(left, right, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def activation_quantization(
    X,
    Data,
    Scales,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    STRIDE_ROW: tl.constexpr,
    STRIDE_COLUMN: tl.constexpr,
    CONTIGUOUS_FULL_GROUPS: tl.constexpr,
    GROUPS_PER_CTA: tl.constexpr,
    SCALE_STRIDE_COLUMN: tl.constexpr,
):
    groups_per_row: tl.constexpr = triton.cdiv(COLUMNS, 128)
    if ROWS * COLUMNS > 0x7FFFFFFF:
        group = tl.program_id(0).to(tl.int64) * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    else:
        group = tl.program_id(0) * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    lane = tl.arange(0, 128)
    valid_group = group < ROWS * groups_per_row
    if CONTIGUOUS_FULL_GROUPS:
        offsets = group[:, None] * 128 + lane[None, :]
        values = tl.load(X + offsets, valid_group[:, None], other=0).to(tl.float32)
        valid = valid_group[:, None]
        output_offsets = offsets
    else:
        row = group // groups_per_row
        column = (group % groups_per_row)[:, None] * 128 + lane[None, :]
        valid = valid_group[:, None] & (column < COLUMNS)
        offsets = row[:, None].to(tl.int64) * STRIDE_ROW + column.to(tl.int64) * STRIDE_COLUMN
        values = tl.load(X + offsets, valid, other=0).to(tl.float32)
        output_offsets = row[:, None] * COLUMNS + column
    maximum = tl.reduce(tl.abs(values), 1, _maximum_nan)
    maximum = tl.maximum(maximum, 1.0e-4, propagate_nan=tl.PropagateNan.ALL)
    reciprocal_448 = tl.full((), 0x3B124925, tl.int32).to(tl.float32, bitcast=True)
    raw_scale = maximum * reciprocal_448
    bits = raw_scale.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, 1), 254)
    scales = (exponent << 23).to(tl.float32, bitcast=True)
    inverse = tl.inline_asm_elementwise(
        "div.full.f32 $0, 0f3f800000, $1;",
        constraints="=f,f",
        args=[scales],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    normalized = values * inverse[:, None]
    encoded = normalized.to(tl.float8e4nv)
    tl.store(Data + output_offsets, encoded, valid)
    if SCALE_STRIDE_COLUMN:
        scale_offsets = group // groups_per_row + (group % groups_per_row) * SCALE_STRIDE_COLUMN
    else:
        scale_offsets = group
    tl.store(Scales + scale_offsets, scales, valid_group)
