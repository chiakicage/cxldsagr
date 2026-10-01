"""Independent Hopper FP8 block-scaled linear and grouped expert operators.

Checkpoint weights are E4M3 with one FP32 multiplier per 128x128 block.
Activations are quantized per row and 128 input channels, rounding the scale
up to a power of two (UE8M0). Outputs are BF16. CUDA uses Triton tensor cores;
CPU dispatches to explicit FP32 reference arithmetic for small model tests.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _quantize_activation(X, Y, S, M, K: tl.constexpr, ROW: tl.constexpr, COL: tl.constexpr):
    groups: tl.constexpr = tl.cdiv(K, 128)
    block = tl.program_id(0) * 4 + tl.arange(0, 4)
    row, group = block // groups, block % groups
    channel = group[:, None] * 128 + tl.arange(0, 128)[None, :]
    value = tl.load(
        X + row[:, None] * ROW + channel * COL,
        (row[:, None] < M) & (channel < K),
        other=0,
    ).to(tl.float32)
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(value), 1), 1e-4), 448.0)
    # Exact upward rounding to a power of two, without approximate log2.
    bits = scale.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (tl.minimum(tl.maximum(exponent, 1), 254) << 23).to(tl.float32, bitcast=True)
    quantized = (value / scale[:, None]).to(tl.float8e4nv)
    tl.store(Y + row[:, None] * K + channel, quantized, (row[:, None] < M) & (channel < K))
    tl.store(S + block, scale, row < M)


@triton.jit
def _fp8_linear(
    X,
    XS,
    W,
    WS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    W_ROW: tl.constexpr,
    W_COL: tl.constexpr,
    WS_ROW: tl.constexpr,
    WS_COL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    channel = tl.arange(0, 128)
    groups: tl.constexpr = tl.cdiv(K, 128)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for block in range(groups):
        kk = block * 128 + channel
        a = tl.load(X + rows[:, None] * K + kk[None, :], (rows[:, None] < M) & (kk < K), 0.0)
        b = tl.load(
            W + columns[None, :] * W_ROW + kk[:, None] * W_COL,
            (columns[None, :] < N) & (kk[:, None] < K),
            0.0,
        )
        a_scale = tl.load(XS + rows * groups + block, rows < M, 0)
        b_scale = tl.load(WS + (columns // 128) * WS_ROW + block * WS_COL, columns < N, 0)
        partial = tl.dot(a, b)
        accumulator += partial * a_scale[:, None] * b_scale[None, :]
    tl.store(
        Y + rows[:, None] * N + columns[None, :], accumulator, (rows[:, None] < M) & (columns < N)
    )


@triton.jit
def _grouped_fp8_linear(
    X,
    XS,
    W,
    WS,
    SORTED_ROUTES,
    ROW_OFFSETS,
    TILE_OFFSETS,
    Y,
    EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    W_EXPERT: tl.constexpr,
    W_ROW: tl.constexpr,
    W_COL: tl.constexpr,
    WS_EXPERT: tl.constexpr,
    WS_ROW: tl.constexpr,
    WS_COL: tl.constexpr,
    ROUTE_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    total_tiles = tl.load(TILE_OFFSETS + EXPERTS)
    if tile < total_tiles:
        # Device-side binary search avoids reading expert counts back to CPU.
        low, high = 0, EXPERTS
        while low < high:
            middle = (low + high) // 2
            boundary = tl.load(TILE_OFFSETS + middle + 1)
            right = tile >= boundary
            low = tl.where(right, middle + 1, low)
            high = tl.where(right, high, middle)
        expert = low
        row_begin = tl.load(ROW_OFFSETS + expert)
        row_end = tl.load(ROW_OFFSETS + expert + 1)
        tile_begin = tl.load(TILE_OFFSETS + expert)
        sorted_row = row_begin + (tile - tile_begin) * BLOCK_M + tl.arange(0, BLOCK_M)
        route = tl.load(SORTED_ROUTES + sorted_row, sorted_row < row_end, 0)
        if ROUTE_INPUT:
            input_row = route
        else:
            input_row = route // TOPK
        columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        channel = tl.arange(0, 128)
        groups: tl.constexpr = tl.cdiv(K, 128)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        weight = W + expert.to(tl.int64) * W_EXPERT
        weight_scale = WS + expert.to(tl.int64) * WS_EXPERT
        for block in range(groups):
            kk = block * 128 + channel
            a = tl.load(
                X + input_row[:, None] * K + kk[None, :],
                (sorted_row[:, None] < row_end) & (kk < K),
                0.0,
            )
            b = tl.load(
                weight + columns[None, :] * W_ROW + kk[:, None] * W_COL,
                (columns[None, :] < N) & (kk[:, None] < K),
                0.0,
            )
            a_scale = tl.load(XS + input_row * groups + block, sorted_row < row_end, 0)
            b_scale = tl.load(
                weight_scale + (columns // 128) * WS_ROW + block * WS_COL,
                columns < N,
                0,
            )
            partial = tl.dot(a, b)
            accumulator += partial * a_scale[:, None] * b_scale[None, :]
        tl.store(
            Y + route[:, None] * N + columns[None, :],
            accumulator,
            (sorted_row[:, None] < row_end) & (columns < N),
        )


def _require_sm90(device):
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise NotImplementedError("FP8 linear CUDA execution requires SM90/Hopper")


def _validate_weight(x, weight, scales, *, grouped=False):
    expected_dims = 3 if grouped else 2
    if weight.ndim != expected_dims or scales.ndim != expected_dims:
        raise ValueError("Invalid weight or block scale dimensions")
    if x.ndim < 2 or x.shape[-1] != weight.shape[-1] or not x.shape[-1]:
        raise ValueError("Activation and weight input dimensions must agree and be positive")
    if weight.shape[-2] == 0:
        raise ValueError("Output dimension must be positive")
    expected_scales = (
        *weight.shape[:-2],
        triton.cdiv(weight.shape[-2], 128),
        triton.cdiv(weight.shape[-1], 128),
    )
    if scales.shape != expected_scales:
        raise ValueError(f"Expected 128x128 block scales with shape {expected_scales}")
    if weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
        raise ValueError("Weights must be float8_e4m3fn and scales must be FP32")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Activations must be BF16, FP16, or FP32")
    if x.device != weight.device or x.device != scales.device:
        raise ValueError("Activations, weights, and scales must share a device")


def reference_quantize_fp8_activation(x):
    """Independent PyTorch UE8M0 quantization, also usable as a CPU oracle."""
    if x.ndim != 2 or not x.shape[1]:
        raise ValueError("Expected [rows, positive input channels] activation")
    rows, columns = x.shape
    groups = triton.cdiv(columns, 128)
    blocks = F.pad(x.float(), (0, groups * 128 - columns)).reshape(rows, groups, 128)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    data = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return data.reshape(rows, groups * 128)[:, :columns].contiguous(), scale.contiguous()


def quantize_fp8_activation(x):
    """Quantize a 2D activation to FP8 data and row-by-K-block FP32 scales."""
    if x.ndim != 2 or not x.shape[1]:
        raise ValueError("Expected [rows, positive input channels] activation")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Activations must be BF16, FP16, or FP32")
    if x.device.type == "cpu":
        return reference_quantize_fp8_activation(x)
    _require_sm90(x.device)
    rows, columns = x.shape
    groups = triton.cdiv(columns, 128)
    data = torch.empty((rows, columns), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, groups), device=x.device, dtype=torch.float32)
    if rows:
        with torch.cuda.device(x.device):
            _quantize_activation[(triton.cdiv(rows * groups, 4),)](
                x, data, scales, rows, columns, *x.stride(), num_warps=4
            )
    return data, scales


def _dequantize_weight(weight, scales):
    expanded = scales.repeat_interleave(128, -2).repeat_interleave(128, -1)
    return weight.float() * expanded[..., : weight.shape[-2], : weight.shape[-1]]


def _reference_activation(x):
    data, scales = reference_quantize_fp8_activation(x)
    return data.float() * scales.repeat_interleave(128, -1)[:, : x.shape[-1]]


@torch.no_grad()
def reference_fp8_linear(x, weight, scales):
    """FP32 matmul of independently dequantized weights and FP8 activations."""
    _validate_weight(x, weight, scales)
    with torch.autocast(device_type=x.device.type, enabled=False):
        activation = _reference_activation(x.reshape(-1, x.shape[-1]))
        result = activation @ _dequantize_weight(weight, scales).T
    return result.reshape(*x.shape[:-1], weight.shape[0]).to(torch.bfloat16)


def fp8_linear(x, weight, scales):
    """``[..., K] @ [N, K].T -> [..., N]`` with checkpoint FP8 block scales.

    CPU uses the reference for small correctness tests. GPU never falls back
    to dequantized dense weights or a CPU implementation.
    """
    _validate_weight(x, weight, scales)
    if x.device.type == "cpu":
        return reference_fp8_linear(x, weight, scales)
    _require_sm90(x.device)
    activation = x.reshape(-1, x.shape[-1])
    data, activation_scales = quantize_fp8_activation(activation)
    rows, columns = activation.shape[0], weight.shape[0]
    output = torch.empty((rows, columns), device=x.device, dtype=torch.bfloat16)
    if rows:
        block_m = 32 if rows < 64 else 64
        with torch.cuda.device(x.device):
            _fp8_linear[(triton.cdiv(rows, block_m), triton.cdiv(columns, 128))](
                data,
                activation_scales,
                weight,
                scales,
                output,
                rows,
                columns,
                x.shape[-1],
                *weight.stride(),
                *scales.stride(),
                block_m,
                128,
                num_warps=4,
                num_stages=2,
            )
    return output.reshape(*x.shape[:-1], columns)


@dataclass(frozen=True)
class ExpertRouting:
    """GPU route ordering reusable by the gate, up, and down expert GEMMs.

    ``expert_ids`` must remain unchanged while the plan is reused. Invalid
    expert IDs are padding; they are sorted after valid routes and yield zero.
    No counts or offsets are copied to CPU to construct this plan.
    """

    expert_ids: torch.Tensor
    num_experts: int
    sorted_routes: torch.Tensor
    row_offsets: torch.Tensor
    tile_offsets: torch.Tensor
    block_m: int = 32


def prepare_expert_routing(expert_ids, num_experts):
    """Prepare ``[tokens, topk]`` routing entirely on its current device."""
    if expert_ids.ndim != 2 or expert_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert_ids must be an int32/int64 [tokens, topk] tensor")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    block_m = 32
    flat = expert_ids.reshape(-1).long()
    keys = torch.where((flat >= 0) & (flat < num_experts), flat, num_experts)
    sorted_routes = torch.argsort(keys)
    counts = torch.zeros(num_experts + 1, device=keys.device, dtype=torch.int32)
    counts.scatter_add_(0, keys, torch.ones_like(keys, dtype=torch.int32))
    zero = counts.new_zeros(1)
    row_offsets = torch.cat((zero, counts[:-1].cumsum(0, dtype=torch.int32)))
    tiles = torch.div(counts[:-1] + block_m - 1, block_m, rounding_mode="floor")
    tile_offsets = torch.cat((zero, tiles.cumsum(0, dtype=torch.int32)))
    return ExpertRouting(expert_ids, num_experts, sorted_routes, row_offsets, tile_offsets, block_m)


def _validate_grouped(x, expert_ids, weights, scales):
    _validate_weight(x, weights, scales, grouped=True)
    if expert_ids.ndim != 2 or expert_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert_ids must be an int32/int64 [tokens, topk] tensor")
    if expert_ids.device != x.device or x.ndim not in (2, 3):
        raise ValueError("Expected token or token-route activations on the routing device")
    if x.shape[0] != expert_ids.shape[0] or (x.ndim == 3 and x.shape[1] != expert_ids.shape[1]):
        raise ValueError("Activation and routing token/route dimensions do not agree")
    if weights.shape[0] <= 0:
        raise ValueError("At least one expert is required")


@torch.no_grad()
def reference_grouped_fp8_linear(x, expert_ids, weights, scales):
    """Small CPU/GPU numerical oracle, explicitly separate from the CUDA path."""
    _validate_grouped(x, expert_ids, weights, scales)
    tokens, topk = expert_ids.shape
    output = torch.zeros((tokens, topk, weights.shape[1]), device=x.device, dtype=torch.bfloat16)
    with torch.autocast(device_type=x.device.type, enabled=False):
        quantized = _reference_activation(x.reshape(-1, x.shape[-1]))
        if x.ndim == 2:
            quantized = quantized[:, None, :].expand(tokens, topk, x.shape[-1])
        else:
            quantized = quantized.reshape(tokens, topk, x.shape[-1])
        for expert in range(weights.shape[0]):
            selected = expert_ids == expert
            if selected.any():
                weight = _dequantize_weight(weights[expert], scales[expert])
                output[selected] = (quantized[selected] @ weight.T).to(torch.bfloat16)
    return output


def grouped_fp8_linear(x, expert_ids, weights, scales, *, routing=None):
    """Apply selected experts, returning BF16 ``[tokens, topk, out_channels]``.

    ``x`` may be ``[tokens, in_channels]`` for gate/up projections, or
    ``[tokens, topk, in_channels]`` for down projections. ``routing`` can be
    shared across all three calls. Invalid expert IDs produce zero routes.
    """
    _validate_grouped(x, expert_ids, weights, scales)
    if x.device.type == "cpu":
        return reference_grouped_fp8_linear(x, expert_ids, weights, scales)
    _require_sm90(x.device)
    if routing is None:
        routing = prepare_expert_routing(expert_ids, weights.shape[0])
    if routing.expert_ids is not expert_ids or routing.num_experts != weights.shape[0]:
        raise ValueError("Routing plan must reference the same expert_ids tensor and expert count")
    tokens, topk = expert_ids.shape
    routes, columns = expert_ids.numel(), weights.shape[1]
    # Zero initialization also covers invalid expert IDs, whose routes are not launched.
    output = torch.zeros((tokens, topk, columns), device=x.device, dtype=torch.bfloat16)
    if not routes:
        return output
    data, activation_scales = quantize_fp8_activation(x.reshape(-1, x.shape[-1]))
    max_tiles = triton.cdiv(routes, routing.block_m) + weights.shape[0]
    with torch.cuda.device(x.device):
        _grouped_fp8_linear[(max_tiles, triton.cdiv(columns, 128))](
            data,
            activation_scales,
            weights,
            scales,
            routing.sorted_routes,
            routing.row_offsets,
            routing.tile_offsets,
            output,
            weights.shape[0],
            topk,
            columns,
            x.shape[-1],
            *weights.stride(),
            *scales.stride(),
            x.ndim == 3,
            routing.block_m,
            128,
            num_warps=4,
            num_stages=2,
        )
    return output
