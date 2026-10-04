"""Checkpoint FP8 linears using upstream DeepGEMM on Hopper.

GPU activation quantization uses the local handwritten Triton kernel. Matrix
multiplication uses DeepGEMM's dense and grouped public APIs; local code adapts
padding and route layout. CPU arithmetic remains an independent small oracle.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from . import quantization


def _cdiv(value, divisor):
    return (value + divisor - 1) // divisor


@lru_cache(maxsize=1)
def _deep_gemm():
    import deep_gemm

    return deep_gemm


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
        _cdiv(weight.shape[-2], 128),
        _cdiv(weight.shape[-1], 128),
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
    groups = _cdiv(columns, 128)
    blocks = F.pad(x.float(), (0, groups * 128 - columns)).reshape(rows, groups, 128)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    data = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return data.reshape(rows, groups * 128)[:, :columns].contiguous(), scale.contiguous()


def quantize_fp8_activation(x):
    """Return owned FP8 data and FP32 scales, preserving the CPU reference."""
    if x.device.type == "cpu":
        if x.ndim != 2 or not x.shape[1]:
            raise ValueError("Expected [rows, positive input channels] activation")
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Activations must be BF16, FP16, or FP32")
        return reference_quantize_fp8_activation(x)
    return quantization.quantize(x)


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


def _pad_weight(weight):
    """Satisfy TMA alignment for small test shapes; real checkpoint weights fit."""
    n, k = weight.shape[-2:]
    padded_n, padded_k = _cdiv(n, 64) * 64, _cdiv(k, 128) * 128
    if (n, k) != (padded_n, padded_k):
        weight = F.pad(weight.view(torch.uint8), (0, padded_k - k, 0, padded_n - n)).view(
            torch.float8_e4m3fn
        )
    return weight.contiguous()


@dataclass(slots=True)
class _PreparedFP8Activation:
    """One adjacent linear may consume this unchanged, call-local activation."""

    source: torch.Tensor
    data: torch.Tensor
    scales: torch.Tensor
    shape: tuple
    stride: tuple
    pointer: int
    version: int | None
    consumed: bool = False

    @classmethod
    def capture(cls, source, data, scales):
        return cls(
            source,
            data,
            scales,
            tuple(source.shape),
            source.stride(),
            source.data_ptr(),
            None if torch.is_inference(source) else source._version,
        )

    def consume(self, source):
        if self.consumed:
            raise ValueError("Prepared FP8 activation has already been consumed")
        if (
            source is not self.source
            or tuple(source.shape) != self.shape
            or source.stride() != self.stride
            or source.data_ptr() != self.pointer
            or (self.version is not None and source._version != self.version)
        ):
            raise ValueError("Prepared FP8 activation requires its unchanged source tensor")
        self.consumed = True
        return self.data, self.scales


def _validate_linear_options(x, weight, scales, out, quantized, return_quantized):
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.numel()
        or x.stride(-1) != 1
        or weight.shape[0] % 64
        or weight.shape[1] % 128
        or torch.is_grad_enabled()
    ):
        raise ValueError("FP8 linear options require nonempty aligned BF16 CUDA inference [Q,K]")
    if quantized is not None and (
        not isinstance(quantized, _PreparedFP8Activation) or return_quantized
    ):
        raise ValueError("Expected a prepared FP8 activation for one consuming linear")
    if out is None:
        return
    if (
        out.shape != (len(x), weight.shape[0])
        or out.device != x.device
        or out.dtype != torch.bfloat16
        or out.stride(-1) != 1
        or out.stride(0) < out.shape[1]
        or out.data_ptr() % 16
        or out.stride(0) * out.element_size() % 16
    ):
        raise ValueError("FP8 output must have aligned, disjoint BF16 row-major [Q,N] storage")
    sources = [x, weight, scales]
    if quantized is not None:
        sources.extend((quantized.data, quantized.scales))
    if out.untyped_storage().data_ptr() in {t.untyped_storage().data_ptr() for t in sources}:
        raise ValueError("FP8 output storage must not alias activation, weight or scale storage")


def fp8_linear(x, weight, scales, *, out=None, quantized=None, return_quantized=False):
    """Official FP8 1D2D, optionally storing directly and sharing one quantization.

    Optional arguments are restricted to aligned BF16 SM90 inference. A returned
    prepared activation has one-call lifetime; keep its source unchanged until
    the adjacent consuming linear. Inference tensors have no version counter.
    """
    _validate_weight(x, weight, scales)
    options = out is not None or quantized is not None or return_quantized
    if options:
        _validate_linear_options(x, weight, scales, out, quantized, return_quantized)
    if x.device.type == "cpu":
        return reference_fp8_linear(x, weight, scales)
    _require_sm90(x.device)
    activation = x.reshape(-1, x.shape[-1])
    rows, columns = activation.shape[0], weight.shape[0]
    if not rows:
        return torch.empty((*x.shape[:-1], columns), device=x.device, dtype=torch.bfloat16)
    with torch.cuda.device(x.device):
        padded_weight = _pad_weight(weight)
        padded_k = padded_weight.shape[-1]
        if activation.shape[-1] != padded_k:
            activation = F.pad(activation, (0, padded_k - activation.shape[-1]))
        if quantized is None:
            data, activation_scales = quantize_fp8_activation(activation)
        else:
            data, activation_scales = quantized.consume(x)
        output = (
            torch.empty((rows, padded_weight.shape[0]), device=x.device, dtype=torch.bfloat16)
            if out is None
            else out
        )
        _deep_gemm().fp8_gemm_nt(
            (data, activation_scales),
            (padded_weight, scales.contiguous()),
            output,
            recipe=(1, 128, 128),
        )
    result = (
        output
        if out is not None
        else output[:, :columns].contiguous().reshape(*x.shape[:-1], columns)
    )
    if return_quantized:
        return result, _PreparedFP8Activation.capture(x, data, activation_scales)
    return result


@dataclass(frozen=True)
class ExpertRouting:
    """Reusable route order and padded DeepGEMM layout, built without CPU counts."""

    expert_ids: torch.Tensor
    num_experts: int
    sorted_routes: torch.Tensor
    row_offsets: torch.Tensor
    tile_offsets: torch.Tensor
    block_m: int = 32
    packed_routes: torch.Tensor | None = None
    grouped_layout: torch.Tensor | None = None
    output_slots: torch.Tensor | None = None
    valid_routes: torch.Tensor | None = None


def prepare_expert_routing(expert_ids, num_experts):
    """Sort and pad expert routes on-device for the upstream grouped GEMM API."""
    if expert_ids.ndim != 2 or expert_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert_ids must be an int32/int64 [tokens, topk] tensor")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    block_m = 32
    if expert_ids.device.type == "cuda":
        _require_sm90(expert_ids.device)
        with torch.cuda.device(expert_ids.device):
            block_m = _deep_gemm().get_mk_alignment_for_contiguous_layout()
    flat = expert_ids.reshape(-1).long()
    valid_routes = (flat >= 0) & (flat < num_experts)
    keys = torch.where(valid_routes, flat, num_experts)
    sorted_routes = torch.argsort(keys)
    counts = torch.zeros(num_experts + 1, device=keys.device, dtype=torch.int32)
    counts.scatter_add_(0, keys, torch.ones_like(keys, dtype=torch.int32))
    zero = counts.new_zeros(1)
    row_offsets = torch.cat((zero, counts[:-1].cumsum(0, dtype=torch.int32)))
    tiles = torch.div(counts[:-1] + block_m - 1, block_m, rounding_mode="floor")
    tile_offsets = torch.cat((zero, tiles.cumsum(0, dtype=torch.int32)))
    common = (expert_ids, num_experts, sorted_routes, row_offsets, tile_offsets, block_m)
    if expert_ids.device.type == "cpu" or not flat.numel():
        return ExpertRouting(*common)
    # A static upper bound avoids .item()/nonzero()/CPU synchronization for
    # per-expert counts. Extra slots carry group -1 and never produce routes.
    capacity = (_cdiv(flat.numel(), block_m) + num_experts) * block_m
    positions = torch.arange(capacity, device=keys.device, dtype=torch.int32)
    experts = torch.searchsorted(tile_offsets[1:] * block_m, positions, right=True, out_int32=True)
    safe_experts = experts.clamp_max(num_experts - 1).long()
    local = positions - tile_offsets[safe_experts] * block_m
    valid = (experts < num_experts) & (local < counts[safe_experts])
    sorted_positions = (row_offsets[safe_experts] + local).clamp(0, flat.numel() - 1).long()
    packed_routes = sorted_routes[sorted_positions]
    grouped_layout = torch.where(valid, experts, -1).contiguous()
    rank = torch.empty_like(sorted_routes)
    rank.scatter_(0, sorted_routes, torch.arange(flat.numel(), device=keys.device))
    safe_keys = keys.clamp_max(num_experts - 1)
    slots = tile_offsets[safe_keys] * block_m + rank - row_offsets[safe_keys]
    output_slots = torch.where(valid_routes, slots, 0).long()
    return ExpertRouting(*common, packed_routes, grouped_layout, output_slots, valid_routes)


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
    """Upstream grouped GEMM plus route packing and inverse mapping.

    Invalid expert IDs return zero. GPU packing, quantization, scale-layout
    conversion, GEMM and inverse mapping are all part of this operator call.
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
    columns = weights.shape[1]
    if not expert_ids.numel():
        return torch.empty((tokens, topk, columns), device=x.device, dtype=torch.bfloat16)
    with torch.cuda.device(x.device):
        padded_weight = _pad_weight(weights)
        activation = x.reshape(-1, x.shape[-1])
        padded_k = padded_weight.shape[-1]
        if activation.shape[-1] != padded_k:
            activation = F.pad(activation, (0, padded_k - activation.shape[-1]))
        data, activation_scales = quantize_fp8_activation(activation)
        input_rows = routing.packed_routes if x.ndim == 3 else routing.packed_routes // topk
        valid = routing.grouped_layout >= 0
        packed_data = data.view(torch.uint8).index_select(0, input_rows)
        packed_data = torch.where(valid[:, None], packed_data, 0).view(torch.float8_e4m3fn)
        packed_scales = activation_scales.index_select(0, input_rows)
        packed_scales = torch.where(valid[:, None], packed_scales, 1.0)
        packed_output = torch.empty(
            (packed_data.shape[0], padded_weight.shape[1]), device=x.device, dtype=torch.bfloat16
        )
        _deep_gemm().m_grouped_fp8_gemm_nt_contiguous(
            (packed_data, packed_scales),
            (padded_weight, scales.contiguous()),
            packed_output,
            routing.grouped_layout,
            recipe=(1, 128, 128),
        )
        output = packed_output.index_select(0, routing.output_slots)[:, :columns]
        output = torch.where(routing.valid_routes[:, None], output, 0)
    return output.reshape(tokens, topk, columns)
