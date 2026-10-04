"""Checkpoint-backed DeepSeek V3.2 attention for standalone ECHO-style extend.

The MLA record is a BF16 latent plus rotary key (576 elements for V3.2).
Indexer Q/K use RoPE followed by FP8 quantization, without Hadamard rotation.
The explicit BF16 backend block-dequantizes checkpoint weights once. The FP8
backend retains checkpoint weights and quantizes activations in the SM90
linear operator. Neither backend imports SGLang or a legacy experiment.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch
from safetensors import safe_open
from torch.nn import functional as F

from models.deepseek_v32.nonmatrix import rms_norm
from operators.deepseek_v32.indexer.quantization import quantize_index


@dataclass(frozen=True)
class Config:
    dim: int = 7168
    n_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    num_hidden_layers: int = 61
    vocab_size: int = 129280
    norm_eps: float = 1e-6
    max_seq_len: int = 163840
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    scale_fmt: str | None = "ue8m0"
    weight_block_size: tuple[int, int] = (128, 128)
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    first_k_dense_replace: int = 3
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    routed_scaling_factor: float = 2.5
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> Config:
        path = Path(path)
        raw = json.loads((path / "config.json" if path.is_dir() else path).read_text())
        rope = raw.get("rope_scaling") or {}
        if rope and rope.get("rope_type", rope.get("type", "yarn")) != "yarn":
            raise ValueError("Only DeepSeek V3.2 YaRN rotary scaling is supported")
        quant = raw.get("quantization_config") or {}
        if quant.get("quant_method", "fp8") != "fp8":
            raise ValueError("This runner requires BF16 or block-FP8 weights, not AWQ weights")
        mapping = {
            "dim": "hidden_size",
            "n_heads": "num_attention_heads",
            "norm_eps": "rms_norm_eps",
            "max_seq_len": "max_position_embeddings",
        }
        fields = (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "num_hidden_layers",
            "vocab_size",
            "rope_theta",
            "intermediate_size",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "first_k_dense_replace",
            "norm_topk_prob",
            "scoring_func",
            "routed_scaling_factor",
        )
        values = {dst: raw[src] for dst, src in mapping.items() if src in raw}
        values.update({name: raw[name] for name in fields if name in raw})
        for dst, src in (
            ("original_seq_len", "original_max_position_embeddings"),
            ("rope_factor", "factor"),
            ("beta_fast", "beta_fast"),
            ("beta_slow", "beta_slow"),
            ("mscale", "mscale"),
        ):
            if src in rope:
                values[dst] = rope[src]
        values["scale_fmt"] = quant.get("scale_fmt")
        values["weight_block_size"] = tuple(quant.get("weight_block_size", (128, 128)))
        values["raw"] = raw
        return cls(**values)

    def __post_init__(self):
        dimensions = (
            self.dim,
            self.n_heads,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.index_n_heads,
            self.index_head_dim,
            self.index_topk,
            self.num_hidden_layers,
            self.vocab_size,
            self.max_seq_len,
            self.original_seq_len,
        )
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in dimensions):
            raise ValueError("Model dimensions and context lengths must be positive integers")
        if self.qk_rope_head_dim % 2 or self.qk_rope_head_dim > self.index_head_dim:
            raise ValueError("RoPE dimension must be even and fit the indexer head")
        if len(self.weight_block_size) != 2 or any(x <= 0 for x in self.weight_block_size):
            raise ValueError("weight_block_size must have two positive dimensions")
        if self.scale_fmt not in (None, "ue8m0"):
            raise ValueError("Unsupported indexer quantization scale format")

    @property
    def qk_head_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def attention_scale(self) -> float:
        scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
        if self.max_seq_len > self.original_seq_len:
            scale *= (1 + 0.1 * self.mscale * math.log(self.rope_factor)) ** 2
        return scale

    @property
    def hidden_size(self) -> int:
        return self.dim


class CheckpointReader:
    """Locate tensors using existing shard headers, without requiring an index.

    Missing unrelated shards are allowed for attention-layer profiling. A
    requested missing tensor fails explicitly; this does not certify that an
    entire 61-layer checkpoint is available. Tensor data is loaded only on get.
    """

    def __init__(self, model_path: Path | str):
        self.model_path = Path(model_path)
        self.tensor_files: dict[str, Path] = {}
        self.tensor_metadata: dict[str, dict] = {}
        for path in sorted(self.model_path.glob("*.safetensors")):
            with path.open("rb") as handle:
                prefix = handle.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"Incomplete safetensors header: {path}")
                size = struct.unpack("<Q", prefix)[0]
                if not 0 < size <= min(path.stat().st_size - 8, 100_000_000):
                    raise ValueError(f"Invalid safetensors header size: {path}")
                header = json.loads(handle.read(size))
            for name, metadata in header.items():
                if name == "__metadata__":
                    continue
                if name in self.tensor_files:
                    raise ValueError(f"Duplicate checkpoint tensor {name}")
                self.tensor_files[name] = path
                self.tensor_metadata[name] = metadata
        if not self.tensor_files:
            raise FileNotFoundError(f"No safetensors weights found in {self.model_path}")

    def get_tensor(self, name: str) -> torch.Tensor:
        try:
            path = self.tensor_files[name]
        except KeyError as exc:
            raise KeyError(f"Required tensor {name!r} is absent from {self.model_path}") from exc
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def linear_weight(
        self, stem: str, *, device, block_size: tuple[int, int] = (128, 128)
    ) -> torch.Tensor:
        data = self.get_tensor(stem + ".weight").to(device)
        if data.ndim != 2:
            raise ValueError(f"Linear weight must be a matrix: {stem}")
        if data.dtype == torch.float8_e4m3fn:
            scales = self.get_tensor(stem + ".weight_scale_inv").to(device)
            expected = tuple((n + b - 1) // b for n, b in zip(data.shape, block_size))
            if tuple(scales.shape) != expected:
                raise ValueError(f"Invalid FP8 scale shape for {stem}: expected {expected}")
            # Slice after expansion: kv_a_proj has 576 rows (a partial 128-row block).
            scale = scales.float().repeat_interleave(block_size[0], 0)
            scale = scale.repeat_interleave(block_size[1], 1)[: data.shape[0], : data.shape[1]]
            data = data.float() * scale
        elif data.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Unsupported checkpoint weight dtype {data.dtype}: {stem}")
        return data.to(torch.bfloat16).contiguous()


class CheckpointLinear:
    """Retain checkpoint FP8 data for the GPU backend or explicitly dequantize."""

    def __init__(self, reader, stem, *, device, backend="bf16", block_size=(128, 128)):
        if backend not in ("bf16", "fp8"):
            raise ValueError("linear backend must be bf16 or fp8")
        self.backend = backend
        self.scales = None
        if backend == "bf16":
            self.weight = reader.linear_weight(stem, device=device, block_size=block_size)
        else:
            if torch.device(device).type not in ("cpu", "cuda"):
                raise ValueError("The FP8 linear backend supports SM90 or an explicit CPU oracle")
            if tuple(block_size) != (128, 128):
                raise ValueError("The FP8 linear backend requires 128-by-128 checkpoint blocks")
            self.weight = reader.get_tensor(stem + ".weight").to(device).contiguous()
            if self.weight.dtype == torch.float8_e4m3fn:
                self.scales = reader.get_tensor(stem + ".weight_scale_inv").to(device).contiguous()
                expected = tuple((n + 127) // 128 for n in self.weight.shape)
                if tuple(self.scales.shape) != expected:
                    raise ValueError(f"Invalid FP8 scale shape for {stem}: expected {expected}")
            elif self.weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
                self.weight = self.weight.bfloat16()
            else:
                raise ValueError(f"Unsupported checkpoint weight dtype: {stem}")

    def __call__(self, x, *, out=None, quantized=None, return_quantized=False):
        if self.scales is not None:
            from operators.deepseek_v32.linear.fp8 import fp8_linear

            return fp8_linear(
                x,
                self.weight,
                self.scales,
                out=out,
                quantized=quantized,
                return_quantized=return_quantized,
            )
        if out is not None or quantized is not None or return_quantized:
            raise ValueError("Prepared input and output views require the checkpoint FP8 path")
        return F.linear(x, self.weight)


def normalized_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal Sylvester Hadamard transform, accumulated in FP32."""
    width = x.shape[-1]
    if not width or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    y = x.float()
    stride = 1
    while stride < width:
        shape = y.shape
        pairs = y.reshape(*shape[:-1], -1, 2, stride)
        a, b = pairs.unbind(-2)
        y = torch.stack((a + b, a - b), dim=-2).reshape(shape)
        stride *= 2
    return (y * width**-0.5).to(x.dtype)


def rotary_frequencies(config: Config, *, device) -> torch.Tensor:
    dim = config.qk_rope_head_dim
    freq = config.rope_theta ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    if config.max_seq_len > config.original_seq_len:

        def correction(rotations):
            return (
                dim
                * math.log(config.original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(config.rope_theta))
            )

        low = max(math.floor(correction(config.beta_fast)), 0)
        high = min(math.ceil(correction(config.beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, device=device).float() - low) / (high - low)).clamp(0, 1)
        freq = freq * (1 - ramp) + freq / config.rope_factor * ramp
    return freq


def apply_rope(x: torch.Tensor, angles: torch.Tensor, *, interleaved: bool) -> torch.Tensor:
    """Rotate [token, ..., rotary-dim], matching official MLA/indexer pairing."""
    cos = angles.cos().reshape(angles.shape[0], *([1] * (x.ndim - 2)), -1)
    sin = angles.sin().reshape_as(cos)
    a, b = (x.float()[..., ::2], x.float()[..., 1::2]) if interleaved else x.float().chunk(2, -1)
    real, imag = a * cos - b * sin, b * cos + a * sin
    result = (
        torch.stack((real, imag), -1).flatten(-2) if interleaved else torch.cat((real, imag), -1)
    )
    return result.to(x.dtype)


def prepare_rotary_cache(angles):
    """Build one per-call trig table, shared by MLA and indexer rotations."""
    return (
        torch.arange(len(angles), device=angles.device, dtype=torch.int64),
        torch.cat((angles.cos(), angles.sin()), dim=-1),
    )


def apply_rope_pair(q, k, angles, *, interleaved, cache=None):
    from operators import flashinfer

    if (
        flashinfer.can_use_flashinfer(q)
        and q.shape[-1] in (32, 64, 128, 256, 512)
        and q.shape[-1] == k.shape[-1]
        and angles.shape[-1] * 2 <= q.shape[-1]
    ):
        positions, trig = prepare_rotary_cache(angles) if cache is None else cache
        return flashinfer.rotary_pair(q, k, positions, trig, interleaved=interleaved)
    width = angles.shape[-1] * 2
    return tuple(
        torch.cat(
            (apply_rope(x[..., :width], angles, interleaved=interleaved), x[..., width:]),
            dim=-1,
        )
        for x in (q, k)
    )


@dataclass
class Projected:
    q: torch.Tensor
    kv: torch.Tensor
    index_q: torch.Tensor
    index_k: torch.Tensor
    index_weights: torch.Tensor
    index_scale: torch.Tensor


class CheckpointAttention:
    """One real checkpoint attention layer, including its input RMSNorm.

    ``project`` consumes the residual-stream hidden states. It returns logical
    Q/K/indexer records; cache ownership and selected-record fetching are left
    to the caller. ``output`` applies W_VB and W_O, without adding the residual.
    """

    precision: ClassVar[dict[str, str | bool]] = {
        "projection": "checkpoint_block_fp8_dequant_to_bf16_then_bf16_gemm",
        "projection_activation_quantization": False,
        "main_kv": "bf16_latent_and_rope_no_fp8_roundtrip",
        "indexer": "bf16_rope_then_fp8_e4m3_no_hadamard",
        "indexer_head_weights": "fp32_linear_from_checkpoint_weight",
    }

    def __init__(
        self,
        model_path: Path | str,
        layer_idx: int = 0,
        device="cuda",
        *,
        reader: CheckpointReader | None = None,
        linear_backend="bf16",
    ):
        self.config = self.cfg = Config.from_checkpoint(model_path)
        cfg = self.cfg
        if not isinstance(layer_idx, int) or not 0 <= layer_idx < cfg.num_hidden_layers:
            raise ValueError("Layer index is outside the checkpoint configuration")
        self.layer_idx = layer_idx
        self.device = torch.device(device)
        self.reader = reader or CheckpointReader(model_path)
        self.linear_backend = linear_backend
        self.precision = dict(type(self).precision)
        if linear_backend == "fp8":
            self.precision["projection"] = "checkpoint_block_fp8_weight_and_fp8_activation_gemm"
            self.precision["projection_activation_quantization"] = True
        stem = f"model.layers.{layer_idx}."
        attn = stem + "self_attn."

        def tensor(name):
            return self.reader.get_tensor(name).to(device=self.device, dtype=torch.float32)

        def linear(name):
            return CheckpointLinear(
                self.reader,
                attn + name,
                device=self.device,
                backend=linear_backend,
                block_size=cfg.weight_block_size,
            )

        self.input_norm_weight = tensor(stem + "input_layernorm.weight")
        self.q_norm_weight = tensor(attn + "q_a_layernorm.weight")
        self.kv_norm_weight = tensor(attn + "kv_a_layernorm.weight")
        self.index_norm_weight = tensor(attn + "indexer.k_norm.weight")
        self.index_norm_bias = tensor(attn + "indexer.k_norm.bias")
        self.wq_a, self.wq_b = linear("q_a_proj"), linear("q_b_proj")
        self.wkv_a = linear("kv_a_proj_with_mqa")
        wkv_b = self.reader.linear_weight(
            attn + "kv_b_proj", device=self.device, block_size=cfg.weight_block_size
        ).reshape(cfg.n_heads, cfg.qk_nope_head_dim + cfg.v_head_dim, cfg.kv_lora_rank)
        self.wk_b = wkv_b[:, : cfg.qk_nope_head_dim].contiguous()
        self.wv_b = wkv_b[:, cfg.qk_nope_head_dim :].transpose(1, 2).contiguous()
        self.wo = linear("o_proj")
        self.index_wq, self.index_wk = linear("indexer.wq_b"), linear("indexer.wk")
        self.index_head_weight = tensor(attn + "indexer.weights_proj.weight")
        self.frequencies = rotary_frequencies(cfg, device=self.device)

    @torch.inference_mode()
    def embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype not in (torch.int32, torch.int64) or token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional integer tensor")
        # CPU mmap keeps the full embedding table out of HBM; only requested rows move.
        ids = token_ids.to(device="cpu", dtype=torch.int64)
        if ids.numel() and (ids.min().item() < 0 or ids.max().item() >= self.cfg.vocab_size):
            raise ValueError("Token id is outside the checkpoint vocabulary")
        table = self.reader.get_tensor("model.embed_tokens.weight")
        return table[ids].to(device=self.device, dtype=torch.bfloat16)

    @torch.inference_mode()
    def project(self, hidden: torch.Tensor, start_pos: int, *, normalized=False) -> Projected:
        self._validate_project_hidden(hidden)
        count = hidden.shape[0]
        if (
            not isinstance(start_pos, int)
            or start_pos < 0
            or start_pos + count > self.cfg.max_seq_len
        ):
            raise ValueError("Token positions exceed the checkpoint context limit")
        positions = torch.arange(start_pos, start_pos + count, device=hidden.device).float()
        return self.project_positions(hidden, positions, normalized=normalized)

    def _validate_project_hidden(self, hidden):
        cfg = self.cfg
        if hidden.ndim != 2 or hidden.shape[1] != cfg.dim or not hidden.shape[0]:
            raise ValueError(f"hidden must have nonempty shape [tokens, {cfg.dim}]")
        if hidden.device != self.wq_a.weight.device or hidden.dtype != torch.bfloat16:
            raise ValueError("hidden must be BF16 on the attention layer's device")

    @torch.inference_mode()
    def project_positions(self, hidden, positions, *, normalized=False) -> Projected:
        """Pure projection with explicit FP32 positions; caller validates their values.

        The serving graph caller validates its integer start and context bound
        before updating a device scalar. No tensor value is read on the CPU.
        """
        self._validate_project_hidden(hidden)
        if (
            positions.shape != (len(hidden),)
            or positions.device != hidden.device
            or positions.dtype != torch.float32
        ):
            raise ValueError("positions must be one FP32 value per token on the hidden device")
        cfg = self.cfg
        count = hidden.shape[0]
        x = hidden if normalized else rms_norm(hidden, self.input_norm_weight, cfg.norm_eps)
        qr = rms_norm(self.wq_a(x), self.q_norm_weight, cfg.norm_eps)
        angles = positions[:, None] * self.frequencies[None, :]
        rotary_cache = prepare_rotary_cache(angles) if hidden.is_cuda else None
        q = self.wq_b(qr).reshape(count, cfg.n_heads, -1)
        q_nope, q_pe = q.split((cfg.qk_nope_head_dim, cfg.qk_rope_head_dim), -1)
        if hidden.is_cuda:
            # cuBLAS accepts this strided output directly. Writing into the
            # final token-major layout avoids repacking a head-major Q tensor.
            projected_q = torch.empty(
                (count, cfg.n_heads, cfg.qk_head_dim), device=hidden.device, dtype=hidden.dtype
            )
            torch.bmm(
                q_nope.transpose(0, 1),
                self.wk_b,
                out=projected_q[..., : cfg.kv_lora_rank].transpose(0, 1),
            )
        else:
            q_latent = torch.bmm(q_nope.transpose(0, 1), self.wk_b).transpose(0, 1)
        kv = self.wkv_a(x)
        latent, k_pe = kv.split((cfg.kv_lora_rank, cfg.qk_rope_head_dim), -1)
        latent = rms_norm(latent, self.kv_norm_weight, cfg.norm_eps)
        direct_rotary_output = hidden.is_cuda and q_pe.shape[1:] == (128, 64)
        if direct_rotary_output:
            from operators.flashinfer import rotary_pair_into

            q_pe, k_pe = rotary_pair_into(
                q_pe,
                k_pe,
                *rotary_cache,
                projected_q[..., cfg.kv_lora_rank :],
                interleaved=True,
            )
        else:
            q_pe, k_pe = apply_rope_pair(q_pe, k_pe, angles, interleaved=True, cache=rotary_cache)
        index_q = self.index_wq(qr).reshape(count, cfg.index_n_heads, cfg.index_head_dim)
        index_k = F.layer_norm(
            self.index_wk(x).float(),
            (cfg.index_head_dim,),
            self.index_norm_weight,
            self.index_norm_bias,
            cfg.norm_eps,
        ).bfloat16()
        index_q, index_k = apply_rope_pair(
            index_q, index_k, angles, interleaved=False, cache=rotary_cache
        )
        index_q, q_scale = quantize_index(index_q, cfg.scale_fmt)
        index_k, k_scale = quantize_index(index_k, cfg.scale_fmt)
        weights = F.linear(x.float(), self.index_head_weight) * cfg.index_n_heads**-0.5
        weights = weights * q_scale[..., 0] * cfg.index_head_dim**-0.5
        if hidden.is_cuda:
            if not direct_rotary_output:
                projected_q[..., cfg.kv_lora_rank :].copy_(q_pe)
        else:
            projected_q = torch.cat((q_latent, q_pe), -1).contiguous()
        return Projected(
            q=projected_q,
            kv=torch.cat((latent, k_pe), -1).contiguous(),
            index_q=index_q,
            index_k=index_k,
            index_weights=weights.contiguous(),
            index_scale=k_scale[:, 0].contiguous(),
        )

    @torch.inference_mode()
    def _expand_values(self, attention: torch.Tensor, *, out=None) -> torch.Tensor:
        """Expand absorbed values, optionally into owned graph input storage."""
        cfg = self.cfg
        if attention.ndim != 3 or attention.shape[1:] != (cfg.n_heads, cfg.kv_lora_rank):
            raise ValueError("Attention output must have shape [tokens, heads, kv_lora_rank]")
        if attention.device != self.wo.weight.device or attention.dtype != torch.bfloat16:
            raise ValueError("Attention output must be BF16 on the attention layer's device")
        if out is not None:
            if (
                out.shape != (attention.shape[0], cfg.n_heads, cfg.v_head_dim)
                or out.dtype != attention.dtype
                or out.device != attention.device
                or not out.is_contiguous()
            ):
                raise ValueError("Expanded output must be contiguous BF16 [tokens, heads, values]")
            torch.bmm(attention.transpose(0, 1), self.wv_b, out=out.transpose(0, 1))
            return out
        if attention.is_cuda:
            # Write into the final token-major layout so flattening for o_proj
            # does not repack a head-major BMM result.
            head_out = torch.empty(
                (attention.shape[0], cfg.n_heads, cfg.v_head_dim),
                device=attention.device,
                dtype=attention.dtype,
            )
            torch.bmm(attention.transpose(0, 1), self.wv_b, out=head_out.transpose(0, 1))
        else:
            head_out = torch.bmm(attention.transpose(0, 1), self.wv_b).transpose(0, 1)
        return head_out

    @torch.inference_mode()
    def output(self, attention: torch.Tensor) -> torch.Tensor:
        head_out = self._expand_values(attention)
        return self.wo(head_out.reshape(attention.shape[0], -1))
