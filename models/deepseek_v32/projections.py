"""Checkpoint MLA and indexer projections, without cache ownership."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from torch.nn import functional as F

from models.deepseek_v32.checkpoint import CheckpointLinear, CheckpointReader
from models.deepseek_v32.config import Config
from models.deepseek_v32.nonmatrix import rms_norm
from models.deepseek_v32.rotary import apply_rope_pair, prepare_rotary_cache, rotary_frequencies
from operators.deepseek_v32.indexer.quantization import quantize_index


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
        self._q1_projection_stream = (
            torch.cuda.Stream(device=self.device)
            if linear_backend == "fp8"
            and self.device.type == "cuda"
            and torch.cuda.get_device_capability(self.device) == (9, 0)
            and (cfg.n_heads, cfg.qk_rope_head_dim) == (128, 64)
            else None
        )

    def _project_kv(self, x):
        cfg = self.cfg
        kv = self.wkv_a(x)
        latent, k_pe = kv.split((cfg.kv_lora_rank, cfg.qk_rope_head_dim), -1)
        latent = rms_norm(latent, self.kv_norm_weight, cfg.norm_eps)
        return kv, latent, k_pe

    def _project_index_k(self, x):
        return F.layer_norm(
            self.index_wk(x).float(),
            (self.cfg.index_head_dim,),
            self.index_norm_weight,
            self.index_norm_bias,
            self.cfg.norm_eps,
        ).bfloat16()

    @contextmanager
    def _projection_branches(self, x):
        side = self._q1_projection_stream
        if side is None or len(x) != 1 or not torch.cuda.is_current_stream_capturing():
            yield None
            return
        current = torch.cuda.current_stream(x.device)
        side.wait_stream(current)
        x.record_stream(side)
        try:
            with torch.cuda.stream(side):
                kv, latent, k_pe = self._project_kv(x)
                index_k = self._project_index_k(x)
            yield kv, latent, k_pe, index_k
        except BaseException as primary:
            try:
                current.wait_stream(side)
            except BaseException as cleanup:  # noqa: BLE001 -- retain both exception objects
                raise BaseExceptionGroup(
                    "Projection execution and stream join failed", [primary, cleanup]
                ) from None
            raise
        current.wait_stream(side)
        # The caller consumes side allocations after this captured join.
        for tensor in (kv, latent, index_k):
            tensor.record_stream(current)

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
        with self._projection_branches(x) as branches:
            qr = rms_norm(self.wq_a(x), self.q_norm_weight, cfg.norm_eps)
            angles = positions[:, None] * self.frequencies[None, :]
            rotary_cache = prepare_rotary_cache(angles) if hidden.is_cuda else None
            q = self.wq_b(qr).reshape(count, cfg.n_heads, -1)
            q_nope, q_pe = q.split((cfg.qk_nope_head_dim, cfg.qk_rope_head_dim), -1)
            if hidden.is_cuda:
                # Write directly to token-major Q with the original strided cuBLAS output.
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
            if branches is not None:
                index_q = self.index_wq(qr).reshape(count, cfg.index_n_heads, cfg.index_head_dim)
        if branches is None:
            _kv, latent, k_pe = self._project_kv(x)
        else:
            _kv, latent, k_pe, index_k = branches
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
        if branches is None:
            index_q = self.index_wq(qr).reshape(count, cfg.index_n_heads, cfg.index_head_dim)
            index_k = self._project_index_k(x)
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
