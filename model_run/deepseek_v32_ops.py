"""FlashInfer normalization/RoPE adapters for the synthetic V3.2 runner.

RoPE uses the official inference checkpoint's layouts: interleaved for MLA,
split-half for the indexer. Frequency construction is outside the timed step.
"""

import math

import torch


def yarn_cos_sin_cache(cfg, length: int, device) -> torch.Tensor:
    dim = cfg.qk_rope_head_dim
    freq = cfg.rope_theta ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    if cfg.max_seq_len > cfg.original_seq_len:

        def correction(rotations):
            return (
                dim
                * math.log(cfg.original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(cfg.rope_theta))
            )

        low = max(math.floor(correction(cfg.beta_fast)), 0)
        high = min(math.ceil(correction(cfg.beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, device=device).float() - low) / (high - low)).clamp(0, 1)
        freq = freq * (1 - ramp) + freq / cfg.rope_factor * ramp
    angles = torch.outer(torch.arange(length, device=device).float(), freq)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()


def attention_scale(cfg) -> float:
    scale = (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5
    if cfg.max_seq_len > cfg.original_seq_len:
        scale *= (1 + 0.1 * cfg.mscale * math.log(cfg.rope_factor)) ** 2
    return scale


def quantize_index(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One power-of-two dequantization scale per 128-element indexer head."""
    amax = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


class FlashInferV32Ops:
    def __init__(self, cfg, device="cuda"):
        try:
            from flashinfer import norm, rope
        except ImportError as exc:
            raise ImportError(
                "V3.2 norm/RoPE requires FlashInfer; install model_run/requirements.txt"
            ) from exc
        self.norm = norm
        self.rope = rope
        self.cfg = cfg
        # Unit affine parameters for the synthetic benchmark, not checkpoint weights.
        self.q_weight = torch.ones(cfg.q_lora_rank, device=device, dtype=torch.bfloat16)
        self.kv_weight = torch.ones(cfg.kv_lora_rank, device=device, dtype=torch.bfloat16)
        self.index_weight = torch.ones(cfg.index_head_dim, device=device, dtype=torch.float32)
        self.index_bias = torch.zeros_like(self.index_weight)
        self.cos_sin_cache = yarn_cos_sin_cache(cfg, cfg.max_seq_len, device)

    def q_norm(self, x):
        return self.norm.rmsnorm(x.contiguous(), self.q_weight, eps=self.cfg.norm_eps)

    def kv_norm(self, x):
        return self.norm.rmsnorm(x.contiguous(), self.kv_weight, eps=self.cfg.norm_eps)

    def index_norm(self, x):
        return self.norm.layernorm(
            x.contiguous(), self.index_weight, self.index_bias, eps=self.cfg.norm_eps
        )

    def apply_rope(self, q, k, positions, *, is_neox):
        """Rotate just the supplied 64-d slices; accept differing Q/K head counts."""
        dim = self.cfg.qk_rope_head_dim
        q_out, k_out = self.rope.apply_rope_with_cos_sin_cache(
            positions,
            q.reshape(q.shape[0], -1).contiguous(),
            k.reshape(k.shape[0], -1).contiguous(),
            dim,
            self.cos_sin_cache,
            is_neox=is_neox,
        )
        return q_out.reshape(q.shape), k_out.reshape(k.shape)
