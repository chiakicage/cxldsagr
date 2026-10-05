"""DeepSeek checkpoint dimensions, quantization and rotary configuration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


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
