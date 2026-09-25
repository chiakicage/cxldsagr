"""NOSA checkpoint metadata and supported configuration."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class NosaConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    bos_token_id: int = 1
    eos_token_id: tuple[int, ...] = (2,)
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if self.tie_word_embeddings:
            raise ValueError("This NOSA runner requires independent embedding and lm_head weights")
        for name in ("rms_norm_eps", "rope_theta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        eos = self.eos_token_id
        object.__setattr__(self, "eos_token_id", (eos,) if isinstance(eos, int) else tuple(eos))
        if any(token < 0 or token >= self.vocab_size for token in self.eos_token_id):
            raise ValueError("eos_token_id is outside the vocabulary")
        scaling = self.rope_scaling
        if scaling is not None:
            if scaling.get("rope_type", scaling.get("type")) != "longrope":
                raise ValueError("Only default RoPE and NOSA's static LongRoPE are supported")
            for name in ("short_factor", "long_factor"):
                factors = scaling.get(name, [])
                if len(factors) != self.head_dim // 2 or any(
                    not math.isfinite(x) or x <= 0 for x in factors
                ):
                    raise ValueError(f"LongRoPE {name} needs head_dim / 2 positive finite values")
            # NOSA-8B uses identical arrays. Changing frequencies after caching K
            # would invalidate that cache; do not silently generalize this policy.
            if list(scaling["short_factor"]) != list(scaling["long_factor"]):
                raise ValueError("NOSA static LongRoPE requires equal short_factor and long_factor")
            attention_factor = scaling.get("attention_factor")
            if (
                attention_factor is None
                or not math.isfinite(attention_factor)
                or attention_factor <= 0
            ):
                raise ValueError("NOSA LongRoPE requires an explicit positive attention_factor")

    @classmethod
    def from_dict(cls, data: dict):
        if data.get("model_type", "llama") != "llama":
            raise ValueError("Expected a NOSA Llama checkpoint")
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError("Only the NOSA SwiGLU (silu) MLP is supported")
        if data.get("pretraining_tp", 1) != 1:
            raise ValueError("pretraining_tp must be 1")
        values = {field.name: data[field.name] for field in fields(cls) if field.name in data}
        values.setdefault("head_dim", data["hidden_size"] // data["num_attention_heads"])
        values.setdefault("num_key_value_heads", data["num_attention_heads"])
        return cls(**values)

    @classmethod
    def from_pretrained(cls, model_path: str | Path):
        return cls.from_dict(json.loads((Path(model_path) / "config.json").read_text()))
