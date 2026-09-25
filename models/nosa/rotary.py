"""NOSA static LongRoPE and Llama split-half rotation."""

import torch

from models.nosa.config import NosaConfig


def rotary_cos_sin(config: NosaConfig, positions: torch.Tensor, dtype: torch.dtype):
    """FP32 LongRoPE phases, then cast before applying Llama split-half rotation."""
    exponent = (
        torch.arange(0, config.head_dim, 2, device=positions.device).float() / config.head_dim
    )
    factors = 1.0
    attention_factor = 1.0
    if config.rope_scaling is not None:
        factors = torch.tensor(config.rope_scaling["short_factor"], device=positions.device)
        attention_factor = config.rope_scaling["attention_factor"]
    inv_freq = 1.0 / (factors * config.rope_theta**exponent)
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return (angles.cos() * attention_factor).to(dtype), (angles.sin() * attention_factor).to(dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    first, second = x.chunk(2, dim=-1)
    return x * cos[:, None, :] + torch.cat((-second, first), dim=-1) * sin[:, None, :]
