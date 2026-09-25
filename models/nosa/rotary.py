"""NOSA static LongRoPE and Llama split-half rotation."""

import torch
from torch import nn

from models.nosa.config import NosaConfig
from operators import flashinfer


def _phases(config: NosaConfig, positions: torch.Tensor):
    exponent = (
        torch.arange(0, config.head_dim, 2, device=positions.device).float() / config.head_dim
    )
    factors = 1.0
    attention_factor = 1.0
    if config.rope_scaling is not None:
        factors = torch.tensor(
            config.rope_scaling["short_factor"], device=positions.device, dtype=torch.float32
        )
        attention_factor = config.rope_scaling["attention_factor"]
    inv_freq = 1.0 / (factors * config.rope_theta**exponent)
    angles = positions.float()[:, None] * inv_freq[None, :]
    return angles.cos() * attention_factor, angles.sin() * attention_factor


class NosaRotaryEmbedding(nn.Module):
    """Reuse FP32 static LongRoPE phases across layers, chunks and requests.

    Buffers are derived from configuration, never checkpoint state. Position
    slices on the warm path require neither a device allocation nor a host sync.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("positions", None, persistent=False)
        self.register_buffer("cos_sin_cache", None, persistent=False)
        self._cache_key = None

    def _apply(self, fn, recurse=True):
        # Module.to(dtype) must not round the FP32 phase cache to the model's
        # weight dtype. Drop derived state and rebuild at the next use instead.
        self.positions = None
        self.cos_sin_cache = None
        self._cache_key = None
        return super()._apply(fn, recurse=recurse)

    def forward(self, config: NosaConfig, start: int, end: int, *, device):
        if not 0 <= start <= end:
            raise ValueError("RoPE positions require 0 <= start <= end")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        scaling = config.rope_scaling
        signature = (
            config.head_dim,
            config.rope_theta,
            None if scaling is None else tuple(scaling["short_factor"]),
            1.0 if scaling is None else scaling["attention_factor"],
        )
        key = (device, signature)
        capacity = max(config.max_position_embeddings, end)
        if key != self._cache_key or self.positions is None or self.positions.numel() < capacity:
            positions = torch.arange(capacity, device=device, dtype=torch.int64)
            cos, sin = _phases(config, positions)
            self.positions = positions
            self.cos_sin_cache = torch.cat((cos, sin), dim=-1)
            self._cache_key = key
        return self.positions[start:end], self.cos_sin_cache


def _has_aligned_head_rows(x):
    """Dense heads within each row; allow gaps between packed QKV token rows."""
    return (
        x.ndim == 3
        and x.stride(2) == 1
        and x.stride(1) == x.shape[2]
        and x.stride(0) >= x.shape[1] * x.shape[2]
        and x.data_ptr() % 16 == 0
        and (x.stride(0) * x.element_size()) % 16 == 0
    )


def apply_rotary_qk(q, k, positions, cos_sin_cache):
    """Rotate fresh Q/K projections together, with one rounding at the store."""
    if (
        q.is_cuda
        and flashinfer.can_use_flashinfer(q)
        and q.dtype == k.dtype
        and q.device == k.device == positions.device == cos_sin_cache.device
        and _has_aligned_head_rows(q)
        and _has_aligned_head_rows(k)
        and q.shape[-1] in (64, 128, 256, 512)
    ):
        return flashinfer.apply_rope_with_cos_sin_cache(q, k, positions, cos_sin_cache)

    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]

    def rotate(x):
        # Preserve double precision for independent mathematical reference runs.
        working = x if x.dtype == torch.float64 else x.float()
        first, second = working.chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(
            x.dtype
        )

    return rotate(q), rotate(k)


def rotary_cos_sin(config: NosaConfig, positions: torch.Tensor, dtype: torch.dtype):
    """FP32 LongRoPE phases, then cast before applying Llama split-half rotation."""
    cos, sin = _phases(config, positions)
    return torch.cat((cos, cos), dim=-1).to(dtype), torch.cat((sin, sin), dim=-1).to(dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    first, second = x.chunk(2, dim=-1)
    return x * cos[:, None, :] + torch.cat((-second, first), dim=-1) * sin[:, None, :]
