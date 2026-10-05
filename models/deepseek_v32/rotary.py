"""DeepSeek YaRN and distinct MLA/indexer rotary pairing."""

import math

import torch

from models.deepseek_v32.config import Config


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
