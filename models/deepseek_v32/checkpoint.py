"""Lazy checkpoint tensor loading and BF16/FP8 linear weights."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F


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
