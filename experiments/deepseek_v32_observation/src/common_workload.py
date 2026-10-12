"""Resident checkpoint operator fixtures, captured from the production dense block.

The fixture builder executes production projection/MLP preparation once and
retains its leaf inputs. Independent leaf calls are not an additive model
timeline: the packed production MLP reuses one gate/up quantization, while its
independent linear cases each include their own quantization.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.deepseek_v32_mfu.src.operator_flops import (
    OperatorWork,
    batched_matmul_work,
    linear_work,
    non_matmul_work,
)
from models.deepseek_v32 import layers, projections
from models.deepseek_v32.checkpoint import CheckpointLinear, CheckpointReader
from models.deepseek_v32.layers import CheckpointBlock
from models.deepseek_v32.nonmatrix import _reference_rms_norm, residual_rms_norm
from models.deepseek_v32.rotary import apply_rope
from operators import flashinfer
from operators.deepseek_v32.indexer.quantization import reference_quantize_index
from operators.deepseek_v32.linear.fp8 import (
    reference_fp8_linear,
    reference_quantize_fp8_activation,
)
from operators.deepseek_v32.linear.quantization import quantize


@dataclass
class OperatorCase:
    name: str
    call: Callable[[], Any]
    work: OperatorWork
    verify: Callable[[Any], dict]
    metadata: dict = field(default_factory=dict)


def tensor_metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def sample_indices(count, limit, device):
    samples = min(count, limit)
    return (
        torch.arange(samples, device=device, dtype=torch.int64) * (count - 1) // max(samples - 1, 1)
    )


def sample_columns(count, device):
    """Three complete aligned blocks keep the FP8 block-scale identity intact."""
    blocks = sorted({0, (count - 1) // 256, (count - 1) // 128})
    return torch.cat(
        [
            torch.arange(block * 128, min((block + 1) * 128, count), device=device)
            for block in blocks
        ]
    )


def compare(actual, expected, *, rtol=8e-3, atol=0.0625, relative_l2_limit=None):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("Reference shape or dtype differs")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("Operator or reference produced a nonfinite value")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    difference = actual.float() - expected.float()
    relative_l2 = (difference.norm() / expected.float().norm().clamp_min(1e-30)).item()
    if relative_l2_limit is not None and relative_l2 >= relative_l2_limit:
        raise AssertionError(f"relative L2 {relative_l2} exceeds {relative_l2_limit}")
    return {
        "checked_shape": list(actual.shape),
        "bitwise_equal": bool(
            torch.equal(
                actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
            )
        ),
        "max_abs": difference.abs().max().item() if difference.numel() else 0.0,
        "relative_l2": relative_l2,
        "rtol": rtol,
        "atol": atol,
        "relative_l2_limit": relative_l2_limit,
    }


def compare_bits(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("Exact reference shape or dtype differs")
    if not torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ):
        raise AssertionError("Exact reference bytes differ")
    return {"checked_shape": list(actual.shape), "bitwise_equal": True}


def linear_verifier(linear, source):
    def verify(actual):
        rows = sample_indices(len(source), 8, source.device)
        columns = sample_columns(linear.weight.shape[0], source.device)
        if linear.scales is not None:
            blocks = torch.div(columns[::128], 128, rounding_mode="floor")
            expected = reference_fp8_linear(
                source[rows], linear.weight[columns], linear.scales[blocks]
            )
            reference = "independent UE8M0 activation and block-weight dequantization, FP32 matmul"
        else:
            expected = F.linear(source[rows].float(), linear.weight[columns].float()).to(
                actual.dtype
            )
            reference = "FP32 matmul of BF16 operands, one final BF16 conversion"
        return {
            **compare(actual[rows][:, columns].contiguous(), expected.contiguous()),
            "reference": reference,
            "coverage": "sampled operator output; not full output or complete-model acceptance",
            "row_indices": rows.cpu().tolist(),
            "column_indices": columns.cpu().tolist(),
            "full_output_shape": list(actual.shape),
            "input_channels_checked": source.shape[-1],
        }

    return verify


def float_linear_verifier(source, weight):
    def verify(actual):
        rows = sample_indices(len(source), 8, source.device)
        columns = sample_columns(weight.shape[0], source.device)
        expected = F.linear(source[rows].double(), weight[columns].double()).to(actual.dtype)
        return {
            **compare(
                actual[rows][:, columns].contiguous(), expected.contiguous(), rtol=2e-4, atol=1e-4
            ),
            "reference": "FP64 matmul of the actual FP32 input and weight",
            "coverage": "sampled output, all input channels",
            "row_indices": rows.cpu().tolist(),
            "column_indices": columns.cpu().tolist(),
            "full_output_shape": list(actual.shape),
        }

    return verify


def bmm_verifier(source, weight):
    def verify(actual):
        rows = sample_indices(source.shape[1], 8, source.device)
        expected = torch.bmm(source[:, rows].float(), weight.float()).to(actual.dtype)
        return {
            **compare(actual[:, rows].contiguous(), expected.contiguous()),
            "reference": "FP32 per-head batched matmul of BF16 operands",
            "coverage": "all heads, sampled token rows, all input/output channels",
            "row_indices": rows.cpu().tolist(),
            "full_output_shape": list(actual.shape),
        }

    return verify


def prepared_fp8_verifier(data, activation_scales, weight, weight_scales):
    """Check the GEMM separately from quantization using its exact FP8 operands."""

    def verify(actual):
        rows = sample_indices(len(data), 8, data.device)
        columns = sample_columns(weight.shape[0], data.device)
        x = data[rows].float() * activation_scales[rows].repeat_interleave(128, -1)
        scales = weight_scales[columns // 128].repeat_interleave(128, -1)
        expected = (x @ (weight[columns].float() * scales).T).bfloat16()
        return {
            **compare(actual[rows][:, columns].contiguous(), expected),
            "reference": "independent FP32 matmul of the exact prepared FP8 data and FP32 block scales",
            "coverage": "sampled rows and aligned output blocks; all input channels",
            "row_indices": rows.cpu().tolist(),
            "column_indices": columns.cpu().tolist(),
            "full_output_shape": list(actual.shape),
        }

    return verify


def add_fp8_linear_leaves(
    owner, name, instance, source, *, family, prepared=None, quantization_parent=None
):
    """Capture the public GEMM operands of one real CheckpointLinear invocation."""
    import deep_gemm

    captured = {}
    original = deep_gemm.fp8_gemm_nt

    def capture(a, b, output, **kwargs):
        captured.update(a=a, b=b, output=output, kwargs=kwargs)
        return original(a, b, output, **kwargs)

    if prepared is None:
        with patch.object(deep_gemm, "fp8_gemm_nt", capture):
            instance(source)
    else:
        captured.update(prepared)
    data, scales = captured["a"]
    weight, weight_scales = captured["b"]
    rows, k = data.shape
    n = weight.shape[0]
    consumer = scales.stride(0) == 1
    quantize_call = lambda: quantize(source.reshape(-1, k), consumer_scale_layout=consumer)
    if quantization_parent is None:
        owner._add(
            name + "_activation_quantize",
            quantize_call,
            non_matmul_work(name + "_activation_quantize"),
            quantization_verifier(source.reshape(-1, k), reference_quantize_fp8_activation),
            inputs=(source,),
            operator_kind="non_gemm",
            operator_family="quantization",
            parent_api=name,
            consumer_scale_layout=consumer,
            boundary="production activation quantization on the exact linear input; GEMM excluded",
        )
    transformed = deep_gemm.transform_sf_into_required_layout(
        scales, rows, k, (1, 128, 128), is_sfa=True
    )
    layout_changed = transformed.data_ptr() != scales.data_ptr()
    if layout_changed:
        owner._add(
            name + "_scale_layout",
            lambda: deep_gemm.transform_sf_into_required_layout(
                scales, rows, k, (1, 128, 128), is_sfa=True
            ),
            non_matmul_work(name + "_scale_layout"),
            lambda actual: {**compare_bits(actual, scales), "coverage": "all logical scale values"},
            inputs=(scales,),
            operator_kind="non_gemm",
            operator_family="scale_layout",
            parent_api=name,
            boundary="official SM90 activation-scale layout conversion; GEMM excluded",
            **(
                {"supported_condition": "S > 1"}
                if (k, n)
                in {
                    (7168, 1536),
                    (1536, 24576),
                    (7168, 576),
                    (1536, 8192),
                    (7168, 128),
                    (16384, 7168),
                    (7168, 18432),
                    (18432, 7168),
                }
                else {}
            ),
        )
    output = torch.empty_strided(
        captured["output"].shape,
        captured["output"].stride(),
        device=source.device,
        dtype=captured["output"].dtype,
    )

    def gemm():
        original((data, transformed), (weight, weight_scales), output, **captured["kwargs"])
        return output

    owner._add(
        name + "_gemm",
        gemm,
        linear_work(
            name + "_gemm",
            rows=rows,
            in_features=k,
            out_features=n,
            precision="fp8",
            local_fp8_kernel=True,
        ),
        prepared_fp8_verifier(data, scales, weight, weight_scales),
        inputs=(data, transformed, weight, weight_scales, output),
        operator_kind="gemm",
        operator_family=family,
        parent_api=name,
        quantization_operation=(quantization_parent or name) + "_activation_quantize",
        scale_layout_operation=name + "_scale_layout" if layout_changed else None,
        scale_layout_no_kernel=not layout_changed,
        boundary="official DeepGEMM GEMM with prepared FP8 operands/scales/output; quantization and scale-layout excluded",
    )


def norm_verifier(source, weight, eps):
    def verify(actual):
        expected = _reference_rms_norm(source, weight, eps)
        return {
            **compare(actual, expected, rtol=8e-3, atol=0.015625, relative_l2_limit=5e-5),
            "reference": "production CPU-oracle RMSNorm equations in independent FP32 Torch ops",
            "coverage": "full output",
        }

    return verify


def residual_norm_verifier(source, residual, weight, eps):
    def verify(actual):
        summed = source.float() + residual.float()
        expected = _reference_rms_norm(summed, weight, eps).bfloat16()
        return {
            "normalized": compare(
                actual[0], expected, rtol=8e-3, atol=0.015625, relative_l2_limit=5e-5
            ),
            "saved_residual": compare_bits(actual[1], summed.bfloat16()),
            "reference": "unrounded FP32 residual sum, FP32 RMSNorm, independent BF16 outputs",
            "coverage": "full output and saved residual",
        }

    return verify


def quantization_verifier(source, reference):
    def verify(actual):
        expected = reference(source)
        return {
            "data": compare_bits(actual[0], expected[0]),
            "scales": compare_bits(actual[1], expected[1]),
            "reference": "independent FP32 Torch UE8M0 reduction, scaling and FP8 conversion",
            "coverage": "full FP8 data and FP32 scales",
        }

    return verify


def rope_verifier(query, key, angles, *, interleaved):
    def verify(actual):
        checks = []
        width = angles.shape[-1] * 2
        for source, result in zip((query, key), actual, strict=True):
            expected = apply_rope(source[..., :width], angles, interleaved=interleaved)
            check = compare(
                result[..., :width].contiguous(),
                expected.contiguous(),
                rtol=8e-3,
                atol=1e-4,
                relative_l2_limit=5e-4,
            )
            if source.shape[-1] > width:
                check["unrotated_suffix"] = compare_bits(result[..., width:], source[..., width:])
            checks.append(check)
        return {
            "outputs": checks,
            "reference": "independent model FP32 RoPE equations",
            "coverage": "full Q and K",
        }

    return verify


class CommonWorkload:
    def __init__(self, s, device, seed, checkpoint):
        if type(s) is not int or s < 1:
            raise ValueError("S must be a positive integer")
        self.s, self.device, self.seed = s, torch.device(device), seed
        if self.device.type != "cuda" or torch.cuda.get_device_capability(self.device) != (9, 0):
            raise RuntimeError("DeepSeek resident operator observation requires SM90/Hopper")
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("Observation FP32 matmul requires allow_tf32=False")
        self.checkpoint = Path(checkpoint)
        self.reader = CheckpointReader(checkpoint)
        self.block = CheckpointBlock(
            checkpoint, 0, self.device, reader=self.reader, capacity=s, allocate_cache=False
        )
        self.attention = self.block.attention_layer
        self.cfg = self.block.cfg
        self.embedding = self.reader.get_tensor("model.embed_tokens.weight").to(self.device)
        self.head = self.reader.get_tensor("lm_head.weight").to(self.device)
        self.final_norm = self.reader.get_tensor("model.norm.weight").to(self.device)
        generator = self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.hidden = torch.randn(
            (s, self.cfg.dim), device=self.device, generator=generator
        ).bfloat16()
        self.residual = torch.randn(
            (s, self.cfg.dim), device=self.device, generator=generator
        ).bfloat16()
        self.token_ids = torch.randint(
            self.cfg.vocab_size, (s,), device=self.device, generator=generator
        )
        # Position values represent a long-context call; S-only fixtures do not
        # scan H and do not claim task-derived hidden-state distributions.
        self.positions = torch.arange(65536, 65536 + s, device=self.device).float()
        self.cases = []
        self._capture_cases()
        self.metadata = {
            "family": "common",
            "S": s,
            "checkpoint": str(self.checkpoint),
            "source_layer": 0,
            "input_distribution": "seeded Gaussian BF16 activations, real checkpoint weights",
            "positions": "65536 + arange(S); position values do not alter operator shapes",
            "no_cache_allocation": self.block.cache is None and self.block.attention is None,
            "measurement_semantics": "independent operator fixtures; overlapping module aggregates are not additive",
        }

    def _add(self, name, call, work, verify, *, inputs=(), scope="operator", **metadata):
        kind = metadata.get(
            "operator_kind", "non_gemm" if work.useful_flops is None else "api_aggregate"
        )
        if "operator_family" not in metadata:
            if name.startswith("mlp_") or name == "dense_mlp":
                family = "dense_mlp"
            elif name in ("q_absorb", "v_expand"):
                family = "projection_transform"
                kind = "gemm"
            elif name == "o_proj":
                family = "output_projection"
            elif name.endswith("_proj") or name == "attention_projection":
                family = "input_projection"
            elif "quantize" in name:
                family = "quantization"
            elif "norm" in name:
                family = "norm"
            elif "cast" in name:
                family = "cast"
            elif "scale" in name:
                family = "scale"
            elif "rotary" in name or "rope" in name:
                family = "rope"
            elif name in ("embedding", "lm_head"):
                family = name
            else:
                family = "layout"
            metadata["operator_family"] = family
        metadata.setdefault("operator_kind", kind)
        metadata.setdefault(
            "parent_api",
            {
                "index_weights_input_cast": "index_weights_proj",
                "index_layer_norm_input_cast": "index_layer_norm",
                "index_layer_norm_output_cast": "index_layer_norm",
            }.get(name, name),
        )
        scope = "module_aggregate" if metadata["operator_kind"] == "api_aggregate" else "operator"
        self.cases.append(
            OperatorCase(
                name,
                call,
                work,
                verify,
                {
                    "family": "common",
                    "S": self.s,
                    "source_layer": 0,
                    "measure_scope": scope,
                    "scope": scope,
                    "not_additive": True,
                    "inputs": [tensor_metadata(tensor) for tensor in inputs],
                    **metadata,
                },
            )
        )

    def _capture_cases(self):
        attn, cfg, mlp = self.attention, self.cfg, self.block.mlp
        linear_names = {
            id(getattr(attn, attribute)): name
            for attribute, name in (
                ("wq_a", "q_a_proj"),
                ("wq_b", "q_b_proj"),
                ("wkv_a", "kv_a_proj"),
                ("index_wq", "index_q_proj"),
                ("index_wk", "index_k_proj"),
                ("wo", "o_proj"),
            )
        }
        linear_names.update(
            {id(getattr(mlp, name)): "mlp_" + name for name in ("gate", "up", "down")}
        )
        norms = {
            attn.input_norm_weight.data_ptr(): "input_rms_norm",
            attn.q_norm_weight.data_ptr(): "query_rms_norm",
            attn.kv_norm_weight.data_ptr(): "kv_rms_norm",
        }
        captured = {}
        original_linear, original_bmm = CheckpointLinear.__call__, torch.bmm
        original_norm = projections.rms_norm
        original_layer_norm = F.layer_norm
        original_rope, original_rope_into = projections.apply_rope_pair, flashinfer.rotary_pair_into
        original_quantize = projections.quantize_index
        original_trig = projections.prepare_rotary_cache
        original_silu = layers.silu_mul_packed
        original_mul = torch.Tensor.__mul__
        original_float, original_bfloat16 = torch.Tensor.float, torch.Tensor.bfloat16
        original_cat = torch.cat

        def linear(instance, x, *args, **kwargs):
            name = linear_names.get(id(instance))
            if name is not None:
                captured[name] = (instance, x)
                import deep_gemm

                original_gemm = deep_gemm.fp8_gemm_nt

                def gemm(a, b, output, **gemm_kwargs):
                    captured["matrix:" + name] = {
                        "a": a,
                        "b": b,
                        "output": output,
                        "kwargs": gemm_kwargs,
                    }
                    return original_gemm(a, b, output, **gemm_kwargs)

                with patch.object(deep_gemm, "fp8_gemm_nt", gemm):
                    return original_linear(instance, x, *args, **kwargs)
            return original_linear(instance, x, *args, **kwargs)

        def bmm(a, b, *args, **kwargs):
            if b.data_ptr() in (attn.wk_b.data_ptr(), attn.wv_b.data_ptr()):
                captured["q_absorb" if b.data_ptr() == attn.wk_b.data_ptr() else "v_expand"] = (
                    a,
                    b,
                    kwargs,
                )
            return original_bmm(a, b, *args, **kwargs)

        def norm(x, weight, eps):
            if weight.data_ptr() in norms:
                captured[norms[weight.data_ptr()]] = (x, weight, eps)
            return original_norm(x, weight, eps)

        def layer_norm(x, shape, weight, bias, eps):
            captured["index_layer_norm"] = (x, shape, weight, bias, eps)
            return original_layer_norm(x, shape, weight, bias, eps)

        def rope(q, k, angles, **kwargs):
            captured["index_rope"] = (q, k, angles, kwargs)
            return original_rope(q, k, angles, **kwargs)

        def rope_into(q, k, positions, trig, out, **kwargs):
            captured["main_rope"] = (q, k, positions, trig, out, kwargs)
            return original_rope_into(q, k, positions, trig, out, **kwargs)

        def index_quantize(x, fmt):
            captured["index_query_quantize" if x.ndim == 3 else "index_key_quantize"] = (x, fmt)
            return original_quantize(x, fmt)

        def trig(angles):
            captured["rotary_cache"] = angles
            return original_trig(angles)

        def silu(packed):
            captured["mlp_silu_mul"] = packed
            return original_silu(packed)

        def mul(left, right):
            if (
                left.shape == (self.s, 1)
                and isinstance(right, torch.Tensor)
                and right.shape == (1, cfg.qk_rope_head_dim // 2)
            ):
                captured["rotary_angles"] = (left, right)
            elif left.shape == (self.s, cfg.index_n_heads):
                if isinstance(right, (float, int)) and right == cfg.index_n_heads**-0.5:
                    captured["index_weights_head_scale"] = (left, right)
                elif isinstance(right, torch.Tensor) and right.shape == left.shape:
                    captured["index_weights_query_scale"] = (left, right)
                elif isinstance(right, (float, int)) and right == cfg.index_head_dim**-0.5:
                    captured["index_weights_dim_scale"] = (left, right)
            return original_mul(left, right)

        def float_(value, *args, **kwargs):
            if value.dtype == torch.bfloat16 and value.shape == (self.s, cfg.index_head_dim):
                captured["index_layer_norm_input_cast"] = value
            elif value.dtype == torch.bfloat16 and value.shape == (self.s, cfg.dim):
                captured["index_weights_input_cast"] = value
            return original_float(value, *args, **kwargs)

        def bfloat16(value, *args, **kwargs):
            if value.dtype == torch.float32 and value.shape == (self.s, cfg.index_head_dim):
                captured["index_layer_norm_output_cast"] = value
            return original_bfloat16(value, *args, **kwargs)

        def cat(tensors, *args, **kwargs):
            if (
                len(tensors) == 2
                and tensors[0].shape == (self.s, cfg.kv_lora_rank)
                and tensors[1].shape == (self.s, cfg.qk_rope_head_dim)
            ):
                captured["main_kv_concat"] = (tensors, args, kwargs)
            return original_cat(tensors, *args, **kwargs)

        with ExitStack() as stack:
            for target, name, replacement in (
                (CheckpointLinear, "__call__", linear),
                (torch, "bmm", bmm),
                (projections, "rms_norm", norm),
                (F, "layer_norm", layer_norm),
                (projections, "apply_rope_pair", rope),
                (flashinfer, "rotary_pair_into", rope_into),
                (projections, "quantize_index", index_quantize),
                (projections, "prepare_rotary_cache", trig),
                (layers, "silu_mul_packed", silu),
                (torch.Tensor, "__mul__", mul),
                (torch.Tensor, "float", float_),
                (torch.Tensor, "bfloat16", bfloat16),
                (torch, "cat", cat),
            ):
                stack.enter_context(patch.object(target, name, replacement))
            projected = attn.project_positions(self.hidden, self.positions)
            self.attention_input = torch.randn(
                (self.s, cfg.n_heads, cfg.kv_lora_rank),
                device=self.device,
                generator=self.generator,
            ).bfloat16()
            attn.output(self.attention_input)
            mlp(self.hidden)
        self.projected = projected
        self.captured = captured

        for name in (
            "rotary_angles",
            "index_weights_head_scale",
            "index_weights_query_scale",
            "index_weights_dim_scale",
        ):
            left, right = captured[name]
            self._add(
                name,
                lambda left=left, right=right: original_mul(left, right),
                non_matmul_work(name),
                lambda actual, left=left, right=right: {
                    **compare_bits(
                        actual,
                        (
                            left.double()
                            * (
                                right.double()
                                if isinstance(right, torch.Tensor)
                                else torch.tensor(
                                    right, device=left.device, dtype=left.dtype
                                ).double()
                            )
                        ).to(left.dtype),
                    ),
                    "coverage": "full output",
                    "reference": "FP64 multiply of actual FP32 operands rounded once to FP32",
                },
                inputs=(left, right) if isinstance(right, torch.Tensor) else (left,),
                boundary="captured production FP32 multiplication leaf",
                scalar=right if isinstance(right, (float, int)) else None,
            )
        for name, conversion in (
            ("index_layer_norm_input_cast", original_float),
            ("index_layer_norm_output_cast", original_bfloat16),
            ("index_weights_input_cast", original_float),
        ):
            source = captured[name]
            self._add(
                name,
                lambda source=source, conversion=conversion: conversion(source),
                non_matmul_work(name),
                lambda actual, source=source, conversion=conversion: {
                    **compare_bits(actual, conversion(source)),
                    "coverage": "full output",
                },
                inputs=(source,),
                boundary="captured production activation dtype conversion leaf",
            )
        tensors, concat_args, concat_kwargs = captured["main_kv_concat"]
        self._add(
            "main_kv_concat",
            lambda: original_cat(tensors, *concat_args, **concat_kwargs),
            non_matmul_work("main_kv_concat"),
            lambda actual: {
                "latent": compare_bits(actual[:, : cfg.kv_lora_rank], tensors[0]),
                "rope_key": compare_bits(actual[:, cfg.kv_lora_rank :], tensors[1]),
                "coverage": "full packed main KV record",
            },
            inputs=tuple(tensors),
            boundary="production concatenation of normalized BF16 latent and RoPE key",
        )

        for name in linear_names.values():
            instance, source = captured[name]
            n, k = instance.weight.shape
            self._add(
                name,
                lambda instance=instance, source=source: instance(source),
                linear_work(
                    name,
                    rows=self.s,
                    in_features=k,
                    out_features=n,
                    precision="fp8",
                    local_fp8_kernel=True,
                ),
                linear_verifier(instance, source),
                inputs=(source, instance.weight),
                boundary="independent CheckpointLinear API including activation quantization and scale preparation",
                production_packed_mlp_reuses_gate_up_quantization=name in ("mlp_gate", "mlp_up"),
            )
            family = (
                "dense_mlp"
                if name.startswith("mlp_")
                else ("output_projection" if name == "o_proj" else "input_projection")
            )
            add_fp8_linear_leaves(
                self,
                name,
                instance,
                source,
                family=family,
                prepared=captured["matrix:" + name],
                quantization_parent="mlp_gate" if name == "mlp_up" else None,
            )
        for name in ("q_absorb", "v_expand"):
            source, weight, kwargs = captured[name]
            self._add(
                name,
                lambda source=source, weight=weight, kwargs=kwargs: torch.bmm(
                    source, weight, **kwargs
                ),
                batched_matmul_work(
                    name,
                    batch=source.shape[0],
                    m=source.shape[1],
                    n=weight.shape[-1],
                    k=source.shape[-1],
                ),
                bmm_verifier(source, weight),
                inputs=(source, weight, kwargs["out"]),
                boundary="production cuBLAS BMM with token-major destination strides; caller allocation excluded",
                operator_kind="gemm",
            )
        for name in norms.values():
            source, weight, eps = captured[name]
            self._add(
                name,
                lambda source=source, weight=weight, eps=eps: original_norm(source, weight, eps),
                non_matmul_work(name),
                norm_verifier(source, weight, eps),
                inputs=(source, weight),
                boundary="production RMSNorm adapter including any layout preparation",
            )
        x, shape, weight, bias, eps = captured["index_layer_norm"]
        self._add(
            "index_layer_norm",
            lambda x=x, shape=shape, weight=weight, bias=bias, eps=eps: original_layer_norm(
                x, shape, weight, bias, eps
            ),
            non_matmul_work("index_layer_norm"),
            lambda actual, x=x, shape=shape, weight=weight, bias=bias, eps=eps: {
                **compare(
                    actual,
                    original_layer_norm(
                        x.double(), shape, weight.double(), bias.double(), eps
                    ).float(),
                    rtol=2e-4,
                    atol=1e-5,
                ),
                "coverage": "full output",
                "reference": "FP64 Torch LayerNorm",
            },
            inputs=(x, weight, bias),
            boundary="production FP32 LayerNorm leaf; surrounding BF16/FP32 casts excluded",
        )
        q, k, positions, trig_cache, out, kwargs = captured["main_rope"]
        angles = captured["rotary_cache"]
        self._add(
            "main_rope",
            lambda: original_rope_into(q, k, positions, trig_cache, out, **kwargs),
            non_matmul_work("main_rope"),
            rope_verifier(q, k, angles, interleaved=True),
            inputs=(q, k, positions, trig_cache, out),
            boundary="production direct-output main MLA RoPE adapter",
        )
        iq, ik, index_angles, index_kwargs = captured["index_rope"]
        self._add(
            "index_rope",
            lambda: original_rope(iq, ik, index_angles, **index_kwargs),
            non_matmul_work("index_rope"),
            rope_verifier(iq, ik, index_angles, interleaved=False),
            inputs=(iq, ik, index_angles),
            boundary="production split-half indexer RoPE, preserving unrotated suffix",
        )
        self._add(
            "rotary_cache",
            lambda: original_trig(angles),
            non_matmul_work("rotary_cache"),
            lambda actual: {
                "positions": compare_bits(actual[0], torch.arange(self.s, device=self.device)),
                "trig": compare_bits(actual[1], torch.cat((angles.cos(), angles.sin()), -1)),
                "coverage": "full output",
            },
            inputs=(angles,),
            boundary="production per-call cos/sin cache construction; angle multiplication excluded",
            operator_kind="api_aggregate",
            operator_family="rope",
        )
        cosine, sine = angles.cos(), angles.sin()
        for name, call, expected in (
            (
                "rotary_positions",
                lambda: torch.arange(len(angles), device=self.device, dtype=torch.int64),
                torch.arange(len(angles), device=self.device, dtype=torch.int64),
            ),
            ("rotary_cos", lambda: angles.cos(), cosine),
            ("rotary_sin", lambda: angles.sin(), sine),
            (
                "rotary_trig_concat",
                lambda: torch.cat((cosine, sine), -1),
                torch.cat((cosine, sine), -1),
            ),
        ):
            self._add(
                name,
                call,
                non_matmul_work(name),
                lambda actual, expected=expected: {
                    **compare_bits(actual, expected),
                    "coverage": "full output",
                    "reference": "accepted production trig/position fixture on unchanged input",
                },
                inputs=(angles,)
                if name in ("rotary_cos", "rotary_sin")
                else ((cosine, sine) if name == "rotary_trig_concat" else ()),
                operator_kind="non_gemm",
                operator_family="layout"
                if name in ("rotary_positions", "rotary_trig_concat")
                else "rope",
                parent_api="rotary_cache",
                boundary="one production trig-cache construction operation; other operations excluded",
            )
        for name in ("index_query_quantize", "index_key_quantize"):
            source, fmt = captured[name]
            self._add(
                name,
                lambda source=source, fmt=fmt: original_quantize(source, fmt),
                non_matmul_work(name),
                quantization_verifier(
                    source, lambda source, fmt=fmt: reference_quantize_index(source, fmt)
                ),
                inputs=(source,),
                boundary="production FP8 indexer quantization API, full data/scales",
            )
        for width in (
            cfg.dim,
            cfg.q_lora_rank,
            cfg.n_heads * cfg.v_head_dim,
            cfg.intermediate_size,
        ):
            source = torch.randn(
                (self.s, width), device=self.device, generator=self.generator
            ).bfloat16()
            for consumer in (False, True) if self.s == 1 else (False,):
                name = f"linear_activation_quantize_k{width}" + (
                    "_consumer_scales" if consumer else ""
                )
                self._add(
                    name,
                    lambda source=source, consumer=consumer: quantize(
                        source, consumer_scale_layout=consumer
                    ),
                    non_matmul_work(name),
                    quantization_verifier(source, reference_quantize_fp8_activation),
                    inputs=(source,),
                    boundary="production activation quantization API; also included in independent FP8 linear cases",
                    consumer_scale_layout=consumer,
                    supported_condition="S == 1" if consumer else "all requested S",
                )
        packed = captured["mlp_silu_mul"]
        self._add(
            "mlp_silu_mul",
            lambda: original_silu(packed),
            non_matmul_work("mlp_silu_mul"),
            lambda actual: {
                **compare(
                    actual,
                    (
                        F.silu(packed[:, : cfg.intermediate_size].float())
                        * packed[:, cfg.intermediate_size :].float()
                    ).bfloat16(),
                    rtol=8e-3,
                    atol=1e-5,
                    relative_l2_limit=1e-5,
                ),
                "coverage": "full output",
                "reference": "FP32 Torch SiLU and multiply, one BF16 conversion",
            },
            inputs=(packed,),
            boundary="production packed SwiGLU API; gate/up packing excluded because producer supplies packed output",
        )
        for name, weight in (
            ("post_attention_residual_norm", self.block.post_norm_weight),
            ("final_residual_norm", self.final_norm),
        ):
            self._add(
                name,
                lambda weight=weight: residual_rms_norm(
                    self.hidden, self.residual, weight, cfg.norm_eps
                ),
                non_matmul_work(name),
                residual_norm_verifier(self.hidden, self.residual, weight, cfg.norm_eps),
                inputs=(self.hidden, self.residual, weight),
                boundary="production residual RMSNorm, full S normalized output and saved residual",
            )
        self._add(
            "embedding",
            lambda: F.embedding(self.token_ids, self.embedding),
            non_matmul_work("embedding"),
            lambda actual: {
                **compare_bits(actual, self.embedding[self.token_ids]),
                "coverage": "full output",
                "reference": "resident checkpoint table integer row selection",
            },
            inputs=(self.token_ids, self.embedding),
            boundary="production resident GPU embedding gather; no CPU mmap or H2D",
        )
        index_weight_input = captured["index_weights_input_cast"]
        self._add(
            "index_weights_proj",
            lambda: F.linear(index_weight_input.float(), attn.index_head_weight),
            linear_work(
                "index_weights_proj",
                rows=self.s,
                in_features=cfg.dim,
                out_features=cfg.index_n_heads,
                precision="fp32",
            ),
            float_linear_verifier(index_weight_input.float(), attn.index_head_weight),
            inputs=(index_weight_input, attn.index_head_weight),
            boundary="production FP32 head-weight linear and BF16-to-FP32 input conversion; subsequent scale factors excluded",
        )
        index_float = index_weight_input.float()
        index_output = torch.empty(
            (self.s, cfg.index_n_heads), device=self.device, dtype=torch.float32
        )
        self._add(
            "index_weights_gemm",
            lambda: torch.mm(index_float, attn.index_head_weight.T, out=index_output),
            linear_work(
                "index_weights_gemm",
                rows=self.s,
                in_features=cfg.dim,
                out_features=cfg.index_n_heads,
                precision="fp32",
            ),
            float_linear_verifier(index_float, attn.index_head_weight),
            inputs=(index_float, attn.index_head_weight, index_output),
            operator_kind="gemm",
            operator_family="input_projection",
            parent_api="index_weights_proj",
            boundary="FP32 head-weight GEMM with prepared FP32 input/output; input cast excluded",
        )
        head_linear = type("BF16Head", (), {"weight": self.head, "scales": None})()

        def verify_head(actual):
            if actual.dtype != torch.float32 or not torch.equal(actual, actual.bfloat16().float()):
                raise AssertionError("LM head output must preserve BF16 values in FP32 storage")
            return linear_verifier(head_linear, self.hidden)(actual.bfloat16())

        self._add(
            "lm_head",
            lambda: F.linear(self.hidden, self.head).float(),
            linear_work(
                "lm_head",
                rows=self.s,
                in_features=cfg.dim,
                out_features=cfg.vocab_size,
                precision="bf16",
            ),
            verify_head,
            inputs=(self.hidden, self.head),
            boundary="production BF16 LM head plus FP32 output conversion on all S rows; not last-token model boundary",
        )
        head_output = torch.empty(
            (self.s, cfg.vocab_size), device=self.device, dtype=torch.bfloat16
        )
        self._add(
            "lm_head_gemm",
            lambda: torch.mm(self.hidden, self.head.T, out=head_output),
            linear_work(
                "lm_head_gemm",
                rows=self.s,
                in_features=cfg.dim,
                out_features=cfg.vocab_size,
                precision="bf16",
            ),
            linear_verifier(head_linear, self.hidden),
            inputs=(self.hidden, self.head, head_output),
            operator_kind="gemm",
            operator_family="lm_head",
            parent_api="lm_head",
            boundary="BF16 LM-head GEMM with prepared output; FP32 output conversion excluded",
        )
        torch.mm(self.hidden, self.head.T, out=head_output)
        self._add(
            "lm_head_output_cast",
            lambda: head_output.float(),
            non_matmul_work("lm_head_output_cast"),
            lambda actual: {**compare_bits(actual, head_output.float()), "coverage": "full output"},
            inputs=(head_output,),
            operator_kind="non_gemm",
            operator_family="cast",
            parent_api="lm_head",
            boundary="LM-head BF16-to-FP32 output conversion; GEMM excluded",
        )

        def verify_mlp(actual):
            before = mlp.pack_gate_up
            try:
                mlp.pack_gate_up = False
                expected = mlp(self.hidden)
            finally:
                mlp.pack_gate_up = before
            return {
                **compare_bits(actual, expected),
                "coverage": "full module output against unpacked production; matrix accuracy covered by independent sampled leaf checks",
                "reference": "same production MLP without shared gate/up quantization and packed output",
            }

        total = sum(
            case.work.useful_flops
            for case in self.cases
            if case.name in ("mlp_gate", "mlp_up", "mlp_down")
        )
        self._add(
            "dense_mlp",
            lambda: mlp(self.hidden),
            OperatorWork(
                "dense_mlp",
                "fp8",
                total,
                None,
                "6 * S * dim * intermediate_size",
                dimensions={
                    "S": self.s,
                    "dim": cfg.dim,
                    "intermediate_size": cfg.intermediate_size,
                },
                notes="Whole production MLP; quantization and SwiGLU included in time, excluded from matrix FLOPs.",
            ),
            verify_mlp,
            inputs=(self.hidden,),
            scope="module_aggregate",
            boundary="complete production packed dense MLP; overlaps independent mlp_gate/up/down/silu cases",
        )
        projection_names = (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj",
            "index_q_proj",
            "index_k_proj",
            "q_absorb",
            "index_weights_proj",
        )
        precision_flops = {}
        for case in self.cases:
            if case.name in projection_names:
                precision_flops[case.work.precision] = (
                    precision_flops.get(case.work.precision, 0) + case.work.useful_flops
                )

        def verify_projection(actual):
            result = {}
            for name in ("q", "kv", "index_q", "index_k", "index_weights", "index_scale"):
                result[name] = compare_bits(getattr(actual, name), getattr(projected, name))
            return {
                "outputs": result,
                "coverage": "full production projection replay; arithmetic independently validated by leaf checks",
                "reference": "original production fixture outputs on the same unchanged input",
            }

        self._add(
            "attention_projection",
            lambda: attn.project_positions(self.hidden, self.positions),
            OperatorWork(
                "attention_projection",
                "mixed",
                sum(precision_flops.values()),
                None,
                "sum of query/KV/indexer projection linears, absorbed Q BMM and FP32 indexer weights projection",
                dimensions={
                    "S": self.s,
                    **{f"{name}_flops": value for name, value in precision_flops.items()},
                },
                notes="Complete production projection, including all norms, casts, RoPE, quantization, scales, concatenation and API-internal copies. Mixed precision ideal-time sum.",
            ),
            verify_projection,
            inputs=(self.hidden, self.positions),
            scope="module_aggregate",
            precision_flops=precision_flops,
            boundary="complete no-cache production attention projection; overlaps all projection components",
        )

    def input_identity(self):
        tensors = {
            "hidden": self.hidden,
            "residual": self.residual,
            "token_ids": self.token_ids,
            "positions": self.positions,
            "attention_input": self.attention_input,
        }
        inputs = {}
        for name, tensor in tensors.items():
            raw = tensor.contiguous().view(torch.uint8).flatten()
            positions = sample_indices(len(raw), 4096, tensor.device)
            payload = raw[positions].cpu().numpy().tobytes()
            metadata = tensor_metadata(tensor)
            metadata.pop("device")
            inputs[name] = {
                **metadata,
                "sampled_bytes": len(payload),
                "sample_sha256": hashlib.sha256(payload).hexdigest(),
                "hash_method": "4096 or fewer evenly spaced bytes of contiguous logical tensor",
            }
        checkpoint = {
            name: {
                "shard": self.reader.tensor_files[name].name,
                "shape": item["shape"],
                "dtype": item["dtype"],
            }
            for name, item in self.reader.tensor_metadata.items()
            if name.startswith("model.layers.0.")
            or name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
        }
        return {
            "workload": self.metadata,
            "seed": self.seed,
            "inputs": inputs,
            "checkpoint_tensors": checkpoint,
        }

    def close(self):
        torch.cuda.synchronize(self.device)
        self.cases.clear()
        self.captured.clear()
        self.block = self.attention = self.projected = None
        self.embedding = self.head = self.final_norm = None
        self.hidden = self.residual = self.token_ids = self.positions = self.attention_input = None


@torch.inference_mode()
def build_common(s, device="cuda:0", seed=20261011, checkpoint="/preset-models"):
    return CommonWorkload(s, device, seed, checkpoint)
