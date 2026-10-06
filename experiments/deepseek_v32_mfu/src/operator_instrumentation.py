"""Diagnostic-only wrappers around the real ECHO calls, with NVTX and useful FLOPs.

No CUDA tensor is copied or reduced to compute the annotations. In particular,
causal sparse-pair counts follow the exact top-k contract without synchronizing
the GPU inside an operator to inspect selection values.
"""

import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.deepseek_v32_mfu.src.operator_flops import (
    batched_matmul_work,
    indexer_work,
    linear_work,
    non_matmul_work,
    selected_pairs,
    sparse_mla_work,
)


def _work_details(work):
    details = work.as_dict()
    details["operator_work_name"] = details.pop("name")
    if details["precision"] is not None:
        details["precision"] = details["precision"].upper()
    details["flop_reason"] = work.formula
    return details


def _precision(dtype):
    try:
        return {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
            torch.float8_e4m3fn: "fp8",
        }[dtype]
    except KeyError:
        raise ValueError(f"Unaccounted matrix precision: {dtype}") from None


def _storage_key(q):
    return q.device, q.untyped_storage().data_ptr()


@dataclass(frozen=True)
class QueryOrigin:
    """Resolve recursive unit-step query slices using host-visible metadata.

    The real attention path creates contiguous Q and only slices its first
    dimension during offload retries. Retaining shape/pointer integers avoids
    a GPU reduction or copy, and avoids keeping old Q allocations alive.
    """

    storage_key: tuple
    data_ptr: int
    stride: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    element_size: int
    start_pos: int

    @classmethod
    def capture(cls, q, start_pos):
        if q.ndim != 3 or len(q) == 0 or q.stride(0) <= 0:
            raise ValueError("Projected Q must be a nonempty rank-three tensor")
        if isinstance(start_pos, bool) or not isinstance(start_pos, int) or start_pos < 0:
            raise ValueError("Projected query start must be a nonnegative integer")
        return cls(
            _storage_key(q),
            q.data_ptr(),
            tuple(q.stride()),
            tuple(q.shape),
            q.dtype,
            q.element_size(),
            start_pos,
        )

    def resolve_start(self, q):
        if (
            q.ndim != 3
            or _storage_key(q) != self.storage_key
            or q.dtype != self.dtype
            or tuple(q.shape[1:]) != self.shape[1:]
            or tuple(q.stride()) != self.stride
            or not len(q)
        ):
            raise ValueError("MLA Q must be a nonempty unit-step row slice of its projected Q")
        row_bytes = self.stride[0] * self.element_size
        byte_offset = q.data_ptr() - self.data_ptr
        if byte_offset < 0 or byte_offset % row_bytes:
            raise ValueError("MLA Q pointer does not align to a projected query row")
        row_offset = byte_offset // row_bytes
        if row_offset + len(q) > self.shape[0]:
            raise ValueError("MLA Q slice extends outside its projected query range")
        return self.start_pos + row_offset


def mla_call_details(q, indices, origin, value_dim):
    """Annotate successful MLA leaves; failed capacity probes do no MLA work."""
    start = origin.resolve_start(q)
    if indices.ndim != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError("MLA selection must have one row per query")
    pairs = selected_pairs(len(q), start, indices.shape[1])
    work = sparse_mla_work(
        query_tokens=len(q),
        heads=q.shape[1],
        qk_dim=q.shape[2],
        value_dim=value_dim,
        selected_slots=indices.shape[1],
        valid_selected_pairs=pairs,
        precision=_precision(q.dtype),
    )
    return {
        **_work_details(work),
        "query_start": start,
        "query_tokens": len(q),
        "selected_slots": indices.shape[1],
        "valid_selected_pairs": pairs,
        "selection_count_source": "exact causal top-k contract; query slice offset from tensor metadata",
    }


class OperatorScopes:
    def __init__(self, mode, phase):
        self.mode, self.phase = mode, phase
        self.layer = "shared"
        self.calls = []

    @contextmanager
    def __call__(self, stage, *, useful_flops=None, precision=None, **details):
        if re.fullmatch(r"layer_\d+", stage):
            previous, self.layer = self.layer, stage
            try:
                # Attribute inter-stage allocation/concatenation to this layer.
                with (
                    torch.cuda.nvtx.range(f"echo/{self.mode}/{self.phase}/{stage}"),
                    self("layer_misc"),
                ):
                    yield
            finally:
                self.layer = previous
            return
        stage = {
            "indexer": "indexer_aux",
            "indexer_prefetch": "indexer_prefetch_aux",
            "sparse_mla": "sparse_mla_aux",
        }.get(stage, stage)
        if useful_flops is None:
            defaults = _work_details(
                non_matmul_work(
                    stage,
                    notes="Only exclusive GPU work belongs here; nested matrix calls have their own scopes.",
                )
            )
            defaults.pop("useful_flops")
            defaults.pop("precision")
            details = {**defaults, **details}
        call_id = len(self.calls)
        label = f"echo/{self.mode}/{self.phase}/{self.layer}/{stage}"
        if self.mode not in ("resident", "offload"):
            label += f"/call_{call_id}"
        self.calls.append(
            {
                "call_id": call_id,
                "nvtx": label,
                "mode": self.mode,
                "phase": self.phase,
                "layer": self.layer,
                "stage": stage,
                "useful_flops": useful_flops,
                "precision": precision,
                **details,
            }
        )
        with torch.cuda.nvtx.range(label):
            yield


class InstrumentOperators:
    """Install wrappers for one annotated forward and restore them on exit."""

    def __init__(self, model, scopes):
        self.model, self.scopes = model, scopes
        self.linears, self.bmms, self.float_linears = {}, {}, {}
        self.queries = {}
        for block in model.blocks:
            attn = block.attention.attention
            for attribute, label in (
                ("wq_a", "q_a_proj"),
                ("wq_b", "q_b_proj"),
                ("wkv_a", "kv_a_proj"),
                ("index_wq", "index_q_proj"),
                ("index_wk", "index_k_proj"),
                ("wo", "o_proj"),
            ):
                self.linears[id(getattr(attn, attribute))] = label
            for attribute in ("gate", "up", "down"):
                self.linears[id(getattr(block.mlp, attribute))] = "mlp_" + attribute
            self.bmms[attn.wk_b.data_ptr()] = "q_absorb"
            self.bmms[attn.wv_b.data_ptr()] = "v_expand"
            self.float_linears[attn.index_head_weight.data_ptr()] = "index_weights_proj"
        self.float_linears[model.head_weight.data_ptr()] = "lm_head"

    def __enter__(self):
        from models.deepseek_v32 import attention as echo_attention
        from models.deepseek_v32 import checkpoint, projections, rotary
        from models.deepseek_v32 import layers as echo_block
        from models.deepseek_v32 import model as echo_infer
        from operators.deepseek_v32.attention.offload import mla as offload_mla
        from operators.deepseek_v32.indexer import echo

        self.stack = ExitStack()
        original_linear = checkpoint.CheckpointLinear.__call__
        original_bmm, original_functional = torch.bmm, F.linear
        original_project = projections.CheckpointAttention.project
        original_logits, original_mla = echo.logits, echo_attention.sparse_mla

        def linear(instance, x, *args, **kwargs):
            label = self.linears.get(id(instance))
            if label is None:
                return original_linear(instance, x, *args, **kwargs)
            n, k = instance.weight.shape
            m = x.numel() // k
            work = linear_work(
                label,
                rows=m,
                in_features=k,
                out_features=n,
                precision="fp8" if instance.scales is not None else _precision(x.dtype),
                local_fp8_kernel=instance.scales is not None,
            )
            with self.scopes(label, shape=[m, n, k], **_work_details(work)):
                return original_linear(instance, x, *args, **kwargs)

        def bmm(a, b, *args, **kwargs):
            label = self.bmms.get(b.data_ptr())
            if label is None:
                return original_bmm(a, b, *args, **kwargs)
            batch, m, k = a.shape
            n = b.shape[-1]
            work = batched_matmul_work(
                label, batch=batch, m=m, n=n, k=k, precision=_precision(a.dtype)
            )
            with self.scopes(label, shape=[batch, m, n, k], **_work_details(work)):
                return original_bmm(a, b, *args, **kwargs)

        def functional(x, weight, *args, **kwargs):
            label = self.float_linears.get(weight.data_ptr())
            if label is None:
                return original_functional(x, weight, *args, **kwargs)
            n, k = weight.shape
            m = x.numel() // k
            if x.dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
                raise ValueError(
                    "FP32 MFU requires TF32 disabled, matching the experiment contract"
                )
            work = linear_work(
                label, rows=m, in_features=k, out_features=n, precision=_precision(x.dtype)
            )
            with self.scopes(label, shape=[m, n, k], **_work_details(work)):
                return original_functional(x, weight, *args, **kwargs)

        def project(instance, hidden, start_pos, **kwargs):
            result = original_project(instance, hidden, start_pos, **kwargs)
            self.queries[_storage_key(result.q)] = QueryOrigin.capture(result.q, start_pos)
            return result

        def logits(
            q, k, weights, kscale, query_start, prefetch=None, *, _bounds=None, _pad_to_stride=False
        ):
            if tuple(q.shape[1:]) != (64, 128):
                raise ValueError("The native ECHO indexer requires 64 heads and dimension 128")
            work = indexer_work(
                query_tokens=len(q),
                query_start=query_start,
                kv_tokens=len(k),
                prefetch=prefetch is not None,
            )
            with self.scopes(
                "indexer_fused" if prefetch is not None else "indexer_qk",
                query_start=query_start,
                query_tokens=len(q),
                kv_tokens=len(k),
                **_work_details(work),
            ):
                return original_logits(
                    q,
                    k,
                    weights,
                    kscale,
                    query_start,
                    prefetch=prefetch,
                    _bounds=_bounds,
                    _pad_to_stride=_pad_to_stride,
                )

        def mla(q, kv, indices, scale, value_dim=512):
            try:
                origin = self.queries[_storage_key(q)]
            except KeyError:
                raise ValueError("MLA Q has no recorded projection origin") from None
            with self.scopes("mla_qk_pv", **mla_call_details(q, indices, origin, value_dim)):
                return original_mla(q, kv, indices, scale, value_dim)

        def wrap(function, label):
            def wrapped(*args, **kwargs):
                with self.scopes(label, **_work_details(non_matmul_work(label))):
                    return function(*args, **kwargs)

            return wrapped

        targets = [
            (checkpoint.CheckpointLinear, "__call__", linear),
            (projections.CheckpointAttention, "project", project),
            (torch, "bmm", bmm),
            (F, "linear", functional),
            (echo, "logits", logits),
            (echo_attention, "sparse_mla", mla),
            (offload_mla, "sparse_mla", mla),
        ]
        for module, names in (
            (projections, ("apply_rope_pair", "prepare_rotary_cache", "quantize_index")),
            (
                rotary,
                ("apply_rope", "apply_rope_pair", "prepare_rotary_cache", "normalized_hadamard"),
            ),
        ):
            for name in names:
                if hasattr(module, name):
                    targets.append((module, name, wrap(getattr(module, name), name)))
        for module in (projections, echo_block, echo_infer):
            if hasattr(module, "rms_norm"):
                targets.append((module, "rms_norm", wrap(module.rms_norm, "rms_norm")))
        if hasattr(echo_infer, "residual_rms_norm"):
            targets.append(
                (
                    echo_infer,
                    "residual_rms_norm",
                    wrap(echo_infer.residual_rms_norm, "residual_rms_norm"),
                )
            )
        # Wrap the model adapters, so packing, FP32 copies and output casts
        # remain inside the nonmatrix operator's measured range.
        for name in ("residual_rms_norm", "silu_mul", "silu_mul_packed"):
            if hasattr(echo_block, name):
                label = "silu_mul" if name == "silu_mul_packed" else name
                targets.append((echo_block, name, wrap(getattr(echo_block, name), label)))
        targets.append((F, "layer_norm", wrap(F.layer_norm, "index_layer_norm")))
        for target, attribute, replacement in targets:
            self.stack.enter_context(patch.object(target, attribute, replacement))
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)
