"""Auditable matrix FLOPs for the real DeepSeek V3.2 ECHO execution path.

One multiply-add counts as two FLOPs. ``useful_flops`` is the logical model
work; ``executed_matmul_flops`` additionally counts padding known from the
local kernel source. The latter excludes scalar work and is not model FLOPs.
cuBLAS internal tiles are deliberately left unknown. No CUDA or Torch import
is needed to analyze saved shapes.

MFU must use a measured duration from the corresponding invocation and a
precision-specific *dense* peak. Peaks are explicit inputs: the GPU model,
clock convention, FP32/TF32 policy, and peak source belong in run metadata.
An FP32 operation must not silently use the FP8 or BF16 Tensor Core peak.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from operators.deepseek_v32.attention._config import padded_selection_count


def _integer(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class OperatorWork:
    """FLOPs for one operator invocation, independent of profiling overhead."""

    name: str
    precision: str | None
    useful_flops: int | None
    executed_matmul_flops: int | None
    formula: str
    executed_formula: str | None = None
    dimensions: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def mfu(self, gpu_ms: float, peaks_tflops: Mapping[str, float]) -> dict:
        """Return useful MFU and separately labelled padded-matmul utilization.

        ``gpu_ms`` may be a GPU kernel duration or a sum of GPU kernel
        durations within this invocation. The caller must record which; an
        NVTX wall interval with submission gaps is a different denominator.
        Non-matrix work returns N/A, not zero percent Tensor Core MFU.
        """
        if not math.isfinite(gpu_ms) or gpu_ms < 0:
            raise ValueError("gpu_ms must be finite and nonnegative")
        result = {
            "gpu_ms": gpu_ms,
            "peak_tflops": None,
            "useful_tflops": None,
            "mfu_pct": None,
            "executed_matmul_utilization_pct": None,
            "mfu_na_reason": None,
        }
        if self.useful_flops is None:
            result["mfu_na_reason"] = "non-matmul operator"
            return result
        if gpu_ms == 0:
            raise ValueError("A matrix operator needs a positive measured duration")
        result["useful_tflops"] = self.useful_flops / gpu_ms / 1e9
        peak = peaks_tflops.get(self.precision)
        if peak is None:
            result["mfu_na_reason"] = f"no explicit {self.precision} peak supplied"
            return result
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError("The precision-specific dense peak must be finite and positive")
        result["peak_tflops"] = peak
        result["mfu_pct"] = self.useful_flops / gpu_ms / 1e9 / peak * 100
        if self.executed_matmul_flops is not None:
            result["executed_matmul_utilization_pct"] = (
                self.executed_matmul_flops / gpu_ms / 1e9 / peak * 100
            )
        return result


def non_matmul_work(name: str, *, notes: str = "") -> OperatorWork:
    """Represent norms, transforms, selection, cache control, and data copies."""
    return OperatorWork(name, None, None, None, "N/A (no matrix multiply)", notes=notes)


def linear_work(
    name: str,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    precision: str,
    local_fp8_kernel: bool = False,
) -> OperatorWork:
    """Count ``[M,K] @ [N,K].T`` using actual checkpoint weight precision.

    ``local_fp8_kernel`` identifies the model adapter, which now dispatches
    official DeepGEMM. Its runtime-selected padding is not inferred here.
    Activation quantization/scaling are scalar work, excluded from FLOPs.
    """
    m = _integer(rows, "rows")
    n = _integer(out_features, "out_features", positive=True)
    k = _integer(in_features, "in_features", positive=True)
    if not precision:
        raise ValueError("The observed matrix arithmetic precision is required")
    dimensions = {"M": m, "N": n, "K": k}
    if local_fp8_kernel and precision != "fp8":
        raise ValueError("The block-FP8 adapter requires precision='fp8'")
    return OperatorWork(
        name,
        precision,
        2 * m * n * k,
        None,
        "2 * M * N * K",
        None,
        dimensions,
        "Scalar activation quantization, block scaling and conversion are excluded from FLOPs. "
        + (
            "Official DeepGEMM selects TMA/GEMM geometry at runtime; executed padding is unknown."
            if local_fp8_kernel
            else "cuBLAS padding is unknown."
        ),
    )


def batched_matmul_work(
    name: str, *, batch: int, m: int, n: int, k: int, precision: str = "bf16"
) -> OperatorWork:
    """Count the per-head absorbed Q projection or latent-to-V expansion."""
    dims = {
        "B": _integer(batch, "batch", positive=True),
        "M": _integer(m, "m"),
        "N": _integer(n, "n", positive=True),
        "K": _integer(k, "k", positive=True),
    }
    return OperatorWork(
        name,
        precision,
        2 * dims["B"] * dims["M"] * dims["N"] * dims["K"],
        None,
        "2 * B * M * N * K",
        dimensions=dims,
        notes="torch.bmm/cuBLAS internal padding is unknown.",
    )


def causal_pairs(query_tokens: int, query_start: int) -> int:
    """Count visible logical (query, key) pairs for contiguous causal queries."""
    q = _integer(query_tokens, "query_tokens")
    p = _integer(query_start, "query_start")
    return q * p + q * (q + 1) // 2


def selected_pairs(query_tokens: int, query_start: int, selected_slots: int) -> int:
    """Count exact top-k pairs, excluding early-prefix causal padding.

    The identity assumes finite scores for each visible key and an exact
    causal top-k. Saved selection validity should be checked against it.
    Repeated selected IDs, if present, are separate attention operands.
    """
    q = _integer(query_tokens, "query_tokens")
    p = _integer(query_start, "query_start")
    slots = _integer(selected_slots, "selected_slots")
    unsaturated = min(q, max(0, slots - p))
    return unsaturated * p + unsaturated * (unsaturated + 1) // 2 + (q - unsaturated) * slots


def indexer_work(
    *, query_tokens: int, query_start: int, kv_tokens: int | None = None, prefetch: bool = False
) -> OperatorWork:
    """Count native 64-head, D128 FP8 indexer QK; prefetch adds no matrix FLOPs."""
    q = _integer(query_tokens, "query_tokens", positive=True)
    p = _integer(query_start, "query_start")
    visible = p + q if kv_tokens is None else _integer(kv_tokens, "kv_tokens", positive=True)
    if visible < p + q:
        raise ValueError("kv_tokens must cover every query's causal endpoint")
    useful_pairs = causal_pairs(q, p)
    # load_schedule uses the largest endpoint of the two rows; the final
    # partial group still executes two WGMMA rows (second row is zero padded).
    # Official DeepGEMM's resident SM90 kernel tiles 256 KV rows; the fused
    # ECHO prefetch implementation retains its 128-row tile.
    block_kv = 128 if prefetch else 256
    padded_pairs = sum(2 * _round_up(p + min(row + 2, q), block_kv) for row in range(0, q, 2))
    return OperatorWork(
        "indexer_prefetch" if prefetch else "indexer",
        "fp8",
        2 * 64 * 128 * useful_pairs,
        2 * 64 * 128 * padded_pairs,
        "2 * 64 * 128 * (Q * P + Q * (Q + 1) / 2)",
        "2 * 64 * 128 * sum_groups(2 * round_up(P + min(group_start + 2, Q),BLOCK_KV))",
        {
            "Q": q,
            "P": p,
            "KV": visible,
            "valid_pairs": useful_pairs,
            "padded_pairs": padded_pairs,
            "BLOCK_KV": block_kv,
        },
        "Only QK dot products count; ReLU, head weighting/reduction, cleanup, histogram, "
        "and fused host prefetch are excluded from FLOPs but included when their kernels "
        "are inside the measured duration. Causal invalid positions within a tile are padding.",
    )


def sparse_mla_work(
    *,
    query_tokens: int,
    heads: int,
    qk_dim: int,
    value_dim: int,
    selected_slots: int,
    valid_selected_pairs: int,
    precision: str = "bf16",
) -> OperatorWork:
    """Count the single fused sparse MLA kernel's QK and PV together.

    Neither QK nor PV has a separately measured duration inside this kernel;
    assigning the full duration to each and calling both operator MFU would
    double-count time. Their FLOP contributions remain explicit below.
    """
    q = _integer(query_tokens, "query_tokens")
    h = _integer(heads, "heads", positive=True)
    d = _integer(qk_dim, "qk_dim", positive=True)
    v = _integer(value_dim, "value_dim", positive=True)
    slots = _integer(selected_slots, "selected_slots")
    pairs = _integer(valid_selected_pairs, "valid_selected_pairs")
    if h not in (64, 128) or d != 576 or v != 512 or precision != "bf16":
        raise ValueError("FlashMLA SM90 sparse prefill requires BF16 H64/H128, D576 and V512")
    if pairs > q * slots:
        raise ValueError("valid_selected_pairs exceeds the selection tensor capacity")
    padded_slots = padded_selection_count(slots)
    useful_qk, useful_pv = 2 * pairs * h * d, 2 * pairs * h * v
    executed = 2 * q * h * padded_slots * (d + v)
    return OperatorWork(
        "sparse_mla",
        precision,
        useful_qk + useful_pv,
        executed,
        "QK: 2 * valid_pairs * H * D; PV: 2 * valid_pairs * H * V",
        "2 * Q * H * round_up(selected_slots,128) * (D + V)",
        {
            "Q": q,
            "H": h,
            "D": d,
            "V": v,
            "selected_slots": slots,
            "valid_pairs": pairs,
            "qk_flops": useful_qk,
            "pv_flops": useful_pv,
            "slots_padded": padded_slots,
            "BLOCK_H": 64,
            "BLOCK_K": 64,
            "selection_alignment": 128,
            "threads": 384,
        },
        "Useful FLOPs exclude invalid selection slots; official FlashMLA executes padded slots. "
        "Softmax, address lookup, and KV loads add time but no counted matrix FLOPs. "
        "Offload query splitting preserves per-query work; sum only successful MLA invocations. "
        "FlashMLA SM90 sparse prefill uses two alternating 64-token KV tiles with "
        "topk_length=None; the adapter pads capacity to 128 without removing duplicate IDs.",
    )


def _dimension(config, name: str, alias: str | None = None) -> int:
    for key in (name, alias):
        if key is not None:
            value = config.get(key) if isinstance(config, Mapping) else getattr(config, key, None)
            if value is not None:
                return _integer(value, name, positive=True)
    raise ValueError(f"Missing model dimension {name}")


def build_operator_work(
    config,
    *,
    query_tokens: int,
    query_start: int,
    linear_dtypes: Mapping[str, str],
    kv_tokens: int | None = None,
    selected_slots: int | None = None,
    valid_selected_pairs: int | None = None,
    prefetch: bool = False,
    lm_head_tokens: int = 0,
) -> dict[str, OperatorWork]:
    """Describe one real layer/chunk, using caller-observed linear precision.

    Names are suitable for NVTX leaf ranges. ``linear_dtypes`` must name each
    listed linear and use ``fp8`` only when CheckpointLinear.scales is present;
    local FP8 tile accounting then follows linear/fp8.py. The head-weight
    projection is FP32 unless the caller explicitly records a TF32 policy.
    ``lm_head_tokens`` defaults to zero because the head executes once outside
    the transformer layers, often for only the last token.

    This intentionally covers dense layers only; do not apply it to MoE
    layers, reconstruct 61-layer totals from three layers, or infer precision
    solely from the command-line backend name.
    """
    q = _integer(query_tokens, "query_tokens", positive=True)
    p = _integer(query_start, "query_start")
    dim = _dimension(config, "dim", "hidden_size")
    heads = _dimension(config, "n_heads", "num_attention_heads")
    qr = _dimension(config, "q_lora_rank")
    latent = _dimension(config, "kv_lora_rank")
    nope = _dimension(config, "qk_nope_head_dim")
    rope = _dimension(config, "qk_rope_head_dim")
    value = _dimension(config, "v_head_dim")
    ih = _dimension(config, "index_n_heads")
    idim = _dimension(config, "index_head_dim")
    intermediate = _dimension(config, "intermediate_size")
    if (ih, idim) != (64, 128):
        raise ValueError("The native ECHO indexer requires 64 heads and dimension 128")
    visible = p + q if kv_tokens is None else kv_tokens
    slots = (
        min(_dimension(config, "index_topk"), visible) if selected_slots is None else selected_slots
    )
    pairs = selected_pairs(q, p, slots) if valid_selected_pairs is None else valid_selected_pairs
    linear_shapes = {
        "q_a_proj": (q, dim, qr),
        "q_b_proj": (q, qr, heads * (nope + rope)),
        "kv_a_proj": (q, dim, latent + rope),
        "index_q_proj": (q, qr, ih * idim),
        "index_k_proj": (q, dim, idim),
        "o_proj": (q, heads * value, dim),
        "mlp_gate": (q, dim, intermediate),
        "mlp_up": (q, dim, intermediate),
        "mlp_down": (q, intermediate, dim),
    }
    if _integer(lm_head_tokens, "lm_head_tokens"):
        linear_shapes["lm_head"] = (lm_head_tokens, dim, _dimension(config, "vocab_size"))
    missing = linear_shapes.keys() - linear_dtypes.keys()
    if missing:
        raise ValueError(f"Observed linear precision is required for: {', '.join(sorted(missing))}")
    result = {
        name: linear_work(
            name,
            rows=rows,
            in_features=k,
            out_features=n,
            precision=linear_dtypes[name],
            local_fp8_kernel=linear_dtypes[name] == "fp8",
        )
        for name, (rows, k, n) in linear_shapes.items()
    }
    result["index_weights_proj"] = linear_work(
        "index_weights_proj",
        rows=q,
        in_features=dim,
        out_features=ih,
        precision=linear_dtypes.get("index_weights_proj", "fp32"),
    )
    result["q_absorb"] = batched_matmul_work("q_absorb", batch=heads, m=q, n=latent, k=nope)
    result["v_expand"] = batched_matmul_work("v_expand", batch=heads, m=q, n=value, k=latent)
    indexer = indexer_work(query_tokens=q, query_start=p, kv_tokens=visible, prefetch=prefetch)
    result[indexer.name] = indexer
    result["sparse_mla"] = sparse_mla_work(
        query_tokens=q,
        heads=heads,
        qk_dim=latent + rope,
        value_dim=latent,
        selected_slots=slots,
        valid_selected_pairs=pairs,
    )
    return result
