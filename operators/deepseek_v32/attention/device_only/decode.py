"""Explicit sparse Q1 partitioning and FP32 natural-log LSE merge on SM90."""

from functools import cache

import torch

from operators.deepseek_v32.attention._config import DECODE_SPLITS

_observed = {}


def runtime_info():
    """Export retained launch artifacts outside measurement; never compile or launch."""
    if not _observed:
        return None
    import dataclasses
    import hashlib
    import importlib.metadata

    def plain(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if dataclasses.is_dataclass(value):
            return plain(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if hasattr(value, "_asdict"):
            return plain(value._asdict())
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        raise TypeError(f"Unsupported decode Triton metadata: {type(value).__name__}")

    specializations = []
    for key, compiled in sorted(_observed.items()):
        values = vars(compiled)
        asm = values["asm"]
        if not {"ptx", "cubin"} <= asm.keys():
            raise RuntimeError("Observed decode combine is missing PTX or CUBIN")
        artifacts = {}
        for kind in sorted({"ptx", "cubin", "ttir", "ttgir", "llir"} & asm.keys()):
            value = asm[kind]
            if not isinstance(value, (str, bytes)):
                raise TypeError(f"Unsupported decode {kind} artifact type")
            artifacts[kind] = hashlib.sha256(
                value.encode() if isinstance(value, str) else value
            ).hexdigest()
        specializations.append(
            {
                "hash": key,
                "name": values["name"],
                "metadata": plain(values["metadata"]),
                "artifact_sha256": artifacts,
                "module_handle_set": values.get("module") is not None,
                "function_handle_set": values.get("function") is not None,
            }
        )
    return {
        "policy": "sparse-q1-split16-natural-lse-v1",
        "triton_version": importlib.metadata.version("triton"),
        "specializations": specializations,
    }


@cache
def _combine_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def combine(Partials, Lse, Out, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
        head = tl.program_id(0)
        split = tl.arange(0, S)
        dim = tl.arange(0, D)
        lse = tl.load(Lse + split * H + head)
        # Official sparse prefill returns +inf for a shard with no valid IDs.
        present = lse != float("inf")
        lse = tl.where(present, lse, -float("inf"))
        peak = tl.max(lse, 0)
        peak = tl.where(peak == -float("inf"), 0.0, peak)
        weight = tl.exp(lse - peak)
        denominator = tl.sum(weight, 0)
        value = tl.load(Partials + (split[:, None] * H + head) * D + dim[None, :])
        value = tl.where(present[:, None], value.to(tl.float32), 0.0)
        result = tl.sum(value * weight[:, None], 0) / tl.maximum(denominator, 1.0)
        tl.store(Out + head * D + dim, result)

    return combine


def split_kv(q, kv, indices, scale, backend):
    """Consume already prepared Q1/H64-or-H128/2048-aligned selected slots."""
    heads = q.shape[1]
    repeated_q = q.repeat(DECODE_SPLITS, 1, 1)
    partials, _, lse = backend.flash_mla_sparse_fwd(
        repeated_q,
        kv.unsqueeze(1),
        indices.view(DECODE_SPLITS, 1, -1),
        scale,
        512,
        topk_length=None,
    )
    output = torch.empty((1, heads, 512), dtype=q.dtype, device=q.device)
    compiled = _combine_kernel()[(heads,)](
        partials, lse, output, heads, DECODE_SPLITS, 512, num_warps=4
    )
    # Retain the actual launch result only. Hashing/metadata export is explicit
    # and happens later through runtime_info(), outside measured execution.
    _observed[compiled.hash] = compiled
    return output
