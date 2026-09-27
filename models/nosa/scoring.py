"""Checkpoint-trained NOSA query-agnostic (CIS) token scores.

This matches ``cxl_recsys.models.nosa_ops.cis_scores``: project flattened V
with delta in the model dtype, apply softplus and A in FP32, and round the
result back to V's dtype. CIS itself is the additive attention log bias;
applying log-sigmoid here would change the checkpoint's attention semantics.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NosaAttentionState:
    """Causal resident CIS view, shaped ``[token, KV head]``.

    Selection mean-compresses these scores, while main attention uses the
    uncompressed scores as an additive bias. This view owns no cache state.
    """

    cis_scores: torch.Tensor

    @property
    def cis_bias(self) -> torch.Tensor:
        return self.cis_scores


def cis_scores(
    values: torch.Tensor,
    delta: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``A * softplus(delta(flatten(V)))`` for each token/KV head.

    ``values`` is ``[token, KV head, head_dim]``, ``delta`` is
    ``[KV head, KV head * head_dim]``, and checkpoint ``A`` (``scale``) is
    ``[KV head]``. No sign restriction is imposed on learned A.
    """
    if values.ndim != 3 or values.shape[1] <= 0 or values.shape[2] <= 0:
        raise ValueError("V must have [token, KV head, head_dim] shape")
    heads, dimension = values.shape[1:]
    if delta.shape != (heads, heads * dimension) or scale.shape != (heads,):
        raise ValueError("NOSA delta/A shapes must match V's KV heads and head dimension")
    if any(not tensor.is_floating_point() for tensor in (values, delta, scale)):
        raise ValueError("NOSA CIS requires floating-point V, delta and A")
    if any(tensor.device != values.device for tensor in (delta, scale)):
        raise ValueError("NOSA CIS V, delta and A must share a device")
    if delta.dtype != values.dtype:
        raise ValueError("NOSA delta and V must share a dtype")
    if bias is not None and (
        bias.shape != (heads,) or bias.device != values.device or bias.dtype != values.dtype
    ):
        raise ValueError("NOSA delta bias must have [KV head] shape and V's dtype/device")
    # Explicitly disable caller autocast: the projection's checkpoint dtype
    # rounding is part of NOSA's scoring definition, even for FP32 inputs.
    with torch.autocast(device_type=values.device.type, enabled=False):
        projected = F.linear(values.flatten(1), delta, bias)
        return (F.softplus(projected.float()) * scale.float()).to(values.dtype)


def compress_sequence(values: torch.Tensor) -> torch.Tensor:
    """32-token means at stride 16, retaining only complete windows.

    Unlike the existing query-aware-only FP32 analysis path, full NOSA retains
    model-dtype compression rounding, as in the source implementation.
    """
    if len(values) < 32:
        return values[:0]
    return values.unfold(0, 32, 16).mean(-1)
