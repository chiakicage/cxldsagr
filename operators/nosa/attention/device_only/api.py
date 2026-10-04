"""NOSA sparse attention over device-resident K/V and logical selections."""

import torch

from layers.attention import BlockSelection
from operators.nosa.attention.common import _validate_inputs


def nosa_block_sparse_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    selection: BlockSelection,
    query_start: int = 0,
    cis_bias: torch.Tensor | None = None,
    *,
    workspace=None,
) -> torch.Tensor:
    """Fused inference-only SM90 implementation of the reference operation.

    Q/K/V use BF16 or FP16 with D=64/128 and at most 32 Q heads per KV head.
    Tensor Cores compute QK and AV; online softmax and accumulators are FP32.
    Probabilities are rounded to Q's dtype for AV, so results are not bitwise
    identical to the FP32 reference. K/V remain resident and are read in place.

    The native CuTe implementation specializes D=128 and GQA=16 with TMA
    aligned pointers and row/head strides. Four adjacent queries share the
    union of their selected blocks while retaining independent causal masks,
    selection membership and softmax state. Packages with little overlap or
    nonfinite values requiring separate masking use the per-query kernel.
    Other supported shapes/layouts use Triton. CXLDSAGR_SM90_BACKEND=triton
    selects that implementation explicitly.

    An explicit workspace uses reserved scratch and requires the native
    BF16/D128/GQA16 route with matching K/V layouts. The caller owns the
    execution lease and must await work before reusing or releasing storage.
    """
    _validate_inputs(q, keys, values, selection, query_start, cis_bias)
    if workspace is not None:
        from operators.nosa.attention.workspace import NosaAttentionWorkspace

        if not isinstance(workspace, NosaAttentionWorkspace):
            raise TypeError("workspace must be a NosaAttentionWorkspace")
        scratch = workspace.validate_attention(q, keys, values)
        from operators.nosa.attention.device_only._fa3 import _launch_validated_workspace

        # The workspace guard checks Hopper, native selection, dtype, layout,
        # capacity, capture state and scratch aliasing before this import.
        return _launch_validated_workspace(
            q, keys, values, selection, query_start, cis_bias, workspace, scratch
        )
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise NotImplementedError("The fused NOSA block attention backend requires SM90/Hopper")
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or q.shape[-1] not in (64, 128)
        or q.shape[1] // keys.shape[1] > 32
    ):
        raise ValueError("Fused NOSA requires BF16/FP16, D=64/128 and GQA groups <=32")
    if any(t.stride(-1) != 1 for t in (q, keys, values)):
        raise ValueError("Fused NOSA requires contiguous Q/K/V innermost dimensions")
    from operators.nosa._native import native_enabled

    if native_enabled():
        from operators.nosa.attention.device_only import _cuda as native_attention

        if native_attention.supports_native_attention(q, keys, values):
            return native_attention.launch_nosa_block_attention(
                q, keys, values, selection, query_start, cis_bias
            )
    # CPU-only users of the mathematical reference do not import Triton.
    from operators.nosa.attention.device_only._triton import launch_nosa_block_attention

    return launch_nosa_block_attention(q, keys, values, selection, query_start, cis_bias)
