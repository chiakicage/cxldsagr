"""Independent FP32 sparse MLA oracle using only PyTorch.

Selected IDs address logical records. Duplicate slots participate independently;
negative and out-of-range IDs are padding. This module does not import device
kernels, Triton, or native extension loaders.
"""

import torch

from operators.deepseek_v32.attention._validation import _validate


@torch.no_grad()
def reference_sparse_mla(q, kv, indices, scale, value_dim=512):
    """FP32 oracle with the same selected-slot and padding semantics."""
    _validate(q, kv, indices, scale, value_dim)
    output = torch.zeros((*q.shape[:2], value_dim), dtype=q.dtype, device=q.device)
    # Keep arithmetic FP32 even if the surrounding caller uses autocast.
    with torch.autocast(device_type=q.device.type, enabled=False):
        for row in range(q.shape[0]):
            selected = indices[row]
            selected = selected[(selected >= 0) & (selected < kv.shape[0])].long()
            if not selected.numel():
                continue
            records = kv[selected].float()
            logits = q[row].float() @ records.T * scale
            output[row] = (logits.softmax(dim=-1) @ records[:, :value_dim]).to(q.dtype)
    return output
