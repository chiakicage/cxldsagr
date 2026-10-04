"""DeepSeek V3.2 sparse prefill attention using the official FlashMLA backend.

The model supplies absorbed BF16 queries and shared latent/RoPE KV records.
Selection is already causal and may contain logical resident IDs or physical
IDs after exact offload recall. This adapter handles layout and index-width
conversion; it does not change selection or implement another attention kernel.
"""

import hashlib
import importlib
from functools import cache
from pathlib import Path

import torch
from torch.nn import functional as F

from operators.deepseek_v32.attention._config import padded_selection_count
from operators.deepseek_v32.attention._validation import _validate


@cache
def _flash_mla():
    try:
        return importlib.import_module("flash_mla")
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek sparse MLA requires the FlashMLA submodule's compiled SM90 backend; "
            "prepare the shared third-party dependencies and run uv sync"
        ) from exc


def build_info():
    """Identify the actual loaded official Python interface and native library."""
    package = _flash_mla()
    native = importlib.import_module("flash_mla.cuda")
    interface = importlib.import_module("flash_mla.flash_mla_interface")
    files = {
        "package": Path(package.__file__).resolve(),
        "interface": Path(interface.__file__).resolve(),
        "native": Path(native.__file__).resolve(),
    }
    identities = {}
    for name, path in files.items():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        identities[name] = {"path": str(path), "sha256": digest}
    return {
        "backend": "flash_mla.flash_mla_sparse_fwd",
        "package_version": package.__version__,
        "files": identities,
    }


def sparse_mla(q, kv, indices, scale, value_dim=512):
    """Return BF16 [Q,H,512] sparse attention on SM90 using FlashMLA.

    Supported queries have 64 or 128 heads and dimension 576. KV records have
    dimension 576, with the first 512 elements used as values. Negative and
    out-of-range IDs are padding; duplicates participate independently and
    all-padding rows return zero. Strided inputs and int64 IDs are converted
    inside this call. These copies and selected-axis padding are operator work.
    The independent CPU oracle is attention.reference.torch.
    """
    _validate(q, kv, indices, scale, value_dim)
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise NotImplementedError("FlashMLA sparse prefill requires an SM90/Hopper CUDA device")
    if q.dtype != torch.bfloat16:
        raise ValueError("FlashMLA sparse prefill requires BF16 queries and KV records")
    if q.shape[1] not in (64, 128) or q.shape[2] != 576 or value_dim != 512:
        raise ValueError("DeepSeek FlashMLA requires H=64 or 128, QK dimension 576 and V=512")
    if kv.shape[0] > torch.iinfo(torch.int32).max:
        raise ValueError("FlashMLA sparse prefill requires the KV row count to fit int32")
    if not q.shape[0] or not kv.shape[0] or not indices.shape[1]:
        return torch.zeros((*q.shape[:2], value_dim), dtype=q.dtype, device=q.device)

    with torch.cuda.device(q.device):
        q, kv = q.contiguous(), kv.contiguous()
        # A contiguous view can retain a two-byte storage offset. Native TMA
        # and vector KV loads also need a 16-byte aligned base address.
        if q.data_ptr() % 16:
            q = q.clone()
        if kv.data_ptr() % 16:
            kv = kv.clone()
        if indices.dtype == torch.int64:
            # Narrow only after masking: large invalid int64 IDs can wrap into
            # valid int32 token IDs and must never fetch those records.
            valid = (indices >= 0) & (indices < kv.shape[0])
            indices = indices.masked_fill(~valid, -1).to(torch.int32)
        indices = indices.contiguous()
        padded = padded_selection_count(indices.shape[1])
        if padded != indices.shape[1]:
            indices = F.pad(indices, (0, padded - indices.shape[1]), value=-1)
        output, _, _ = _flash_mla().flash_mla_sparse_fwd(
            q,
            kv.unsqueeze(1),
            indices.unsqueeze(1),
            float(scale),
            value_dim,
            topk_length=None,
        )
    return output
