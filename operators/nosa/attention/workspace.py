"""Bounded FA3 scratch storage; importing this module needs only PyTorch.

The CPU allocation path exists for storage and lifecycle tests. Attention still
requires the native SM90/BF16/D128/GQA16 specialization. This object owns no
model state; its caller must serialize users and await work before releasing it.
"""

import torch


def _integer(name, value, *, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value < 2**31:
        raise ValueError(f"{name} must be {'a nonnegative' if allow_zero else 'a positive'} int32")


def _layout(max_queries, kv_heads, head_dim, dtype):
    _integer("max_queries", max_queries)
    _integer("kv_heads", kv_heads)
    _integer("head_dim", head_dim)
    if kv_heads > 65535:
        raise ValueError("kv_heads exceeds the CUDA grid limit")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("NOSA workspace requires a floating point dtype")
    batches = ((max_queries + 7) // 8) * kv_heads
    _integer("query batch capacity", batches)
    return {
        "fallback": ((max_queries + 3) // 4, kv_heads),
        "pages": (batches, 8 * 64),
        "members": (batches, 8 * 64),
        # Even when the largest batch is not 256, a smaller call may need the
        # second counts region used by native work ordering at 256 batches.
        "counts": (2 * batches,),
    }


class NosaAttentionWorkspace:
    """One explicitly bounded set of fallback/pages/members/counts tensors."""

    def __init__(self, max_queries, kv_heads, head_dim=128, *, device, dtype):
        layout = _layout(max_queries, kv_heads, head_dim, dtype)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise NotImplementedError("NOSA workspace storage supports CPU or CUDA")
        if self.device.type == "cuda":
            from operators.nosa._native import native_enabled

            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.get_device_capability(self.device) != (9, 0):
                raise NotImplementedError("Shared NOSA attention requires SM90/Hopper")
            if not native_enabled():
                raise NotImplementedError("Shared NOSA attention requires the native FA3 backend")
            if dtype != torch.bfloat16 or head_dim != 128:
                raise ValueError("Shared NOSA attention requires BF16 and head_dim=128")
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise NotImplementedError("Shared NOSA attention does not support CUDA Graph")
        self.max_queries = max_queries
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self._storage = {
            name: torch.empty(shape, dtype=torch.int32, device=self.device)
            for name, shape in layout.items()
        }
        self.empty_mask = torch.empty(0, dtype=torch.bool, device=self.device)

    @staticmethod
    def allocation_sizes(max_queries, kv_heads, head_dim=128, *, dtype):
        """Pure payload sizes in ``tensors()`` order, including the empty mask.

        Each nonzero entry is one independent tensor allocation. Allocator
        rounding belongs to the caller's device-specific reservation policy.
        """
        from math import prod

        return (
            *(
                prod(shape) * 4
                for shape in _layout(max_queries, kv_heads, head_dim, dtype).values()
            ),
            0,
        )

    @staticmethod
    def estimate_capacity_bytes(max_queries, kv_heads, head_dim=128, *, dtype):
        """Pure storage estimate, including counts capacity for every smaller q."""
        return sum(
            NosaAttentionWorkspace.allocation_sizes(max_queries, kv_heads, head_dim, dtype=dtype)
        )

    def slices(self, queries, kv_heads=None, *, device=None, dtype=None):
        """Return contiguous views shaped for the actual query/head geometry."""
        _integer("queries", queries, allow_zero=True)
        heads = self.kv_heads if kv_heads is None else kv_heads
        _integer("kv_heads", heads)
        if queries > self.max_queries or heads > self.kv_heads:
            raise ValueError("NOSA attention request exceeds reserved query/head capacity")
        if device is not None:
            actual_device = torch.device(device)
            if actual_device.type == "cuda" and actual_device.index is None:
                actual_device = torch.device("cuda", torch.cuda.current_device())
            if actual_device != self.device:
                raise ValueError("NOSA attention workspace device mismatch")
        if dtype is not None and dtype != self.dtype:
            raise ValueError("NOSA attention workspace dtype mismatch")
        batches = ((queries + 7) // 8) * heads
        fallback_elements = ((queries + 3) // 4) * heads
        counts = batches * (2 if batches == 256 else 1)
        return {
            "fallback": self._storage["fallback"].view(-1)[:fallback_elements].view(-1, heads),
            "pages": self._storage["pages"][:batches],
            "members": self._storage["members"][:batches],
            "counts": self._storage["counts"][:counts],
        }

    def validate_attention(self, q, keys, values):
        """Validate the reserved FA3 route and return its actual-shape views.

        The views are reused by the immediate launch. Creating and discarding
        the same views at each dispatch layer adds host work without checking a
        different invariant.
        """
        if not q.is_cuda or self.device.type != "cuda":
            raise NotImplementedError(
                "CPU NOSA workspace is allocation-only; use reference attention"
            )
        from operators.nosa._native import native_enabled
        from operators.nosa.attention.device_only._cuda import supports_native_attention

        if not native_enabled():
            raise NotImplementedError("Shared NOSA attention requires the native FA3 backend")
        if torch.cuda.get_device_capability(q.device) != (9, 0):
            raise NotImplementedError("Shared NOSA attention requires SM90/Hopper")
        if (
            q.ndim != 3
            or keys.ndim != 3
            or values.shape != keys.shape
            or any(t.device != self.device or t.dtype != self.dtype for t in (q, keys, values))
            or any(t.stride(-1) != 1 for t in (q, keys, values))
            or q.dtype != torch.bfloat16
            or q.shape[-1] != 128
            or q.shape[-1] != self.head_dim
            or not supports_native_attention(q, keys, values)
            or keys.stride() != values.stride()
        ):
            raise ValueError(
                "Shared NOSA attention requires native BF16/D128/GQA16 matching K/V layouts"
            )
        scratch = self.slices(len(q), keys.shape[1], device=q.device, dtype=q.dtype)
        storages = {
            tensor.untyped_storage().data_ptr() for tensor in self.tensors() if tensor.numel()
        }
        if any(t.untyped_storage().data_ptr() in storages for t in (q, keys, values)):
            raise ValueError("NOSA attention inputs must not alias writable scratch storage")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise NotImplementedError("Shared NOSA attention does not support CUDA Graph")
        return scratch

    def tensors(self):
        """Enumerate full storage tensors, excluding transient slices."""
        return (*self._storage.values(), self.empty_mask)

    @property
    def capacity_bytes(self):
        return sum(t.untyped_storage().nbytes() for t in self.tensors())
