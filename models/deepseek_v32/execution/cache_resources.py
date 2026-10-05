"""Allocation-free resource bounds for the ECHO shared-cache execution path.

The cache owns indexer score/top-k workspace for admission purposes. These
bounds deliberately include concurrently live intermediates rather than just
retained tensors observed after a layer returns. Weights and ordinary model
activations are outside this ledger and must fit physical device memory too.
"""

from dataclasses import dataclass

import torch

CACHE_POLICY_REVISION = "echo-global-pages-fifo-v1"
PAGE_SIZE = 64


def dense_staging_allocation_bytes(logical_bytes, device):
    """Reserve one dense stage storage's CUDA block, including an unsplit tail.

    This bound covers the native allocator's default 512-B rounding and at most
    1 MiB of an unsplit large block. Allocation validates the actual block once;
    an allocator configuration that exceeds this bound is rejected there.
    Existing ECHO allocations retain their separately validated accounting.
    """
    if type(logical_bytes) is not int or logical_bytes < 1:
        raise ValueError("dense staging bytes must be a positive integer")
    device = torch.device(device)
    if device.type == "cpu":
        return logical_bytes
    if device.type != "cuda":
        raise ValueError("dense staging supports CPU reference or CUDA")
    rounded = (logical_bytes + 511) // 512 * 512
    return rounded + ((1 << 20) if rounded > 1 << 20 else 0)


def check_dense_staging_allocation(tensor, reserved_bytes):
    """Find the owning active CUDA block and reject unreserved capacity.

    Run once during shared allocation, outside request execution. A missing or
    ambiguous snapshot record is not a capacity proof and must reject too.
    """
    storage = tensor.untyped_storage()
    logical = storage.nbytes()
    if type(reserved_bytes) is not int or reserved_bytes < logical:
        raise ValueError("dense staging reservation must cover its owning storage")
    if tensor.device.type == "cpu":
        return logical
    address = storage.data_ptr()
    # Avoid exporting allocator trace histories for an allocation-capacity check.
    snapshot = torch._C._cuda_memorySnapshot((0, 0, False))
    blocks = [
        block
        for segment in snapshot["segments"]
        if segment["device"] == tensor.device.index
        for block in segment["blocks"]
        if block["address"] == address and block["state"] == "active_allocated"
    ]
    if len(blocks) != 1:
        raise RuntimeError("cannot identify one active allocator block for dense staging")
    actual = blocks[0]["size"]
    if type(actual) is not int or not logical <= actual <= reserved_bytes:
        raise RuntimeError("dense staging allocator block exceeds its reservation")
    return actual


def padded_tokens(capacity):
    if type(capacity) is not int or capacity < 1:
        raise ValueError("token capacity must be a positive integer")
    return (capacity + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE


@dataclass(frozen=True)
class ExecutionReservation:
    query_tokens: int
    context_tokens: int
    indexer_bytes: int
    copy_source_bytes: int
    cpu_indexer_bytes: int = 8
    cpu_scalar_bytes: int = 8
    cpu_metrics_bytes: int = 24

    @property
    def hbm(self):
        return self.indexer_bytes + self.copy_source_bytes

    @property
    def dram(self):
        # Independent observed bounds, conservatively summed: pageable
        # isfinite/aten::ne scalar, synchronous CUDA scalar's pinned bin, and
        # the three-int64 metrics copy. These are execution scratch, not KV.
        return self.cpu_indexer_bytes + self.cpu_scalar_bytes + self.cpu_metrics_bytes

    @property
    def cpu_workspace_metadata(self):
        return {
            "workspace_cpu_indexer_bytes": self.cpu_indexer_bytes,
            "workspace_cpu_scalar_bytes": self.cpu_scalar_bytes,
            "workspace_cpu_metrics_bytes": self.cpu_metrics_bytes,
            "workspace_cpu_bytes": self.dram,
        }


def execution_reservation(query_tokens, context_tokens, *, topk, width, max_inflight_writes=2):
    """Conservative storage bound, invariant across a fixed-pool chunk sweep.

    Scores (native padded output, masked copy and mask/finite reduction), top-k
    values/IDs, global-ID remaps, union/sort storage, and bounded append source
    leases all count. The implementation may use less, but must never allocate
    beyond this shape contract or grow a buffer outside this reservation.
    """
    if any(
        type(x) is not int or x < 1
        for x in (query_tokens, context_tokens, topk, width, max_inflight_writes)
    ):
        raise ValueError("execution dimensions must be positive integers")
    columns = (context_tokens + 127) // 128 * 128
    selected = min(topk, context_tokens)
    # 16 bytes covers all concurrent logits/mask/finite-hint intermediates;
    # 64 per selected element includes PyTorch sort/unique and remap temporaries.
    indexer = 16 * query_tokens * columns + 64 * query_tokens * selected + 32 * columns
    copies = max_inflight_writes * query_tokens * width * 2
    return ExecutionReservation(query_tokens, context_tokens, indexer, copies)
