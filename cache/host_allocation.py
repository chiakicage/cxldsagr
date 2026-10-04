"""Owned host capacity for PyTorch's CUDA CachingHostAllocator.

The pinned allocator in the project's pinned PyTorch version rounds each
allocation to a power of two (ATen/core/CachingHostAllocator.h). Logical tensor
bytes can therefore understate owned DRAM. Cached blocks without a tensor owner
belong to process allocator statistics, not this cache-capacity accounting.
"""

import math

import torch


def pinned_allocation_bytes(nbytes):
    """Return the allocator bin for one pinned allocation, without allocating."""
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError("allocation bytes must be a nonnegative integer")
    return 1 << (nbytes - 1).bit_length() if nbytes else 0


def storage_allocation_bytes(tensor):
    """Count the complete owning storage, including a pinned allocator's bin."""
    nbytes = tensor.untyped_storage().nbytes()
    return (
        pinned_allocation_bytes(nbytes)
        if tensor.device.type == "cpu" and tensor.is_pinned()
        else nbytes
    )


def allocate_host_tensor(shape, *, dtype, pin_memory):
    """Expose the entire reserved pinned bin through the logical view's storage.

    Callers reserve this capacity before allocation. Requesting the bin itself
    lets post-allocation storage inspection check its capacity; a shorter view
    retains the requested record shape and all transfer byte counts. Ordinary
    CPU reference allocations keep their unrounded capacity.
    """
    shape = tuple(shape)
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("host tensor dimensions must be nonnegative integers")
    numel = math.prod(shape)
    itemsize = dtype.itemsize
    logical_bytes = numel * itemsize
    reserved = pinned_allocation_bytes(logical_bytes) if pin_memory else logical_bytes
    if reserved % itemsize:
        raise ValueError("host allocation bin must contain whole tensor elements")
    backing = torch.empty(reserved // itemsize, dtype=dtype, device="cpu", pin_memory=pin_memory)
    result = backing[:numel].view(shape)
    if result.untyped_storage().nbytes() != reserved:
        raise RuntimeError("host storage capacity differs from its pre-allocation reservation")
    if logical_bytes and result.is_pinned() != bool(pin_memory):
        raise RuntimeError("host allocation does not have the requested pinned state")
    if storage_allocation_bytes(result) != reserved:
        raise RuntimeError("host allocator bin differs from its pre-allocation reservation")
    return result
