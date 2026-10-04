"""Pinned-bin reservation and observable backing, without requiring a GPU."""

import pytest
import torch

from cache.host_allocation import allocate_host_tensor, pinned_allocation_bytes
from cache.sparse_token_pool import _tensor_bytes


@pytest.mark.parametrize(
    "requested, expected",
    [(0, 0), (1, 1), (4095, 4096), (4096, 4096), (4097, 8192), (1050624 * 1152, 2**31)],
)
def test_pinned_bin_boundaries(requested, expected):
    assert pinned_allocation_bytes(requested) == expected


def test_cpu_reference_keeps_unrounded_storage_and_deduplicates_views():
    tensor = allocate_host_tensor((3, 7), dtype=torch.float32, pin_memory=False)
    assert tensor.shape == (3, 7) and not tensor.is_pinned()
    assert tensor.untyped_storage().nbytes() == 84
    assert _tensor_bytes([tensor[:1], tensor[1:], tensor, None]) == {"hbm": 0, "dram": 84}


def test_pinned_backing_requests_full_bin_but_keeps_logical_shape(monkeypatch):
    # Emulate pinning only. A real CPU storage verifies the allocated capacity,
    # logical view and alias identity independently of the estimator formula.
    empty = torch.empty
    pinned = set()
    requests = []

    def allocate(*args, **kwargs):
        pin = kwargs.pop("pin_memory", False)
        result = empty(*args, **kwargs)
        if pin:
            requests.append(result.numel() * result.element_size())
            pinned.add(result.untyped_storage().data_ptr())
        return result

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(
        torch.Tensor, "is_pinned", lambda tensor: tensor.untyped_storage().data_ptr() in pinned
    )
    tensor = allocate_host_tensor((3, 7), dtype=torch.float32, pin_memory=True)
    assert requests == [128]
    assert tensor.shape == (3, 7) and tensor.is_contiguous()
    assert tensor.numel() * tensor.element_size() == 84
    assert tensor.untyped_storage().nbytes() == 128
    assert _tensor_bytes([tensor[:1], tensor[1:]]) == {"hbm": 0, "dram": 128}


@pytest.mark.parametrize("fault", ["capacity", "pin_state"])
def test_post_allocation_contract_rejects_unexpected_storage(monkeypatch, fault):
    empty = torch.empty

    def allocate(*args, **kwargs):
        pin = kwargs.pop("pin_memory", False)
        if pin and fault == "capacity":
            return empty(args[0] + 1, **kwargs)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    with pytest.raises(RuntimeError, match="capacity|pinned state"):
        allocate_host_tensor((3, 7), dtype=torch.float32, pin_memory=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_pinned_backing_retains_bin_after_stream_ordered_copy():
    tensor = allocate_host_tensor((137,), dtype=torch.uint8, pin_memory=True)
    assert tensor.is_pinned() and tensor.shape == (137,)
    assert tensor.untyped_storage().nbytes() == 256
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.arange(137, device="cuda", dtype=torch.uint8)
        tensor.copy_(source, non_blocking=True)
    stream.synchronize()
    torch.testing.assert_close(tensor, torch.arange(137, dtype=torch.uint8))
    assert _tensor_bytes([tensor[:1], tensor[1:]]) == {"hbm": 0, "dram": 256}
