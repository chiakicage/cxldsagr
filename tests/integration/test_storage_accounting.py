"""Both DeepSeek storage scans retain exact capacity and alias semantics."""

import pytest
import torch

from cache.sparse_token_pool import _tensor_bytes
from models.deepseek_v32.execution.adapter import _storage_bytes


@pytest.mark.parametrize("measure", [_tensor_bytes, _storage_bytes])
def test_storage_scan_counts_current_whole_allocation_aliases_and_empty_views(measure):
    storage = torch.empty(65, dtype=torch.uint8)
    other = torch.empty(7, dtype=torch.uint8)
    view = storage[10:11]
    assert measure([view, storage[:0], other, None]) == {"hbm": 0, "dram": 72}
    storage.untyped_storage().resize_(128)
    assert measure([view, storage[:0], other, torch.empty(0)]) == {"hbm": 0, "dram": 135}


@pytest.mark.parametrize("measure", [_tensor_bytes, _storage_bytes])
def test_storage_scan_queries_once_per_tensor_and_retains_device_and_pinned_bins(measure):
    class Storage:
        def data_ptr(self):
            return 1234

        def nbytes(self):
            return 257

    class Tensor:
        def __init__(self, device, pinned=False):
            self.device = torch.device(device)
            self.pinned = pinned
            self.queries = 0

        def untyped_storage(self):
            self.queries += 1
            return Storage()

        def is_pinned(self):
            return self.pinned

    tensors = [Tensor("cpu", True), Tensor("cuda:0"), Tensor("cuda:1")]
    assert measure(tensors) == {"hbm": 514, "dram": 512}
    assert [tensor.queries for tensor in tensors] == [1, 1, 1]
