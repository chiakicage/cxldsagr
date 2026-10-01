import pytest
import torch

from operators.common.kv_transfer import gather_host_records


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("host_offset,device_offset", [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize(
    "dtype,width", [(torch.bfloat16, 8), (torch.bfloat16, 576), (torch.float32, 19)]
)
def test_cuda_gather_contiguous_storage_offsets_preserves_other_bytes(
    dtype, width, host_offset, device_offset
):
    rows = 4
    # Keep guards around both views, including an offset smaller than uint4.
    host_storage = torch.arange(rows * width + 8, dtype=torch.float32).to(dtype).pin_memory()
    host = host_storage[host_offset : host_offset + rows * width].view(rows, width)
    device_storage = torch.full((rows * width + 8,), -17, dtype=dtype, device="cuda")
    device = device_storage[device_offset : device_offset + rows * width].view(rows, width)
    expected = device_storage.cpu()
    source_before = host_storage.clone()
    host_ids = torch.tensor([3, 0, 2], dtype=torch.int64, device="cuda")
    device_ids = torch.tensor([1, 3, 0], dtype=torch.int64, device="cuda")
    expected_view = expected[device_offset : device_offset + rows * width].view(rows, width)
    expected_view[[1, 3, 0]] = host[[3, 0, 2]]

    assert host.is_contiguous() and device.is_contiguous() and host.is_pinned()
    gather_host_records(host, device, host_ids, device_ids)
    torch.cuda.synchronize()

    torch.testing.assert_close(device_storage.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(host_storage, source_before, rtol=0, atol=0)
