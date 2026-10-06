import pytest
import torch

from operators.common.kv_transfer import copy_host_records_async, gather_host_records


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("host_offset,device_offset", [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("max_ctas", [None, 1, 2])
@pytest.mark.parametrize(
    "dtype,width", [(torch.bfloat16, 8), (torch.bfloat16, 576), (torch.float32, 19)]
)
def test_cuda_gather_contiguous_storage_offsets_preserves_other_bytes(
    dtype, width, host_offset, device_offset, max_ctas
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
    gather_host_records(host, device, host_ids, device_ids, max_ctas=max_ctas)
    torch.cuda.synchronize()

    torch.testing.assert_close(device_storage.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(host_storage, source_before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count_dtype", [torch.int64, torch.uint32])
@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("max_ctas", [None, 1])
def test_cuda_counted_gather_never_reads_uninitialized_tail(count_dtype, count, max_ctas):
    host = torch.arange(6 * 576, dtype=torch.float32).bfloat16().reshape(6, 576).pin_memory()
    device = torch.full((6, 576), -17, dtype=host.dtype, device="cuda")
    # Invalid tail IDs make any access beyond the counted prefix a hard failure.
    host_ids = torch.full((8,), -1, dtype=torch.int64, device="cuda")
    device_ids = torch.full_like(host_ids, -1)
    host_ids[:count] = torch.tensor([4, 1, 5][:count], device="cuda")
    device_ids[:count] = torch.tensor([1, 3, 2][:count], device="cuda")
    valid_count = torch.tensor([count], dtype=count_dtype, device="cuda")
    expected = device.cpu()
    expected[[1, 3, 2][:count]] = host[[4, 1, 5][:count]]

    gather_host_records(
        host, device, host_ids, device_ids, max_ctas=max_ctas, valid_count=valid_count
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(device.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("max_ctas", [0, -1, True, 1.5, 2**31])
def test_gather_rejects_invalid_grid_before_accessing_tensors(max_ctas):
    with pytest.raises(ValueError, match="max_ctas"):
        gather_host_records(None, None, None, None, max_ctas=max_ctas)


@pytest.mark.parametrize("name", ["host_start", "device_start", "count"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_contiguous_copy_rejects_invalid_span_before_accessing_tensors(name, value):
    kwargs = {"host_start": 0, "device_start": 0, "count": 0}
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        copy_host_records_async(None, None, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "dtype,width", [(torch.bfloat16, 576), (torch.float32, 7), (torch.uint8, 13)]
)
@pytest.mark.parametrize("count", [0, 3])
def test_cuda_contiguous_copy_on_private_stream_preserves_guards_and_source(dtype, width, count):
    # Unaligned tensor storage offsets and unequal row offsets must remain exact.
    host_storage = torch.arange(8 * width + 3, dtype=torch.float32).to(dtype).pin_memory()
    host = host_storage[1 : 1 + 8 * width].view(8, width)
    device_storage = torch.full((8 * width + 5,), 17, dtype=dtype, device="cuda")
    device = device_storage[2 : 2 + 8 * width].view(8, width)
    expected = device_storage.cpu()
    expected[2 : 2 + 8 * width].view(8, width)[3 : 3 + count] = host[1 : 1 + count]
    source_before = host_storage.clone()
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        copy_host_records_async(host, device, host_start=1, device_start=3, count=count)
        ready = torch.cuda.Event()
        ready.record()
    with torch.cuda.stream(consumer):
        consumer.wait_event(ready)
        observed = device_storage.clone()
    consumer.synchronize()
    torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(host_storage, source_before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "host_start,device_start,count", [(4, 0, 1), (0, 4, 1), (3, 0, 2), (0, 3, 2)]
)
def test_cuda_contiguous_copy_rejects_out_of_bounds_span(host_start, device_start, count):
    host = torch.zeros(4, 7, pin_memory=True)
    device = torch.zeros_like(host, device="cuda")
    with pytest.raises(ValueError, match="span exceeds"):
        copy_host_records_async(
            host, device, host_start=host_start, device_start=device_start, count=count
        )
