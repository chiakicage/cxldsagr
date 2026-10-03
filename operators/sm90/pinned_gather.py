"""GPU-issued indexed reads from pinned Host rows into contiguous HBM rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(host, indices, destination, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    source = tl.load(indices + row).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    # Promote before multiplication: a pinned pool can span well beyond 4 GiB.
    values = tl.load(host + source * WIDTH + columns, columns < WIDTH, other=0)
    tl.store(destination + row.to(tl.int64) * WIDTH + columns, values, columns < WIDTH)


def gather_pinned_rows(host, indices, destination):
    """Caller owns index validity, stream dependencies and allocation lifetimes.

    Like ECHO recall, the GPU loads pinned system memory itself, not via a
    CPU gather or cudaMemcpyAsync. No staging tensor or allocator is used here.
    """
    if (
        host.device.type != "cpu"
        or not host.is_pinned()
        or not destination.is_cuda
        or not indices.is_cuda
        or indices.device != destination.device
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.ndim != 1
        or host.ndim != 3
        or destination.ndim != 3
        or host.shape[1:] != destination.shape[1:]
        or host.dtype != destination.dtype
        or host.dtype != torch.bfloat16
        or not all(x.is_contiguous() for x in (host, indices, destination))
        or destination.shape[0] != indices.numel()
    ):
        raise ValueError("expected pinned BF16 rows, CUDA indices and matching contiguous output")
    if indices.numel():
        width = host.shape[1] * host.shape[2]
        with torch.cuda.device(destination.device):
            _gather[(indices.numel(),)](
                host, indices, destination, width, triton.next_power_of_2(width), num_warps=4
            )
