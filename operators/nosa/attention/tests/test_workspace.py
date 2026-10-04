"""Storage bounds and actual-shape views for shared native attention scratch."""

import subprocess
import sys

import pytest
import torch

from operators.nosa.attention.offload.api import NosaFetchWorkspace, estimate_trace_rows
from operators.nosa.attention.workspace import NosaAttentionWorkspace


def _storage_bytes(workspace):
    # Inspect actual storage rather than deriving another copy of the formula.
    storages = {
        (tensor.device, tensor.untyped_storage().data_ptr()): tensor.untyped_storage().nbytes()
        for tensor in workspace.tensors()
        if tensor.numel()
    }
    return sum(storages.values())


def test_workspace_import_does_not_load_gpu_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, torch; before = set(sys.modules); "
                "from operators.nosa.attention.workspace import NosaAttentionWorkspace; "
                "from operators.nosa.attention.offload.api import NosaFetchWorkspace; "
                "added = set(sys.modules) - before; "
                "assert not any(x == 'tvm_ffi' or x.startswith('tvm_ffi.') or "
                "x == 'triton' or x.startswith('triton.') or x.endswith('._fa3') for x in added)"
            ),
        ],
        check=True,
    )


def test_workspace_estimates_do_not_allocate_or_inspect_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pure estimate attempted allocation or CUDA inspection")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    assert NosaAttentionWorkspace.estimate_capacity_bytes(1032, 2, dtype=torch.bfloat16) > 0
    rows = estimate_trace_rows(2048, 1032, 2)
    assert rows >= estimate_trace_rows(2048, 1024, 2)
    assert (
        NosaFetchWorkspace.estimate_capacity_bytes(
            2048, 2, 128, dtype=torch.bfloat16, max_queries=1032, trace_capacity=rows
        )
        > rows * 32
    )


@pytest.mark.parametrize("head_dim,dtype", [(128, torch.bfloat16), (4, torch.float32)])
def test_cpu_actual_storage_equals_reservation(head_dim, dtype):
    queries, heads, capacity = 1032, 2, 2048
    scratch = NosaAttentionWorkspace(queries, heads, head_dim, device="cpu", dtype=dtype)
    assert _storage_bytes(scratch) == scratch.capacity_bytes
    assert scratch.capacity_bytes == scratch.estimate_capacity_bytes(
        queries, heads, head_dim, dtype=dtype
    )
    assert scratch.allocation_sizes(queries, heads, head_dim, dtype=dtype) == tuple(
        tensor.untyped_storage().nbytes() for tensor in scratch.tensors()
    )
    rows = estimate_trace_rows(capacity, queries, heads)
    fetch = NosaFetchWorkspace(
        capacity,
        heads,
        head_dim,
        device="cpu",
        dtype=dtype,
        max_queries=queries,
        bounded=True,
        trace_capacity=rows,
    )
    assert _storage_bytes(fetch) == fetch.capacity_bytes
    assert fetch.capacity_bytes == fetch.estimate_capacity_bytes(
        capacity, heads, head_dim, dtype=dtype, max_queries=queries, trace_capacity=rows
    )
    assert fetch.allocation_sizes(
        capacity, heads, head_dim, dtype=dtype, max_queries=queries, trace_capacity=rows
    ) == tuple(tensor.untyped_storage().nbytes() for tensor in fetch.tensors())
    # CPU tensors prove accounting only; the CUDA kernel remains unavailable.
    with pytest.raises(NotImplementedError, match="allocation-only"):
        fetch.run(None, None, None, None, None, None, None, 0)


@pytest.mark.parametrize("queries,heads", [(0, 2), (1, 1), (9, 2), (1024, 2), (1032, 2)])
def test_slices_have_actual_shapes_without_allocation(monkeypatch, queries, heads):
    workspace = NosaAttentionWorkspace(1032, 2, device="cpu", dtype=torch.bfloat16)
    pointers = {t.untyped_storage().data_ptr() for t in workspace.tensors()}

    def forbidden(*args, **kwargs):
        raise AssertionError("slicing allocated a new tensor")

    monkeypatch.setattr(torch, "empty", forbidden)
    views = workspace.slices(queries, heads, device="cpu", dtype=torch.bfloat16)
    batches = ((queries + 7) // 8) * heads
    assert views["fallback"].shape == ((queries + 3) // 4, heads)
    assert views["pages"].shape == views["members"].shape == (batches, 512)
    assert views["counts"].shape == (batches * (2 if batches == 256 else 1),)
    assert all(t.is_contiguous() for t in views.values())
    assert all(t.untyped_storage().data_ptr() in pointers for t in views.values())
    # Max q=1032 has 258 batches, but q=1024 needs 512 counts, not 258.
    assert workspace._storage["counts"].numel() == 516


def test_bounded_workspace_rejects_growth_without_storage_mutation():
    workspace = NosaFetchWorkspace(
        2048,
        2,
        128,
        device="cpu",
        dtype=torch.bfloat16,
        max_queries=1032,
        bounded=True,
        trace_capacity=512,
    )
    before = [(t.data_ptr(), t.shape) for t in workspace.tensors()]
    workspace.reserve(1024, trace_rows=512)
    with pytest.raises(ValueError, match="cannot grow"):
        workspace.reserve(1033)
    with pytest.raises(ValueError, match="cannot grow"):
        workspace.reserve(1024, trace_rows=513)
    with pytest.raises(ValueError, match="reserved query/head"):
        workspace._attention_scratch.slices(1033)
    with pytest.raises(ValueError, match="reserved query/head"):
        workspace._attention_scratch.slices(1, 3)
    with pytest.raises(ValueError, match="dtype mismatch"):
        workspace._attention_scratch.slices(1, dtype=torch.float32)
    assert [(t.data_ptr(), t.shape) for t in workspace.tensors()] == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_queries": 0},
        {"max_queries": True},
        {"max_queries": 2049},
        {"trace_capacity": -1},
        {"trace_capacity": True},
        {"bounded": True},
    ],
)
def test_fetch_invalid_reservation_fails_before_allocation(monkeypatch, kwargs):
    def forbidden(*args, **kw):
        raise AssertionError("invalid reservation reached allocation")

    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(ValueError):
        NosaFetchWorkspace(2048, 2, 128, device="cpu", dtype=torch.bfloat16, **kwargs)
