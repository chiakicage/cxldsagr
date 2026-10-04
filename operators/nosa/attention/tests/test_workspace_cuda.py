"""GPU verification of shared FA3 storage; these checks are not experiments."""

from types import SimpleNamespace

import pytest
import torch
import tvm_ffi

from layers.attention import BlockSelection
from operators.nosa.attention.device_only import _fa3
from operators.nosa.attention.device_only.api import nosa_block_sparse_attention
from operators.nosa.attention.offload.api import NosaFetchWorkspace
from operators.nosa.attention.workspace import NosaAttentionWorkspace


@pytest.fixture
def hopper(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("Shared NOSA workspace checks require SM90/Hopper")
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    torch.manual_seed(1337)


def _inputs(queries, heads=2):
    q = torch.randn(queries, heads * 16, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(64 + queries, heads, 128, device="cuda", dtype=q.dtype)
    values = torch.randn_like(keys)
    selection = BlockSelection(
        torch.zeros((queries, heads, 1), device="cuda", dtype=torch.int32), 64
    )
    return q, keys, values, selection


def test_cuda_workspace_matches_owned_for_varying_query_and_head_shapes(hopper, monkeypatch):
    workspace = NosaAttentionWorkspace(1032, 2, device="cuda", dtype=torch.bfloat16)
    pointers = [t.data_ptr() for t in workspace.tensors()]
    retained = []
    validate = workspace.validate_attention
    slices = workspace.slices
    checks, views = [], []

    def checked(*args):
        checks.append(len(args[0]))
        return validate(*args)

    def sliced(*args, **kwargs):
        views.append(args[0])
        return slices(*args, **kwargs)

    monkeypatch.setattr(workspace, "validate_attention", checked)
    monkeypatch.setattr(workspace, "slices", sliced)
    for queries, heads in ((1032, 2), (1024, 2), (1, 1), (0, 2), (9, 2), (1024, 2)):
        q, keys, values, selection = _inputs(queries, heads)
        expected = nosa_block_sparse_attention(q, keys, values, selection, 64)
        original_empty = torch.empty
        allocations = []

        def record(*args, _empty=original_empty, _allocations=allocations, **kwargs):
            result = _empty(*args, **kwargs)
            _allocations.append((tuple(result.shape), result.dtype))
            return result

        with monkeypatch.context() as context:
            context.setattr(torch, "empty", record)
            actual = nosa_block_sparse_attention(
                q, keys, values, selection, 64, workspace=workspace
            )
        torch.cuda.synchronize()
        assert allocations == [(tuple(q.shape), q.dtype)]
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert [t.data_ptr() for t in workspace.tensors()] == pointers
        retained.append((actual, expected))
    assert all(torch.equal(actual, expected) for actual, expected in retained)
    assert checks == views == [1032, 1024, 1, 0, 9, 1024]


def test_cuda_workspace_rejects_unbudgeted_dispatch_before_launch(hopper, monkeypatch):
    workspace = NosaAttentionWorkspace(8, 2, device="cuda", dtype=torch.bfloat16)
    q, keys, values, selection = _inputs(9)
    with pytest.raises(ValueError, match="reserved query/head"):
        nosa_block_sparse_attention(q, keys, values, selection, 64, workspace=workspace)
    q, keys, values, selection = _inputs(8)
    with pytest.raises(ValueError, match="BF16/D128/GQA16"):
        nosa_block_sparse_attention(
            q.half(), keys.half(), values.half(), selection, 64, workspace=workspace
        )
    distinct_stride = torch.empty((len(values), 2, 256), device="cuda", dtype=values.dtype)[
        ..., :128
    ]
    distinct_stride.copy_(values)
    with pytest.raises(ValueError, match="matching K/V layouts"):
        nosa_block_sparse_attention(q, keys, distinct_stride, selection, 64, workspace=workspace)
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "triton")
    with pytest.raises(NotImplementedError, match="native FA3"):
        nosa_block_sparse_attention(q, keys, values, selection, 64, workspace=workspace)


def test_direct_fa3_workspace_entry_still_validates_before_launch(hopper, monkeypatch):
    from operators.nosa.attention.device_only import _fa3

    workspace = NosaAttentionWorkspace(8, 2, device="cuda", dtype=torch.bfloat16)
    q, keys, values, selection = _inputs(9)

    def forbidden():
        raise AssertionError("invalid capacity reached the native module")

    monkeypatch.setattr(_fa3, "_module", forbidden)
    with pytest.raises(ValueError, match="reserved query/head"):
        _fa3.launch_nosa_fa3_attention(q, keys, values, selection, 64, None, workspace=workspace)


def test_workspace_capture_and_scratch_alias_rejections_precede_launch(hopper, monkeypatch):
    from operators.nosa.attention.device_only import _fa3

    workspace = NosaAttentionWorkspace(128, 2, device="cuda", dtype=torch.bfloat16)
    q, keys, values, selection = _inputs(8)

    def forbidden():
        raise AssertionError("invalid workspace use reached the native module")

    monkeypatch.setattr(_fa3, "_module", forbidden)
    alias = workspace._storage["pages"].view(torch.bfloat16).flatten()[: q.numel()].view(q.shape)
    with pytest.raises(ValueError, match="alias writable scratch"):
        nosa_block_sparse_attention(alias, keys, values, selection, 64, workspace=workspace)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(NotImplementedError, match="CUDA Graph"):
        nosa_block_sparse_attention(q, keys, values, selection, 64, workspace=workspace)


def test_cuda_bounded_fetch_rejects_queries_and_trace_before_mutation(hopper):
    workspace = NosaFetchWorkspace(
        128, 2, 128, device="cuda", dtype=torch.bfloat16, bounded=True, max_queries=8
    )
    workspace.keys.fill_(73)
    initial_bytes = workspace.capacity_bytes
    for queries, profile, message in ((9, False, "reserved query"), (8, True, "trace")):
        q, keys, values, selection = _inputs(queries)
        host_keys = keys.cpu().pin_memory()
        host_values = values.cpu().pin_memory()
        workspace.profile_work_intervals = profile
        with pytest.raises(ValueError, match=message):
            workspace.run(q, selection, host_keys, host_values, keys[64:], values[64:], None, 64)
        assert workspace.capacity_bytes == initial_bytes
        assert torch.all(workspace.keys == 73).item()


def _stream_inputs():
    torch.manual_seed(8128)
    q = torch.randn(128, 32, 128, device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn(256, 2, 128, device="cuda:0", dtype=q.dtype)
    v = torch.randn_like(k)
    ids = (
        torch.arange(4, device="cuda:0", dtype=torch.int32)
        .view(1, 1, 4)
        .expand(128, 2, 4)
        .contiguous()
    )
    return q, k, v, BlockSelection(ids, 64)


@torch.inference_mode()
def test_workspace_two_nondefault_streams_preserve_outputs_ffi_and_recorded_lifetimes(
    hopper, monkeypatch
):
    q, k, v, selection = _stream_inputs()
    qs = [q, q * 0.75]
    expected = [nosa_block_sparse_attention(x, k, v, selection, 128) for x in qs]
    workspace = NosaAttentionWorkspace(128, 2, device=q.device, dtype=q.dtype)
    storage_ids = {id(t) for t in workspace.tensors()}
    pointers = [t.data_ptr() for t in workspace.tensors()]
    ready = torch.cuda.Event()
    ready.record()
    streams = [torch.cuda.Stream(device=q.device), torch.cuda.Stream(device=q.device)]
    device = tvm_ffi.device("cuda", q.device.index)
    native = _fa3._module()
    record = torch.Tensor.record_stream
    records, submissions = [], []

    def recorded(tensor, stream):
        if id(tensor) in storage_ids:
            records.append((id(tensor), stream.cuda_stream))
        return record(tensor, stream)

    def submitted(*args):
        current = torch.cuda.current_stream(q.device.index)
        assert tvm_ffi.get_raw_stream(device) == current.cuda_stream
        submissions.append(current.cuda_stream)
        return native.fa3_forward(*args)

    monkeypatch.setattr(torch.Tensor, "record_stream", recorded)
    monkeypatch.setattr(_fa3, "_module", lambda: SimpleNamespace(fa3_forward=submitted))
    retained = []
    for index, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            stream.wait_event(ready)  # Serialize shared workspace, including cross-stream reuse.
            with tvm_ffi.use_raw_stream(device, streams[1 - index].cuda_stream):
                previous_ffi = tvm_ffi.get_raw_stream(device)
                output = nosa_block_sparse_attention(
                    qs[index], k, v, selection, 128, workspace=workspace
                )
                assert tvm_ffi.get_raw_stream(device) == previous_ffi
                assert torch.cuda.current_stream(q.device.index) == stream
                retained.append(output)
            ready.record(stream)
    torch.cuda.current_stream(q.device.index).wait_event(ready)
    torch.cuda.synchronize(q.device)
    assert submissions == [s.cuda_stream for s in streams]
    assert records == [(id(t), s.cuda_stream) for s in streams for t in workspace.tensors()]
    assert [t.data_ptr() for t in workspace.tensors()] == pointers
    for actual, control in zip(retained, expected, strict=True):
        torch.testing.assert_close(actual, control, atol=0, rtol=0)
    assert retained[0].data_ptr() != retained[1].data_ptr()


@torch.inference_mode()
def test_workspace_native_exception_restores_ffi_and_torch_stream(hopper, monkeypatch):
    q, k, v, selection = _stream_inputs()
    workspace = NosaAttentionWorkspace(128, 2, device=q.device, dtype=q.dtype)
    stream, prior = torch.cuda.Stream(device=q.device), torch.cuda.Stream(device=q.device)
    device = tvm_ffi.device("cuda", q.device.index)
    ready = torch.cuda.Event()
    ready.record()

    def fail(*args):
        assert tvm_ffi.get_raw_stream(device) == stream.cuda_stream
        raise RuntimeError("injected native workspace failure")

    with torch.cuda.stream(stream), tvm_ffi.use_raw_stream(device, prior.cuda_stream):
        stream.wait_event(ready)
        with monkeypatch.context() as patch:
            patch.setattr(_fa3, "_module", lambda: SimpleNamespace(fa3_forward=fail))
            with pytest.raises(RuntimeError, match="injected native workspace failure"):
                nosa_block_sparse_attention(q, k, v, selection, 128, workspace=workspace)
        assert tvm_ffi.get_raw_stream(device) == prior.cuda_stream
        assert torch.cuda.current_stream(q.device.index) == stream
        actual = nosa_block_sparse_attention(q, k, v, selection, 128, workspace=workspace)
        ready.record(stream)
    torch.cuda.current_stream(q.device.index).wait_event(ready)
    expected = nosa_block_sparse_attention(q, k, v, selection, 128)
    torch.cuda.synchronize(q.device)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
