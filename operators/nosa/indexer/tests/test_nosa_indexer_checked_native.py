"""Checked ranked submission preserves failure bytes, stream ordering and async fallback."""

import pytest
import torch

from operators.nosa._native import load_module
from operators.nosa.indexer._indexer_checked_cuda import select_prepared_out
from operators.nosa.indexer._prepare_cuda import PreparationScratch
from operators.nosa.indexer._prepare_ranked_cuda import prepare_ranked_out
from operators.nosa.indexer.compression import update_compressed_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Hopper CUDA required")


def buffers(rows=1024, heads=2):
    torch.manual_seed(681)
    prefix, length = 32768, 32768 + rows
    # Preserve a QKV-style gap between token rows.
    q_storage = torch.randn((rows, heads * 16 + 4, 128), device="cuda", dtype=torch.bfloat16)
    q = q_storage[:, : heads * 16].view(rows, heads, 16, 128)
    k = torch.randn((length, heads, 128), device="cuda", dtype=torch.bfloat16)
    cis = torch.randn((length, heads), device="cuda", dtype=torch.bfloat16)
    count, stable = length // 16 - 1, (length - 16) // 64
    ck = torch.full((count, heads, 128), 17, device="cuda", dtype=q.dtype)
    cc = torch.full((count, heads), 17, device="cuda", dtype=q.dtype)
    pool = torch.full((stable, heads), 17, device="cuda", dtype=q.dtype)
    update_compressed_cache(
        k[:prefix], cis[:prefix], ck, cc, pool, compressed_start=0, pooled_start=0
    )
    workspace = torch.full((rows * heads, (length + 63) // 64), 17, device="cuda", dtype=q.dtype)
    normalizers = torch.full((1, rows, heads, 16, 2), 17, device="cuda", dtype=torch.float32)
    ids = torch.full((rows, heads, 64), 17, device="cuda", dtype=torch.int64)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    ranking = torch.full((heads, 64), 17, device="cuda", dtype=torch.int32)
    args = (q, k, cis, ck, cc, pool, workspace, normalizers, ids, valid, ranking)
    options = {
        "query_start": prefix,
        "validated_start": prefix,
        "compressed_start": prefix // 16 - 1,
        "pooled_start": (prefix - 16) // 64,
        "scratch": PreparationScratch.allocate("cuda", include_host=True),
    }
    return args, options


def byte_snapshots(tensors):
    return [tensor.view(torch.uint8).clone() for tensor in tensors]


def assert_bytes(tensors, snapshots):
    for actual, expected in zip(tensors, snapshots, strict=True):
        torch.testing.assert_close(actual.view(torch.uint8), expected, rtol=0, atol=0)


def separate_reference(args, options):
    q, k, cis, *outputs = args
    copies = tuple(t.clone() for t in outputs)
    ck, cc, pool, work, norms, ids, valid, ranking = copies
    finite = prepare_ranked_out(q, k, cis, ck, cc, pool, ranking, **options)
    assert bool(finite)
    import tvm_ffi

    with torch.cuda.device(q.device), tvm_ffi.use_torch_stream():
        load_module("nosa_indexer").indexer_ranked_out(
            q, ck, cc, pool, work, norms, ids, valid, ranking, options["query_start"]
        )
    torch.cuda.synchronize()
    return copies


@pytest.mark.parametrize("rows,heads", [(128, 2), (512, 2), (1024, 2), (1088, 2), (1024, 1)])
@pytest.mark.parametrize("custom_stream", [False, True])
@torch.inference_mode()
def test_cuda_checked_ranked_matches_separate_calls_exactly(rows, heads, custom_stream):
    args, options = buffers(rows, heads)
    expected = separate_reference(args, options)
    stream = torch.cuda.Stream() if custom_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Queue work before entry to verify preparation consumes this stream's state.
        args[0].mul_(1)
        ids, valid = select_prepared_out(*args, **options)
    stream.synchronize()
    assert ids.data_ptr() == args[8].data_ptr()
    assert valid.data_ptr() == args[9].data_ptr()
    assert_bytes(args[3:], byte_snapshots(expected))


@pytest.mark.parametrize("field", [0, 1, 2])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("rows", [128, 1024])
@torch.inference_mode()
def test_cuda_nonfinite_inputs_leave_all_derived_ranking_and_downstream_bytes_unchanged(
    field, bad, rows
):
    args, options = buffers(rows)
    protected = args[3:]
    before = byte_snapshots(protected)
    tensor = args[field]
    index = (-1,) * tensor.ndim
    tensor[index] = bad
    with pytest.raises(ValueError, match="finite Q, K and CIS"):
        select_prepared_out(*args, **options)
    assert_bytes(protected, before)
    assert options["scratch"].host_finite.item() is False


@pytest.mark.parametrize("overlap", ["partial_workspace", "finite_ranking", "partial_keys"])
@torch.inference_mode()
def test_cuda_validation_scratch_must_not_alias_inputs_or_protected_outputs(overlap):
    args, options = buffers()
    scratch = options["scratch"]
    if overlap == "partial_workspace":
        partial, finite = args[6].view(torch.uint8).flatten()[:3072], scratch.finite
    elif overlap == "finite_ranking":
        partial, finite = (
            scratch.partial,
            args[-1].view(torch.uint8).flatten()[:1].view(torch.bool).reshape(()),
        )
    else:
        partial, finite = args[1].view(torch.uint8).flatten()[:3072], scratch.finite
    options["scratch"] = PreparationScratch(partial, finite, scratch.host_finite)
    before = byte_snapshots(args[1:])
    with pytest.raises(Exception, match="must not overlap"):
        select_prepared_out(*args, **options)
    assert_bytes(args[1:], before)


@pytest.mark.parametrize("rows", [128, 1024])
@pytest.mark.parametrize(
    "left,right",
    [
        (10, 6),  # prepared ranking and score workspace
        (10, 7),  # prepared ranking and normalizers
        (10, 8),  # prepared ranking and selected IDs
        (10, 9),  # prepared ranking and validity
        (10, 3),  # prepared ranking and derived keys
        (10, 2),  # prepared ranking and input CIS
        (4, 6),  # derived CIS and score workspace
        (5, 3),  # pooled CIS and compressed keys
        (9, 8),  # validity and selected IDs
        (6, 1),  # score workspace and input keys
        (7, 0),  # normalizers and strided input query
    ],
)
@torch.inference_mode()
def test_cuda_writable_buffers_must_be_disjoint_before_any_write(rows, left, right):
    args, options = buffers(rows)
    changed = list(args)
    target = changed[left]
    if right == 0:
        # Q's token stride has a gap, but its storage still owns the full span.
        source_bytes = changed[right].untyped_storage()
        alias = torch.empty(0, device="cuda", dtype=target.dtype).set_(
            source_bytes, 0, target.shape, target.stride()
        )
    else:
        source = changed[right].view(torch.uint8).flatten()
        size = target.numel() * target.element_size()
        assert size <= source.numel()
        alias = source[:size].view(target.dtype).view(target.shape)
    changed[left] = alias
    scratch = options["scratch"]
    protected = (*args, *changed, scratch.partial, scratch.finite, scratch.host_finite)
    before = byte_snapshots(protected)
    with pytest.raises(Exception, match="must not overlap"):
        select_prepared_out(*changed, **options)
    assert_bytes(protected, before)


@torch.inference_mode()
def test_cuda_checked_rejects_graph_capture_before_any_write_but_async_prepare_remains_capturable():
    args, options = buffers()
    load_module("nosa_indexer_checked")
    expected = separate_reference(args, options)
    before = byte_snapshots(args[3:])
    with (
        pytest.raises(Exception, match="cannot be CUDA Graph captured"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        select_prepared_out(*args, **options)
    assert_bytes(args[3:], before)
    q, k, cis, ck, cc, pool, _, _, _, _, ranking = args
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        finite = prepare_ranked_out(q, k, cis, ck, cc, pool, ranking, **options)
    graph.replay()
    torch.cuda.synchronize()
    assert bool(finite)
    assert_bytes((ck, cc, pool, ranking), byte_snapshots((*expected[:3], expected[-1])))


@pytest.mark.parametrize(
    "problem", ["unpinned", "wrong_normalizers", "wrong_query", "noninteger_bounds"]
)
@torch.inference_mode()
def test_cuda_checked_rejects_invalid_contract_before_writing(problem):
    args, options = buffers()
    before = byte_snapshots(args[3:])
    changed = list(args)
    if problem == "unpinned":
        scratch = options["scratch"]
        options["scratch"] = PreparationScratch(
            scratch.partial, scratch.finite, torch.empty((), dtype=torch.bool)
        )
    elif problem == "wrong_normalizers":
        changed[7] = changed[7][:, :-1]
    elif problem == "wrong_query":
        changed[0] = changed[0][:, :, :8]
    else:
        options["validated_start"] = True
    with pytest.raises(Exception, match="Check failed|pinned host bool|bounds must"):
        select_prepared_out(*changed, **options)
    assert_bytes(args[3:], before)
