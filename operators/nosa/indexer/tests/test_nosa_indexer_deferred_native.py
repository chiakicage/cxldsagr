"""Finite-byte equality and failed-preparation isolation for short prefixes."""

import pytest
import torch

from operators.nosa.indexer import _indexer_deferred_cuda as deferred
from operators.nosa.indexer._prepare_cuda import PreparationScratch, prepare_out
from operators.nosa.indexer._prepare_ranked_cuda import prepare_ranked_out
from operators.nosa.indexer.api import _launch_scores, select_contiguous_blocks
from operators.nosa.indexer.compression import update_compressed_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Hopper CUDA required")


def buffers(rows, prefix):
    torch.manual_seed(947)
    length, heads = prefix + rows, 2
    q_storage = torch.randn((rows, heads * 16 + 4, 128), device="cuda", dtype=torch.bfloat16)
    q = q_storage[:, : heads * 16].view(rows, heads, 16, 128)
    k = torch.randn((length, heads, 128), device="cuda", dtype=q.dtype)
    cis = torch.randn((length, heads), device="cuda", dtype=q.dtype)
    count, stable, blocks = (
        max(0, length // 16 - 1),
        max(0, (length - 16) // 64),
        (length + 63) // 64,
    )
    # Full capacities include an unused tail; the deferred preflight protects it.
    ck = torch.full((count + 17, heads, 128), 17, device="cuda", dtype=q.dtype)
    cc = torch.full((count + 17, heads), 17, device="cuda", dtype=q.dtype)
    pool = torch.full((stable + 17, heads), 17, device="cuda", dtype=q.dtype)
    if prefix:
        update_compressed_cache(
            k[:prefix], cis[:prefix], ck, cc, pool, compressed_start=0, pooled_start=0
        )
    work = (
        torch.full((rows * heads, blocks), 17, device="cuda", dtype=q.dtype)
        if blocks > 64
        else None
    )
    norms = (
        torch.full((1, rows, heads, 16, 2), 17, device="cuda", dtype=torch.float32)
        if blocks > 64
        else None
    )
    ids = torch.full((rows, heads, 64), 17, device="cuda", dtype=torch.int64)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    ranking = torch.full((heads, 64), 17, device="cuda", dtype=torch.int32) if blocks > 64 else None
    args = (q, k, cis, ck, cc, pool, work, norms, ids, valid, ranking)
    options = {
        "query_start": prefix,
        "validated_start": prefix,
        "compressed_start": max(0, prefix // 16 - 1),
        "pooled_start": max(0, (prefix - 16) // 64),
        "scratch": PreparationScratch.allocate("cuda"),
    }
    return args, options


def snapshots(tensors):
    return [value.view(torch.uint8).clone() for value in tensors if value is not None]


def assert_bytes(tensors, before):
    for value, expected in zip((t for t in tensors if t is not None), before, strict=True):
        torch.testing.assert_close(value.view(torch.uint8), expected, rtol=0, atol=0)


def ordinary_reference(args, options):
    q, k, cis = args[:3]
    copies = tuple(t.clone() if t is not None else None for t in args[3:])
    ck, cc, pool, work, norms, _, _, ranking = copies
    count, stable = max(0, len(k) // 16 - 1), max(0, (len(k) - 16) // 64)
    prepare_options = {key: value for key, value in options.items() if key != "query_start"}
    if ranking is None:
        finite = prepare_out(q, k, cis, ck, cc, pool, **prepare_options)
    else:
        finite = prepare_ranked_out(
            q, k, cis, ck, cc, pool, ranking, query_start=options["query_start"], **prepare_options
        )
    assert bool(finite)
    ids, valid = select_contiguous_blocks(
        q,
        ck[:count],
        cc[:count],
        options["query_start"],
        len(k),
        return_valid_mask=True,
        pooled_cis=pool[:stable],
        workspace=None if work is None else work.flatten(),
        prepared_ranking=ranking,
    )
    return ck, cc, pool, work, norms, ids, valid, ranking


@pytest.mark.parametrize("length", [1024 * n for n in range(1, 32)])
@torch.inference_mode()
def test_each_affected_q1024_prefix_matches_original_score_and_selection_bytes(length):
    args, options = buffers(1024, length - 1024)
    expected = ordinary_reference(args, options)
    deferred.select_prepared_out(*args, **options)
    torch.cuda.synchronize()
    assert options["scratch"].finite.item()
    # Every affected Q1024 prefix uses the original Triton score path, whose
    # unused normalizer bytes remain unchanged on both sides.
    assert_bytes(args[3:], snapshots(expected))


@pytest.mark.parametrize("length", [4096, 4224, 8176, 8192, 8208, 32752])
@torch.inference_mode()
def test_q128_short_ranked_and_native_score_thresholds_match_original_bytes(length):
    args, options = buffers(128, length - 128)
    expected = ordinary_reference(args, options)
    deferred.select_prepared_out(*args, **options)
    torch.cuda.synchronize()
    assert options["scratch"].finite.item()
    # The original wrapper's native normalizers are internal, so compare all
    # externally visible derived/score/ranking/selection bytes here.
    assert_bytes((*args[3:7], *args[8:]), snapshots((*expected[:4], *expected[5:])))


@pytest.mark.parametrize("rows", [128, 1024])
@pytest.mark.parametrize("count", [510, 511, 512, 2046, 2047, 2048])
@torch.inference_mode()
def test_guarded_standalone_score_bytes_match_original_around_both_native_cutoffs(rows, count):
    torch.manual_seed(1053)
    q = torch.randn((rows, 2, 16, 128), device="cuda", dtype=torch.bfloat16)
    ck = torch.randn((count, 2, 128), device="cuda", dtype=q.dtype)
    length, start = (count + 1) * 16, (count + 1) * 16 - rows
    blocks = (length + 63) // 64
    expected = torch.empty((rows * 2, blocks), device="cuda", dtype=q.dtype)
    actual = torch.empty_like(expected)
    norms = torch.empty((1, rows, 2, 16, 2), device="cuda")
    finite = torch.ones((), device="cuda", dtype=torch.bool)
    _launch_scores(q, ck, None, expected, start, blocks, pool_output=True)
    deferred._launch_guarded_scores(q, ck, actual, norms, finite, start, blocks)
    torch.cuda.synchronize()
    assert_bytes((actual,), snapshots((expected,)))


@pytest.mark.parametrize("rows,prefix", [(1024, 0), (1024, 4096), (1024, 30720), (128, 8192)])
@pytest.mark.parametrize("field", [0, 1, 2])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@torch.inference_mode()
def test_nonfinite_preserves_full_derived_score_ranking_and_normalizer_capacities(
    rows, prefix, field, bad
):
    args, options = buffers(rows, prefix)
    if args[-1] is not None:
        args[-1].zero_()  # Packed zero would encode an invalid block if consumed.
    args[field][(-1,) * args[field].ndim] = bad
    protected = (*args[3:8], args[-1])
    before = snapshots(protected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        deferred.select_prepared_out(*args, **options)
        selected = args[8].clone()
    stream.synchronize()
    assert not options["scratch"].finite.item()
    assert_bytes(protected, before)
    assert (selected == -1).all()
    assert not args[9].any()


@pytest.mark.parametrize("rows,prefix", [(1024, 0), (1024, 4096), (128, 8192)])
@pytest.mark.parametrize("field", [0, 1, 2])
@torch.inference_mode()
def test_nonfinite_never_reads_uninitialized_outputs_for_compute_sanitizer(rows, prefix, field):
    args, options = buffers(rows, prefix)
    changed = list(args)
    # Do not snapshot these allocations: initcheck must see genuinely unwritten
    # derived, ranking, workspace and normalizer storage on the false path.
    for index in (3, 4, 5, 6, 7, 10):
        if changed[index] is not None:
            changed[index] = torch.empty_like(changed[index])
    changed[field][(-1,) * changed[field].ndim] = torch.nan
    deferred.select_prepared_out(*changed, **options)
    torch.cuda.synchronize()
    assert not options["scratch"].finite.item()
    assert (changed[8] == -1).all()
    assert not changed[9].any()


@pytest.mark.parametrize(
    "problem",
    ["partial_work", "partial_unused_tail", "ranking_unused_tail", "finite_ids", "mask_ids"],
)
@torch.inference_mode()
def test_aliases_rejected_before_any_producer_write_including_unused_capacity(problem):
    args, options = buffers(1024, 4096)
    changed = list(args)
    scratch = options["scratch"]
    count = len(args[1]) // 16 - 1
    tail = args[3][count:].view(torch.uint8).flatten()
    if problem == "partial_work":
        scratch = PreparationScratch(args[6].view(torch.uint8).flatten()[:3072], scratch.finite)
    elif problem == "partial_unused_tail":
        scratch = PreparationScratch(tail[:3072], scratch.finite)
    elif problem == "ranking_unused_tail":
        changed[-1] = tail[:512].view(torch.int32).view(2, 64)
    elif problem == "finite_ids":
        flag = args[8].view(torch.uint8).flatten()[:1].view(torch.bool).reshape(())
        scratch = PreparationScratch(scratch.partial, flag)
    else:
        changed[9] = (
            args[8].view(torch.uint8).flatten()[: args[9].numel()].view(torch.bool).view_as(args[9])
        )
    options["scratch"] = scratch
    protected = (*args[1:], scratch.partial, scratch.finite)
    before = snapshots(protected)
    with pytest.raises(ValueError, match="must not overlap"):
        deferred.select_prepared_out(*changed, **options)
    assert_bytes(protected, before)


@pytest.mark.parametrize("prefix", [0, 4096])
@torch.inference_mode()
def test_guarded_short_submission_captures_and_replays_safe_failure(prefix):
    args, options = buffers(1024, prefix)
    expected = ordinary_reference(args, options)
    deferred.select_prepared_out(*args, **options)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deferred.select_prepared_out(*args, **options)
    graph.replay()
    torch.cuda.synchronize()
    assert_bytes(args[3:], snapshots(expected))
    args[0][-1, -1, -1, -1] = torch.nan
    before = snapshots((*args[3:8], args[-1]))
    graph.replay()
    torch.cuda.synchronize()
    assert not options["scratch"].finite.item()
    assert_bytes((*args[3:8], args[-1]), before)
    assert (args[8] == -1).all()
    assert not args[9].any()
