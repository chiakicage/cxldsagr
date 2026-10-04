"""CPU contract checks against the unchanged ordinary score dispatcher."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from operators.nosa.indexer import _indexer_deferred_cuda as deferred
from operators.nosa.indexer import _scores_cuda, api


@pytest.mark.parametrize("rows", [128, 1024])
@pytest.mark.parametrize("count", [510, 511, 512, 2046, 2047, 2048])
def test_guarded_dispatch_matches_ordinary_launch_arguments(monkeypatch, rows, count):
    import tvm_ffi

    query = torch.empty((rows, 2, 16, 128), dtype=torch.bfloat16)
    keys = torch.empty((count, 2, 128), dtype=query.dtype)
    blocks = ((count + 1) * 16 + 63) // 64
    output = torch.empty((rows * 2, blocks), dtype=query.dtype)
    normalizers = torch.empty((1, rows, 2, 16, 2))
    finite = torch.empty((), dtype=torch.bool)
    ordinary, guarded = [], []
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    monkeypatch.setattr(api._scores, "run", lambda *a, **kw: ordinary.append(("triton", a, kw)))
    monkeypatch.setattr(
        _scores_cuda, "scores_out", lambda *a, **kw: ordinary.append(("native", a, kw))
    )
    monkeypatch.setattr(
        deferred._guarded_scores, "run", lambda *a, **kw: guarded.append(("triton", a, kw))
    )
    module = SimpleNamespace(scores_guarded_out=lambda *a, **kw: guarded.append(("native", a, kw)))
    monkeypatch.setattr(deferred, "load_module", lambda name: module)
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(tvm_ffi, "use_torch_stream", nullcontext)
    start = (count + 1) * 16 - rows
    api._launch_scores(query, keys, None, output, start, blocks, pool_output=True)
    deferred._launch_guarded_scores(query, keys, output, normalizers, finite, start, blocks)
    assert len(ordinary) == len(guarded) == 1
    assert guarded[0][0] == ordinary[0][0]
    if ordinary[0][0] == "triton":
        expected, actual = ordinary[0][1], guarded[0][1][1:]
        for left, right in zip(actual, expected, strict=True):
            assert left is right if isinstance(left, torch.Tensor) else left == right
        assert guarded[0][2] == ordinary[0][2]
    else:
        assert guarded[0][1] == (
            query,
            keys,
            query,
            output,
            normalizers,
            finite,
            start,
            blocks,
            True,
            True,
        )


def test_alias_check_covers_unused_derived_capacity_and_strided_input_gaps():
    storage = torch.empty((64, 2, 128), dtype=torch.bfloat16)
    ranking_in_unused_tail = storage[32:].view(torch.uint8).flatten()[:512].view(torch.int32)
    with pytest.raises(ValueError, match="must not overlap"):
        deferred._validate_disjoint(
            (), (("full compressed capacity", storage), ("ranking", ranking_in_unused_tail))
        )
    input_storage = torch.empty((8, 36, 128), dtype=torch.bfloat16)
    query = input_storage[:, :32]
    gap = input_storage[0, 32:]
    with pytest.raises(ValueError, match="must not overlap"):
        deferred._validate_disjoint((("strided Q", query),), (("output in Q span", gap),))


def test_disjoint_views_of_one_slab_and_empty_spans_are_allowed():
    slab = torch.empty(4096, dtype=torch.uint8)
    deferred._validate_disjoint(
        (("input", slab[:512]),),
        (("work", slab[512:1024]), ("partial", slab[1024:]), ("empty", slab[:0])),
    )
