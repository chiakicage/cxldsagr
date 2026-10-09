"""Production Q1 dispatch, causal selection, and graph replay on Hopper."""

import pytest
import torch

from operators.deepseek_v32.indexer import echo
from operators.deepseek_v32.indexer.selection import exact_topk
from operators.deepseek_v32.indexer.tests.test_echo_indexer import inputs, require_sm90


def nonpaged_reference(native, data, starts, ends, width):
    q, k, weights, scales = data
    if len(k) % 4:
        scales = torch.nn.functional.pad(scales, (0, 4 - len(k) % 4))
    result = native((q, None), (k, scales[: len(k)]), weights, starts, ends, max_seqlen_k=width)
    mask = torch.arange(width, device=q.device)[None, :] >= ends[:, None]
    return result.masked_fill(mask, -torch.inf)


def assert_scores_and_selection(actual, expected):
    torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32), atol=0, rtol=0)
    for left, right in zip(exact_topk(actual, 2048), exact_topk(expected, 2048), strict=True):
        if left.dtype == torch.float32:
            left, right = left.view(torch.int32), right.view(torch.int32)
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize("columns,query_start", [(32768, 32767), (32769, 32768), (32769, 32767)])
@pytest.mark.parametrize("pad_to_stride", [False, True])
@torch.inference_mode()
def test_cuda_paged_q1_production_matches_nonpaged_and_graph(
    columns, query_start, pad_to_stride, monkeypatch
):
    require_sm90()
    import deep_gemm

    data = inputs(1, columns)
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.tensor([query_start + 1], dtype=torch.int32, device="cuda")
    width = (columns + 255) // 256 * 256 if pad_to_stride else columns
    native = deep_gemm.fp8_fp4_mqa_logits
    expected = nonpaged_reference(native, data, starts, ends, width)
    original_paged = echo._paged_q1_logits
    calls = []

    def paged(*args):
        # The tail-page ABI cannot accept nonpaged's extra scale padding.
        assert args[3] is data[3] and args[3].shape == (columns,)
        calls.append(True)
        return original_paged(*args)

    def forbidden_nonpaged(*args, **kwargs):
        raise AssertionError("Eligible production Q1 fell through to nonpaged MQA")

    monkeypatch.setattr(echo, "_paged_q1_logits", paged)
    monkeypatch.setattr(deep_gemm, "fp8_fp4_mqa_logits", forbidden_nonpaged)

    def run():
        return echo.logits(*data, query_start, _bounds=(starts, ends), _pad_to_stride=pad_to_stride)

    actual = run()
    assert actual.shape == (1, width)
    assert actual.stride(0) == (columns + 255) // 256 * 256
    assert_scores_and_selection(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(4):
        graph.replay()
        torch.cuda.synchronize()
        assert_scores_and_selection(captured, expected)
    # Packing and scheduler generation must execute on replay with current inputs.
    data[1].zero_()
    data[3].fill_(0.75)
    if query_start < columns - 1:
        ends.fill_(query_start)
    graph.replay()
    changed = nonpaged_reference(native, data, starts, ends, width)
    torch.cuda.synchronize()
    assert_scores_and_selection(captured, changed)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "rows,columns,query_start", [(1, 32767, 32766), (2, 32769, 32767), (1, 32769, 1023)]
)
@torch.inference_mode()
def test_cuda_short_or_multiquery_still_calls_nonpaged(rows, columns, query_start, monkeypatch):
    require_sm90()
    import deep_gemm

    data = inputs(rows, columns)
    original = deep_gemm.fp8_fp4_mqa_logits
    calls = []

    def nonpaged(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(deep_gemm, "fp8_fp4_mqa_logits", nonpaged)
    monkeypatch.setattr(
        echo, "_paged_q1_logits", lambda *args: pytest.fail("Unsupported shape used paged MQA")
    )
    actual = echo.logits(*data, query_start)
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.arange(query_start + 1, query_start + rows + 1, dtype=torch.int32, device="cuda")
    expected = nonpaged_reference(original, data, starts, ends, columns)
    assert_scores_and_selection(actual, expected)
    assert len(calls) == 1
