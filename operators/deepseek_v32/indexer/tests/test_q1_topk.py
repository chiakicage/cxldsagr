"""Q1 dispatch boundaries and bitwise ordering across native stream/capture use."""

from contextlib import nullcontext

import pytest
import torch

from operators.deepseek_v32.indexer import q1_topk_cub, selection


@pytest.mark.parametrize(
    "rows,columns,k,cub",
    [
        (1, 32767, 2048, False),
        (1, 32768, 2048, True),
        (1, 32769, 2048, True),
        (1, 131072, 2048, True),
        (2, 32768, 2048, False),
        (1, 32768, 2047, False),
        (1, 2047, 2048, False),
    ],
)
def test_dispatch_preserves_geometry_abi_layout_and_output_ownership(
    monkeypatch, rows, columns, k, cub
):
    # CPU tensors exercise the host dispatcher with mocked CUDA metadata only.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    scores = torch.empty(rows, columns * 2)[:, ::2]
    count = min(k, columns)
    values = torch.empty(rows, count)
    indices = torch.empty(rows, count, dtype=torch.int32)
    calls = []

    def native(actual, selected, *, sorted_output=True, deterministic=True):
        assert actual.is_contiguous() and actual.shape == scores.shape
        assert selected == count
        assert (sorted_output, deterministic) == (not cub, not cub)
        calls.append("selection")
        return values, indices

    def postprocess(name):
        def invoke(actual_values, actual_indices):
            assert actual_values is values and actual_indices is indices
            calls.append(name)

        return invoke

    monkeypatch.setattr(selection, "_official_topk_int32", native)
    monkeypatch.setattr(selection, "_mask_nonfinite_indices_", postprocess("original-mask"))
    monkeypatch.setattr(q1_topk_cub, "sort_mask_", postprocess("cub-sort-mask"))
    actual_values, actual_indices = selection.exact_topk(scores, k)
    assert actual_values is values and actual_indices is indices
    assert calls == ["selection", "cub-sort-mask" if cub else "original-mask"]


def test_cub_failure_propagates_without_repeating_selection_or_mask(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    failure = RuntimeError("native launch failed")
    calls = []

    def native(*args, **kwargs):
        calls.append("selection")
        return torch.empty(1, 2048), torch.empty(1, 2048, dtype=torch.int32)

    def fail(*args):
        calls.append("sort")
        raise failure

    monkeypatch.setattr(selection, "_official_topk_int32", native)
    monkeypatch.setattr(q1_topk_cub, "sort_mask_", fail)
    monkeypatch.setattr(selection, "_mask_nonfinite_indices_", lambda *args: calls.append("mask"))
    with pytest.raises(RuntimeError) as caught:
        selection.exact_topk(torch.empty(1, 32768), 2048)
    assert caught.value is failure and calls == ["selection", "sort"]


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("DeepSeek Q1 selection requires Hopper SM90")


def reference(scores):
    backend = selection._topk_backend()
    values, indices = backend.top_k(
        scores.contiguous(),
        2048,
        sorted=True,
        deterministic=True,
        tie_break=backend.TopKTieBreak.SMALL,
    )
    return values, indices.int().masked_fill(~torch.isfinite(values), -1)


def assert_bits(actual, expected):
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left.view(torch.int32), right.view(torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("columns", [32767, 32768, 32769, 65536, 131072])
@pytest.mark.parametrize("pattern", ["ties", "signed_zero", "causal", "extreme"])
def test_cuda_q1_threshold_layout_and_value_bits(columns, pattern):
    require_sm90()
    storage = torch.empty(1, columns * 2 + 2, device="cuda")
    scores = storage[:, 1:-1:2]
    if pattern == "ties":
        scores.copy_(torch.arange(columns, device="cuda").remainder(19))
    elif pattern == "signed_zero":
        scores.zero_()
        scores.view(torch.int32)[:, ::2] = -2147483648
    elif pattern == "causal":
        scores.copy_(torch.randn_like(scores))
        scores[:, 7:] = -torch.inf
    else:
        values = torch.tensor(
            [
                torch.finfo(torch.float32).min,
                -1e30,
                -1e-38,
                -0.0,
                0.0,
                1e-38,
                1e30,
                torch.finfo(torch.float32).max,
                -torch.inf,
            ],
            device="cuda",
        )
        scores.copy_(values[torch.arange(columns, device="cuda").remainder(len(values))])
    before = scores.clone()
    assert_bits(selection.exact_topk(scores, 2048), reference(scores))
    torch.testing.assert_close(scores.view(torch.int32), before.view(torch.int32), rtol=0, atol=0)


def test_cuda_q1_changed_graph_inputs_on_nondefault_stream():
    require_sm90()
    scores = torch.randn(1, 32769, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            selection.exact_topk(scores, 2048)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = selection.exact_topk(scores, 2048)
        for pattern in ("random", "ties", "invalid", "short", "signed_zero"):
            if pattern == "random":
                scores.normal_()
            elif pattern == "invalid":
                scores.fill_(-torch.inf)
            else:
                scores.zero_()
                if pattern == "short":
                    scores[:, 7:] = -torch.inf
                elif pattern == "signed_zero":
                    scores.view(torch.int32)[:, ::2] = -2147483648
            graph.replay()
            assert_bits(output, reference(scores))
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
