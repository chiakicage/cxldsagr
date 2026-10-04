"""Exact ECHO hint values, score ownership and stream ordering."""

import pytest
import torch

from operators.deepseek_v32.indexer.prefetch_hint import update_prefetch_hint


def _device(name):
    if name == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0)
    ):
        pytest.skip("Hopper GPU is required")
    return name


def _bytes(tensor):
    return tensor.contiguous().view(torch.uint8)


def _expected(scores, offset):
    result = offset.clone()
    tail = scores[-4:]
    finite = torch.isfinite(tail)
    result[0] = tail.masked_fill(~finite, 0).sum() / finite.sum().clamp_min(1)
    return result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(1, 1), (3, 121), (4, 2049), (128, 65664)])
@pytest.mark.parametrize("pattern", ["causal", "nonfinite", "subnormal", "overflow", "zeros"])
@torch.inference_mode()
def test_hint_matches_checked_bits_without_mutating_scores(device, shape, pattern):
    device = _device(device)
    q, n = shape
    source = torch.randn(q + 1, n + 7, device=device)
    scores = source[1:, 1 : n + 1]
    if pattern == "causal":
        scores.masked_fill_(
            torch.arange(n, device=device)[None, :]
            >= torch.arange(n - q + 1, n + 1, device=device)[:, None],
            -torch.inf,
        )
    elif pattern == "nonfinite":
        scores[:, ::7] = torch.nan
        scores[:, ::11] = torch.inf
        scores[:, ::13] = -torch.inf
    elif pattern == "subnormal":
        bits = (
            torch.tensor(
                [1, 2, 0x7FFFFF, 0x800001, 0x80000001, 0x807FFFFF], device=device, dtype=torch.int64
            )
            .int()
            .view(torch.float32)
        )
        scores.copy_(bits[torch.arange(n, device=device) % len(bits)][None, :])
    elif pattern == "overflow":
        scores.fill_(torch.finfo(torch.float32).max)
        scores[:, ::2].neg_()
    else:
        scores.zero_()
        scores[:, ::2] = -0.0
    original = source.clone()
    offset = torch.arange(16, dtype=torch.float32, device=device)
    expected = _expected(scores, offset)
    update_prefetch_hint(scores, offset)
    assert torch.equal(_bytes(offset), _bytes(expected))
    assert torch.equal(_bytes(source), _bytes(original))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_nonunit_inner_stride_retains_checked_path(device, monkeypatch):
    device = _device(device)
    import operators.deepseek_v32.indexer.prefetch_hint as module

    def forbidden():
        raise AssertionError("unsupported stride must use checked expression")

    monkeypatch.setattr(module, "_kernels", forbidden)
    scores = torch.randn(4, 34, device=device)[:, ::2]
    offset = torch.arange(16, dtype=torch.float32, device=device)
    expected = _expected(scores, offset)
    update_prefetch_hint(scores, offset)
    assert torch.equal(_bytes(offset), _bytes(expected))


@torch.inference_mode()
def test_hint_nondefault_stream_orders_predecessor_and_consumer():
    device = _device("cuda")
    scores = torch.empty(128, 65664, device=device)
    reference = torch.randn_like(scores)
    reference[:, ::11] = -torch.inf
    offset = torch.arange(16, dtype=torch.float32, device=device)
    expected = _expected(reference, offset)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000)
        scores.copy_(reference)
        update_prefetch_hint(scores, offset)
        consumed = offset.clone()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(_bytes(consumed), _bytes(expected))
