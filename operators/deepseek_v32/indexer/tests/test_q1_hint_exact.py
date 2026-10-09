"""Bitwise hint specialization, dispatch boundaries and graph/stream consumers."""

import subprocess
import sys

import pytest
import torch

from operators.deepseek_v32.indexer import prefetch_hint, q1_hint_exact


def _hopper():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper GPU required")


def _bits(value):
    return value.contiguous().view(torch.uint8)


def _oracle(scores, offset):
    result = offset.clone()
    finite = torch.isfinite(scores[-4:])
    result[0] = scores[-4:].masked_fill(~finite, 0).sum() / finite.sum().clamp_min(1)
    return result


def test_import_and_cpu_reference_do_not_load_native_compilers():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys, torch; "
                "from operators.deepseek_v32.indexer import prefetch_hint; "
                "prefetch_hint.update_prefetch_hint(torch.ones(1, 65537), torch.zeros(16)); "
                "assert 'operators.deepseek_v32.indexer.q1_hint_exact' not in sys.modules; "
                "assert 'tvm_ffi' not in sys.modules; assert 'triton' not in sys.modules"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize("start", [0, 4])
@pytest.mark.parametrize(
    "pattern", ["mixed", "nonfinite", "zeros", "subnormal", "overflow", "cancel", "tree", "tail"]
)
@torch.inference_mode()
def test_exact_q1_padded_views_match_installed_torch(start, pattern, monkeypatch):
    _hopper()
    owner = torch.full((2, 65548), 123.0, device="cuda")
    scores = owner[1:2, start : start + 65537]
    patterns = {
        "mixed": [0x3F800000, 0xC0000000, 0x7FC01234, 0x7F800000, 0xFF800000, 0x80000000],
        "nonfinite": [0x7FC01234, 0x7F800000, 0xFF800000, 0x7F800001],
        "zeros": [0, 0x80000000],
        "subnormal": [1, 2, 0x7FFFFF, 0x800001, 0x80000001, 0x807FFFFF],
        "overflow": [0x7F7FFFFF],
        "cancel": [0x7F7FFFFF, 0x7F7FFFFF, 0xFF7FFFFF, 0xFF7FFFFF],
    }
    if pattern in patterns:
        values = torch.tensor(patterns[pattern], dtype=torch.int64, device="cuda")
        values = values.int().view(torch.float32)
        scores.copy_(values[torch.arange(65537, device="cuda") % len(values)][None])
    else:
        scores.zero_()
        if pattern == "tail":
            scores[0, -1] = -13.25
        else:
            positions = [
                0,
                1,
                3,
                4,
                31,
                32,
                127,
                128,
                511,
                512,
                1023,
                1024,
                2047,
                2048,
                32767,
                32768,
                65532,
                65533,
                65534,
                65535,
                65536,
            ]
            values = torch.tensor([1e20, -1e20, 1.0, -1.0, 1e-30], device="cuda")
            scores[0, positions] = values[torch.arange(len(positions), device="cuda") % 5]
    original = owner.clone()
    offset = torch.arange(16, device="cuda", dtype=torch.float32)
    expected = _oracle(scores, offset)
    calls = []
    native_update = q1_hint_exact.update

    def observed(*args):
        calls.append(True)
        return native_update(*args)

    monkeypatch.setattr(q1_hint_exact, "update", observed)
    prefetch_hint.update_prefetch_hint(scores, offset)
    assert calls == [True]
    assert torch.equal(_bits(offset), _bits(expected))
    assert torch.equal(_bits(owner), _bits(original))


@pytest.mark.parametrize("case", ["unaligned", "strided", "short", "long", "q2", "grad"])
@torch.inference_mode()
def test_unsupported_inputs_preserve_normal_dispatch(case, monkeypatch):
    _hopper()
    shape = {"short": (1, 65536), "long": (1, 65538), "q2": (2, 65537)}.get(case, (1, 65537))
    scores = torch.randn(shape, device="cuda")
    if case == "unaligned":
        scores = torch.randn(1, 65538, device="cuda")[:, 1:]
    elif case == "strided":
        scores = torch.randn(1, 131074, device="cuda")[:, ::2]
    offset = torch.arange(16, device="cuda", dtype=torch.float32)

    def forbidden(*args):
        raise AssertionError("Unsupported input reached native execution")

    monkeypatch.setattr(q1_hint_exact, "module", forbidden)
    with torch.set_grad_enabled(case == "grad"):
        expected = _oracle(scores, offset)
        prefetch_hint.update_prefetch_hint(scores, offset)
        assert not q1_hint_exact.supported(scores, offset)
        with pytest.raises(ValueError, match="unsupported"):
            q1_hint_exact.update(scores, offset)
    assert torch.equal(_bits(offset), _bits(expected))


@torch.inference_mode()
def test_native_failure_propagates_original_exception(monkeypatch):
    _hopper()
    scores, offset = torch.ones(1, 65537, device="cuda"), torch.zeros(16, device="cuda")
    expected = RuntimeError("native build failed")

    def fail():
        raise expected

    monkeypatch.setattr(q1_hint_exact, "module", fail)
    with pytest.raises(RuntimeError) as caught:
        prefetch_hint.update_prefetch_hint(scores, offset)
    assert caught.value is expected
    assert torch.equal(offset, torch.zeros_like(offset))


@torch.inference_mode()
def test_changed_graph_inputs_and_nondefault_stream_consumer():
    _hopper()
    scores = torch.randn(1, 65537, device="cuda")
    offset = torch.arange(16, dtype=torch.float32, device="cuda")
    prefetch_hint.update_prefetch_hint(scores, offset)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prefetch_hint.update_prefetch_hint(scores, offset)
        consumed = offset.clone()
    for factor in (-3.0, 0.0, 7.0):
        scores.copy_(torch.randn_like(scores) * factor)
        scores[:, ::17] = torch.nan
        offset.copy_(torch.arange(16, dtype=torch.float32, device="cuda") * factor)
        expected = _oracle(scores, offset)
        graph.replay()
        assert torch.equal(_bits(consumed), _bits(expected))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    changed = torch.randn_like(scores)
    expected = _oracle(changed, offset)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000)
        scores.copy_(changed)
        prefetch_hint.update_prefetch_hint(scores, offset)
        consumed = offset.clone()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(_bits(consumed), _bits(expected))
