"""Integrated norm dispatch, exact vendor agreement and output lifetime."""

import os
import subprocess
import sys

import pytest
import torch

from models.deepseek_v32 import nonmatrix
from operators.deepseek_v32.norm import api


def _require_hopper():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; explicit GPU regression checks availability first")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("Typed norm acceptance requires SM90/Hopper")


def _bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    integer = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.contiguous().view(integer), expected.contiguous().view(integer))


def _fixture(rows, width, strided):
    pitch = width + 64 if strided else width
    xowner = torch.full((rows, pitch), 7, device="cuda", dtype=torch.bfloat16)
    rowner = torch.full_like(xowner, -11)
    x, residual = xowner[:, :width], rowner[:, :width]
    x.copy_(torch.randn_like(x))
    residual.copy_(torch.randn_like(residual))
    weight = torch.linspace(0.73, 1.31, width, device="cuda", dtype=torch.float32)
    return x, residual, weight, xowner, rowner


def _oracle(x, residual, weight, fused):
    from flashinfer import norm

    widened = x.to(dtype=torch.float32, copy=True).contiguous()
    if not fused:
        return (norm.rmsnorm(widened, weight, eps=1e-6).to(x.dtype),)
    saved = residual.to(dtype=torch.float32, copy=True).contiguous()
    norm.fused_add_rmsnorm(widened, saved, weight, eps=1e-6)
    return widened.to(x.dtype), saved.to(x.dtype)


def _invoke(x, residual, weight, fused):
    if fused:
        return nonmatrix.residual_rms_norm(x, residual, weight, 1e-6)
    return (nonmatrix.rms_norm(x, weight, 1e-6),)


def test_cpu_import_and_autograd_do_not_load_cuda_norm_tooling():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
from models.deepseek_v32.nonmatrix import rms_norm, residual_rms_norm
from operators.deepseek_v32.norm.api import can_use_typed_norm
x = torch.randn(2, 512, requires_grad=True)
r = torch.randn_like(x, requires_grad=True)
w = torch.ones(512, requires_grad=True)
assert not can_use_typed_norm(x, w)
y, s = residual_rms_norm(x, r, w, 1e-6)
(y.square().sum() + s.square().sum() + rms_norm(x, w, 1e-6).sum()).backward()
assert x.grad is not None and r.grad is not None and w.grad is not None
assert not torch.cuda.is_initialized()
assert not any(n == 'cutlass' or n.startswith('cutlass.') for n in sys.modules)
assert not any(n == 'flashinfer' or n.startswith('flashinfer.') for n in sys.modules)
""",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fused,width", [(False, 512), (False, 1536), (False, 7168), (True, 7168)])
@pytest.mark.parametrize("rows", [1, 9, 121, 128, 129, 1024])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_cuda_typed_norm_exact_vendor_output_and_full_input_storage(fused, width, rows, strided):
    _require_hopper()
    x, residual, weight, xowner, rowner = _fixture(rows, width, strided)
    guards = tuple(t.clone() for t in (xowner, rowner, weight))
    assert api.can_use_typed_norm(x, weight, residual if fused else None)
    expected = _oracle(x, residual, weight, fused)
    first = _invoke(x, residual, weight, fused)
    second = _invoke(x * 0.75, residual * -0.5, weight, fused)
    for value, reference in zip(first, expected, strict=True):
        _bits(value, reference)
    pointers = [value.untyped_storage().data_ptr() for value in (*first, *second)]
    assert len(pointers) == len(set(pointers))
    assert not set(pointers).intersection(
        t.untyped_storage().data_ptr() for t in (xowner, rowner, weight)
    )
    for value, guard in zip((xowner, rowner, weight), guards, strict=True):
        _bits(value, guard)


@pytest.mark.parametrize("fused,width", [(False, 512), (True, 7168)])
@torch.inference_mode()
def test_cuda_typed_norm_nondefault_stream_and_changed_graph_inputs(fused, width):
    _require_hopper()
    x, residual, weight, _, _ = _fixture(128, width, True)
    expected = _oracle(x, residual, weight, fused)
    main, side = torch.cuda.current_stream(), torch.cuda.Stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        first = _invoke(x, residual, weight, fused)
        consumed = tuple(value.clone() for value in first)
        for _ in range(3):
            _invoke(x, residual, weight, fused)
    main.wait_stream(side)
    for actual, reference in zip((*first, *consumed), (*expected, *expected), strict=True):
        _bits(actual, reference)
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        captured = _invoke(x, residual, weight, fused)
    side.synchronize()
    for factor in (0.5, -1.25, 0.75):
        x.mul_(factor)
        residual.add_(0.125)
        expected = _oracle(x, residual, weight, fused)
        graph.replay()
        retained = tuple(value.clone() for value in captured)
        graph.replay()
        for actual, reference in zip(retained, expected, strict=True):
            _bits(actual, reference)


@torch.inference_mode()
def test_cuda_unmeasured_layout_and_fp32_keep_ordinary_path(monkeypatch):
    _require_hopper()

    def unexpected(*args):
        pytest.fail("Unsupported input reached the typed norm execution")

    monkeypatch.setattr(api, "rms_norm", unexpected)
    monkeypatch.setattr(api, "residual_rms_norm", unexpected)
    weight = torch.linspace(0.73, 1.31, 7168, device="cuda")
    cases = [
        torch.randn(9, 7168, device="cuda"),
        torch.randn(9, 7168 * 2, device="cuda", dtype=torch.bfloat16)[:, ::2],
        torch.randn(9 * 7168 + 8, device="cuda", dtype=torch.bfloat16)[4:-4].view(9, 7168),
        torch.randn(9, 7168 + 128, device="cuda", dtype=torch.bfloat16)[:, :7168],
    ]
    for x in cases:
        residual = x.clone()
        assert not api.can_use_typed_norm(x, weight)
        for fused in (False, True):
            actual = _invoke(x, residual, weight, fused)
            expected = _oracle(x, residual, weight, fused)
            for value, reference in zip(actual, expected, strict=True):
                _bits(value, reference)
