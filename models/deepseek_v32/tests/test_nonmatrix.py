"""Numerical and ownership checks for DeepSeek's ordinary-operation adapters."""

import pytest
import torch
from torch.nn import functional as F

from models.deepseek_v32.nonmatrix import residual_rms_norm, rms_norm, silu_mul, silu_mul_packed


def reference_norm(x, weight, eps=1e-6):
    value = x.float()
    inv_rms = (value.square().mean(-1, keepdim=True) + eps).rsqrt()
    return (value * inv_rms * weight.float()).to(x.dtype)


def require_hopper():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; explicit GPU regression requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("DeepSeek FlashInfer validation requires SM90/Hopper")


def assert_norm_close(actual, expected):
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)
    assert error < 5e-5
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    else:
        torch.testing.assert_close(actual, expected, rtol=8e-3, atol=0.015625)


def test_cpu_preserves_fp32_weights_unrounded_sum_and_inputs():
    torch.manual_seed(738)
    x = torch.randn(3, 129).bfloat16()
    residual = torch.randn_like(x)
    weight = torch.linspace(0.73, 1.31, 129)
    expected_sum = x.float() + residual.float()
    x_before, residual_before = x.clone(), residual.clone()
    actual, saved = residual_rms_norm(x, residual, weight, 1e-6)
    torch.testing.assert_close(
        actual, reference_norm(expected_sum, weight).bfloat16(), rtol=0, atol=0
    )
    torch.testing.assert_close(saved, expected_sum.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close((x, residual), (x_before, residual_before), rtol=0, atol=0)
    direct, same_input = residual_rms_norm(x, None, weight, 1e-6)
    assert same_input is x
    torch.testing.assert_close(direct, reference_norm(x, weight), rtol=0, atol=0)


@pytest.mark.parametrize("rows,width", [(1, 512), (129, 1536), (1024, 7168)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_cuda_norm_with_true_fp32_weights_and_strided_rows(rows, width, dtype):
    require_hopper()
    torch.manual_seed(641)
    x = torch.randn(rows * 2, width, device="cuda", dtype=torch.bfloat16)[::2].to(dtype)
    x[0] = 0
    weight = torch.linspace(0.73, 1.31, width, device="cuda")
    before = x.clone()
    actual = rms_norm(x, weight, 1e-6)
    assert_norm_close(actual, reference_norm(x, weight))
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_cuda_fused_residual_norm_uses_unrounded_sum_and_owns_buffers(dtype):
    require_hopper()
    torch.manual_seed(153)
    x = torch.randn(129, 7168, device="cuda", dtype=torch.bfloat16).to(dtype)
    residual = torch.randn_like(x)
    weight = torch.linspace(0.73, 1.31, x.shape[-1], device="cuda")
    before = x.clone(), residual.clone()
    summed = x.float() + residual.float()
    expected = reference_norm(summed, weight).to(dtype)
    actual, saved = residual_rms_norm(x, residual, weight, 1e-6)
    assert_norm_close(actual, expected)
    torch.testing.assert_close(saved, summed.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close((x, residual), before, rtol=0, atol=0)
    if dtype == torch.bfloat16:
        prematurely_rounded = reference_norm(summed.bfloat16(), weight)
        wrong_error = (
            prematurely_rounded.float() - expected.float()
        ).norm() / expected.float().norm()
        assert wrong_error > 1e-3


@pytest.mark.parametrize("shape", [(5, 2048), (128, 18432), (17, 8, 2048)])
@torch.inference_mode()
def test_cuda_silu_mul_keeps_fp32_intermediate_and_route_dimensions(shape):
    require_hopper()
    torch.manual_seed(722)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 3
    up = torch.randn_like(gate)
    before = gate.clone(), up.clone()
    expected = (F.silu(gate.float()) * up.float()).bfloat16()
    actual = silu_mul(gate, up)
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=1e-5)
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error < 1e-5
    torch.testing.assert_close((gate, up), before, rtol=0, atol=0)
    assert torch.count_nonzero(F.silu(gate) * up != expected) > 0


def test_cpu_silu_mul_and_empty_shapes():
    gate = torch.tensor([[-7.0, -0.3, 0.2, 2.0, 8.0]]).bfloat16()
    up = torch.tensor([[3.0, 7.0, 0.2, -0.9, 1.7]]).bfloat16()
    torch.testing.assert_close(
        silu_mul(gate, up), (F.silu(gate.float()) * up.float()).bfloat16(), rtol=0, atol=0
    )
    x = torch.empty(0, 128, dtype=torch.bfloat16)
    weight = torch.ones(128)
    assert rms_norm(x, weight, 1e-6).shape == x.shape
    assert residual_rms_norm(x, x, weight, 1e-6)[0].shape == x.shape
    assert silu_mul(x, x).shape == x.shape


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cpu_packed_silu_matches_reference_and_preserves_autograd(dtype):
    packed = torch.linspace(-4, 4, 96).reshape(3, 32).to(dtype).requires_grad_()
    original = packed.detach().clone()
    gate, up = packed.split(16, dim=-1)
    expected = (F.silu(gate.float()) * up.float()).bfloat16()
    actual = silu_mul_packed(packed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    assert packed.grad is not None
    torch.testing.assert_close(packed.detach(), original, rtol=0, atol=0)
    assert actual.untyped_storage().data_ptr() != packed.untyped_storage().data_ptr()
    assert silu_mul_packed(torch.empty(0, 32, dtype=dtype)).shape == (0, 16)


def test_packed_silu_rejects_malformed_width():
    for value in (torch.ones(()), torch.empty(3, 0), torch.ones(3, 31)):
        with pytest.raises(ValueError, match="positive even"):
            silu_mul_packed(value)


@torch.inference_mode()
def test_cuda_packed_silu_matches_original_api_and_owns_return():
    require_hopper()
    packed = torch.randn(121, 36864, device="cuda", dtype=torch.bfloat16)
    original = packed.clone()
    expected = silu_mul(*packed.split(18432, dim=-1))
    actual = silu_mul_packed(packed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
    packed.zero_()
    later = silu_mul_packed(packed)
    assert actual.untyped_storage().data_ptr() not in (
        packed.untyped_storage().data_ptr(),
        later.untyped_storage().data_ptr(),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_cuda_fp16_silu_does_not_insert_an_extra_fp16_output_rounding():
    require_hopper()
    torch.manual_seed(131)
    gate = torch.randn(5, 2048, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    expected = (F.silu(gate.float()) * up.float()).bfloat16()
    torch.testing.assert_close(silu_mul(gate, up), expected, rtol=0, atol=0)
