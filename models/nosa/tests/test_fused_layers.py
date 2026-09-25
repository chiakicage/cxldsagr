"""Residual scheduling and FlashInfer math; CPU mocks only validate wiring."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from layers.feed_forward import SwiGLU
from layers.normalization import RMSNorm
from models.nosa.tests.test_model import initialized_model, reference_logits, tiny_config
from operators import flashinfer


def fp32_norm(x, weight, eps):
    values = x.float()
    return values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * weight.float()


def fp32_swiglu(layer, x):
    gate_weight, up_weight = layer.gate_up_proj.weight.float().chunk(2, dim=0)
    gate_bias, up_bias = (
        (None, None)
        if layer.gate_up_proj.bias is None
        else layer.gate_up_proj.bias.float().chunk(2, dim=0)
    )
    gate = F.linear(x.float(), gate_weight, gate_bias)
    up = F.linear(x.float(), up_weight, up_bias)
    down_bias = None if layer.down_proj.bias is None else layer.down_proj.bias.float()
    return F.linear(gate * gate.sigmoid() * up, layer.down_proj.weight.float(), down_bias)


@pytest.fixture
def cpu_fused_wiring(monkeypatch):
    """Emulate the kernels' mutation contract, without claiming CUDA validation."""
    calls = {"rmsnorm": [], "fused_add_rmsnorm": [], "silu_and_mul": []}

    def norm(x, weight, eps):
        calls["rmsnorm"].append(x)
        return fp32_norm(x, weight, eps).to(x.dtype)

    def add_norm(x, residual, weight, eps):
        assert x.data_ptr() != residual.data_ptr()
        calls["fused_add_rmsnorm"].append((x, residual))
        summed = x.float() + residual.float()
        x.copy_(fp32_norm(summed, weight, eps))
        residual.copy_(summed)
        return x, residual

    def activate(packed):
        assert packed.is_contiguous()
        calls["silu_and_mul"].append(packed)
        gate, up = packed.float().chunk(2, dim=-1)
        return (gate * gate.sigmoid() * up).to(packed.dtype)

    monkeypatch.setattr(flashinfer, "can_use_flashinfer", lambda x: True)
    monkeypatch.setattr(flashinfer, "rmsnorm", norm)
    monkeypatch.setattr(flashinfer, "fused_add_rmsnorm", add_norm)
    monkeypatch.setattr(flashinfer, "silu_and_mul", activate)
    return calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_cpu_rmsnorm_preserves_inputs_and_eager_rounding(dtype):
    generator = torch.Generator().manual_seed(20)
    x = torch.randn(7, 32, generator=generator).to(dtype)
    residual = torch.randn(7, 32, generator=generator).to(dtype)
    x_before, residual_before = x.clone(), residual.clone()
    layer = RMSNorm(32, 1e-6, device="cpu", dtype=dtype)
    layer.weight.copy_(torch.randn(32, generator=generator) * 0.2 + 1)
    assert not flashinfer.can_use_flashinfer(x)

    def eager_norm(values):
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(-1, keepdim=True) + layer.eps
        )
        return normalized.to(dtype) * layer.weight

    torch.testing.assert_close(layer(x), eager_norm(x), rtol=0, atol=0)
    normalized, summed = layer(x, residual)
    torch.testing.assert_close(summed, x_before + residual_before, rtol=0, atol=0)
    torch.testing.assert_close(normalized, eager_norm(summed), rtol=0, atol=0)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)


@torch.inference_mode()
def test_cpu_mock_fused_norm_returns_the_mutated_buffers(cpu_fused_wiring):
    x = torch.linspace(-3, 2, 96).reshape(3, 32)
    residual = torch.linspace(1, -2, 96).reshape(3, 32)
    layer = RMSNorm(32, 1e-6, device="cpu", dtype=torch.float32)
    layer.weight.copy_(torch.linspace(0.5, 1.5, 32))
    expected_sum = x + residual
    expected_norm = fp32_norm(expected_sum, layer.weight, layer.eps)
    actual, actual_residual = layer(x, residual)
    assert actual is x and actual_residual is residual
    torch.testing.assert_close(actual, expected_norm)
    torch.testing.assert_close(actual_residual, expected_sum)


@pytest.mark.parametrize(
    "layout",
    ["rank1", "noncontiguous", "residual3d", "input_offset", "residual_offset", "weight_offset"],
)
@torch.inference_mode()
def test_cpu_mock_norm_unsupported_layouts_use_eager(cpu_fused_wiring, layout):
    torch.manual_seed(25)
    x = torch.randn(4, 16)
    residual = None
    layer = RMSNorm(16, 1e-6, device="cpu", dtype=torch.float32)
    if layout == "rank1":
        x = x[0]
    elif layout == "noncontiguous":
        x = torch.randn(16, 4).T
    elif layout == "residual3d":
        x = x.reshape(2, 2, 16)
        residual = torch.randn_like(x)
    elif layout == "input_offset":
        x = torch.randn(65)[1:].reshape(4, 16)
    elif layout == "residual_offset":
        residual = torch.randn(65)[1:].reshape(4, 16)
    else:
        layer.weight = nn.Parameter(torch.randn(17)[1:])
    before = x.clone()
    residual_before = None if residual is None else residual.clone()
    summed = x if residual is None else x + residual
    expected = fp32_norm(summed, layer.weight, layer.eps)
    actual = layer(x, residual)
    if residual is not None:
        actual, actual_sum = actual
        torch.testing.assert_close(actual_sum, summed)
        torch.testing.assert_close(residual, residual_before, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, before, atol=0, rtol=0)
    assert not cpu_fused_wiring["rmsnorm"]
    assert not cpu_fused_wiring["fused_add_rmsnorm"]


@torch.inference_mode()
def test_cpu_mock_swiglu_unaligned_projection_width_uses_eager(cpu_fused_wiring):
    torch.manual_seed(26)
    layer = SwiGLU(16, 23, bias=True, device="cpu", dtype=torch.float32)
    x = torch.randn(4, 16)
    torch.testing.assert_close(layer(x), fp32_swiglu(layer, x), rtol=2e-5, atol=2e-6)
    assert not cpu_fused_wiring["silu_and_mul"]


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shape", [(1, 16), (7, 16), (2, 3, 16)])
@torch.inference_mode()
def test_cpu_mock_packed_swiglu_preserves_parameters_and_observes_updates(
    cpu_fused_wiring, bias, shape
):
    torch.manual_seed(21)
    layer = SwiGLU(16, 24, bias=bias, device="cpu", dtype=torch.float32)
    x = torch.randn(shape)
    parameters = dict(layer.named_parameters())
    expected_keys = {f"{name}.weight" for name in ("gate_up_proj", "down_proj")}
    if bias:
        expected_keys |= {f"{name}.bias" for name in ("gate_up_proj", "down_proj")}
    assert set(layer.state_dict()) == expected_keys
    assert all(
        isinstance(getattr(layer, name), nn.Linear) for name in ("gate_up_proj", "down_proj")
    )
    projections = []
    handle = layer.gate_up_proj.register_forward_hook(
        lambda module, args, output: projections.append(output)
    )
    try:
        for invocation, update in enumerate((False, True), start=1):
            if update:
                # Mutation after first use catches stale copies of packed weights/biases.
                for parameter in layer.parameters():
                    parameter.add_(0.1)
            actual = layer(x)
            torch.testing.assert_close(actual, fp32_swiglu(layer, x), rtol=2e-5, atol=2e-6)
            packed = cpu_fused_wiring["silu_and_mul"][-1]
            assert len(projections) == invocation
            assert packed.data_ptr() == projections[-1].data_ptr()
            assert (
                packed.untyped_storage().data_ptr() == projections[-1].untyped_storage().data_ptr()
            )
            gate_weight, up_weight = layer.gate_up_proj.weight.chunk(2, dim=0)
            gate_bias, up_bias = (
                (None, None)
                if layer.gate_up_proj.bias is None
                else layer.gate_up_proj.bias.chunk(2, dim=0)
            )
            for part, weight, projection_bias in zip(
                packed.chunk(2, -1), (gate_weight, up_weight), (gate_bias, up_bias), strict=True
            ):
                torch.testing.assert_close(
                    part.reshape(-1, 24), F.linear(x.reshape(-1, 16), weight, projection_bias)
                )
            assert set(layer.state_dict()) == expected_keys
            assert all(layer.get_parameter(name) is value for name, value in parameters.items())
    finally:
        handle.remove()


@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_cpu_mock_decoder_residual_boundaries_slicing_and_cache(cpu_fused_wiring, bias):
    config = tiny_config(attention_bias=bias, mlp_bias=bias)
    model = initialized_model(config)
    tokens = torch.tensor([1, 7, 3, 8, 2, 9, 5, 4, 6])
    expected = reference_logits(config, model.state_dict(), tokens)
    boundary = {}

    def last_layer_output(module, args, output):
        boundary["layer_output"] = output

    def final_norm_input(module, args):
        boundary["norm_input"] = args

    handles = [
        model.model.layers[-1].register_forward_hook(last_layer_output),
        model.model.norm.register_forward_pre_hook(final_norm_input),
    ]
    try:
        actual = model(tokens, logits_to_keep=3)
        torch.testing.assert_close(actual, expected[-3:], rtol=3e-5, atol=3e-6)
        pending, residual = boundary["layer_output"]
        final_pending, final_residual = boundary["norm_input"]
        assert final_pending.shape == final_residual.shape == (3, config.hidden_size)
        assert final_pending.data_ptr() == pending[-3:].data_ptr()
        assert final_residual.data_ptr() == residual[-3:].data_ptr()
        assert len(cpu_fused_wiring["rmsnorm"]) == 1
        assert len(cpu_fused_wiring["fused_add_rmsnorm"]) == 2 * config.num_hidden_layers
        assert len(cpu_fused_wiring["silu_and_mul"]) == config.num_hidden_layers
    finally:
        for handle in handles:
            handle.remove()

    hidden = model(tokens, return_hidden=True)
    torch.testing.assert_close(model.lm_head(hidden).float(), expected, rtol=3e-5, atol=3e-6)
    for return_hidden in (False, True):
        cache = model.new_cache(len(tokens))
        chunks = []
        start = 0
        for size in (3, 4, 1, 1):
            chunks.append(model(tokens[start : start + size], cache, return_hidden=return_hidden))
            start += size
            assert cache.length == start
        reference = hidden if return_hidden else expected
        torch.testing.assert_close(torch.cat(chunks), reference, rtol=3e-5, atol=3e-6)
        model.cache_manager.release(cache)


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for actual FlashInfer kernels"
)


@pytest.fixture(scope="module")
def cuda_kernels(record_testsuite_property):
    backend = pytest.importorskip("flashinfer")
    gpu = torch.cuda.get_device_properties(0)
    for name, value in {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "flashinfer_version": backend.__version__,
        "gpu": gpu.name,
        "compute_capability": f"{gpu.major}.{gpu.minor}",
        "gpu_memory_bytes": gpu.total_memory,
    }.items():
        record_testsuite_property(name, str(value))


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 17])
@torch.inference_mode()
def test_cuda_norms_match_fp32_reference_and_preserve_alias_contract(cuda_kernels, dtype, tokens):
    generator = torch.Generator(device="cuda").manual_seed(22)
    x = torch.randn(tokens, 4096, device="cuda", generator=generator).to(dtype)
    residual = torch.randn(tokens, 4096, device="cuda", generator=generator).to(dtype)
    layer = RMSNorm(4096, 1e-6, device="cuda", dtype=dtype)
    layer.weight.copy_(torch.randn(4096, device="cuda", generator=generator) * 0.2 + 1)
    assert flashinfer.can_use_flashinfer(x)
    with torch.enable_grad():
        assert not flashinfer.can_use_flashinfer(x)
    assert not flashinfer.can_use_flashinfer(x.float())
    tolerance = 0.008 if dtype == torch.bfloat16 else 0.001
    x_before, residual_before = x.clone(), residual.clone()
    torch.testing.assert_close(
        layer(x).float(),
        fp32_norm(x, layer.weight, layer.eps),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    summed = x_before.float() + residual_before.float()
    normalized, updated_residual = layer(x, residual)
    assert normalized is x and updated_residual is residual
    torch.testing.assert_close(updated_residual, summed.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(
        normalized.float(),
        fp32_norm(summed, layer.weight, layer.eps),
        atol=tolerance,
        rtol=tolerance,
    )


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 17])
@torch.inference_mode()
def test_cuda_silu_and_mul_nosa_width_matches_fp32_reference(cuda_kernels, dtype, tokens):
    generator = torch.Generator(device="cuda").manual_seed(23)
    # Actual NOSA activation width, without allocating three 4096 x 16384 weights.
    packed = torch.randn(tokens, 2 * 16384, device="cuda", generator=generator).to(dtype)
    before = packed.clone()
    gate, up = packed.float().chunk(2, dim=-1)
    expected = gate * gate.sigmoid() * up
    actual = flashinfer.silu_and_mul(packed)
    assert actual.shape == (tokens, 16384) and actual.dtype == dtype
    tolerance = 0.008 if dtype == torch.bfloat16 else 0.001
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(packed, before, rtol=0, atol=0)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("tokens", [1, 9])
@torch.inference_mode()
def test_cuda_packed_swiglu_matches_fp32_reference(cuda_kernels, dtype, bias, tokens):
    torch.manual_seed(24)
    layer = SwiGLU(256, 384, bias=bias, device="cuda", dtype=dtype)
    x = torch.randn(tokens, 256, device="cuda", dtype=dtype)
    expected = fp32_swiglu(layer, x)
    actual = layer(x)
    atol, rtol = (0.004, 0.02) if dtype == torch.bfloat16 else (0.0005, 0.003)
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_cuda_model_fused_matches_rounding_reference_and_dense_math(
    cuda_kernels, monkeypatch, dtype
):
    config = tiny_config(hidden_size=256, head_dim=64, intermediate_size=384, mlp_bias=True)
    # Both sides share the explicit dense attention reference to isolate ordinary layers.
    model = initialized_model(config, device="cuda", dtype=dtype)
    tokens = torch.tensor([1, 7, 3, 8, 2, 9, 5, 4, 6], device="cuda")

    def norm_reference(x, weight, eps):
        return fp32_norm(x, weight, eps).to(x.dtype)

    def add_norm_reference(x, residual, weight, eps):
        # FlashInfer normalizes the FP32 sum before rounding the residual store.
        summed = x.float() + residual.float()
        x.copy_(fp32_norm(summed, weight, eps))
        residual.copy_(summed)
        return x, residual

    def activation_reference(packed):
        gate, up = packed.float().chunk(2, dim=-1)
        return (gate * gate.sigmoid() * up).to(packed.dtype)

    with monkeypatch.context() as patch:
        patch.setattr(flashinfer, "rmsnorm", norm_reference)
        patch.setattr(flashinfer, "fused_add_rmsnorm", add_norm_reference)
        patch.setattr(flashinfer, "silu_and_mul", activation_reference)
        expected_hidden = model(tokens, return_hidden=True)
        expected_logits = model(tokens)
    with monkeypatch.context() as patch:
        patch.setattr(flashinfer, "can_use_flashinfer", lambda x: False)
        eager_logits = model(tokens)

    fused_logits = model(tokens)
    dense_logits = reference_logits(
        config, {name: value.cpu() for name, value in model.state_dict().items()}, tokens.cpu()
    )
    # Eager rounds after more operations. Check both implementations against
    # independent dense math without treating their different rounding as bugs.
    for output in (eager_logits, fused_logits):
        difference = output.cpu().double() - dense_logits.double()
        nrmse = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            dense_logits.double()
        )
        assert nrmse <= 4 * torch.finfo(dtype).eps

    atol, rtol = (0.06, 0.035) if dtype == torch.bfloat16 else (0.008, 0.005)
    torch.testing.assert_close(fused_logits, expected_logits, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        model(tokens, logits_to_keep=2), expected_logits[-2:], atol=atol, rtol=rtol
    )
    for return_hidden in (False, True):
        expected = expected_hidden if return_hidden else expected_logits
        torch.testing.assert_close(
            model(tokens, return_hidden=return_hidden), expected, atol=atol, rtol=rtol
        )
        cache = model.new_cache(len(tokens))
        chunks = []
        start = 0
        for size in (3, 4, 1, 1):
            chunks.append(model(tokens[start : start + size], cache, return_hidden=return_hidden))
            start += size
            assert cache.length == start
        torch.testing.assert_close(torch.cat(chunks), expected, atol=atol, rtol=rtol)
        model.cache_manager.release(cache)
