"""Q1 scale-layout exactness, graph freshness, and prepared-activation lifetime."""

import pytest
import torch

from operators.deepseek_v32.linear import fp8, quantization
from operators.deepseek_v32.linear.tests.test_deepseek_linear import require_sm90
from operators.deepseek_v32.linear.tests.test_linear_quantization import _edge_values


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(
        a.clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8),
        b.clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8),
    )


def row_major_linear(x, weight, scales):
    data, activation_scales = fp8.quantize_fp8_activation(x)
    result = torch.empty((len(x), len(weight)), device=x.device, dtype=torch.bfloat16)
    fp8._deep_gemm().fp8_gemm_nt(
        (data, activation_scales), (weight, scales), result, recipe=(1, 128, 128)
    )
    return result


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("columns", (129, 7168))
@torch.inference_mode()
def test_cuda_q1_scale_bits_and_official_no_copy_layout(dtype, columns):
    require_sm90()
    owner = torch.zeros((1, columns * 2 + 16), device="cuda", dtype=dtype)
    x = owner[:, 8 : 8 + columns * 2 : 2]
    x.copy_(_edge_values(dtype, 1, columns))
    saved = owner.clone()
    expected = fp8.quantize_fp8_activation(x)
    actual = quantization.quantize(x, consumer_scale_layout=True)
    for a, b in zip(actual, expected, strict=True):
        exact(a, b)
    assert expected[1].is_contiguous()
    assert actual[1].stride() == (1, 4)
    official_view = fp8._deep_gemm().get_mn_major_tma_aligned_tensor(actual[1])
    assert official_view.data_ptr() == actual[1].data_ptr()
    assert official_view.stride() == (1, 4)
    exact(owner, saved)


@pytest.mark.parametrize("k,n", sorted(fp8._Q1_TMA_SCALE_SHAPES))
@torch.inference_mode()
def test_cuda_q1_complete_linear_matches_original_and_changed_graph(k, n, monkeypatch):
    require_sm90()
    torch.manual_seed(k + n)
    x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scales = torch.full(((n + 127) // 128, k // 128), 0.125, device="cuda")
    expected = row_major_linear(x, weight, scales)
    seen = []
    original = quantization.quantize

    def observed(source, *, consumer_scale_layout=False):
        seen.append(consumer_scale_layout)
        return original(source, consumer_scale_layout=consumer_scale_layout)

    monkeypatch.setattr(quantization, "quantize", observed)
    actual = fp8.fp8_linear(x, weight, scales)
    assert seen == [True]
    exact(actual, expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fp8.fp8_linear(x, weight, scales)
    torch.cuda.current_stream().wait_stream(stream)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        actual = fp8.fp8_linear(x, weight, scales)
    for _ in range(3):
        x.mul_(0.75).add_(0.125)
        capture.replay()
        torch.cuda.synchronize()
        exact(actual, row_major_linear(x, weight, scales))


@pytest.mark.parametrize(
    "rows,dtype,k,n",
    ((2, torch.bfloat16, 1536, 8192), (1, torch.float16, 1536, 8192), (1, torch.bfloat16, 128, 64)),
)
@torch.inference_mode()
def test_cuda_non_q1_target_linears_keep_default_layout(rows, dtype, k, n, monkeypatch):
    require_sm90()
    x = torch.randn((rows, k), device="cuda", dtype=dtype)
    weight = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(((n + 127) // 128, k // 128), device="cuda")
    seen = []
    original = quantization.quantize

    def observed(source, *, consumer_scale_layout=False):
        seen.append(consumer_scale_layout)
        return original(source, consumer_scale_layout=consumer_scale_layout)

    monkeypatch.setattr(quantization, "quantize", observed)
    fp8.fp8_linear(x, weight, scales)
    assert seen == [False]


@torch.inference_mode()
def test_cuda_q1_prepared_gate_scales_are_consumed_once_without_requantization(monkeypatch):
    require_sm90()
    x = torch.randn((1, 7168), device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn((18432, 7168), device="cuda").to(torch.float8_e4m3fn) for _ in range(2)]
    scales = torch.ones((144, 56), device="cuda")
    expected = [row_major_linear(x, weight, scales) for weight in weights]
    gate, prepared = fp8.fp8_linear(x, weights[0], scales, return_quantized=True)
    assert prepared.scales.stride() == (1, 4)

    def forbidden(*args, **kwargs):
        pytest.fail("Prepared Q1 scale layout must be reused")

    monkeypatch.setattr(quantization, "quantize", forbidden)
    up = fp8.fp8_linear(x, weights[1], scales, quantized=prepared)
    exact(gate, expected[0])
    exact(up, expected[1])
    with pytest.raises(ValueError, match="already been consumed"):
        fp8.fp8_linear(x, weights[1], scales, quantized=prepared)
