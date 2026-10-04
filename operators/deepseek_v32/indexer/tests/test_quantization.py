"""Independent exact indexer quantization checks, without performance timing."""

import subprocess
import sys

import pytest
import torch

from operators.deepseek_v32.indexer.quantization import (
    quantize_index,
    reference_quantize_index,
)


def oracle(x, scale_fmt):
    values = x.to(torch.float32)
    maxima = torch.amax(torch.abs(values), dim=-1, keepdim=True)
    scales = torch.clamp_min(maxima, 1e-4) / 448.0
    if scale_fmt == "ue8m0":
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    output = torch.clamp(values / scales, -448.0, 448.0).to(torch.float8_e4m3fn)
    return output.contiguous(), scales.contiguous()


def assert_bits_equal(actual, expected):
    data, scales = actual
    ref_data, ref_scales = expected
    assert data.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert data.is_contiguous() and scales.is_contiguous()
    assert data.shape == ref_data.shape and scales.shape == ref_scales.shape
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(scales.view(torch.int32), ref_scales.view(torch.int32))


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("Indexer quantization tests require Hopper SM90")


@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cpu_reference_layout_dtype_and_strides(scale_fmt, dtype):
    source = torch.randn(7, 4, 258, generator=torch.Generator().manual_seed(221)).to(dtype)
    for x in (source[..., ::2], source[:, ::2, :128], source[0, 0, :128]):
        expected = oracle(x, scale_fmt)
        assert_bits_equal(quantize_index(x, scale_fmt), expected)
        assert_bits_equal(reference_quantize_index(x, scale_fmt), expected)


def test_reference_import_does_not_load_triton():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from operators.deepseek_v32.indexer.quantization import "
                "reference_quantize_index; assert 'triton' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_unsupported_scale_format_is_rejected():
    for implementation in (quantize_index, reference_quantize_index):
        with pytest.raises(ValueError, match="scale format"):
            implementation(torch.ones(1, 128), "packed")


@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
@pytest.mark.parametrize("tokens,heads", [(0, 64), (1, 1), (17, 64), (128, 1), (1024, 64)])
def test_cuda_random_and_zero_rows(scale_fmt, tokens, heads):
    require_sm90()
    generator = torch.Generator(device="cuda").manual_seed(222)
    shape = (tokens, heads, 128) if heads != 1 else (tokens, 128)
    x = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    if tokens:
        x[0] = 0
    assert_bits_equal(quantize_index(x, scale_fmt), oracle(x, scale_fmt))


@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
def test_cuda_every_bf16_value_as_row_maximum(scale_fmt):
    require_sm90()
    # Every BF16 encoding covers the amax floor, exponent boundaries,
    # subnormals, infinities and all NaN payloads. Also exercise signed zero.
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = values.to("cuda")
    x = values[:, None].expand(-1, 128).contiguous()
    x[:, 1::2] *= -1
    assert_bits_equal(quantize_index(x, scale_fmt), oracle(x, scale_fmt))


@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
def test_cuda_nonfinite_values_poison_the_row_like_reference(scale_fmt):
    require_sm90()
    x = torch.ones(8, 128, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = torch.inf
    x[1, 0] = -torch.inf
    x[2, 0] = torch.nan
    x[3].fill_(torch.nan)
    x[4].fill_(-torch.inf)
    x[5].fill_(-0.0)
    x[6, 0] = -torch.nan
    x[7, 127] = torch.nan
    assert_bits_equal(quantize_index(x, scale_fmt), oracle(x, scale_fmt))


@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
def test_cuda_fp8_rounding_boundaries_and_tiny_values(scale_fmt):
    require_sm90()
    positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoint = (positive[:-1] + positive[1:]) / 2
    # BF16 nearest neighbors around every FP8 midpoint and several row scales.
    middle = midpoint.to(torch.bfloat16)
    samples = torch.stack(
        (
            torch.nextafter(middle, torch.full_like(middle, -torch.inf)),
            middle,
            torch.nextafter(middle, torch.full_like(middle, torch.inf)),
        )
    ).flatten()
    rows = []
    for factor in (2.0**-30, 2.0**-20, 2.0**-10, 1.0, 2.0**10, 2.0**60):
        for maximum in (448.0, 450.0, 510.0):
            x = torch.zeros(samples.numel(), 128, dtype=torch.bfloat16)
            x[:, 0] = maximum * factor
            x[:, 1] = samples * factor
            x[:, 2] = -samples * factor
            rows.append(x)
    x = torch.cat(rows).to("cuda")
    assert_bits_equal(quantize_index(x, scale_fmt), oracle(x, scale_fmt))


@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cuda_reference_fallback_for_layouts_and_dtypes(scale_fmt, dtype):
    require_sm90()
    source = torch.randn(17, 4, 256, device="cuda", dtype=dtype)
    for x in (source[..., ::2], source.transpose(0, 1), source[:, :, :127], source[0, 0]):
        assert_bits_equal(quantize_index(x, scale_fmt), oracle(x, scale_fmt))
