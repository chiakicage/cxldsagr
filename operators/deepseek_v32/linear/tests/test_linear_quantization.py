"""Public quantizer contract and focused handwritten-Triton regressions.

The compiled DeepGEMM helper is an oracle confined to CUDA tests. Graph replay,
stream ordering and prepared-output ownership are covered by the existing
component and consumer suites.
"""

import importlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from operators.deepseek_v32.linear import fp8

REPO = Path(fp8.__file__).resolve().parents[3]
QUANTIZATION_MODULE = "operators.deepseek_v32.linear.quantization"
DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _forbid_cuda_implementation(*args, **kwargs):
    pytest.fail("This path must not compile or call the CUDA quantizer")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
def test_cpu_public_quantization_known_group_scales_and_signed_zero(dtype, strided, monkeypatch):
    monkeypatch.setattr(fp8, "_deep_gemm", _forbid_cuda_implementation)
    monkeypatch.setattr(torch, "compile", _forbid_cuda_implementation)
    owner = torch.full((2, 514 if strided else 257), 13, dtype=dtype)
    x = owner[:, ::2] if strided else owner
    x.zero_()
    x[0, 0], x[0, 127] = 1.0, -1.0
    x[0, 128], x[0, 255] = 448.0, -448.0
    x[0, 256] = math.ldexp(1.0, -22)
    x[1, ::2] = -0.0
    source_bits = owner.view(torch.uint8).clone()

    data, scales = fp8.quantize_fp8_activation(x)

    expected = torch.zeros((2, 257), dtype=torch.uint8)
    expected[0, 0], expected[0, 127] = 0x78, 0xF8
    expected[0, 128], expected[0, 255] = 0x7E, 0xFE
    expected[0, 256] = 0x38
    expected[1, ::2] = 0x80
    expected_scales = torch.tensor(
        [[2.0**-8, 1.0, 2.0**-22], [2.0**-22, 2.0**-22, 2.0**-22]], dtype=torch.float32
    )
    assert data.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert data.is_contiguous() and scales.is_contiguous()
    assert torch.equal(data.view(torch.uint8), expected)
    assert torch.equal(scales.view(torch.int32), expected_scales.view(torch.int32))
    assert torch.equal(owner.view(torch.uint8), source_bits)


@pytest.mark.parametrize("shape", [(), (128,), (2, 1, 128), (2, 0)])
def test_cpu_public_quantization_rejects_invalid_rank_or_empty_channels(shape, monkeypatch):
    monkeypatch.setattr(fp8, "_deep_gemm", _forbid_cuda_implementation)
    monkeypatch.setattr(torch, "compile", _forbid_cuda_implementation)
    with pytest.raises(ValueError, match="Expected.*activation"):
        fp8.quantize_fp8_activation(torch.empty(shape, dtype=torch.bfloat16))


@pytest.mark.parametrize("dtype", [torch.int32, torch.float64, torch.bool, torch.complex64])
def test_cpu_public_quantization_rejects_unsupported_dtype(dtype, monkeypatch):
    monkeypatch.setattr(fp8, "_deep_gemm", _forbid_cuda_implementation)
    monkeypatch.setattr(torch, "compile", _forbid_cuda_implementation)
    with pytest.raises(ValueError, match="BF16, FP16, or FP32"):
        fp8.quantize_fp8_activation(torch.empty((2, 128), dtype=dtype))


def test_cpu_public_quantization_empty_rows():
    data, scales = fp8.quantize_fp8_activation(torch.empty((0, 129), dtype=torch.bfloat16))
    assert data.shape == (0, 129) and data.dtype == torch.float8_e4m3fn
    assert scales.shape == (0, 2) and scales.dtype == torch.float32
    assert data.device.type == scales.device.type == "cpu"


@pytest.mark.parametrize("capability", [(8, 0), (9, 1), (10, 0)])
def test_cpu_non_sm90_public_call_rejects_before_cuda_implementation(capability, monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    # Real tensor metadata exercises the public CUDA guard without allocating
    # CUDA storage or requiring a device to exist on the CPU regression host.
    x = FakeTensor(
        FakeTensorMode(),
        torch.empty((2, 128), dtype=torch.bfloat16, device="meta"),
        torch.device("cuda:0"),
    )
    observed_devices = []

    def device_capability(device):
        observed_devices.append(device)
        return capability

    monkeypatch.setattr(torch.cuda, "get_device_capability", device_capability)
    monkeypatch.setattr(fp8, "_deep_gemm", _forbid_cuda_implementation)
    monkeypatch.setattr(torch, "compile", _forbid_cuda_implementation)
    with pytest.raises(NotImplementedError, match="SM90/Hopper"):
        fp8.quantize_fp8_activation(x)
    assert observed_devices == [x.device]


def test_cpu_public_import_and_fallback_do_not_load_gpu_dependencies():
    program = r"""
import importlib.abc
import json
import subprocess
import sys
import torch

class DenyGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"deep_gemm", "tvm_ffi", "triton"}:
            raise AssertionError(f"CPU path tried to import {fullname}")

def forbidden(*args, **kwargs):
    raise AssertionError("CPU path tried to compile or launch a subprocess")

assert "deep_gemm" not in sys.modules and "tvm_ffi" not in sys.modules
assert not torch.cuda.is_initialized()
sys.path.insert(0, sys.argv[1])
sys.meta_path.insert(0, DenyGPUImports())
torch.compile = forbidden
subprocess.Popen = forbidden
from operators.deepseek_v32.linear.fp8 import quantize_fp8_activation
data, scales = quantize_fp8_activation(torch.ones((2, 129), dtype=torch.bfloat16))
assert data.shape == (2, 129) and scales.shape == (2, 2)
assert "deep_gemm" not in sys.modules and "tvm_ffi" not in sys.modules
assert not torch.cuda.is_initialized()
print(json.dumps({"cpu_fallback": True, "cuda_initialized": False}))
"""
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program, str(REPO)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {"cpu_fallback": True, "cuda_initialized": False}


def _signed_word(value, bits):
    return value if value < (1 << (bits - 1)) else value - (1 << bits)


def _edge_values(dtype, rows, columns):
    """Preserve signed/signaling NaN encodings while constructing CPU inputs."""
    bits = 32 if dtype == torch.float32 else 16
    raw_dtype = torch.int32 if bits == 32 else torch.int16
    sign = 1 << (bits - 1)
    special = {
        torch.bfloat16: (0x7F80, 0xFF80, 0x7FC1, 0xFFC1, 0x7F81, 0xFF81),
        torch.float16: (0x7C00, 0xFC00, 0x7E01, 0xFE01, 0x7C01, 0xFC01),
        torch.float32: (0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC00001, 0x7F800001, 0xFF800001),
    }[dtype]
    raw = torch.zeros((9, 128), dtype=raw_dtype)
    raw[:, 1::2] = _signed_word(sign, bits)
    values = raw.view(dtype)
    finite_witness = torch.finfo(dtype).max if dtype == torch.float16 else 2.0**127
    for group, (subject, position) in enumerate(
        zip(special, (127, 32, 63, 64, 95, 96), strict=True)
    ):
        values[group, 0], values[group, 1] = finite_witness, -finite_witness
        values[group, 2], values[group, 3] = 0.0, -0.0
        raw[group, 4], raw[group, 5] = 1, _signed_word(sign | 1, bits)
        values[group, 6], values[group, 7] = 1.0, -1.0
        raw[group, position] = _signed_word(subject, bits)
    values[6, 0], values[6, 1] = torch.finfo(dtype).max, -torch.finfo(dtype).max
    values[7, 0], values[7, 1] = 2.0**-14, -(2.0**-14)
    values[8, 127] = 448.0
    ties = torch.tensor([1.0625, 1.1875, -1.0625, -1.1875], dtype=dtype)
    values[8, :4] = ties
    values[8, 4:8] = torch.nextafter(ties, torch.full_like(ties, -float("inf")))
    values[8, 8:12] = torch.nextafter(ties, torch.full_like(ties, float("inf")))
    logical = raw.flatten().repeat((columns + raw.numel() - 1) // raw.numel())[:columns]
    logical = logical.repeat(rows, 1).contiguous()
    if columns % 128:
        logical[:, -1] = _signed_word(special[3], bits)
    return logical.view(dtype)


def _cpu_case(dtype, rows, columns, layout):
    values = _edge_values(dtype, rows, columns)
    strides = {
        "contiguous": (columns, 1),
        "misaligned": (columns, 1),
        "pitched": (columns + 4, 1),
        "column_stride2": (2 * columns + 8, 2),
        "transposed": (1, rows + 2),
        "broadcast_rows": (0, 1),
    }[layout]
    offset = 1 if layout == "misaligned" else 16
    span = (rows - 1) * strides[0] + (columns - 1) * strides[1] + 1
    owner = torch.full((offset + span + 16,), 13.0, dtype=dtype)
    x = owner.as_strided((rows, columns), strides, offset)
    if layout == "broadcast_rows":
        x[0].copy_(values[0])
    else:
        x.copy_(values)
    return owner, x


CUDA_CASES = (
    *(
        pytest.param(torch.bfloat16, width, "contiguous", id=f"dense-bf16-k{width}")
        for width in (1536, 7168, 16384, 18432)
    ),
    pytest.param(torch.bfloat16, 1536, "misaligned", id="bf16-offset2bytes"),
    pytest.param(torch.bfloat16, 1536, "pitched", id="bf16-pitched"),
    pytest.param(torch.bfloat16, 1536, "column_stride2", id="bf16-column-stride2"),
    pytest.param(torch.bfloat16, 1536, "broadcast_rows", id="bf16-broadcast-rows"),
    pytest.param(torch.bfloat16, 128, "contiguous", id="bf16-single-group"),
    pytest.param(torch.bfloat16, 129, "contiguous", id="bf16-tail1"),
    pytest.param(torch.bfloat16, 1535, "contiguous", id="bf16-tail127"),
    pytest.param(torch.float16, 1536, "contiguous", id="fp16-generic"),
    pytest.param(torch.float16, 257, "transposed", id="fp16-transposed-tail1"),
    pytest.param(torch.float32, 1536, "contiguous", id="fp32-generic"),
    pytest.param(torch.float32, 257, "column_stride2", id="fp32-stride2-tail1"),
)


@pytest.fixture
def compiled_official_oracle():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("Triton quantizer regression requires SM90/Hopper")
    import deep_gemm

    helper = deep_gemm.per_token_cast_to_fp8
    pinned = REPO / "3rdparty/DeepGEMM/deep_gemm/utils/math.py"
    assert Path(inspect.getfile(helper)).read_bytes() == pinned.read_bytes()

    def official(x):
        return helper(x, use_ue8m0=True, gran_k=128, use_packed_ue8m0=False)

    # Independent dtype/stride guards must not exhaust one Dynamo code cache.
    torch.compiler.reset()
    try:
        yield torch.compile(
            official, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False}
        )
    finally:
        torch.compiler.reset()


def _assert_diagnostic_probes(pair, dtype, columns):
    data, scales = pair
    raw = data.view(torch.uint8)
    scale_bits = scales.view(torch.int32)
    assert scale_bits[0, 0].item() == (254 << 23)
    assert raw[0, 2].item() == 0x00 and raw[0, 3].item() == 0x80
    if dtype in (torch.bfloat16, torch.float32):
        # 1 / 2^127 is FP32 subnormal 0x00400000. Replacing it with zero
        # incorrectly erases these finite witnesses beside Inf/NaN.
        assert raw[0, 0].item() == 0x38 and raw[0, 1].item() == 0xB8
    if columns >= 1024:
        assert torch.all(scale_bits[0, :6] == (254 << 23)).item()
        assert scale_bits[0, 6].item() == ((135 if dtype == torch.float16 else 247) << 23)
        assert scale_bits[0, 7].item() == (105 << 23)
        assert raw[0, 896].item() == 0x78 and raw[0, 897].item() == 0xF8


@pytest.mark.parametrize("dtype,columns,layout", CUDA_CASES)
@torch.inference_mode()
def test_cuda_public_quantizer_exact_adversarial_inputs(
    dtype, columns, layout, compiled_official_oracle, monkeypatch
):
    quantization = importlib.import_module(QUANTIZATION_MODULE)
    cpu_owner, cpu_x = _cpu_case(dtype, 3, columns, layout)
    owner = cpu_owner.to("cuda")
    x = owner.as_strided(cpu_x.shape, cpu_x.stride(), cpu_x.storage_offset())
    before = owner.view(torch.uint8).clone()
    expected = compiled_official_oracle(x)
    _assert_diagnostic_probes(expected, dtype, columns)

    calls = []
    original = quantization.quantize

    def observed(source):
        calls.append(source)
        return original(source)

    monkeypatch.setattr(quantization, "quantize", observed)
    monkeypatch.setattr(fp8, "_deep_gemm", _forbid_cuda_implementation)
    monkeypatch.setattr(torch, "compile", _forbid_cuda_implementation)
    actual = fp8.quantize_fp8_activation(x)
    assert len(calls) == 1 and calls[0] is x
    data, scales = actual
    assert data.shape == x.shape and scales.shape == (len(x), (columns + 127) // 128)
    assert data.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert data.is_contiguous() and scales.is_contiguous()
    assert data.device == scales.device == x.device
    _assert_diagnostic_probes(actual, dtype, columns)
    for result, oracle in zip(actual, expected, strict=True):
        assert torch.equal(result.view(torch.uint8), oracle.contiguous().view(torch.uint8))
    assert torch.equal(owner.view(torch.uint8), before)
