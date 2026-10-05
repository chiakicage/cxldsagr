"""Production packed dense MLP bytes, stream ordering and graph ownership."""

import pytest
import torch

from models.deepseek_v32.checkpoint import CheckpointLinear
from models.deepseek_v32.layers import CheckpointMLP


def make_mlp(columns, channels, device):
    mlp = object.__new__(CheckpointMLP)
    mlp.pack_gate_up = True
    for name, shape in (
        ("gate", (columns, channels)),
        ("up", (columns, channels)),
        ("down", (channels, columns)),
    ):
        linear = object.__new__(CheckpointLinear)
        linear.weight = (torch.randn(shape, device=device) * 8).to(torch.float8_e4m3fn)
        linear.scales = torch.full(
            tuple((n + 127) // 128 for n in shape), 0.03125, device=device, dtype=torch.float32
        )
        setattr(mlp, name, linear)
    return mlp


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; explicit GPU regression requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("Packed MLP production checks require SM90/Hopper")


def assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def test_cpu_and_empty_dispatch_preserve_original_path():
    mlp = make_mlp(64, 128, "cpu")
    x = torch.randn(3, 128).bfloat16()
    assert not mlp._can_pack_gate_up(x)
    actual = mlp(x)
    mlp.pack_gate_up = False
    assert_bits(actual, mlp(x))
    assert mlp(torch.empty(0, 128).bfloat16()).shape == (0, 128)


@pytest.mark.parametrize("rows,columns,channels", [(1, 64, 128), (128, 256, 128), (129, 192, 256)])
@torch.inference_mode()
def test_cuda_packed_mlp_exact_lifetime_stream_and_graph(rows, columns, channels):
    require_sm90()
    torch.manual_seed(847)
    mlp = make_mlp(columns, channels, "cuda")
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    original = x.clone()
    before = {
        name: (getattr(mlp, name).weight.clone(), getattr(mlp, name).scales.clone())
        for name in ("gate", "up", "down")
    }
    mlp.pack_gate_up = False
    expected = mlp(x)
    mlp.pack_gate_up = True
    assert mlp._can_pack_gate_up(x)
    actual = mlp(x)
    assert_bits(actual, expected)
    x.mul_(0.5)
    later = mlp(x)
    assert actual.data_ptr() != later.data_ptr()
    assert_bits(actual, expected)
    x.copy_(original)
    current, stream = torch.cuda.current_stream(), torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        x.add_(0.125)
        actual = mlp(x)
        mlp.pack_gate_up = False
        expected_stream = mlp(x)
        mlp.pack_gate_up = True
        consumed = actual.clone()
        done = torch.cuda.Event()
        done.record()
    current.wait_event(done)
    assert_bits(consumed, expected_stream)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(3):
            mlp(x)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = mlp(x)
    stream.synchronize()
    retained = []
    for scale in (0.5, -1.25, 2.0):
        x.mul_(scale)
        graph.replay()
        mlp.pack_gate_up = False
        expected_graph = mlp(x)
        mlp.pack_gate_up = True
        assert_bits(captured, expected_graph)
        retained.append((captured.clone(), expected_graph))
    for owned, expected_graph in retained:
        assert_bits(owned, expected_graph)
    for name, pair in before.items():
        assert_bits(getattr(mlp, name).weight, pair[0])
        assert_bits(getattr(mlp, name).scales, pair[1])
    assert set(vars(mlp)) == {"pack_gate_up", "gate", "up", "down"}
