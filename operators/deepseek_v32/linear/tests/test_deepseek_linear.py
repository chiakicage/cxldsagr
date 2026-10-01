"""FP8 projection and grouped expert numerical correctness, without timing."""

import pytest
import torch

from operators.deepseek_v32.linear.fp8 import (
    fp8_linear,
    grouped_fp8_linear,
    prepare_expert_routing,
    quantize_fp8_activation,
    reference_fp8_linear,
    reference_grouped_fp8_linear,
    reference_quantize_fp8_activation,
)


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("FP8 DeepSeek projection checks require SM90/Hopper")


def make_weight(shape, device):
    weight = (torch.randn(shape, device=device) * 20).to(torch.float8_e4m3fn)
    scale_shape = (*shape[:-2], (shape[-2] + 127) // 128, (shape[-1] + 127) // 128)
    scales = torch.rand(scale_shape, device=device, dtype=torch.float32) * 0.02 + 0.003
    return weight, scales


def test_cpu_linear_known_block_scales_and_activation_rounding():
    x = torch.full((2, 257), 0.5, dtype=torch.bfloat16)
    weight = torch.ones(130, 257).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output = fp8_linear(x, weight, scales)
    expected = torch.cat((torch.full((2, 128), 193.5), torch.full((2, 2), 579.0)), dim=1)
    torch.testing.assert_close(output, expected.to(torch.bfloat16), rtol=0, atol=0)


def test_cpu_grouped_broadcast_and_route_inputs_with_empty_experts():
    torch.manual_seed(13)
    x = torch.randn(4, 24, dtype=torch.bfloat16)
    ids = torch.tensor([[2, 0], [0, 0], [-1, 2], [8, 2]], dtype=torch.int32)
    weights, scales = make_weight((4, 31, 24), "cpu")
    output = grouped_fp8_linear(x, ids, weights, scales)
    assert torch.count_nonzero(output[2, 0]) == 0
    assert torch.count_nonzero(output[3, 0]) == 0
    for token, route in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (3, 1)]:
        expert = ids[token, route]
        expected = fp8_linear(x[token : token + 1], weights[expert], scales[expert])
        torch.testing.assert_close(output[token, route], expected[0], rtol=0, atol=0)
    routed = x[:, None, :].expand(-1, 2, -1).clone()
    torch.testing.assert_close(
        grouped_fp8_linear(routed, ids, weights, scales), output, rtol=0, atol=0
    )


def test_routing_offsets_cover_only_valid_routes_and_pad_each_expert():
    ids = torch.tensor([[3, 0, 3], [0, 3, -1], [2, 9, 3]], dtype=torch.int32)
    plan = prepare_expert_routing(ids, 4)
    assert plan.row_offsets.tolist() == [0, 2, 2, 3, 7]
    assert plan.tile_offsets.tolist() == [0, 1, 1, 2, 3]
    sorted_experts = ids.flatten()[plan.sorted_routes[:7]].tolist()
    assert sorted_experts == [0, 0, 2, 3, 3, 3, 3]


@pytest.mark.parametrize("columns", [1, 24, 128, 257, 7168])
def test_cuda_quantization_matches_independent_oracle(columns):
    require_sm90()
    torch.manual_seed(22)
    x = torch.randn(5, columns * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    x[0] = 0
    expected, expected_scale = reference_quantize_fp8_activation(x)
    actual, actual_scale = quantize_fp8_activation(x)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("rows", "out_channels", "in_channels"),
    [(1, 17, 24), (7, 137, 257), (65, 512, 1024), (128, 1536, 7168)],
)
def test_cuda_dense_matches_fp32_oracle(rows, out_channels, in_channels):
    require_sm90()
    torch.manual_seed(56)
    x = torch.randn(rows, in_channels, device="cuda", dtype=torch.bfloat16)
    weights, scales = make_weight((out_channels, in_channels), "cuda")
    expected = reference_fp8_linear(x, weights, scales)
    actual = fp8_linear(x, weights, scales)
    torch.testing.assert_close(actual, expected, atol=0.0625, rtol=8e-3)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.003


@pytest.mark.parametrize("route_input", [False, True])
@pytest.mark.parametrize(
    ("tokens", "topk", "experts", "in_channels", "out_channels"),
    [(5, 3, 4, 24, 17), (67, 4, 9, 257, 137), (128, 8, 16, 2048, 512)],
)
def test_cuda_grouped_matches_fp32_oracle(
    tokens, topk, experts, in_channels, out_channels, route_input
):
    require_sm90()
    torch.manual_seed(119)
    x_shape = (tokens, topk, in_channels) if route_input else (tokens, in_channels)
    x = torch.randn(x_shape, device="cuda", dtype=torch.bfloat16)
    weights, scales = make_weight((experts, out_channels, in_channels), "cuda")
    ids = torch.randint(0, experts - 1, (tokens, topk), device="cuda", dtype=torch.int32)
    ids[0] = -1
    ids[-1, 0] = experts + 1
    routing = prepare_expert_routing(ids, experts)
    expected = reference_grouped_fp8_linear(x, ids, weights, scales)
    actual = grouped_fp8_linear(x, ids, weights, scales, routing=routing)
    torch.testing.assert_close(actual, expected, atol=0.0625, rtol=8e-3)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.003
    assert torch.count_nonzero(actual[0]) == 0


def test_cuda_grouped_reuses_routing_for_down_projection():
    require_sm90()
    torch.manual_seed(310)
    x = torch.randn(35, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 4, (35, 3), device="cuda", dtype=torch.int32)
    plan = prepare_expert_routing(ids, 4)
    up_w, up_s = make_weight((4, 256, 128), "cuda")
    down_w, down_s = make_weight((4, 128, 256), "cuda")
    intermediate = grouped_fp8_linear(x, ids, up_w, up_s, routing=plan)
    actual = grouped_fp8_linear(intermediate, ids, down_w, down_s, routing=plan)
    expected = reference_grouped_fp8_linear(intermediate, ids, down_w, down_s)
    torch.testing.assert_close(actual, expected, atol=0.0625, rtol=8e-3)
    with pytest.raises(ValueError, match="same expert_ids"):
        grouped_fp8_linear(x, ids.clone(), up_w, up_s, routing=plan)


@pytest.mark.parametrize("grouped", [False, True])
def test_empty_tokens_cpu_and_cuda(grouped):
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        if device == "cuda":
            require_sm90()
        x = torch.empty(0, 24, device=device, dtype=torch.bfloat16)
        if grouped:
            weights, scales = make_weight((3, 16, 24), device)
            ids = torch.empty(0, 2, device=device, dtype=torch.int32)
            result = grouped_fp8_linear(x, ids, weights, scales)
            assert result.shape == (0, 2, 16)
        else:
            weights, scales = make_weight((16, 24), device)
            assert fp8_linear(x, weights, scales).shape == (0, 16)
