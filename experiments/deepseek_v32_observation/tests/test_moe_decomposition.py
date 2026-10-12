"""References for quantized routed GEMM and explicit MoE leaf inventories."""

from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_mfu.src.operator_flops import linear_work, non_matmul_work
from experiments.deepseek_v32_observation.src.moe_workload import (
    MoEWorkload,
    prepared_grouped_verifier,
)
from models.deepseek_v32.config import Config
from models.deepseek_v32.layers import route_experts


def test_prepared_grouped_reference_follows_actual_route_slots_and_scales():
    generator = torch.Generator().manual_seed(222)
    ids = torch.tensor([[2, 0], [1, 2], [0, 1]])
    # Padding slots and expert-order packing do not follow logical route order.
    slots = torch.tensor([7, 0, 4, 8, 1, 5])
    data = torch.randn(10, 256, generator=generator).to(torch.float8_e4m3fn)
    scale = torch.rand(10, 2, generator=generator) + 0.5
    weight = torch.randn(3, 256, 256, generator=generator).to(torch.float8_e4m3fn)
    weight_scale = torch.rand(3, 2, 2, generator=generator) + 0.25
    output = torch.full((10, 256), torch.nan, dtype=torch.bfloat16)
    for route, expert in enumerate(ids.flatten()):
        slot = slots[route]
        x = data[slot].float() * scale[slot].repeat_interleave(128)
        w = weight[expert].float() * weight_scale[expert].repeat_interleave(
            128, 0
        ).repeat_interleave(128, 1)
        output[slot] = (x @ w.T).bfloat16()
    routing = SimpleNamespace(expert_ids=ids, output_slots=slots)
    verify = prepared_grouped_verifier(data, scale, weight, weight_scale, routing)

    receipt = verify(output)
    assert receipt["packed_row_indices"] == slots.tolist()
    assert receipt["checked_shape"] == [6, 256]
    assert receipt["bitwise_equal"]
    corrupted = output.clone()
    corrupted[slots[3], 140] += 32
    with pytest.raises(AssertionError):
        verify(corrupted)


def test_routing_leaf_fixtures_preserve_unbiased_weights_and_gemm_scope():
    owner = MoEWorkload.__new__(MoEWorkload)
    owner.s, owner.layer_idx, owner.cases = 3, 3, []
    cfg = Config(
        dim=16,
        n_routed_experts=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        routed_scaling_factor=2.5,
    )
    generator = torch.Generator().manual_seed(994)
    hidden = torch.randn(3, 16, generator=generator).bfloat16()
    gate = torch.randn(8, 16, generator=generator)
    bias = torch.tensor([0, 0, 0, 0, 8, 8, 8, 8], dtype=torch.float32)
    owner._routing_leaves(hidden, gate, bias, cfg)
    leaves = {case.name: case for case in owner.cases}
    for case in leaves.values():
        case.verify(case.call())
        assert case.metadata["scope"] == "operator"
        assert case.metadata["parent_api"] == "moe_routing"
    matrix = leaves["moe_routing_gate_gemm"]
    assert matrix.metadata["operator_kind"] == "gemm"
    assert matrix.work.useful_flops == 2 * 3 * 16 * 8
    _, expected_weights = route_experts(hidden, gate, bias, cfg)
    torch.testing.assert_close(leaves["moe_routing_weight_scale"].call(), expected_weights)
    assert leaves["moe_routing_expert_topk"].call().min() >= 4


def test_aggregate_and_simple_leaf_metadata_have_distinct_scopes():
    owner = MoEWorkload.__new__(MoEWorkload)
    owner.s, owner.layer_idx, owner.cases = 1, 3, []
    owner._add(
        "moe_grouped_gate",
        lambda: None,
        linear_work("moe_grouped_gate", rows=8, in_features=128, out_features=128, precision="fp8"),
        lambda actual: {},
        operator_kind="api_aggregate",
        operator_family="moe_gemm",
    )
    owner._add(
        "moe_output_cast",
        lambda: None,
        non_matmul_work("moe_output_cast"),
        lambda actual: {},
        parent_api="moe",
    )
    aggregate, cast = owner.cases
    assert aggregate.metadata["scope"] == "module_aggregate"
    assert aggregate.metadata["parent_api"] == "moe_grouped_gate"
    assert aggregate.metadata["not_additive"]
    assert cast.metadata["scope"] == "operator"
    assert cast.metadata["operator_kind"] == "non_gemm"
    assert cast.metadata["operator_family"] == "moe_cast"
    assert cast.metadata["parent_api"] == "moe"


def test_metadata_primitives_prepare_permutation_and_reset_accumulation():
    owner = MoEWorkload.__new__(MoEWorkload)
    owner.s, owner.layer_idx, owner.cases, owner._before_calls = 3, 3, [], {}
    owner.cfg = SimpleNamespace(n_routed_experts=3)
    owner.expert_ids = torch.tensor([[2, 0], [1, 2], [0, 1]])
    # A small static packed layout uses two rows per expert and trailing padding.
    owner.routing = SimpleNamespace(
        expert_ids=owner.expert_ids,
        block_m=2,
        sorted_routes=torch.tensor([1, 4, 2, 5, 0, 3]),
        row_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        packed_routes=torch.tensor([1, 4, 2, 5, 0, 3, 0, 0]),
    )
    owner._metadata_primitives()
    for case in owner.cases:
        reset = owner.before_call(case.name)
        for _ in range(2):
            if reset is not None:
                reset()
            case.verify(case.call())
        assert case.metadata["operator_kind"] == "non_gemm"
        assert case.metadata["scope"] == "operator"
    assert owner.before_call("moe_metadata_counts_scatter_add") is not None
