"""Independent checks for decomposed matrix arithmetic and scope metadata."""

from dataclasses import replace

import pytest
import torch

from experiments.deepseek_v32_mfu.src.operator_flops import linear_work, non_matmul_work
from experiments.deepseek_v32_observation.src.common_workload import (
    CommonWorkload,
    prepared_fp8_verifier,
)


def test_prepared_gemm_reference_uses_actual_quantized_operands():
    generator = torch.Generator().manual_seed(917)
    data = torch.randn(3, 256, generator=generator).to(torch.float8_e4m3fn)
    weights = torch.randn(256, 256, generator=generator).to(torch.float8_e4m3fn)
    activation_scales = torch.tensor([[0.5, 2.0], [1.0, 4.0], [2.0, 0.25]])
    weight_scales = torch.tensor([[0.25, 2.0], [4.0, 0.5]])
    expanded_x = data.float() * activation_scales.repeat_interleave(128, -1)
    expanded_weights = weights.float() * weight_scales.repeat_interleave(128, 0).repeat_interleave(
        128, 1
    )
    output = (expanded_x @ expanded_weights.T).bfloat16()
    verify = prepared_fp8_verifier(data, activation_scales, weights, weight_scales)

    receipt = verify(output)
    assert receipt["checked_shape"] == [3, 256]
    assert receipt["bitwise_equal"]
    corrupted = output.clone()
    corrupted[1, 129] += 16
    with pytest.raises(AssertionError):
        verify(corrupted)


def test_metadata_keeps_complete_linear_separate_from_gemm_and_cast():
    owner = CommonWorkload.__new__(CommonWorkload)
    owner.s = 1
    owner.cases = []
    matrix = linear_work("q_a_proj", rows=1, in_features=128, out_features=64, precision="fp8")
    for name, work, metadata in (
        ("q_a_proj", matrix, {}),
        (
            "q_a_proj_gemm",
            replace(matrix, name="q_a_proj_gemm"),
            {
                "operator_kind": "gemm",
                "operator_family": "input_projection",
                "parent_api": "q_a_proj",
            },
        ),
        ("index_weights_input_cast", non_matmul_work("index_weights_input_cast"), {}),
    ):
        owner._add(name, lambda: None, work, lambda value: {}, **metadata)

    aggregate, gemm, cast = owner.cases
    assert aggregate.metadata["scope"] == "module_aggregate"
    assert aggregate.metadata["operator_kind"] == "api_aggregate"
    assert aggregate.metadata["parent_api"] == "q_a_proj"
    assert gemm.metadata["scope"] == "operator"
    assert gemm.metadata["operator_kind"] == "gemm"
    assert cast.metadata["operator_family"] == "cast"
    assert cast.metadata["parent_api"] == "index_weights_proj"
