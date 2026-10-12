"""CPU checks for immutable eager snapshots and exact static-output comparison."""

from dataclasses import dataclass

import pytest
import torch

from experiments.deepseek_v32_observation.src.graph_validation import equal_output, snapshot


@dataclass
class Output:
    scores: torch.Tensor
    details: dict


def test_snapshot_preserves_nested_eager_values_after_buffer_reuse():
    scores = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    indices = torch.tensor([3, 1], dtype=torch.int64)
    eager = Output(scores, {"selected": (indices, {"count": 2}), "policy": "static"})
    expected = snapshot(eager)

    equal_output(eager, expected)
    assert expected["scores"].data_ptr() != scores.data_ptr()
    assert expected["details"]["selected"][0].data_ptr() != indices.data_ptr()
    scores.fill_(0)
    indices.fill_(-1)
    eager.details["selected"][1]["count"] = 0

    torch.testing.assert_close(expected["scores"], torch.arange(12).reshape(3, 4).T.float())
    assert expected["details"]["selected"][1]["count"] == 2
    with pytest.raises(AssertionError, match="bits differ"):
        equal_output(eager, expected)


def test_equal_output_compares_logical_values_with_different_strides():
    actual = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    expected = actual.contiguous()
    assert actual.stride() != expected.stride()
    equal_output(actual, expected)


def test_equal_output_accepts_fp8_and_identical_masked_score_tails():
    fp8 = torch.tensor([0.0, 0.5, -2.0]).to(torch.float8_e4m3fn)
    scores = torch.tensor([1.5, -float("inf"), -float("inf")])
    equal_output({"fp8": fp8, "scores": scores}, snapshot({"fp8": fp8, "scores": scores}))


@pytest.mark.parametrize("replacement", [float("nan"), float("inf"), -float("inf")])
def test_equal_output_rejects_injected_nonfinite_values(replacement):
    expected = torch.tensor([1.0, 2.0])
    actual = expected.clone()
    actual[1] = replacement
    with pytest.raises(AssertionError, match="bits differ"):
        equal_output(actual, expected)


def test_equal_output_rejects_signed_zero_change():
    # Numerical equality alone would conceal this change in a static replay.
    with pytest.raises(AssertionError, match="bits differ"):
        equal_output(torch.tensor([-0.0]), torch.tensor([0.0]))


@pytest.mark.parametrize(
    "actual, expected, message",
    [
        (torch.ones(1, 2), torch.ones(2), "shape or dtype"),
        (torch.ones(2, dtype=torch.bfloat16), torch.ones(2), "shape or dtype"),
        ({"scores": 1, "extra": 2}, {"scores": 1}, "fields differ"),
        ([1], [1, 2], "arity differs"),
        ({"policy": "changed"}, {"policy": "static"}, "scalar metadata"),
    ],
)
def test_equal_output_rejects_output_contract_changes(actual, expected, message):
    with pytest.raises(AssertionError, match=message):
        equal_output(actual, expected)
