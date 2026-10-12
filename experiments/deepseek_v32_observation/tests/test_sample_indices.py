"""Sampling retains valid endpoints even beyond float32 integer precision."""

import pytest
import torch

from experiments.deepseek_v32_observation.src.common_workload import sample_indices


@pytest.mark.parametrize("count", [2**27, 2**27 + 1])
def test_large_logical_population_has_exact_in_range_endpoints(count):
    indices = sample_indices(count, 4096, "cpu")

    assert indices.dtype == torch.int64
    assert indices.device.type == "cpu"
    assert indices.numel() == 4096
    assert indices[0].item() == 0
    assert indices[-1].item() == count - 1
    assert bool(((indices >= 0) & (indices < count)).all())
    assert bool((indices[1:] > indices[:-1]).all())
    assert indices.unique().numel() == indices.numel()


@pytest.mark.parametrize("count, limit", [(1, 4096), (2, 4096), (3, 3), (17, 64)])
def test_small_population_is_covered_without_repeated_indices(count, limit):
    indices = sample_indices(count, limit, "cpu")
    torch.testing.assert_close(indices, torch.arange(count, dtype=torch.int64))


def test_single_sample_uses_first_valid_item():
    torch.testing.assert_close(sample_indices(2**27, 1, "cpu"), torch.tensor([0]))
