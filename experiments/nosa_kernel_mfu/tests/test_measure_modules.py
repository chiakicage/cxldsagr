"""Complete-module metrics must separate device kernels from host intervals."""

import pytest
import torch

from experiments.nosa_kernel_mfu.src.measure_modules import (
    assert_same_output,
    kernel_activity,
)
from layers.attention import BlockSelection


def test_kernel_sum_includes_auxiliaries_without_counting_launch_or_copy_twice():
    trace = {
        "traceEvents": [
            {"cat": "cpu_op", "name": "module", "dur": 500},
            {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "dur": 20},
            {"cat": "kernel", "name": "validate", "dur": 5},
            {"cat": "gpu_memcpy", "name": "Memcpy DtoH", "dur": 3},
            {"cat": "kernel", "name": "compress", "dur": 7},
            {"cat": "kernel", "name": "score", "dur": 80},
            {"cat": "kernel", "name": "select", "dur": 20},
            {"ph": "M", "name": "process_name"},
        ]
    }
    result = kernel_activity(trace)
    assert result["kernel_ms"] == pytest.approx(0.112)
    assert result["memcpy_ms"] == pytest.approx(0.003)
    assert [item["name"] for item in result["kernels"]] == [
        "validate",
        "compress",
        "score",
        "select",
    ]


def test_repeated_selection_acceptance_checks_validity_as_well_as_ids():
    ids = torch.tensor([[[0, -1]]])
    valid = ids >= 0
    assert_same_output(BlockSelection(ids, 64, valid), (ids.clone(), valid.clone()))
    with pytest.raises(AssertionError):
        assert_same_output(BlockSelection(ids, 64, ~valid), (ids, valid))
