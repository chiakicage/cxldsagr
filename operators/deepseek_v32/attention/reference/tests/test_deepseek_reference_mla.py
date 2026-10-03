"""PyTorch sparse MLA semantics without GPU kernel imports."""

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from operators.deepseek_v32.attention.reference.torch import reference_sparse_mla


def test_reference_import_and_execution_do_not_load_gpu_backends():
    script = """
import importlib.abc
import sys
import torch

class RejectGPUBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'triton', 'tvm_ffi'}:
            raise AssertionError(f'Reference imported GPU backend: {fullname}')

assert not any(name.split('.')[0] in {'triton', 'tvm_ffi'} for name in sys.modules)
sys.meta_path.insert(0, RejectGPUBackends())
from operators.deepseek_v32.attention.reference.torch import reference_sparse_mla

q = torch.zeros(1, 1, 24)
kv = torch.ones(2, 24)
ids = torch.tensor([[1, -1]])
torch.testing.assert_close(reference_sparse_mla(q, kv, ids, 1.0, 16), torch.ones(1, 1, 16))
assert not any(name.split('.')[0] in {'triton', 'tvm_ffi'} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[5],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reference_selected_slots_include_duplicates_and_ignore_padding():
    q = torch.zeros(3, 2, 24)
    kv = torch.zeros(3, 24)
    kv[0] = float("nan")
    kv[1, :16] = 2
    kv[2, :16] = 5
    indices = torch.tensor([[1, 2, 2, -1], [-1, -1, 10, -1], [1, -1, -1, -1]])
    actual = reference_sparse_mla(q, kv, indices, 1.0, value_dim=16)
    torch.testing.assert_close(actual[0], torch.full((2, 16), 4.0))
    torch.testing.assert_close(actual[1], torch.zeros(2, 16))
    torch.testing.assert_close(actual[2], torch.full((2, 16), 2.0))


def test_reference_position_dimensions_affect_scores_but_are_not_values():
    q = torch.zeros(1, 1, 24)
    q[..., 23] = 1
    kv = torch.zeros(2, 24)
    kv[1, :16] = 1
    kv[1, 23] = math.log(3)
    output = reference_sparse_mla(q, kv, torch.tensor([[0, 1]]), 1.0, value_dim=16)
    torch.testing.assert_close(output, torch.full((1, 1, 16), 0.75))


@pytest.mark.parametrize("bad_case", ["dimension", "queries", "dtype", "ids", "value", "scale"])
def test_invalid_inputs_are_rejected(bad_case):
    q, kv, ids, scale, value_dim = (
        torch.zeros(2, 3, 24),
        torch.zeros(4, 24),
        torch.zeros(2, 4, dtype=torch.int32),
        0.2,
        16,
    )
    if bad_case == "dimension":
        kv = kv[:, :20]
    elif bad_case == "queries":
        ids = ids[:1]
    elif bad_case == "dtype":
        kv = kv.double()
    elif bad_case == "ids":
        ids = ids.float()
    elif bad_case == "value":
        value_dim = 25
    else:
        scale = float("inf")
    with pytest.raises(ValueError):
        reference_sparse_mla(q, kv, ids, scale, value_dim)
