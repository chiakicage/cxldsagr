"""Attention on exactly recalled, physically remapped HBM records."""

import pytest
import torch

from operators.deepseek_v32.attention.device_only.mla import sparse_mla
from operators.deepseek_v32.attention.offload.mla import sparse_mla_from_pool


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("DeepSeek sparse MLA checks require SM90/Hopper")


def test_cuda_physical_remapping_preserves_attention():
    require_sm90()
    torch.manual_seed(734)
    q = torch.randn(4, 19, 576, device="cuda", dtype=torch.bfloat16)
    resident = torch.randn(1031, 576, device="cuda", dtype=torch.bfloat16)
    logical = torch.randint(0, len(resident), (4, 129), device="cuda", dtype=torch.int32)
    logical[:, -4:] = -1
    unique = torch.unique(logical[logical >= 0]).long()
    physical = torch.searchsorted(unique, logical).int().masked_fill(logical < 0, -1)
    compact = resident[unique]
    resident_output = sparse_mla(q, resident, logical, 0.07)
    compact_output = sparse_mla_from_pool(q, compact, physical, 0.07)
    torch.testing.assert_close(compact_output, resident_output, atol=0, rtol=0)
