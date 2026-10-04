"""Official FlashMLA adapter correctness; these are not performance experiments."""

import subprocess
import sys

import pytest
import torch

from operators.deepseek_v32.attention.device_only.mla import sparse_mla
from operators.deepseek_v32.attention.reference.torch import reference_sparse_mla


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("DeepSeek FlashMLA checks require SM90/Hopper")


def test_kernel_explicitly_requires_hopper():
    with pytest.raises(NotImplementedError, match="SM90"):
        sparse_mla(
            torch.zeros(1, 128, 576, dtype=torch.bfloat16),
            torch.zeros(5, 576, dtype=torch.bfloat16),
            torch.zeros(1, 4).int(),
            1,
        )


def test_reference_and_adapter_imports_do_not_load_native_backends():
    script = """
import sys
sys.modules["flash_mla"] = None
sys.modules["triton"] = None
from operators.deepseek_v32.attention.device_only.mla import sparse_mla
from operators.deepseek_v32.attention.reference.torch import reference_sparse_mla
import torch
q = torch.zeros(1, 2, 16)
kv = torch.ones(2, 16)
ids = torch.tensor([[0, 1]])
assert torch.equal(reference_sparse_mla(q, kv, ids, 1, 16), torch.ones_like(q))
try:
    sparse_mla(q, kv, ids, 1, 16)
except NotImplementedError as exc:
    assert "SM90" in str(exc)
else:
    raise AssertionError("CPU must not dispatch a native attention backend")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    ("heads", "queries", "selected"),
    [(64, 3, 1), (64, 3, 127), (128, 3, 129), (128, 3, 2048), (128, 65, 137)],
)
def test_cuda_matches_fp32_reference_with_strides_masks_and_duplicates(heads, queries, selected):
    require_sm90()
    torch.manual_seed(427)
    q = torch.randn(queries, heads * 2, 1152, device="cuda", dtype=torch.bfloat16)[:, ::2, ::2]
    kv = torch.randn(846, 1152, device="cuda", dtype=torch.bfloat16)[::2, ::2]
    ids = torch.randint(1, len(kv), (queries, selected * 2), device="cuda", dtype=torch.int32)[
        :, ::2
    ]
    kv[0] = float("nan")
    ids[0] = -1
    ids[1, ::7] = len(kv) + 10
    if selected >= 3:
        ids[2, :3] = torch.tensor([1, 1, 2], device="cuda", dtype=torch.int32)
    expected = reference_sparse_mla(q, kv, ids, 576**-0.5)
    actual = sparse_mla(q, kv, ids, 576**-0.5)
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=2e-2)
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    # Exact offload query splitting preserves this same per-query reduction.
    split = torch.cat(
        [
            sparse_mla(q[start : start + 16], kv, ids[start : start + 16], 576**-0.5)
            for start in range(0, len(q), 16)
        ]
    )
    torch.testing.assert_close(actual, split, atol=0, rtol=0)


def test_cuda_int64_invalid_ids_cannot_wrap_into_valid_records():
    require_sm90()
    q = torch.zeros(2, 64, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((4, 576), 7, device="cuda", dtype=torch.bfloat16)
    kv[0] = float("nan")
    kv[1, :512] = 2
    ids = torch.full((2, 129), 2**32 + 1, device="cuda", dtype=torch.int64)
    ids[0, 1] = -(2**32) + 1
    ids[1, 0] = 1
    actual = sparse_mla(q, kv, ids, 1)
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    torch.testing.assert_close(actual[1], torch.full_like(actual[1], 2), atol=0, rtol=0)


def test_cuda_contiguous_storage_offsets_are_aligned_for_native_loads():
    require_sm90()
    torch.manual_seed(294)
    q = torch.randn(3 * 128 * 576 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(3, 128, 576)
    kv = torch.randn(17 * 576 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(17, 576)
    assert q.is_contiguous() and kv.is_contiguous()
    assert q.data_ptr() % 16 and kv.data_ptr() % 16
    ids = torch.randint(0, len(kv), (3, 129), device="cuda", dtype=torch.int32)
    expected = reference_sparse_mla(q, kv, ids, 576**-0.5)
    actual = sparse_mla(q, kv, ids, 576**-0.5)
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=2e-2)


def test_cuda_empty_tiles_and_large_logits_keep_online_softmax_stable():
    require_sm90()
    q = torch.zeros(1, 64, 576, device="cuda", dtype=torch.bfloat16)
    q[..., 575] = 100
    kv = torch.zeros(3, 576, device="cuda", dtype=torch.bfloat16)
    kv[0] = float("nan")
    kv[1, :512], kv[1, 575] = 2, 100
    kv[2, :512], kv[2, 575] = 5, -100
    ids = torch.full((1, 193), -1, device="cuda", dtype=torch.int32)
    ids[0, 65:67] = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    actual = sparse_mla(q, kv, ids, 1)
    torch.testing.assert_close(actual, torch.full_like(actual, 2), atol=0, rtol=0)


@pytest.mark.parametrize(
    ("heads", "dimension", "value_dim", "dtype", "message"),
    [
        (128, 576, 512, torch.float16, "BF16"),
        (128, 576, 512, torch.float32, "BF16"),
        (96, 576, 512, torch.bfloat16, "H=64 or 128"),
        (128, 512, 512, torch.bfloat16, "dimension 576"),
        (128, 576, 448, torch.bfloat16, "V=512"),
    ],
)
def test_cuda_unsupported_layouts_and_dtypes_fail_explicitly(
    heads, dimension, value_dim, dtype, message
):
    require_sm90()
    q = torch.zeros(1, heads, dimension, device="cuda", dtype=dtype)
    kv = torch.zeros(5, dimension, device="cuda", dtype=dtype)
    ids = torch.zeros(1, 4, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match=message):
        sparse_mla(q, kv, ids, 1, value_dim)


@pytest.mark.parametrize(("queries", "tokens", "selected"), [(0, 5, 3), (2, 0, 3), (2, 5, 0)])
def test_cuda_empty_inputs(queries, tokens, selected):
    require_sm90()
    q = torch.empty(queries, 128, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.empty(tokens, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.full((queries, selected), -1, device="cuda", dtype=torch.int32)
    actual = sparse_mla(q, kv, indices, 0.2)
    assert actual.shape == (queries, 128, 512)
    assert torch.equal(actual, torch.zeros_like(actual))
