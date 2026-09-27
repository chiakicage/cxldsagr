"""Finite-input reduction on logical strided NOSA tensor views."""

import pytest
import torch

from operators.sm90.nosa_validation import all_finite

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_cuda_nosa_finite_scan_matches_torch_without_mutation(dtype, strided):
    generator = torch.Generator(device="cuda").manual_seed(891)
    q = torch.randn((7, 32, 128), generator=generator, device="cuda", dtype=dtype)
    keys = torch.randn((66559, 2, 128), generator=generator, device="cuda", dtype=dtype)
    cis = torch.randn((len(keys), 2), generator=generator, device="cuda", dtype=dtype)
    if strided:
        # NaN storage holes are not logical input elements. Slice every axis,
        # including the feature dimension, without copying the selected views.
        views = []
        for value in (q, keys, cis):
            storage = torch.full(
                tuple(2 * size for size in value.shape), torch.nan, device="cuda", dtype=dtype
            )
            view = storage[tuple(slice(None, None, 2) for _ in value.shape)]
            view.copy_(value)
            views.append(view)
        q, keys, cis = views
    before = [tensor.clone() for tensor in (q, keys, cis)]
    finite = all_finite(q, keys, cis)
    assert finite.shape == () and finite.dtype == torch.bool and finite.device == q.device
    assert finite.item()
    for tensor, original in zip((q, keys, cis), before, strict=True):
        torch.testing.assert_close(tensor, original, atol=0, rtol=0)
    for tensor in (q, keys, cis):
        index = tuple(size - 1 for size in tensor.shape)
        original = tensor[index].clone()
        for bad in (torch.nan, torch.inf, -torch.inf):
            tensor[index] = bad
            expected = (
                torch.isfinite(q).all() & torch.isfinite(keys).all() & torch.isfinite(cis).all()
            )
            torch.testing.assert_close(all_finite(q, keys, cis), expected, atol=0, rtol=0)
        tensor[index] = original
    assert all_finite(q, keys, cis).item()


@requires_cuda
@pytest.mark.parametrize("length", [1, 31, 32, 33, 47, 48])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_cuda_nosa_finite_scan_accepts_extrema_and_checks_uncompressed_tail(length, dtype):
    q = torch.zeros((1, 4, 64), device="cuda", dtype=dtype)
    keys = torch.zeros((length, 2, 64), device="cuda", dtype=dtype)
    cis = torch.zeros((length, 2), device="cuda", dtype=dtype)
    limits = torch.finfo(dtype)
    q[0, 0, :4] = torch.tensor(
        [limits.max, limits.min, limits.tiny, -0.0], device="cuda", dtype=dtype
    )
    assert all_finite(q, keys, cis).item()
    for tensor in (keys, cis):
        index = tuple(size - 1 for size in tensor.shape)
        tensor[index] = torch.nan
        assert not all_finite(q, keys, cis).item()
        tensor[index] = 0


def test_nosa_finite_scan_rejects_cpu():
    with pytest.raises(ValueError, match="CUDA"):
        all_finite(torch.zeros((1, 4, 64)), torch.zeros((1, 2, 64)), torch.zeros((1, 2)))


@requires_cuda
@torch.inference_mode()
def test_cuda_nosa_finite_scan_accepts_an_already_validated_key_prefix():
    # Exceeds 1024 partial CTAs, so the last element is reached by a later
    # grid-stride iteration while the final reduction remains bounded.
    q = torch.zeros((8193, 16, 64), device="cuda", dtype=torch.bfloat16)
    keys = torch.empty((0, 2, 64), device=q.device, dtype=q.dtype)
    cis = torch.empty((0, 2), device=q.device, dtype=q.dtype)
    assert all_finite(q, keys, cis).item()
    q[-1, -1, -1] = torch.inf
    assert not all_finite(q, keys, cis).item()
