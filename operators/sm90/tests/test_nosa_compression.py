"""Incremental compression, immutable pools, and append boundary correctness."""

import pytest
import torch

from models.nosa.scoring import compress_sequence
from operators.sm90.nosa_compression import update_compressed_cache

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def pooled_reference(cis, count):
    if not count:
        return cis[:0]
    return torch.stack([cis[max(0, 4 * b - 1) : 4 * b + 4].amax(0) for b in range(count)])


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_cuda_incremental_compression_matches_full_means_and_preserves_prefix(dtype, strided):
    generator = torch.Generator(device="cuda").manual_seed(374)
    length, heads, dim = 4217, 2, 64
    keys = torch.randn((length, heads, dim), generator=generator, device="cuda", dtype=dtype)
    cis = torch.randn((length, heads), generator=generator, device="cuda", dtype=dtype)
    if strided:
        # Real logical rows are finite; unused storage must not enter means.
        storage = torch.full((length * 2, heads * 2, dim), torch.nan, device="cuda", dtype=dtype)
        storage[::2, ::2] = keys
        keys = storage[::2, ::2]
        storage_cis = torch.full((length * 2, heads * 2), torch.nan, device="cuda", dtype=dtype)
        storage_cis[::2, ::2] = cis
        cis = storage_cis[::2, ::2]
    count, stable = length // 16 - 1, (length - 16) // 64
    ck = torch.full((count, heads, dim), torch.nan, device="cuda", dtype=dtype)
    cc = torch.full((count, heads), torch.nan, device="cuda", dtype=dtype)
    pool = torch.full((stable, heads), torch.nan, device="cuda", dtype=dtype)
    old_c = old_s = 0
    for end in (1, 31, 32, 47, 48, 63, 64, 79, 80, 95, 96, 129, 4095, 4096, 4097, 4112, length):
        before = ck[:old_c].clone(), cc[:old_c].clone(), pool[:old_s].clone()
        update_compressed_cache(
            keys[:end],
            cis[:end],
            ck,
            cc,
            pool,
            compressed_start=old_c,
            pooled_start=old_s,
        )
        for actual, expected in zip((ck[:old_c], cc[:old_c], pool[:old_s]), before, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        old_c, old_s = max(0, end // 16 - 1), max(0, (end - 16) // 64)
        expected_k, expected_c = compress_sequence(keys[:end]), compress_sequence(cis[:end])
        torch.testing.assert_close(ck[:old_c], expected_k)
        torch.testing.assert_close(cc[:old_c], expected_c)
        # Pool must use already rounded compressed values.
        torch.testing.assert_close(
            pool[:old_s], pooled_reference(cc[:old_c], old_s), atol=0, rtol=0
        )
    full_k, full_c, full_pool = torch.empty_like(ck), torch.empty_like(cc), torch.empty_like(pool)
    update_compressed_cache(
        keys, cis, full_k, full_c, full_pool, compressed_start=0, pooled_start=0
    )
    for incremental, full in zip((ck, cc, pool), (full_k, full_c, full_pool), strict=True):
        torch.testing.assert_close(incremental, full, atol=0, rtol=0)


@requires_cuda
@torch.inference_mode()
def test_cuda_compression_can_catch_up_pools_without_rewriting_compressed_prefix():
    keys = torch.randn((65553, 2, 128), device="cuda", dtype=torch.bfloat16)
    cis = torch.randn((len(keys), 2), device="cuda", dtype=keys.dtype)
    ck = compress_sequence(keys).contiguous()
    cc = compress_sequence(cis).contiguous()
    pool = torch.empty(((len(keys) - 16) // 64, 2), device="cuda", dtype=keys.dtype)
    before = ck.clone(), cc.clone()
    update_compressed_cache(keys, cis, ck, cc, pool, compressed_start=len(ck), pooled_start=0)
    torch.testing.assert_close(ck, before[0], atol=0, rtol=0)
    torch.testing.assert_close(cc, before[1], atol=0, rtol=0)
    torch.testing.assert_close(pool, pooled_reference(cc, len(pool)), atol=0, rtol=0)
