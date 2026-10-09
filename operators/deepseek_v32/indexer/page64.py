"""Byte-preserving adaptation to the official page64 index-key ABI."""

from functools import cache

import torch


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def pack_page64(Keys, Scales, Packed, TOKENS: tl.constexpr):
        page = tl.program_id(0)
        key_word = tl.arange(0, 2048)
        token = page * 64 + key_word // 32
        key = tl.load(Keys + page * 2048 + key_word, token < TOKENS, 0)
        tl.store(Packed + page * 2112 + key_word, key)
        scale_word = tl.arange(0, 64)
        scale = tl.load(Scales + page * 64 + scale_word, page * 64 + scale_word < TOKENS, 0)
        tl.store(Packed + page * 2112 + 2048 + scale_word, scale)

    return pack_page64


def pack_q1_keys(k, scales):
    """Pack validated contiguous K[N,128]/scale[N], zeroing page padding."""
    pages = (len(k) + 63) // 64
    packed = torch.empty((pages, 8448), dtype=torch.uint8, device=k.device)
    _kernel()[(pages,)](
        k.view(torch.int32),
        scales.view(torch.int32),
        packed.view(torch.int32),
        TOKENS=len(k),
        num_warps=4,
    )
    return packed
