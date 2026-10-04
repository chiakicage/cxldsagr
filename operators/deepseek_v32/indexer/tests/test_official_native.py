"""ABI and source-isolation checks for ECHO's untouched native kernels."""

import pytest
import torch

from operators.deepseek_v32.indexer.official import module, topk_module


def _require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; explicit GPU regression requires Hopper")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.fail("Official ECHO native validation requires Hopper SM90")


def test_official_fused_logits_transport_and_mainline_coexistence():
    _require_sm90()
    import deep_gemm

    mainline_file = deep_gemm.__file__
    native = module()
    torch.manual_seed(17)
    history, queries, slots = 4096, 16, 8192
    tokens = history + queries
    missing = torch.iinfo(torch.int32).max
    q = torch.randn((queries, 64, 128), device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn((tokens, 128), device="cuda").to(torch.float8_e4m3fn)
    scale = torch.ones(tokens, device="cuda")
    weights = torch.randn((queries, 64), device="cuda")
    starts = torch.zeros(queries, device="cuda", dtype=torch.int32)
    ends = torch.arange(history + 1, tokens + 1, device="cuda", dtype=torch.int32)
    host = torch.empty((tokens, 1, 576), pin_memory=True, dtype=torch.bfloat16).normal_()
    device = torch.zeros((slots + 1, 1, 576), device="cuda", dtype=torch.bfloat16)
    device_to_host = torch.full((slots + 1,), missing, device="cuda", dtype=torch.int64)
    host_to_device = torch.full((tokens + 1,), missing, device="cuda", dtype=torch.int32)
    priority = torch.zeros(slots + 1, device="cuda", dtype=torch.int32)
    priority[0] = missing
    counter = torch.zeros(1, device="cuda", dtype=torch.uint32)
    logits = native.fp8_mqa_logits_fuse_prefetch(
        q,
        (k, scale),
        weights,
        starts,
        ends,
        page_table_1=torch.arange(tokens, device="cuda", dtype=torch.int32)[None],
        extend_seq_lens=torch.full((1,), queries, device="cuda", dtype=torch.int32),
        extend_seq_to_req=torch.zeros(queries, device="cuda", dtype=torch.int32),
        device_pool_buf=device,
        host_pool_buf=host,
        device_pool_loc_alloc_buf=torch.zeros(slots, device="cuda", dtype=torch.int32),
        device_pool_priority=priority,
        device_pool_loc_small_priority=torch.arange(1, slots + 1, device="cuda", dtype=torch.int32),
        device_token_to_host=device_to_host,
        host_token_to_device=host_to_device,
        recall_counter=counter,
        extend_logits_offsets=torch.zeros(16, device="cuda"),
        clean_logits=True,
    )
    torch.cuda.synchronize()
    valid = torch.arange(tokens, device="cuda")[None] < ends[:, None]
    reference = native.fp8_mqa_logits(q, (k, scale), weights, starts, ends, True)
    torch.testing.assert_close(logits[valid], reference[valid], atol=0, rtol=0)
    recalled_slots = torch.where(device_to_host < missing)[0]
    recalled_hosts = device_to_host[recalled_slots]
    assert 0 < len(recalled_slots) <= history
    assert bool((recalled_hosts < history).all())
    assert recalled_hosts.unique().numel() == recalled_hosts.numel()
    assert torch.equal(host_to_device[recalled_hosts], recalled_slots.int())
    torch.testing.assert_close(
        device[recalled_slots].cpu(), host[recalled_hosts.cpu()], atol=0, rtol=0
    )
    mainline = deep_gemm.fp8_fp4_mqa_logits(
        (q, None), (k, scale), weights, starts, ends, max_seqlen_k=tokens
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(logits[valid], mainline[valid], atol=0, rtol=0)
    assert deep_gemm.__file__ == mainline_file


def test_official_bounded_argmin_preserves_priority_values_and_sentinels():
    _require_sm90()
    torch.manual_seed(23)
    score = torch.randint(0, 10_000_001, (65537,), device="cuda", dtype=torch.int32)
    score[:1000] = torch.iinfo(torch.int32).max
    score[1000:4000] = 0
    count = 4096
    output = torch.full_like(score, -1)
    topk_module().fast_argmin_bounded(
        score, output, torch.tensor([count], device="cuda", dtype=torch.int32)
    )
    torch.cuda.synchronize()
    selected = output[:count].long()
    assert selected.unique().numel() == count
    expected = torch.topk(score, count, largest=False).values.sort().values
    actual = score[selected].sort().values
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_official_exact_topk_uses_lengths_with_padded_logit_stride():
    _require_sm90()
    torch.manual_seed(31)
    score = torch.randn((4, 8192 + 128), device="cuda")[:, :8192]
    lengths = torch.tensor([10, 2048, 4097, 8192], device="cuda", dtype=torch.int32)
    selected = torch.empty((4, 2048), device="cuda", dtype=torch.int32)
    topk_module().fast_topk(score, selected, lengths)
    torch.cuda.synchronize()
    for row, length in enumerate(lengths.tolist()):
        count = min(length, 2048)
        positions = selected[row, :count].long()
        assert positions.unique().numel() == count
        assert bool(((positions >= 0) & (positions < length)).all())
        assert bool((selected[row, count:] == -1).all())
        expected = torch.topk(score[row, :length], count).values.sort().values
        actual = score[row, positions].sort().values
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
