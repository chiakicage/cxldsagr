"""Independent mathematical checks for NOSA query-aware block selection."""

import math
from dataclasses import asdict

import pytest
import torch

from layers.attention import AttentionContext, ResidentLayerView
from models.nosa.indexer import NosaIndexer, NosaSelectionPolicy


def select(q, keys, query_start, *, query_chunk_size=64, block_budget=64):
    return NosaIndexer(query_chunk_size, block_budget=block_budget)(
        q,
        ResidentLayerView(3, keys=keys),
        AttentionContext(3, query_start, len(q)),
    )


def mathematical_oracle(q, keys, query_start, *, block_budget=64):
    """Use per-head distributions and token-interval overlap, without pooling kernels.

    Only fully causal 32-token intervals are constructed for each query. An
    interval contributes to every 64-token block it intersects. The oracle
    independently expresses the semantics in token coordinates, rather than
    using the implementation's compressed-index gather or vectorized masks.
    """
    q, keys = q.float().cpu(), keys.float().cpu()
    kv_heads = keys.shape[1]
    group_size = q.shape[1] // kv_heads
    result = torch.full((len(q), kv_heads, block_budget), -1, dtype=torch.long)
    for row, position in enumerate(range(query_start, query_start + len(q))):
        window_starts = list(range(0, position + 2 - 32, 16))
        compressed = [keys[start : start + 32].sum(dim=0) / 32 for start in window_starts]
        last_block = position // 64
        mandatory = {0} | set(range(max(0, last_block - 15), last_block + 1))
        for head in range(kv_heads):
            scores = torch.zeros(len(compressed), dtype=torch.float32)
            for query_head in range(head * group_size, (head + 1) * group_size):
                if compressed:
                    logits = torch.stack(
                        [torch.dot(q[row, query_head], key[head]) for key in compressed]
                    ) / math.sqrt(q.shape[-1])
                    scores += logits.softmax(dim=0)
            block_scores = {}
            for block in range(last_block + 1):
                if block in mandatory:
                    continue
                overlaps = [
                    index
                    for index, start in enumerate(window_starts)
                    if start < (block + 1) * 64 and start + 32 > block * 64
                ]
                block_scores[block] = max((scores[index].item() for index in overlaps), default=0)
            dynamic = sorted(block_scores, key=lambda block: (-block_scores[block], block))[
                : block_budget - 17
            ]
            chosen = sorted(mandatory | set(dynamic))
            result[row, head, : len(chosen)] = torch.tensor(chosen)
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_budget", [32, 64])
def test_query_aware_selection_matches_independent_gqa_oracle(dtype, block_budget):
    generator = torch.Generator().manual_seed(712)
    # A strided layout like the model's packed Q/K/V projection views.
    q = torch.randn((7, 4, 16), generator=generator).to(dtype)[..., ::2]
    q = q * torch.tensor([0.2, 5.0, 1.0, 3.0], dtype=dtype)[None, :, None]
    keys = torch.randn((75 * 64 + 19, 2, 16), generator=generator).to(dtype)[..., ::2]
    start = len(keys) - len(q)
    actual = select(q, keys, start, query_chunk_size=3, block_budget=block_budget)
    expected = mathematical_oracle(q, keys, start, block_budget=block_budget)
    assert actual.block_size == 64
    assert actual.block_ids.dtype == torch.long
    assert actual.valid_mask.dtype == torch.bool
    torch.testing.assert_close(actual.block_ids, expected, rtol=0, atol=0)
    assert actual.valid_mask.all()
    # The two KV heads carry separate distributions and selections.
    assert not torch.equal(actual.block_ids[:, 0], actual.block_ids[:, 1])


@pytest.mark.parametrize("start", [4095, 4111, 4127, 4159])
@pytest.mark.parametrize("block_budget", [32, 64])
def test_compression_and_local_boundaries_use_absolute_causal_positions(start, block_budget):
    generator = torch.Generator().manual_seed(129)
    q = torch.randn((2, 4, 8), generator=generator)
    keys = torch.randn((start + len(q), 2, 8), generator=generator)
    actual = select(q, keys, start, block_budget=block_budget)
    expected = mathematical_oracle(q, keys, start, block_budget=block_budget)
    torch.testing.assert_close(actual.block_ids, expected, rtol=0, atol=0)


@pytest.mark.parametrize("block_budget", [32, 64])
def test_ties_choose_smallest_ids_and_local_is_exactly_sixteen_aligned_blocks(block_budget):
    q = torch.zeros((64, 4, 8))
    keys = torch.zeros((81 * 64, 2, 8))
    actual = select(q, keys, 80 * 64, block_budget=block_budget)
    expected = torch.tensor([*range(block_budget - 16), *range(65, 81)]).expand(64, 2, block_budget)
    torch.testing.assert_close(actual.block_ids, expected, rtol=0, atol=0)
    assert actual.valid_mask.all()
    # Block 64 would appear if "16 local" meant 16 previous plus the current.
    assert not (actual.block_ids == 64).any()


@pytest.mark.parametrize(
    ("block_budget", "length"),
    [
        (budget, length)
        for budget in (32, 64)
        for length in (1, 31, 32, 63, 64, 65, 1024, 1025, budget * 64)
    ],
)
def test_short_context_selects_all_visible_blocks_and_pads(length, block_budget):
    q = torch.zeros((1, 4, 8))
    keys = torch.zeros((length, 2, 8))
    actual = select(q, keys, length - 1, block_budget=block_budget)
    visible = (length + 63) // 64
    expected = torch.full((1, 2, block_budget), -1, dtype=torch.long)
    expected[:, :, :visible] = torch.arange(visible)
    torch.testing.assert_close(actual.block_ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.valid_mask, expected >= 0, rtol=0, atol=0)


@pytest.mark.parametrize("block_budget", [32, 64])
def test_chunking_and_later_k_do_not_change_earlier_selection_or_cache_state(block_budget):
    generator = torch.Generator().manual_seed(512)
    start = 81 * 64 + 13
    q = torch.randn((35, 4, 8), generator=generator)
    keys = torch.randn((start + len(q) + 20, 2, 8), generator=generator)
    keys_before = keys.clone()
    cache = ResidentLayerView(3, keys=keys)
    state = object()
    cache.set_layer_state(3, state)
    context = AttentionContext(3, start, len(q), auxiliary_state=state)
    full = NosaIndexer(block_budget=block_budget)(q, cache, context)
    singleton_chunks = NosaIndexer(query_chunk_size=1, block_budget=block_budget)(q, cache, context)
    torch.testing.assert_close(full.block_ids, singleton_chunks.block_ids, rtol=0, atol=0)
    # Individually truncating the resident view is an independent causal check:
    # the remaining queries were physically absent from each such selection.
    for row in [0, 2, 18, 34]:
        one = select(
            q[row : row + 1], keys[: start + row + 1], start + row, block_budget=block_budget
        )
        torch.testing.assert_close(full.block_ids[row : row + 1], one.block_ids, rtol=0, atol=0)
    assert cache.get_layer_state(3) is state
    torch.testing.assert_close(keys, keys_before, rtol=0, atol=0)


@pytest.mark.parametrize("block_budget", [32, 64])
def test_cpu_autocast_does_not_lower_reference_scoring_precision(block_budget):
    generator = torch.Generator().manual_seed(127)
    keys = torch.randn((80 * 64, 2, 8), generator=generator)
    q = torch.randn((3, 4, 8), generator=generator)
    expected = mathematical_oracle(q, keys, len(keys) - len(q), block_budget=block_budget)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = select(q, keys, len(keys) - len(q), block_budget=block_budget)
    torch.testing.assert_close(actual.block_ids, expected, rtol=0, atol=0)


def test_nonresident_cache_fails_explicitly():
    with pytest.raises(NotImplementedError, match="resident"):
        NosaIndexer()(torch.zeros((1, 4, 8)), object(), AttentionContext(0, 0, 1))


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_chunk_size_must_be_a_positive_integer(chunk_size):
    with pytest.raises(ValueError, match="positive integer"):
        NosaIndexer(chunk_size)


@pytest.mark.parametrize(
    ("mode", "backend", "device", "expected"),
    [
        ("query_aware", "reference", "cpu", 64),
        ("query_aware", "reference", "cuda", 64),
        ("nosa", "reference", "cpu", 64),
        ("nosa", "reference", "cuda", 64),
        ("nosa", "auto", "cpu", 64),
        ("nosa", "auto", "cuda", None),
        ("nosa", "triton", "cuda", None),
    ],
)
def test_default_query_chunk_size_follows_execution_backend(mode, backend, device, expected):
    indexer = NosaIndexer(mode=mode, backend=backend)
    assert indexer.effective_query_chunk_size(torch.device(device)) == expected
    for size in (1, 64, 1536):
        explicit = NosaIndexer(size, mode=mode, backend=backend)
        assert explicit.effective_query_chunk_size(torch.device(device)) == (
            None if expected is None else size
        )
        if expected is None:
            assert explicit.effective_query_chunk_size(torch.device(device), 8193) == 8193


def test_budget_derives_dynamic_topk_and_preserves_default_policy_fields():
    assert asdict(NosaIndexer().policy) == asdict(NosaSelectionPolicy())
    smaller = asdict(NosaIndexer(block_budget=32).policy)
    expected = asdict(NosaSelectionPolicy()) | {"block_budget": 32, "topk_blocks": 15}
    assert smaller == expected


@pytest.mark.parametrize("budget", [0, 16, 17, 31, 33, 63, 65, 32.0, True, None])
def test_unsupported_or_noninteger_budgets_fail(budget):
    with pytest.raises(ValueError, match="block_budget"):
        NosaIndexer(block_budget=budget)


def test_32_budget_is_a_per_query_subset_not_truncation_of_sorted_64_ids():
    generator = torch.Generator().manual_seed(817)
    start = 80 * 64 - 5
    q = torch.randn((12, 4, 8), generator=generator)
    keys = torch.randn((start + len(q), 2, 8), generator=generator)
    smaller = select(q, keys, start, block_budget=32)
    larger = select(q, keys, start, block_budget=64)
    expected = mathematical_oracle(q, keys, start, block_budget=32)
    torch.testing.assert_close(smaller.block_ids, expected, rtol=0, atol=0)
    assert smaller.valid_mask.all() and larger.valid_mask.all()
    for row, position in enumerate(range(start, start + len(q))):
        mandatory = {0} | set(range(position // 64 - 15, position // 64 + 1))
        for head in range(keys.shape[1]):
            selected = set(smaller.block_ids[row, head].tolist())
            assert mandatory <= selected <= set(larger.block_ids[row, head].tolist())
            # Sorted IDs no longer carry score rank: truncating them drops the
            # mandatory local window and picks the wrong query-aware blocks.
            assert selected != set(larger.block_ids[row, head, :32].tolist())


@pytest.mark.parametrize(
    ("q_shape", "key_length", "start", "query_length", "message"),
    [
        ((1, 3, 8), 1, 0, 1, "KV groups"),
        ((1, 4, 7), 1, 0, 1, "head dimensions"),
        ((1, 4, 8), 1, -1, 1, "query_start"),
        ((1, 4, 8), 1, 0, 2, "query_length"),
        ((1, 4, 8), 1, 1, 1, "cover the query positions"),
    ],
)
def test_invalid_query_metadata_fails(q_shape, key_length, start, query_length, message):
    with pytest.raises(ValueError, match=message):
        NosaIndexer()(
            torch.zeros(q_shape),
            ResidentLayerView(0, keys=torch.zeros((key_length, 2, 8))),
            AttentionContext(0, start, query_length),
        )


@pytest.mark.parametrize("value", [torch.nan, torch.inf])
def test_nonfinite_scores_are_not_silently_ranked(value):
    q = torch.zeros((1, 4, 8))
    q[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        select(q, torch.zeros((1, 2, 8)), 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("block_budget", [32, 64])
def test_cuda_reference_matches_cpu_oracle_and_restores_tf32(block_budget):
    generator = torch.Generator().manual_seed(127)
    keys = torch.randn((80 * 64, 2, 8), generator=generator)
    q = torch.randn((3, 4, 8), generator=generator)
    expected = mathematical_oracle(q, keys, len(keys) - len(q), block_budget=block_budget)
    original = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actual = select(q.cuda(), keys.cuda(), len(keys) - len(q), block_budget=block_budget)
        assert torch.backends.cuda.matmul.fp32_precision == "tf32"
        torch.testing.assert_close(actual.block_ids.cpu(), expected, rtol=0, atol=0)
    finally:
        torch.backends.cuda.matmul.fp32_precision = original
