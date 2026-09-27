"""Hopper NOSA score and deterministic two-stage selector correctness."""

import pytest
import torch

from layers.attention import AttentionContext, ResidentLayerView
from models.nosa.indexer import NosaIndexer, compressed_scores_reference

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def rank_oracle(scores, cis, positions, length):
    scores, cis, positions = scores.cpu(), cis.cpu(), positions.cpu()
    result = torch.full((*scores.shape[:2], min(64, (length + 63) // 64)), -1, dtype=torch.long)
    for row, position in enumerate(positions.tolist()):
        last = position // 64
        mandatory = {0} | set(range(max(0, last - 16), last + 1))
        for head in range(scores.shape[1]):
            query_rank, cis_rank = {}, {}
            for block in range(last + 1):
                overlap = [c for c in range(max(0, block * 4 - 1), min(len(cis), block * 4 + 4))]
                query_rank[block] = max(
                    (float(scores[row, head, c]) for c in overlap), default=-float("inf")
                )
                cis_rank[block] = max((float(cis[c, head]) for c in overlap), default=-float("inf"))
            first = mandatory | set(
                sorted(set(range(last + 1)) - mandatory, key=lambda b: (-query_rank[b], b))[
                    : max(0, 33 - len(mandatory))
                ]
            )
            final = sorted(
                first
                | set(
                    sorted(set(range(last + 1)) - first, key=lambda b: (-cis_rank[b], b))[
                        : max(0, 64 - len(first))
                    ]
                )
            )
            result[row, head, : len(final)] = torch.tensor(final)
    return result


@requires_cuda
@pytest.mark.parametrize("blocks", [65, 129, 4097])
@torch.inference_mode()
def test_cuda_nosa_selection_matches_oracle_including_ties_and_large_context(blocks):
    from operators.sm90.nosa_indexer import select_from_scores

    length = blocks * 64 - 7
    count = (length - 32) // 16 + 1
    generator = torch.Generator().manual_seed(26)
    # Rounded integer scores exercise ties and the signed composite ranking
    # keys. The long case takes the split pool/promote/partial-topk path.
    scores = torch.randint(-3, 4, (2, 2, count), generator=generator).float()
    cis = torch.randint(-8, 1, (count, 2), generator=generator).float()
    positions = torch.tensor([0, length - 1])
    expected = rank_oracle(scores, cis, positions, length)
    actual = select_from_scores(scores.cuda(), cis.cuda(), positions.cuda(), length)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize(
    ("dtype", "groups", "dimension"), [(torch.bfloat16, 4, 64), (torch.float16, 16, 128)]
)
@torch.inference_mode()
def test_cuda_nosa_compressed_scores_match_fp32_gqa_reference(dtype, groups, dimension):
    from operators.sm90.nosa_indexer import compressed_scores

    generator = torch.Generator(device="cuda").manual_seed(137)
    query = torch.randn((5, 2, groups, dimension), generator=generator, device="cuda", dtype=dtype)
    keys = torch.randn((289, 2, dimension), generator=generator, device="cuda", dtype=dtype)
    positions = torch.tensor([0, 31, 1023, 4096, 4639], device="cuda")
    expected = compressed_scores_reference(query, keys, positions)
    actual = compressed_scores(query, keys, positions)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.012, atol=0.001)
    assert torch.isfinite(actual).all()
    assert actual[0].count_nonzero() == 0


@requires_cuda
@torch.inference_mode()
def test_cuda_full_nosa_indexer_preserves_33_64_policy_on_exact_score_ties():
    start = 100 * 64
    q = torch.zeros((3, 4, 64), device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros((start + len(q), 2, 64), device="cuda", dtype=torch.bfloat16)
    cis = (
        torch.arange(len(keys), device="cuda", dtype=torch.float32)[:, None]
        .expand(-1, 2)
        .to(torch.bfloat16)
    )
    cache = ResidentLayerView(0, keys=keys, cis_scores=cis)
    context = AttentionContext(0, start, len(q))
    reference = NosaIndexer(mode="nosa")(q, cache, context)
    actual = NosaIndexer(mode="nosa", backend="triton")(q, cache, context)
    torch.testing.assert_close(actual.block_ids, reference.block_ids, atol=0, rtol=0)
    assert actual.valid_mask.all()


@requires_cuda
@pytest.mark.parametrize("blocks", [129, 4097])
@torch.inference_mode()
def test_cuda_nosa_launches_on_tensor_device_and_preserves_current_device(blocks):
    from operators.sm90.nosa_indexer import compressed_scores, select_from_scores

    original = torch.cuda.current_device()
    # A single-GPU machine still checks restoration; two visible devices
    # additionally exercise launch on a GPU other than the current one.
    target = torch.device("cuda", (original + 1) % torch.cuda.device_count())
    length = blocks * 64 - 7
    count = (length - 32) // 16 + 1
    query = torch.zeros((1, 1, 2, 64), dtype=torch.bfloat16, device=target)
    keys = torch.zeros((count, 1, 64), dtype=query.dtype, device=target)
    positions = torch.tensor([length - 1], device=target)
    cis = -torch.arange(count, dtype=torch.float32, device=target)[:, None]
    expected_scores = compressed_scores_reference(query, keys, positions)
    scores = compressed_scores(query, keys, positions)
    assert torch.cuda.current_device() == original
    assert scores.device == target
    torch.testing.assert_close(scores, expected_scores)
    expected = rank_oracle(scores, cis, positions, length)
    actual = select_from_scores(scores, cis, positions, length)
    assert torch.cuda.current_device() == original
    assert actual.device == target
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
