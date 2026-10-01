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
@pytest.mark.parametrize("blocks", [65, 129, 4096])
@torch.inference_mode()
def test_cuda_nosa_selection_matches_oracle_including_ties_and_large_context(blocks):
    from operators.nosa.indexer.api import select_from_scores

    length = blocks * 64 - 7
    count = (length - 32) // 16 + 1
    generator = torch.Generator().manual_seed(26)
    # Rounded integer scores exercise stable ties, including block counts
    # up to the full maximum model context.
    scores = torch.randint(-3, 4, (2, 2, count), generator=generator).float()
    cis = torch.randint(-8, 1, (count, 2), generator=generator).float()
    positions = torch.tensor([0, length - 1])
    expected = rank_oracle(scores, cis, positions, length)
    actual = select_from_scores(scores.cuda(), cis.cuda(), positions.cuda(), length)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize(
    ("dtype", "groups", "dimension", "rows"),
    [(torch.bfloat16, 4, 64, 5), (torch.float16, 16, 128, 5), (torch.bfloat16, 16, 128, 1024)],
)
@torch.inference_mode()
def test_cuda_nosa_compressed_scores_match_fp32_gqa_reference(dtype, groups, dimension, rows):
    from operators.nosa.indexer.api import compressed_scores

    generator = torch.Generator(device="cuda").manual_seed(137)
    query = torch.randn(
        (rows, 2, groups, dimension), generator=generator, device="cuda", dtype=dtype
    )
    keys = torch.randn((289, 2, dimension), generator=generator, device="cuda", dtype=dtype)
    positions = (
        torch.tensor([0, 31, 1023, 4096, 4639], device="cuda")
        if rows == 5
        else torch.linspace(0, 4639, rows, device="cuda", dtype=torch.int64)
    )
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
@pytest.mark.parametrize("blocks", [129, 4096])
@torch.inference_mode()
def test_cuda_nosa_launches_on_tensor_device_and_preserves_current_device(blocks):
    from operators.nosa.indexer.api import compressed_scores, select_from_scores

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


@requires_cuda
@pytest.mark.parametrize(
    ("length", "start", "rows"), [(66560, 65536, 7), (6403, 6396, 7), (65, 0, 5), (31, 26, 5)]
)
@torch.inference_mode()
def test_cuda_contiguous_nosa_selection_matches_positions_and_rank_oracle(length, start, rows):
    from operators.nosa.indexer.api import (
        compressed_scores,
        select_blocks,
        select_contiguous_blocks,
    )

    count = max(0, (length - 32) // 16 + 1)
    generator = torch.Generator(device="cuda").manual_seed(163)
    # Preserve the contiguous feature dimension while exercising strided
    # token/head views, as produced by packed model projections.
    query = torch.randn(
        (rows * 2, 4, 16, 128), generator=generator, device="cuda", dtype=torch.bfloat16
    )[::2, ::2]
    keys = torch.randn((count * 2, 4, 128), generator=generator, device="cuda", dtype=query.dtype)[
        ::2, ::2
    ]
    cis = torch.randn((count, 2), generator=generator, device="cuda", dtype=query.dtype)
    positions = torch.arange(start, start + rows, device="cuda")
    scores = compressed_scores(query, keys, positions)
    expected = rank_oracle(scores, cis, positions, length)
    by_position = select_blocks(query, keys, cis, positions, length)
    contiguous = select_contiguous_blocks(query, keys, cis, start, length)
    torch.testing.assert_close(by_position.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(contiguous.cpu(), expected, rtol=0, atol=0)


def test_contiguous_nosa_operator_rejects_cpu_even_for_short_context():
    from operators.nosa.indexer.api import select_contiguous_blocks

    with pytest.raises(ValueError, match="CUDA"):
        select_contiguous_blocks(
            torch.zeros((1, 2, 16, 128), dtype=torch.bfloat16),
            torch.zeros((0, 2, 128), dtype=torch.bfloat16),
            torch.zeros((0, 2), dtype=torch.bfloat16),
            0,
            1,
        )


@requires_cuda
@pytest.mark.parametrize("start", [-1, True, 1.5, None, 32])
@torch.inference_mode()
def test_cuda_contiguous_nosa_operator_rejects_invalid_or_out_of_range_start(start):
    from operators.nosa.indexer.api import select_contiguous_blocks

    with pytest.raises(ValueError, match="query_start|within total_length|query positions"):
        select_contiguous_blocks(
            torch.zeros((2, 2, 16, 128), device="cuda", dtype=torch.bfloat16),
            torch.zeros((1, 2, 128), device="cuda", dtype=torch.bfloat16),
            torch.zeros((1, 2), device="cuda", dtype=torch.bfloat16),
            start,
            33,
        )


@requires_cuda
@pytest.mark.parametrize("length", [65, 6403])
@pytest.mark.parametrize("invalid_position", [-1, "past_end"])
@torch.inference_mode()
def test_cuda_nosa_tensor_positions_remain_range_checked(length, invalid_position):
    from operators.nosa.indexer.api import select_blocks, select_from_scores

    count = (length - 32) // 16 + 1
    query = torch.zeros((2, 2, 16, 128), device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros((count, 2, 128), device="cuda", dtype=query.dtype)
    cis = torch.zeros((count, 2), device="cuda", dtype=query.dtype)
    scores = torch.zeros((2, 2, count), device="cuda", dtype=query.dtype)
    bad = length if invalid_position == "past_end" else invalid_position
    positions = torch.tensor([0, bad], device="cuda")
    with pytest.raises(ValueError, match="within total_length"):
        select_blocks(query, keys, cis, positions, length)
    with pytest.raises(ValueError, match="within total_length"):
        select_from_scores(scores, cis, positions, length)


def _pooled_reference(values, blocks):
    padded = torch.nn.functional.pad(
        values, (1, 4 * blocks - values.shape[-1]), value=-float("inf")
    )
    return padded.unfold(-1, 5, 4).amax(-1)


def _literal_selection_reference(scores, cis, positions, length):
    """Literal masked stable sorts also preserve natural infinity semantics."""
    scores, cis, positions = scores.cpu(), cis.cpu(), positions.cpu()
    blocks = (length + 63) // 64
    block = torch.arange(blocks)
    last = positions[:, None, None] // 64
    causal = block <= last
    mandatory = (block == 0) | (causal & (last <= block + 16))
    qa = _pooled_reference(scores.float(), blocks)
    shared = _pooled_reference(cis.T.float(), blocks)
    qa = qa.masked_fill(mandatory, float("inf")).masked_fill(~causal, -float("inf"))
    shared = shared[None].expand_as(qa).clone()
    shared.masked_fill_(mandatory, float("inf")).masked_fill_(~causal, -float("inf"))
    first = qa.argsort(dim=-1, descending=True, stable=True)[..., :33]
    promoted = torch.where(first <= last, float("inf"), -float("inf"))
    shared.scatter_(-1, first, promoted)
    selected = shared.argsort(dim=-1, descending=True, stable=True)[..., :64].sort(-1).values
    return selected.masked_fill(selected > last, -1)


@requires_cuda
@pytest.mark.parametrize(
    ("score_dtype", "cis_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.bfloat16),
    ],
)
@torch.inference_mode()
def test_cuda_flashinfer_selection_preserves_zero_infinity_ties_and_mixed_precision(
    score_dtype, cis_dtype
):
    from operators.nosa.indexer.api import select_from_scores

    length = 8257
    count = (length - 32) // 16 + 1
    generator = torch.Generator().manual_seed(831)
    choices = torch.tensor([-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf")])
    scores = choices[torch.randint(len(choices), (5, 2, count), generator=generator)].to(
        score_dtype
    )
    cis = choices[torch.randint(len(choices), (count, 2), generator=generator)].to(cis_dtype)
    scores[1].zero_()
    scores[2].fill_(-float("inf"))
    scores[3].fill_(float("inf"))
    positions = torch.tensor([0, 32 * 64, 63 * 64, 99 * 64, length - 1])
    expected = _literal_selection_reference(scores, cis, positions, length)
    actual = select_from_scores(scores.cuda(), cis.cuda(), positions.cuda(), length)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    # These FP32 CIS differences disappear if selection uses a half-precision
    # QA workspace. Preserve them when promoting mixed score/CIS dtypes.
    if cis_dtype == torch.float32:
        scores.zero_()
        cis = 1.0 + torch.arange(count, dtype=torch.float32)[:, None].expand(-1, 2) * 1e-5
        expected = _literal_selection_reference(scores, cis, positions, length)
        actual = select_from_scores(scores.cuda(), cis.cuda(), positions.cuda(), length)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize(
    ("length", "rows", "dtype"),
    [
        (2049, 17, torch.bfloat16),
        (2112, 17, torch.float16),
        (2113, 17, torch.bfloat16),
        (65537, 7, torch.bfloat16),
        (65552, 7, torch.float16),
        (65568, 7, torch.bfloat16),
        (66560, 1024, torch.bfloat16),
    ],
)
@torch.inference_mode()
def test_cuda_fused_pool_preserves_standalone_rounding_and_compression_tile_tails(
    length, rows, dtype
):
    from operators.nosa.indexer.api import compressed_scores, pooled_scores

    count = (length - 32) // 16 + 1
    blocks = (length + 63) // 64
    start = length - rows
    generator = torch.Generator(device="cuda").manual_seed(391)
    query = torch.randn((rows, 2, 16, 128), device="cuda", dtype=dtype, generator=generator)
    keys = torch.randn((count, 2, 128), device="cuda", dtype=dtype, generator=generator)
    positions = torch.arange(start, length, device="cuda")
    scores = compressed_scores(query, keys, positions)
    expected = _pooled_reference(scores.float(), blocks)
    block = torch.arange(blocks, device="cuda")
    last = positions[:, None, None] // 64
    mandatory = (block == 0) | ((block <= last) & (last <= block + 16))
    expected.masked_fill_(mandatory, float("inf")).masked_fill_(block > last, -float("inf"))
    # Deliberately initialize every cell to NaN: incomplete tail stores must
    # fail even if a later top-k happened to discard their contents.
    workspace = torch.full((rows * 2, blocks), float("nan"), device="cuda", dtype=torch.float32)
    pooled_scores(query, keys, None, start, length, workspace)
    torch.testing.assert_close(workspace.view_as(expected), expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("cached", [False, True])
@torch.inference_mode()
def test_cuda_contiguous_selection_reuses_flat_workspace_and_stable_cis_without_position_allocation(
    cached, monkeypatch
):
    from operators.nosa.indexer.api import compressed_scores, select_contiguous_blocks

    generator = torch.Generator(device="cuda").manual_seed(817)
    rows, heads = 9, 2
    capacity_blocks = 1041
    workspace = torch.empty(rows * heads * capacity_blocks, dtype=torch.float32, device="cuda")
    for length in (66560, 6403, 65537, 65552):
        count = (length - 32) // 16 + 1
        blocks = (length + 63) // 64
        start = length - rows
        query = torch.randn(
            (rows, heads, 4, 64), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        keys = torch.randn(
            (count, heads, 64), dtype=query.dtype, device="cuda", generator=generator
        )
        # A strided FP32 CIS view checks both stride handling and precision.
        cis = torch.randn((count * 2, heads * 2), device="cuda", generator=generator)[::2, ::2]
        positions = torch.arange(start, length, device="cuda")
        scores = compressed_scores(query, keys, positions)
        expected = _literal_selection_reference(scores, cis, positions, length)
        stable = max(0, (length - 16) // 64)
        pooled = _pooled_reference(cis.T, blocks).T[:stable] if cached else None

        def forbidden(*args, **kwargs):
            raise AssertionError("contiguous selection must use scalar positions")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "arange", forbidden)
            ids, valid = select_contiguous_blocks(
                query,
                keys,
                cis,
                start,
                length,
                pooled_cis=pooled,
                workspace=workspace,
                return_valid_mask=True,
            )
        torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(valid.cpu(), expected >= 0)


@requires_cuda
@pytest.mark.parametrize(
    ("length", "start", "rows"), [(1, 0, 1), (65, 0, 2), (4096, 4032, 64), (65537, 65537, 0)]
)
@torch.inference_mode()
def test_cuda_contiguous_selection_short_context_and_empty_query_validity(length, start, rows):
    from operators.nosa.indexer.api import select_contiguous_blocks

    count = max(0, (length - 32) // 16 + 1)
    query = torch.zeros((rows, 2, 4, 64), device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros((count, 2, 64), device="cuda", dtype=query.dtype)
    cis = torch.zeros((count, 2), device="cuda", dtype=query.dtype)
    ids, valid = select_contiguous_blocks(query, keys, cis, start, length, return_valid_mask=True)
    block = torch.arange(64, device="cuda")[None, None, :]
    last = torch.arange(start, start + rows, device="cuda")[:, None, None] // 64
    expected_valid = (block <= last).expand(rows, 2, 64)
    torch.testing.assert_close(valid, expected_valid)
    torch.testing.assert_close(ids, torch.where(expected_valid, block, -1).expand_as(ids))


@requires_cuda
@pytest.mark.parametrize("invalid", ["small", "dtype", "matrix", "strided", "misaligned"])
@torch.inference_mode()
def test_cuda_contiguous_selection_rejects_invalid_scratch_before_launch(invalid):
    from operators.nosa.indexer.api import select_contiguous_blocks

    length, rows, heads = 66560, 2, 2
    count, blocks = (length - 32) // 16 + 1, (length + 63) // 64
    query = torch.zeros((rows, heads, 4, 64), device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros((count, heads, 64), device="cuda", dtype=query.dtype)
    cis = torch.zeros((count, heads), device="cuda", dtype=query.dtype)
    size = rows * heads * blocks
    if invalid == "small":
        workspace = torch.empty(size - 1, device="cuda", dtype=query.dtype)
    elif invalid == "dtype":
        workspace = torch.empty(size, device="cuda", dtype=torch.float32)
    elif invalid == "matrix":
        workspace = torch.empty((rows * heads, blocks), device="cuda", dtype=query.dtype)
    elif invalid == "strided":
        workspace = torch.empty(size * 2, device="cuda", dtype=query.dtype)[::2]
    else:
        workspace = torch.empty(size + 1, device="cuda", dtype=query.dtype)[1:]
    with pytest.raises(ValueError, match="workspace"):
        select_contiguous_blocks(query, keys, cis, length - rows, length, workspace=workspace)


@requires_cuda
@pytest.mark.parametrize("overflow", ["total_length", "compressed_windows", "query_rows"])
@torch.inference_mode()
def test_cuda_nosa_rejects_above_256k_before_launch(overflow, monkeypatch):
    from operators.nosa.indexer import api as operator

    length = 262144 + (overflow == "total_length")
    count = 16383 + (overflow == "compressed_windows")
    rows = 262145 if overflow == "query_rows" else 1
    query = torch.zeros((1, 2, 4, 64), device="cuda", dtype=torch.bfloat16).expand(rows, -1, -1, -1)
    keys = torch.zeros((count, 2, 64), device="cuda", dtype=query.dtype)
    cis = torch.zeros((count, 2), device="cuda", dtype=query.dtype)
    positions = torch.empty(rows, device="cuda", dtype=torch.int64)

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized geometry must fail before GPU kernels")

    monkeypatch.setattr(operator, "_launch_scores", forbidden)
    monkeypatch.setattr(operator, "_select_workspace", forbidden)
    with pytest.raises(ValueError, match="262144|16383"):
        operator.select_contiguous_blocks(query, keys, cis, 0, length)
    if overflow != "total_length":
        with pytest.raises(ValueError, match="262144|16383"):
            operator.compressed_scores(query, keys, positions)
    if overflow == "total_length":
        scores = torch.zeros((1, 2, count), device="cuda", dtype=query.dtype)
        with pytest.raises(ValueError, match="262144"):
            operator.select_from_scores(scores, cis, positions, length)
