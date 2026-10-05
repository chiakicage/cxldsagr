"""CIS scoring and full NOSA selection checked against token-interval oracles."""

import math

import pytest
import torch

from models.attention_contracts import AttentionContext
from models.nosa.attention import ResidentLayerView
from models.nosa.indexer import NosaIndexer
from models.nosa.scoring import NosaAttentionState, cis_scores, compress_sequence


def select(q, keys, cis, start, *, chunk=64, backend="reference", use_state=False):
    records = {"keys": keys}
    state = NosaAttentionState(cis) if use_state else None
    if not use_state:
        records["cis_scores"] = cis
    return NosaIndexer(query_chunk_size=chunk, mode="nosa", backend=backend)(
        q, ResidentLayerView(1, **records), AttentionContext(1, start, len(q), state)
    )


def full_selection_oracle(q, keys, cis, start):
    """Rank independent per-head distributions by token-window intersections.

    Explicit Python sets model both selection stages; there are no vectorized
    pooling gathers, promoted infinities, or tensor-level sort masks here.
    """
    q, keys, cis = q.cpu(), keys.cpu(), cis.cpu()
    groups = q.shape[1] // keys.shape[1]
    output = torch.full((len(q), keys.shape[1], 64), -1, dtype=torch.long)
    for row, position in enumerate(range(start, start + len(q))):
        last = position // 64
        mandatory = {0} | set(range(max(0, last - 16), last + 1))
        windows = list(range(0, position + 2 - 32, 16))
        means = [keys[s : s + 32].float().mean(0).to(keys.dtype) for s in windows]
        cis_means = [cis[s : s + 32].float().mean(0).to(cis.dtype) for s in windows]
        for head in range(keys.shape[1]):
            probabilities = torch.zeros(len(windows))
            for qhead in range(head * groups, (head + 1) * groups):
                logits = torch.tensor(
                    [
                        torch.dot(q[row, qhead].float(), key[head].float()).item()
                        / math.sqrt(q.shape[-1])
                        for key in means
                    ]
                )
                probabilities += logits.softmax(-1)
            probabilities = probabilities.to(q.dtype)
            query_ranks, cis_ranks = {}, {}
            for block in range(last + 1):
                overlap = [
                    i for i, s in enumerate(windows) if s < (block + 1) * 64 and s + 32 > block * 64
                ]
                query_ranks[block] = max((float(probabilities[i]) for i in overlap), default=0)
                cis_ranks[block] = max((float(cis_means[i][head]) for i in overlap), default=0)
            first = mandatory | set(
                sorted(set(range(last + 1)) - mandatory, key=lambda b: (-query_ranks[b], b))[
                    : max(0, 33 - len(mandatory))
                ]
            )
            final = first | set(
                sorted(set(range(last + 1)) - first, key=lambda b: (-cis_ranks[b], b))[
                    : max(0, 64 - len(first))
                ]
            )
            ordered = sorted(final)
            output[row, head, : len(ordered)] = torch.tensor(ordered)
    return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_cis_checkpoint_projection_rounding_and_stable_softplus(dtype, with_bias):
    values = torch.tensor([[[1000.0, -1000.0], [0.5, 0.25]], [[0.0, 0.0], [2.0, -1.0]]]).to(dtype)
    delta = torch.tensor([[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 1.0, 0.5]]).to(dtype)
    scale = torch.tensor([-0.25, -2.0]).to(dtype)
    bias = torch.tensor([-1.0, 0.25]).to(dtype) if with_bias else None
    projected = values.flatten(1).float() @ delta.float().T
    if bias is not None:
        projected += bias.float()
    projected = projected.to(dtype).double()
    expected = (torch.logaddexp(torch.zeros_like(projected), projected) * scale.double()).to(dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = cis_scores(values, delta, scale, bias)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype and torch.isfinite(actual).all()
    assert NosaAttentionState(actual).cis_bias is actual


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_selection_matches_independent_gqa_and_cis_oracle(dtype):
    generator = torch.Generator().manual_seed(611)
    start = 100 * 64 + 12
    q = torch.randn((3, 4, 8), generator=generator).to(dtype)
    keys = torch.randn((start + len(q), 2, 8), generator=generator).to(dtype)
    cis = (-torch.rand((len(keys), 2), generator=generator) * 12).to(dtype)
    actual = select(q, keys, cis, start)
    expected = full_selection_oracle(q, keys, cis, start)
    torch.testing.assert_close(actual.block_ids, expected, atol=0, rtol=0)
    assert actual.valid_mask.all()


def test_cis_fills_after_query_stage_and_source_local_boundary_is_inclusive():
    start = 100 * 64
    q = torch.zeros((1, 4, 8))
    keys = torch.zeros((start + 1, 2, 8))
    ramp = torch.arange(len(keys), dtype=torch.float32)
    cis = torch.stack((ramp, -ramp), dim=-1)
    result = select(q, keys, cis, start).block_ids
    torch.testing.assert_close(result[0, 0], torch.tensor([*range(16), *range(53, 101)]))
    torch.testing.assert_close(result[0, 1], torch.tensor([*range(47), *range(84, 101)]))
    assert (result == 84).any(-1).all()
    assert NosaIndexer(mode="nosa").policy.local_blocks == 17
    assert NosaIndexer().policy.local_blocks == 16


@pytest.mark.parametrize("length", [1, 31, 32, 47, 48, 63, 64, 65, 4096, 4097])
def test_full_short_context_padding_and_compression_boundaries(length):
    q, keys = torch.zeros((1, 4, 8)), torch.zeros((length, 2, 8))
    cis = torch.zeros((length, 2))
    selected = select(q, keys, cis, length - 1)
    count = min(64, (length + 63) // 64)
    assert selected.block_ids.shape == (1, 2, 64)
    assert selected.valid_mask.sum().item() == count * 2
    assert ((selected.block_ids < 0) == ~selected.valid_mask).all()
    compressed = compress_sequence(keys)
    assert len(compressed) == max(0, (length - 32) // 16 + 1)


def test_full_selection_causal_chunking_and_ephemeral_state():
    generator = torch.Generator().manual_seed(918)
    start = 82 * 64 - 5
    q = torch.randn((23, 4, 8), generator=generator)
    keys = torch.randn((start + len(q) + 48, 2, 8), generator=generator)
    cis = torch.randn((len(keys), 2), generator=generator)
    state = NosaAttentionState(cis)
    cache = ResidentLayerView(1, keys=keys)
    sentinel = object()
    cache.set_layer_state(1, sentinel)
    expected = NosaIndexer(mode="nosa")(q, cache, AttentionContext(1, start, len(q), state))
    actual = select(q, keys, cis, start, chunk=1)
    torch.testing.assert_close(actual.block_ids, expected.block_ids, atol=0, rtol=0)
    for row in (0, 6, 22):
        stop = start + row + 1
        one = select(q[row : row + 1], keys[:stop], cis[:stop], stop - 1, use_state=True)
        torch.testing.assert_close(one.block_ids[0], actual.block_ids[row], atol=0, rtol=0)
    assert cache.get_layer_state(1) is sentinel


def test_full_selection_requires_cis_and_supported_mode_budget():
    with pytest.raises(ValueError, match="cis_scores"):
        NosaIndexer(mode="nosa")(
            torch.zeros((1, 4, 8)),
            ResidentLayerView(1, keys=torch.zeros((1, 2, 8))),
            AttentionContext(1, 0, 1),
        )
    with pytest.raises(ValueError, match="block_budget=64"):
        NosaIndexer(mode="nosa", block_budget=32)
    with pytest.raises(ValueError, match="mode"):
        NosaIndexer(mode="typo")
    with pytest.raises(ValueError, match="backend"):
        NosaIndexer(mode="nosa", backend="typo")


def test_explicit_triton_indexer_rejects_cpu_even_for_short_context():
    with pytest.raises(ValueError, match="CUDA"):
        select(
            torch.zeros((1, 4, 64)),
            torch.zeros((1, 2, 64)),
            torch.zeros((1, 2)),
            0,
            backend="triton",
        )


@pytest.mark.parametrize(
    ("device", "backend"),
    [
        ("cpu", "reference"),
        pytest.param(
            "cuda",
            "triton",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("name", ["Q", "K", "CIS"])
@pytest.mark.parametrize("value", [torch.nan, torch.inf, -torch.inf])
def test_full_nosa_nonfinite_inputs_are_rejected(device, backend, name, value):
    # The short-context fast path must still reject nonfinite inputs even
    # though every visible block will be selected without computing scores.
    q = torch.zeros((2, 32, 128), device=device, dtype=torch.bfloat16)
    keys = torch.zeros((33, 2, 128), device=device, dtype=q.dtype)
    cis = torch.zeros((33, 2), device=device, dtype=q.dtype)
    inputs = {"Q": q, "K": keys, "CIS": cis}
    inputs[name].reshape(-1)[-1] = value
    with pytest.raises(ValueError, match="finite"):
        select(q, keys, cis, 31, chunk=None, backend=backend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("length", "start", "rows", "strided"),
    [(66560, 65536, 1024, True), (66559, 65536, 1023, False), (63, 0, 63, True)],
)
@torch.inference_mode()
def test_cuda_default_nosa_chunks_preserve_64_query_block_selections(length, start, rows, strided):
    generator = torch.Generator(device="cuda").manual_seed(857)
    q = torch.randn(
        (rows, 40 if strided else 32, 128),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )[:, :32]
    keys = torch.randn(
        (length, 4 if strided else 2, 128),
        generator=generator,
        device="cuda",
        dtype=q.dtype,
    )
    if strided:
        keys = keys[:, ::2]
    cis = -torch.rand((length, 2), generator=generator, device="cuda", dtype=q.dtype)
    expected = select(q, keys, cis, start, chunk=64, backend="triton")
    for backend in ("auto", "triton"):
        actual = select(q, keys, cis, start, chunk=None, backend=backend)
        torch.testing.assert_close(actual.block_ids, expected.block_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual.valid_mask, expected.valid_mask, rtol=0, atol=0)
        assert actual.block_ids.shape == (rows, 2, 64)
    if length < 64:
        assert expected.valid_mask.sum() == rows * 2
        assert (expected.block_ids[..., 0] == 0).all()
