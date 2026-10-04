"""Official exact selection, causal padding and disclosed index-tie policy."""

from types import SimpleNamespace

import pytest
import torch

from operators.deepseek_v32.indexer import selection
from operators.deepseek_v32.indexer.selection import exact_topk


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("DeepSeek selection tests require Hopper SM90")


def test_selection_rejects_unsupported_metadata():
    with pytest.raises(NotImplementedError, match="Hopper"):
        exact_topk(torch.ones(2, 7), 3)
    for k in (0, 2049, True, 1.5):
        with pytest.raises(ValueError, match="k must"):
            exact_topk(torch.ones(2, 7), k)
    with pytest.raises(ValueError, match="FP32"):
        exact_topk(torch.ones(2, 7, dtype=torch.bfloat16), 3)
    with pytest.raises(ValueError, match="nonempty"):
        exact_topk(torch.empty(2, 0), 1)


def test_int32_adapter_preserves_official_ordering_abi_and_output_storage(monkeypatch):
    scores = torch.tensor([[3.0, 1.0, 3.0], [-1.0, -2.0, -3.0]])
    native_ids = torch.tensor([[0, 2], [0, 1]], dtype=torch.int32)
    native_values = torch.tensor([[3.0, 3.0], [-1.0, -2.0]])
    states = torch.empty(16, dtype=torch.uint8)

    def cache_buffer(name, size, device, *, zero_init):
        assert (name, size, device, zero_init) == (
            "radix_topk_row_states_cpu",
            1024 * 1024,
            scores.device,
            True,
        )
        return states

    def radix(
        input_scores, k, sorted_output, deterministic, tie_break, scratch, values, graph_safe
    ):
        assert input_scores is scores and scratch is states
        assert (k, sorted_output, deterministic, tie_break, graph_safe) == (2, True, True, 1, False)
        values.copy_(native_values)
        return native_ids

    backend = SimpleNamespace(
        _get_cache_buf=cache_buffer,
        get_topk_module=lambda: SimpleNamespace(radix_topk=radix),
        TopKTieBreak=SimpleNamespace(SMALL=1),
    )
    monkeypatch.setattr(selection, "_topk_backend", lambda: backend)
    values, indices = selection._official_topk_int32(scores, 2)
    assert indices is native_ids
    torch.testing.assert_close(values, native_values, rtol=0, atol=0)


@pytest.mark.parametrize("rows,columns,k", [(3, 17, 17), (5, 4099, 2048), (128, 65664, 2048)])
def test_cuda_wrapper_matches_public_flashinfer_bits_and_policy(rows, columns, k):
    require_sm90()
    backend = selection._topk_backend()
    scores = torch.arange(rows * columns, device="cuda", dtype=torch.float32)
    scores = -scores.remainder(19).reshape(rows, columns)
    scores[0].fill_(-torch.inf)
    scores[1, :4] = torch.tensor([-0.0, 0.0, -0.0, 0.0], device="cuda")
    scores[1, 31:] = -torch.inf
    before = scores.clone()
    expected_values, expected_indices = backend.top_k(
        scores, k, sorted=True, deterministic=True, tie_break=backend.TopKTieBreak.SMALL
    )
    expected_indices = torch.where(torch.isfinite(expected_values), expected_indices, -1).int()
    values, indices = exact_topk(scores, k)
    torch.testing.assert_close(
        values.view(torch.int32), expected_values.view(torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(scores.view(torch.int32), before.view(torch.int32), rtol=0, atol=0)


def test_cuda_nonfinite_mask_preserves_all_value_bits_and_finite_indices():
    require_sm90()
    bits = torch.tensor(
        [0, -2147483648, 0x7F7FFFFF, -8388609, 0x7F800000, -8388608, 0x7FC00001, -4194303],
        device="cuda",
        dtype=torch.int32,
    )
    values = bits.view(torch.float32)
    before = bits.clone()
    indices = torch.arange(8, device="cuda", dtype=torch.int32)
    expected = torch.where(torch.isfinite(values), indices, -1)
    selection._mask_nonfinite_indices_(values, indices)
    torch.testing.assert_close(values.view(torch.int32), before, rtol=0, atol=0)
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)


def test_cuda_exact_selection_graph_replay_and_nondefault_stream():
    require_sm90()
    scores = torch.randn(4, 4096, device="cuda")
    scores[:, -31:] = -torch.inf
    expected = exact_topk(scores, 2048)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        exact_topk(scores, 2048)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = exact_topk(scores, 2048)
    for _ in range(3):
        graph.replay()
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                result.view(torch.int32), reference.view(torch.int32), rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "rows,columns,k",
    [
        (1, 1, 1),
        (3, 17, 2048),
        (5, 257, 32),
        (32, 4099, 1024),
        (1024, 1024, 2048),
        (1024, 65536, 1),
        (1024, 66560, 2048),
    ],
)
def test_cuda_exact_values_causal_visibility_and_unique_ids(rows, columns, k):
    require_sm90()
    generator = torch.Generator(device="cuda").manual_seed(191)
    scores = torch.randn(rows, columns, device="cuda", generator=generator)
    start = max(0, columns - rows)
    ends = (start + torch.arange(rows, device="cuda") + 1).clamp_max(columns)
    scores.masked_fill_(torch.arange(columns, device="cuda")[None] >= ends[:, None], -torch.inf)
    values, indices = exact_topk(scores, k)
    reference = scores.topk(min(k, columns), dim=-1)
    torch.testing.assert_close(values, reference.values, atol=0, rtol=0)
    assert indices.dtype == torch.int32 and values.dtype == torch.float32
    valid = torch.isfinite(values)
    assert (indices[~valid] == -1).all()
    assert ((indices >= 0) == valid).all()
    assert (indices < ends[:, None]).all()
    gathered = scores.gather(-1, indices.clamp_min(0).long())
    torch.testing.assert_close(gathered[valid], values[valid], atol=0, rtol=0)
    ordered = indices.sort(-1).values
    assert not ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()


def test_cuda_every_k_uses_exact_small_index_ties():
    require_sm90()
    scores = -torch.arange(2049, device="cuda", dtype=torch.float32).remainder(19)
    scores = scores.expand(3, -1).clone()
    scores[1].fill_(-torch.inf)
    scores[2, 31:] = -torch.inf
    reference_ids = torch.argsort(scores, dim=-1, descending=True, stable=True)
    reference_values = scores.gather(-1, reference_ids)
    for k in range(1, 2049):
        values, indices = exact_topk(scores, k)
        expected_ids = (
            reference_ids[:, :k].int().masked_fill(~torch.isfinite(reference_values[:, :k]), -1)
        )
        torch.testing.assert_close(values, reference_values[:, :k], rtol=0, atol=0)
        torch.testing.assert_close(indices, expected_ids, rtol=0, atol=0)


def test_cuda_large_tie_overflow_and_all_invalid_rows_are_deterministic():
    require_sm90()
    scores = torch.zeros(3, 66560, device="cuda")
    scores[1].fill_(-torch.inf)
    scores[2, 7:] = -torch.inf
    expected = torch.arange(2048, device="cuda", dtype=torch.int32).expand(3, -1).clone()
    expected[1].fill_(-1)
    expected[2, 7:] = -1
    for _ in range(4):
        values, indices = exact_topk(scores, 2048)
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)
        assert values[0].eq(0).all() and values[1].isneginf().all()


def test_cuda_noncontiguous_extreme_finite_values_and_signed_zero():
    require_sm90()
    base = torch.tensor(
        [
            torch.finfo(torch.float32).min,
            -1e30,
            -1e-30,
            -0.0,
            0.0,
            1e-30,
            1e30,
            torch.finfo(torch.float32).max,
            -torch.inf,
        ],
        device="cuda",
    )
    scores = base.expand(5, -1).clone().T.contiguous().T
    assert not scores.is_contiguous()
    before = scores.clone()
    for k in (1, 2, 4, 8, 9):
        values, indices = exact_topk(scores, k)
        torch.testing.assert_close(values, scores.topk(k, dim=-1).values, atol=0, rtol=0)
        assert (indices[~torch.isfinite(values)] == -1).all()
    torch.testing.assert_close(scores, before, atol=0, rtol=0)


def test_cuda_empty_query_batch():
    require_sm90()
    values, indices = exact_topk(torch.empty(0, 7, device="cuda"), 2048)
    assert values.shape == indices.shape == (0, 7)
    assert indices.dtype == torch.int32
