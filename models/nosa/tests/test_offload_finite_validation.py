"""Offload finite checks reject inputs before derived reservation and remain bounded."""

import pytest
import torch

from models.attention_contracts import AttentionContext
from models.nosa.cache.offload import NosaOffloadCache
from models.nosa.config import NosaConfig
from models.nosa.indexer import NosaIndexer

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    ),
]


def setup(device):
    heads, dim = (32, 128) if device == "cuda" else (4, 4)
    config = NosaConfig(
        hidden_size=heads * dim,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=2,
        head_dim=dim,
        vocab_size=32,
        max_position_embeddings=160,
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cache = NosaOffloadCache(config, 160, device=device, dtype=dtype)
    keys = torch.randn((95, 2, dim), dtype=dtype, device=device)
    cis = torch.randn((95, 2), dtype=dtype, device=device)
    # Exercise a real strided query without reading unused storage holes.
    query = torch.randn((15, heads, dim * 2), dtype=dtype, device=device)[..., :dim]
    cache.begin_step(80)
    cache.write_layer(0, keys=keys[:80], values=keys[:80], cis_scores=cis[:80])
    cache.commit_step()
    return cache, keys, cis, query


def append(cache, keys, cis):
    cache.begin_step(15)
    cache.write_layer(0, keys=keys[80:], values=keys[80:], cis_scores=cis[80:])


def snapshot(cache):
    return (
        cache.indexer_cache.layer_state(0),
        {name: tensor.clone() for name, tensor in cache.indexer_cache.layer_view(0).items()},
        cache.indexer_cache.stats(),
    )


def unchanged(cache, saved):
    state, tensors, stats = saved
    assert cache.indexer_cache.layer_state(0) == state
    assert cache.indexer_cache.stats() == stats
    for name, tensor in cache.indexer_cache.layer_view(0).items():
        torch.testing.assert_close(tensor, tensors[name], atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("target", ["query", "keys", "cis"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_bad_query_or_uncompressed_tail_fails_before_any_derived_reservation(
    monkeypatch, device, target, value
):
    cache, keys, cis, query = setup(device)
    saved = snapshot(cache)
    {"query": query, "keys": keys, "cis": cis}[target][-1].fill_(value)
    append(cache, keys, cis)

    def unexpected_reservation(*args, **kwargs):
        pytest.fail("nonfinite input reached derived reservation")

    monkeypatch.setattr(cache.indexer_cache, "reserve_layer", unexpected_reservation)
    indexer = NosaIndexer(mode="nosa", backend="triton" if device == "cuda" else "reference")
    try:
        with pytest.raises(ValueError, match="finite Q, K and CIS"):
            indexer(query, cache, AttentionContext(0, 80, 15))
        unchanged(cache, saved)
        cache.abort_step()
        assert cache.length == cache.indexer_cache.length == 80
    finally:
        cache.release()


@pytest.mark.parametrize("device", DEVICES)
def test_prepared_records_still_reject_each_new_query(monkeypatch, device):
    cache, keys, cis, query = setup(device)
    append(cache, keys, cis)
    indexer = NosaIndexer(mode="nosa", backend="triton" if device == "cuda" else "reference")
    indexer(query, cache, AttentionContext(0, 80, 15))
    saved = snapshot(cache)
    query[-1, -1, -1] = torch.nan

    def unexpected_reservation(*args, **kwargs):
        pytest.fail("prepared path must not reserve again")

    monkeypatch.setattr(cache.indexer_cache, "reserve_layer", unexpected_reservation)
    try:
        with pytest.raises(ValueError, match="finite Q, K and CIS"):
            indexer(query, cache, AttentionContext(0, 80, 15))
        unchanged(cache, saved)
        cache.abort_step()
        assert cache.length == cache.indexer_cache.length == 80
    finally:
        cache.release()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("target", ["keys", "cis"])
def test_direct_commit_without_queries_checks_new_tail(monkeypatch, device, target):
    cache, keys, cis, _ = setup(device)
    saved = snapshot(cache)
    {"keys": keys, "cis": cis}[target][-1].fill_(torch.nan)
    append(cache, keys, cis)
    try:
        with pytest.raises(ValueError, match="finite Q, K and CIS"):
            cache.commit_step()
        unchanged(cache, saved)
        assert cache.length == cache.indexer_cache.length == 80
        cache.abort_step()
    finally:
        cache.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_combines_unvalidated_inputs_and_only_rechecks_q_when_prepared(monkeypatch):
    from operators.nosa.indexer import validation

    cache, keys, cis, query = setup("cuda")
    append(cache, keys, cis)
    calls = []
    original = validation.all_finite

    def finite(q, k, c, **kwargs):
        calls.append((q, len(k), len(c)))
        return original(q, k, c, **kwargs)

    monkeypatch.setattr(validation, "all_finite", finite)
    indexer = NosaIndexer(mode="nosa", backend="triton")
    try:
        first = indexer(query, cache, AttentionContext(0, 80, 15))
        second = indexer(query, cache, AttentionContext(0, 80, 15))
        assert len(calls) == 2
        assert calls[0] == (query, 15, 15)
        assert calls[1] == (query, 0, 0)
        torch.testing.assert_close(first.block_ids, second.block_ids, atol=0, rtol=0)
        cache.commit_step()
        assert len(calls) == 2  # Already prepared commit has no new values to validate.
        assert cache.length == cache.indexer_cache.length == 95
    finally:
        cache.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("queries", [1, 128, 1024])
def test_cuda_actual_finite_scratch_fits_existing_session_reservation(monkeypatch, queries):
    from cache.allocator.budget import allocation_bytes
    from models.nosa.execution.session_budget import session_budget_breakdown
    from operators.nosa.indexer.validation import all_finite

    q = torch.zeros((queries, 32, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((queries, 2, 128), device="cuda", dtype=torch.bfloat16)
    cis = torch.zeros((queries, 2), device="cuda", dtype=torch.bfloat16)
    sizes = []
    original = torch.empty

    def empty(*args, **kwargs):
        tensor = original(*args, **kwargs)
        if tensor.is_cuda:
            sizes.append(tensor.untyped_storage().nbytes())
        return tensor

    monkeypatch.setattr(torch, "empty", empty)
    assert all_finite(q, k, cis).item()
    assert len(sizes) == 2 and sizes[1] == 1
    bound = session_budget_breakdown(
        capacity=65536 + queries,
        queries=queries,
        layers=32,
        kv_heads=2,
        query_heads=32,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda",
        scheme="overlap",
    )
    assert sum(allocation_bytes(size, "cuda") for size in sizes) <= bound["finite_hbm"]
