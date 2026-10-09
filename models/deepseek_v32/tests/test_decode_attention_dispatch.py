"""Original decode batches and capacity-split prefill leaves keep distinct APIs."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_cache import WorkingSetTooLarge
from models.attention_contracts import TokenSelection
from models.deepseek_v32.attention import EchoAttentionRunner


@pytest.mark.parametrize("offload", [False, True])
def test_explicit_decode_uses_original_selection_after_recall(monkeypatch, offload):
    import models.deepseek_v32.attention as module

    records = torch.arange(12).reshape(3, 4)
    logical = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    physical = torch.tensor([[2, 1, -1]], dtype=torch.int32)
    runner = object.__new__(EchoAttentionRunner)
    runner.cfg = SimpleNamespace(attention_scale=0.25)
    runner.cache = SimpleNamespace(offload=offload, records=records, ensure=lambda ids: physical)
    q = torch.ones(1, 2, 4)
    calls = []

    def decode(query, kv, ids, scale):
        calls.append(True)
        assert query is q and kv is records and scale == 0.25
        assert torch.equal(ids, physical if offload else logical)
        return query

    monkeypatch.setattr(module, "sparse_mla_decode", decode)
    monkeypatch.setattr(module, "sparse_mla", lambda *a: pytest.fail("Unexpected prefill"))
    result = runner._consume(q, TokenSelection(logical), lambda _: nullcontext(), decode=True)
    assert result is q and calls == [True]


def test_capacity_split_single_query_leaves_keep_prefill(monkeypatch):
    import models.deepseek_v32.attention as module

    runner = object.__new__(EchoAttentionRunner)
    runner.cfg = SimpleNamespace(attention_scale=1.0)

    def ensure(ids):
        if len(ids) > 1:
            raise WorkingSetTooLarge("test pool fits one query at a time")
        return ids

    runner.cache = SimpleNamespace(
        offload=True,
        records=torch.zeros(5, 4),
        ensure=ensure,
        stats=SimpleNamespace(capacity_splits=0),
    )
    calls = []

    def prefill(q, records, ids, scale):
        calls.append(len(q))
        return q

    monkeypatch.setattr(module, "sparse_mla_from_pool", prefill)
    monkeypatch.setattr(module, "sparse_mla_decode", lambda *a: pytest.fail("Unexpected decode"))
    q = torch.arange(24).reshape(3, 2, 4)
    ids = torch.tensor([[0], [1], [2]], dtype=torch.int32)
    output = runner._consume(q, TokenSelection(ids), lambda _: nullcontext())
    assert torch.equal(output, q)
    assert calls == [1, 1, 1] and runner.cache.stats.capacity_splits == 2
