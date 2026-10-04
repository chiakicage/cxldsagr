"""Cross-phase model contracts exercised with the generic CPU cache oracle."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_pool import SharedSparseTokenPool
from models.deepseek_v32.echo_attention import EchoAttentionRunner


@pytest.mark.parametrize("compute_callbacks", [False, True])
def test_one_outer_batch_preserves_prefetch_topk_append_recall_order(
    monkeypatch, compute_callbacks
):
    import operators.deepseek_v32.indexer.echo as indexer

    pool = SharedSparseTokenPool(64, 4, 1, 8, device="cpu")
    session = pool.allocate_session(16)
    cache = session.layer(0)
    cfg = SimpleNamespace(
        kv_lora_rank=2, qk_rope_head_dim=2, index_head_dim=2, index_topk=4, attention_scale=1.0
    )
    calls = []

    def project(hidden, position, normalized=False):
        count = len(hidden)
        values = torch.arange(position, position + count).bfloat16()
        return SimpleNamespace(
            q=values[:, None, None],
            kv=values[:, None].expand(-1, 4).contiguous(),
            index_k=values[:, None].expand(-1, 2).to(torch.float8_e4m3fn).contiguous(),
            index_q=values,
            index_scale=torch.ones(count),
            index_weights=torch.ones(count),
        )

    attention = SimpleNamespace(
        cfg=cfg, device=torch.device("cpu"), project=project, output=lambda value: value
    )
    runner = EchoAttentionRunner(attention, 16, offload=True, slots=8, chunk_size=2, cache=cache)

    def logits(q, k, weights, scales, start, prefetch=None):
        calls.append((start, len(q)))
        assert cache.written == start
        assert cache.indexer_visible_end == start + len(q)
        if prefetch is None:
            assert cache.all_history_resident
        else:
            assert prefetch["history_length"] == start
            # Only written history may be fetched while current index-K is visible.
            cache.prefetch_reference(range(start))
        return torch.arange(len(k)).float().expand(len(q), -1).clone()

    def consume(q, indices, scope):
        assert cache.written == cache.indexer_visible_end
        physical = cache.ensure(indices)
        valid = indices >= 0
        torch.testing.assert_close(
            cache.records[physical[valid].long(), 0], indices[valid].bfloat16(), rtol=0, atol=0
        )
        return q

    monkeypatch.setattr(indexer, "logits", logits)
    runner._consume = consume
    callbacks = {}
    if compute_callbacks:

        def unexpected_compute(*args, **kwargs):
            raise AssertionError("ordinary computation must not run beside the callbacks")

        attention.project = attention.output = unexpected_compute
        callbacks = {
            "project_callback": project,
            "output_callback": lambda value: (value, value.clone()),
        }
    for count in (6, 2):
        cache.begin_step(count)
        result = runner.forward(
            torch.zeros(count, 2, dtype=torch.bfloat16), capture_indices=True, **callbacks
        )
        if compute_callbacks:
            assert isinstance(result, tuple) and len(result) == 2
            torch.testing.assert_close(result[0], result[1], rtol=0, atol=0)
        cache.commit()
    assert calls == [(0, 6), (6, 2)]  # No hidden scoring split at runner.chunk_size=2.
    assert cache.metrics()["recalled_records"] == 0
    assert cache.metrics()["written_records"] == 8
    assert len(runner.last_indices) == 1
    session.release()
    pool.close()


def test_exact_union_capacity_split_keeps_every_query_selection():
    from cache.sparse_token_cache import WorkingSetTooLarge

    runner = object.__new__(EchoAttentionRunner)
    consumed = []

    class SelectionCache:
        offload = True
        stats = SimpleNamespace(capacity_splits=0)
        records = torch.empty(1)

        def ensure(self, indices):
            if len(torch.unique(indices)) > 3:
                raise WorkingSetTooLarge()
            consumed.append(indices.clone())
            return indices

    runner.cache = SelectionCache()
    runner.cfg = SimpleNamespace(attention_scale=1.0)
    # A separate test mock for the computation leaves cache selection invariant.
    from contextlib import nullcontext
    from unittest.mock import patch

    import models.deepseek_v32.echo_attention as module

    indices = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]], dtype=torch.int32)
    q = torch.arange(4).reshape(4, 1, 1)
    with patch.object(module, "sparse_mla_from_pool", lambda q, *args: q):
        output = runner._consume(q, indices, lambda _: nullcontext())
    torch.testing.assert_close(torch.cat(consumed), indices)
    torch.testing.assert_close(output, q)
    assert runner.cache.stats.capacity_splits == 3
    with pytest.raises(ValueError, match="one query"):
        runner._consume(q[:1], torch.arange(4).reshape(1, -1), lambda _: nullcontext())
