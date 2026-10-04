"""Opt-in diagnostics distinguish batch unions, residency, and actual rereads."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_pool import SharedSparseTokenPool
from models.deepseek_v32.echo_attention import EchoAttentionRunner


def diagnostic_runner(monkeypatch, *, enabled, slots=3, prefetch_ids=(), candidate_slots=0):
    import models.deepseek_v32.echo_attention as attention_module
    import operators.deepseek_v32.indexer.echo as indexer

    pool = SharedSparseTokenPool(64, 4, 1, slots, device="cpu", candidate_slots=candidate_slots)
    cache = pool.allocate_session(16).layer(0)
    cfg = SimpleNamespace(
        kv_lora_rank=2, qk_rope_head_dim=2, index_head_dim=2, index_topk=3, attention_scale=1.0
    )

    def project(hidden, position, normalized=False):
        values = torch.arange(position, position + len(hidden)).bfloat16()
        return SimpleNamespace(
            q=values[:, None, None],
            kv=values[:, None].expand(-1, 4).contiguous(),
            index_q=values,
            index_k=values[:, None].expand(-1, 2).to(torch.float8_e4m3fn).contiguous(),
            index_scale=torch.ones(len(hidden)),
            index_weights=torch.ones(len(hidden)),
        )

    attention = SimpleNamespace(
        cfg=cfg, device=torch.device("cpu"), project=project, output=lambda output: output
    )
    runner = EchoAttentionRunner(
        attention,
        16,
        offload=True,
        slots=slots,
        chunk_size=3,
        cache=cache,
        collect_cache_diagnostics=enabled,
    )

    def logits(q, k, weights, scales, start, prefetch=None):
        assert start == 6 and len(q) == 3
        cache.prefetch_reference(prefetch_ids)
        # The union is six records; each query still has its full exact top-3.
        scores = torch.full((3, len(k)), -100.0)
        scores[0, :3] = scores[2, :3] = torch.tensor([9.0, 8.0, 7.0])
        scores[1, 3:6] = torch.tensor([9.0, 8.0, 7.0])
        return scores

    monkeypatch.setattr(indexer, "logits", logits)
    monkeypatch.setattr(attention_module, "sparse_mla_from_pool", lambda q, *args: q)
    cache.begin_step(6)
    for start in (0, 3):
        cache.append(torch.arange(start, start + 3).bfloat16()[:, None].expand(-1, 4).contiguous())
    cache.commit()
    return pool, cache, runner


def test_union_initial_hits_and_cross_group_rereads_are_independent(monkeypatch):
    pool, cache, runner = diagnostic_runner(monkeypatch, enabled=True)
    cache.begin_step(3)
    runner.forward(torch.zeros(3, 2, dtype=torch.bfloat16))
    cache.commit()
    metrics = runner.cache_diagnostics
    assert metrics["unique_selection_records"] == 6
    assert metrics["fused_before_resident_selection_records"] == 3
    assert metrics["fused_before_hbm_token_hit_ratio"] == 0.5
    assert metrics["historical_selection_records"] == 6
    assert metrics["remaining_miss_after_append"] == 6
    assert metrics["prefetched_records"] == 0  # Q=P leaves no prefetch headroom.
    assert metrics["consumer_groups"] == 3
    assert metrics["group_selection_records"] == 9
    assert metrics["group_recalled_records"] == metrics["recalled_records"] == 9
    assert metrics["cross_group_reread_records"] == 3
    assert metrics["cross_group_reread_bytes"] == 3 * 4 * 2
    assert metrics["host_to_device_bytes"] == 9 * 4 * 2
    assert metrics["device_to_host_bytes"] == 3 * 4 * 2
    assert runner._diagnostic_state is None
    assert all(not isinstance(value, torch.Tensor) for value in metrics.values())
    pool.close()


def test_default_path_never_enters_intrusive_diagnostic_helpers(monkeypatch):
    pool, cache, runner = diagnostic_runner(monkeypatch, enabled=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("formal execution entered diagnostic observation")

    runner._start_cache_diagnostics = forbidden
    runner._diagnose_selection = forbidden
    runner._diagnose_after_append = forbidden
    runner._diagnose_group_before_recall = forbidden
    runner._finish_cache_diagnostics = forbidden
    cache.begin_step(3)
    output = runner.forward(torch.zeros(3, 2, dtype=torch.bfloat16))
    cache.commit()
    assert output.flatten().tolist() == [6, 7, 8]
    assert runner.cache_diagnostics is runner._diagnostic_state is None
    pool.close()


def test_prefetch_delta_is_not_confused_with_initial_residency(monkeypatch):
    pool, cache, runner = diagnostic_runner(monkeypatch, enabled=True, slots=4, prefetch_ids=(0,))
    cache.begin_step(3)
    runner.forward(torch.zeros(3, 2, dtype=torch.bfloat16))
    cache.commit()
    metrics = runner.cache_diagnostics
    assert metrics["unique_selection_records"] == 6
    assert metrics["fused_before_resident_selection_records"] == 4
    assert metrics["prefetched_records"] == 1
    assert metrics["remaining_miss_after_append"] == 5
    assert metrics["group_recalled_records"] == metrics["recalled_records"]
    assert metrics["host_to_device_bytes"] == (1 + metrics["recalled_records"]) * 8
    pool.close()


def test_failed_layer_releases_diagnostic_tensors_and_drops_stale_results(monkeypatch):
    pool, cache, runner = diagnostic_runner(monkeypatch, enabled=True)
    cache.begin_step(3)

    def fail(*args, **kwargs):
        raise RuntimeError("injected attention failure")

    runner._consume = fail
    runner.cache_diagnostics = {"stale": True}
    with pytest.raises(RuntimeError, match="attention failure"):
        runner.forward(torch.zeros(3, 2, dtype=torch.bfloat16))
    assert runner.cache_diagnostics is runner._diagnostic_state is None
    cache.rollback()
    pool.close()


def test_transient_diagnostics_do_not_map_candidate_ids_to_host(monkeypatch):
    import operators.deepseek_v32.indexer.echo as indexer

    pool, cache, runner = diagnostic_runner(monkeypatch, enabled=True, candidate_slots=3)
    original = cache.host_records()

    def logits(q, k, weights, scales, start, prefetch=None):
        assert prefetch["history_length"] == 6
        result = torch.full((3, len(k)), -torch.inf)
        for row, selection in enumerate(((0, 1, 6), (0, 6, 7), (1, 7, 8))):
            result[row, list(selection)] = torch.tensor([9.0, 8.0, 7.0])
        return result

    monkeypatch.setattr(indexer, "logits", logits)
    cache.begin_transient(3)
    runner.forward(torch.zeros(3, 2, dtype=torch.bfloat16))
    cache.discard_transient()
    metrics = runner.cache_diagnostics
    assert metrics["unique_selection_records"] == 5
    assert metrics["historical_selection_records"] == 2
    assert metrics["consumer_groups"] == 1
    assert metrics["group_selection_records"] == 5
    assert metrics["device_to_host_bytes"] == 0
    assert metrics["capacity_splits"] == 0
    assert cache.length == 6 and torch.equal(cache.host_records(), original)
    pool.close()
