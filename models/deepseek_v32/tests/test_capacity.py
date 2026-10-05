"""CPU checks for full-population fixed-pool capacity planning."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_pool import SharedSparseTokenPool
from models.deepseek_v32.execution.capacity import (
    MAX_DEVICE_SLOTS,
    MAX_HOST_TOKENS,
    EchoCapacityPlanner,
)
from models.deepseek_v32.tests.test_cache_resources import planned_backend


def planner(**changes):
    cfg = SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_topk=16,
        max_seq_len=4096,
    )
    args = {
        "num_layers": 3,
        "history_tokens": 120,
        "candidate_tokens": 8,
        "chunk_size": 8,
        "total_hbm_bytes": 1 << 30,
        "model_hbm_bytes": 0,
        "dram_budget_bytes": 1 << 36,
        "hbm_fraction": 1,
        "allocator_headroom_bytes": 0,
    }
    args.update(changes)
    return EchoCapacityPlanner(cfg, **args)


def test_full_population_matches_shared_candidate_and_history_private_reservations():
    plan = planner()
    ledger = plan.estimate(sparse_pool_tokens=32, host_arena_tokens=384)
    backend = planned_backend()
    backend.device = torch.device("cuda")
    backend.scheme, backend.num_layers, backend.max_seq_len = "echo", 3, 4096
    backend.cfg = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128)
    private = backend.estimate_session_bytes(120)
    shared = SharedSparseTokenPool.estimate_shared_bytes(
        384, 576, 3, 32, device="cuda", candidate_slots=8
    )
    workspace = SharedSparseTokenPool.estimate_execution_workspace_bytes(384, 32)
    merged_indexer = 128 * (128 + 4)
    assert ledger["full_sessions"] == 3
    assert ledger["hbm"]["reservation_bytes"] == (
        shared["hbm"] + 3 * private["hbm"] + workspace + plan.execution.hbm + merged_indexer
    )
    assert ledger["dram"]["total_bytes"] == shared["dram"] + 3 * private["dram"] + 40
    assert ledger["hbm"]["allocator_allowance_bytes"] > 0
    assert ledger["hbm"]["total_bytes"] > ledger["hbm"]["reservation_bytes"]


def test_planning_does_not_allocate_tensors_or_initialize_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("capacity planning touched allocation or CUDA")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    monkeypatch.setattr(SharedSparseTokenPool, "__init__", forbidden)
    result = planner().max_host_capacity(sparse_pool_tokens=32)
    assert result["feasible"]
    assert not result["validation"]["gpu_execution_performed"]
    assert not result["validation"]["physical_peak_proven"]


def test_host_search_stops_at_per_layer_pinned_bin_boundary():
    three_users = planner().estimate(sparse_pool_tokens=32, host_arena_tokens=384)
    plan = planner(dram_budget_bytes=three_users["dram"]["total_bytes"])
    result = plan.max_host_capacity(sparse_pool_tokens=32)
    assert result["max_host_arena_tokens"] == 384
    assert result["max_full_sessions"] == 3
    assert result["limiting_constraints"] == ["dram_budget"]
    assert result["selected"]["dram"]["host_record_pinned_bin_bytes_per_layer"] == 524288
    assert result["next_candidate"]["dram"]["host_record_pinned_bin_bytes_per_layer"] == 1048576
    assert result["selected"]["dram"]["remaining_bytes"] == 0
    # The full next session must be tested; NH is not rounded to a partial user.
    assert result["next_candidate"]["host_arena_tokens"] == 512


def test_all_resident_indexers_constrain_host_capacity_even_with_free_dram():
    three_users = planner().estimate(sparse_pool_tokens=32, host_arena_tokens=384)
    budget = three_users["hbm"]["total_bytes"]
    result = planner(total_hbm_bytes=budget).max_host_capacity(sparse_pool_tokens=32)
    assert result["max_full_sessions"] == 3
    assert result["limiting_constraints"] == ["hbm_budget"]
    next_hbm = result["next_candidate"]["hbm"]
    indexers = next_hbm["components"]["session_index_keys"]
    assert next_hbm["total_bytes"] > budget
    # A one-session indexer ledger would incorrectly accept this population.
    assert next_hbm["total_bytes"] - indexers * 3 // 4 < budget


def test_auto_query_workspace_reduces_capacity_when_chunk_or_candidate_grows():
    narrow = planner().estimate(sparse_pool_tokens=32, host_arena_tokens=512)
    budget = narrow["hbm"]["total_bytes"]
    result = planner(total_hbm_bytes=budget, chunk_size=16).max_host_capacity(sparse_pool_tokens=32)
    assert result["max_full_sessions"] < 4
    wide_candidate = planner(history_tokens=112, candidate_tokens=16)
    wide_chunk = planner(chunk_size=16)
    assert wide_candidate.query_tokens == wide_chunk.query_tokens == 16
    assert wide_candidate.execution == wide_chunk.execution
    assert result["selected"]["workspace_query_tokens"] == 16


def test_device_search_checks_the_next_single_slot():
    chosen = planner().estimate(sparse_pool_tokens=80, host_arena_tokens=256)
    result = planner(total_hbm_bytes=chosen["hbm"]["total_bytes"]).max_device_capacity(
        host_arena_tokens=256
    )
    assert result["raw_alloc_max_P"] == result["useful_max_P"] == 80
    assert result["limiting_constraints"] == ["hbm_budget"]
    assert result["selected"]["hbm"]["remaining_bytes"] == 0
    assert result["next_candidate"]["sparse_pool_tokens"] == 81
    assert result["next_candidate"]["hbm"]["remaining_bytes"] < 0


def test_useful_device_capacity_is_separate_from_raw_allocation_limit():
    result = planner().max_device_capacity(host_arena_tokens=256)
    assert result["raw_alloc_max_P"] > 256
    assert result["useful_max_P"] == 256
    assert result["useful_selected"]["sparse_pool_tokens"] == 256
    assert result["useful_selected"]["feasible"]
    assert result["useful_limiting_constraints"] == ["host_token_capacity"]
    assert result["selected"]["host_arena_tokens"] == 256


def test_unaligned_logical_context_and_unused_arena_pages():
    plan = planner(history_tokens=121)
    ledger = plan.estimate(sparse_pool_tokens=32, host_arena_tokens=448)
    assert ledger["session_capacity_tokens"] == 121
    assert ledger["execution_context_tokens"] == 129
    assert ledger["session_page_tokens"] == 128
    assert ledger["full_sessions"] == 3
    assert ledger["unused_host_tokens"] == 64
    assert ledger["hbm"]["components"]["session_index_keys"] == 3 * 3 * 121 * 128
    assert ledger["dram"]["components"]["session_cpu_page_tables"] == 3 * 2 * 4
    result = plan.max_device_capacity(host_arena_tokens=448)
    assert result["selected"]["full_sessions"] == 3
    assert result["selected"]["unused_host_tokens"] == 64


def test_fraction_applies_to_total_hbm_and_model_is_subtracted_in_full():
    plan = planner(
        total_hbm_bytes=10_000_001,
        model_hbm_bytes=2_000_000,
        hbm_fraction=0.9,
        noncache_headroom_bytes=1_000_000,
        allocator_headroom_bytes=321,
    )
    assert plan.budgets["total_hbm_bytes"] == 10_000_001
    assert plan.budgets["model_hbm_bytes"] == 2_000_000
    assert plan.budgets["fraction_hbm_limit_bytes"] == 9_000_000
    assert plan.budgets["raw_hbm_envelope_bytes"] == 7_000_000
    assert plan.budgets["hbm_envelope_bytes"] == 7_000_000
    assert plan.budgets["fraction_reserve_bytes"] == 1_000_001
    assert plan.budgets["cache_hbm_budget_bytes"] == 6_000_000
    hbm = plan.estimate(sparse_pool_tokens=32, host_arena_tokens=128)["hbm"]
    assert hbm["budget_bytes"] == 7_000_000
    assert hbm["total_bytes"] == hbm["cache_total_bytes"] + 1_000_000
    assert hbm["allocator_allowance_bytes"] == hbm["persistent_allocator_allowance_bytes"] + 321


@pytest.mark.parametrize("budget_name", ["total_hbm_bytes", "dram_budget_bytes"])
def test_no_session_fit_has_no_selected_capacity_and_reports_first_failure(budget_name):
    result = planner(**{budget_name: 0}).max_host_capacity(sparse_pool_tokens=32)
    assert not result["feasible"]
    assert result["selected"] is None
    assert result["max_host_arena_tokens"] == result["max_full_sessions"] == 0
    assert result["next_candidate"]["full_sessions"] == 1
    assert result["limiting_constraints"]


def test_device_search_reports_fixed_dram_failure_instead_of_claiming_a_maximum():
    result = planner(dram_budget_bytes=0).max_device_capacity(host_arena_tokens=256)
    assert not result["feasible"]
    assert result["raw_alloc_max_P"] == result["useful_max_P"] == 0
    assert result["useful_selected"] is None
    assert result["limiting_constraints"] == ["dram_budget"]
    assert result["next_candidate"]["sparse_pool_tokens"] == 16


@pytest.mark.parametrize("model_bytes", [900_000, 950_000, 1_100_000])
def test_nonpositive_cache_envelope_is_explicitly_infeasible(model_bytes):
    plan = planner(total_hbm_bytes=1_000_000, model_hbm_bytes=model_bytes, hbm_fraction=0.9)
    assert plan.budgets["raw_hbm_envelope_bytes"] == 900_000 - model_bytes
    assert plan.budgets["hbm_envelope_bytes"] == 0
    assert plan.budgets["fraction_reserve_bytes"] == 100_000
    host = plan.max_host_capacity(sparse_pool_tokens=32)
    device = plan.max_device_capacity(host_arena_tokens=256)
    for result in (host, device):
        assert not result["feasible"]
        assert result["selected"] is None
        assert "hbm_budget" in result["limiting_constraints"]


def test_native_integer_limits_are_search_boundaries_without_large_allocations():
    plan = planner(total_hbm_bytes=1 << 70, dram_budget_bytes=1 << 70)
    device = plan.max_device_capacity(host_arena_tokens=256)
    assert device["raw_alloc_max_P"] == MAX_DEVICE_SLOTS - plan.candidate_tokens
    assert device["limiting_constraints"] == ["native_device_index_limit"]
    assert (
        device["next_candidate"]["sparse_pool_tokens"]
        == MAX_DEVICE_SLOTS - plan.candidate_tokens + 1
    )
    host = plan.max_host_capacity(sparse_pool_tokens=32)
    assert host["max_host_arena_tokens"] == MAX_HOST_TOKENS // 128 * 128
    assert host["limiting_constraints"] == ["native_host_index_limit"]


@pytest.mark.parametrize("fraction", [0, -1, 1.01, float("nan"), float("inf"), True])
def test_invalid_fractions_are_rejected(fraction):
    with pytest.raises(ValueError, match="hbm_fraction"):
        planner(hbm_fraction=fraction)


def test_shape_and_indexing_constraints_are_explicit():
    plan = planner()
    with pytest.raises(ValueError, match="fixed P"):
        plan.max_host_capacity(sparse_pool_tokens=8)
    with pytest.raises(ValueError, match="fixed NH"):
        plan.max_device_capacity(host_arena_tokens=129)
    with pytest.raises(ValueError, match="complete session"):
        plan.max_device_capacity(host_arena_tokens=64)
    with pytest.raises(ValueError, match="context"):
        planner(history_tokens=4096)
    with pytest.raises(ValueError, match="integer"):
        planner(num_layers=True)
    ledger = plan.estimate(sparse_pool_tokens=8, host_arena_tokens=64)
    assert ledger["violated_constraints"] == ["minimum_device_pool", "full_session_fit"]


def test_candidate_growth_does_not_consume_host_pages_or_private_indexer_rows():
    small = planner(history_tokens=64, candidate_tokens=8).estimate(
        sparse_pool_tokens=32, host_arena_tokens=192
    )
    larger = planner(history_tokens=64, candidate_tokens=24).estimate(
        sparse_pool_tokens=32, host_arena_tokens=192
    )
    assert small["full_sessions"] == larger["full_sessions"] == 3
    assert small["session_capacity_tokens"] == larger["session_capacity_tokens"] == 64
    assert small["session_page_tokens"] == larger["session_page_tokens"] == 64
    assert small["dram"] == larger["dram"]
    for name in ("session_index_keys", "session_index_scales", "session_page_tables"):
        assert small["hbm"]["components"][name] == larger["hbm"]["components"][name]
    assert small["execution_context_tokens"] == 72
    assert larger["execution_context_tokens"] == 88
    assert larger["candidate_persistence"] == "gpu_transient"
    assert larger["candidate_device_to_host_bytes"] == larger["candidate_host_tokens"] == 0


def test_candidate_and_merged_storages_are_shared_across_users():
    one = planner().estimate(sparse_pool_tokens=32, host_arena_tokens=128)
    four = planner().estimate(sparse_pool_tokens=32, host_arena_tokens=512)
    expected = {
        "candidate_kv_records": 3 * 8 * 576 * 2,
        "merged_index_keys": 128 * 128,
        "merged_index_scales": 128 * 4,
    }
    for name, size in expected.items():
        assert one["hbm"]["components"][name] == four["hbm"]["components"][name] == size
    assert "candidate_index_keys" not in one["hbm"]["components"]
    assert "candidate_index_scales" not in one["hbm"]["components"]
    assert (
        four["hbm"]["components"]["session_index_keys"]
        == 4 * one["hbm"]["components"]["session_index_keys"]
    )


def test_native_selection_scratch_is_shared_and_counter_slab_tracks_all_sessions():
    one = planner().estimate(sparse_pool_tokens=32, host_arena_tokens=128)
    four = planner().estimate(sparse_pool_tokens=32, host_arena_tokens=512)
    ten_layers = planner(num_layers=10).estimate(sparse_pool_tokens=32, host_arena_tokens=512)
    for ledger in (one, four, ten_layers):
        components = ledger["hbm"]["components"]
        assert components["resident_selection_bitmap"] == 8
        assert components["resident_selection_count"] == 4
        assert components["append_orders"] == ledger["num_layers"] * 32 * 8
        assert components["session_prefetch_counters"] == (
            ledger["full_sessions"] * ledger["num_layers"] * 24
        )
        assert components["session_native_counters"] == (
            ledger["full_sessions"] * ledger["num_layers"] * 32
        )


def test_selection_bitmap_includes_sentinel_and_candidate_at_word_boundaries():
    aligned = planner(candidate_tokens=8).estimate(sparse_pool_tokens=55, host_arena_tokens=128)
    tail = planner(candidate_tokens=9).estimate(sparse_pool_tokens=55, host_arena_tokens=128)
    assert aligned["hbm"]["components"]["resident_selection_bitmap"] == 8
    assert tail["hbm"]["components"]["resident_selection_bitmap"] == 12
    target = planner(candidate_tokens=128).estimate(sparse_pool_tokens=65536, host_arena_tokens=128)
    components = target["hbm"]["components"]
    assert components["resident_selection_bitmap"] + components["resident_selection_count"] == 8216


def test_counter_allocator_allowance_uses_one_combined_slab_per_session(monkeypatch):
    plan = planner()

    def allocation_bytes(size, device):
        assert device == "cuda"
        # Only the complete [layers, 7] slab receives this synthetic rounding
        # charge. Separately charging its views would yield a different result.
        return size + (13 if size == 3 * 56 else 0)

    monkeypatch.setattr(
        "models.deepseek_v32.execution.capacity.dense_staging_allocation_bytes", allocation_bytes
    )
    ledger = plan.estimate(sparse_pool_tokens=32, host_arena_tokens=384)
    assert ledger["hbm"]["persistent_allocator_allowance_bytes"] == 3 * 13
