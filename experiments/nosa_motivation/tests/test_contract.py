"""Reject weakened capacities, misleading revisit identities, and persisted candidates."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.nosa_motivation.src.config import (
    BACKEND_SCHEMES,
    METHODS,
    configuration,
    parser,
    resource_limits,
)
from experiments.nosa_motivation.src.measure import (
    check_diagnostics,
    check_graph_replays,
    check_request,
    check_resource_plan,
    compare_hidden,
    validate_warmup,
)


def config(**changes):
    args = parser().parse_args(["--run-id", "test_run"])
    for key, value in changes.items():
        setattr(args, key, value)
    return configuration(args)


def diagnostics(method, history=65536, *, recalled=False):
    h2d, d2h = (64 if recalled else 0), (0 if method == "hbm" or recalled else 128)
    return {
        "candidate_persistence": "gpu_transient",
        "retained_length": history,
        "transfer_metrics_status": "complete",
        "host_to_device_bytes": h2d,
        "prefix_host_to_device_bytes": 0,
        "candidate_host_to_device_bytes": h2d,
        "candidate_main_kv_host_to_device_bytes": h2d,
        "device_to_host_bytes": d2h,
        "prefix_device_to_host_bytes": d2h,
        "candidate_device_to_host_bytes": 0,
    }


def test_defaults_match_deepseek_trace_and_fixed_capacities():
    values = config()
    assert (
        values["history_tokens"],
        values["candidate_tokens"],
        values["chunk_size"],
        values["num_users"],
        values["rounds"],
        values["sparse_pool_tokens"],
        values["host_arena_tokens"],
    ) == (65536, 128, 1024, 16, 2, 65536, 16777216)
    assert values["methods"] == list(METHODS)
    assert resource_limits(values) == {
        "max_session_capacity": 65664,
        "max_history_tokens": 65536,
        "max_candidate_tokens": 128,
    }
    assert values["byte_subbudgets"] is None


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"rounds": 1}, "two users and two rounds"),
        ({"num_users": 1}, "two users and two rounds"),
        ({"sparse_pool_tokens": 65535}, "multiples of 64"),
        ({"sparse_pool_tokens": 32768}, "complete padded history"),
        ({"host_arena_tokens": 65536}, "retain every requested history"),
        ({"history_tokens": 262144, "sparse_pool_tokens": 262144}, "context limit"),
        ({"seed": -1}, "nonnegative"),
        ({"history_tokens": 65535}, "multiples of 64"),
        ({"chunk_size": 1000}, "multiples of 64"),
        ({"peak_bf16_tflops": float("nan")}, "finite and positive"),
    ],
)
def test_reject_invalid_config(changes, match):
    with pytest.raises(ValueError, match=match):
        config(**changes)


@pytest.mark.parametrize("method", METHODS)
def test_exact_fixed_quota_and_transient_contract(method):
    values = config()
    plan = SimpleNamespace(
        metadata={
            "sparse_pool_tokens": 65536,
            "host_arena_tokens": 16777216,
            "candidate_persistence": "gpu_transient",
        },
        hbm_tokens=65536 if method == "hbm" else 0,
        host_pages=0 if method == "hbm" else 262144,
    )
    check_resource_plan(method, plan, values)
    changed = deepcopy(plan)
    changed.hbm_tokens = changed.host_pages = 0
    with pytest.raises(AssertionError, match="admission"):
        check_resource_plan(method, changed, values)
    changed = deepcopy(plan)
    changed.metadata["candidate_persistence"] = "truncate_committed"
    with pytest.raises(AssertionError, match="transient GPU"):
        check_resource_plan(method, changed, values)


def test_hbm_evicted_revisit_remains_a_revisit():
    values = config()
    request = {"request_id": 16, "user_id": 0, "input_sha256": "ids"}
    row = {
        **request,
        "method": "hbm",
        "scheme": "hbm",
        "visit_index": 1,
        "is_revisit": True,
        "prefix_cache_hit": False,
        "stable_prefix_tokens": 65536,
        "candidate_suffix_tokens": 128,
        "resource_mode": "fixed_pools",
        "hbm_budget_bytes": None,
        "dram_budget_bytes": None,
        "cached_users": 1,
        "cache_diagnostics": diagnostics("hbm"),
    }
    check_request(request, row, values)
    row["is_revisit"] = False
    with pytest.raises(AssertionError, match="is_revisit"):
        check_request(request, row, values)


@pytest.mark.parametrize("method", METHODS)
def test_candidate_d2h_and_phase_misattribution_rejected(method):
    counters = diagnostics(method)
    counters["candidate_device_to_host_bytes"] = 16
    counters["device_to_host_bytes"] += 16
    with pytest.raises(AssertionError, match="must not"):
        check_diagnostics(method, counters, config())
    counters = diagnostics(method)
    counters["host_to_device_bytes"] += 1
    with pytest.raises(AssertionError, match="do not sum"):
        check_diagnostics(method, counters, config())


@pytest.mark.parametrize("method", METHODS)
def test_warmup_includes_host_recall_at_default_pool(method):
    values = config()
    requests = [
        {"request_id": index, "user_id": index % 16, "input_sha256": str(index)}
        for index in range(32)
    ]
    rows = []
    for order, index in enumerate((0, 1, 16)):
        rows.append(
            {
                **requests[index],
                "scheme": BACKEND_SCHEMES[method],
                "visit_index": int(order == 2),
                "prefix_cache_hit": order == 2 and method != "hbm",
                "cache_diagnostics": diagnostics(method, recalled=order == 2 and method != "hbm"),
            }
        )
    validate_warmup(method, rows, values, requests)
    if method != "hbm":
        # Indexer boundary copies alone do not exercise main-KV recall.
        rows[-1]["cache_diagnostics"]["candidate_main_kv_host_to_device_bytes"] = 0
        with pytest.raises(AssertionError, match="main-KV host recall"):
            validate_warmup(method, rows, values, requests)
        rows[-1]["cache_diagnostics"] = diagnostics(method)
        with pytest.raises(AssertionError, match="host recall"):
            validate_warmup(method, rows, values, requests)


def test_full_hidden_finiteness_and_tolerance_are_enforced():
    reference = torch.zeros(2, 4, dtype=torch.bfloat16)
    actual = reference.clone()
    actual[-1, -1] = 0.01
    result = compare_hidden(actual, reference, config())
    assert result["passed"] and not result["exact"] and result["elements"] == 8
    actual[-1, -1] = 0.1
    with pytest.raises(AssertionError):
        compare_hidden(actual, reference, config())
    actual[-1, -1] = float("nan")
    with pytest.raises(AssertionError, match="finite"):
        compare_hidden(actual, reference, config())


def test_graph_replay_gate_requires_both_islands_for_all_layers_and_queries():
    values = config(compute_graphs=True)
    before = {
        "enabled": True,
        "eager_fallbacks": 0,
        "graphs": 128,
        "query_sizes": [128, 1024],
        "project_replays": 0,
        "finish_replays": 0,
        "policy_revision": "nosa_compute_graphs_deferred_validation_v2",
        "finite_validation": "per-layer device flags; one host decision before commit",
        "static_storage_bytes": 1024,
        "static_allocated_bytes": 2048,
        "private_reserved_bytes": 4096,
    }
    after = {**before, "project_replays": 32 * 65, "finish_replays": 32 * 65}
    check_graph_replays(before, after, values, {"prefix_cache_hit": False})
    after["finish_replays"] -= 1
    with pytest.raises(AssertionError, match="replay coverage"):
        check_graph_replays(before, after, values, {"prefix_cache_hit": False})
    after["finish_replays"] += 1
    after["private_reserved_bytes"] += 512
    with pytest.raises(AssertionError, match="policy or reserved storage"):
        check_graph_replays(before, after, values, {"prefix_cache_hit": False})
