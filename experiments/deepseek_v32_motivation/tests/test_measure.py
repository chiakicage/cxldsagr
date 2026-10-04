from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_motivation.src.measure import (
    INDEXER_DISPATCH_POLICY,
    check_compute_graph_replays,
    check_request,
    configuration,
    configure_precision,
    parser,
    resource_limits,
    validate_warmup_trace,
    warmup,
)


def test_formal_precision_disables_tf32_and_records_available_settings():
    matmul = SimpleNamespace(
        allow_tf32=True,
        fp32_precision="none",
        allow_bf16_reduced_precision_reduction=True,
        allow_fp16_reduced_precision_reduction=True,
    )
    state = {"precision": "high"}
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=matmul),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
        set_float32_matmul_precision=lambda value: state.update(precision=value),
        get_float32_matmul_precision=lambda: state["precision"],
    )
    settings = configure_precision(torch)
    assert matmul.allow_tf32 is False
    assert settings["cuda_matmul_allow_tf32"] is False
    assert settings["float32_matmul_precision"] == "highest"
    assert settings["bf16_reduced_precision_reduction"] is True
    assert settings["cudnn_allow_tf32"] is True
    assert "unavailable" in settings["cudnn_fp32_precision"]


def test_default_workload_does_not_fill_host_arena():
    config = configuration(parser().parse_args(["--run-id", "test"]))
    assert config["num_users"] == 16
    assert config["requests_per_scheme"] == 32
    assert config["sparse_pool_tokens"] == 65536
    assert config["host_arena_tokens"] == 16777216
    assert config["num_users"] * config["padded_history_tokens"] < config["host_arena_tokens"]
    assert config["byte_subbudgets"] is None
    assert config["warmup_requests_per_scheme"] == 3
    assert config["warmup_request_indices"] == [0, 1, 16]
    assert config["indexer_dispatch_policy"] == INDEXER_DISPATCH_POLICY
    assert resource_limits(config) == {
        "max_session_capacity": 65664,
        "max_history_tokens": 65536,
        "max_candidate_tokens": 128,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["--rounds", "1"],
        ["--num-users", "1"],
        ["--sparse-pool-tokens", "32768"],
        ["--host-arena-tokens", "65536"],
        ["--host-arena-tokens", "1048577"],
    ],
)
def test_invalid_capacity_or_trace_rejected(arguments):
    with pytest.raises(ValueError):
        configuration(parser().parse_args(["--run-id", "test", *arguments]))


def test_evicted_hbm_user_is_still_a_revisit():
    config = configuration(parser().parse_args(["--run-id", "test"]))
    request = {"request_id": 16, "user_id": 10}
    metrics = {
        **request,
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
        "cache_diagnostics": {"host_to_device_bytes": 0, "device_to_host_bytes": 0},
    }
    check_request(request, metrics, config)
    with pytest.raises(AssertionError, match="is_revisit"):
        check_request(request, {**metrics, "is_revisit": False}, config)
    with pytest.raises(AssertionError, match="prefix_cache_hit"):
        check_request(request, {**metrics, "prefix_cache_hit": True}, config)


@pytest.mark.parametrize("cache_hit", [False, True])
@pytest.mark.parametrize("candidate", [128, 1024])
def test_compute_replay_audit_covers_every_history_chunk_and_independent_layer(
    cache_hit, candidate
):
    config = configuration(
        parser().parse_args(
            [
                "--run-id",
                "test",
                "--compute-graphs",
                "--history-tokens",
                "2305",
                "--candidate-tokens",
                str(candidate),
            ]
        )
    )
    before = {
        "enabled": True,
        "allocated": True,
        "eager_fallbacks": 0,
        "graphs": [
            {"layer": layer, "queries": queries, "projection_replays": 9, "finish_replays": 9}
            for layer in range(10)
            for queries in {1024, 257, candidate}
        ],
    }
    after = deepcopy(before)
    for graph in after["graphs"]:
        delta = int(graph["queries"] == candidate)
        if not cache_hit:
            delta += {1024: 2, 257: 1}.get(graph["queries"], 0)
        graph["projection_replays"] += delta
        graph["finish_replays"] += delta
    metrics = {"prefix_cache_hit": cache_hit}
    check_compute_graph_replays(before, after, config, metrics)
    for defect in ("missing_layer", "duplicate", "fallback", "uncaptured", "reduced_work"):
        invalid = deepcopy(after)
        if defect == "missing_layer":
            invalid["graphs"].pop()
        elif defect == "duplicate":
            invalid["graphs"].append(invalid["graphs"][0])
        elif defect == "fallback":
            invalid["eager_fallbacks"] = 1
        elif defect == "uncaptured":
            invalid["allocated"] = False
        else:
            invalid["graphs"][0]["finish_replays"] -= 1
        with pytest.raises(ValueError, match="compute graph"):
            check_compute_graph_replays(before, invalid, config, metrics)


@pytest.mark.parametrize("scheme", ["hbm", "echo", "serial_sparse", "dense_prefetch"])
def test_warmup_exercises_host_recall_then_releases_cache(scheme):
    config = configuration(parser().parse_args(["--run-id", "test"]))
    events = []
    backend = SimpleNamespace(
        scheme=scheme,
        synchronize=lambda: events.append("synchronize"),
        close=lambda: events.append("release_shared"),
    )

    class Runner:
        def __init__(self, actual_backend, *, resource_limits):
            assert actual_backend is backend
            assert resource_limits["max_history_tokens"] == 65536

        def __enter__(self):
            return self

        def execute(self, request):
            events.append(request["request_id"])
            miss = request["is_revisit"] and scheme != "hbm"
            return SimpleNamespace(
                metrics={
                    **request,
                    "prefix_cache_hit": miss,
                    "cache_diagnostics": {
                        "candidate_persistence": "gpu_transient",
                        "retained_length": 65536,
                        "host_to_device_bytes": 3456 if miss else 0,
                        "device_to_host_bytes": 0,
                        "prefetched_records": 2 if miss else 0,
                        "recalled_records": 1 if miss else 0,
                    },
                }
            )

        def __exit__(self, *_):
            events.append("release_sessions")

    workload = SimpleNamespace(
        requests=[
            {
                "request_id": index,
                "user_id": index % 16,
                "visit_index": index // 16,
                "is_revisit": index >= 16,
                "input_sha256": f"request-{index}",
            }
            for index in range(32)
        ]
    )
    trace = warmup(backend, workload, config, Runner)
    assert events == [0, 1, 16, "release_sessions", "synchronize", "release_shared"]
    assert [row["request_id"] for row in trace] == [0, 1, 16]
    if scheme != "hbm":
        assert trace[2]["host_to_device_bytes"] > 0 and trace[2]["recalled_records"] > 0
        trace[2]["recalled_records"] = 0
        with pytest.raises(ValueError, match="did not exercise host recall"):
            validate_warmup_trace(scheme, trace, config, workload.requests)
