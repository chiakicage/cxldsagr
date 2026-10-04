"""Reject missing independent work even when aggregate FLOPs still match."""

from copy import deepcopy

import pytest

from experiments.deepseek_v32_motivation.src.flops import phase_ledger
from experiments.deepseek_v32_motivation.src.verify_profile_flops import verify_calls
from experiments.deepseek_v32_motivation.tests.test_flops import MODEL, PRECISIONS


def captured_candidate():
    config = {
        "history_tokens": 4096,
        "candidate_tokens": 128,
        "chunk_size": 1024,
        "layers": 4,
        "sparse_pool_tokens": 4096,
        "padded_history_tokens": 4096,
        "num_users": 2,
    }
    work = {"config": config, "model_config": MODEL, "linear_precisions_by_source": PRECISIONS}
    capture = {
        "capture_index": 3,
        "scheme": "echo",
        "phase": "revisit",
        "runner_metrics": {"prefix_cache_hit": True},
    }
    calls = []
    for row in phase_ledger(MODEL, config, PRECISIONS, scheme="echo", phase="candidate"):
        if row["useful_flops"] is None:
            continue
        layers = (
            [-1]
            if row["source_layer"] == -1
            else [i for i in range(4) if i % 3 == row["source_layer"]]
        )
        for layer in layers:
            calls.append(
                {
                    **capture,
                    "call_id": len(calls),
                    "segment": "candidate",
                    "chunk": "0",
                    "layer": str(layer),
                    "stage": row["name"],
                    "precision": row["precision"].upper(),
                    "useful_flops": row["useful_flops"] // len(layers),
                    "graph_api": row["name"] not in ("indexer_prefetch", "sparse_mla", "lm_head"),
                }
            )
    return calls, [capture], work


def test_graph_and_eager_work_mix_conserves_every_independent_layer():
    calls, captures, work = captured_candidate()
    result = verify_calls(calls, captures, work)
    assert result["passed"] and result["graph_matrix_calls"] > 0
    assert result["matrix_calls"] == len(calls)
    # Duplicating layer 0 and dropping its independent copy preserves total
    # work, source-layer work, dtype counts, and API count. It must still fail.
    for row in calls:
        if row["layer"] == "3":
            row["layer"] = "0"
    with pytest.raises(ValueError, match="operator groups"):
        verify_calls(calls, captures, work)


def test_sparse_mla_splits_must_preserve_useful_flops():
    calls, captures, work = captured_candidate()
    row = next(row for row in calls if row["stage"] == "sparse_mla")
    tail = deepcopy(row)
    tail["call_id"] = len(calls)
    row["useful_flops"] //= 2
    tail["useful_flops"] -= row["useful_flops"]
    calls.append(tail)
    assert verify_calls(calls, captures, work)["passed"]
    tail["useful_flops"] += 1
    assert not verify_calls(calls, captures, work)["passed"]


def test_wrong_precision_and_missing_history_fail_before_mfu():
    calls, captures, work = captured_candidate()
    calls[0]["precision"] = "FP32"
    with pytest.raises(ValueError, match="operator groups"):
        verify_calls(calls, captures, work)
    calls, captures, work = captured_candidate()
    captures[0]["runner_metrics"]["prefix_cache_hit"] = False
    with pytest.raises(ValueError, match="admission trajectory"):
        verify_calls(calls, captures, work)
