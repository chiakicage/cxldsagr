"""Graph replay attribution must conserve real GPU activities and work."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_motivation.src.graph_attribution import (
    attribute_graph_replays,
    original_node,
)
from experiments.deepseek_v32_motivation.src.graph_instrumentation import (
    CaptureGraphOperators,
    instrument_graph_queries,
)
from experiments.deepseek_v32_motivation.src.operator_mfu import invocation_row


def replay_fixture():
    replay = {
        "nvtx": "replay",
        "graph_replay": True,
        "graph_id": 22,
        "graph_gpu_node_ids": [10, 11, 12],
    }
    call = {
        "graph_api": True,
        "graph_replay_nvtx": "replay",
        "graph_node_ids": [10, 11],
        "capture_index": 2,
        "scheme": "hbm",
        "phase": "cold",
        "segment": "history",
        "stage": "q_a_proj",
        "precision": "FP8",
        "chunk": "0",
        "layer": "0",
        "call_id": 4,
        "nvtx": "replay/api",
        "useful_flops": 197900000,
        "executed_matmul_flops": None,
        "dimensions": {"M": 1, "N": 1, "K": 1},
    }
    activities = [
        {
            "id": index,
            "graph_id": 22,
            "graph_node_id": index + 20,
            "kind": "kernel",
            "name": name,
            "start": 1000 + index * 200,
            "end": 1100 + index * 200,
            "device_id": 0,
            "scope": {"label": "replay", "start": 0, "end": 10},
        }
        for index, name in enumerate(("sm90_fp8_gemm_1d2d_impl", "quantizer", "norm"))
    ]
    return [replay, call], activities, {20: 10, 21: 11, 22: 12}


def test_graph_lineage_assigns_actual_replay_intervals_and_preserves_nonmatrix_nodes():
    calls, activities, parents = replay_fixture()
    before = [(a["id"], a["start"], a["end"]) for a in activities]
    audit = attribute_graph_replays(activities, calls, parents)
    assert audit["gpu_node_activities"] == 3
    assert before == [(a["id"], a["start"], a["end"]) for a in activities]
    assert activities[-1]["graph_call"] is None
    matrix = [a for a in activities if a["graph_call"] is not None]
    row = invocation_row(calls[1], matrix, {"fp8": 1979})
    assert row["useful_flops"] == calls[1]["useful_flops"]
    assert row["operator_gpu_active_ns"] == 200
    assert row["operator_gpu_span_ns"] == 300
    assert row["primary_kernel_ns"] == 100


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unknown", "wrong_graph"])
def test_graph_node_inventory_rejects_partial_or_ambiguous_timing(corruption):
    calls, activities, parents = replay_fixture()
    if corruption == "missing":
        activities.pop()
    elif corruption == "duplicate":
        activities.append(deepcopy(activities[0]))
    elif corruption == "unknown":
        activities[0]["graph_node_id"] = 100
    else:
        activities[0]["graph_id"] = 99
    with pytest.raises(ValueError):
        attribute_graph_replays(activities, calls, parents)


def test_graph_api_ownership_and_clone_cycles_fail_closed():
    calls, activities, parents = replay_fixture()
    calls.append(deepcopy(calls[1]))
    with pytest.raises(ValueError, match="same node"):
        attribute_graph_replays(activities, calls, parents)
    with pytest.raises(ValueError, match="cyclic"):
        original_node(1, {1: 2, 2: 1})


def test_replay_expansion_preserves_work_and_has_no_fake_cpu_scope():
    registry = CaptureGraphOperators(None)
    registry.graphs["projection", 3, 128] = {
        "executable_graph_id": 9,
        "capture_graph_id": 8,
        "gpu_node_ids": [100, 101],
        "operators": [
            {"stage": "q_a_proj", "precision": "FP8", "useful_flops": 500, "graph_node_ids": [100]},
            {
                "stage": "q_absorb",
                "precision": "BF16",
                "useful_flops": 400,
                "graph_node_ids": [101],
            },
        ],
    }
    expanded = []
    for index in range(2):
        replay = {
            "stage": "compute_graph_projection_layer_3_q_128",
            "call_id": index * 3,
            "nvtx": f"replay_{index}",
            "cpu_start_ns": 100 + index,
        }
        expanded.extend(registry.expand_replay(replay, index * 3 + 1))
        assert replay["graph_replay"] and replay["graph_id"] == 9
    assert len({row["nvtx"] for row in expanded}) == len(expanded)
    assert len({row["call_id"] for row in expanded}) == len(expanded)
    assert sum(row["useful_flops"] for row in expanded) == 1800
    assert all(row["cpu_inclusive_ns"] == 0 for row in expanded)
    assert all(row["graph_api"] and not row["graph_replay"] for row in expanded)


def test_graph_query_origin_tracks_each_replay_position_and_slices_before_mla():
    q = torch.empty((4, 2, 8))
    operators = SimpleNamespace(queries={})
    runner = SimpleNamespace(cache=SimpleNamespace(written=64))
    observed = []

    def original(layer, supplied_runner, hidden, residual, **kwargs):
        assert supplied_runner is runner
        origin = next(iter(operators.queries.values()))
        observed.append(origin.resolve_start(q[2:]))
        return "result"

    bank = SimpleNamespace(
        pairs={(0, 4): SimpleNamespace(projected=SimpleNamespace(q=q))}, forward_block=original
    )
    with instrument_graph_queries(bank, operators):
        assert bank.forward_block(0, runner, [0] * 4, None) == "result"
        runner.cache.written = 128
        bank.forward_block(0, runner, [0] * 4, None)
    assert observed == [66, 130]
    assert bank.forward_block is original
