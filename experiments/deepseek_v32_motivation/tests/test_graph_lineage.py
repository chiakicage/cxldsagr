"""Process, clone and NVTX evidence jointly identify every replayed node."""

import sqlite3

import pytest

from experiments.deepseek_v32_motivation.src.graph_attribution import (
    GraphLineage,
    attribute_graph_replays,
    original_node,
    read_lineage,
)

PROCESS = 12 << 24
STAGE = "compute_graph_projection_layer_0_q_128"


def replay_fixture():
    calls, activities, scopes = [], [], []
    parents = GraphLineage({401: 301, 301: 101, 499: 302, 302: 102})
    parents.process = PROCESS
    for index in range(2):
        label = f"replay{index}"
        replay = {
            "nvtx": label,
            "stage": STAGE,
            "graph_replay": True,
            "graph_id": 7,
            "graph_gpu_node_ids": [101, 102],
        }
        api = {
            "nvtx": label + "/api",
            "stage": "q_a_proj",
            "graph_api": True,
            "graph_replay_nvtx": label,
            "graph_node_ids": [101],
        }
        calls.extend((replay, api))
        scope = {"label": label, "stage": STAGE}
        scopes.append(scope)
        activities.extend(
            [
                {
                    "scope": scope,
                    "graph_id": 7,
                    "graph_node_id": 401,
                    "process": PROCESS,
                    "kind": "kernel",
                },
                {
                    "scope": scope,
                    "graph_id": None,
                    "graph_node_id": 499,
                    "process": PROCESS,
                    "kind": "memset",
                },
            ]
        )
    return calls, activities, parents, scopes


def test_repeated_replays_resolve_clone_chains_and_copy_nodes_without_graph_id():
    calls, activities, parents, scopes = replay_fixture()
    audit = attribute_graph_replays(activities, calls, parents, scopes=scopes, require_replays=True)
    assert audit["replays"] == 2
    assert audit["gpu_node_activities"] == 4
    assert audit["copy_nodes_without_graph_id"] == 2
    assert activities[0]["graph_call"] is calls[1]
    assert activities[1]["graph_call"] is None
    assert activities[2]["graph_call"] is calls[3]


@pytest.mark.parametrize(
    "damage", ["process", "kernel_id", "lineage", "nvtx", "all_work", "all_records"]
)
def test_missing_or_conflicting_evidence_fails(damage):
    calls, activities, parents, scopes = replay_fixture()
    if damage == "process":
        activities[0]["process"] += 1 << 24
    elif damage == "kernel_id":
        activities[0]["graph_id"] = None
    elif damage == "lineage":
        parents.clear()
    elif damage == "nvtx":
        calls, activities = calls[2:], activities[2:]
    elif damage == "all_work":
        activities = []
    elif damage == "all_records":
        calls, activities, scopes = [], [], []
    with pytest.raises(ValueError):
        attribute_graph_replays(activities, calls, parents, scopes=scopes, require_replays=True)


def setup_db(path, rows):
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE CUDA_GRAPH_NODE_EVENTS(graphNodeId INTEGER, originalGraphNodeId INTEGER, globalTid INTEGER)"
        )
        db.executemany("INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (?,?,?)", rows)
    return path


def test_setup_files_require_one_process_and_unambiguous_clone_parents(tmp_path):
    first = setup_db(tmp_path / "one.sqlite", [(10, None, PROCESS + 7), (21, 10, PROCESS + 7)])
    second = setup_db(tmp_path / "two.sqlite", [(40, 21, PROCESS + 8)])
    edges = read_lineage([first, second])
    assert edges.process == PROCESS
    assert original_node(40, edges) == 10
    assert original_node(41, edges) == 41  # Adjacent numeric IDs are not evidence.
    wrong = setup_db(tmp_path / "wrong.sqlite", [(40, 21, PROCESS + (1 << 24))])
    with pytest.raises(ValueError, match="different processes"):
        read_lineage([first, wrong])
    ambiguous = setup_db(tmp_path / "ambiguous.sqlite", [(21, 11, PROCESS)])
    with pytest.raises(ValueError, match="ambiguous"):
        read_lineage([first, ambiguous])
    missing = setup_db(tmp_path / "missing.sqlite", [(21, 10, None)])
    with pytest.raises(ValueError, match="lacks process identity"):
        read_lineage([missing])
