"""Resolve replayed GPU nodes through Nsight's recorded clone lineage."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

from experiments.deepseek_v32_echo_prefill.src.analyze_nsys import _PROCESS_MASK
from experiments.deepseek_v32_motivation.src.graph_instrumentation import REPLAY_PATTERN


class GraphLineage(dict):
    """Node clone edges from exactly one traced process."""

    process = None


def read_lineage(paths):
    parents = GraphLineage()
    for path in paths:
        with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
            tables = {
                row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "CUDA_GRAPH_NODE_EVENTS" not in tables:
                raise ValueError(f"graph setup capture lacks CUDA_GRAPH_NODE_EVENTS: {path}")
            for node, parent, thread in db.execute(
                "SELECT graphNodeId, originalGraphNodeId, globalTid FROM CUDA_GRAPH_NODE_EVENTS"
            ):
                if thread is None:
                    raise ValueError("graph node lineage lacks process identity")
                process = thread & _PROCESS_MASK
                if parents.process is not None and parents.process != process:
                    raise ValueError("graph setup captures contain different processes")
                parents.process = process
                if parent is None:
                    continue
                if node in parents and parents[node] != parent:
                    raise ValueError("ambiguous CUDA graph clone lineage")
                parents[node] = parent
    return parents


def original_node(node, parents):
    seen = set()
    while node in parents:
        if node in seen:
            raise ValueError("cyclic CUDA graph clone lineage")
        seen.add(node)
        node = parents[node]
    return node


def attribute_graph_replays(activities, calls, parents, *, scopes=None, require_replays=False):
    """Require each expected GPU node exactly once for every observed replay.

    Existing launch correlation supplies the replay's actual CPU scope. Only
    measured GPU activity intervals are used; synthetic API ledger entries have
    no claimed CPU duration or NVTX matrix range during replay.
    """
    replays = {row["nvtx"]: row for row in calls if row.get("graph_replay")}
    if len(replays) != sum(bool(row.get("graph_replay")) for row in calls):
        raise ValueError("duplicate graph replay scope")
    if require_replays and not replays:
        raise ValueError("graph-enabled request lacks replay records")
    if scopes is not None:
        labels = [scope["label"] for scope in scopes if REPLAY_PATTERN.fullmatch(scope["stage"])]
        if len(labels) != len(set(labels)) or set(labels) != replays.keys():
            raise ValueError("observed NVTX graph replay scopes differ from the ledger")
    api_nodes = defaultdict(dict)
    for call in calls:
        if not call.get("graph_api"):
            continue
        label = call["graph_replay_nvtx"]
        if label not in replays:
            raise ValueError("graph matrix API has no replay")
        for node in call["graph_node_ids"]:
            if node in api_nodes[label]:
                raise ValueError("graph matrix APIs claim the same node")
            api_nodes[label][node] = call
    observed = defaultdict(list)
    node_only_copy_activities = 0
    for activity in activities:
        if not activity.get("graph_id") and not activity.get("graph_node_id"):
            continue
        scope = activity.get("scope")
        label = scope["label"] if scope else None
        if label not in replays:
            raise ValueError("graph GPU activity has no instrumented replay scope")
        replay = replays[label]
        if (
            getattr(parents, "process", None) is not None
            and activity.get("process") != parents.process
        ):
            raise ValueError("graph setup and replay activities belong to different processes")
        graph_id = activity.get("graph_id")
        if graph_id:
            if graph_id != replay["graph_id"]:
                raise ValueError("replayed graph ID differs from the allocated executable")
        elif activity["kind"] in ("memcpy", "memset"):
            # Nsight's memcpy/memset tables may export graphNodeId without
            # graphId. Exact node lineage, process, launch and complete graph
            # membership still identify the work; do not decode a numeric ID.
            node_only_copy_activities += 1
        else:
            raise ValueError("graph kernel activity lacks executable graph ID")
        node = original_node(activity["graph_node_id"], parents)
        observed[label].append(node)
        activity["graph_capture_node_id"] = node
        activity["graph_call"] = api_nodes[label].get(node)
        if activity["graph_call"] is not None:
            activity["graph_stage"] = activity["graph_call"]["stage"]
    for label, replay in replays.items():
        expected = set(replay["graph_gpu_node_ids"])
        actual = observed[label]
        if not expected or len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"missing, duplicate or unknown GPU graph nodes for replay {label}")
        if not api_nodes[label] or not api_nodes[label].keys() <= expected:
            raise ValueError("graph API nodes do not match the complete graph")
    return {
        "replays": len(replays),
        "matrix_api_invocations": sum(bool(row.get("graph_api")) for row in calls),
        "gpu_node_activities": sum(map(len, observed.values())),
        "clone_edges": len(parents),
        "copy_nodes_without_graph_id": node_only_copy_activities,
        "every_replay_gpu_node_verified": True,
    }


def profile_graph_inputs(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    setups = metadata.get("graph_setup_captures", [])
    if not setups:
        raise ValueError("graph replay analysis requires recorded setup captures")
    paths = [directory / item["sqlite"] for item in setups]
    calls = json.loads((directory / "operator_calls.json").read_text())
    return calls, read_lineage(paths), paths
