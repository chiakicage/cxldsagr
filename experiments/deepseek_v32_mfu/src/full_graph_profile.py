"""Capture and verify full-extend graph stage ownership through native node IDs."""

from __future__ import annotations

import re
from contextlib import contextmanager

import torch

from experiments.deepseek_v32_motivation.src.graph_attribution import original_node
from experiments.deepseek_v32_motivation.src.graph_instrumentation import GraphInspector

REPLAY_PATTERN = re.compile(r"extend_graph_replay_q_(\d+)")
STAGES = {
    "indexer": "indexer_qk",
    "indexer_prefetch": "indexer_fused",
    "input_residual_norm": "residual_rms_norm",
    "post_attention_residual_norm": "residual_rms_norm",
    "dense_mlp": "mlp",
}


class FullExtendGraphCapture:
    """Inspect only active capture; retain innermost scope for every GPU node."""

    def __init__(self, method, *, record_operators=False):
        self.method = method
        self.inspector = GraphInspector()
        self.graph_id = None
        self.nodes = {}
        self.owners = {}
        self.stack = []
        self.template = None
        self.record_operators = record_operators
        self.operators = []

    @contextmanager
    def __call__(self, stage, **details):
        if not torch.cuda.is_current_stream_capturing():
            yield
            return
        if details:
            # InstrumentOperators also wraps scalar helpers. Preserve the model's
            # stage ownership while recording only complete matrix API boundaries.
            if not self.record_operators or details.get("useful_flops") is None:
                yield
            else:
                with self.matrix_scope(stage, details):
                    yield
            return
        stream = torch.cuda.current_stream().cuda_stream
        graph_id, before = self.inspector.snapshot(stream)
        if self.graph_id is None:
            if stage != "extend_graph_body" or before:
                raise RuntimeError("full graph scope must enclose every captured GPU node")
            self.graph_id = graph_id
        if graph_id != self.graph_id:
            raise RuntimeError("full graph stage changed the active capture")
        inherited = self.stack[-1][1] if self.stack else "shared"
        layer = stage if re.fullmatch(r"layer_\d+", stage) else inherited
        self.stack.append((stage, layer))
        try:
            yield
            after_id, after = self.inspector.snapshot(stream)
            if after_id != graph_id or not before.keys() <= after.keys():
                raise RuntimeError("full graph node identity changed during a stage")
            for node in after.keys() - before.keys():
                if after[node] in (0, 1, 2) and node not in self.owners:
                    self.owners[node] = {
                        "stage": STAGES.get(stage, stage),
                        "source_stage": stage,
                        "stage_path": [name for name, _ in self.stack],
                        "layer": layer,
                    }
            self.nodes = after
        finally:
            self.stack.pop()

    @contextmanager
    def matrix_scope(self, stage, details):
        if not self.stack or self.graph_id is None:
            raise RuntimeError("matrix API capture must be inside the full graph body")
        stream = torch.cuda.current_stream().cuda_stream
        graph_id, before = self.inspector.snapshot(stream)
        yield
        after_id, after = self.inspector.snapshot(stream)
        if graph_id != self.graph_id or after_id != graph_id or not before.keys() <= after.keys():
            raise RuntimeError("matrix API changed full graph identity or removed nodes")
        nodes = sorted(node for node in after.keys() - before.keys() if after[node] in (0, 1, 2))
        if not nodes:
            raise RuntimeError(f"matrix API {stage} captured no GPU nodes")
        self.operators.append(
            {
                "stage": stage,
                "layer": self.stack[-1][1],
                **details,
                "graph_node_ids": nodes,
            }
        )

    def finalize(self, graph):
        gpu_nodes = sorted(node for node, kind in self.nodes.items() if kind in (0, 1, 2))
        if not gpu_nodes or set(gpu_nodes) != self.owners.keys() or self.stack:
            raise RuntimeError("full graph node ownership is incomplete")
        if any(owner["stage"] == "extend_graph_body" for owner in self.owners.values()):
            raise RuntimeError("full graph contains GPU work outside every named stage")
        operators = getattr(self, "operators", [])
        claimed = [node for operator in operators for node in operator["graph_node_ids"]]
        if len(claimed) != len(set(claimed)) or not set(claimed) <= set(gpu_nodes):
            raise RuntimeError("full graph matrix API node ownership is ambiguous")
        if getattr(self, "record_operators", False) and not operators:
            raise RuntimeError("full graph matrix API capture contains no operators")
        self.template = {
            "method": self.method,
            "capture_graph_id": self.graph_id,
            "executable_graph_id": self.inspector.executable_id(graph.graph),
            "gpu_node_ids": gpu_nodes,
            "node_types": self.nodes,
            "node_owners": {str(node): owner for node, owner in self.owners.items()},
            "provider": self.inspector.provenance,
            "runtime": graph.describe(),
        }
        if getattr(self, "record_operators", False):
            self.template["operators"] = operators
        return self.template

    def expand_replay(self, replay, next_call_id):
        if not REPLAY_PATTERN.fullmatch(replay["stage"]):
            return []
        if self.template is None:
            raise RuntimeError("full graph replay lacks a finalized capture ledger")
        replay.update(
            graph_replay=True,
            full_extend_graph=True,
            graph_id=self.template["executable_graph_id"],
            graph_capture_id=self.template["capture_graph_id"],
            graph_gpu_node_ids=self.template["gpu_node_ids"],
            graph_node_owners=self.template["node_owners"],
        )
        return [
            {
                **replay,
                **operator,
                "call_id": next_call_id + index,
                "parent_call_id": replay["call_id"],
                "graph_replay": False,
                "full_extend_graph": False,
                "graph_api": True,
                "graph_replay_nvtx": replay["nvtx"],
                "nvtx": f"{replay['nvtx']}/graph_api_{index}_{operator['stage']}",
                "cpu_inclusive_ns": 0,
                "cpu_scope_kind": "capture_template_instantiated_for_observed_replay",
            }
            for index, operator in enumerate(self.template.get("operators", []))
        ]


def attribute_full_graph_replays(activities, calls, parents, *, scopes=None, apis=None):
    """Verify exact node membership and one launch; annotate GPU stage/layer."""
    replays = [row for row in calls if row.get("full_extend_graph")]
    if len(replays) != 1:
        raise ValueError("extend must have exactly one full graph replay")
    replay = replays[0]
    node_calls = {}
    for call in calls:
        if not call.get("graph_api"):
            continue
        if call.get("graph_replay_nvtx") != replay["nvtx"]:
            raise ValueError("matrix API belongs to an unexpected full graph replay")
        nodes = call.get("graph_node_ids")
        if not nodes or len(nodes) != len(set(nodes)):
            raise ValueError("full graph matrix API has missing or duplicate nodes")
        for node in nodes:
            if node not in replay["graph_gpu_node_ids"] or node in node_calls:
                raise ValueError("full graph matrix API node ownership is ambiguous")
            node_calls[node] = call
    if scopes is not None:
        observed = [
            row["label"] for row in scopes if REPLAY_PATTERN.fullmatch(row.get("stage") or "")
        ]
        if observed != [replay["nvtx"]]:
            raise ValueError("observed full graph replay scopes differ from the ledger")
    if apis is not None:
        launches = [
            api
            for api in apis
            if "GraphLaunch" in api["name"] and api.get("scope", {}).get("label") == replay["nvtx"]
        ]
        if len(launches) != 1:
            raise ValueError("extend replay does not contain exactly one CUDA graph launch")
    seen = []
    outside = []
    for activity in activities:
        node_id = activity.get("graph_node_id")
        graph_id = activity.get("graph_id")
        if not node_id and not graph_id:
            outside.append(activity)
            continue
        if activity.get("scope", {}).get("label") != replay["nvtx"]:
            raise ValueError("full graph node lies outside its replay scope")
        if graph_id and graph_id != replay["graph_id"]:
            raise ValueError("extend launched an unexpected executable graph")
        if (
            getattr(parents, "process", None) is not None
            and activity.get("process") != parents.process
        ):
            raise ValueError("full graph setup and replay belong to different processes")
        node = original_node(node_id, parents)
        owner = replay["graph_node_owners"].get(str(node))
        if owner is None:
            raise ValueError("full graph replay has an unknown GPU node")
        seen.append(node)
        activity["graph_capture_node_id"] = node
        activity["graph_stage"] = owner["stage"]
        activity["graph_source_stage"] = owner.get("source_stage", owner["stage"])
        activity["graph_scope_path"] = owner.get("stage_path", [])
        activity["graph_layer"] = owner["layer"]
        activity["full_extend_graph"] = True
        # Only capture-time matrix API boundaries establish operator ownership.
        # Stage-only gap captures deliberately keep graph_call unset.
        activity["graph_call"] = node_calls.get(node)
    if len(seen) != len(set(seen)) or set(seen) != set(replay["graph_gpu_node_ids"]):
        raise ValueError("missing or duplicate GPU nodes in full graph replay")
    unexpected = [
        activity
        for activity in outside
        if activity.get("scope", {}).get("stage") != "extend_graph_inputs"
    ]
    if unexpected:
        raise ValueError("extend executed GPU work outside its full graph/input staging")
    return {
        "replays": 1,
        "gpu_node_activities": len(seen),
        "every_replay_gpu_node_verified": True,
        "full_extend_graph": True,
        "outside_graph_activities": len(outside),
        "outside_graph_stages": sorted({row.get("scope", {}).get("stage") for row in outside}),
        "matrix_api_count": sum(bool(call.get("graph_api")) for call in calls),
        "matrix_api_gpu_node_activities": len(node_calls),
    }
