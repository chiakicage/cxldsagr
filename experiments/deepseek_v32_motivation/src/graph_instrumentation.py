"""Capture matrix API ownership without changing executable CUDA Graph nodes."""

from __future__ import annotations

import ctypes as ct
import hashlib
import re
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPLAY_PATTERN = re.compile(r"compute_graph_(projection|finish)_layer_(\d+)_q_(\d+)")


@contextmanager
def instrument_graph_queries(bank, operators):
    """Record the real replay query origin before eager MLA consumes it.

    Replay bypasses CheckpointAttention.project, which normally supplies this
    metadata. The graph owns a stable Q allocation; the cache's host-visible
    written length is the same position passed to the projection callback.
    """
    from experiments.deepseek_v32_echo_prefill.src.operator_instrumentation import QueryOrigin

    original = bank.forward_block

    def forward_block(layer, runner, hidden, residual, **kwargs):
        pair = bank.pairs[layer, len(hidden)]
        origin = QueryOrigin.capture(pair.projected.q, runner.cache.written)
        operators.queries[origin.storage_key] = origin
        return original(layer, runner, hidden, residual, **kwargs)

    with patch.object(bank, "forward_block", forward_block):
        yield


def _loaded_library(stem, *, nsight=False):
    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if stem in line and line.split()[-1].startswith("/")
    }
    if nsight:
        paths = {path for path in paths if "/nsight-systems/" in path}
    if len(paths) != 1:
        raise RuntimeError(f"need one already-loaded {stem} provider, found {sorted(paths)}")
    path = paths.pop()
    return ct.CDLL(path), path


class GraphInspector:
    """Use the same loaded CUPTI as Nsight; never subscribe or alter nodes."""

    def __init__(self):
        self.runtime, runtime_path = _loaded_library("libcudart.so")
        self.cupti, cupti_path = _loaded_library("libcupti.so", nsight=True)
        pointer = ct.c_void_p
        self.runtime.cudaRuntimeGetVersion.argtypes = [ct.POINTER(ct.c_int)]
        version = ct.c_int()
        self._check(self.runtime.cudaRuntimeGetVersion(ct.byref(version)), "runtime version")
        # CUDA 13 exports the seven-argument API without the v2 suffix.
        self.capture_v3 = version.value >= 13000
        name = "cudaStreamGetCaptureInfo" if self.capture_v3 else "cudaStreamGetCaptureInfo_v2"
        self.capture_info = getattr(self.runtime, name)
        args = [
            pointer,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_ulonglong),
            ct.POINTER(pointer),
            ct.POINTER(pointer),
        ]
        if self.capture_v3:
            args.append(ct.POINTER(pointer))
        self.capture_info.argtypes = [*args, ct.POINTER(ct.c_size_t)]
        self.runtime.cudaGraphGetNodes.argtypes = [
            pointer,
            ct.POINTER(pointer),
            ct.POINTER(ct.c_size_t),
        ]
        self.runtime.cudaGraphNodeGetType.argtypes = [pointer, ct.POINTER(ct.c_int)]
        self.cupti.cuptiGetGraphId.argtypes = [pointer, ct.POINTER(ct.c_uint32)]
        self.cupti.cuptiGetGraphExecId.argtypes = [pointer, ct.POINTER(ct.c_uint32)]
        self.cupti.cuptiGetGraphNodeId.argtypes = [pointer, ct.POINTER(ct.c_uint64)]
        self.provenance = {
            "cuda_runtime_version": version.value,
            "capture_info_api": name,
            "libraries": {
                path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in (runtime_path, cupti_path)
            },
        }

    @staticmethod
    def _check(result, operation):
        if result:
            raise RuntimeError(f"{operation} failed with status {result}")

    def snapshot(self, stream):
        status, capture_id, graph = ct.c_int(), ct.c_ulonglong(), ct.c_void_p()
        dependencies, size = ct.c_void_p(), ct.c_size_t()
        args = [
            stream,
            ct.byref(status),
            ct.byref(capture_id),
            ct.byref(graph),
            ct.byref(dependencies),
        ]
        if self.capture_v3:
            args.append(None)
        self._check(self.capture_info(*args, ct.byref(size)), "capture info")
        if status.value != 1 or not graph.value:
            raise RuntimeError("graph API attribution requires a live capture")
        graph_id = ct.c_uint32()
        self._check(self.cupti.cuptiGetGraphId(graph, ct.byref(graph_id)), "CUPTI graph ID")
        size = ct.c_size_t()
        self._check(self.runtime.cudaGraphGetNodes(graph, None, ct.byref(size)), "graph node count")
        nodes = (ct.c_void_p * size.value)()
        if size.value:
            self._check(self.runtime.cudaGraphGetNodes(graph, nodes, ct.byref(size)), "graph nodes")
        result = {}
        for node in nodes:
            node_id, kind = ct.c_uint64(), ct.c_int()
            self._check(self.cupti.cuptiGetGraphNodeId(node, ct.byref(node_id)), "CUPTI node ID")
            self._check(self.runtime.cudaGraphNodeGetType(node, ct.byref(kind)), "graph node type")
            if kind.value not in (0, 1, 2, 5, 6, 7):
                raise RuntimeError(
                    f"unsupported graph node type {kind.value}; no inferred ownership"
                )
            result[node_id.value] = kind.value
        return graph_id.value, result

    def executable_id(self, graph):
        result = ct.c_uint32()
        self._check(
            self.cupti.cuptiGetGraphExecId(graph.raw_cuda_graph_exec(), ct.byref(result)),
            "CUPTI executable graph ID",
        )
        return result.value


def operator_model(backend):
    return SimpleNamespace(
        blocks=[
            SimpleNamespace(attention=SimpleNamespace(attention=attn), mlp=block.mlp)
            for attn, block in zip(backend.attentions, backend.blocks, strict=True)
        ],
        head_weight=backend.head_weight,
    )


def independent_weight_identity(attention, block):
    """Identify the matrix operands actually captured for this layer."""
    weights = {}
    for name in ("wq_a", "wq_b", "wkv_a", "index_wq", "index_wk", "wo"):
        weights[f"attention.{name}"] = getattr(attention, name).weight
    for name in ("wk_b", "wv_b", "index_head_weight"):
        weights[f"attention.{name}"] = getattr(attention, name)
    for name in ("gate", "up", "down"):
        weights[f"mlp.{name}"] = getattr(block.mlp, name).weight
    return {
        name: {
            "data_ptr": weight.data_ptr(),
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
        }
        for name, weight in weights.items()
    }


class CaptureGraphOperators:
    """Install only while allocating graphs, before measured request scopes."""

    def __init__(self, backend):
        self.backend = backend
        self.graphs = {}
        self.current = None
        self.inspector = None

    @contextmanager
    def graph_scope(self, bank, pair, part):
        import torch

        if self.current is not None:
            raise RuntimeError("nested graph captures are not supported")
        key = part, pair.layer, pair.queries
        if key in self.graphs:
            raise RuntimeError("duplicate graph identity during one allocation")
        stream = torch.cuda.current_stream(bank.device).cuda_stream
        graph_id, before = self.inspector.snapshot(stream)
        if before:
            raise RuntimeError("capture hook must precede every graph node")
        self.current = {"key": key, "graph_id": graph_id, "stream": stream, "operators": []}
        try:
            yield
            end_id, nodes = self.inspector.snapshot(stream)
            if end_id != graph_id:
                raise RuntimeError("graph identity changed during capture")
            operators = self.current["operators"]
            claimed = [node for op in operators for node in op["graph_node_ids"]]
            gpu_nodes = sorted(node for node, kind in nodes.items() if kind in (0, 1, 2))
            if (
                not operators
                or len(claimed) != len(set(claimed))
                or not set(claimed) <= set(gpu_nodes)
            ):
                raise RuntimeError("captured matrix node ownership is absent or ambiguous")
            self.graphs[key] = {
                "part": part,
                "layer": pair.layer,
                "queries": pair.queries,
                "capture_graph_id": graph_id,
                "gpu_node_ids": gpu_nodes,
                "node_types": nodes,
                "operators": operators,
                "residual_present": pair.residual_present,
                "independent_weight_identity": independent_weight_identity(
                    bank.attentions[pair.layer], bank.blocks[pair.layer]
                ),
            }
        finally:
            self.current = None

    @contextmanager
    def __call__(self, stage, **details):
        # Warmup and nonmatrix calls still execute, but are not matrix templates.
        if self.current is None or details.get("useful_flops") is None:
            yield
            return
        graph_id, before = self.inspector.snapshot(self.current["stream"])
        yield
        end_id, after = self.inspector.snapshot(self.current["stream"])
        if (
            graph_id != end_id
            or graph_id != self.current["graph_id"]
            or not before.keys() <= after.keys()
        ):
            raise RuntimeError("captured API changed graph identity or removed nodes")
        nodes = sorted(node for node in after.keys() - before.keys() if after[node] in (0, 1, 2))
        if not nodes:
            raise RuntimeError(f"matrix API {stage} captured no GPU nodes")
        self.current["operators"].append({"stage": stage, **details, "graph_node_ids": nodes})

    def __enter__(self):
        from experiments.deepseek_v32_echo_prefill.src.operator_instrumentation import (
            InstrumentOperators,
        )
        from models.deepseek_v32.execution.compute_graphs import DeepSeekComputeGraphs

        self.inspector = GraphInspector()
        self.stack = ExitStack()

        def capture_scope(bank, pair, part):
            return self.graph_scope(bank, pair, part)

        try:
            self.stack.enter_context(
                patch.object(DeepSeekComputeGraphs, "_capture_scope", capture_scope)
            )
            self.stack.enter_context(InstrumentOperators(operator_model(self.backend), self))
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def finalize(self):
        bank = self.backend._compute_graphs
        expected = {
            (part, layer, q) for layer, q in bank.pairs for part in ("projection", "finish")
        }
        if self.graphs.keys() != expected:
            raise RuntimeError("not every allocated compute graph has a capture ledger")
        identities = {}
        for key, entry in self.graphs.items():
            part, layer, queries = key
            for name, weight in entry["independent_weight_identity"].items():
                pointer = weight["data_ptr"]
                previous = identities.setdefault((name, pointer), layer)
                if previous != layer:
                    raise RuntimeError("independent layer graphs share a matrix weight operand")
            graph = getattr(bank.pairs[layer, queries], f"{part}_graph")
            entry["executable_graph_id"] = self.inspector.executable_id(graph)
        return {
            "scheme": self.backend.scheme,
            "provider": self.inspector.provenance,
            "graphs": list(self.graphs.values()),
        }

    def expand_replay(self, replay, next_call_id):
        match = REPLAY_PATTERN.fullmatch(replay["stage"])
        if not match:
            return []
        key = match[1], int(match[2]), int(match[3])
        if key not in self.graphs:
            raise RuntimeError(f"missing capture-time graph template: {key}")
        graph = self.graphs[key]
        replay.update(
            graph_replay=True,
            graph_id=graph["executable_graph_id"],
            graph_capture_id=graph["capture_graph_id"],
            graph_gpu_node_ids=graph["gpu_node_ids"],
        )
        return [
            {
                **replay,
                **op,
                "call_id": next_call_id + index,
                "parent_call_id": replay["call_id"],
                "graph_replay": False,
                "graph_api": True,
                "graph_replay_nvtx": replay["nvtx"],
                "nvtx": f"{replay['nvtx']}/graph_api_{index}_{op['stage']}",
                "cpu_inclusive_ns": 0,
                "cpu_scope_kind": "capture_template_instantiated_for_observed_replay",
            }
            for index, op in enumerate(graph["operators"])
        ]
