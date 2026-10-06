"""Backend-owned CUDA Graphs for pure DeepSeek block computation.

Cache allocation, indexing, selection, recall, writeback and transactions remain
eager. Graph outputs are borrowed; persistent KV writes receive an owned copy.
Value expansion writes directly into the finish graph's smaller static input.
The first three blocks pass activations through graph-owned output storage;
all layers share immutable positions and one dynamic start per query shape.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from time import perf_counter

import torch

from cache.prefix_pool import CacheBudgetExceeded
from models.deepseek_v32.execution.cache_resources import dense_staging_allocation_bytes
from models.deepseek_v32.nonmatrix import residual_rms_norm

DEFAULT_PRIVATE_LIMIT_BYTES = 12 * 2**30
GRAPH_POLICY_REVISION = "deepseek-compute-islands-v4-bound-inputs"


def _precision_policy():
    """Freeze policies which eager matrix calls consume when graphs are built."""
    matmul = torch.backends.cuda.matmul
    result = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_enabled": torch.is_autocast_enabled("cuda"),
        "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
    }
    for owner, name, label in (
        (matmul, "fp32_precision", "cuda_matmul_fp32_precision"),
        (matmul, "allow_tf32", "cuda_matmul_allow_tf32"),
        (torch.backends, "fp32_precision", "backends_fp32_precision"),
        (matmul, "allow_bf16_reduced_precision_reduction", "bf16_reduced_precision_reduction"),
        (matmul, "allow_fp16_reduced_precision_reduction", "fp16_reduced_precision_reduction"),
    ):
        try:
            result[label] = getattr(owner, name)
        except (AttributeError, RuntimeError) as error:
            # PyTorch rejects some mixed uses of its old and new TF32 APIs.
            # Preserve that state so changing APIs also invalidates a capture.
            result[label] = {"unavailable": str(error)}
    return result


def _shared_input_shapes(queries):
    return {
        "start": ((), torch.int64),
        "positions": ((queries,), torch.int64),
        "index_starts": ((queries,), torch.int32),
    }


def _owned_input_shapes(cfg, queries, layer):
    shapes = {
        "expanded": ((queries, cfg.n_heads, cfg.v_head_dim), torch.bfloat16),
    }
    # Layers 1 and 2 consume the previous finish graph's retained outputs.
    # A caller supplying independent activations can still copy into that
    # borrowed storage; graph outputs have always been borrowed, not durable.
    if layer not in (1, 2):
        shapes["hidden"] = ((queries, cfg.dim), torch.bfloat16)
        if layer % 3 != 0:
            shapes["residual"] = ((queries, cfg.dim), torch.bfloat16)
    return shapes


def plan_compute_graphs(cfg, layers, chunk_size, history, candidate, private_limit):
    """Pure shape/capacity plan; the private limit is a chosen allocation bound."""
    if any(
        type(value) is not int or value < 1
        for value in (layers, chunk_size, history, candidate, private_limit)
    ):
        raise ValueError("compute graph dimensions and private limit must be positive integers")
    queries = {min(chunk_size, history), candidate}
    if history % chunk_size:
        queries.add(history % chunk_size)
    static_storage = static_limit = 0
    for count in queries:
        shape_groups = [_shared_input_shapes(count)]
        shape_groups.extend(_owned_input_shapes(cfg, count, layer) for layer in range(layers))
        for shapes in shape_groups:
            for shape, dtype in shapes.values():
                elements = 1
                for size in shape:
                    elements *= size
                logical = elements * dtype.itemsize
                static_storage += logical
                static_limit += dense_staging_allocation_bytes(logical, "cuda")
    return {
        "compute_graphs_enabled": True,
        "compute_graph_policy_revision": GRAPH_POLICY_REVISION,
        "compute_graph_query_sizes": sorted(queries),
        "compute_graph_count": 2 * layers * len(queries),
        "compute_graph_static_storage_bytes": static_storage,
        "compute_graph_static_allocation_limit_bytes": static_limit,
        "compute_graph_private_limit_bytes": private_limit,
        "compute_graph_reserved_limit_bytes": static_limit + private_limit,
        "compute_graph_private_limit_kind": "chosen_upper_limit_not_observed_fixed_overhead",
    }


@dataclass
class _GraphPair:
    layer: int
    queries: int
    residual_present: bool
    inputs: dict
    owned_inputs: dict
    projection_graph: object = None
    finish_graph: object = None
    projected: object = None
    saved: object = None
    indexer_bounds: object = None
    output: object = None
    projection_replays: int = 0
    finish_replays: int = 0


class DeepSeekComputeGraphs:
    """Serial, backend-owned compute islands with independent layer weights."""

    def __init__(self, attentions, blocks, device, metadata):
        self.attentions = tuple(attentions)
        self.blocks = tuple(blocks)
        if len(self.attentions) != len(self.blocks) or not self.blocks:
            raise ValueError("compute graphs require matching independent layers")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("compute graphs require CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.metadata = dict(metadata)
        self.pairs = {}
        self._shared_inputs = {}
        self._start_positions = {}
        self.allocated = False
        self.closed = False
        self.failed = False
        self._active = False
        self._stream = self._last_stream = self._last_event = None
        self._capture_stream = None
        self.static_storage_bytes = self.static_allocated_bytes = self.private_reserved_bytes = 0
        self.memory_at_allocation = None
        self.setup_seconds = None
        self.eager_fallbacks = 0
        self.input_copies = self.position_updates = 0
        self.precision_policy = None

    def _check(self):
        if self.closed or self.failed:
            raise RuntimeError("compute graph resource is closed or failed")

    @torch.inference_mode()
    def allocate(self):
        self._check()
        if self.allocated or self.pairs:
            raise RuntimeError("compute graph resource has already allocated")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compute graph setup cannot run inside capture")
        free, total = torch.cuda.mem_get_info(self.device)
        if self.metadata["compute_graph_reserved_limit_bytes"] > free:
            raise CacheBudgetExceeded("chosen compute graph allocation limit exceeds free HBM")
        started = perf_counter()
        self.precision_policy = _precision_policy()
        self._capture_stream = torch.cuda.Stream(device=self.device)
        try:
            for queries in self.metadata["compute_graph_query_sizes"]:
                inputs = self._shared_inputs[queries] = {}
                for name, (shape, dtype) in _shared_input_shapes(queries).items():
                    inputs[name] = torch.zeros(shape, dtype=dtype, device=self.device)
                inputs["positions"].copy_(torch.arange(queries, device=self.device))
                self._start_positions[queries] = 0
            for layer, (attention, block) in enumerate(
                zip(self.attentions, self.blocks, strict=True)
            ):
                if block.is_moe:
                    raise ValueError("serving compute graphs require dense checkpoint blocks")
                for queries in self.metadata["compute_graph_query_sizes"]:
                    owned_inputs = {
                        name: torch.zeros(shape, dtype=dtype, device=self.device)
                        for name, (shape, dtype) in _owned_input_shapes(
                            attention.cfg, queries, layer
                        ).items()
                    }
                    inputs = {**self._shared_inputs[queries], **owned_inputs}
                    if layer in (1, 2):
                        inputs["hidden"], inputs["residual"] = self.pairs[layer - 1, queries].output
                    pair = _GraphPair(layer, queries, layer % 3 != 0, inputs, owned_inputs)
                    # Retain every partial allocation before any warmup/capture can fail.
                    self.pairs[layer, queries] = pair
                    self._capture_pair(pair, attention, block)
                    self._audit_capacity()
            torch.cuda.synchronize(self.device)
            self._check_precision_policy()
            self._audit_capacity()
            free, total = torch.cuda.mem_get_info(self.device)
            self.memory_at_allocation = {
                "pytorch_allocated": torch.cuda.memory_allocated(self.device),
                "pytorch_reserved": torch.cuda.memory_reserved(self.device),
                "device_used": total - free,
                "device_total": total,
            }
            self.allocated = True
        except BaseException:
            self.failed = True
            raise
        finally:
            self.setup_seconds = perf_counter() - started

    def _capture_pair(self, pair, attention, block):
        inputs = pair.inputs

        def project():
            normalized, saved = residual_rms_norm(
                inputs["hidden"],
                inputs.get("residual"),
                attention.input_norm_weight,
                attention.cfg.norm_eps,
            )
            logical_positions = inputs["positions"] + inputs["start"]
            positions = logical_positions.float()
            # The same dynamic start drives RoPE and the exclusive causal ends.
            # Computing these inside the projection graph avoids two eager
            # metadata launches before every indexer, including HBM-only.
            ends = (logical_positions + 1).to(torch.int32)
            return attention.project_positions(normalized, positions, normalized=True), saved, ends

        def finish():
            output = attention.wo(inputs["expanded"].reshape(pair.queries, -1))
            normalized, residual = residual_rms_norm(
                pair.saved, output, block.post_norm_weight, block.cfg.norm_eps
            )
            return block.mlp(normalized), residual

        current = torch.cuda.current_stream(self.device)
        self._capture_stream.wait_stream(current)
        with torch.cuda.stream(self._capture_stream):
            for _ in range(3):
                project()
        self._capture_stream.synchronize()
        pair.projection_graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(pair.projection_graph, stream=self._capture_stream),
            self._capture_scope(pair, "projection"),
        ):
            pair.projected, pair.saved, ends = project()
            pair.indexer_bounds = (inputs["index_starts"], ends)
        # Only the projection capture owns the saved tensor bound by finish.
        # Its no-residual branch aliases static hidden; otherwise it lives in
        # the projection's private pool. Both remain valid through finish.
        with torch.cuda.stream(self._capture_stream):
            pair.projection_graph.replay()
            for _ in range(3):
                finish()
        self._capture_stream.synchronize()
        pair.finish_graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(pair.finish_graph, stream=self._capture_stream),
            self._capture_scope(pair, "finish"),
        ):
            pair.output = finish()
        current.wait_stream(self._capture_stream)

    def _capture_scope(self, pair, part):
        """Optional diagnostic instrumentation while the capture graph is live."""
        return nullcontext()

    def _audit_capacity(self):
        snapshot = torch.cuda.memory_snapshot()
        segments = [item for item in snapshot if item["device"] == self.device.index]
        pool_ids = {
            tuple(graph.pool())
            for pair in self.pairs.values()
            for graph in (pair.projection_graph, pair.finish_graph)
            if graph is not None
        }
        pool_segments = [
            item for item in segments if tuple(item.get("segment_pool_id", (0, 0))) in pool_ids
        ]
        present = {tuple(item["segment_pool_id"]) for item in pool_segments}
        if present != pool_ids:
            raise RuntimeError("cannot identify every CUDA Graph private allocation pool")
        self.private_reserved_bytes = sum(item["total_size"] for item in pool_segments)
        blocks = {
            item["address"]: item["size"]
            for segment in segments
            for item in segment["blocks"]
            if item["state"] == "active_allocated"
        }
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for inputs in (
                *self._shared_inputs.values(),
                *(pair.owned_inputs for pair in self.pairs.values()),
            )
            for tensor in inputs.values()
        }
        if not storages.keys() <= blocks.keys():
            raise RuntimeError("cannot identify every static compute graph input allocation")
        self.static_storage_bytes = sum(storages.values())
        self.static_allocated_bytes = sum(blocks[address] for address in storages)
        if self.private_reserved_bytes > self.metadata["compute_graph_private_limit_bytes"]:
            raise CacheBudgetExceeded(
                "CUDA Graph private reserved capacity exceeds its chosen limit"
            )
        if (
            self.static_allocated_bytes
            > self.metadata["compute_graph_static_allocation_limit_bytes"]
        ):
            raise CacheBudgetExceeded("CUDA Graph input allocation exceeds its planned capacity")

    @contextmanager
    def execution(self):
        self._check()
        if not self.allocated or self._active:
            raise RuntimeError("compute graphs require one active backend execution")
        self._check_precision_policy()
        current = torch.cuda.current_stream(self.device)
        if self._last_stream is not None and self._last_stream != current:
            try:
                current.wait_event(self._last_event)
            except BaseException:
                self.failed = True
                raise
        self._active = True
        self._stream = current
        body_error = None
        try:
            yield
        except BaseException as error:
            body_error = error
            raise
        finally:
            try:
                self._last_event = torch.cuda.Event()
                self._last_event.record(current)
                self._last_stream = current
            except BaseException as completion_error:
                self.failed = True
                if body_error is not None:
                    raise BaseExceptionGroup(
                        "DeepSeek graph execution and completion both failed",
                        [body_error, completion_error],
                    ) from None
                raise
            finally:
                self._active = False
                self._stream = None

    def _check_precision_policy(self):
        if self.precision_policy != _precision_policy():
            raise RuntimeError("compute graph precision policy changed after capture")

    def supports(self, layer, hidden, residual):
        pair = self.pairs.get((layer, len(hidden)))
        return pair is not None and pair.residual_present == (residual is not None)

    @torch.inference_mode()
    def forward_block(self, layer, runner, hidden, residual, *, scope=None):
        self._check()
        if not self._active or torch.cuda.current_stream(self.device) != self._stream:
            raise RuntimeError("compute replay requires its backend execution stream")
        pair = self.pairs[layer, len(hidden)]
        if pair.residual_present != (residual is not None):
            raise ValueError("incoming residual does not match the captured block branch")
        if runner.attention is not self.attentions[layer]:
            raise ValueError("compute graph weight identity does not match this session layer")
        scope = scope or (lambda _: nullcontext())

        def project(source, position, *, normalized=False):
            if normalized or source is not hidden:
                raise ValueError("compute graph block projection requires its raw hidden input")
            attention = self.attentions[layer]
            attention._validate_project_hidden(hidden)
            if (
                type(position) is not int
                or position < 0
                or position + len(hidden) > attention.cfg.max_seq_len
            ):
                raise ValueError("Token positions exceed the checkpoint context limit")
            if residual is not None and (
                residual.shape != hidden.shape
                or residual.dtype != hidden.dtype
                or residual.device != hidden.device
            ):
                raise ValueError("Residual must match hidden shape, device, and BF16 dtype")
            with scope("compute_graph_projection_inputs"):
                if pair.inputs["hidden"] is not hidden:
                    pair.inputs["hidden"].copy_(hidden)
                    self.input_copies += 1
                if residual is not None and pair.inputs["residual"] is not residual:
                    pair.inputs["residual"].copy_(residual)
                    self.input_copies += 1
                if self._start_positions[pair.queries] != position:
                    pair.inputs["start"].fill_(position)
                    self._start_positions[pair.queries] = position
                    self.position_updates += 1
            with scope(f"compute_graph_projection_layer_{layer}_q_{len(hidden)}"):
                pair.projection_graph.replay()
            pair.projection_replays += 1
            projected = pair.projected
            if runner.cache.offload and getattr(runner.cache, "transient_start", None) is None:
                # D2H tickets must own bytes that a later graph cannot overwrite.
                with scope("compute_graph_owned_writeback_source"):
                    projected = replace(projected, kv=projected.kv.clone())
            return projected

        def finish(raw_attention):
            with scope("compute_graph_value_expansion"):
                self.attentions[layer]._expand_values(raw_attention, out=pair.inputs["expanded"])
            with scope(f"compute_graph_finish_layer_{layer}_q_{len(hidden)}"):
                pair.finish_graph.replay()
            pair.finish_replays += 1
            return pair.output

        return runner.forward(
            hidden,
            scope=scope,
            normalized=False,
            project_callback=project,
            output_callback=finish,
            indexer_bounds=pair.indexer_bounds,
        )

    def shared_bytes(self):
        return {"hbm": self.static_allocated_bytes + self.private_reserved_bytes, "dram": 0}

    def describe(self):
        return {
            "enabled": True,
            "policy_revision": GRAPH_POLICY_REVISION,
            "allocated": self.allocated,
            "static_storage_bytes": self.static_storage_bytes,
            "static_allocated_bytes": self.static_allocated_bytes,
            "private_reserved_bytes": self.private_reserved_bytes,
            "chosen_private_limit_bytes": self.metadata["compute_graph_private_limit_bytes"],
            "memory_at_allocation": self.memory_at_allocation,
            "setup_seconds": self.setup_seconds,
            "eager_fallbacks": self.eager_fallbacks,
            "input_copies": self.input_copies,
            "position_updates": self.position_updates,
            "captured_precision_policy": self.precision_policy,
            "profiling_status": "graph_node_timing_attribution_pending",
            "graphs": [
                {
                    "layer": pair.layer,
                    "queries": pair.queries,
                    "residual_present": pair.residual_present,
                    "projection_replays": pair.projection_replays,
                    "finish_replays": pair.finish_replays,
                }
                for pair in self.pairs.values()
            ],
        }

    def close(self):
        if self.closed:
            return
        if self._active:
            raise RuntimeError("cannot close active compute graphs")
        try:
            torch.cuda.synchronize(self.device)
        except BaseException:
            self.failed = True
            raise
        self.pairs.clear()
        self._shared_inputs.clear()
        self._start_positions.clear()
        self._capture_stream = self._last_stream = self._last_event = None
        self.static_storage_bytes = self.static_allocated_bytes = self.private_reserved_bytes = 0
        self.allocated = False
        self.closed = True
