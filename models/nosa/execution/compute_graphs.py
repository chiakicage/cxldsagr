"""Pure computation graphs and pre-commit validation; cache/selection/IO stay eager."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter

import torch

POLICY_REVISION = "nosa_compute_graphs_deferred_validation_v2"


def _precision():
    return (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
    )


@dataclass
class _Pair:
    layer: int
    queries: int
    inputs: dict
    project_graph: object = None
    finish_graph: object = None
    q: object = None
    records: dict = field(default_factory=dict)
    residual: object = None
    output: object = None
    project_replays: int = 0
    finish_replays: int = 0


class NosaComputeGraphs:
    """One projection and finish graph per independent layer and query shape.

    Static inputs and each graph's private allocator pool remain owned until
    close. The limit is a chosen capacity bound, not a measured fixed overhead.
    Inputs/weights/policies may not change placement, identity or precision after
    capture. Graph outputs are borrowed until the next execution of that pair;
    model.forward copies its final public hidden output before returning.
    """

    def __init__(self, model, query_sizes, *, private_limit_bytes=8 * 2**30):
        if not query_sizes or any(type(n) is not int or n <= 0 for n in query_sizes):
            raise ValueError("Compute graph query sizes must be positive integers")
        if type(private_limit_bytes) is not int or private_limit_bytes <= 0:
            raise ValueError("Compute graph private allocation limit must be positive")
        self.model = model
        self.layers = tuple(model.model.layers)
        weight = model.model.embed_tokens.weight
        self.device, self.dtype = weight.device, weight.dtype
        if self.device.type != "cuda" or self.dtype != torch.bfloat16:
            raise ValueError("NOSA compute graphs require CUDA BF16")
        if model.attention_mode != "sparse":
            raise ValueError("NOSA compute graphs require the full sparse model")
        self.query_sizes = tuple(sorted(set(query_sizes)))
        if max(self.query_sizes) > model.config.max_position_embeddings:
            raise ValueError("Compute graph query size exceeds model capacity")
        self.private_limit_bytes = private_limit_bytes
        self.pairs = {}
        self._closing_pools = set()
        self._active = self.closed = self.failed = self.allocated = False
        self._stream = self._last_stream = self._last_event = None
        self.private_reserved_bytes = self.static_allocated_bytes = self.static_storage_bytes = 0
        self.setup_seconds = None
        self.precision = _precision()
        self.weight_identity = self._weight_identity()
        self.config = deepcopy(model.config)
        self.layer_policy = deepcopy(self._layer_policy())
        self.cos_sin = None
        from models.nosa.execution.deferred_validation import DeferredValidation

        self.validation = DeferredValidation(len(self.layers), self.device)
        from cache.allocator.budget import allocation_bytes

        self.static_allocation_limit_bytes = (
            allocation_bytes(len(self.layers), self.device) + allocation_bytes(1, self.device)
        ) + sum(
            allocation_bytes(count * width * itemsize, self.device)
            for layer in range(len(self.layers))
            for count in self.query_sizes
            for width, itemsize in (
                (self.config.hidden_size, self.dtype.itemsize),
                (1, 8),
                (self.config.num_attention_heads * self.config.head_dim, self.dtype.itemsize),
                *(([(self.config.hidden_size, self.dtype.itemsize)]) if layer else []),
            )
        )

    def _layer_policy(self):
        return tuple(
            (
                layer.input_layernorm.eps,
                layer.post_attention_layernorm.eps,
                layer.self_attn.config,
                layer.self_attn.with_cis,
            )
            for layer in self.layers
        )

    def _weight_identity(self):
        try:
            return tuple(
                (name, id(p), p.data_ptr(), p._version, p.dtype, p.device, tuple(p.shape))
                for name, p in self.model.named_parameters()
            )
        except RuntimeError as error:
            raise ValueError(
                "Load graph model weights outside inference_mode to track parameter mutation"
            ) from error

    @property
    def pool_ids(self):
        return {
            tuple(graph.pool())
            for pair in self.pairs.values()
            for graph in (pair.project_graph, pair.finish_graph)
            if graph is not None
        }

    def _audit_memory(self):
        segments = [
            item for item in torch.cuda.memory_snapshot() if item["device"] == self.device.index
        ]
        pools = [s for s in segments if tuple(s.get("segment_pool_id", (0, 0))) in self.pool_ids]
        if {tuple(s["segment_pool_id"]) for s in pools} != self.pool_ids:
            raise RuntimeError("Cannot identify every NOSA compute graph allocation pool")
        self.private_reserved_bytes = sum(s["total_size"] for s in pools)
        blocks = {
            b["address"]: b["size"]
            for segment in segments
            for b in segment["blocks"]
            if b["state"] == "active_allocated"
        }
        storages = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for pair in self.pairs.values()
            for t in pair.inputs.values()
        }
        storages.update(
            (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            for t in self.validation.tensors
        )
        if not storages.keys() <= blocks.keys():
            raise RuntimeError("Cannot identify every NOSA graph static input allocation")
        self.static_storage_bytes = sum(storages.values())
        self.static_allocated_bytes = sum(blocks[address] for address in storages)
        if self.static_allocated_bytes > self.static_allocation_limit_bytes:
            raise RuntimeError("NOSA graph static allocation exceeds its planned capacity")
        if self.private_reserved_bytes > self.private_limit_bytes:
            raise RuntimeError("NOSA graph private reserved capacity exceeds its chosen limit")

    @torch.inference_mode()
    def allocate(self):
        if self.pairs or self.closed or self.failed or self.allocated:
            raise RuntimeError("NOSA compute graphs can be allocated only once")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Compute graph setup cannot run within another capture")
        started = perf_counter()
        free, _ = torch.cuda.mem_get_info(self.device)
        if self.static_allocation_limit_bytes + self.private_limit_bytes > free:
            raise RuntimeError("Chosen NOSA compute graph allocation limits exceed free HBM")
        current = torch.cuda.current_stream(self.device)
        capture = torch.cuda.Stream(device=self.device)
        _, self.cos_sin = self.model.model.rotary(
            self.config, 0, max(self.query_sizes), device=self.device
        )
        try:
            self.validation.allocate()
            for idx, layer in enumerate(self.layers):
                for queries in self.query_sizes:
                    inputs = {
                        "hidden": torch.zeros(
                            queries, self.config.hidden_size, device=self.device, dtype=self.dtype
                        ),
                        "positions": torch.arange(queries, device=self.device),
                        "attended": torch.zeros(
                            queries,
                            self.config.num_attention_heads,
                            self.config.head_dim,
                            device=self.device,
                            dtype=self.dtype,
                        ),
                    }
                    if idx:
                        inputs["residual"] = torch.zeros_like(inputs["hidden"])
                    pair = _Pair(idx, queries, inputs)
                    self.pairs[idx, queries] = pair
                    self._capture(pair, layer, current, capture)
                    self._audit_memory()
            current.wait_stream(capture)
            torch.cuda.synchronize(self.device)
            self._last_event = torch.cuda.Event()
            self._last_event.record(current)
            self._last_stream = current
            self.allocated = True
        except BaseException:
            self.failed = True
            raise
        finally:
            self.setup_seconds = perf_counter() - started

    def _capture(self, pair, layer, current, capture):
        inputs = pair.inputs

        def project():
            if pair.layer:
                normalized, residual = layer.input_layernorm(inputs["hidden"], inputs["residual"])
            else:
                normalized, residual = layer.input_layernorm(inputs["hidden"]), inputs["hidden"]
            q, records = layer.self_attn.project(normalized, inputs["positions"], self.cos_sin)
            return q, records, residual

        def finish():
            x = layer.self_attn.o_proj(inputs["attended"].reshape(pair.queries, -1))
            x, residual = layer.post_attention_layernorm(x, pair.residual)
            return layer.mlp(x), residual

        capture.wait_stream(current)
        with torch.cuda.stream(capture):
            for _ in range(3):
                project()
        capture.synchronize()
        pair.project_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(pair.project_graph, stream=capture):
            pair.q, pair.records, pair.residual = project()
        with torch.cuda.stream(capture):
            pair.project_graph.replay()
            for _ in range(3):
                finish()
        capture.synchronize()
        pair.finish_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(pair.finish_graph, stream=capture):
            pair.output = finish()
        current.wait_stream(capture)

    def _check(self):
        if not self.allocated or self.closed or self.failed:
            raise RuntimeError("NOSA compute graphs are not available")
        if _precision() != self.precision:
            raise RuntimeError("NOSA graph precision policy changed after capture")
        if (
            self._weight_identity() != self.weight_identity
            or tuple(self.model.model.layers) != self.layers
        ):
            raise RuntimeError("NOSA graph model parameters changed after capture")
        if (
            self.model.config != self.config
            or self.model.model.rotary.cos_sin_cache is not self.cos_sin
            or self._layer_policy() != self.layer_policy
        ):
            raise RuntimeError("NOSA graph model configuration or RoPE storage changed")

    @contextmanager
    def execution(self):
        self._check()
        if self._active or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NOSA compute graphs require exclusive eager execution")
        current = torch.cuda.current_stream(self.device)
        if current != self._last_stream:
            current.wait_event(self._last_event)
        self._active, self._stream = True, current
        body_error = record_error = None
        try:
            try:
                yield self
            except BaseException as error:  # noqa: BLE001 -- record completion before propagating.
                body_error = error
            try:
                self._last_event.record(current)
                self._last_stream = current
            except BaseException as error:  # noqa: BLE001 -- keep failed graph storage reachable.
                self.failed = True
                record_error = error
        finally:
            self._active, self._stream = False, None
        if body_error is not None and record_error is not None:
            raise BaseExceptionGroup(
                "NOSA graph execution and completion recording both failed",
                [body_error, record_error],
            ) from None
        if record_error is not None:
            raise record_error
        if body_error is not None:
            raise body_error

    @torch.inference_mode()
    def forward_layer(self, idx, hidden, residual, positions, attention, cache, indexer):
        if not self._active or torch.cuda.current_stream(self.device) != self._stream:
            raise RuntimeError("NOSA graph replay requires its active execution stream")
        pair = self.pairs.get((idx, len(hidden)))
        if pair is None or (residual is None) != (idx == 0):
            raise ValueError("No NOSA compute graph for this layer/query/residual geometry")
        for name, value in (("hidden", hidden), ("positions", positions), ("residual", residual)):
            if value is None:
                continue
            target = pair.inputs[name]
            if (value.shape, value.dtype, value.device) != (
                target.shape,
                target.dtype,
                target.device,
            ):
                raise ValueError(f"NOSA graph input {name} differs from captured geometry")
            target.copy_(value)
        pair.project_graph.replay()
        pair.project_replays += 1
        attended = self.layers[idx].self_attn.attend(
            pair.q, pair.records, attention, cache, idx, indexer=indexer
        )
        pair.inputs["attended"].copy_(attended)
        pair.finish_graph.replay()
        pair.finish_replays += 1
        return pair.output

    def shared_bytes(self):
        return {"hbm": self.static_allocated_bytes + self.private_reserved_bytes, "dram": 0}

    def describe(self):
        return {
            "enabled": True,
            "policy_revision": POLICY_REVISION,
            "query_sizes": self.query_sizes,
            "static_storage_bytes": self.static_storage_bytes,
            "static_allocated_bytes": self.static_allocated_bytes,
            "private_reserved_bytes": self.private_reserved_bytes,
            "chosen_private_limit_bytes": self.private_limit_bytes,
            "static_allocation_limit_bytes": self.static_allocation_limit_bytes,
            "setup_seconds": self.setup_seconds,
            "eager_fallbacks": 0,
            "project_replays": sum(p.project_replays for p in self.pairs.values()),
            "finish_replays": sum(p.finish_replays for p in self.pairs.values()),
            "graphs": len(self.pairs) * 2,
            "finite_validation": "per-layer device flags; one host decision before commit",
        }

    def close(self):
        if self.closed:
            return
        if self._active:
            raise RuntimeError("Cannot close active NOSA computation graphs")
        try:
            torch.cuda.synchronize(self.device)
        except BaseException:
            self.failed = True
            raise
        owned_pools = self.pool_ids | self._closing_pools
        self._closing_pools = owned_pools
        self.pairs.clear()
        self.validation.close()
        # Graph destruction releases ownership but native allocator segments
        # remain cached under their private IDs. Return those inactive segments
        # outside request timing before another backend validates pool identity.
        torch.cuda.empty_cache()
        if any(
            tuple(segment.get("segment_pool_id", (0, 0))) in owned_pools
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == self.device.index
        ):
            self.failed = True
            raise RuntimeError("NOSA graph allocations remain live after graph close")
        self.static_allocated_bytes = self.private_reserved_bytes = self.static_storage_bytes = 0
        self._closing_pools.clear()
        self.closed, self.allocated = True, False
