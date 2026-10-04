"""Persistent NOSA sessions for the serial multi-user GR serving runner.

The shared serving cache manager owns admission and eviction. This adapter
owns model-specific cache layout and execution; it never reads GR requests.
All schemes execute complete NOSA sparse attention, including CIS. ``dense``
describes the volume transferred, not a different attention mask.
"""

from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType

import torch

from models.nosa.cache import NosaKVCache
from models.nosa.model import NosaForCausalLM
from models.nosa.offload_cache import NosaOffloadCache
from models.nosa.serving_resources import POLICY_REVISION, NosaExecutionResources


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


class NosaDensePrefetchCache(NosaOffloadCache):
    """Pinned historical K/V with two alternating, full-layer HBM buffers.

    Layer zero is queued at the start of each append. Once the current layer has
    written its suffix, its stream waits for that layer's history and queues the
    next layer's complete history on a copy stream. The copy stream waits for all
    previous consumers before reusing a slot. Thus next-layer copies can overlap
    the current layer's indexer, attention, and MLP. No sparse fetch is performed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._execution_resources is not None:
            if self._execution_resources.scheme != "dense_prefetch":
                raise ValueError("Dense prefetch cache requires matching execution resources")
            self._stage_keys = self._stage_values = None
            self._prefetch_stream = None
            self._ready = [None, None]
            self._staged_layers = [None, None]
            self.last_prefetch_bytes = 0
            return
        shape = (2, self.max_seq_len, self.config.num_key_value_heads, self.config.head_dim)
        self._stage_keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        self._stage_values = torch.empty_like(self._stage_keys)
        self._prefetch_stream = None
        self._ready = [None, None]
        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
            self._ready = [torch.cuda.Event(), torch.cuda.Event()]
        self._staged_layers = [None, None]
        self.last_prefetch_bytes = 0

    @property
    def attention_workspace(self):
        raise RuntimeError("Dense prefetch uses double staging, not sparse fetch workspace")

    def begin_step(self, token_count):
        result = super().begin_step(token_count)
        self._staged_layers = [None, None]
        self.last_prefetch_bytes = 0
        try:
            if self._execution_resources is not None:
                self._execution_resources.staging_lease.reset()
            self._prefetch(0)
        except BaseException:
            self.abort_step()
            raise
        return result

    @torch.no_grad()
    def _prefetch(self, layer_idx):
        if layer_idx >= self.spec.num_layers:
            return
        slot = layer_idx % 2
        host = self.host_layer_view(layer_idx)
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self)
            lease = self._execution_resources.staging_lease
            previous = lease.submitted_copy_bytes
            try:
                lease.prefetch(
                    layer_idx,
                    sources={name: host[name][: self.length] for name in ("keys", "values")},
                )
            finally:
                self._transfer_metrics.add_payload(
                    "main_kv_host_to_device_bytes", lease.submitted_copy_bytes - previous
                )
            self.last_prefetch_bytes += (
                2
                * self.length
                * self.config.num_key_value_heads
                * self.config.head_dim
                * self.dtype.itemsize
            )
            self._staged_layers[slot] = layer_idx
            return

        def copy_history():
            self._stage_keys[slot, : self.length].copy_(
                host["keys"][: self.length], non_blocking=self.device.type == "cuda"
            )
            self._stage_values[slot, : self.length].copy_(
                host["values"][: self.length], non_blocking=self.device.type == "cuda"
            )

        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            # The current stream includes every earlier attention consumer.
            # This fences slot reuse, while later current-layer compute can
            # overlap the next-layer history copy queued below.
            self._prefetch_stream.wait_stream(current)
            with torch.cuda.stream(self._prefetch_stream):
                copy_history()
                self._ready[slot].record(self._prefetch_stream)
        else:
            copy_history()
        self.last_prefetch_bytes += (
            2
            * self.length
            * self.config.num_key_value_heads
            * self.config.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )
        self._staged_layers[slot] = layer_idx

    @torch.no_grad()
    def write_layer(self, layer_idx, **records):
        if self._staged_layers[layer_idx % 2] != layer_idx:
            raise RuntimeError("Dense prefetch requires layers in increasing order")
        super().write_layer(layer_idx, **records)
        slot = layer_idx % 2
        if self._execution_resources is not None:
            stage = self._execution_resources.staging_lease.wait_ready(layer_idx)
            for name in ("keys", "values"):
                stage[name][self.length : self._pending_end].copy_(records[name])
            self._prefetch(layer_idx + 1)
            return
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_event(self._ready[slot])
        self._stage_keys[slot, self.length : self._pending_end].copy_(records["keys"])
        self._stage_values[slot, self.length : self._pending_end].copy_(records["values"])
        self._prefetch(layer_idx + 1)

    def dense_layer_view(self, layer_idx):
        end = self.visible_length(layer_idx)
        slot = layer_idx % 2
        if self._staged_layers[slot] != layer_idx:
            raise RuntimeError("Requested layer no longer owns its dense staging slot")
        if self._execution_resources is not None:
            self._execution_resources.check_execution(self)
            return {
                **self._execution_resources.staging_lease.view(layer_idx, end=end),
                "cis_scores": self.cis_scores[layer_idx, :end],
            }
        return {
            "keys": self._stage_keys[slot, :end],
            "values": self._stage_values[slot, :end],
            "cis_scores": self.cis_scores[layer_idx, :end],
        }

    def stats(self):
        result = super().stats()
        staging = sum(_bytes(t) for t in (self._stage_keys, self._stage_values) if t is not None)
        result["resident_bytes"] += staging
        result["capacity_bytes"] += staging
        return result

    def release(self):
        if self.released:
            return
        super().release()
        self._stage_keys = self._stage_values = None
        self._prefetch_stream = None
        self._ready = [None, None]
        self._staged_layers = [None, None]


class _DensePrefetchAttention:
    def __call__(self, q, selection, cache_access, context):
        if not isinstance(cache_access, NosaDensePrefetchCache):
            raise TypeError("Dense prefetch attention requires its matching cache")
        records = cache_access.dense_layer_view(context.layer_idx)
        options = {}
        if q.is_cuda:
            from operators.nosa.attention.device_only.api import nosa_block_sparse_attention

            attention = nosa_block_sparse_attention
            resources = cache_access._execution_resources
            if resources is not None:
                resources.check_execution(cache_access, len(q))
                options["workspace"] = resources.attention_workspace
        else:
            from operators.nosa.attention.reference.torch import (
                reference_nosa_block_sparse_attention,
            )

            attention = reference_nosa_block_sparse_attention
        return attention(
            q,
            records["keys"],
            records["values"],
            selection,
            query_start=context.query_start,
            cis_bias=records["cis_scores"],
            **options,
        )


class NosaServingBackend:
    """Full NOSA model with persistent, independently evictable user sessions.

    ``estimate_session_bytes`` reserves cache allocation blocks, including owned
    pending appends, indexer validation/selection and CIS helper temporaries.
    Model weights, ordinary QKV activations and attention outputs are separate.
    There is no per-request weight loading, CPU snapshotting, or prefix replay on
    a hit. Caller-owned LRU eviction discards both tiers of one user's session.
    """

    schemes = ("hbm", "serial_sparse", "dense_prefetch", "overlap")

    def __init__(self, model, scheme, *, chunk_size=1024):
        if scheme not in self.schemes:
            raise ValueError(f"NOSA serving scheme must be one of {self.schemes}")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if model.attention_mode != "sparse":
            raise ValueError("All NOSA serving schemes require the complete sparse NOSA policy")
        self.model = model
        self.scheme = scheme
        self.chunk_size = chunk_size
        self.config = model.config
        self.max_seq_len = self.config.max_position_embeddings
        self.device = model.model.embed_tokens.weight.device
        self.dtype = model.model.embed_tokens.weight.dtype
        self.compute_graphs = None
        self._dense_attention = _DensePrefetchAttention()
        cfg = self.config
        self.resources = NosaExecutionResources(
            scheme=scheme,
            layers=cfg.num_hidden_layers,
            kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            query_heads=cfg.num_attention_heads,
            max_seq_len=self.max_seq_len,
            chunk_size=chunk_size,
            dtype=self.dtype,
            device=self.device,
        )

    @classmethod
    def from_pretrained(
        cls, checkpoint, *, scheme, device="cuda:0", chunk_size=1024, max_seq_len=None
    ):
        model = NosaForCausalLM.from_pretrained(
            checkpoint,
            device=device,
            dtype=torch.bfloat16,
            attention_mode="sparse",
            sparse_backend="auto",
        )
        if (model.config.num_hidden_layers, len(model.model.layers)) != (32, 32):
            raise ValueError("NOSA-8B serving requires all 32 checkpoint layers")
        if max_seq_len is not None:
            if type(max_seq_len) is not int or not 0 < max_seq_len <= 262144:
                raise ValueError("max_seq_len must be an integer in [1,262144]")
            model.config = replace(model.config, max_position_embeddings=max_seq_len)
        backend = cls(model, scheme, chunk_size=chunk_size)
        backend.checkpoint = str(checkpoint)
        return backend

    def _validate_capacity(self, capacity):
        self.resources.validate_capacity(capacity)

    def plan_resources(self, budgets, limits):
        if (
            self.device.type == "cuda"
            and getattr(self.model.main_attention, "backend", None) == "reference"
        ):
            raise NotImplementedError("Shared CUDA serving requires native attention")
        plan = self.resources.plan_resources(budgets, limits)
        if self.compute_graphs is None:
            return plan
        if budgets is not None:
            raise NotImplementedError("NOSA compute graphs currently require fixed P/NH mode")
        from cache.prefix_pool import CacheFootprint

        self._graph_base_plan = plan
        shared = self.compute_graphs.shared_bytes()["hbm"]
        description = {
            key: value
            for key, value in self.compute_graphs.describe().items()
            if key not in ("project_replays", "finish_replays", "eager_fallbacks")
        }
        self._graph_plan = replace(
            plan,
            shared=CacheFootprint(plan.shared.hbm + shared, plan.shared.dram),
            metadata=MappingProxyType({**plan.metadata, "compute_graphs": description}),
        )
        return self._graph_plan

    def allocate_shared(self, plan):
        if self.compute_graphs is not None:
            base = getattr(self, "_graph_base_plan", None)
            if base is None or plan != getattr(self, "_graph_plan", None):
                raise ValueError("Compute graph allocation requires its matching resource plan")
            self.resources.allocate_shared(base)
        else:
            self.resources.allocate_shared(plan)

    def shared_bytes(self):
        result = self.resources.shared_bytes()
        if self.compute_graphs is not None:
            result["hbm"] += self.compute_graphs.shared_bytes()["hbm"]
        return result

    def enable_compute_graphs(self, query_sizes, *, private_limit_bytes=8 * 2**30):
        """Capture pure compute before runner admission or session allocation."""
        if (
            self.compute_graphs is not None
            or self.resources.plan is not None
            or self.resources._sessions
        ):
            raise RuntimeError("Enable compute graphs before planning or allocating sessions")
        from models.nosa.compute_graphs import NosaComputeGraphs

        graphs = NosaComputeGraphs(self.model, query_sizes, private_limit_bytes=private_limit_bytes)
        self.compute_graphs = graphs
        try:
            graphs.allocate()
            self.resources.allowed_graph_pool_ids = graphs.pool_ids
        except BaseException as error:
            import traceback

            # Setup tracebacks can retain the last graph pair and private output
            # tensors after capture has unwound. Release those local aliases
            # before checking that close returned its allocator segments.
            traceback.clear_frames(error.__traceback__)
            graphs.close()
            self.compute_graphs = None
            raise

    def bind_owner(self, owner):
        self.resources.bind_owner(owner)

    def unbind_owner(self, owner, *, rollback=False):
        self.resources.unbind_owner(owner, rollback=rollback)

    def estimate_session_host_pages(self, capacity):
        self._validate_capacity(capacity)
        return 0

    def session_host_pages(self, session):
        self.resources.check_session(session)
        return 0

    def estimate_session_bytes(self, capacity, prefix_tokens):
        self._validate_capacity(capacity)
        if type(prefix_tokens) is not int or not 0 < prefix_tokens < capacity:
            raise ValueError("Prefix must leave a nonempty candidate suffix")
        cfg = self.config
        queries = max(min(self.chunk_size, prefix_tokens), capacity - prefix_tokens)
        if self.resources.plan is not None:
            metadata = self.resources.plan.metadata
            if capacity - prefix_tokens > metadata["max_candidate_tokens"]:
                raise ValueError("Candidate suffix exceeds planned execution capacity")
            if queries > metadata["max_query_tokens"]:
                raise ValueError("Query length exceeds planned execution capacity")
        from models.nosa.session_budget import estimate_session_bytes

        return estimate_session_bytes(
            capacity=capacity,
            queries=queries,
            layers=cfg.num_hidden_layers,
            kv_heads=cfg.num_key_value_heads,
            query_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
            dtype=self.dtype,
            device=self.device,
            scheme=self.scheme,
        )

    def create_session(self, capacity):
        self._validate_capacity(capacity)
        if self.resources.plan is None:
            raise RuntimeError("Plan and allocate shared resources before creating a session")
        options = {
            "device": self.device,
            "dtype": self.dtype,
            "with_cis": True,
            "execution_resources": self.resources,
        }
        if self.scheme == "hbm":
            session = NosaKVCache(self.config, capacity, **options)
        else:
            cache_type = (
                NosaDensePrefetchCache if self.scheme == "dense_prefetch" else NosaOffloadCache
            )
            session = cache_type(
                self.config,
                capacity,
                overlap=self.scheme == "overlap",
                query_tile_size=128,
                **options,
            )
        self.resources.attach(session)
        return session

    @contextmanager
    def execution_lease(self, session, *, phase="candidate"):
        from models.nosa.attention import NosaSparseAttention

        metrics, started = None, False
        try:
            with self.resources.lease(session):
                metrics = session._transfer_metrics
                started = True
                metrics.begin(phase)
                original = self.model.main_attention
                self.model.main_attention = (
                    self._dense_attention
                    if self.scheme == "dense_prefetch"
                    else NosaSparseAttention(workspace=self.resources.attention_workspace)
                )
                try:
                    yield self.resources
                finally:
                    self.model.main_attention = original
        except BaseException:
            if started and metrics._phase is not None:
                metrics.finish(success=False)
            raise
        else:
            metrics.finish(success=True)

    def _validate_ids(self, session, ids, *, prefill):
        self.resources.check_session(session)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or not len(ids):
            raise ValueError("Tokens must be a nonempty one-dimensional tensor")
        if ids.dtype not in (torch.int32, torch.int64) or ids.device != self.device:
            raise ValueError("Tokens must be integral and on the backend device")
        if session.length + len(ids) > session.max_seq_len:
            raise ValueError("Input exceeds session capacity")
        if not prefill and len(ids) > self.resources.plan.metadata["max_candidate_tokens"]:
            raise ValueError("Candidate suffix exceeds planned execution capacity")
        queries = min(self.chunk_size, len(ids)) if prefill else len(ids)
        if queries > self.resources.plan.metadata["max_query_tokens"]:
            raise ValueError("Query length exceeds planned execution capacity")

    def _forward(self, session, ids):
        if self.compute_graphs is None:
            return self.model(ids, session, return_hidden=True)
        with self.compute_graphs.execution():
            return self.model(ids, session, return_hidden=True, compute_graphs=self.compute_graphs)

    @torch.inference_mode()
    def prefill(self, session, ids):
        self._validate_ids(session, ids, prefill=True)
        if session.length != 0:
            raise ValueError("Prefix construction requires an empty session")
        if ids.ndim != 1 or not len(ids):
            raise ValueError("Prefix must be a nonempty one-dimensional token tensor")
        with self.execution_lease(session, phase="prefix"):
            for start in range(0, len(ids), self.chunk_size):
                self._forward(session, ids[start : start + self.chunk_size])

    @torch.inference_mode()
    def extend(self, session, ids):
        self._validate_ids(session, ids, prefill=False)
        with self.execution_lease(session):
            return self._forward(session, ids)

    def truncate(self, session, prefix):
        self.resources.check_session(session)
        # Resident truncate publishes metadata only, so synchronize before the
        # next user or a possible eviction may reuse/free its tensors.
        self.synchronize()
        session.truncate(prefix)

    def session_storage_bytes(self, session):
        """Logical owning storage bytes, separate from allocator capacity."""
        self.resources.check_session(session, allow_released=True)
        stats = session.stats()
        return {"hbm": stats["resident_bytes"], "dram": stats["host_bytes"]}

    def session_bytes(self, session):
        """HBM tensor storage and owned DRAM capacity for admission checks.

        CUDA allocator padding is covered by the reservation and intrusive
        allocation audit. Pinned host bins have an exact power-of-two capacity,
        which is reported here even when a tensor exposes a shorter storage.
        """
        from models.nosa.allocation_budget import pinned_allocation_bytes

        result = self.session_storage_bytes(session)
        tensors = [*session.buffers.values(), getattr(session, "_native_indexer_host_flag", None)]
        seen = set()
        for tensor in tensors:
            if tensor is None or tensor.device.type != "cpu" or not tensor.is_pinned():
                continue
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in seen:
                seen.add(storage.data_ptr())
                result["dram"] += (
                    pinned_allocation_bytes(storage.nbytes(), "cuda") - storage.nbytes()
                )
        return result

    def session_metrics(self, session):
        """Observe last-request KV payload after execution, outside its timing.

        Prefix construction is retained for its immediately following candidate;
        a later candidate starts a revisit with zero prefix bytes. ``truncate``
        retains these counters. CUDA scalar observation is a diagnostic D2H and
        is excluded from the KV payload fields, as are metadata and link overhead.
        """
        self.resources.check_session(session)
        return session._transfer_metrics.snapshot()

    def release_session(self, session):
        self.resources.check_session(session, allow_released=True)
        self.synchronize()
        session.release()

    def close(self):
        self.resources.close()
        if self.compute_graphs is not None:
            self.compute_graphs.close()
            self.compute_graphs = None
            self.resources.allowed_graph_pool_ids = set()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def describe(self):
        return {
            "model": "nosa",
            "checkpoint": getattr(self, "checkpoint", None),
            "scheme": self.scheme,
            "layers": len(self.model.model.layers),
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "dtype": str(self.dtype),
            "device": str(self.device),
            "max_seq_len": self.max_seq_len,
            "prefix_chunk_size": self.chunk_size,
            "policy_revision": POLICY_REVISION,
            "workspace_scope": "backend",
            "host_scope": "session",
            "resource_plan": None
            if self.resources.plan is None
            else dict(self.resources.plan.metadata),
            "shared_cache_bytes": self.shared_bytes(),
            "attention_policy": "full NOSA: 33 QA incl sink/local, then CIS to 64 blocks",
            "output": "all candidate normalized hidden states; no LM head or decode",
            "cache_eviction": "whole-user session; both HBM and DRAM released",
            "cache_budget_scope": "tensor capacities, derived records, staging, owned append, scratch",
            "dense_prefetch": "full next-layer historical K/V on copy stream, two fenced HBM slots",
            "sparse_offload": "native full-batch unique sparse union; overlap uses cooperative FA3",
            "compute_graphs": {"enabled": False}
            if self.compute_graphs is None
            else self.compute_graphs.describe(),
        }
