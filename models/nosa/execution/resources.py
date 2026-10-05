"""Backend-owned, bounded execution storage for serial NOSA serving.

CPU allocations exercise layout and ownership only. Their ``hbm`` ledger is a
resource role, not a measurement of GPU memory. Model weights and ordinary
activations are outside this module; persistent histories remain session-owned.
"""

from contextlib import contextmanager
from dataclasses import replace
from math import prod
from types import MappingProxyType

import torch

from cache.allocator.budget import allocation_bytes, validate_allocator
from cache.capacity import AllocationSpec, CacheFootprint, CapacityPolicy, ResourcePlan
from cache.lifecycle import ResourceLifecycle
from cache.prefix_pool import CacheBudgetExceeded

POLICY_REVISION = "nosa_backend_workspace_v2"


def _positive(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _lifecycle_property(name):
    """Keep diagnostic/test access attached to the single lifecycle state."""
    return property(
        lambda self: getattr(self.lifecycle, name),
        lambda self, value: setattr(self.lifecycle, name, value),
    )


class NosaExecutionResources:
    """One fixed workspace, with independent session identities and one lease."""

    plan = _lifecycle_property("plan")
    generation = _lifecycle_property("generation")
    poisoned = _lifecycle_property("poisoned")
    _sessions = _lifecycle_property("sessions")
    _identity = _lifecycle_property("identity")
    _active_session = _lifecycle_property("active_session")
    _admission_owner = _lifecycle_property("admission_owner")
    _admission_initial_plan = _lifecycle_property("admission_initial_plan")

    def __init__(
        self,
        *,
        scheme,
        layers,
        kv_heads,
        head_dim,
        query_heads,
        max_seq_len,
        chunk_size,
        dtype,
        device,
        query_tile_size=128,
        fetch_ctas=96,
    ):
        if scheme not in ("hbm", "serial_sparse", "dense_prefetch", "overlap"):
            raise ValueError("Unknown NOSA serving scheme")
        for name, value in (
            ("layers", layers),
            ("kv_heads", kv_heads),
            ("head_dim", head_dim),
            ("query_heads", query_heads),
            ("max_seq_len", max_seq_len),
            ("chunk_size", chunk_size),
            ("query_tile_size", query_tile_size),
            ("fetch_ctas", fetch_ctas),
        ):
            _positive(name, value)
        self.scheme, self.layers = scheme, layers
        self.kv_heads, self.head_dim, self.query_heads = kv_heads, head_dim, query_heads
        self.max_seq_len, self.chunk_size = max_seq_len, chunk_size
        self.device, self.dtype = torch.device(device), dtype
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.query_tile_size, self.fetch_ctas = query_tile_size, fetch_ctas
        self.lifecycle = ResourceLifecycle("NOSA execution")
        self.fetch_workspace = self.attention_workspace = self.staging = None
        self._allocation_ready = None
        self._allocator_key = None
        self.allowed_graph_pool_ids = set()
        self.staging_lease = None
        self._allocation_failure = None

    def validate_native(self):
        if self.device.type == "cpu":
            return
        from operators.nosa._native import native_enabled

        if (
            self.device.type != "cuda"
            or not native_enabled()
            or torch.cuda.get_device_capability(self.device) != (9, 0)
            or self.dtype != torch.bfloat16
            or self.head_dim != 128
            or self.query_heads != self.kv_heads * 16
        ):
            raise NotImplementedError("Shared NOSA serving requires native SM90/BF16/D128/GQA16")

    def _validate_allocator(self):
        evidence = validate_allocator(self.device, allowed_graph_pools=self.allowed_graph_pool_ids)
        if self._allocator_key is not None and evidence["configuration_key"] != self._allocator_key:
            raise RuntimeError("CUDA allocator configuration changed while NOSA resources are live")
        return evidence

    def _plan_geometry(self, limits):
        """Validate execution geometry without selecting a capacity policy."""
        self.validate_native()
        allocator = self._validate_allocator()
        capacity = limits.get("max_session_capacity", self.max_seq_len)
        candidate = limits.get("max_candidate_tokens", capacity)
        _positive("max_session_capacity", capacity)
        _positive("max_candidate_tokens", candidate)
        if not candidate <= capacity <= min(self.max_seq_len, 262144):
            raise ValueError("C/A limits must satisfy 0 < A <= C <= model context limit")
        queries = max(min(self.chunk_size, capacity), candidate)
        trace_rows = limits.get("trace_capacity", 0)
        if type(trace_rows) is not int or trace_rows < 0:
            raise ValueError("trace_capacity must be a nonnegative row count")
        stages = {"hbm": 0, "dense_prefetch": 2}.get(self.scheme, 1)
        if trace_rows and stages != 1:
            raise ValueError("Trace storage is supported only by sparse fetch schemes")
        return {
            "policy_revision": POLICY_REVISION,
            "allocator_policy": allocator["policy"],
            "pool_scope": "backend_workspace",
            "workspace_scope": "backend",
            "host_scope": "session",
            "scheme": self.scheme,
            "max_session_capacity": capacity,
            "max_candidate_tokens": candidate,
            "max_query_tokens": queries,
            "prefix_chunk_size": self.chunk_size,
            "stage_count": stages,
            "trace_capacity": trace_rows,
            "layers": self.layers,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "query_heads": self.query_heads,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "query_tile_size": self.query_tile_size,
            "fetch_ctas": self.fetch_ctas,
        }

    def _allocation(self, name, dtype, shape, *, alias_of=None):
        payload = prod(shape) * dtype.itemsize
        return AllocationSpec(
            name=name,
            owner="shared",
            dtype=str(dtype),
            shape=shape,
            device=str(self.device),
            lifetime="resource_plan",
            storage_bytes=payload,
            charged_bytes=0 if alias_of is not None else allocation_bytes(payload, self.device),
            alias_of=alias_of,
            accounting_tier="hbm",
        )

    def _workspace_allocations(self, metadata):
        """Declare kernel-owned scratch and staging using their stable names."""
        capacity, queries = metadata["max_session_capacity"], metadata["max_query_tokens"]
        if self.scheme in ("serial_sparse", "overlap"):
            from operators.nosa.attention.offload.api import NosaFetchWorkspace

            layout = NosaFetchWorkspace.allocation_layout(
                capacity,
                self.kv_heads,
                self.head_dim,
                dtype=self.dtype,
                query_tile_size=self.query_tile_size,
                max_queries=queries,
                trace_capacity=metadata["trace_capacity"],
            )
            prefix = "fetch"
        else:
            from operators.nosa.attention.workspace import NosaAttentionWorkspace

            layout = NosaAttentionWorkspace.allocation_layout(
                queries, self.kv_heads, self.head_dim, dtype=self.dtype
            )
            prefix = "attention"
        return tuple(
            self._allocation(f"{prefix}.{name}", dtype, shape)
            for name, (dtype, shape) in layout.items()
        )

    def plan_resources(self, budgets, limits):
        """Estimate from dimensions alone; do not allocate or mutate a live plan."""
        if not isinstance(budgets, CacheFootprint):
            raise TypeError("budgets must be a CacheFootprint")
        policy = CapacityPolicy.byte_budget(budgets)
        metadata = self._plan_geometry(limits)
        allocations = self._workspace_allocations(metadata)
        if self.scheme == "dense_prefetch":
            shape = (2, metadata["max_session_capacity"], self.kv_heads, self.head_dim)
            allocations += tuple(
                self._allocation(f"staging.{name}", self.dtype, shape)
                for name in ("keys", "values")
            )
        logical_shared = sum(item.storage_bytes for item in allocations)
        shared = sum(item.charged_bytes for item in allocations)
        metadata["logical_shared_hbm_bytes"] = logical_shared
        result = ResourcePlan(
            shared=CacheFootprint(hbm=shared),
            host_pages=0,
            metadata=MappingProxyType(metadata),
            allocations=allocations,
            policy=policy,
        )
        if not result.shared.fits(budgets):
            raise CacheBudgetExceeded(f"Shared NOSA workspace {result.shared} exceeds {budgets}")
        return result

    @property
    def record_shapes(self):
        return {name: (self.kv_heads, self.head_dim) for name in ("keys", "values")}

    def allocate_shared(self, plan, *, owner=None):
        self.lifecycle.check_access(owner, allow_closed=True)
        if not isinstance(plan, ResourcePlan):
            raise TypeError("Expected ResourcePlan")
        budget = plan.shared if plan.policy is None else plan.policy.budget
        expected = self.plan_resources(budget, plan.metadata)
        if expected != plan:
            raise ValueError("Shared plan does not match this backend's layout")
        # Admission budgets can change while the immutable storage layout is
        # reused. They do not create a different workspace allocation identity.
        if self.plan is not None and replace(self.plan, policy=plan.policy) == plan:
            return
        if self._sessions or self._active_session is not None:
            raise RuntimeError("Cannot replace a plan with live sessions or lease")
        if self.plan is not None:
            raise RuntimeError("Close the existing shared plan before allocating a different plan")
        self._drop_storage()
        metadata = plan.metadata
        capacity, queries = metadata["max_session_capacity"], metadata["max_query_tokens"]
        try:
            if self.scheme in ("serial_sparse", "overlap"):
                from operators.nosa.attention.offload.api import NosaFetchWorkspace

                self.fetch_workspace = NosaFetchWorkspace(
                    capacity,
                    self.kv_heads,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                    query_tile_size=self.query_tile_size,
                    fetch_ctas=self.fetch_ctas,
                    overlap=self.scheme == "overlap",
                    max_queries=queries,
                    bounded=True,
                    trace_capacity=metadata["trace_capacity"],
                )
            else:
                from operators.nosa.attention.workspace import NosaAttentionWorkspace

                self.attention_workspace = NosaAttentionWorkspace(
                    queries,
                    self.kv_heads,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                if self.scheme == "dense_prefetch":
                    from cache.staging import DoubleBufferStaging

                    self.staging = DoubleBufferStaging(
                        capacity,
                        self.record_shapes,
                        device=self.device,
                        dtype=self.dtype,
                    )
            actual = CacheFootprint.from_mapping(self.shared_bytes())
            logical = CacheFootprint(hbm=metadata["logical_shared_hbm_bytes"])
            if actual != logical or not actual.fits(plan.shared):
                raise RuntimeError(
                    f"Shared storage {actual} differs from layout {logical} / reserve {plan.shared}"
                )
            allocator = self._validate_allocator()
            self._allocator_key = allocator["configuration_key"]
            if self.device.type == "cuda":
                self._allocation_ready = torch.cuda.Event()
                self._allocation_ready.record(torch.cuda.current_stream(self.device))
        except BaseException as error:
            # A failed constructor can own partially initialized CUDA tensors
            # only through its traceback. Retain them until the device drains;
            # on a failed drain the poisoned owner must keep those aliases.
            self._allocation_failure = error
            self.lifecycle.cleanup_after_failure(error, self._drain_and_drop_storage)
            raise
        self.lifecycle.allocated(plan, owner=owner)

    def storage_tensors(self):
        tensors = []
        for workspace in (self.fetch_workspace, self.attention_workspace):
            if workspace is not None:
                tensors.extend(workspace.tensors())
        if self.staging is not None:
            tensors.extend(self.staging.storage_tensors())
        return tuple(tensors)

    def shared_bytes(self):
        storages = {}
        for tensor in self.storage_tensors():
            storage = tensor.untyped_storage()
            storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return {"hbm": sum(storages.values()), "dram": 0}

    def validate_capacity(self, capacity):
        _positive("Session capacity", capacity)
        limit = (
            self.max_seq_len if self.plan is None else self.plan.metadata["max_session_capacity"]
        )
        if capacity > limit:
            raise ValueError("Session capacity exceeds planned context limit")

    def validate_cache(self, *, capacity, device, dtype, kv_heads, head_dim, **fetch_options):
        if self.plan is None:
            raise RuntimeError("Plan and allocate shared resources before creating a session")
        if self.poisoned:
            raise RuntimeError("NOSA execution resources are poisoned")
        self.validate_capacity(capacity)
        self._validate_allocator()
        if (torch.device(device), dtype, kv_heads, head_dim) != (
            self.device,
            self.dtype,
            self.kv_heads,
            self.head_dim,
        ):
            raise ValueError("Borrowed cache layout differs from execution resources")
        for name, expected in (
            ("query_tile_size", self.query_tile_size),
            ("fetch_ctas", self.fetch_ctas),
            ("overlap", self.scheme == "overlap"),
        ):
            if name in fetch_options and fetch_options[name] != expected:
                raise ValueError(f"Borrowed cache {name} differs from execution resources")

    def attach(self, session, *, owner=None):
        self.lifecycle.check_access(owner)
        if session in self._sessions or session.released:
            raise RuntimeError("Expected a new live session")
        if getattr(session, "_execution_resources", None) is not self or hasattr(
            session, "_serving_identity"
        ):
            raise ValueError("Cannot attach a foreign or previously owned NOSA session")
        session._serving_identity = self._identity
        session._execution_generation = self.generation
        session._execution_ready = None
        # Register before CUDA can fail so cleanup can release the new session,
        # or retain it under this owner when completion cannot be established.
        session._lifecycle_registration = self.lifecycle.register(session, owner=owner)
        try:
            if self.device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
                session._execution_ready = ready
        except BaseException as attachment_error:

            def cleanup():
                torch.cuda.synchronize(self.device)
                with self.mutation(session, owner=owner, releasing=True):
                    session.release()

            self.lifecycle.cleanup_after_failure(attachment_error, cleanup)
            raise

    def check_session(self, session, *, allow_released=False):
        self.lifecycle.check_session(
            session,
            getattr(session, "_lifecycle_registration", None),
            released=getattr(session, "released", False),
            allow_released=allow_released,
        )

    def check_execution(self, session, token_count=None):
        self.check_session(session)
        self.lifecycle.check_execution(session)
        if token_count is not None:
            _positive("query length", token_count)
            if token_count > self.plan.metadata["max_query_tokens"]:
                raise ValueError("Query length exceeds planned execution capacity")

    def check_mutation(self, session, *, releasing=False):
        self.check_session(session)
        self.lifecycle.check_mutation(session, releasing=releasing)

    def detach(self, session):
        self.lifecycle.detach(session)
        session._execution_ready = None

    @contextmanager
    def lease(self, session, *, owner=None):
        self.check_session(session)
        self.lifecycle.check_access(owner)
        self.validate_native()
        self._validate_allocator()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise NotImplementedError(
                        "Shared NOSA serving does not support CUDA Graph capture"
                    )

        def prepare():
            if self.device.type == "cuda":
                current = torch.cuda.current_stream(self.device)
                for event in (self._allocation_ready, session._execution_ready):
                    if event is not None:
                        current.wait_event(event)
            if self.staging is not None:
                self.staging_lease = self.staging.lease(session, generation=self.generation)

        def drain():
            failures = []
            if self.staging_lease is not None:
                try:
                    self.staging_lease.close()
                except BaseException as error:  # noqa: BLE001 -- still attempt the device drain.
                    failures.append(error)
            # Includes commit-time compression and all queued D2H/H2D work,
            # even a prefetch that an exceptional caller never consumed.
            if self.device.type == "cuda":
                try:
                    torch.cuda.synchronize(self.device)
                except BaseException as error:  # noqa: BLE001 -- poison before propagating.
                    failures.append(error)
            try:
                self._validate_allocator()
            except BaseException as error:  # noqa: BLE001 -- reject invalidated bounds.
                failures.append(error)
            if failures:
                cause = (
                    failures[0]
                    if len(failures) == 1
                    else BaseExceptionGroup("NOSA completion failures", failures)
                )
                raise RuntimeError("Unable to drain NOSA execution resources") from cause
            if not self.poisoned:
                self.staging_lease = None

        with self.lifecycle.execution(
            session, session._lifecycle_registration, owner=owner, prepare=prepare, drain=drain
        ):
            yield self

    def mutation(self, session, *, owner=None, releasing=False):
        self.check_session(session)
        return self.lifecycle.mutation(
            session, session._lifecycle_registration, owner=owner, releasing=releasing
        )

    def fetch(self, session):
        self.check_execution(session)
        if self.fetch_workspace is None:
            raise RuntimeError("This scheme has no sparse fetch workspace")
        return self.fetch_workspace

    def bind_owner(self, owner):
        self.lifecycle.bind_owner(owner)

    def unbind_owner(self, owner, *, rollback=False):
        self.lifecycle.unbind_owner(
            owner, rollback=rollback, release_new=self._drain_and_drop_storage
        )

    def _drain_and_drop_storage(self):
        if self.poisoned:
            raise RuntimeError("Cannot safely release poisoned NOSA execution resources")
        try:
            if self.device.type == "cuda" and (
                self.plan is not None
                or self._allocation_failure is not None
                or any(
                    resource is not None
                    for resource in (self.fetch_workspace, self.attention_workspace, self.staging)
                )
            ):
                torch.cuda.synchronize(self.device)
            self._drop_storage()
        except BaseException as error:
            self.poisoned = True
            raise RuntimeError("Unable to drain NOSA shared allocation") from error

    def _drop_storage(self):
        if self.staging is not None:
            self.staging.close()
        self.fetch_workspace = self.attention_workspace = self.staging = None
        self._allocation_ready = None
        self._allocator_key = None
        self._allocation_failure = None
        self.lifecycle.drop_plan()

    def close(self):
        self.lifecycle.close(self._drain_and_drop_storage)
