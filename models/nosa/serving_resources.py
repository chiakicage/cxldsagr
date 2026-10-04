"""Backend-owned, bounded execution storage for serial NOSA serving.

CPU allocations exercise layout and ownership only. Their ``hbm`` ledger is a
resource role, not a measurement of GPU memory. Model weights and ordinary
activations are outside this module; persistent histories remain session-owned.
"""

from contextlib import contextmanager
from threading import Lock
from types import MappingProxyType

import torch

from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint
from executor.serving_backend import SharedCachePlan
from models.nosa.allocation_budget import allocation_bytes, validate_allocator

POLICY_REVISION = "nosa_backend_workspace_v2"


def _positive(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class NosaExecutionResources:
    """One fixed workspace, with independent session identities and one lease."""

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
        self.plan = None
        self.generation = 0
        self.fetch_workspace = self.attention_workspace = self.staging = None
        self._allocation_ready = None
        self._allocator_key = None
        self.allowed_graph_pool_ids = set()
        self.staging_lease = None
        self._sessions = set()
        self._identity = object()
        self._active_session = None
        self._lock = Lock()
        self._admission_owner = None
        self._admission_initial_plan = None
        self._allocation_failure = None
        self.poisoned = False

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

    def plan_resources(self, budgets, limits):
        """Estimate from dimensions alone; do not allocate or mutate a live plan."""
        if not isinstance(budgets, CacheFootprint):
            raise TypeError("budgets must be a CacheFootprint")
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
        from operators.nosa.attention.workspace import NosaAttentionWorkspace

        stages = {"hbm": 0, "dense_prefetch": 2}.get(self.scheme, 1)
        if stages == 1:
            from operators.nosa.attention.offload.api import NosaFetchWorkspace

            allocations = NosaFetchWorkspace.allocation_sizes(
                capacity,
                self.kv_heads,
                self.head_dim,
                dtype=self.dtype,
                query_tile_size=self.query_tile_size,
                max_queries=queries,
                trace_capacity=trace_rows,
            )
        else:
            if trace_rows:
                raise ValueError("Trace storage is supported only by sparse fetch schemes")
            allocations = NosaAttentionWorkspace.allocation_sizes(
                queries,
                self.kv_heads,
                self.head_dim,
                dtype=self.dtype,
            )
            if stages:
                # DoubleBufferStaging owns one [2, C, H, D] storage per record.
                stage_bytes = 2 * capacity * self.kv_heads * self.head_dim * self.dtype.itemsize
                allocations = (*allocations, stage_bytes, stage_bytes)
        logical_shared = sum(allocations)
        shared = sum(allocation_bytes(size, self.device) for size in allocations)
        result = SharedCachePlan(
            shared=CacheFootprint(hbm=shared),
            host_pages=0,
            metadata=MappingProxyType(
                {
                    "policy_revision": POLICY_REVISION,
                    "allocator_policy": allocator["policy"],
                    "logical_shared_hbm_bytes": logical_shared,
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
            ),
        )
        if not result.shared.fits(budgets):
            raise CacheBudgetExceeded(f"Shared NOSA workspace {result.shared} exceeds {budgets}")
        return result

    @property
    def record_shapes(self):
        return {name: (self.kv_heads, self.head_dim) for name in ("keys", "values")}

    def allocate_shared(self, plan):
        if self.poisoned:
            raise RuntimeError("NOSA execution resources are poisoned")
        if not isinstance(plan, SharedCachePlan):
            raise TypeError("Expected SharedCachePlan")
        expected = self.plan_resources(plan.shared, plan.metadata)
        if expected != plan:
            raise ValueError("Shared plan does not match this backend's layout")
        if self.plan == plan:
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
            self.plan = plan
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
            self._drain_and_drop_storage()
            raise
        self.generation += 1

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

    def attach(self, session):
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
        self._sessions.add(session)
        try:
            if self.device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
                session._execution_ready = ready
        except BaseException:
            try:
                torch.cuda.synchronize(self.device)
                session.release()
            except BaseException as error:
                self.poisoned = True
                raise RuntimeError("Unable to drain failed NOSA session attachment") from error
            raise

    def check_session(self, session, *, allow_released=False):
        if getattr(session, "_serving_identity", None) is not self._identity:
            raise ValueError("Foreign session belongs to another backend")
        if allow_released and session.released:
            return
        if (
            session not in self._sessions
            or session.released
            or session._execution_generation != self.generation
        ):
            raise RuntimeError("Stale or released NOSA session")

    def check_execution(self, session, token_count=None):
        self.check_session(session)
        if self.poisoned:
            raise RuntimeError("NOSA execution resources are poisoned")
        if self._active_session is not session:
            raise RuntimeError("Borrowed NOSA cache requires an active execution lease")
        if token_count is not None:
            _positive("query length", token_count)
            if token_count > self.plan.metadata["max_query_tokens"]:
                raise ValueError("Query length exceeds planned execution capacity")

    def check_mutation(self, session, *, releasing=False):
        self.check_session(session)
        if self.poisoned:
            raise RuntimeError("Cannot mutate a session with poisoned execution resources")
        if self._active_session is not None and (releasing or self._active_session is not session):
            raise RuntimeError("Session mutation conflicts with an execution lease")

    def detach(self, session):
        self._sessions.remove(session)
        session._execution_ready = None

    @contextmanager
    def lease(self, session):
        self.check_session(session)
        self.validate_native()
        self._validate_allocator()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise NotImplementedError(
                        "Shared NOSA serving does not support CUDA Graph capture"
                    )
        if self.poisoned:
            raise RuntimeError("NOSA execution resources are poisoned")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent execution lease is not supported")
        self._active_session = session
        try:
            if self.device.type == "cuda":
                current = torch.cuda.current_stream(self.device)
                for event in (self._allocation_ready, session._execution_ready):
                    if event is not None:
                        current.wait_event(event)
            if self.staging is not None:
                self.staging_lease = self.staging.lease(session, generation=self.generation)
            yield self
        finally:
            failures = []
            try:
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
                    self.poisoned = True
                    # Preserve ownership and aliases while completion is unknown.
                    raise RuntimeError("Unable to drain NOSA execution resources") from failures[0]
                self.staging_lease = None
                self._active_session = None
            finally:
                self._lock.release()

    def fetch(self, session):
        self.check_execution(session)
        if self.fetch_workspace is None:
            raise RuntimeError("This scheme has no sparse fetch workspace")
        return self.fetch_workspace

    def bind_owner(self, owner):
        if owner is None:
            raise ValueError("Admission requires a non-None owner identity")
        if self.poisoned:
            raise RuntimeError("Cannot bind poisoned NOSA execution resources")
        if self._admission_owner is not None:
            raise RuntimeError("Backend already has an admission owner")
        if self._sessions or self._active_session is not None:
            raise RuntimeError("Cannot bind admission with existing direct sessions")
        self._admission_owner = owner
        self._admission_initial_plan = self.plan

    def unbind_owner(self, owner, *, rollback=False):
        if owner is None or self._admission_owner is not owner:
            raise ValueError("Foreign admission owner")
        if self._sessions or self._active_session is not None:
            raise RuntimeError("Release sessions and leases before unbinding admission")
        if self.poisoned:
            raise RuntimeError("Cannot unbind poisoned NOSA execution resources")
        if rollback and self._admission_initial_plan is None:
            self._drain_and_drop_storage()
        self._admission_owner = None
        self._admission_initial_plan = None

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
        self.plan = None

    def close(self):
        if self._sessions or self._active_session is not None or self._admission_owner is not None:
            raise RuntimeError(
                "Cannot close resources with live sessions, lease or admission owner"
            )
        self._drain_and_drop_storage()
