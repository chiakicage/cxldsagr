"""Explicit bridge from existing model loops to the token runtime contract."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Literal

from cache.capacity import (
    CacheFootprint,
    CapacityPolicy,
    ResourcePlan,
    ResourceUsage,
    SessionPlan,
)
from executor.contracts import ExecutionLimits, ExecutionResult, OutputSpec, RequestShape


def _require_methods(target, names):
    # Construction-time contract validation, never execution strategy discovery.
    for name in names:
        if not callable(getattr(target, name, None)):
            raise TypeError(f"backend must implement {name}")


class BackendAdapter:
    """Keep established computation, allocation and synchronization boundaries.

    The model builder must declare shared ownership, candidate strategy, output
    workload and length access. Missing capabilities are errors; no method or
    report metadata changes the declared execution policy.
    """

    def __init__(
        self,
        backend: Any,
        *,
        shared: bool,
        candidate_mode: Literal["gpu_transient", "append_truncate"],
        session_length: Callable[[Any], int],
        chunk_size: int,
        run_lm_head: bool = False,
        last_logits: Callable[[], Any] | None = None,
        diagnostics: Callable[[Any], dict] | None = None,
        owner_aware: bool = False,
        session_planner: Callable[[ResourcePlan, RequestShape, Hashable], SessionPlan]
        | None = None,
        session_factory: Callable[..., Any] | None = None,
    ):
        if candidate_mode not in ("gpu_transient", "append_truncate"):
            raise ValueError("candidate mode must be explicit")
        if type(shared) is not bool or type(run_lm_head) is not bool:
            raise TypeError("shared and run_lm_head must be boolean")
        if run_lm_head and last_logits is None:
            raise ValueError("LM head execution requires an explicit logits accessor")
        if (session_planner is None) != (session_factory is None):
            raise ValueError("native session planning and allocation must be provided together")
        _require_methods(
            backend,
            (
                "estimate_session_bytes",
                "create_session",
                "prefill",
                "extend",
                "truncate",
                "session_bytes",
                "release_session",
                "synchronize",
            ),
        )
        if shared:
            _require_methods(
                backend,
                (
                    "plan_resources",
                    "allocate_shared",
                    "bind_owner",
                    "unbind_owner",
                    "close",
                    "shared_bytes",
                    "estimate_session_host_pages",
                    "session_host_pages",
                ),
            )
        if candidate_mode == "gpu_transient":
            _require_methods(backend, ("extend_candidate",))
        self.backend = backend
        self.shared = shared
        self.candidate_mode = candidate_mode
        self.session_length = session_length
        self.device = backend.device
        self.max_seq_len = backend.max_seq_len
        self.chunk_size = chunk_size
        self.output_spec = OutputSpec(run_lm_head=run_lm_head)
        self.prefill_output_spec = OutputSpec(hidden="none", run_lm_head=run_lm_head)
        self._last_logits = last_logits
        self._diagnostics = diagnostics
        self._owner_aware = owner_aware
        self._owner = None
        self._session_planner = session_planner
        self._session_factory = session_factory
        self._plan: ResourcePlan | None = None
        self._model_plan: ResourcePlan | None = None

    def plan_resources(self, policy: CapacityPolicy, limits: Mapping[str, int]) -> ResourcePlan:
        if not self.shared and policy.mode == "fixed_pools":
            raise ValueError(
                "fixed-pools admission requires a shared backend with host pages or HBM tokens"
            )
        requested = ExecutionLimits.from_mapping(
            limits, context=self.max_seq_len, chunk_size=self.chunk_size
        )
        model_plan = (
            self.backend.plan_resources(policy.budget, limits) if self.shared else ResourcePlan()
        )
        if not isinstance(model_plan, ResourcePlan):
            raise TypeError("plan_resources must return a ResourcePlan (ResourcePlan)")
        if model_plan.hbm_tokens:
            _require_methods(self.backend, ("estimate_session_hbm_tokens", "session_hbm_tokens"))
        metadata = dict(model_plan.metadata)
        effective = ExecutionLimits.from_mapping(
            {
                "max_seq_len": requested.context_tokens,
                "max_session_capacity": min(
                    requested.request_tokens,
                    metadata.get("max_session_capacity", requested.request_tokens),
                ),
                "max_history_tokens": min(
                    requested.history_tokens,
                    metadata.get("max_history_tokens", requested.history_tokens),
                ),
                "max_candidate_tokens": min(
                    requested.candidate_tokens,
                    metadata.get("max_candidate_tokens", requested.candidate_tokens),
                ),
                "max_query_tokens": min(
                    requested.query_tokens,
                    metadata.get(
                        "max_query_tokens",
                        metadata.get("workspace_query_tokens", requested.query_tokens),
                    ),
                ),
            },
            context=requested.context_tokens,
            chunk_size=self.chunk_size,
        )
        # Derived report metadata is never read to select a candidate entrypoint.
        metadata["candidate_persistence"] = (
            "gpu_transient" if self.candidate_mode == "gpu_transient" else "committed"
        )
        self._model_plan = model_plan
        self._plan = replace(
            model_plan, policy=policy, limits=effective, metadata=MappingProxyType(metadata)
        )
        return self._plan

    def allocate_resources(self, plan: ResourcePlan) -> None:
        if plan is not self._plan:
            raise ValueError("resource allocation requires the exact planned resource identity")
        if self.shared:
            self.backend.allocate_shared(self._model_plan, **self._owner_options())

    def _owner_options(self):
        return {"owner": self._owner} if self._owner_aware else {}

    def bind_owner(self, owner):
        if self.shared:
            self.backend.bind_owner(owner)
        self._owner = owner

    def unbind_owner(self, owner, *, rollback=False):
        if self.shared:
            self.backend.unbind_owner(owner, rollback=rollback)
        self._owner = None

    def _quota(self, capacity: int) -> tuple[int, int]:
        pages = self.backend.estimate_session_host_pages(capacity) if self._plan.host_pages else 0
        tokens = self.backend.estimate_session_hbm_tokens(capacity) if self._plan.hbm_tokens else 0
        if type(pages) is not int or pages < 0:
            raise ValueError("session host pages must be a nonnegative integer")
        if type(tokens) is not int or tokens < 0:
            raise ValueError("session HBM tokens must be a nonnegative integer")
        return pages, tokens

    def maximum_session_quota(self, plan: ResourcePlan) -> tuple[int, int]:
        capacity = (
            plan.limits.history_tokens
            if self.candidate_mode == "gpu_transient"
            else plan.limits.request_tokens
        )
        return self._quota(capacity)

    def plan_session(self, shape: RequestShape, identity: Hashable) -> SessionPlan:
        self._plan.limits.validate(shape)
        if self._session_planner is not None:
            plan = self._session_planner(self._plan, shape, identity)
            if not isinstance(plan, SessionPlan) or plan.resource_identity is not self._plan:
                raise ValueError("model session planner must retain resource plan identity")
            if plan.history_tokens != shape.history_tokens or plan.history_identity != identity:
                raise ValueError("model session planner changed the history identity")
            return plan
        capacity = (
            shape.history_tokens if self.candidate_mode == "gpu_transient" else shape.total_tokens
        )
        reservation = CacheFootprint.from_mapping(
            self.backend.estimate_session_bytes(capacity, shape.history_tokens)
        )
        pages, tokens = self._quota(capacity)
        return SessionPlan(
            self._plan,
            identity,
            shape.history_tokens,
            capacity,
            reservation,
            pages,
            tokens,
            model_plan=capacity,
        )

    def create_session(self, plan: SessionPlan):
        if plan.resource_identity is not self._plan:
            raise ValueError("session plan belongs to different resources")
        if self._session_factory is not None:
            return self._session_factory(plan, **self._owner_options())
        return self.backend.create_session(plan.model_plan, **self._owner_options())

    def session_usage(self, session):
        return ResourceUsage(
            CacheFootprint.from_mapping(self.backend.session_bytes(session)),
            host_pages=self.backend.session_host_pages(session) if self._plan.host_pages else 0,
            hbm_tokens=self.backend.session_hbm_tokens(session) if self._plan.hbm_tokens else 0,
        )

    def shared_usage(self):
        charged = (
            CacheFootprint.from_mapping(self.backend.shared_bytes())
            if self.shared
            else CacheFootprint()
        )
        return ResourceUsage(charged, owner="shared")

    def validate_output(self, output: OutputSpec, *, prefill=False):
        if output.run_lm_head != self.output_spec.run_lm_head:
            raise ValueError("output spec would change this model adapter's LM head workload")
        if output.logits == "all":
            raise ValueError("this adapter only computes last-token logits")
        if output.ownership != "owned":
            raise ValueError("this adapter exposes owned outputs only")
        if prefill and output.hidden != "none":
            raise ValueError("this adapter discards prefill hidden outputs")

    def _result(self, hidden, output):
        if output.hidden == "none":
            hidden = None
        elif output.hidden == "last":
            hidden = hidden[-1:]
        logits = self._last_logits() if output.logits == "last" else None
        return ExecutionResult(hidden, logits)

    def prefill(self, session, ids, output):
        self.validate_output(output, prefill=True)
        hidden = self.backend.prefill(session, ids, **self._owner_options())
        return self._result(hidden, output)

    def candidate(self, session, ids, output):
        self.validate_output(output)
        if self.candidate_mode == "gpu_transient":
            hidden = self.backend.extend_candidate(session, ids, **self._owner_options())
        else:
            hidden = self.backend.extend(session, ids, **self._owner_options())
        return self._result(hidden, output)

    def finish_candidate(self, session, history_tokens):
        # Keep the established cleanup call and its model-specific drain behavior.
        self.backend.truncate(session, history_tokens, **self._owner_options())

    def append(self, session, ids, output):
        self.validate_output(output)
        return self._result(self.backend.extend(session, ids, **self._owner_options()), output)

    def diagnostics(self, session):
        return {} if self._diagnostics is None else self._diagnostics(session)

    def release_session(self, session):
        self.backend.release_session(session, **self._owner_options())

    def synchronize(self):
        self.backend.synchronize()

    def close(self):
        if self.shared:
            self.backend.close()
