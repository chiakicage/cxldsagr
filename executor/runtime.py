"""Model-neutral token runtime, preserving driver transaction boundaries."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from cache.capacity import CapacityPolicy, ResourcePlan, SessionPlan
from executor.contracts import ModelDriver, OutputSpec, RequestShape, StageObserver


@dataclass
class _Session:
    value: Any
    plan: SessionPlan
    failed: bool = False


class TokenRuntime:
    """Validate inputs, plan identity, session state and owned output contracts.

    Drivers retain their native compute loop, lease, synchronization and commit
    scope. No runtime operation commits model state a second time. Candidate
    observations preserve the old execution/cleanup measurement boundary.
    """

    def __init__(self, driver: ModelDriver):
        self.driver = driver
        self.plan: ResourcePlan | None = None
        self._sessions: dict[int, _Session] = {}
        self._allocated = False
        self._closed = False

    def plan_resources(self, policy: CapacityPolicy, limits: Mapping[str, int]) -> ResourcePlan:
        if self._closed or self._allocated:
            raise RuntimeError("cannot replan a closed or allocated runtime")
        self.plan = self.driver.plan_resources(policy, limits)
        return self.plan

    def allocate_resources(self, plan: ResourcePlan) -> None:
        if self._closed or plan is not self.plan:
            raise ValueError("resource plan identity does not match this runtime")
        self.driver.allocate_resources(plan)
        self._allocated = True

    def plan_session(self, shape: RequestShape, identity: Hashable) -> SessionPlan:
        if not self._allocated or self._closed:
            raise RuntimeError("allocate resources before planning sessions")
        self.plan.limits.validate(shape)
        return self.driver.plan_session(shape, identity)

    def create_session(self, plan: SessionPlan):
        if self._closed or not self._allocated or plan.resource_identity is not self.plan:
            raise ValueError("session plan belongs to inactive or foreign resources")
        session = self.driver.create_session(plan)
        self._sessions[id(session)] = _Session(session, plan)
        return session

    def _check(self, session, *, allow_failed=False):
        state = self._sessions.get(id(session))
        if state is None or state.value is not session:
            raise ValueError("foreign or released runtime session")
        if self._closed or state.failed and not allow_failed:
            raise RuntimeError("runtime session cannot execute after failure or close")
        return state

    def _tokens(self, ids):
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or not len(ids):
            raise ValueError("tokens must be a nonempty one-dimensional tensor")
        if ids.dtype not in (torch.int32, torch.int64) or ids.device != self.driver.device:
            raise ValueError("tokens must be integral and on the backend device")

    def session_usage(self, session):
        self._check(session, allow_failed=True)
        return self.driver.session_usage(session)

    def shared_usage(self):
        return self.driver.shared_usage()

    def prefill(self, session, ids, output_spec: OutputSpec | None = None):
        state = self._check(session)
        self._tokens(ids)
        if self.driver.session_length(session) != 0:
            raise ValueError("prefill requires an empty session")
        if len(ids) != state.plan.history_tokens:
            raise ValueError("prefill must build the planned history")
        output_spec = output_spec or self.driver.prefill_output_spec
        self.driver.validate_output(output_spec, prefill=True)
        try:
            result = self.driver.prefill(session, ids, output_spec)
            if self.driver.session_length(session) != len(ids):
                raise RuntimeError("prefill did not commit the complete history")
            return result
        except BaseException:
            state.failed = True
            raise

    @staticmethod
    def _check_result(result, output, count):
        rows = count if output.hidden == "all" else 1
        if output.hidden != "none" and (
            result.hidden is None or result.hidden.ndim != 2 or result.hidden.shape[0] != rows
        ):
            raise RuntimeError("backend must return every selected token's hidden state")
        if output.logits == "last" and (
            result.logits is None or result.logits.ndim != 2 or result.logits.shape[0] != 1
        ):
            raise RuntimeError("backend must return the requested last-token logits")
        if result.ownership != output.ownership:
            raise RuntimeError("backend output ownership differs from output spec")

    def candidate(
        self,
        session,
        ids,
        output_spec: OutputSpec | None = None,
        *,
        observer: StageObserver | None = None,
    ):
        state = self._check(session)
        self._tokens(ids)
        history = self.driver.session_length(session)
        if history != state.plan.history_tokens:
            raise ValueError("candidate requires the complete planned history")
        self.plan.limits.validate(RequestShape(history, len(ids)))
        if (
            self.driver.candidate_mode == "append_truncate"
            and history + len(ids) > state.plan.retained_capacity
        ):
            raise ValueError("candidate exceeds session allocation capacity")
        output_spec = output_spec or self.driver.output_spec
        self.driver.validate_output(output_spec)
        try:
            with torch.inference_mode():
                result = self.driver.candidate(session, ids, output_spec)
            self.driver.synchronize()
            self._check_result(result, output_spec, len(ids))
            if observer is not None:
                observer("executed")
            self.driver.finish_candidate(session, history)
            self.driver.synchronize()
            if self.driver.session_length(session) != history:
                raise RuntimeError("candidate changed committed history length")
            if observer is not None:
                observer("cleaned")
            return result
        except BaseException:
            state.failed = True
            raise

    def append(self, session, ids, output_spec: OutputSpec | None = None):
        state = self._check(session)
        self._tokens(ids)
        old_length = self.driver.session_length(session)
        if not old_length:
            raise ValueError("append requires a committed prefix")
        if old_length + len(ids) > min(
            state.plan.retained_capacity, self.plan.limits.context_tokens
        ):
            raise ValueError("append exceeds session allocation capacity")
        if len(ids) > self.plan.limits.candidate_tokens:
            raise ValueError("append exceeds planned execution capacity")
        output_spec = output_spec or self.driver.output_spec
        self.driver.validate_output(output_spec)
        try:
            with torch.inference_mode():
                result = self.driver.append(session, ids, output_spec)
            self.driver.synchronize()
            self._check_result(result, output_spec, len(ids))
            if self.driver.session_length(session) != old_length + len(ids):
                raise RuntimeError("append did not commit all tokens")
            return result
        except BaseException:
            state.failed = True
            raise

    def release_session(self, session):
        self._check(session, allow_failed=True)
        self.driver.release_session(session)
        del self._sessions[id(session)]

    def close(self):
        if self._sessions:
            raise RuntimeError("release runtime sessions before closing resources")
        self.driver.close()
        self._closed = True

    def retire(self):
        """End one runner runtime while its backend keeps shared allocations."""
        if self._sessions:
            raise RuntimeError("release runtime sessions before retiring admission")
        self._closed = True
