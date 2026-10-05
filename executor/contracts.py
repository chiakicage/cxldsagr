"""Token execution contracts; no GR requests or concrete model dependencies."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from cache.capacity import CapacityPolicy, ResourcePlan, ResourceUsage, SessionPlan

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class RequestShape:
    history_tokens: int
    candidate_tokens: int

    def __post_init__(self):
        for name in ("history_tokens", "candidate_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def total_tokens(self) -> int:
        return self.history_tokens + self.candidate_tokens


@dataclass(frozen=True)
class ExecutionLimits:
    context_tokens: int
    request_tokens: int
    history_tokens: int
    candidate_tokens: int
    history_chunk_tokens: int
    query_tokens: int

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.request_tokens > self.context_tokens:
            raise ValueError("planned request exceeds backend context limit")
        if max(self.history_tokens, self.candidate_tokens) > self.request_tokens:
            raise ValueError("history/candidate limit exceeds planned request capacity")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, int], *, context: int, chunk_size: int
    ) -> ExecutionLimits:
        if any(type(value) is not int or value <= 0 for value in values.values()):
            raise ValueError("resource limits must be positive integers")
        total = values.get("max_session_capacity", context)
        return cls(
            min(context, values.get("max_seq_len", context)),
            total,
            values.get("max_history_tokens", total),
            values.get("max_candidate_tokens", total),
            chunk_size,
            values.get("max_query_tokens", values.get("workspace_query_tokens", total)),
        )

    def validate(self, shape: RequestShape) -> None:
        if shape.total_tokens > self.context_tokens:
            raise ValueError("request exceeds backend context limit")
        if shape.total_tokens > self.request_tokens:
            raise ValueError("request exceeds planned session capacity")
        if shape.history_tokens > self.history_tokens:
            raise ValueError("history exceeds planned retained capacity")
        if shape.candidate_tokens > self.candidate_tokens:
            raise ValueError("candidate suffix exceeds planned execution capacity")


@dataclass(frozen=True)
class OutputSpec:
    hidden: Literal["none", "last", "all"] = "all"
    logits: Literal["none", "last", "all"] = "none"
    run_lm_head: bool = False
    ownership: Literal["owned", "borrowed"] = "owned"

    def __post_init__(self):
        if self.hidden not in ("none", "last", "all") or self.logits not in ("none", "last", "all"):
            raise ValueError("output selection must be none, last or all")
        if type(self.run_lm_head) is not bool:
            raise TypeError("run_lm_head must be boolean")
        if self.logits != "none" and not self.run_lm_head:
            raise ValueError("logits require LM head execution")
        if self.ownership not in ("owned", "borrowed"):
            raise ValueError("output ownership must be owned or borrowed")


@dataclass(frozen=True)
class ExecutionResult:
    hidden: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    ownership: Literal["owned", "borrowed"] = "owned"


StageObserver = Callable[[Literal["executed", "cleaned"]], None]


class ModelDriver(Protocol):
    """Explicit model operations; transaction scope and GPU leases stay local.

    Migration adapters implement this protocol without replacing model loops or
    committing a second time. ResourceLifecycle providers are composed beneath
    this interface, so the runtime does not duplicate provider synchronization.
    """

    device: torch.device
    max_seq_len: int
    chunk_size: int
    output_spec: OutputSpec
    prefill_output_spec: OutputSpec
    candidate_mode: Literal["gpu_transient", "append_truncate"]

    def plan_resources(self, policy: CapacityPolicy, limits: Mapping[str, int]) -> ResourcePlan: ...
    def allocate_resources(self, plan: ResourcePlan) -> None: ...
    def bind_owner(self, owner: Any) -> None: ...
    def unbind_owner(self, owner: Any, *, rollback: bool = False) -> None: ...
    def maximum_session_quota(self, plan: ResourcePlan) -> tuple[int, int]: ...
    def plan_session(self, shape: RequestShape, identity: Hashable) -> SessionPlan: ...
    def create_session(self, plan: SessionPlan) -> Any: ...
    def session_usage(self, session: Any) -> ResourceUsage: ...
    def shared_usage(self) -> ResourceUsage: ...
    def session_length(self, session: Any) -> int: ...
    def prefill(self, session: Any, ids: torch.Tensor, output: OutputSpec) -> ExecutionResult: ...
    def candidate(self, session: Any, ids: torch.Tensor, output: OutputSpec) -> ExecutionResult: ...
    def finish_candidate(self, session: Any, history_tokens: int) -> None: ...
    def append(self, session: Any, ids: torch.Tensor, output: OutputSpec) -> ExecutionResult: ...
    def validate_output(self, output: OutputSpec, *, prefill: bool = False) -> None: ...
    def diagnostics(self, session: Any) -> dict: ...
    def release_session(self, session: Any) -> None: ...
    def synchronize(self) -> None: ...
    def close(self) -> None: ...


class ServingBackend(Protocol):
    """A model entrypoint declares its complete runtime driver before admission."""

    scheme: str
    device: torch.device
    max_seq_len: int

    def runtime_driver(self, policy: CapacityPolicy) -> ModelDriver: ...
    def describe(self) -> dict: ...
