"""Token-only model adapter contract for persistent prefix execution."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from cache.prefix_pool import CacheFootprint


@dataclass(frozen=True)
class SharedCachePlan:
    """Allocation-free plan for cache resources owned by the backend.

    ``shared`` reserves the fixed arena, device pools, and reusable execution
    workspace once. Session estimates exclude these bytes. ``host_pages`` is
    an independent admission quota, not another charge for the arena's DRAM.
    ``hbm_tokens`` bounds retained resident history independently of its byte
    footprint; disposable candidate storage does not consume this token quota.
    Backend-specific dimensions and workspace bounds belong in ``metadata``.
    """

    shared: CacheFootprint = field(default_factory=CacheFootprint)
    host_pages: int = 0
    page_size: int = 64
    metadata: Mapping[str, Any] = field(default_factory=dict)
    hbm_tokens: int = 0

    def __post_init__(self):
        if not isinstance(self.shared, CacheFootprint):
            raise TypeError("shared must be a CacheFootprint")
        if type(self.host_pages) is not int or self.host_pages < 0:
            raise ValueError("host_pages must be a nonnegative integer")
        if type(self.page_size) is not int or self.page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        if type(self.hbm_tokens) is not int or self.hbm_tokens < 0:
            raise ValueError("hbm_tokens must be a nonnegative integer")


class SharedServingBackend(Protocol):
    """Optional resource extension; older adapters use a zero-shared plan.

    Planning is pure and precedes allocation. ``allocate_shared`` may reuse an
    identical live plan, but must not grow beyond its reservation. The backend,
    rather than an individual session or runner lease, owns shared storage.

    One non-None admission owner binds before allocation and remains bound even
    when its session pool is empty. Binding fails without mutation if another
    owner, direct session, or execution lease exists. Unbinding requires all
    sessions and leases to have finished. Normal unbinding preserves storage;
    rollback releases only storage allocated since that binding, preserving an
    already allocated plan. A failed drain retains ownership and rejects reuse.
    The outer backend owner calls ``close`` after the runner has closed.

    ``budgets=None`` requests fixed pools without cache byte sub-budgets. A
    supporting backend must require explicit pool capacities and return a
    positive host-page or retained-HBM-token quota that fits one maximum-size
    session. Unsupported
    backends reject this mode before allocating. Byte reservations and their
    allocation audits still apply; they do not establish whole-machine fit.
    """

    def plan_resources(
        self, budgets: CacheFootprint | None, limits: Mapping[str, int]
    ) -> SharedCachePlan: ...

    def allocate_shared(self, plan: SharedCachePlan) -> None: ...

    def bind_owner(self, owner: Any) -> None: ...

    def unbind_owner(self, owner: Any, *, rollback: bool = False) -> None: ...

    def close(self) -> None: ...

    def shared_bytes(self) -> Mapping[str, int]: ...

    def estimate_session_host_pages(self, capacity: int) -> int: ...

    def session_host_pages(self, session: Any) -> int: ...

    # Optional for existing shared backends; required by a positive hbm_tokens
    # plan. The runner treats an absent pair as zero retained HBM token use.
    def estimate_session_hbm_tokens(self, capacity: int) -> int: ...

    def session_hbm_tokens(self, session: Any) -> int: ...


class ServingBackend(Protocol):
    """Persistent token execution, optionally specialized for disposable GR suffixes.

    A backend may expose ``retained_session_capacity(total, prefix)`` so the
    runner admits only persistent history, plus ``extend_candidate`` to consume
    GPU-only suffix state without committing it. Ordinary ``extend`` keeps its
    persistent append contract. These optional hooks do not alter adapters
    which have not implemented disposable suffixes.
    """

    scheme: str
    device: torch.device
    max_seq_len: int

    def estimate_session_bytes(self, capacity: int, prefix_tokens: int) -> dict[str, int]: ...

    def create_session(self, capacity: int) -> Any: ...

    def prefill(self, session: Any, ids: torch.Tensor) -> None: ...

    def extend(self, session: Any, ids: torch.Tensor) -> torch.Tensor: ...

    def truncate(self, session: Any, prefix: int) -> None: ...

    def session_bytes(self, session: Any) -> dict[str, int]: ...

    def release_session(self, session: Any) -> None: ...

    def synchronize(self) -> None: ...

    def describe(self) -> dict: ...
