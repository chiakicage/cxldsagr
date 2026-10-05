"""Pure capacity declarations, independent of models, allocators and admission."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from executor.contracts import ExecutionLimits


_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2fnuz": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


def _immutable(*args, **kwargs):
    raise TypeError("resource plan metadata is immutable")


class _FrozenDict(dict):
    """Keep normal JSON/equality behavior while preventing in-place plan edits."""

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class _FrozenList(list):
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _immutable
    append = clear = extend = insert = pop = remove = reverse = sort = _immutable

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _freeze_metadata(value):
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("resource plan metadata keys must be strings")
        return _FrozenDict((key, _freeze_metadata(item)) for key, item in value.items())
    if isinstance(value, list):
        return _FrozenList(_freeze_metadata(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata(item) for item in value)
    raise TypeError("resource plan metadata must contain only immutable data declarations")


def nonnegative(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class CacheFootprint:
    hbm: int = 0
    dram: int = 0

    def __post_init__(self):
        nonnegative(self.hbm, "hbm byte count")
        nonnegative(self.dram, "dram byte count")

    @classmethod
    def from_mapping(cls, value: Mapping[str, int]) -> CacheFootprint:
        return cls(hbm=value["hbm"], dram=value["dram"])

    def fits(self, budget: CacheFootprint) -> bool:
        return self.hbm <= budget.hbm and self.dram <= budget.dram

    def __add__(self, other: CacheFootprint) -> CacheFootprint:
        return CacheFootprint(self.hbm + other.hbm, self.dram + other.dram)


@dataclass(frozen=True)
class CapacityPolicy:
    mode: Literal["budget", "fixed_pools"]
    budget: CacheFootprint | None = None

    def __post_init__(self):
        if self.mode not in ("budget", "fixed_pools"):
            raise ValueError("unknown capacity policy")
        if (self.mode == "budget") != isinstance(self.budget, CacheFootprint):
            raise ValueError("byte budget policy requires a CacheFootprint; fixed pools omit it")
        if self.mode == "fixed_pools" and self.budget is not None:
            raise ValueError("fixed pools cannot carry a cache byte sub-budget")

    @classmethod
    def byte_budget(cls, budget: CacheFootprint) -> CapacityPolicy:
        return cls("budget", budget)

    @classmethod
    def fixed_pools(cls) -> CapacityPolicy:
        return cls("fixed_pools")


@dataclass(frozen=True)
class AllocationSpec:
    """One named storage or alias, with the model's allocator reservation.

    A ``peak_group`` contains mutually exclusive phases on the same device.
    Allocations in one ``phase`` coexist and add; only the largest phase counts.
    With no phase label, each item is an independent alternative. Aliases carry
    no additional charge. ``accounting_tier`` permits the existing CPU reference
    tests to express device/host storage roles without claiming physical HBM.
    Storage size and allocator charge are separate, including conservative bounds.
    """

    name: str
    owner: Literal["shared", "session"]
    dtype: str
    shape: tuple[int, ...]
    device: str
    lifetime: str
    storage_bytes: int
    charged_bytes: int
    alias_of: str | None = None
    peak_group: str | None = None
    phase: str | None = None
    accounting_tier: Literal["hbm", "dram"] | None = None

    def __post_init__(self):
        if not self.name or not self.dtype or not self.device or not self.lifetime:
            raise ValueError("allocation name, dtype, device and lifetime must be explicit")
        if self.owner not in ("shared", "session"):
            raise ValueError("allocation owner must be shared or session")
        if self.dtype.removeprefix("torch.") not in _DTYPE_BYTES:
            raise ValueError("allocation dtype must have an explicit element size")
        if self.device.split(":", 1)[0] not in ("cpu", "cuda"):
            raise ValueError("allocation device must be CPU or CUDA")
        if self.accounting_tier not in (None, "hbm", "dram"):
            raise ValueError("allocation accounting tier must be hbm or dram")
        if (
            self.peak_group == ""
            or self.phase == ""
            or self.phase is not None
            and self.peak_group is None
        ):
            raise ValueError("allocation phase requires a nonempty peak group")
        if not isinstance(self.shape, tuple):
            raise TypeError("allocation shape must be an immutable tuple")
        for dimension in self.shape:
            nonnegative(dimension, "allocation dimension")
        nonnegative(self.storage_bytes, "storage_bytes")
        nonnegative(self.charged_bytes, "charged_bytes")
        if self.logical_bytes > self.storage_bytes:
            raise ValueError("allocation storage cannot contain its declared dtype and shape")
        if self.alias_of == self.name:
            raise ValueError("an allocation cannot alias itself")
        if self.alias_of is not None and self.charged_bytes:
            raise ValueError("an alias must not charge its owning storage again")
        if self.alias_of is None and self.charged_bytes < self.storage_bytes:
            raise ValueError("allocation charge must cover its owning storage")

    @property
    def logical_bytes(self) -> int:
        return prod(self.shape) * _DTYPE_BYTES[self.dtype.removeprefix("torch.")]

    @property
    def tier(self) -> Literal["hbm", "dram"]:
        return self.accounting_tier or ("hbm" if self.device.split(":", 1)[0] == "cuda" else "dram")


def allocation_footprint(allocations: tuple[AllocationSpec, ...]) -> CacheFootprint:
    by_name = {item.name: item for item in allocations}
    if len(by_name) != len(allocations):
        raise ValueError("allocation names must be unique")
    totals = {"hbm": 0, "dram": 0}
    phases: dict[tuple[str, str, str, str], dict[str, int]] = {}
    for item in allocations:
        if item.alias_of is not None:
            target = by_name.get(item.alias_of)
            if target is None or target.alias_of is not None:
                raise ValueError("allocation alias must identify an owning storage")
            if (
                target.device != item.device
                or target.owner != item.owner
                or target.tier != item.tier
            ):
                raise ValueError("allocation alias must retain storage device and owner")
            if item.storage_bytes > target.storage_bytes:
                raise ValueError("allocation alias exceeds owning storage")
            continue
        tier = item.tier
        if item.peak_group is None:
            totals[tier] += item.charged_bytes
        else:
            key = (tier, item.owner, item.device, item.peak_group)
            group = phases.setdefault(key, {})
            phase = item.phase if item.phase is not None else item.name
            group[phase] = group.get(phase, 0) + item.charged_bytes
    for (tier, _, _, _), group in phases.items():
        totals[tier] += max(group.values())
    return CacheFootprint(**totals)


@dataclass(frozen=True)
class ResourcePlan:
    """Shared reservations and independent admission quotas.

    ``metadata`` holds the immutable model plan. It never selects execution
    behavior; the model driver declares that before planning or allocation.
    The optional policy/limits fields permit explicit adaptation of existing
    model planners while they migrate to this contract.
    """

    shared: CacheFootprint = field(default_factory=CacheFootprint)
    host_pages: int = 0
    page_size: int = 64
    metadata: Mapping[str, Any] = field(default_factory=dict)
    hbm_tokens: int = 0
    allocations: tuple[AllocationSpec, ...] = ()
    policy: CapacityPolicy | None = None
    limits: ExecutionLimits | None = None

    def __post_init__(self):
        if not isinstance(self.shared, CacheFootprint):
            raise TypeError("shared must be a CacheFootprint")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("resource plan metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        nonnegative(self.host_pages, "host_pages")
        nonnegative(self.hbm_tokens, "hbm_tokens")
        if type(self.page_size) is not int or self.page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        if not isinstance(self.allocations, tuple):
            raise TypeError("resource allocations must be an immutable tuple")
        if any(item.owner != "shared" for item in self.allocations):
            raise ValueError("resource plan must contain only shared allocations")
        if self.allocations and not allocation_footprint(self.allocations).fits(self.shared):
            raise ValueError("shared allocations exceed their reservation")


@dataclass(frozen=True)
class SessionPlan:
    """One admission decision, consumed unchanged by session allocation."""

    resource_identity: object
    history_identity: Hashable
    history_tokens: int
    retained_capacity: int
    reservation: CacheFootprint
    host_pages: int = 0
    hbm_tokens: int = 0
    allocations: tuple[AllocationSpec, ...] = ()
    model_plan: Any = None

    def __post_init__(self):
        if type(self.history_tokens) is not int or self.history_tokens <= 0:
            raise ValueError("history_tokens must be a positive integer")
        if type(self.retained_capacity) is not int or self.retained_capacity < self.history_tokens:
            raise ValueError("retained capacity must cover the history")
        hash(self.history_identity)
        if not isinstance(self.reservation, CacheFootprint):
            raise TypeError("session reservation must be a CacheFootprint")
        nonnegative(self.host_pages, "session host_pages")
        nonnegative(self.hbm_tokens, "session hbm_tokens")
        if not isinstance(self.allocations, tuple):
            raise TypeError("session allocations must be an immutable tuple")
        if any(item.owner != "session" for item in self.allocations):
            raise ValueError("session plan must contain only session allocations")
        if self.allocations and not allocation_footprint(self.allocations).fits(self.reservation):
            raise ValueError("session allocations exceed their reservation")


@dataclass(frozen=True)
class ResourceUsage:
    """Owned cache storage and admission charge; process metrics stay separate."""

    charged: CacheFootprint
    storage: CacheFootprint | None = None
    host_pages: int = 0
    hbm_tokens: int = 0
    owner: Literal["shared", "session"] = "session"

    def __post_init__(self):
        if not isinstance(self.charged, CacheFootprint):
            raise TypeError("resource usage charge must be a CacheFootprint")
        if self.storage is not None and (
            not isinstance(self.storage, CacheFootprint) or not self.storage.fits(self.charged)
        ):
            raise ValueError("resource usage charge must cover its storage")
        nonnegative(self.host_pages, "used host_pages")
        nonnegative(self.hbm_tokens, "used hbm_tokens")
        if self.owner not in ("shared", "session"):
            raise ValueError("resource usage owner must be shared or session")

    def as_mapping(self) -> dict[str, int]:
        return {"hbm": self.charged.hbm, "dram": self.charged.dram}
