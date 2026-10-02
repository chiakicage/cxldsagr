"""Budgeted ownership of persistent prefix sessions, independent of model layout."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheFootprint:
    hbm: int = 0
    dram: int = 0

    def __post_init__(self):
        for name in ("hbm", "dram"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer byte count")

    @classmethod
    def from_mapping(cls, value: Mapping[str, int]) -> CacheFootprint:
        return cls(hbm=value["hbm"], dram=value["dram"])

    def fits(self, budget: CacheFootprint) -> bool:
        return self.hbm <= budget.hbm and self.dram <= budget.dram

    def __add__(self, other: CacheFootprint) -> CacheFootprint:
        return CacheFootprint(self.hbm + other.hbm, self.dram + other.dram)


class CacheBudgetExceeded(ValueError):
    """The requested session cannot fit even after all other users are evicted."""


@dataclass
class PrefixEntry:
    session: Any
    signature: Hashable
    capacity: int
    reservation: CacheFootprint
    ready: bool = False


@dataclass(frozen=True)
class PrefixLease:
    entry: PrefixEntry
    hit: bool
    evicted: tuple[Hashable, ...]


class PrefixSessionPool:
    """Serial LRU with admission reservations covering each session's peak cache use.

    A lease is unpublished until ``mark_ready`` follows successful prefix build.
    Callers discard a session on any execution failure. The supplied release
    callback must finish asynchronous use before freeing cache storage.
    """

    def __init__(
        self,
        budget: CacheFootprint,
        *,
        allocate: Callable[[int], Any],
        release: Callable[[Any], None],
        measure: Callable[[Any], Mapping[str, int]],
    ):
        self.budget = budget
        self._allocate, self._release, self._measure = allocate, release, measure
        self._entries: OrderedDict[Hashable, PrefixEntry] = OrderedDict()
        self.evictions = 0

    @property
    def reserved(self) -> CacheFootprint:
        result = CacheFootprint()
        for entry in self._entries.values():
            result += entry.reservation
        return result

    def __len__(self):
        return len(self._entries)

    def discard(self, key: Hashable) -> None:
        entry = self._entries.get(key)
        if entry is not None:
            self._release(entry.session)
            del self._entries[key]

    def acquire(
        self,
        key: Hashable,
        signature: Hashable,
        capacity: int,
        reservation: CacheFootprint,
    ) -> PrefixLease:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        # An impossible request must not destroy any existing reusable state.
        if not reservation.fits(self.budget):
            raise CacheBudgetExceeded(
                f"session requires {reservation}, exceeds cache budget {self.budget}"
            )
        prior = self._entries.get(key)
        if (
            prior is not None
            and prior.ready
            and prior.signature == signature
            and prior.capacity >= capacity
        ):
            self._entries.move_to_end(key)
            return PrefixLease(prior, True, ())
        evicted = []
        if prior is not None:
            self.discard(key)
            evicted.append(key)
        while not (self.reserved + reservation).fits(self.budget):
            victim = next(iter(self._entries))
            self.discard(victim)
            evicted.append(victim)
        self.evictions += len(evicted)
        session = self._allocate(capacity)
        entry = PrefixEntry(session, signature, capacity, reservation)
        self._entries[key] = entry
        try:
            self.audit()
        except BaseException:
            self.discard(key)
            raise
        return PrefixLease(entry, False, tuple(evicted))

    def mark_ready(self, key: Hashable) -> None:
        self._entries[key].ready = True

    def audit(self) -> CacheFootprint:
        """Check actual allocated cache bytes against both admission and global caps."""
        actual = CacheFootprint()
        for key, entry in self._entries.items():
            measured = CacheFootprint.from_mapping(self._measure(entry.session))
            if not measured.fits(entry.reservation):
                raise RuntimeError(
                    f"cache allocation for {key!r} exceeded reservation: "
                    f"actual={measured}, reserved={entry.reservation}"
                )
            actual += measured
        if not actual.fits(self.budget) or not self.reserved.fits(self.budget):
            raise RuntimeError("global cache budget invariant violated")
        return actual

    def close(self) -> None:
        for key in tuple(self._entries):
            self.discard(key)
