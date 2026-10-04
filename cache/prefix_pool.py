"""Persistent prefix ownership with byte budgets or fixed storage quotas."""

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
    """The requested resources cannot fit even after all users are evicted."""


@dataclass
class PrefixEntry:
    session: Any
    signature: Hashable
    capacity: int
    reservation: CacheFootprint
    host_pages: int = 0
    ready: bool = False
    hbm_tokens: int = 0


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
    Shared storage remains owned by the backend after ``close`` releases users.
    Host page reservations track arena occupancy separately from physical DRAM.
    ``budget=None`` uses host pages or retained HBM tokens for LRU;
    allocation reservations remain checked, without a cache byte sub-budget.
    """

    def __init__(
        self,
        budget: CacheFootprint | None,
        *,
        allocate: Callable[[int], Any],
        release: Callable[[Any], None],
        measure: Callable[[Any], Mapping[str, int]],
        shared: CacheFootprint | None = None,
        measure_shared: Callable[[], Mapping[str, int]] | None = None,
        host_page_capacity: int = 0,
        measure_host_pages: Callable[[Any], int] | None = None,
        hbm_token_capacity: int = 0,
        measure_hbm_tokens: Callable[[Any], int] | None = None,
    ):
        shared = CacheFootprint() if shared is None else shared
        if type(host_page_capacity) is not int or host_page_capacity < 0:
            raise ValueError("host_page_capacity must be a nonnegative integer")
        if type(hbm_token_capacity) is not int or hbm_token_capacity < 0:
            raise ValueError("hbm_token_capacity must be a nonnegative integer")
        if budget is None and host_page_capacity == hbm_token_capacity == 0:
            raise ValueError(
                "fixed-pools admission requires a positive host page capacity or HBM token capacity"
            )
        if budget is not None and not shared.fits(budget):
            raise CacheBudgetExceeded(
                f"shared cache requires {shared}, exceeds cache budget {budget}"
            )
        self.budget = budget
        self.shared = shared
        self.host_page_capacity = host_page_capacity
        self.hbm_token_capacity = hbm_token_capacity
        self._measure_shared = measure_shared
        self._measure_host_pages = measure_host_pages
        self._measure_hbm_tokens = measure_hbm_tokens
        self._allocate, self._release, self._measure = allocate, release, measure
        self._entries: OrderedDict[Hashable, PrefixEntry] = OrderedDict()
        self.evictions = 0

    @property
    def reserved(self) -> CacheFootprint:
        """Fixed shared storage plus all independent session reservations."""
        return self.shared + self.session_reserved

    @property
    def session_reserved(self) -> CacheFootprint:
        result = CacheFootprint()
        for entry in self._entries.values():
            result += entry.reservation
        return result

    @property
    def reserved_host_pages(self) -> int:
        return sum(entry.host_pages for entry in self._entries.values())

    @property
    def available_host_pages(self) -> int:
        return self.host_page_capacity - self.reserved_host_pages

    @property
    def reserved_hbm_tokens(self) -> int:
        return sum(entry.hbm_tokens for entry in self._entries.values())

    @property
    def available_hbm_tokens(self) -> int:
        return self.hbm_token_capacity - self.reserved_hbm_tokens

    def shared_bytes(self) -> CacheFootprint:
        if self._measure_shared is None:
            return self.shared
        actual = CacheFootprint.from_mapping(self._measure_shared())
        if not actual.fits(self.shared):
            raise RuntimeError(
                f"shared cache allocation exceeded reservation: actual={actual}, "
                f"reserved={self.shared}"
            )
        return actual

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
        *,
        host_pages: int = 0,
        hbm_tokens: int = 0,
    ) -> PrefixLease:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(host_pages) is not int or host_pages < 0:
            raise ValueError("host_pages must be a nonnegative integer")
        if type(hbm_tokens) is not int or hbm_tokens < 0:
            raise ValueError("hbm_tokens must be a nonnegative integer")
        if self.budget is None and host_pages == hbm_tokens == 0:
            raise ValueError(
                "fixed-pools admission requires a positive session host page count or HBM token count"
            )
        if (host_pages or hbm_tokens) and reservation == CacheFootprint():
            raise ValueError("fixed-pool sessions require a nonzero per-session byte reservation")
        # An impossible request must not destroy any existing reusable state.
        if self.budget is not None and not (self.shared + reservation).fits(self.budget):
            raise CacheBudgetExceeded(
                f"session requires {reservation} plus shared {self.shared}, "
                f"exceeds cache budget {self.budget}"
            )
        if host_pages > self.host_page_capacity:
            raise CacheBudgetExceeded(
                f"session requires {host_pages} host pages, exceeds arena capacity "
                f"{self.host_page_capacity}"
            )
        if hbm_tokens > self.hbm_token_capacity:
            raise CacheBudgetExceeded(
                f"session requires {hbm_tokens} HBM tokens, exceeds pool capacity "
                f"{self.hbm_token_capacity}"
            )
        prior = self._entries.get(key)
        if (
            prior is not None
            and prior.ready
            and prior.signature == signature
            and prior.capacity >= capacity
            and reservation.fits(prior.reservation)
            and prior.host_pages >= host_pages
            and prior.hbm_tokens >= hbm_tokens
        ):
            self._entries.move_to_end(key)
            return PrefixLease(prior, True, ())
        evicted = []
        if prior is not None:
            self.discard(key)
            evicted.append(key)
        while (
            self.budget is not None
            and not (self.reserved + reservation).fits(self.budget)
            or self.reserved_host_pages + host_pages > self.host_page_capacity
            or self.reserved_hbm_tokens + hbm_tokens > self.hbm_token_capacity
        ):
            victim = next(iter(self._entries))
            self.discard(victim)
            evicted.append(victim)
        self.evictions += len(evicted)
        session = self._allocate(capacity)
        entry = PrefixEntry(
            session, signature, capacity, reservation, host_pages, hbm_tokens=hbm_tokens
        )
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
        """Check allocation reservations, storage quotas, and configured byte caps."""
        actual = self.shared_bytes()
        actual_host_pages = 0
        actual_hbm_tokens = 0
        for key, entry in self._entries.items():
            measured = CacheFootprint.from_mapping(self._measure(entry.session))
            if not measured.fits(entry.reservation):
                raise RuntimeError(
                    f"cache allocation for {key!r} exceeded reservation: "
                    f"actual={measured}, reserved={entry.reservation}"
                )
            actual += measured
            pages = (
                entry.host_pages
                if self._measure_host_pages is None
                else self._measure_host_pages(entry.session)
            )
            if type(pages) is not int or pages < 0 or pages > entry.host_pages:
                raise RuntimeError(
                    f"cache host pages for {key!r} exceeded reservation or are invalid: "
                    f"actual={pages}, reserved={entry.host_pages}"
                )
            actual_host_pages += pages
            tokens = (
                entry.hbm_tokens
                if self._measure_hbm_tokens is None
                else self._measure_hbm_tokens(entry.session)
            )
            if type(tokens) is not int or tokens < 0 or tokens > entry.hbm_tokens:
                raise RuntimeError(
                    f"cache HBM tokens for {key!r} exceeded reservation or are invalid: "
                    f"actual={tokens}, reserved={entry.hbm_tokens}"
                )
            actual_hbm_tokens += tokens
        if self.budget is not None and (
            not actual.fits(self.budget) or not self.reserved.fits(self.budget)
        ):
            raise RuntimeError("global cache budget invariant violated")
        if (
            actual_host_pages > self.host_page_capacity
            or self.reserved_host_pages > self.host_page_capacity
        ):
            raise RuntimeError("global host page quota invariant violated")
        if (
            actual_hbm_tokens > self.hbm_token_capacity
            or self.reserved_hbm_tokens > self.hbm_token_capacity
        ):
            raise RuntimeError("global HBM token quota invariant violated")
        return actual

    def close(self) -> None:
        for key in tuple(self._entries):
            self.discard(key)
