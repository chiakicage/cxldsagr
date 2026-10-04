"""Serial GR service with bounded cross-request prefix reuse and real timings."""

from __future__ import annotations

import hashlib
import struct
import sys
import time
from array import array
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import repeat
from operator import is_
from typing import Any

import torch

from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint, PrefixSessionPool
from executor.serving_backend import ServingBackend, SharedCachePlan
from serving import token_validation


def token_digest(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()


def _packed_prefix_digest(packed: array, prefix: int) -> str:
    """Keep prefix identity little-endian while tensor storage stays native."""
    if sys.byteorder == "little":
        return hashlib.sha256(memoryview(packed)[:prefix]).hexdigest()
    history = packed[:prefix]
    history.byteswap()
    return hashlib.sha256(history).hexdigest()


def _prepare_token_input(ids: list[int], prefix: int) -> tuple[tuple[int, str], torch.Tensor]:
    """Encode one private token buffer for both hashing and the CPU tensor.

    The tensor retains its writable array owner. Neither aliases the caller's
    list; the existing device transfer still happens before cache admission.
    """
    try:
        packed = array("q", ids)
    except OverflowError:
        packed = None
    if packed is None or packed.itemsize != 8:
        # Preserve the original errors outside the array exception context:
        # prefix struct.pack first, then candidate overflow in torch.tensor.
        # The same fallback also supports an unusual native long-long ABI.
        return (prefix, token_digest(ids[:prefix])), torch.tensor(ids, dtype=torch.long)
    signature = (prefix, _packed_prefix_digest(packed, prefix))
    return signature, torch.frombuffer(packed, dtype=torch.long)


@dataclass
class ServingResult:
    metrics: dict[str, Any]
    hidden: torch.Tensor


class PersistentGRRunner:
    """One active request; preserve only the stable history between requests.

    ``latency_ms`` includes input validation/transfer, admission/eviction,
    prefix construction on a miss, all candidate hidden states, and rollback to
    the stable prefix. Trace generation, model loading, and warmup are external.
    ``visit_index`` counts prior user requests even if their caches were evicted.
    Omitting both byte budgets selects fixed-pool admission: the shared backend
    must provide a finite host-page or retained-HBM-token quota. Byte
    reservations are still audited;
    successful execution, rather than this mode, establishes machine capacity.
    """

    def __init__(
        self,
        backend: ServingBackend,
        *,
        hbm_budget_bytes: int | None = None,
        dram_budget_bytes: int | None = None,
        resource_limits: Mapping[str, int] | None = None,
        native_token_validation: bool = False,
    ):
        self.backend = backend
        self.pool = None
        self.closed = True
        self._owner_bound = False
        self._close_complete = False
        self._rollback_on_close = False
        if (hbm_budget_bytes is None) != (dram_budget_bytes is None):
            raise ValueError("provide both cache byte budgets or omit both for fixed pools")
        budget = (
            None
            if hbm_budget_bytes is None
            else CacheFootprint(hbm_budget_bytes, dram_budget_bytes)
        )
        self.resource_mode = "fixed_pools" if budget is None else "budget"
        limits = {
            "max_seq_len": backend.max_seq_len,
            "max_session_capacity": backend.max_seq_len,
            **(resource_limits or {}),
        }
        if any(type(value) is not int or value <= 0 for value in limits.values()):
            raise ValueError("resource limits must be positive integers")
        self.resource_limits = limits
        planner = getattr(backend, "plan_resources", None)
        if budget is None and planner is None:
            raise ValueError(
                "fixed-pools admission requires a shared backend with host pages or HBM tokens"
            )
        self.resource_plan = SharedCachePlan() if planner is None else planner(budget, limits)
        if not isinstance(self.resource_plan, SharedCachePlan):
            raise TypeError("plan_resources must return a SharedCachePlan")
        self._retained_capacity = (
            (lambda capacity, prefix: prefix)
            if self.resource_plan.metadata.get("candidate_persistence") == "gpu_transient"
            else getattr(backend, "retained_session_capacity", lambda capacity, prefix: capacity)
        )
        self._estimate_hbm_tokens = lambda capacity: 0
        measure_hbm_tokens = None
        if self.resource_plan.hbm_tokens:
            for name in ("estimate_session_hbm_tokens", "session_hbm_tokens"):
                if not callable(getattr(backend, name, None)):
                    raise TypeError(f"HBM token quota backend must implement {name}")
            self._estimate_hbm_tokens = backend.estimate_session_hbm_tokens
            measure_hbm_tokens = backend.session_hbm_tokens
        if budget is not None and not self.resource_plan.shared.fits(budget):
            raise CacheBudgetExceeded(
                f"shared cache requires {self.resource_plan.shared}, exceeds cache budget {budget}"
            )
        if planner is not None:
            # Validate the complete optional extension before allocating anything.
            for name in (
                "allocate_shared",
                "bind_owner",
                "unbind_owner",
                "close",
                "shared_bytes",
                "estimate_session_host_pages",
                "session_host_pages",
            ):
                if not callable(getattr(backend, name, None)):
                    raise TypeError(f"shared-resource backend must implement {name}")
        self._estimate_host_pages = (
            backend.estimate_session_host_pages
            if self.resource_plan.host_pages
            else (lambda capacity: 0)
        )
        if budget is None:
            if self.resource_plan.host_pages == self.resource_plan.hbm_tokens == 0:
                raise ValueError(
                    "fixed-pools admission requires a shared backend with host pages or HBM tokens"
                )
            retained_limit = self._retained_capacity(
                limits["max_session_capacity"],
                limits.get("max_history_tokens", limits["max_session_capacity"]),
            )
            maximum_pages = self._estimate_host_pages(retained_limit)
            maximum_tokens = self._estimate_hbm_tokens(retained_limit)
            if type(maximum_pages) is not int or maximum_pages < 0:
                raise ValueError("session host pages must be a nonnegative integer")
            if type(maximum_tokens) is not int or maximum_tokens < 0:
                raise ValueError("session HBM tokens must be a nonnegative integer")
            if maximum_pages == maximum_tokens == 0:
                raise ValueError(
                    "fixed-pools admission requires positive session host pages or HBM tokens"
                )
            if maximum_pages > self.resource_plan.host_pages:
                raise CacheBudgetExceeded("fixed host arena cannot hold one maximum-size session")
            if maximum_tokens > self.resource_plan.hbm_tokens:
                raise CacheBudgetExceeded("fixed HBM pool cannot hold one maximum-size session")
        self.token_validation_identity = {
            "backend": "python_reference",
            "fallback_reason": None,
            "requested_native": native_token_validation,
        }
        if native_token_validation:
            self._token_ids_valid = token_validation.prepare()
            self.token_validation_identity.update(token_validation.runtime_info())
            if self.token_validation_identity["backend"] == "cpython_native":
                self._validate = self._validate_native
        try:
            if planner is not None:
                backend.bind_owner(self)
                self._owner_bound = True
                backend.allocate_shared(self.resource_plan)
            self.pool = PrefixSessionPool(
                budget,
                allocate=backend.create_session,
                release=backend.release_session,
                measure=backend.session_bytes,
                shared=self.resource_plan.shared,
                measure_shared=None if planner is None else backend.shared_bytes,
                host_page_capacity=self.resource_plan.host_pages,
                measure_host_pages=(
                    backend.session_host_pages if self.resource_plan.host_pages else None
                ),
                hbm_token_capacity=self.resource_plan.hbm_tokens,
                measure_hbm_tokens=measure_hbm_tokens,
            )
            self.pool.audit()
        except BaseException as initialization_error:
            try:
                self._close(rollback=True)
            except BaseException as cleanup_error:
                error = RuntimeError("Runner initialization cleanup failed; ownership retained")
                error.add_note(f"Initialization error: {initialization_error!r}")
                raise error from cleanup_error
            raise
        self.visits: Counter = Counter()
        self.closed = False

    def _validate(self, request: Mapping) -> tuple[Any, list[int], int]:
        uid = request["user_id"]
        if not isinstance(uid, (str, int)) or isinstance(uid, bool):
            raise TypeError("user_id must be an integer or string")
        ids = request["input_ids"]
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("input_ids must be a nonempty integer list")
        ids = list(ids)
        if not ids or not all(map(is_, map(type, ids), repeat(int))) or min(ids) < 0:
            raise ValueError("input_ids must contain nonnegative integers")
        prefix = request["stable_prefix_tokens"]
        if type(prefix) is not int or not 0 < prefix < len(ids):
            raise ValueError("stable_prefix_tokens must leave a nonempty suffix")
        if len(ids) > self.backend.max_seq_len:
            raise ValueError("request exceeds backend context limit")
        if len(ids) > self.resource_limits["max_session_capacity"]:
            raise ValueError("request exceeds planned session capacity")
        for source in (self.resource_limits, self.resource_plan.metadata):
            history_limit = source.get("max_history_tokens")
            if history_limit is not None and prefix > history_limit:
                raise ValueError("history exceeds planned retained capacity")
            candidate_limit = source.get("max_candidate_tokens")
            if candidate_limit is not None and len(ids) - prefix > candidate_limit:
                raise ValueError("candidate suffix exceeds planned execution capacity")
        return uid, ids, prefix

    def _validate_native(self, request: Mapping) -> tuple[Any, list[int], int]:
        # Keep this wrapper aligned with _validate; only its token predicate differs.
        uid = request["user_id"]
        if not isinstance(uid, (str, int)) or isinstance(uid, bool):
            raise TypeError("user_id must be an integer or string")
        ids = request["input_ids"]
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("input_ids must be a nonempty integer list")
        ids = list(ids)
        if not self._token_ids_valid(ids):
            raise ValueError("input_ids must contain nonnegative integers")
        prefix = request["stable_prefix_tokens"]
        if type(prefix) is not int or not 0 < prefix < len(ids):
            raise ValueError("stable_prefix_tokens must leave a nonempty suffix")
        if len(ids) > self.backend.max_seq_len:
            raise ValueError("request exceeds backend context limit")
        if len(ids) > self.resource_limits["max_session_capacity"]:
            raise ValueError("request exceeds planned session capacity")
        for source in (self.resource_limits, self.resource_plan.metadata):
            history_limit = source.get("max_history_tokens")
            if history_limit is not None and prefix > history_limit:
                raise ValueError("history exceeds planned retained capacity")
            candidate_limit = source.get("max_candidate_tokens")
            if candidate_limit is not None and len(ids) - prefix > candidate_limit:
                raise ValueError("candidate suffix exceeds planned execution capacity")
        return uid, ids, prefix

    def execute(self, request: Mapping) -> ServingResult:
        if self.closed:
            raise RuntimeError("runner has been closed")
        self.backend.synchronize()
        begin = time.perf_counter()
        uid, raw_ids, prefix = self._validate(request)
        signature, input_cpu = _prepare_token_input(raw_ids, prefix)
        ids = input_cpu.to(device=self.backend.device)
        del input_cpu
        retained_capacity = self._retained_capacity(len(raw_ids), prefix)
        reservation = CacheFootprint.from_mapping(
            self.backend.estimate_session_bytes(retained_capacity, prefix)
        )
        lease = self.pool.acquire(
            uid,
            signature,
            retained_capacity,
            reservation,
            host_pages=self._estimate_host_pages(retained_capacity),
            hbm_tokens=self._estimate_hbm_tokens(retained_capacity),
        )
        session = lease.entry.session
        try:
            self.backend.synchronize()
            admitted = time.perf_counter()
            if not lease.hit:
                self.backend.prefill(session, ids[:prefix])
                self.backend.synchronize()
                self.pool.mark_ready(uid)
            prefixed = time.perf_counter()
            # Capture cache allocation while pending suffix buffers may exist.
            with torch.inference_mode():
                execute_candidate = getattr(self.backend, "extend_candidate", self.backend.extend)
                hidden = execute_candidate(session, ids[prefix:])
            self.backend.synchronize()
            extended = time.perf_counter()
            if hidden.ndim != 2 or hidden.shape[0] != len(raw_ids) - prefix:
                raise RuntimeError("backend must return every candidate token's hidden state")
            before_cleanup = self.pool.audit()
            self.backend.truncate(session, prefix)
            self.backend.synchronize()
            actual = self.pool.audit()
            finished = time.perf_counter()
        except BaseException:
            self.pool.discard(uid)
            raise
        # Diagnostics may synchronize small device counters. Keep that intrusive
        # observation outside the request latency interval. Backends retain
        # bounded counters through truncate and reset them at the next request.
        diagnostics = getattr(self.backend, "session_metrics", None)
        cache_diagnostics = {} if diagnostics is None else diagnostics(session)
        visit_index = self.visits[uid]
        self.visits[uid] += 1
        metrics = {
            key: value
            for key, value in request.items()
            if key not in ("input_ids", "attention_mask", "prompt")
        }
        shared_actual = self.pool.shared_bytes()
        metrics.update(
            scheme=self.backend.scheme,
            user_id=uid,
            visit_index=visit_index,
            visit_number=visit_index + 1,
            is_revisit=visit_index > 0,
            prefix_hit_tier=(
                ("hbm" if self.backend.scheme == "hbm" else "dram") if lease.hit else "miss"
            ),
            prefix_cache_hit=lease.hit,
            evicted_users=list(lease.evicted),
            latency_ms=(finished - begin) * 1000,
            admission_ms=(admitted - begin) * 1000,
            prefix_ms=(prefixed - admitted) * 1000,
            extend_ms=(extended - prefixed) * 1000,
            cleanup_ms=(finished - extended) * 1000,
            stable_prefix_tokens=prefix,
            candidate_suffix_tokens=len(raw_ids) - prefix,
            resource_mode=self.resource_mode,
            hbm_budget_bytes=None if self.pool.budget is None else self.pool.budget.hbm,
            dram_budget_bytes=None if self.pool.budget is None else self.pool.budget.dram,
            cache_hbm_bytes=actual.hbm,
            cache_dram_bytes=actual.dram,
            request_cache_hbm_bytes=before_cleanup.hbm,
            request_cache_dram_bytes=before_cleanup.dram,
            reserved_hbm_bytes=self.pool.reserved.hbm,
            reserved_dram_bytes=self.pool.reserved.dram,
            shared_cache_hbm_bytes=shared_actual.hbm,
            shared_cache_dram_bytes=shared_actual.dram,
            shared_reserved_hbm_bytes=self.pool.shared.hbm,
            shared_reserved_dram_bytes=self.pool.shared.dram,
            session_reserved_hbm_bytes=self.pool.session_reserved.hbm,
            session_reserved_dram_bytes=self.pool.session_reserved.dram,
            cache_host_pages=self.pool.reserved_host_pages,
            host_page_capacity=self.pool.host_page_capacity,
            session_host_pages=lease.entry.host_pages,
            cache_hbm_tokens=self.pool.reserved_hbm_tokens,
            hbm_token_capacity=self.pool.hbm_token_capacity,
            session_hbm_tokens=lease.entry.hbm_tokens,
            host_page_tokens=self.resource_plan.page_size,
            cache_pool_scope=self.resource_plan.metadata.get("pool_scope", "session"),
            cache_diagnostics=cache_diagnostics,
            cached_users=len(self.pool),
        )
        return ServingResult(metrics, hidden)

    def run(self, requests: Iterable[Mapping]) -> Iterator[ServingResult]:
        for request in requests:
            yield self.execute(request)

    def _close(self, *, rollback=False):
        if self._close_complete:
            return
        # A partially closed pool cannot execute again. Keep its owner bound if
        # release/drain fails, allowing close to be retried without re-admission.
        self.closed = True
        self._rollback_on_close |= rollback
        if self.pool is not None:
            self.pool.close()
        if self._owner_bound:
            self.backend.unbind_owner(self, rollback=self._rollback_on_close)
            self._owner_bound = False
        self._close_complete = True

    def close(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
