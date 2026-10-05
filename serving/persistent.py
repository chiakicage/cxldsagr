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
from typing import Any

import torch

from cache.capacity import CapacityPolicy
from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint, PrefixSessionPool
from executor.contracts import RequestShape, ServingBackend
from executor.runtime import TokenRuntime
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
    packed = array("q", ids)
    if packed.itemsize != 8:
        raise RuntimeError("token encoding requires an eight-byte native signed long long")
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
        policy = (
            CapacityPolicy.fixed_pools() if budget is None else CapacityPolicy.byte_budget(budget)
        )
        self.runtime = TokenRuntime(backend.runtime_driver(policy))
        self.driver = self.runtime.driver
        self.resource_plan = self.runtime.plan_resources(policy, limits)
        if budget is not None and not self.resource_plan.shared.fits(budget):
            raise CacheBudgetExceeded(
                f"shared cache requires {self.resource_plan.shared}, exceeds cache budget {budget}"
            )
        if budget is None:
            if self.resource_plan.host_pages == self.resource_plan.hbm_tokens == 0:
                raise ValueError(
                    "fixed-pools admission requires a shared backend with host pages or HBM tokens"
                )
            maximum_pages, maximum_tokens = self.driver.maximum_session_quota(self.resource_plan)
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
        self._token_ids_valid = token_validation.reference
        if native_token_validation:
            self._token_ids_valid = token_validation.prepare()
            self.token_validation_identity.update(token_validation.runtime_info())
        try:
            self.driver.bind_owner(self)
            self._owner_bound = True
            self.runtime.allocate_resources(self.resource_plan)
            self.pool = PrefixSessionPool(
                budget,
                allocate=self.runtime.create_session,
                release=self.runtime.release_session,
                measure=self.runtime.session_usage,
                shared=self.resource_plan.shared,
                measure_shared=lambda: self.runtime.shared_usage().as_mapping(),
                host_page_capacity=self.resource_plan.host_pages,
                hbm_token_capacity=self.resource_plan.hbm_tokens,
            )
            self.pool.audit()
        except BaseException as initialization_error:
            try:
                self._close(rollback=True)
            except BaseException as cleanup_error:  # noqa: BLE001 - retain both original failures
                raise BaseExceptionGroup(
                    "Runner initialization cleanup failed; ownership retained",
                    [initialization_error, cleanup_error],
                ) from None
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
        if not self._token_ids_valid(ids):
            raise ValueError("input_ids must contain nonnegative integers")
        prefix = request["stable_prefix_tokens"]
        if type(prefix) is not int or not 0 < prefix < len(ids):
            raise ValueError("stable_prefix_tokens must leave a nonempty suffix")
        self.resource_plan.limits.validate(RequestShape(prefix, len(ids) - prefix))
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
        plan = self.runtime.plan_session(RequestShape(prefix, len(raw_ids) - prefix), signature)
        lease = self.pool.acquire(
            uid,
            signature,
            plan.retained_capacity,
            plan.reservation,
            host_pages=plan.host_pages,
            hbm_tokens=plan.hbm_tokens,
            plan=plan,
        )
        session = lease.entry.session
        try:
            self.backend.synchronize()
            admitted = time.perf_counter()
            if not lease.hit:
                self.runtime.prefill(session, ids[:prefix])
                self.backend.synchronize()
                self.pool.mark_ready(uid)
            prefixed = time.perf_counter()
            # The observer retains the existing extend/cleanup audit boundary.
            observations = {}

            def observe(stage):
                if stage == "executed":
                    observations["extended"] = time.perf_counter()
                    observations["before_cleanup"] = self.pool.audit()
                else:
                    observations["actual"] = self.pool.audit()
                    observations["finished"] = time.perf_counter()

            result = self.runtime.candidate(session, ids[prefix:], observer=observe)
            hidden = result.hidden
            extended = observations["extended"]
            before_cleanup = observations["before_cleanup"]
            actual = observations["actual"]
            finished = observations["finished"]
        except BaseException as execution_error:
            try:
                self.pool.discard(uid)
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve both failures
                raise BaseExceptionGroup(
                    "request execution and session cleanup failed",
                    [execution_error, cleanup_error],
                ) from None
            raise
        # Diagnostic reads remain outside the measured request interval.
        cache_diagnostics = self.driver.diagnostics(session)
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
            self.driver.unbind_owner(self, rollback=self._rollback_on_close)
            self._owner_bound = False
        self.runtime.retire()
        self._close_complete = True

    def close(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is not None:
                raise BaseExceptionGroup(
                    "runner body and close both failed", [error, cleanup_error]
                ) from None
            raise
