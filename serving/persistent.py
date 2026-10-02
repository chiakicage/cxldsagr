"""Serial GR service with budgeted cross-request prefix reuse and real timings."""

from __future__ import annotations

import hashlib
import struct
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from cache.prefix_pool import CacheFootprint, PrefixSessionPool
from executor.serving_backend import ServingBackend


def token_digest(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()


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
    """

    def __init__(self, backend: ServingBackend, *, hbm_budget_bytes: int, dram_budget_bytes: int):
        self.backend = backend
        self.pool = PrefixSessionPool(
            CacheFootprint(hbm_budget_bytes, dram_budget_bytes),
            allocate=backend.create_session,
            release=backend.release_session,
            measure=backend.session_bytes,
        )
        self.visits: Counter = Counter()
        self.closed = False

    def _validate(self, request: Mapping) -> tuple[Any, list[int], int]:
        uid = request["user_id"]
        if not isinstance(uid, (str, int)) or isinstance(uid, bool):
            raise TypeError("user_id must be an integer or string")
        ids = request["input_ids"]
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("input_ids must be a nonempty integer list")
        if any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("input_ids must contain nonnegative integers")
        prefix = request["stable_prefix_tokens"]
        if type(prefix) is not int or not 0 < prefix < len(ids):
            raise ValueError("stable_prefix_tokens must leave a nonempty suffix")
        if len(ids) > self.backend.max_seq_len:
            raise ValueError("request exceeds backend context limit")
        return uid, list(ids), prefix

    def execute(self, request: Mapping) -> ServingResult:
        if self.closed:
            raise RuntimeError("runner has been closed")
        self.backend.synchronize()
        begin = time.perf_counter()
        uid, raw_ids, prefix = self._validate(request)
        signature = (prefix, token_digest(raw_ids[:prefix]))
        ids = torch.tensor(raw_ids, dtype=torch.long, device=self.backend.device)
        reservation = CacheFootprint.from_mapping(
            self.backend.estimate_session_bytes(len(raw_ids), prefix)
        )
        lease = self.pool.acquire(uid, signature, len(raw_ids), reservation)
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
                hidden = self.backend.extend(session, ids[prefix:])
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
        visit_index = self.visits[uid]
        self.visits[uid] += 1
        metrics = {
            key: value
            for key, value in request.items()
            if key not in ("input_ids", "attention_mask", "prompt")
        }
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
            hbm_budget_bytes=self.pool.budget.hbm,
            dram_budget_bytes=self.pool.budget.dram,
            cache_hbm_bytes=actual.hbm,
            cache_dram_bytes=actual.dram,
            request_cache_hbm_bytes=before_cleanup.hbm,
            request_cache_dram_bytes=before_cleanup.dram,
            reserved_hbm_bytes=self.pool.reserved.hbm,
            reserved_dram_bytes=self.pool.reserved.dram,
            cached_users=len(self.pool),
        )
        return ServingResult(metrics, hidden)

    def run(self, requests: Iterable[Mapping]) -> Iterator[ServingResult]:
        for request in requests:
            yield self.execute(request)

    def close(self):
        self.pool.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
