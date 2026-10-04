"""Strict, test-only CUDA lifetime join for the serial NOSA checkpoint audit.

Kineto and CUDA history use different clock conversions. Do not use a time
tolerance to identify an allocation. Instead require a bijection of *all*
allocation/free-request callbacks, including ordinary activations, in their
single-thread encounter order. Ownership still comes from the generic parser;
active lifetimes come only from CUDA history, through free_completed.

This adapter leaves the shared experiment parser and its public behavior intact.
It is deliberately limited to the controlled, native-allocator test process.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import pairwise
from unittest.mock import patch

from evaluation import cache_memory_audit as generic


def _require(condition, message):
    if not condition:
        raise ValueError(f"NOSA CUDA generation evidence: {message}")


def _callbacks(trace):
    rows = []
    for event in trace["traceEvents"]:
        if event.get("name") != "[memory]":
            continue
        args = event["args"]
        rows.append(
            (
                args.get("Ev Idx"),
                event.get("pid"),
                event.get("tid"),
                event["ts"],
                args["Device Type"],
                args["Device Id"],
                args["Addr"],
                args["Bytes"],
            )
        )
    return rows


def _native_block_fits(requested, block):
    # Independently encode the audited default native allocator's rounding and
    # maximum unsplit tail. The production estimator is not an evidence oracle.
    rounded = (requested + 511) // 512 * 512
    upper = rounded + ((1 << 20) if rounded > (1 << 20) else 0)
    return rounded <= block <= upper and block % 512 == 0


def _history_digest(history):
    return hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()


class CudaGenerationJoin:
    def __init__(self, trace, history, *, baseline):
        _require(history.get("possibly_truncated") is False, "history may be truncated")
        cap = history.get("max_entries_per_device")
        _require(type(cap) is int and cap > 0, "missing history capacity")
        _require("baseTimeNanoseconds" in trace, "missing profiler base timestamp")
        self.base_ns = trace["baseTimeNanoseconds"]
        _require(type(self.base_ns) is int and self.base_ns >= 0, "invalid trace origin")
        self.history = history
        self.history_sha256 = _history_digest(history)
        self.callbacks = _callbacks(trace)
        _require(bool(self.callbacks), "no allocator callbacks")
        previous_index = -1
        by_device = defaultdict(list)
        for row in self.callbacks:
            index, pid, tid, timestamp, kind, ordinal, address, size = row
            _require(type(index) is int and index > previous_index, "callback Ev Idx order")
            previous_index = index
            _require(type(size) is int and size != 0, "invalid callback bytes")
            _require(math.isfinite(timestamp), "invalid callback timestamp")
            _require(kind in (0, 1), "unsupported callback device")
            if kind == 1:
                _require(type(ordinal) is int and ordinal >= 0, "invalid CUDA device")
                _require(type(address) is int and address > 0, "invalid callback address")
                _require(pid is not None and tid is not None, "missing callback thread")
                by_device[f"cuda:{ordinal}"].append(row)
        _require(set(by_device) == set(history["events_by_device"]), "CUDA device sets disagree")
        threads = {(row[1], row[2]) for rows in by_device.values() for row in rows}
        _require(len(threads) == 1, "CUDA callbacks require one CPU thread")
        fixed = {}
        for item in baseline["storages"]:
            if not item["device"].startswith("cuda:"):
                continue
            key = (item["device"], item["address"])
            storage, block = item["storage_bytes"], item["allocator_bytes"]
            _require(key not in fixed, "duplicate baseline owner")
            _require(
                type(item["address"]) is int and item["address"] > 0,
                "invalid baseline address",
            )
            _require(
                type(storage) is int
                and storage > 0
                and type(block) is int
                and _native_block_fits(storage, block),
                "invalid baseline native capacity",
            )
            fixed[key] = item
        self.generations = {}
        self.summary = {
            "method": "all-callback-stream-bijection-v1",
            "callback_thread": list(next(iter(threads))),
            "clock_tolerance_used": False,
            "active_lifetime_clock": "CUDA allocator history only",
            "history_sha256": self.history_sha256,
            "devices": {},
        }
        for device, callbacks in by_device.items():
            events = history["events_by_device"][device]
            _require(len(events) < cap, "history reached recording capacity")
            _require(
                all(a[3] <= b[3] for a, b in pairwise(callbacks)),
                "callback timestamps reverse encounter order",
            )
            live, seen, baseline_freed = {}, set(), set()
            callback_cursor, previous_time = 0, -math.inf
            counts = Counter()
            for event in events:
                action = event["action"]
                timestamp = event["time_us"]
                _require(
                    type(timestamp) is int and timestamp >= previous_time,
                    "history timestamps reverse encounter order",
                )
                previous_time = timestamp
                _require(event.get("pool_id") == [0, 0], "nondefault history pool")
                _require(
                    action
                    in (
                        "alloc",
                        "free_requested",
                        "free_completed",
                        "segment_alloc",
                        "segment_free",
                        "snapshot",
                    ),
                    f"unsupported history action {action}",
                )
                if action not in ("alloc", "free_requested", "free_completed"):
                    continue
                address, requested = event["addr"], event["size"]
                _require(type(address) is int and address > 0, "invalid history address")
                _require(type(requested) is int and requested > 0, "invalid history bytes")
                counts[action] += 1
                callback = None
                if action != "free_completed":
                    _require(callback_cursor < len(callbacks), "history has extra callbacks")
                    callback = callbacks[callback_cursor]
                    callback_cursor += 1
                    expected_action = "alloc" if callback[7] > 0 else "free_requested"
                    _require(
                        (action, address) == (expected_action, callback[6]),
                        f"full callback stream mismatch at {device}/{callback_cursor - 1}",
                    )
                if action == "alloc":
                    _require(address not in live, "address reused before completed free")
                    _require(
                        (device, address) not in fixed or address in baseline_freed,
                        "allocation overlaps an unreleased baseline block",
                    )
                    _require(
                        _native_block_fits(requested, callback[7]),
                        "profiler block is outside the native allocation bound",
                    )
                    key = (device, address, callback[3])
                    _require(key not in self.generations, "ambiguous profiler allocation time")
                    generation = {
                        "requested_bytes": requested,
                        "bytes": callback[7],
                        "start_us": callback[3],
                        "end_us": None,
                        "active_start_us": timestamp,
                        "active_end_us": None,
                        "free_requested": False,
                    }
                    live[address] = generation
                    seen.add(address)
                    self.generations[key] = generation
                else:
                    if address not in live:
                        # The audit retains its fixed baseline conservatively.
                        # A released old indexer slab must nevertheless match
                        # an independently observed pre-capture allocation.
                        _require(action == "free_requested", "completion without request")
                        _require(address not in seen, "duplicate free or missing allocation")
                        _require(address not in baseline_freed, "duplicate baseline free")
                        owner = fixed.get((device, address))
                        _require(owner is not None, "unproven preexisting allocation free")
                        _require(
                            requested == owner["storage_bytes"]
                            and -callback[7] == owner["allocator_bytes"],
                            "baseline free differs from independent storage inventory",
                        )
                        live[address] = {
                            "requested_bytes": requested,
                            "bytes": owner["allocator_bytes"],
                            "free_requested": False,
                        }
                        baseline_freed.add(address)
                    generation = live[address]
                    _require(requested == generation["requested_bytes"], "history free size")
                    if action == "free_requested":
                        _require(not generation["free_requested"], "duplicate free request")
                        _require(-callback[7] == generation["bytes"], "profiler free size")
                        generation["free_requested"] = True
                        generation["end_us"] = callback[3]
                    else:
                        _require(generation["free_requested"], "completion before request")
                        generation["active_end_us"] = timestamp
                        del live[address]
            _require(callback_cursor == len(callbacks), "profiler has extra callbacks")
            # CUDA synchronization need not poll the allocator's deferred-free
            # queue. Pending generations without a recorded free_completed stay
            # active through the end, conservatively, and may never be reused.
            self.summary["devices"][device] = {
                "allocations": counts["alloc"],
                "free_requests": counts["free_requested"],
                "completed_frees": counts["free_completed"],
                "baseline_frees": len(baseline_freed),
                "live_at_end": len(live),
                "pending_free_at_end": sum(item["free_requested"] for item in live.values()),
                "matched_callbacks": callback_cursor,
            }

    def _apply(self, owned, history, *, base_us, errors):
        _require(history is self.history, "history changed after validation")
        matched = set()
        for item in owned:
            if not item["device"].startswith("cuda:"):
                continue
            key = (item["device"], item["address"], item["start_us"])
            generation = self.generations.get(key)
            _require(generation is not None and key not in matched, "ownership generation join")
            _require(
                (item["bytes"], item["end_us"]) == (generation["bytes"], generation["end_us"]),
                "ownership parser disagrees with matched callback generation",
            )
            matched.add(key)
            # Subtracting the display origin preserves all history intervals.
            # It is never used to match either end against the profiler clock.
            item["active_start_us"] = generation["active_start_us"] - base_us
            end = generation["active_end_us"]
            item["active_end_us"] = None if end is None else end - base_us

    def audit(self, trace, *, limits):
        _require(trace["baseTimeNanoseconds"] == self.base_ns, "trace origin changed")
        _require(_callbacks(trace) == self.callbacks, "callbacks changed after validation")
        _require(
            _history_digest(self.history) == self.history_sha256,
            "history changed after validation",
        )
        # Scoped to this isolated test invocation; no shared file is modified.
        with patch.object(generic, "_apply_cuda_history", self._apply):
            report = generic.audit_trace(trace, limits=limits, cuda_history=self.history)
        report["coverage"]["cuda_generation_matching"] = (
            "all-callback stream bijection in single-thread encounter order"
        )
        report["cuda_generation_join"] = self.summary
        return report
