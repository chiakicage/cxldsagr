"""State-based exact recall fixtures; setup and numerical oracles are untimed."""

import importlib.util
import sys
from pathlib import Path
from types import MethodType

import torch

from cache.sparse_token_pool import MISSING, SharedSparseTokenPool
from operators.deepseek_v32.indexer import cache_ops

STATES = ("certified_resident", "restored_all_hit", "cold_sparse_miss")
FAILED_OWNERS = []


def validate_allocation(before, after, selected):
    """Permit slot ties while enforcing protected selection and free-first FIFO."""
    previous = before["host_to_device"][selected]
    protected = previous[previous != MISSING].long()
    if not torch.equal(after["host_to_device"][selected[previous != MISSING]], protected.int()):
        raise AssertionError("recall moved or evicted an already resident selected record")
    missing = selected[previous == MISSING]
    if not len(missing):
        return {"misses": 0, "allocation_ties": False, "priority_threshold": None}
    chosen = after["host_to_device"][missing].long()
    if bool(((chosen < 1) | (chosen >= len(before["device_to_host"]))).any()):
        raise AssertionError("a missing selected record was not allocated")
    eligible = torch.ones(len(before["device_to_host"]), dtype=torch.bool)
    eligible[0] = False
    eligible[protected] = False
    if not bool(eligible[chosen].all()) or len(chosen.unique()) != len(chosen):
        raise AssertionError("allocation consumed protected or duplicate slots")
    ranks = torch.where(before["device_to_host"] == MISSING, -1, before["priority"])
    ordered = ranks[eligible].sort().values
    if len(ordered) < len(chosen):
        raise AssertionError("recall exceeds the unprotected slot capacity")
    torch.testing.assert_close(ranks[chosen].sort().values, ordered[: len(chosen)], rtol=0, atol=0)
    threshold = int(ordered[len(chosen) - 1])
    selected_ranks = ordered[ordered <= threshold]
    tied = len(selected_ranks.unique()) != len(selected_ranks)
    return {"misses": len(chosen), "allocation_ties": tied, "priority_threshold": threshold}


def load_candidate(directory):
    """Load isolated providers without replacing any production module or class."""
    if directory is None:
        return None
    directory = Path(directory).resolve(strict=True)
    modules = []
    for stem in ("cache_ops", "sparse_token_cache"):
        name = f"recall_candidate_{stem}"
        spec = importlib.util.spec_from_file_location(name, directory / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def validate_selection(capture, config):
    ids = capture["indices"]
    if ids.dtype != torch.int32 or ids.shape != (config.append, 2048) or not ids.is_contiguous():
        raise ValueError("recall requires captured contiguous int32 exact top-2048 IDs")
    ends = torch.arange(config.history + 1, config.capacity + 1)[:, None]
    if bool(((ids < -1) | (ids >= ends)).any()):
        raise ValueError("captured selection contains noncausal or invalid IDs")
    if not torch.equal((ids >= 0).sum(1), ends.flatten().clamp_max(2048)):
        raise ValueError("captured exact selection cardinality changed")
    ordered = ids.sort(1).values
    if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
        raise ValueError("captured query selection contains duplicate IDs")


class RecallFixture:
    """One actual layer and one state, with an independent production pool."""

    def __init__(self, capture, config, state, *, candidate=None, device="cuda:0"):
        self.pool = None
        self.cache = None
        self.snapshot = None
        self.closed = False
        if state not in STATES:
            raise ValueError("unknown recall state")
        self.state, self.config = state, config
        self.layer = capture["layer"]
        self.device = torch.device(device)
        try:
            self.records = capture["kv"].to(self.device)
            self.indices = capture["indices"].to(self.device)
            self.logical = torch.unique(capture["indices"])
            self.logical = self.logical[self.logical >= 0]
            self.expected_misses = (
                int((self.logical < config.history).sum()) if state == "cold_sparse_miss" else 0
            )
            self.pool = SharedSparseTokenPool(
                config.slots,
                576,
                1,
                config.slots,
                device=self.device,
                dtype=torch.bfloat16,
                metadata_ops=cache_ops if candidate is None else candidate[0],
            )
            self.session = self.pool.allocate_session(config.capacity)
            self.cache = self.session.layer(0)
            if candidate is not None:
                self.cache._ensure_sparse_from_topk = MethodType(
                    candidate[1].SparseTokenCache._ensure_sparse_from_topk, self.cache
                )
            self._append_prefix()
            if state != "certified_resident":
                self.snapshot = self.pool.snapshot()
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 -- preserve original and cleanup.
                raise BaseExceptionGroup(
                    "recall fixture construction and cleanup failed", [original, cleanup]
                ) from None
            raise

    def _append_prefix(self):
        self.cache.begin_step(self.config.history)
        for start in range(0, self.config.history, self.config.chunk):
            self.cache.append(self.records[start : start + self.config.chunk])
        self.cache.commit()

    def reset(self):
        """Build the exact pre-recall boundary without borrowing allocator snapshots."""
        if self.cache._step_end is not None:
            raise RuntimeError("finish the preceding sample before resetting")
        if self.state == "certified_resident":
            self.cache.truncate(0)
            self._append_prefix()
        else:
            self.pool.restore(self.snapshot)
        if self.state == "cold_sparse_miss":
            with self.cache.operation():
                self.pool.release_ids(0, self.session.global_ids())
        self.cache.begin_step(self.config.append)
        self.cache.declare_indexer_visible(self.config.capacity)
        self.cache.append(self.records[self.config.history :])
        self.cache.drain()
        self.cache.reset_stats()
        torch.cuda.synchronize(self.device)
        if self.cache.all_history_resident != (self.state == "certified_resident"):
            raise RuntimeError("reset produced the wrong CPU residency proof")

    def invoke(self, *, reference=False):
        return (
            self.cache.ensure(self.indices)
            if reference
            else self.cache._ensure_from_topk(self.indices)
        )

    def finish(self):
        """Commit after successful completion, outside the recall timer."""
        self.cache.commit()

    def before_recall(self):
        layer = self.pool.layers[0]
        return {
            name: getattr(layer, name).cpu().clone()
            for name in (
                "host_to_device",
                "device_to_host",
                "priority",
            )
        }

    def inspect(self, physical, before):
        """Compare all consumed KV and complete live maps against logical records."""
        ids, slots = self.indices.cpu(), physical.cpu()
        valid = ids >= 0
        if not torch.equal(slots[~valid], torch.full_like(slots[~valid], -1)):
            raise AssertionError("invalid selection padding changed")
        if bool(((slots[valid] < 1) | (slots[valid] > self.config.slots)).any()):
            raise AssertionError("selected physical slot is outside the pool")
        logical, inverse = torch.unique(ids[valid], sorted=True, return_inverse=True)
        unique_slots = torch.empty(len(logical), dtype=torch.int64)
        unique_slots[inverse] = slots[valid].long()
        if not torch.equal(slots[valid].long(), unique_slots[inverse]):
            raise AssertionError("duplicate logical IDs mapped to different physical slots")
        actual = self.cache.records[unique_slots.to(self.device)].cpu()
        expected = self.records[logical.to(self.device).long()].cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        layer = self.pool.layers[0]
        state = {
            name: getattr(layer, name).cpu().clone()
            for name in (
                "host_to_device",
                "device_to_host",
                "priority",
                "free",
                "clock_tensor",
            )
        }
        h2d, d2h = state["host_to_device"], state["device_to_host"]
        live = torch.where(d2h != MISSING)[0]
        if not torch.equal(h2d[d2h[live]].long(), live):
            raise AssertionError("published cache maps are not inverse")
        mapped_hosts = torch.where(h2d != MISSING)[0]
        mapped_slots = h2d[mapped_hosts].long()
        if bool(((mapped_slots < 1) | (mapped_slots >= len(d2h))).any()) or not torch.equal(
            d2h[mapped_slots], mapped_hosts
        ):
            raise AssertionError("host map contains stale, duplicate, or out-of-range slots")
        if not torch.equal(state["free"][1:], d2h[1:] == MISSING):
            raise AssertionError("free bitmap differs from published maps")
        if bool(state["free"][0]) or state["priority"][0] != MISSING:
            raise AssertionError("padding sentinel metadata changed")
        if int(state["clock_tensor"]) != layer.clock:
            raise AssertionError("CPU and GPU FIFO clocks differ")
        global_ids = self.cache.logical_to_global(
            torch.arange(self.config.capacity, device=self.device)
        ).cpu()
        allocation = validate_allocation(before, state, global_ids[self.logical])
        torch.testing.assert_close(slots[valid], h2d[global_ids[ids[valid].long()]], rtol=0, atol=0)
        mapped = h2d[global_ids]
        resident = mapped != MISSING
        state["logical_priority"] = state["priority"][mapped[resident].long()]
        state["live_records"] = self.cache.records[mapped[resident].to(self.device).long()].cpu()
        torch.testing.assert_close(
            state["live_records"], self.records[resident.to(self.device)].cpu(), rtol=0, atol=0
        )
        metrics = self.cache.metrics()
        expected_counts = {
            "recalled_records": self.expected_misses,
            "selection_records": len(self.logical),
            "resident_selection_records": len(self.logical) - self.expected_misses,
            "max_working_set": len(self.logical),
            "host_to_device_bytes": self.expected_misses * 1152,
            "device_to_host_bytes": 0,
            "evicted_records": 0,
        }
        for key, expected in expected_counts.items():
            if metrics[key] != expected:
                raise AssertionError(f"{key}: {metrics[key]} != {expected}")
        state.update(
            physical=slots,
            clock=layer.clock,
            metrics=metrics,
            allocation=allocation,
            resident_logical=resident,
        )
        return state

    def close(self):
        if self.closed:
            return
        try:
            torch.cuda.synchronize(self.device)
            if self.cache is not None and self.cache._step_end is not None:
                self.cache.rollback()
            if self.pool is not None:
                self.pool.close()
        except BaseException:
            FAILED_OWNERS.append(self)
            raise
        self.cache = self.pool = self.snapshot = None
        self.records = self.indices = None
        self.closed = True


def compare_states(actual, reference):
    """Compare semantic state; only tied allocation may choose different slots."""
    if actual.keys() != reference.keys():
        raise AssertionError("state evidence fields differ")
    for key, value in actual.items():
        if actual["allocation"]["allocation_ties"] and key in {
            "physical",
            "host_to_device",
            "device_to_host",
            "priority",
            "free",
        }:
            continue
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference[key], rtol=0, atol=0, msg=key)
        elif value != reference[key]:
            raise AssertionError(f"reference {key} differs")
