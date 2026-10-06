"""Private recall preserves exact KV, legal priority allocation and failure safety."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_pool import MISSING, PAGE_SIZE, PRIORITY_LIMIT, SharedSparseTokenPool
from cache.tests.test_resident_metadata import _equal, _pools
from operators.common import kv_transfer
from operators.deepseek_v32.indexer import cache_ops

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _LegacyMetadataProvider:
    """Explicit segmented provider; live forwarding preserves stage injection."""

    def __init__(self, *without):
        self.without = frozenset(("sparse_selection_complete", *without))

    def __getattr__(self, name):
        if name in self.without:
            raise AttributeError(name)
        return getattr(cache_ops, name)


def _append(cache, data, *, commit=True):
    cache.begin_step(len(data))
    for piece in data.split(cache.slots):
        cache.append(piece)
    if commit:
        cache.commit()


def _proofs_equal(pools, caches, *, exact=True):
    """GPU-count recall may conservatively discard CPU optimization proofs."""
    if exact:
        _equal(pools, caches)
    for layer_id in range(2):
        states = [pool.layers[layer_id] for pool in pools]
        assert states[0].map_generation >= states[1].map_generation
        assert states[0].lease_owner == states[1].lease_owner
        for owner, end in (("resident_owner", "resident_end"), ("append_owner", "append_cursor")):
            if getattr(states[0], owner) is not None:
                assert getattr(states[0], owner) == getattr(states[1], owner), owner
                assert getattr(states[0], end) == getattr(states[1], end), end
        if states[0].append_owner is not None:
            torch.testing.assert_close(
                states[0].append_order, states[1].append_order, rtol=0, atol=0
            )


_STATE_FIELDS = ("records", "host_to_device", "device_to_host", "priority", "free", "clock_tensor")


def _capture(cache):
    # Enqueue copies only: observing the snapshot must not drain pending host
    # writes or add a host synchronization before the stream-change test.
    logical = torch.arange(cache.host_written_end)
    return {
        "layers": [
            {**{name: getattr(layer, name).clone() for name in _STATE_FIELDS}, "clock": layer.clock}
            for layer in cache._pool.layers
        ],
        "stats": vars(cache.stats).copy(),
        "totals": cache._counter_totals.clone(),
        "globals": cache.session._pages[logical // PAGE_SIZE].long() * PAGE_SIZE
        + logical % PAGE_SIZE,
    }


def _on_cpu(snapshot):
    return {
        **snapshot,
        "layers": [
            {
                name: value.cpu() if isinstance(value, torch.Tensor) else value
                for name, value in layer.items()
            }
            for layer in snapshot["layers"]
        ],
        "totals": snapshot["totals"].cpu(),
        "globals": snapshot["globals"].cpu(),
    }


def _counts(snapshot, record_bytes):
    result = snapshot["stats"].copy()
    prefetched, fused_evictions, failed, selected, resident, maximum, evicted, recalled = snapshot[
        "totals"
    ].tolist()
    result["selection_records"] += selected
    result["resident_selection_records"] += resident
    result["max_working_set"] = max(result["max_working_set"], maximum)
    result["evicted_records"] += fused_evictions + evicted
    result["recalled_records"] += recalled
    result.update(
        prefetched_records=prefetched,
        prefetch_capacity_failures=failed,
        host_to_device_bytes=(prefetched + result["recalled_records"]) * record_bytes,
    )
    return result


def _same_state(left, right):
    return all(
        torch.equal(left[name], right[name])
        if isinstance(left[name], torch.Tensor)
        else left[name] == right[name]
        for name in left
    )


def _event(state, physical):
    if state["clock"] >= PRIORITY_LIMIT:
        live = torch.where(state["device_to_host"][1:] != MISSING)[0] + 1
        values = sorted(set(state["priority"][live].tolist()))
        ranks = {value: rank for rank, value in enumerate(values)}
        state["priority"][live] = torch.tensor(
            [ranks[int(value)] for value in state["priority"][live]]
        )
        state["clock"] = len(values)
    state["priority"][physical] = state["clock"]
    state["priority"][0] = MISSING
    state["clock"] += 1
    state["clock_tensor"].fill_(state["clock"])


def _legal_victims(state, protected, chosen):
    """A small CPU oracle independent of the native sort/radix implementation."""
    assert len(set(chosen.tolist())) == len(chosen), "duplicate allocation slot"
    eligible = set(range(1, len(state["device_to_host"]))) - set(protected.tolist())
    assert set(chosen.tolist()) <= eligible, "allocation used padding/protected/out-of-range slot"
    ranks = {
        slot: -1 if state["device_to_host"][slot] == MISSING else int(state["priority"][slot])
        for slot in eligible
    }
    ordered = sorted(ranks.values())
    assert sorted(ranks[slot] for slot in chosen.tolist()) == ordered[: len(chosen)], (
        "allocation violated free-first/minimum-priority order"
    )
    if not len(chosen):
        return False
    prefix = [rank for rank in ordered if rank <= ordered[len(chosen) - 1]]
    return len(prefix) != len(set(prefix))


def _validate_recall(cache, ids, output, before, *, stage=None, transfer=None):
    """Validate this call against its own trajectory, including partial failures."""
    cache._pool.drain()
    before, after = _on_cpu(before), _on_cpu(_capture(cache))
    layer_id = cache.layer_id
    old, actual = before["layers"][layer_id], after["layers"][layer_id]
    expected = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in old.items()
    }
    logical = torch.unique(ids.cpu().long())
    logical = logical[logical >= 0]
    assert not len(logical) or int(logical[-1]) < cache.written
    selected = before["globals"][logical[logical < cache.host_written_end]]
    previous = old["host_to_device"][selected].long()
    protected = previous[previous != MISSING]
    missing = selected[previous == MISSING]
    _event(expected, protected)
    allocated = stage not in ("wait", "sort")
    published = stage in (None, "map")
    if not allocated:
        chosen, copied = torch.empty(0, dtype=torch.int64), missing[:0]
    elif published:
        copied = missing
        chosen = actual["host_to_device"][missing].long()
    else:
        copied, chosen = (value.cpu().long()[: len(missing)] for value in transfer)
        assert torch.equal(copied.sort().values, missing.sort().values), "wrong transfer IDs"
    tied = _legal_victims(expected, protected, chosen)
    victims = old["device_to_host"][chosen]
    evicted = victims[victims != MISSING]
    expected["host_to_device"][evicted] = MISSING
    expected["device_to_host"][chosen] = MISSING
    expected["priority"][chosen] = -1
    expected["free"][chosen] = True
    expected["records"][chosen] = cache.host[copied].cpu()
    if published:
        expected["host_to_device"][missing] = chosen.int()
        expected["device_to_host"][chosen] = missing
        expected["free"][chosen] = False
        _event(expected, chosen)
    for name in _STATE_FIELDS:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
    assert actual["clock"] == expected["clock"]
    assert actual["device_to_host"][0] == actual["priority"][0] == MISSING
    assert not actual["free"][0]
    assert bool((actual["records"][0] == 0).all())
    assert torch.equal(actual["free"][1:], actual["device_to_host"][1:] == MISSING)
    assert bool((actual["priority"][actual["free"]] == -1).all())
    live = torch.where(actual["device_to_host"] != MISSING)[0]
    global_ids = actual["device_to_host"][live]
    mapped = torch.where(actual["host_to_device"] != MISSING)[0]
    assert torch.equal(mapped.sort().values, global_ids.sort().values)
    torch.testing.assert_close(actual["host_to_device"][global_ids].long(), live, rtol=0, atol=0)
    torch.testing.assert_close(
        actual["records"][live], cache.host[global_ids].cpu(), rtol=0, atol=0
    )
    assert bool((actual["priority"][live] < actual["clock"]).all())
    for index in range(len(before["layers"])):
        if index != layer_id:
            assert _same_state(before["layers"][index], after["layers"][index]), (
                "changed another layer"
            )
    counts = _counts(before, cache.record_bytes)
    counts["selection_records"] += len(logical)
    counts["resident_selection_records"] += len(logical) - len(missing)
    counts["max_working_set"] = max(counts["max_working_set"], len(logical))
    counts["evicted_records"] += len(evicted)
    counts["recalled_records"] += len(missing) if published else 0
    counts["host_to_device_bytes"] += len(missing) * cache.record_bytes if published else 0
    assert _counts(after, cache.record_bytes) == counts
    metrics = cache.metrics()
    assert {name: metrics[name] for name in counts} == counts
    if output is not None:
        ids, output = ids.cpu().long(), output.cpu()
        physical = torch.full_like(ids, -1, dtype=torch.int32)
        history = (ids >= 0) & (ids < cache.host_written_end)
        physical[history] = actual["host_to_device"][before["globals"][ids[history]]]
        candidate = ids >= cache.host_written_end
        physical[candidate] = (ids[candidate] - cache.host_written_end + cache.slots + 1).int()
        torch.testing.assert_close(output, physical, rtol=0, atol=0)
        assert bool((physical[ids >= 0] != MISSING).all())
    return before, tied


def _check_pair(pools, caches, ids, outputs, before):
    checked = [
        _validate_recall(cache, ids, output, snapshot)
        for cache, output, snapshot in zip(caches, outputs, before, strict=True)
    ]
    _proofs_equal(pools, caches, exact=False)
    if (
        not any(tied for _, tied in checked)
        and all(
            _same_state(left, right)
            for left, right in zip(checked[0][0]["layers"], checked[1][0]["layers"], strict=True)
        )
        and _counts(checked[0][0], caches[0].record_bytes)
        == _counts(checked[1][0], caches[1].record_bytes)
    ):
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        _equal(pools, caches)


def _recall_pair(pools, caches, ids):
    before = [_capture(cache) for cache in caches]
    outputs = [cache._ensure_from_topk(ids) for cache in caches]
    _check_pair(pools, caches, ids, outputs, before)
    return outputs


@pytest.mark.parametrize("chosen", [[2, 3], [2, 4]])
def test_recall_oracle_accepts_equal_priority_victims(chosen):
    state = {
        "device_to_host": torch.tensor([MISSING, 10, MISSING, 11, 12, 13]),
        "priority": torch.tensor([MISSING, 4, -1, 2, 2, 9]),
    }
    assert _legal_victims(state, torch.tensor([5]), torch.tensor(chosen))
    assert not _legal_victims(state, torch.tensor([5]), torch.tensor([2]))


@pytest.mark.parametrize("chosen", [[3, 4], [2, 1], [2, 5], [2, 2], [0], [6]])
def test_recall_oracle_rejects_illegal_victims(chosen):
    state = {
        "device_to_host": torch.tensor([MISSING, 10, MISSING, 11, 12, 13]),
        "priority": torch.tensor([MISSING, 4, -1, 2, 2, 9]),
    }
    with pytest.raises(AssertionError):
        _legal_victims(state, torch.tensor([5]), torch.tensor(chosen))


def test_recall_oracle_follows_different_hit_counts_after_legal_ties(monkeypatch):
    pools = [
        SharedSparseTokenPool(128, 3, 2, 4, device="cpu", dtype=torch.float32) for _ in range(2)
    ]
    try:
        pairs = []
        for index, pool in enumerate(pools):
            a, b = [pool.allocate_session(4).layer(0) for _ in range(2)]
            for owner, cache in enumerate((a, b)):
                _append(cache, torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100 * owner)
            ids = torch.tensor([[0]], dtype=torch.int32)
            before = _capture(a)
            with monkeypatch.context() as patch:
                if index == 1:

                    def tied_last_slot(protected, count, _cache=a):
                        assert count == 1
                        chosen = torch.tensor([4])
                        _cache._evict(chosen)
                        return chosen

                    patch.setattr(a, "_available_slots", tied_last_slot)
                output = a._ensure_from_topk(ids)
            assert _validate_recall(a, ids, output, before)[1]
            pairs.append((a, b))
        # The first tie evicted B[0] in only one pool. Both next calls are valid,
        # with different resident-hit and recalled-record increments.
        deltas = []
        for _, cache in pairs:
            before = _capture(cache)
            output = cache._ensure_from_topk(ids)
            _validate_recall(cache, ids, output, before)
            deltas.append(cache.metrics()["recalled_records"] - before["stats"]["recalled_records"])
        assert deltas == [1, 0]
    finally:
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("clock", [None, PRIORITY_LIMIT - 2, PRIORITY_LIMIT - 1, PRIORITY_LIMIT])
def test_recall_oracle_cpu_trajectory_and_clock_rollover(clock):
    pool = SharedSparseTokenPool(128, 3, 2, 4, device="cpu", dtype=torch.float32, candidate_slots=1)
    try:
        caches = [pool.allocate_session(4).layer(0) for _ in range(2)]
        for index, cache in enumerate(caches):
            _append(cache, torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100 * index)
        if clock is not None:
            pool.layers[0].clock = clock
        # Repeated owners include all-hit calls after an eviction trajectory.
        for owner in (0, 1, 1, 0):
            cache = caches[owner]
            cache.begin_transient(1)
            cache.append(torch.full((1, 3), 500.0))
            for ids in (
                torch.tensor([[0, 0, 2, -1, 4]], dtype=torch.int32),
                torch.empty((0, 7), dtype=torch.int32),
            ):
                before = _capture(cache)
                output = cache._ensure_from_topk(ids)
                _validate_recall(cache, ids, output, before)
            cache.discard_transient()
    finally:
        pool.close()


@pytest.mark.parametrize(
    "fault",
    [
        "records",
        "host_to_device",
        "device_to_host",
        "free",
        "priority",
        "clock",
        "counter",
        "padding",
        "candidate",
        "other_layer",
    ],
)
def test_recall_oracle_rejects_corrupt_completed_state(fault):
    pool = SharedSparseTokenPool(128, 3, 2, 4, device="cpu", dtype=torch.float32, candidate_slots=1)
    try:
        a, b = [pool.allocate_session(4).layer(0) for _ in range(2)]
        for index, cache in enumerate((a, b)):
            _append(cache, torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100 * index)
        a.begin_transient(1)
        a.append(torch.full((1, 3), 500.0))
        ids = torch.tensor([[0, 2, -1, 4]], dtype=torch.int32)
        before = _capture(a)
        output = a._ensure_from_topk(ids)
        layer = pool.layers[0]
        slot = int(output[0, 0])
        if fault == "records":
            layer.records[slot, 0] += 1
        elif fault == "host_to_device":
            layer.host_to_device[before["globals"][0]] = MISSING
        elif fault == "device_to_host":
            layer.device_to_host[slot] = MISSING
        elif fault == "free":
            layer.free[slot] = True
        elif fault == "priority":
            layer.priority[slot] += 1
        elif fault == "clock":
            layer.clock += 1
        elif fault == "counter":
            a.stats.resident_selection_records += 1
        elif fault == "padding":
            output[0, 2] = 0
        elif fault == "candidate":
            layer.records[5, 0] += 1
        else:
            pool.layers[1].clock += 1
        with pytest.raises(AssertionError):
            _validate_recall(a, ids, output, before)
    finally:
        pool.close()


def _pair(pools, *, history=137, competitor=150, pending=False):
    pairs = []
    for pool in pools:
        allocations = [pool.allocate_session(64) for _ in range(4)]
        allocations[0].release()
        allocations[2].release()
        session = pool.allocate_session(history)
        if history > 64:
            assert bool((session._pages[1:] < session._pages[:-1]).any())
        a = session.layer(0)
        b = pool.allocate_session(competitor).layer(0)
        xa = (torch.arange(history * pool.width).reshape(history, pool.width) % 197).to(
            device="cuda", dtype=pool.dtype
        )
        xb = (torch.arange(competitor * pool.width).reshape(competitor, pool.width) % 197 + 1).to(
            device="cuda", dtype=pool.dtype
        )
        _append(a, xa, commit=not pending)
        _append(b, xb, commit=not pending)
        pairs.append((a, b))
    return pairs


@pytest.mark.parametrize(
    "dtype,width", [(torch.uint8, 7), (torch.float32, 7), (torch.bfloat16, 576)]
)
@pytest.mark.parametrize("history", [137, 193])
@requires_cuda
def test_cuda_sparse_recall_partial_free_fragmented_pages_and_candidate_tail(dtype, width, history):
    pools = _pools(width=width, dtype=dtype)
    try:
        pairs = _pair(pools, history=history)
        for a, b in pairs:
            b.truncate(80)
            assert not a.all_history_resident
            a.reset_stats()
            a.begin_transient(35)
            a.append(torch.ones(35, width, device="cuda", dtype=dtype))
        caches = [pair[0] for pair in pairs]
        ids = torch.arange(history + 35, device="cuda", dtype=torch.int32).repeat(7, 1)
        ids[:, 0] = -5
        outputs = _recall_pair(pools, caches, ids)
        assert caches[0].metrics()["recalled_records"] > 0
        assert caches[0].metrics()["evicted_records"] > 0
        for cache, output in zip(caches, outputs, strict=True):
            torch.testing.assert_close(
                output[:, history:], ids[:, history:] - history + 194, rtol=0, atol=0
            )
            cache.discard_transient()
        _proofs_equal(pools, caches, exact=False)
        # Alternate owners repeatedly, including a no-miss uncertified revisit.
        for owner in (1, 0, 0, 1, 0):
            caches = [pair[owner] for pair in pairs]
            ids = torch.randint(-3, caches[0].written, (23, 97), device="cuda", dtype=torch.int32)
            _recall_pair(pools, caches, ids)
    finally:
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("clock", [None, PRIORITY_LIMIT - 2, PRIORITY_LIMIT - 1, PRIORITY_LIMIT])
@requires_cuda
def test_cuda_uncertified_all_hit_empty_padding_and_rollover_preserve_proofs(clock):
    pools = _pools(slots=8)
    try:
        caches = []
        for pool in pools:
            cache = pool.allocate_session(8).layer(0)
            _append(cache, torch.ones(8, 7, device="cuda"))
            pool.layers[0].resident_owner = None
            if clock is not None:
                pool.layers[0].clock = clock
            caches.append(cache)
        for ids in (
            torch.tensor([[0, 0, -1, 7]], device="cuda", dtype=torch.int32),
            torch.empty(0, 19, device="cuda", dtype=torch.int32),
            torch.full((7, 11), -3, device="cuda", dtype=torch.int32),
        ):
            before = [pool.layers[0].map_generation for pool in pools]
            _recall_pair(pools, caches, ids)
            assert pools[1].layers[0].map_generation == before[1]
            assert pools[1].layers[0].append_owner == caches[1].session.owner
            native = pools[0].layers[0]
            if native.map_generation != before[0]:
                assert native.map_generation == before[0] + 1
                assert native.resident_owner is native.append_owner is None
                assert native.resident_end == native.append_cursor == 0
    finally:
        for pool in pools:
            pool.close()


@requires_cuda
def test_cuda_sparse_recall_pending_host_writes_and_stream_change():
    pools = _pools(width=576, dtype=torch.bfloat16, slots=193)
    try:
        pairs = _pair(pools, pending=True)
        assert all(pool._writes for pool in pools)
        stream = torch.cuda.Stream()
        for owner in (0, 1, 0):
            caches = [pair[owner] for pair in pairs]
            ids = torch.randint(
                -1, caches[0].written, (128, 2048), device="cuda", dtype=torch.int32
            )
            before = [_capture(cache) for cache in caches]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs = [cache._ensure_from_topk(ids) for cache in caches]
            torch.cuda.current_stream().wait_stream(stream)
            _check_pair(pools, caches, ids, outputs, before)
        for pair in pairs:
            for cache in pair:
                cache.commit()
    finally:
        for pool in pools:
            pool.close()


@requires_cuda
def test_cuda_bounded_recall_never_reads_a_gpu_scalar_on_the_host():
    from torch.utils._python_dispatch import TorchDispatchMode

    class RejectScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("exact recall attempted a GPU scalar read")
            return func(*args, **(kwargs or {}))

    pools = _pools(width=576, dtype=torch.bfloat16)
    try:
        pairs = _pair(pools)
        caches = [pair[0] for pair in pairs]
        ids = torch.tensor([[0, 1, 0, -1]], device="cuda", dtype=torch.int32)
        # Miss and uncertified all-hit calls must both remain asynchronous.
        for _ in range(2):
            before = [_capture(cache) for cache in caches]
            with RejectScalarRead():
                actual = caches[0]._ensure_from_topk(ids)
            expected = caches[1]._ensure_from_topk(ids)
            _check_pair(pools, caches, ids, [actual, expected], before)
        assert caches[0].metrics()["recalled_records"] == 2
    finally:
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("rows", [128, 1024])
@requires_cuda
def test_cuda_large_recall_obeys_priorities_with_full_history_and_candidates(rows):
    slots = 65536
    checked = SimpleNamespace(
        **{
            name: getattr(cache_ops, name)
            for name in ("protect", "mark_misses", "release_ids", "finalize_prefetch")
        }
    )
    pools = [
        SharedSparseTokenPool(
            3 * slots,
            7,
            2,
            slots,
            device="cuda",
            dtype=torch.float32,
            candidate_slots=35,
            metadata_ops=ops,
        )
        for ops in (cache_ops, checked)
    ]
    try:
        pairs = []
        for pool in pools:
            pair = [pool.allocate_session(slots).layer(0) for _ in range(2)]
            for owner, cache in enumerate(pair):
                _append(cache, torch.full((slots, 7), float(owner), device="cuda"))
            pair[1].truncate(slots // 2)
            pair[0].begin_transient(35)
            pair[0].append(torch.full((35, 7), 3.0, device="cuda"))
            pairs.append(pair)
        caches = [pair[0] for pair in pairs]
        generator = torch.Generator(device="cuda").manual_seed(4721)
        for geometry in ((rows, 2048), (1, slots + 35), (0, 2048)):
            ids = (
                torch.arange(slots + 35, device="cuda", dtype=torch.int32)[None]
                if geometry[1] == slots + 35
                else torch.randint(
                    -3, slots + 35, geometry, device="cuda", dtype=torch.int32, generator=generator
                )
            )
            _recall_pair(pools, caches, ids)
        # Scratch is queried without a GPU scalar read and stays inside the
        # reservation that already covers sort/scan execution temporaries.
        scratch = cache_ops._selection_workspace_bytes(slots, torch.device("cuda:0"))
        assert scratch + 8 * slots <= pools[0].execution_workspace_bytes
    finally:
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("invalidate", ["competing_session", "snapshot", "beyond_pool"])
@requires_cuda
def test_cuda_unplanned_append_matches_checked_without_dynamic_host_reads(invalidate):
    from torch.utils._python_dispatch import TorchDispatchMode

    class RejectScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (
                torch.ops.aten._local_scalar_dense.default,
                torch.ops.aten.nonzero.default,
            ) or (
                func is torch.ops.aten.index.Tensor
                and any(index is not None and index.dtype == torch.bool for index in args[1])
            ):
                raise AssertionError("native append attempted a dynamic host read")
            return func(*args, **(kwargs or {}))

    pools = _pools(width=7, slots=193)
    try:
        caches = []
        history = 193 if invalidate == "beyond_pool" else 137
        for pool in pools:
            cache = pool.allocate_session(386).layer(0)
            _append(cache, torch.ones(history, 7, device="cuda"))
            if invalidate == "competing_session":
                other = pool.allocate_session(193).layer(0)
                _append(other, torch.full((150, 7), 2.0, device="cuda"))
            elif invalidate == "snapshot":
                pool.restore(pool.snapshot())
            caches.append(cache)
        suffix = torch.arange(35 * 7, device="cuda", dtype=torch.float32).reshape(35, 7)
        for index, cache in enumerate(caches):
            cache.begin_step(35)
            if index == 0:
                with RejectScalarRead():
                    cache.append(suffix)
            else:
                cache.append(suffix)
            cache.commit()
        _proofs_equal(pools, caches)
        indices = torch.arange(history, history + 35, device="cuda", dtype=torch.int32)[None]
        outputs = [cache._ensure_from_topk(indices) for cache in caches]
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        _proofs_equal(pools, caches)
        for cache, physical in zip(caches, outputs, strict=True):
            torch.testing.assert_close(cache.records[physical.long()], suffix[None])
    finally:
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("stage", ["wait", "sort", "copy", "publish", "map"])
@requires_cuda
def test_cuda_sparse_recall_failure_matches_checked_state_and_other_user(monkeypatch, stage):
    pools = _pools(slots=8)
    pools[0].metadata_ops = _LegacyMetadataProvider()
    try:
        pairs = []
        for pool in pools:
            a = pool.allocate_session(8).layer(0)
            b = pool.allocate_session(8).layer(0)
            _append(a, torch.arange(56, device="cuda", dtype=torch.float32).reshape(8, 7))
            _append(b, torch.arange(56, device="cuda", dtype=torch.float32).reshape(8, 7) + 100)
            pairs.append((a, b))
        ids = torch.arange(4, device="cuda", dtype=torch.int32)[None]
        for index, (pool, (a, _)) in enumerate(zip(pools, pairs, strict=True)):
            before = _capture(a)
            transfer, calls = [], []
            failure = RuntimeError(f"injected {stage} failure")
            if stage == "wait":
                target, name = pool, "wait_host"
            elif stage == "sort":
                target, name = (
                    (cache_ops, "sparse_selection_allocate") if index == 0 else (torch, "argsort")
                )
            elif stage in ("copy", "publish"):
                target, name = (
                    (cache_ops, "sparse_selection_publish")
                    if stage == "publish" and index == 0
                    else (kv_transfer, "gather_host_records")
                )
            else:
                target, name = (
                    (cache_ops, "sparse_selection_map") if index == 0 else (pool, "stamp")
                )
            original = getattr(target, name)

            def fail(
                *args,
                _name=name,
                _index=index,
                _original=original,
                _calls=calls,
                _transfer=transfer,
                _failure=failure,
                **kwargs,
            ):
                _calls.append(_name)
                if _name == "gather_host_records" or (stage == "map" and _index == 1):
                    # At these corresponding checked/native boundaries, record
                    # overwrite or the allocation event has already completed.
                    _original(*args, **kwargs)
                if _name == "gather_host_records":
                    _transfer.extend(value.clone() for value in args[2:4])
                elif _name == "sparse_selection_publish":
                    _transfer.extend(value.clone() for value in args[:2])
                raise _failure

            with monkeypatch.context() as patch:
                patch.setattr(target, name, fail)
                with pytest.raises(RuntimeError, match="injected") as caught:
                    a._ensure_from_topk(ids)
            assert caught.value is failure
            assert calls == [name]
            _validate_recall(a, ids, None, before, stage=stage, transfer=transfer)
        _proofs_equal(pools, [pair[0] for pair in pairs], exact=stage in ("wait", "sort"))
        for a, _ in pairs:
            a.session.release()
        caches = [pair[1] for pair in pairs]
        outputs = _recall_pair(
            pools, caches, torch.arange(8, device="cuda", dtype=torch.int32)[None]
        )
        expected = torch.arange(56, device="cuda", dtype=torch.float32).reshape(1, 8, 7) + 100
        for cache, output in zip(caches, outputs, strict=True):
            torch.testing.assert_close(cache.records[output.long()], expected, rtol=0, atol=0)
    finally:
        for pool in pools:
            pool.close()


@requires_cuda
def test_cuda_invalid_private_selection_traps_in_isolated_process():
    code = r"""
import torch
from cache.sparse_token_pool import SharedSparseTokenPool
from operators.deepseek_v32.indexer import cache_ops
pool = SharedSparseTokenPool(128, 1, 1, 8, device="cuda", dtype=torch.float32, metadata_ops=cache_ops)
a, b = [pool.allocate_session(8).layer(0) for _ in range(2)]
for cache in (a,b):
    cache.begin_step(8); cache.append(torch.ones(8,1,device="cuda")); cache.commit()
a._ensure_from_topk(torch.tensor([[8]],device="cuda",dtype=torch.int32))
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode != 0
    # The sticky device failure may be surfaced by TVM FFI/CUB before PyTorch
    # formats it. Both providers report the same CUDA launch failure.
    assert "launch failure" in result.stderr or (
        "CUDA" in result.stderr and "illegal" in result.stderr
    )
