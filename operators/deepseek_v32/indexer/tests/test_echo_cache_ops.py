"""Native metadata/transport invariants and pinned-upstream helper differentials."""

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from operators.deepseek_v32.indexer import cache_ops
from operators.deepseek_v32.indexer.echo import UPSTREAM_REVISION

MISSING = 2**31 - 1


def require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; explicit GPU entry requires CUDA")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.fail("ECHO transport validation requires Hopper SM90")


def state(slots=64, resident=0, limit=48):
    host = (
        torch.arange(256 * 576, dtype=torch.float32)
        .remainder(997)
        .to(torch.bfloat16)
        .reshape(256, 576)
        .pin_memory()
    )
    records = torch.zeros(slots + 1, 576, dtype=torch.bfloat16, device="cuda")
    h2d = torch.full((256,), MISSING, dtype=torch.int32, device="cuda")
    d2h = torch.full((slots + 1,), MISSING, dtype=torch.int64, device="cuda")
    # Another session's records occupy the oldest slots.
    occupied_ids = torch.arange(128, 128 + resident, device="cuda")
    occupied_slots = torch.arange(1, resident + 1, device="cuda")
    if resident:
        records[occupied_slots] = host[occupied_ids.cpu()].cuda()
        h2d[occupied_ids] = occupied_slots.int()
        d2h[occupied_slots] = occupied_ids
    priority = torch.full((slots + 1,), -1, dtype=torch.int64, device="cuda")
    priority[occupied_slots] = 7
    priority[0] = MISSING
    return {
        "host": host,
        "device": records,
        "host_to_device": h2d,
        "device_to_host": d2h,
        "free_slots": (torch.argsort(priority[1:], stable=True) + 1).int(),
        "allocation_log": torch.full((slots + 1,), MISSING, dtype=torch.int64, device="cuda"),
        "counter": torch.zeros(1, dtype=torch.uint32, device="cuda"),
        "prefetch_stats": torch.zeros(3, dtype=torch.int64, device="cuda"),
        "offset": torch.zeros(16, device="cuda"),
        "page_table": torch.arange(4, dtype=torch.int32, device="cuda"),
        "max_prefetch": limit,
        "priority": priority,
        "free_bitmap": d2h.eq(MISSING).logical_and(torch.arange(slots + 1, device="cuda") > 0),
        "clock": torch.tensor([8], dtype=torch.int64, device="cuda"),
    }


def verify_records_and_maps(lease):
    h2d, d2h = lease["host_to_device"].cpu(), lease["device_to_host"].cpu()
    assert not h2d.eq(-1).any(), "CLAIMED may not escape a completed invocation"
    occupied = torch.nonzero(d2h != MISSING).flatten()
    assert not occupied.eq(0).any()
    assert len(d2h[occupied].unique()) == len(occupied)
    torch.testing.assert_close(h2d[d2h[occupied]].long(), occupied)
    live_ids = torch.nonzero(h2d != MISSING).flatten()
    torch.testing.assert_close(d2h[h2d[live_ids].long()], live_ids)
    torch.testing.assert_close(
        lease["device"].cpu()[occupied], lease["host"][d2h[occupied]], atol=0, rtol=0
    )
    assert lease["device"][0].eq(0).all()
    assert lease["allocation_log"][0].item() == MISSING


@pytest.mark.parametrize("resident", [0, 17, 64])
def test_cuda_prefetch_actual_victims_and_duplicate_global_claims(resident):
    require_sm90()
    lease = state(resident=resident)
    before_h2d = lease["host_to_device"].clone()
    before_d2h = lease["device_to_host"].clone()
    # One warp has a deterministic 11-miss union. Duplicates and invalid IDs
    # must not turn the requested capacity of 48 into 48 evictions.
    candidates = torch.tensor(
        [0, 0, 1, 2, -1, MISSING, 999, *range(3, 11)], dtype=torch.int64, device="cuda"
    )
    cache_ops.prefetch_ids(candidates, lease)
    cache_ops.finalize_prefetch(
        lease["priority"], lease["free_bitmap"], lease["allocation_log"], lease["clock"]
    )
    verify_records_and_maps(lease)
    stats = lease["prefetch_stats"].tolist()
    assert stats == [11, max(0, 11 - (64 - resident)), 0]
    log = lease["allocation_log"].cpu()
    assigned_slots = torch.nonzero(log != MISSING).flatten()
    torch.testing.assert_close(
        assigned_slots.sort().values, lease["free_slots"][:11].cpu().long().sort().values
    )
    untouched = torch.ones(65, dtype=torch.bool, device="cuda")
    untouched[assigned_slots] = False
    torch.testing.assert_close(lease["device_to_host"][untouched], before_d2h[untouched])
    lost_ids = torch.nonzero(
        (before_h2d != MISSING) & (lease["host_to_device"] == MISSING)
    ).flatten()
    assert len(lost_ids) == stats[1]
    assert lease["clock"].item() == 9
    assert lease["priority"][0].item() == MISSING
    assert not lease["free_bitmap"][assigned_slots].any()


@pytest.mark.parametrize("mode", ["empty", "all_hits", "zero_cap"])
def test_cuda_zero_actual_allocations_preserve_pool(mode):
    require_sm90()
    lease = state(resident=64, limit=0 if mode == "zero_cap" else 48)
    candidates = {"empty": [], "all_hits": [128, 128, 129, 145, 191], "zero_cap": [0, 1, 2]}[mode]
    before = {
        key: lease[key].clone()
        for key in ("device", "host_to_device", "device_to_host", "priority", "free_bitmap")
    }
    cache_ops.prefetch_ids(torch.tensor(candidates, dtype=torch.int64, device="cuda"), lease)
    cache_ops.finalize_prefetch(
        lease["priority"], lease["free_bitmap"], lease["allocation_log"], lease["clock"]
    )
    for key, value in before.items():
        torch.testing.assert_close(lease[key], value)
    assert lease["prefetch_stats"][:2].eq(0).all()
    assert lease["clock"].item() == 9
    verify_records_and_maps(lease)


def test_cuda_cap_exhaustion_race_and_exact_counters():
    require_sm90()
    for _ in range(20):
        lease = state(resident=64, limit=7)
        # Thousands of concurrent duplicates contend on the live global map.
        candidates = torch.arange(8192, device="cuda", dtype=torch.int64) % 128
        cache_ops.prefetch_ids(candidates, lease)
        verify_records_and_maps(lease)
        assert lease["prefetch_stats"][:2].tolist() == [7, 7]
        assert lease["counter"].item() == 7 + lease["prefetch_stats"][2].item()
        assert lease["counter"].item() > 7
        assert (lease["allocation_log"] != MISSING).sum().item() == 7


def test_cuda_metadata_protect_release_and_mark_misses_global_ownership():
    require_sm90()
    lease = state(resident=64)
    ids = torch.tensor([128, 129, 129, 0, -1, MISSING], dtype=torch.int64, device="cuda")
    cache_ops.protect(ids, lease["host_to_device"], lease["priority"], lease["clock"])
    assert lease["priority"][1:3].tolist() == [8, 8]
    assert lease["clock"].item() == 9
    output = torch.empty_like(ids)
    cache_ops.mark_misses(ids, lease["host_to_device"], output)
    assert output.tolist() == [MISSING, MISSING, MISSING, 0, MISSING, MISSING]
    cache_ops.release_ids(
        ids,
        lease["host_to_device"],
        lease["device_to_host"],
        lease["priority"],
        lease["free_bitmap"],
    )
    assert lease["device_to_host"][1:3].eq(MISSING).all()
    assert lease["free_bitmap"][1:3].all()
    assert lease["priority"][1:3].eq(-1).all()
    assert lease["clock"].item() == 9
    verify_records_and_maps(lease)
    cache_ops.protect(
        torch.empty(0, dtype=torch.int64, device="cuda"),
        lease["host_to_device"],
        lease["priority"],
        lease["clock"],
    )
    assert lease["clock"].item() == 10


def _official_function(path, name, namespace):
    """Load one pinned official helper, without importing SGLang or its model."""
    tree = ast.parse(path.read_text())
    node = next(
        item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 — pinned read-only reference helper
    return namespace[name]


@pytest.mark.parametrize("allocated_slots", [[], [2, 5, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]])
def test_cuda_post_alloc_and_clock_match_pinned_official_helpers(allocated_slots):
    require_sm90()
    import triton
    import triton.language as tl

    root = Path(__file__).resolve().parents[4]
    official = root / "3rdparty/ECHO"
    assert (
        subprocess.check_output(
            ["git", "-C", str(official), "rev-parse", "HEAD"], text=True
        ).strip()
        == UPSTREAM_REVISION
    )
    official_root = official / "sglang/python/sglang/srt/mem_cache"
    post_alloc = _official_function(
        official_root / "allocator.py",
        "_cuda_graph_allocator_post_alloc_kernel",
        {"triton": triton, "tl": tl},
    )
    priority_event = _official_function(
        official_root / "memory_pool_host.py",
        "update_priority_when_use",
        {"torch": torch, "I32_MAX": MISSING},
    )
    lease = state(slots=9, limit=9)
    slots = torch.tensor(allocated_slots, dtype=torch.int64, device="cuda")
    lease["allocation_log"][slots] = torch.arange(len(slots), device="cuda")
    reference_priority = lease["priority"].clone().unsqueeze(0)
    reference_clock = lease["clock"].clone()
    reference_free = lease["free_bitmap"].clone()
    official_log = torch.zeros(10, dtype=torch.int32, device="cuda")
    official_log[: len(slots)] = slots.int()
    post_alloc[(1,)](official_log, reference_free, 10, BLOCK=256)
    priority_event(
        SimpleNamespace(device_pool_priority=reference_priority, fifo_counter=reference_clock),
        0,
        official_log,
    )
    cache_ops.finalize_prefetch(
        lease["priority"], lease["free_bitmap"], lease["allocation_log"], lease["clock"]
    )
    torch.testing.assert_close(lease["priority"], reference_priority[0])
    torch.testing.assert_close(lease["clock"], reference_clock)
    torch.testing.assert_close(lease["free_bitmap"], reference_free)


def test_cuda_transport_honors_nondefault_stream_and_host_write_dependency():
    require_sm90()
    lease = state(resident=0, limit=16)
    # Compile before starting the deliberately delayed producer.
    cache_ops.prefetch_ids(torch.empty(0, dtype=torch.int64, device="cuda"), lease)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    consumer.wait_stream(torch.cuda.current_stream())
    source = torch.full((16, 576), 123, dtype=torch.bfloat16, device="cuda")
    ready = torch.cuda.Event()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(10_000_000)
        lease["host"][:16].copy_(source, non_blocking=True)
        ready.record()
    with torch.cuda.stream(consumer):
        consumer.wait_event(ready)
        cache_ops.prefetch_ids(torch.arange(16, dtype=torch.int64, device="cuda"), lease)
    consumer.synchronize()
    verify_records_and_maps(lease)
    assert lease["device"][lease["host_to_device"][:16].long()].eq(123).all()


@pytest.mark.parametrize(
    "alias", ["prefix_misses", "prefix_chosen", "misses_chosen", "short_prefix"]
)
def test_sparse_compact_rejects_aliased_or_short_workspace_before_native(monkeypatch, alias):
    def must_not_launch(*args, **kwargs):
        raise AssertionError("invalid scratch reached the native call")

    monkeypatch.setattr(cache_ops, "_call", must_not_launch)
    prefix = torch.empty(8, dtype=torch.int64)
    misses = torch.empty(3, dtype=torch.int64)
    chosen = torch.empty(3, dtype=torch.int64)
    if alias == "prefix_misses":
        misses = prefix[:3]
    elif alias == "prefix_chosen":
        chosen = prefix[:3]
    elif alias == "misses_chosen":
        chosen = misses
    else:
        prefix = prefix[:5]
    with pytest.raises(ValueError, match="distinct storage|miss_prefix"):
        cache_ops.sparse_selection_compact(
            prefix,
            torch.tensor([0], dtype=torch.int32),
            torch.full((64,), MISSING, dtype=torch.int32),
            torch.full((9,), MISSING, dtype=torch.int64),
            torch.full((9,), -1, dtype=torch.int64),
            torch.ones(9, dtype=torch.bool),
            torch.zeros(1, dtype=torch.int64),
            misses,
            chosen,
            history=5,
            timestamp=2,
        )
