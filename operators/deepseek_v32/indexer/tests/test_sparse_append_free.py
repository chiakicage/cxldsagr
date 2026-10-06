"""Native persistent append checks using actual before-state slot eligibility."""

import pytest
import torch

from cache.sparse_token_pool import MISSING
from operators.deepseek_v32.indexer import cache_ops


@pytest.fixture(scope="module", autouse=True)
def require_candidate_hopper():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in CPU regression")
    assert torch.cuda.get_device_capability()[0] == 9, "SM90 acceptance required"
    assert hasattr(cache_ops, "sparse_append_free")


def make_state(slots, count, dtype, width, residency):
    start = slots - count
    pages = torch.tensor([7, 2, 9, 1, 5, 3, 12, 11, 14], dtype=torch.int32)
    assert slots <= len(pages) * 64
    h2d = torch.full((1024,), MISSING, dtype=torch.int32)
    d2h = torch.full((slots + 1,), MISSING, dtype=torch.int64)
    priority = torch.full((slots + 1,), -1, dtype=torch.int64)
    free = torch.ones(slots + 1, dtype=torch.bool)
    priority[0], free[0] = MISSING, False
    records = (torch.arange((slots + 1) * width).reshape(slots + 1, width) % 251).to(dtype)
    logicals = torch.arange(start)
    if residency == "cold":
        logicals = logicals[:0]
    elif residency == "partial":
        logicals = logicals[::3]
    # Permuting occupied slots tests free selection from the actual before-state.
    occupancy = torch.randperm(slots, generator=torch.Generator().manual_seed(614)) + 1
    for logical, slot in zip(logicals.tolist(), occupancy.tolist(), strict=False):
        global_id = int(pages[logical // 64]) * 64 + logical % 64
        h2d[global_id], d2h[slot] = slot, global_id
        priority[slot], free[slot] = logical % 17, False
    source = (torch.arange(count * width).reshape(count, width) % 113 + 4).to(dtype)
    return {
        "source": source,
        "records": records,
        "page_table": pages,
        "host_to_device": h2d,
        "device_to_host": d2h,
        "priority": priority,
        "free_bitmap": free,
        "clock": torch.tensor([17], dtype=torch.int64),
        "evictions": torch.tensor([0], dtype=torch.int64),
        "keys": torch.full((slots,), -71, dtype=torch.int64),
        "sorted_keys": torch.full((slots,), -73, dtype=torch.int64),
    }, start


@pytest.mark.parametrize(
    "slots,count", [(7, 1), (193, 65), (255, 128), (256, 256), (257, 128), (513, 128)]
)
@pytest.mark.parametrize(
    "dtype,width", [(torch.uint8, 7), (torch.float32, 7), (torch.bfloat16, 576)]
)
@pytest.mark.parametrize("residency", ["cold", "partial", "all"])
def test_native_free_append_actual_before_state(slots, count, dtype, width, residency):
    before, start = make_state(slots, count, dtype, width, residency)
    states = [{name: value.cuda() for name, value in before.items()} for _ in range(2)]
    # New suffix rows are absent in both directions before any native launch.
    logical = torch.arange(start, start + count)
    global_ids = before["page_table"][logical // 64].long() * 64 + logical % 64
    assert bool((before["host_to_device"][global_ids] == MISSING).all())
    for function, state in zip(
        (cache_ops.sparse_append, cache_ops.sparse_append_free), states, strict=True
    ):
        function(**state, start=start, timestamp=17)
    torch.cuda.synchronize()
    after = [{name: value.cpu() for name, value in state.items()} for state in states]
    # Both implementations must satisfy logical ownership and free-first
    # allocation; physical tie choices need not be identical.
    for variant in after:
        chosen = variant["host_to_device"][global_ids].long()
        assert len(torch.unique(chosen)) == count
        assert bool(before["free_bitmap"][chosen].all())
        torch.testing.assert_close(variant["records"][chosen], before["source"], rtol=0, atol=0)
        torch.testing.assert_close(variant["device_to_host"][chosen], global_ids, rtol=0, atol=0)
    actual = after[1]
    chosen = actual["host_to_device"][global_ids].long()
    assert len(torch.unique(chosen)) == count
    assert bool(before["free_bitmap"][chosen].all())
    assert bool((before["priority"][chosen] == -1).all())
    assert bool((before["device_to_host"][chosen] == MISSING).all())
    torch.testing.assert_close(actual["records"][chosen], before["source"], rtol=0, atol=0)
    torch.testing.assert_close(actual["device_to_host"][chosen], global_ids, rtol=0, atol=0)
    untouched = torch.ones(slots + 1, dtype=torch.bool)
    untouched[chosen] = False
    for name in ("records", "device_to_host", "priority", "free_bitmap"):
        torch.testing.assert_close(actual[name][untouched], before[name][untouched], rtol=0, atol=0)
    assert actual["clock"].tolist() == [18]
    assert actual["evictions"].tolist() == [0]
    assert bool((actual["priority"][chosen] == 17).all())
    assert not bool(actual["free_bitmap"][chosen].any())
