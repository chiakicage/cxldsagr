import pytest

from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint, PrefixSessionPool


def pool(budget=None):
    if budget is None:
        budget = CacheFootprint(20, 30)
    released = []
    result = PrefixSessionPool(
        budget,
        allocate=lambda capacity: {"capacity": capacity, "hbm": 10, "dram": 10},
        release=lambda session: released.append(session),
        measure=lambda session: session,
    )
    return result, released


def admit(cache, key, signature="prefix", cost=None, capacity=5):
    if cost is None:
        cost = CacheFootprint(10, 10)
    lease = cache.acquire(key, signature, capacity, cost)
    cache.mark_ready(key)
    return lease


def test_lru_and_both_limits():
    cache, released = pool()
    first = admit(cache, "a")
    admit(cache, "b")
    assert admit(cache, "a").entry is first.entry
    lease = admit(cache, "c")
    assert lease.evicted == ("b",)
    assert len(released) == 1
    assert cache.audit() == CacheFootprint(20, 20)
    cache.close()
    assert len(released) == 3
    assert cache.reserved == CacheFootprint()


def test_host_budget_can_force_eviction_before_hbm():
    cache, _ = pool(CacheFootprint(100, 15))
    admit(cache, 0)
    assert admit(cache, 1).evicted == (0,)


def test_impossible_admission_keeps_existing_sessions():
    cache, released = pool()
    original = admit(cache, "a")
    with pytest.raises(CacheBudgetExceeded):
        admit(cache, "b", cost=CacheFootprint(21, 0))
    assert not released
    assert admit(cache, "a").entry is original.entry


@pytest.mark.parametrize("change", ["identity", "capacity", "unfinished"])
def test_unsafe_prefix_reuse_rebuilds(change):
    cache, released = pool()
    original = admit(cache, "a")
    if change == "unfinished":
        original.entry.ready = False
    lease = admit(
        cache,
        "a",
        signature="changed" if change == "identity" else "prefix",
        capacity=6 if change == "capacity" else 5,
    )
    assert not lease.hit
    assert len(released) == 1


def test_allocation_underestimate_fails_and_releases():
    cache, released = pool()
    with pytest.raises(RuntimeError, match="exceeded reservation"):
        admit(cache, "a", cost=CacheFootprint(1, 1))
    assert len(cache) == 0
    assert len(released) == 1


def test_lazy_growth_is_audited():
    cache, _ = pool()
    entry = admit(cache, "a").entry
    entry.session["hbm"] = 11
    with pytest.raises(RuntimeError, match="exceeded reservation"):
        cache.audit()
