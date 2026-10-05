import pytest

from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint, PrefixSessionPool

DEFAULT_SHARED_BUDGET = CacheFootprint(50, 100)


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


def test_audit_and_release_failure_preserves_both_errors_and_session(monkeypatch):
    cache, _ = pool()
    audit_error = RuntimeError("audit failed")
    release_error = RuntimeError("drain failed")

    def failed_audit():
        raise audit_error

    def failed_release(session):
        raise release_error

    monkeypatch.setattr(cache, "audit", failed_audit)
    monkeypatch.setattr(cache, "_release", failed_release)
    with pytest.raises(ExceptionGroup) as caught:
        admit(cache, "a")
    assert caught.value.exceptions == (audit_error, release_error)
    assert len(cache) == 1
    assert not cache._entries["a"].ready


def test_lazy_growth_is_audited():
    cache, _ = pool()
    entry = admit(cache, "a").entry
    entry.session["hbm"] = 11
    with pytest.raises(RuntimeError, match="exceeded reservation"):
        cache.audit()


def shared_pool(*, budget=DEFAULT_SHARED_BUDGET, host_pages=4):
    released = []
    cache = PrefixSessionPool(
        budget,
        shared=CacheFootprint(20, 80),
        measure_shared=lambda: {"hbm": 20, "dram": 80},
        host_page_capacity=host_pages,
        allocate=lambda capacity: {"capacity": capacity, "hbm": 10, "dram": 0},
        release=lambda session: released.append(session),
        measure=lambda session: session,
        measure_host_pages=lambda session: (session["capacity"] + 63) // 64,
    )
    return cache, released


def admit_shared(cache, key, *, pages=1, cost=None, signature="prefix"):
    cost = CacheFootprint(10, 0) if cost is None else cost
    lease = cache.acquire(key, signature, pages * 64, cost, host_pages=pages)
    cache.mark_ready(key)
    return lease


def test_shared_arena_is_charged_once_and_outlives_sessions():
    cache, released = shared_pool()
    admit_shared(cache, "a")
    admit_shared(cache, "b")
    assert cache.audit() == CacheFootprint(40, 80)
    assert cache.reserved == CacheFootprint(40, 80)
    assert cache.session_reserved == CacheFootprint(20, 0)
    assert cache.reserved_host_pages == 2
    assert cache.available_host_pages == 2
    cache.close()
    assert len(released) == 2
    assert cache.audit() == cache.reserved == CacheFootprint(20, 80)
    assert cache.reserved_host_pages == 0


def test_fixed_pools_admit_by_pages_despite_large_byte_reservations():
    cache, released = shared_pool(budget=None, host_pages=2)
    cost = CacheFootprint(1 << 34, 1 << 34)
    first = admit_shared(cache, "a", cost=cost)
    admit_shared(cache, "b", cost=cost)
    assert admit_shared(cache, "a", cost=cost).entry is first.entry
    assert not released
    assert cache.budget is None
    assert cache.audit() == CacheFootprint(40, 80)
    assert admit_shared(cache, "c", cost=cost).evicted == ("b",)
    assert cache.reserved_host_pages == 2
    cache.close()
    assert len(released) == 3


def test_fixed_pools_reject_unbounded_or_unpaged_admission():
    with pytest.raises(ValueError, match="positive host page capacity"):
        shared_pool(budget=None, host_pages=0)
    cache, released = shared_pool(budget=None)
    with pytest.raises(ValueError, match="positive session host page count"):
        cache.acquire("a", "prefix", 64, CacheFootprint(10, 0))
    assert not len(cache) and not released


@pytest.mark.parametrize("overflow", ["session", "shared", "pages"])
def test_fixed_pools_still_audit_allocation_reservations_and_pages(overflow):
    cache, _ = shared_pool(budget=None)
    entry = admit_shared(cache, "a").entry
    if overflow == "session":
        entry.session["hbm"] = 11
    elif overflow == "shared":
        cache._measure_shared = lambda: {"hbm": 21, "dram": 80}
    else:
        entry.session["capacity"] = 65
    with pytest.raises(RuntimeError, match="exceeded reservation"):
        cache.audit()
    cache.close()


def test_fixed_pools_impossible_session_preserves_existing_history_and_lru():
    cache, released = shared_pool(budget=None, host_pages=2)
    first = admit_shared(cache, "a")
    admit_shared(cache, "b")
    with pytest.raises(CacheBudgetExceeded, match="host pages"):
        admit_shared(cache, "a", pages=3, signature="changed")
    assert cache._entries["a"] is first.entry and not released
    assert admit_shared(cache, "c").evicted == ("a",)
    cache.close()


@pytest.mark.parametrize("limit", ["bytes", "pages"])
def test_impossible_shared_request_preserves_same_user_and_lru_order(limit):
    cache, released = shared_pool(host_pages=2)
    first = admit_shared(cache, "a")
    admit_shared(cache, "b")
    with pytest.raises(CacheBudgetExceeded):
        admit_shared(
            cache,
            "a",
            signature="different-prefix",
            pages=3 if limit == "pages" else 1,
            cost=CacheFootprint(31, 0) if limit == "bytes" else CacheFootprint(10, 0),
        )
    assert released == []
    assert cache._entries["a"] is first.entry
    assert admit_shared(cache, "c").evicted == ("a",)


def test_page_admission_evicts_multiple_users_with_available_byte_capacity():
    cache, released = shared_pool(budget=CacheFootprint(100, 100), host_pages=3)
    for uid in ("a", "b", "c"):
        admit_shared(cache, uid)
    admit_shared(cache, "a")
    incoming = admit_shared(cache, "d", pages=2)
    assert incoming.evicted == ("b", "c")
    assert len(released) == 2
    assert cache.reserved_host_pages == 3
    assert cache.audit() == CacheFootprint(40, 80)


def test_byte_admission_can_evict_before_host_page_quota():
    cache, _ = shared_pool(host_pages=10)
    for uid in ("a", "b", "c"):
        admit_shared(cache, uid)
    assert admit_shared(cache, "d").evicted == ("a",)
    assert cache.available_host_pages == 7


def test_paged_session_must_reserve_its_private_metadata():
    cache, released = shared_pool()
    with pytest.raises(ValueError, match="nonzero per-session"):
        admit_shared(cache, "a", cost=CacheFootprint())
    assert released == []
    assert len(cache) == 0


def test_actual_page_allocation_is_checked_separately_from_bytes():
    cache, released = shared_pool()
    with pytest.raises(RuntimeError, match="host pages"):
        cache.acquire("a", "prefix", 65, CacheFootprint(10, 0), host_pages=1)
    assert len(cache) == 0
    assert len(released) == 1


def test_shared_growth_cannot_consume_unreserved_free_budget():
    cache, _ = shared_pool()
    cache._measure_shared = lambda: {"hbm": 21, "dram": 80}
    with pytest.raises(RuntimeError, match="shared cache allocation exceeded"):
        cache.audit()


def test_allocator_failure_does_not_clear_other_users():
    cache, released = shared_pool(host_pages=2)
    admit_shared(cache, "a")
    retained = admit_shared(cache, "b")

    def fail(capacity):
        raise RuntimeError("allocator failure")

    cache._allocate = fail
    with pytest.raises(RuntimeError, match="allocator failure"):
        admit_shared(cache, "c")
    assert len(released) == 1
    assert list(cache._entries) == ["b"]
    assert cache._entries["b"] is retained.entry
    assert cache.audit() == CacheFootprint(30, 80)


def test_larger_execution_reservation_is_not_silently_reused():
    cache, _ = shared_pool()
    original = admit_shared(cache, "a")
    lease = admit_shared(cache, "a", cost=CacheFootprint(15, 0))
    assert not lease.hit
    assert lease.entry is not original.entry
    assert cache.reserved == CacheFootprint(35, 80)


def resident_pool(*, tokens=128, host_pages=0):
    released = []
    cache = PrefixSessionPool(
        None,
        hbm_token_capacity=tokens,
        host_page_capacity=host_pages,
        allocate=lambda capacity: {"capacity": capacity, "hbm": capacity * 8, "dram": 0},
        release=lambda session: released.append(session),
        measure=lambda session: session,
        measure_hbm_tokens=lambda session: session["capacity"],
    )
    return cache, released


def admit_resident(cache, key, *, tokens=64, host_pages=0):
    lease = cache.acquire(
        key,
        "history",
        tokens,
        CacheFootprint(tokens * 8, 0),
        hbm_tokens=tokens,
        host_pages=host_pages,
    )
    cache.mark_ready(key)
    return lease


def test_fixed_hbm_token_quota_evicts_lru_without_host_allocation_or_byte_budget():
    cache, released = resident_pool()
    first = admit_resident(cache, "a")
    admit_resident(cache, "b")
    assert admit_resident(cache, "a").entry is first.entry
    assert admit_resident(cache, "c").evicted == ("b",)
    assert len(released) == 1
    assert cache.reserved_host_pages == cache.host_page_capacity == 0
    assert cache.reserved_hbm_tokens == cache.hbm_token_capacity == 128
    assert cache.available_hbm_tokens == 0
    assert cache.budget is None
    cache.close()
    assert cache.reserved_hbm_tokens == 0
    assert cache.available_hbm_tokens == 128


@pytest.mark.parametrize("tokens,host_pages", [(96, 4), (256, 1)])
def test_host_and_hbm_token_quotas_each_constrain_admission(tokens, host_pages):
    cache, _ = resident_pool(tokens=tokens, host_pages=host_pages)
    admit_resident(cache, "a", host_pages=1)
    assert admit_resident(cache, "b", host_pages=1).evicted == ("a",)
    assert cache.reserved_hbm_tokens == 64
    assert cache.reserved_host_pages == 1


def test_impossible_hbm_token_request_does_not_destroy_resident_history():
    cache, released = resident_pool(tokens=64)
    original = admit_resident(cache, "a")
    with pytest.raises(CacheBudgetExceeded, match="HBM tokens"):
        admit_resident(cache, "a", tokens=65)
    assert not released
    assert admit_resident(cache, "a").entry is original.entry


def test_hbm_token_reservation_is_audited_independently_of_bytes():
    cache, released = resident_pool()
    with pytest.raises(RuntimeError, match="HBM tokens.*exceeded reservation"):
        cache.acquire("a", "history", 64, CacheFootprint(1024, 0), hbm_tokens=63)
    assert len(released) == 1
    assert cache.reserved_hbm_tokens == 0


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_hbm_token_quota_rejects_invalid_counts(value):
    with pytest.raises(ValueError, match="hbm_token_capacity"):
        resident_pool(tokens=value)
    cache, released = resident_pool()
    with pytest.raises(ValueError, match="hbm_tokens"):
        cache.acquire("a", "history", 64, CacheFootprint(512, 0), hbm_tokens=value)
    assert not released and not len(cache)
