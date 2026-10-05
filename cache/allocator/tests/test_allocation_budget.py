"""CPU checks of payload bounds and effective CUDA allocator admission policy."""

import copy
import os
import subprocess
import sys
from contextlib import nullcontext

import pytest
import torch

from cache.allocator import budget


@pytest.mark.parametrize(
    "size,expected",
    [
        (0, 0),
        (1, 512),
        (511, 512),
        (512, 512),
        (513, 1024),
        (1048575, 1048576),
        (1048576, 1048576),
        (1048577, 2097664),
        (2129408, 3177984),
    ],
)
def test_native_bound_handles_pool_boundary_and_inclusive_tail(size, expected):
    assert budget.allocation_bytes(size, "cuda:0") == expected
    assert budget.allocation_bytes(size, torch.device("cuda")) == expected
    assert budget.allocation_bytes(size, "cpu") == size


def test_independent_allocations_are_not_rounded_as_one_combined_storage():
    assert budget.allocation_bytes(1, "cuda") * 2 == 1024
    assert budget.allocation_bytes(2, "cuda") == 512
    # Two measured compressed-K blocks exceeded ceil512(payload), while each
    # remains below the per-allocation bound established by should_split.
    payload = 2129408
    for allocated in (3048960, 2952192, 3177984):
        assert payload < allocated <= budget.allocation_bytes(payload, "cuda")


@pytest.mark.parametrize(
    "size,expected",
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 4),
        (1023, 1024),
        (1024, 1024),
        (1025, 2048),
        (1 << 30, 1 << 30),
        (1075838976, 1 << 31),
        (1090519040, 1 << 31),
    ],
)
def test_pinned_host_capacity_is_one_power_two_bin_per_cuda_role_allocation(size, expected):
    assert budget.pinned_allocation_bytes(size, "cuda:0") == expected
    assert budget.pinned_allocation_bytes(size, torch.device("cuda")) == expected
    assert budget.pinned_allocation_bytes(size, "cpu") == size


def test_independent_pinned_bins_cannot_be_rounded_after_summing_payloads():
    assert budget.pinned_allocation_bytes(1025, "cuda") == 2048
    assert budget.pinned_allocation_bytes(1, "cuda") == 1
    assert budget.pinned_allocation_bytes(1026, "cuda") == 2048
    assert sum(budget.pinned_allocation_bytes(size, "cuda") for size in (1025, 1)) == 2049


@pytest.mark.parametrize("size", [True, 1.0, "1", None])
def test_allocation_size_rejects_noninteger_values(size):
    for function in (budget.allocation_bytes, budget.pinned_allocation_bytes):
        with pytest.raises(TypeError, match="integer"):
            function(size, "cpu")


def test_negative_allocation_and_unsupported_device_fail():
    for function in (budget.allocation_bytes, budget.pinned_allocation_bytes):
        with pytest.raises(ValueError, match="nonnegative"):
            function(-1, "cuda")
    for function, args in (
        (budget.allocation_bytes, (0, "meta")),
        (budget.pinned_allocation_bytes, (0, "meta")),
        (budget.allocator_policy, ("mps",)),
        (budget.validate_allocator, ("xpu",)),
    ):
        with pytest.raises(NotImplementedError, match="only CPU and CUDA"):
            function(*args)


def test_pure_bounds_policy_and_cpu_validation_do_not_query_cuda_or_allocate(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("pure planning must not query CUDA or allocate tensors")

    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", forbidden)
    monkeypatch.setattr(budget, "_allocator_snapshot", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(budget.gc, "get_objects", forbidden)
    assert budget.allocation_bytes(513, "cuda") == 1024
    assert budget.pinned_allocation_bytes(1025, "cuda") == 2048
    assert budget.pinned_allocation_bytes(1025, "cpu") == 1025
    assert budget.allocator_policy("cuda:3") == budget.CUDA_POLICY
    assert budget.allocator_policy("cpu") == budget.CPU_POLICY
    assert budget.validate_allocator("cpu") == {
        "policy": budget.CPU_POLICY,
        "configuration_key": budget.CPU_POLICY,
    }


@pytest.fixture
def cuda_state(monkeypatch):
    # This fixture injects allocator/GC state. Native authentication has its own
    # tests and must not be invoked with substituted builtin module functions.
    monkeypatch.setattr(
        budget._pool_referrers, "prepare", lambda *args: budget._pool_referrers.identity_resolver
    )
    monkeypatch.setattr(budget, "_resolve_pool_referrers", budget._pool_referrers.identity_resolver)
    monkeypatch.setattr(budget, "_pool_scan_state", None)
    settings = {
        "PYTORCH_CUDA_ALLOC_CONF": "",
        "max_split_size": -1,
        "roundup_power2_divisions": {str(1 << index): 0 for index in range(16)},
        "expandable_segments": False,
        "graph_capture_record_stream_reuse": False,
        "garbage_collection_threshold": 0.0,
    }
    state = {"allocator_settings": settings, "segments": []}
    monkeypatch.setattr(torch.version, "git_version", budget.SUPPORTED_TORCH_REVISION)
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch._C, "_cuda_cudaCachingAllocator_is_enabled", lambda: True)
    monkeypatch.setattr(
        torch._C, "_accelerator_getAllocatorSettings", lambda: settings["PYTORCH_CUDA_ALLOC_CONF"]
    )

    def snapshot():
        return copy.deepcopy(state), {"backend": "injected_fixture"}

    monkeypatch.setattr(budget, "_allocator_snapshot", snapshot)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(budget.gc, "get_objects", lambda generation=None: [])
    for name in budget._NO_CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return state


def test_valid_effective_settings_are_independent_of_environment_strings(cuda_state, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "this is not the parsed effective config")
    first = budget.validate_allocator("cuda:0")
    assert first["policy"] == budget.CUDA_POLICY
    assert first["effective_settings"] == cuda_state["allocator_settings"]
    assert len(first["configuration_key"]) == 64
    assert budget.validate_allocator("cuda:0") == first
    # A supported setting change keeps the sizing policy but invalidates an
    # already-allocated plan's exact configuration key.
    cuda_state["allocator_settings"]["garbage_collection_threshold"] = 0.5
    second = budget.validate_allocator("cuda:0")
    assert second["policy"] == first["policy"]
    assert second["configuration_key"] != first["configuration_key"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_split_size", 64 << 20),
        ("max_split_size", True),
        ("roundup_power2_divisions", {"1": 0}),
        ("roundup_power2_divisions", None),
        ("expandable_segments", True),
        ("graph_capture_record_stream_reuse", True),
    ],
)
def test_effective_nondefault_allocator_geometry_is_rejected(cuda_state, key, value):
    cuda_state["allocator_settings"][key] = value
    with pytest.raises(NotImplementedError, match="unlimited splitting"):
        budget.validate_allocator("cuda:0")


def test_nonzero_power_two_rounding_is_rejected_with_empty_environment(cuda_state):
    cuda_state["allocator_settings"]["roundup_power2_divisions"]["1024"] = 4
    with pytest.raises(NotImplementedError, match="512 B rounding"):
        budget.validate_allocator("cuda:0")


def test_effective_pinned_reserve_segment_setting_is_rejected_without_environment(cuda_state):
    cuda_state["allocator_settings"]["PYTORCH_CUDA_ALLOC_CONF"] = (
        "backend:native,pinned_reserve_segment_size_mb:64"
    )
    with pytest.raises(NotImplementedError, match="pinned reserve segments"):
        budget.validate_allocator("cuda:0")


@pytest.mark.parametrize("backend", ["cudaMallocAsync", "pluggable"])
def test_non_native_allocator_is_rejected(cuda_state, monkeypatch, backend):
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: backend)
    with pytest.raises(NotImplementedError, match="native CUDA allocator"):
        budget.validate_allocator("cuda:0")


def test_disabled_caching_and_uncached_environment_are_both_rejected(cuda_state, monkeypatch):
    monkeypatch.setattr(torch._C, "_cuda_cudaCachingAllocator_is_enabled", lambda: False)
    with pytest.raises(NotImplementedError, match="enabled CUDA caching"):
        budget.validate_allocator("cuda:0")
    monkeypatch.setattr(torch._C, "_cuda_cudaCachingAllocator_is_enabled", lambda: True)
    for name in budget._NO_CACHE_VARIABLES:
        with monkeypatch.context() as context:
            context.setenv(name, "1")
            with pytest.raises(NotImplementedError, match="Uncached CUDA"):
                budget.validate_allocator("cuda:0")


@pytest.mark.parametrize("kind", ["empty", "no_split", "custom", "use_on_oom"])
def test_live_python_pools_are_rejected_even_without_snapshot_segments(
    cuda_state, monkeypatch, kind
):
    class Pool:
        pass

    pool = Pool()
    pool.kind = kind
    monkeypatch.setattr(torch.cuda, "MemPool", Pool)
    monkeypatch.setattr(budget.gc, "get_objects", lambda: [object(), pool])
    with pytest.raises(NotImplementedError, match="live Python MemPool"):
        budget.validate_allocator("cuda:0")


def test_normal_frozen_gc_objects_do_not_prevent_default_pool_admission(cuda_state, monkeypatch):
    monkeypatch.setattr(budget.gc, "get_freeze_count", lambda: 1)
    assert budget.validate_allocator("cuda:0")["policy"] == budget.CUDA_POLICY
    # The snapshot independently catches private segments even when a pool's
    # Python wrapper is frozen and therefore absent from gc.get_objects().
    cuda_state["segments"] = [{"segment_pool_id": (0, 7), "is_expandable": False}]
    with pytest.raises(NotImplementedError, match="Private or expandable"):
        budget.validate_allocator("cuda:0")


@pytest.mark.parametrize(
    "snapshot,message",
    [
        (None, "settings and segments mapping"),
        ({"allocator_settings": None, "segments": []}, "effective settings"),
    ],
)
def test_malformed_snapshot_evidence_fails_clearly(cuda_state, monkeypatch, snapshot, message):
    monkeypatch.setattr(
        budget, "_allocator_snapshot", lambda: (snapshot, {"backend": "injected_fixture"})
    )
    with pytest.raises(TypeError, match=message):
        budget.validate_allocator("cuda:0")


def test_missing_snapshot_segment_evidence_fails_clearly(cuda_state):
    del cuda_state["segments"]
    with pytest.raises(TypeError, match="segment ownership evidence"):
        budget.validate_allocator("cuda:0")


@pytest.mark.parametrize(
    "segment",
    [
        {"segment_pool_id": (0, 7), "is_expandable": False},
        {"segment_pool_id": (0, 0), "is_expandable": True},
        {"is_expandable": False},
    ],
)
def test_private_or_expandable_segments_are_rejected(cuda_state, segment):
    cuda_state["segments"] = [segment]
    with pytest.raises(NotImplementedError, match="Private or expandable"):
        budget.validate_allocator("cuda:0")


def test_default_segments_and_api_freshness_are_checked_each_time(cuda_state):
    cuda_state["segments"] = [{"segment_pool_id": (0, 0), "is_expandable": False}]
    budget.validate_allocator("cuda:0")
    cuda_state["segments"][0]["segment_pool_id"] = (1, 2)
    with pytest.raises(NotImplementedError, match="Private or expandable"):
        budget.validate_allocator("cuda:0")


def test_only_explicit_owned_graph_pools_are_allowed_without_changing_geometry_key(cuda_state):
    baseline = budget.validate_allocator("cuda:0")
    cuda_state["segments"] = [{"segment_pool_id": (0, 7), "is_expandable": False}]
    allowed = budget.validate_allocator("cuda:0", allowed_graph_pools={(0, 7)})
    assert allowed["configuration_key"] == baseline["configuration_key"]
    assert allowed["allowed_graph_pool_ids"] == [(0, 7)]
    with pytest.raises(NotImplementedError, match="Private or expandable"):
        budget.validate_allocator("cuda:0", allowed_graph_pools={(0, 8)})
    cuda_state["segments"][0]["is_expandable"] = True
    with pytest.raises(NotImplementedError, match="Private or expandable"):
        budget.validate_allocator("cuda:0", allowed_graph_pools={(0, 7)})
    with pytest.raises(ValueError, match="nondefault"):
        budget.validate_allocator("cuda:0", allowed_graph_pools={(0, 0)})


@pytest.mark.parametrize("generation", [0, 1, 2])
def test_incremental_pool_scan_rejects_new_empty_subclass_after_promotion(
    cuda_state, monkeypatch, generation
):
    class Pool:
        pass

    class SubPool(Pool):
        pass

    objects = [[], [], []]
    collections = [0, 0, 0]
    scans = []

    def get_objects(index=-1):
        scans.append(index)
        return [*objects[0], *objects[1], *objects[2]] if index == -1 else list(objects[index])

    monkeypatch.setattr(torch.cuda, "MemPool", Pool)
    monkeypatch.setattr(budget, "_pool_scan_state", None)
    monkeypatch.setattr(budget.gc, "get_objects", get_objects)
    monkeypatch.setattr(budget.gc, "get_stats", lambda: [{"collections": n} for n in collections])
    monkeypatch.setattr(budget.gc, "get_freeze_count", lambda: 0)
    budget.validate_allocator("cuda:0")
    assert scans == [-1]
    scans.clear()
    budget.validate_allocator("cuda:0")
    assert scans == [0]
    objects[generation].append(SubPool())
    if generation:
        collections[generation - 1] += 1
    with pytest.raises(NotImplementedError, match="live Python MemPool"):
        budget.validate_allocator("cuda:0")


def test_incremental_pool_scan_rechecks_all_after_unfreeze(cuda_state, monkeypatch):
    class Pool:
        pass

    frozen = [1]
    pool = Pool()
    scans = []

    def get_objects(generation=-1):
        scans.append(generation)
        return [pool] if not frozen[0] and generation in (-1, 2) else []

    monkeypatch.setattr(torch.cuda, "MemPool", Pool)
    monkeypatch.setattr(budget, "_pool_scan_state", None)
    monkeypatch.setattr(budget.gc, "get_objects", get_objects)
    monkeypatch.setattr(budget.gc, "get_stats", lambda: [{"collections": 0}] * 3)
    monkeypatch.setattr(budget.gc, "get_freeze_count", lambda: frozen[0])
    budget.validate_allocator("cuda:0")
    frozen[0] = 0
    with pytest.raises(NotImplementedError, match="live Python MemPool"):
        budget.validate_allocator("cuda:0")
    assert scans == [-1, -1]


def test_pool_scan_repeats_if_collection_occurs_during_inspection(cuda_state, monkeypatch):
    class Pool:
        pass

    collections = [0, 0, 0]
    calls = []
    pool = Pool()

    def get_objects(generation=-1):
        calls.append(generation)
        if len(calls) == 1:
            collections[1] += 1
            return []
        return [pool]

    monkeypatch.setattr(torch.cuda, "MemPool", Pool)
    monkeypatch.setattr(budget, "_pool_scan_state", None)
    monkeypatch.setattr(budget.gc, "get_objects", get_objects)
    monkeypatch.setattr(budget.gc, "get_stats", lambda: [{"collections": n} for n in collections])
    monkeypatch.setattr(budget.gc, "get_freeze_count", lambda: 0)
    with pytest.raises(NotImplementedError, match="live Python MemPool"):
        budget.validate_allocator("cuda:0")
    assert calls == [-1, -1]


def test_pool_scan_never_hashes_or_reads_class_attributes_of_unrelated_objects(
    cuda_state, monkeypatch
):
    class UnhashableType(type):
        __hash__ = None

    class Ordinary(metaclass=UnhashableType):
        @property
        def __class__(self):
            pytest.fail("pool scanner must inspect the concrete type without arbitrary attributes")

    value = Ordinary()
    monkeypatch.setattr(budget, "_pool_scan_state", None)
    monkeypatch.setattr(budget.gc, "get_objects", lambda generation=-1: [value])
    assert budget.validate_allocator("cuda:0")["policy"] == budget.CUDA_POLICY


@pytest.mark.parametrize("generation", [0, 1, 2])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_class_referrers_detect_empty_pool_instances_at_all_depths_and_generations(
    generation, depth
):
    class Pool:
        pass

    class Child(Pool):
        __slots__ = ()

    class Grandchild(Child):
        __slots__ = ()

    assert not budget._has_python_pool_referrers(Pool)
    value = (Pool, Child, Grandchild)[depth]()
    if generation:
        budget.gc.collect(generation - 1)
    assert budget.gc.is_tracked(value)
    assert budget._has_python_pool_referrers(Pool)
    del value
    assert not budget._has_python_pool_referrers(Pool)


def test_class_referrers_retry_when_audit_callback_creates_subclass(monkeypatch):
    class Pool:
        pass

    created = []
    calls = []
    original = budget.gc.get_referrers

    def referrers(*kinds):
        calls.append(kinds)
        if not created:

            class Child(Pool):
                pass

            created.append(Child())
        return original(*kinds)

    monkeypatch.setattr(budget.gc, "get_referrers", referrers)
    assert budget._has_python_pool_referrers(Pool)
    assert len(calls) == 2
    assert type(created[0]) not in calls[0]
    assert type(created[0]) in calls[1]


def test_class_referrers_bypass_overridden_subclass_lookup_and_metaclass_hashing():
    class Meta(type):
        __hash__ = None

        def __subclasses__(cls):
            pytest.fail("must call type.__subclasses__ directly")

    class Pool(metaclass=Meta):
        pass

    class Child(Pool):
        pass

    value = Child()
    assert budget._has_python_pool_referrers(Pool)
    assert value is not None


@pytest.mark.skipif(
    sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12),
    reason="Class-referrer optimization is enabled only for CPython 3.12",
)
def test_hybrid_guard_uses_referrers_again_after_generation_one_collection(monkeypatch):
    class Pool:
        pass

    calls = []
    original = budget._has_python_pool_referrers

    def referrers(pool_type):
        calls.append(pool_type)
        return original(pool_type)

    monkeypatch.setattr(torch.cuda, "MemPool", Pool)
    monkeypatch.setattr(budget, "_pool_scan_state", None)
    monkeypatch.setattr(budget, "_has_python_pool_referrers", referrers)
    budget._reject_python_pools()
    initial = len(calls)
    assert initial >= 1
    value = Pool()
    budget.gc.collect(1)
    with pytest.raises(NotImplementedError, match="live Python MemPool"):
        budget._reject_python_pools()
    assert len(calls) > initial
    assert value is not None


def test_capture_version_change_missing_api_and_config_race_fail(cuda_state, monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(torch.cuda, "is_initialized", lambda: True)
        context.setattr(torch.cuda, "device", lambda device: nullcontext())
        context.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(NotImplementedError, match="CUDA Graph"):
            budget.validate_allocator("cuda:0")
    with monkeypatch.context() as context:
        context.setattr(torch.version, "git_version", "different-revision")
        with pytest.raises(NotImplementedError, match="PyTorch revision"):
            budget.validate_allocator("cuda:0")
    with monkeypatch.context() as context:
        context.delattr(torch._C, "_accelerator_getAllocatorSettings")
        with pytest.raises(AttributeError, match="_accelerator_getAllocatorSettings"):
            budget.validate_allocator("cuda:0")
    values = iter(("", "backend:native"))
    monkeypatch.setattr(torch._C, "_accelerator_getAllocatorSettings", lambda: next(values))
    with pytest.raises(RuntimeError, match="changed during validation"):
        budget.validate_allocator("cuda:0")


def test_real_metadata_query_does_not_initialize_cuda_or_allocate_tensors():
    script = """
import torch
import gc
from cache.allocator.budget import CUDA_POLICY, validate_allocator
def forbidden(*args, **kwargs):
    raise AssertionError('allocator validation must not allocate tensors or initialize CUDA')
for name in ('empty', 'zeros', 'ones', 'full', 'tensor'):
    setattr(torch, name, forbidden)
torch.cuda._lazy_init = forbidden
assert not torch.cuda.is_initialized()
assert validate_allocator('cuda:0')['policy'] == CUDA_POLICY
assert not torch.cuda.is_initialized()
ordinary_frozen_object = ['not a pool']
gc.freeze()
assert gc.get_freeze_count() > 0
assert not any(value is ordinary_frozen_object for value in gc.get_objects())
assert validate_allocator('cuda:0')['policy'] == CUDA_POLICY
assert not torch.cuda.is_initialized()
torch._C._accelerator_setAllocatorSettings('pinned_reserve_segment_size_mb:64')
try:
    validate_allocator('cuda:0')
except NotImplementedError as error:
    assert 'pinned reserve segments' in str(error)
else:
    raise AssertionError('effective pinned reserve segment must be rejected')
assert not torch.cuda.is_initialized()
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    for name in (
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_HIP_ALLOC_CONF",
        *budget._NO_CACHE_VARIABLES,
    ):
        environment.pop(name, None)
    subprocess.run([sys.executable, "-c", script], env=environment, check=True, timeout=30)


@pytest.mark.parametrize("backend", ["private_cpp", "torch_official"])
def test_snapshot_backend_evidence_does_not_change_geometry_or_relax_guards(
    cuda_state, monkeypatch, backend
):
    original = budget._allocator_snapshot
    first = budget.validate_allocator("cuda:0")
    runtime = {
        "backend": backend,
        "fallback_reason": "compiler unavailable" if backend == "torch_official" else None,
    }
    monkeypatch.setattr(budget, "_allocator_snapshot", lambda: (original()[0], runtime))
    second = budget.validate_allocator("cuda:0")
    assert second["configuration_key"] == first["configuration_key"]
    assert second["effective_settings"] == first["effective_settings"]
    assert second["snapshot_adapter"] == runtime
    cuda_state["segments"] = [{"segment_pool_id": (1, 2), "is_expandable": False}]
    with pytest.raises(NotImplementedError, match="Private or expandable"):
        budget.validate_allocator("cuda:0")
