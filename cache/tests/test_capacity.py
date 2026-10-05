"""Capacity accounting distinguishes shared aliases, peaks and admission quotas."""

import json
from dataclasses import replace

import pytest

from cache.capacity import (
    AllocationSpec,
    CacheFootprint,
    CapacityPolicy,
    ResourcePlan,
    allocation_footprint,
)


def test_storage_aliases_and_mutually_exclusive_scratch_do_not_multiply_charges():
    owner = AllocationSpec("pool", "shared", "bfloat16", (128, 16), "cuda:0", "backend", 4096, 4096)
    alias = AllocationSpec(
        "layer_view", "shared", "bfloat16", (64, 16), "cuda:0", "lease", 2048, 0, alias_of="pool"
    )
    scratch = AllocationSpec(
        "projection",
        "shared",
        "float32",
        (256,),
        "cuda:0",
        "projection",
        1024,
        2048,
        peak_group="layer_scratch",
    )
    finish = replace(scratch, name="finish", lifetime="finish", charged_bytes=4096)
    host = AllocationSpec("history", "session", "bfloat16", (513,), "cpu", "session", 1026, 2048)
    assert allocation_footprint((owner, alias, scratch, finish, host)) == CacheFootprint(8192, 2048)
    with pytest.raises(ValueError, match="unique"):
        allocation_footprint((owner, owner))
    with pytest.raises(ValueError, match="owning storage"):
        allocation_footprint((alias,))
    with pytest.raises(ValueError, match="device and owner"):
        allocation_footprint((owner, replace(alias, owner="session")))


def test_fixed_policy_cannot_inherit_a_hidden_byte_subbudget():
    with pytest.raises(ValueError):
        CapacityPolicy("fixed_pools", CacheFootprint(1024, 1024))
    with pytest.raises(ValueError):
        CapacityPolicy("budget")
    assert CapacityPolicy.fixed_pools().budget is None


def test_peak_phase_adds_live_allocations_before_comparing_serial_phases():
    scratch = AllocationSpec(
        "projection.input",
        "session",
        "float32",
        (256,),
        "cuda:0",
        "step",
        1024,
        1536,
        peak_group="helper",
        phase="projection",
    )
    same_phase = replace(scratch, name="projection.output")
    later_phase = replace(scratch, name="finite.output", phase="finite", charged_bytes=2048)
    another_device = replace(later_phase, name="peer.output", device="cuda:1")
    assert allocation_footprint((scratch, same_phase, later_phase)) == CacheFootprint(3072, 0)
    assert allocation_footprint(
        (scratch, same_phase, later_phase, another_device)
    ) == CacheFootprint(5120, 0)


def test_allocation_geometry_and_reservation_conserve_storage_roles():
    allocation = AllocationSpec(
        "device_reference",
        "shared",
        "torch.float32",
        (256,),
        "cpu",
        "backend",
        1024,
        1024,
        accounting_tier="hbm",
    )
    assert allocation.logical_bytes == 1024
    assert allocation_footprint((allocation,)) == CacheFootprint(1024, 0)
    ResourcePlan(shared=CacheFootprint(1024), allocations=(allocation,))
    with pytest.raises(ValueError, match="exceed"):
        ResourcePlan(shared=CacheFootprint(512), allocations=(allocation,))
    with pytest.raises(ValueError, match="dtype and shape"):
        replace(allocation, shape=(257,))


def test_planned_metadata_is_detached_deeply_immutable_and_json_compatible():
    source = {"graph": {"sizes": [128, 1024], "reserved": 8192}}
    plan = ResourcePlan(metadata=source)
    source["graph"]["sizes"].append(2048)
    source["graph"]["reserved"] = 0
    assert json.loads(json.dumps(dict(plan.metadata))) == {
        "graph": {"sizes": [128, 1024], "reserved": 8192}
    }
    with pytest.raises(TypeError, match="immutable"):
        plan.metadata["graph"]["reserved"] = 0
    with pytest.raises(TypeError, match="immutable"):
        plan.metadata["graph"]["sizes"].append(2048)
