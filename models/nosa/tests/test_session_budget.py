"""Dimension-derived NOSA session reservations, without CUDA execution."""

import pytest
import torch

from cache.allocator.budget import allocation_bytes
from models.nosa.execution.session_budget import (
    SCHEMES,
    estimate_session_bytes,
    session_budget_breakdown,
)


def options(**changes):
    return {
        "capacity": 66560,
        "queries": 1024,
        "layers": 32,
        "kv_heads": 2,
        "query_heads": 32,
        "head_dim": 128,
        "dtype": torch.bfloat16,
        "device": "cuda:0",
        "scheme": "hbm",
        **changes,
    }


@pytest.mark.parametrize("scheme", SCHEMES)
def test_session_estimate_is_pure_and_charges_individual_records(monkeypatch, scheme):
    def forbidden(*args, **kwargs):
        pytest.fail("Session planning must not allocate or initialize CUDA")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    result = session_budget_breakdown(**options(scheme=scheme))
    # Independent physical tensor shapes: each layer owns three distinct
    # derived records. Combining their payload before allocator rounding would
    # miss at least the individual CIS/pool blocks and large K unsplit tail.
    record_sizes = (4159 * 2 * 128 * 2, 4159 * 2 * 2, 1039 * 2 * 2)
    assert result["derived_record_payloads"] == record_sizes
    assert result["derived_hbm"] == 32 * sum(
        allocation_bytes(size, "cuda:0") for size in record_sizes
    )
    assert result["derived_hbm"] > allocation_bytes(32 * sum(record_sizes), "cuda:0")
    assert estimate_session_bytes(**options(scheme=scheme)) == {
        name: result[name] for name in ("hbm", "dram")
    }


def test_query_bound_covers_smaller_four_split_batch():
    result = session_budget_breakdown(**options(queries=128))
    # q=127 uses four normalizer splits. Its allocation sum exceeds the
    # q=128 one-split joint path even though the actual batch is smaller.
    smaller = sum(
        allocation_bytes(size, "cuda:0")
        for size in (127 * 2 * 64 * 8, 127 * 2 * 64, 4 * 127 * 2 * 16 * 2 * 4)
    )
    larger = sum(
        allocation_bytes(size, "cuda:0")
        for size in (128 * 2 * 64 * 8, 128 * 2 * 64, 128 * 2 * 16 * 2 * 4)
    )
    assert smaller > larger
    assert result["selection_hbm"] >= smaller


@pytest.mark.parametrize("scheme", ("serial_sparse", "dense_prefetch", "overlap"))
def test_offload_reserves_finite_temporaries_boundary_and_every_pending_clone(scheme):
    result = session_budget_breakdown(**options(scheme=scheme))
    elements = 1024 * 32 * 128
    finite_allocations = (2 * elements, elements, elements, elements)
    assert result["finite_hbm"] >= sum(
        allocation_bytes(size, "cuda:0") for size in finite_allocations
    )
    # All L layer appends remain owned until commit. Each K and V is cloned
    # separately; the score/QKV source activation is not retained by this cache.
    assert result["pending_hbm"] == 64 * allocation_bytes(1024 * 2 * 128 * 2, "cuda:0")
    assert result["boundary_hbm"] >= sum(
        allocation_bytes(size, "cuda:0") for size in (31 * 2 * 128 * 2, 1055 * 2 * 128 * 2)
    )
    assert result["phase_hbm"] >= result["finite_hbm"] + result["cis_source_hbm"]
    assert result["workspace_slabs_hbm"] == 2 * allocation_bytes(
        result["workspace_payload"], "cuda:0"
    )
    # Sparse request observation copies both int64 counters to CPU. It follows
    # execution, so reserve max(counter copy, finite-check scalar), not their sum.
    temporary = 16 if scheme in ("serial_sparse", "overlap") else 8
    assert result["host_temporary_dram"] == temporary
    assert result["fixed_dram"] == 4 * 1024**3
    assert result["dram"] == 4 * 1024**3 + temporary
    cpu = session_budget_breakdown(**options(scheme=scheme, device="cpu"))
    assert cpu["host_temporary_dram"] == 0
    assert cpu["dram"] == 2 * 32 * 66560 * 2 * 128 * 2


@pytest.mark.parametrize("capacity,gib", ((65535, 2), (65536, 2), (65537, 4)))
def test_offload_host_capacity_crosses_independent_pinned_bin_boundary(capacity, gib):
    result = session_budget_breakdown(**options(scheme="serial_sparse", capacity=capacity))
    # With 32 layers/H2/D128/BF16, each K or V reaches 1 GiB at 65536 tokens.
    # The next token moves each owning allocation to its 2 GiB pinned bin.
    assert result["fixed_dram"] == gib * 1024**3
    assert result["dram"] == gib * 1024**3 + 16


def test_resident_native_finite_scratch_is_in_workspace_and_host_flag_is_separate():
    result = session_budget_breakdown(**options())
    assert result["finite_hbm"] == result["pending_hbm"] == result["boundary_hbm"] == 0
    assert result["fixed_dram"] == torch.bool.itemsize
    assert result["host_temporary_dram"] == torch.uint8.itemsize
    assert result["dram"] == 2
    assert result["phase_hbm"] >= result["selection_hbm"] + result["cis_source_hbm"]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_budget_covers_smaller_actual_batch_sizes(scheme):
    # Exercise rounding transitions, q=128 normalizer dispatch, and the original
    # 1K query geometry. A bound must cover every smaller admitted batch.
    maxima = session_budget_breakdown(**options(scheme=scheme, queries=1024))
    for queries in (1, 7, 8, 31, 64, 127, 128, 129, 255, 512, 1023):
        actual = session_budget_breakdown(**options(scheme=scheme, queries=queries))
        assert actual["hbm"] <= maxima["hbm"]
        assert actual["dram"] <= maxima["dram"]


def test_cpu_budget_has_role_semantics_without_native_layout_restrictions():
    result = session_budget_breakdown(
        **options(
            capacity=160,
            queries=17,
            layers=4,
            kv_heads=2,
            query_heads=4,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
        )
    )
    assert result["scope"] == "cpu_storage_roles"
    assert result["dram"] == 0


@pytest.mark.parametrize(
    "changes",
    (
        {"capacity": 0},
        {"queries": True},
        {"queries": 66561},
        {"capacity": 262145},
        {"layers": 0},
        {"query_heads": 31},
        {"head_dim": 64},
        {"dtype": torch.float32},
        {"scheme": "unknown"},
        {"device": "meta"},
    ),
)
def test_invalid_native_budget_geometry_fails(changes):
    with pytest.raises(ValueError):
        estimate_session_bytes(**options(**changes))
