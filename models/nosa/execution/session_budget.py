"""Pure NOSA session allocation bounds for the native shared-serving path.

Each independent device tensor and pinned host tensor is charged through its
allocator bound. CUDA bounds require the native allocator configuration checked
by the caller. They cover PyTorch allocation blocks, including an unsplit device
block's tail and each pinned host allocation's power-of-two bin; they do not
include model weights, ordinary activations, inactive allocator segments or
unobserved CPU malloc overhead. CPU results are storage-role budgets only, not
an allocation-peak guarantee for the PyTorch reference implementation.
"""

from dataclasses import dataclass
from math import prod

import torch

from cache.allocator.budget import allocation_bytes, pinned_allocation_bytes
from cache.capacity import AllocationSpec, CacheFootprint, allocation_footprint

SCHEMES = ("hbm", "serial_sparse", "dense_prefetch", "overlap")


def _positive(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _align256(size):
    return (size + 255) // 256 * 256


def session_allocation_layout(
    *,
    capacity,
    queries,
    layers,
    kv_heads,
    query_heads,
    head_dim,
    dtype,
    device,
    scheme,
    host_capacity=None,
    dense_counter=False,
):
    """Bound fixed allocations and serial execution phases without allocating.

    ``queries`` bounds every actual batch, including batches smaller than 128
    that can require four normalizer splits. Two maximum indexer slabs remain
    reserved alongside the maximum helper phase. This intentionally also covers
    an audit that conservatively retains a pre-capture old slab after replacement.
    Helper phases use a maximum, not a sum of sequential Q/K/CIS checks, boundary
    concatenation, selection and CIS projection. Pending K/V clones remain live
    across layers and are therefore charged separately for every layer.
    """
    for name, value in (
        ("capacity", capacity),
        ("queries", queries),
        ("layers", layers),
        ("kv_heads", kv_heads),
        ("query_heads", query_heads),
        ("head_dim", head_dim),
    ):
        _positive(name, value)
    if queries > capacity or capacity > 262144:
        raise ValueError("NOSA requires queries <= capacity <= 262144")
    if query_heads % kv_heads:
        raise ValueError("Query heads must divide into KV groups")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("NOSA session budgeting requires a floating-point dtype")
    if scheme not in SCHEMES:
        raise ValueError("Unknown NOSA serving scheme")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("NOSA session budgeting supports CPU or CUDA")
    if device.type == "cuda" and (
        dtype != torch.bfloat16 or head_dim != 128 or query_heads != 16 * kv_heads
    ):
        raise ValueError("CUDA session budgeting requires native BF16/D128/GQA16")

    if host_capacity is None:
        host_capacity = capacity
    if type(host_capacity) is not int or not 0 < host_capacity <= capacity:
        raise ValueError("Host storage capacity must be within execution capacity")
    size, group = dtype.itemsize, query_heads // kv_heads
    compressed = max(0, capacity // 16 - 1)
    pooled = max(0, (capacity - 16) // 64)
    pages = (capacity + 63) // 64
    allocations = []

    def record(
        name,
        shape,
        record_dtype=dtype,
        *,
        tier="hbm",
        phase=None,
        lifetime="session",
        charged_bytes=None,
    ):
        payload = prod(shape) * record_dtype.itemsize
        if charged_bytes is None:
            charged_bytes = allocation_bytes(payload, device) if tier == "hbm" else payload
        item = AllocationSpec(
            name=name,
            owner="session",
            dtype=str(record_dtype).removeprefix("torch."),
            shape=shape,
            device=str(device) if tier == "hbm" else "cpu",
            lifetime=lifetime,
            storage_bytes=payload,
            charged_bytes=charged_bytes,
            peak_group="helpers" if phase is not None else None,
            phase=phase,
            accounting_tier=tier,
        )
        allocations.append(item)
        return item.charged_bytes

    def footprint(items):
        return allocation_footprint(tuple(items))

    record("cis_scores", (layers, capacity, kv_heads))
    derived_begin = len(allocations)
    derived_shapes = ((compressed, kv_heads, head_dim), (compressed, kv_heads), (pooled, kv_heads))
    for layer in range(layers):
        for name, shape in zip(
            ("compressed_keys", "compressed_cis", "pooled_cis"), derived_shapes, strict=True
        ):
            record(f"indexer.layer{layer}.{name}", shape)
    derived = footprint(allocations[derived_begin:]).hbm
    if scheme == "hbm":
        for name in ("keys", "values"):
            record(name, (layers, capacity, kv_heads, head_dim))
        if device.type == "cuda":
            record("indexer.host_flag", (1,), torch.bool, tier="dram")
    else:
        # Fixed history stores H tokens but keeps the established conservative
        # H+A pinned-bin charge. Tensor payload and reservation stay distinct.
        host_charge = pinned_allocation_bytes(
            layers * capacity * kv_heads * head_dim * size, device
        )
        for name in ("keys", "values"):
            record(
                f"host.{name}",
                (layers, host_capacity, kv_heads, head_dim),
                tier="dram",
                charged_bytes=host_charge,
            )
        if device.type == "cuda" and (scheme in ("serial_sparse", "overlap") or dense_counter):
            record("metrics.copy_counters", (2,), torch.int64)
    fixed = footprint(allocations)

    score = queries * kv_heads * pages * size
    if scheme == "hbm":
        workspace_payload = _align256(_align256(score) + 3328 + kv_heads * 64 * 4)
    else:
        workspace_payload = _align256(score) if pages > 64 else 0
    workspace_begin = len(allocations)
    # Retain the old slab alongside its replacement until completion is known.
    for name in ("current", "retiring"):
        record(
            f"indexer.workspace.{name}",
            (workspace_payload,),
            torch.uint8,
            lifetime="execution_scratch",
        )
    workspace_slabs = footprint(allocations[workspace_begin:]).hbm
    pending_begin = len(allocations)
    if scheme != "hbm":
        for layer in range(layers):
            for name in ("keys", "values"):
                record(
                    f"pending.layer{layer}.{name}",
                    (queries, kv_heads, head_dim),
                    lifetime="pending_append",
                )
    pending = footprint(allocations[pending_begin:]).hbm
    helpers_begin = len(allocations)
    count = queries * kv_heads
    cis_source = allocation_bytes(size * count, device)

    def helper_phase(phase, tensors, *, source=True):
        if source:
            record(f"helper.{phase}.cis_source", (count,), phase=phase, lifetime="execution")
        for name, shape, record_dtype in tensors:
            record(f"helper.{phase}.{name}", shape, record_dtype, phase=phase, lifetime="execution")

    creation_begin = len(allocations)
    helper_phase(
        "cis.softplus", (("input", (count,), torch.float32), ("output", (count,), torch.float32))
    )
    helper_phase(
        "cis.scale",
        (
            ("input", (count,), torch.float32),
            ("a", (kv_heads,), torch.float32),
            ("output", (count,), torch.float32),
        ),
    )
    helper_phase("cis.convert", (("input", (count,), torch.float32), ("output", (count,), dtype)))
    cis_creation = footprint(allocations[creation_begin:]).hbm

    selection_begin = len(allocations)
    candidates = []
    if queries >= 128:
        candidates.append(("large", queries, 1))
    candidates.append(("small", min(queries, 127), max(1, min(4, (compressed + 127) // 128))))
    for label, rows, splits in candidates:
        tensors = [
            ("ids", (rows, kv_heads, 64), torch.int64),
            ("valid", (rows, kv_heads, 64), torch.bool),
            ("normalizers", (splits, rows, kv_heads, group, 2), torch.float32),
        ]
        if scheme != "hbm":
            tensors.append(("ranking", (kv_heads, 64), torch.int32))
        helper_phase(f"selection.{label}", tensors)
    selection = footprint(allocations[selection_begin:]).hbm - cis_source

    finite = boundary = 0
    if scheme != "hbm":
        finite_begin = len(allocations)
        for name, elements, scalar in (
            ("q", queries * query_heads * head_dim, False),
            ("k", queries * kv_heads * head_dim, False),
            ("cis", count, True),
        ):
            retained = [("k_scalar", (1,), torch.bool)] if scalar else []
            helper_phase(
                f"finite.{name}.expression",
                retained
                + [
                    ("abs", (elements,), dtype),
                    ("equal", (elements,), torch.bool),
                    ("not_inf", (elements,), torch.bool),
                    ("mask", (elements,), torch.bool),
                ],
            )
            helper_phase(
                f"finite.{name}.reduction",
                retained
                + [
                    ("mask", (elements,), torch.bool),
                    ("scalar", (1,), torch.bool),
                    ("partials", (elements,), torch.bool),
                    ("semaphore", (1,), torch.int32),
                ],
            )
        helper_phase(
            "finite.combine", [(name, (1,), torch.bool) for name in ("k", "cis", "combined")]
        )
        finite = footprint(allocations[finite_begin:]).hbm - cis_source
        boundary_begin = len(allocations)
        boundary_tokens = min(31, capacity)
        helper_phase(
            "boundary",
            [
                ("history", (boundary_tokens, kv_heads, head_dim), dtype),
                ("combined", (queries + boundary_tokens, kv_heads, head_dim), dtype),
            ],
        )
        boundary = footprint(allocations[boundary_begin:]).hbm - cis_source
    phase = footprint(allocations[helpers_begin:]).hbm

    host_temporary = 0
    if device.type == "cuda":
        host_temporary = 1 if scheme == "hbm" else 8
        if scheme in ("serial_sparse", "overlap"):
            host_temporary = max(host_temporary, 16)
        record(
            "host.observation_temporary",
            (host_temporary,),
            torch.uint8,
            tier="dram",
            lifetime="execution",
        )
    total = footprint(allocations)
    breakdown = {
        "hbm": total.hbm,
        "dram": total.dram,
        "fixed_hbm": fixed.hbm,
        "fixed_dram": fixed.dram,
        "derived_hbm": derived,
        "derived_record_payloads": tuple(prod(shape) * size for shape in derived_shapes),
        "pending_hbm": pending,
        "workspace_payload": workspace_payload,
        "workspace_slabs_hbm": workspace_slabs,
        "phase_hbm": phase,
        "cis_creation_hbm": cis_creation,
        "cis_source_hbm": cis_source,
        "selection_hbm": selection,
        "finite_hbm": finite,
        "boundary_hbm": boundary,
        "host_temporary_dram": host_temporary,
        "scope": "native_allocation_bound" if device.type == "cuda" else "cpu_storage_roles",
    }
    return tuple(allocations), breakdown


@dataclass(frozen=True)
class NosaSessionGeometry:
    """Constructor dimensions and provider generation from one admission plan."""

    provider_identity: object
    generation: int
    history_tokens: int
    capacity: int
    execution_capacity: int
    host_capacity: int
    queries: int
    scheme: str
    allocations: tuple[AllocationSpec, ...] = ()
    reservation: CacheFootprint | None = None
    host_pages: int = 0
    hbm_tokens: int = 0


def session_budget_breakdown(**options):
    """Return the established byte ledger derived from named allocation phases."""
    return session_allocation_layout(**options)[1]


def estimate_session_bytes(
    *, capacity, queries, layers, kv_heads, query_heads, head_dim, dtype, device, scheme
):
    """Return the shared-serving session reservation, excluding shared resources."""
    result = session_budget_breakdown(
        capacity=capacity,
        queries=queries,
        layers=layers,
        kv_heads=kv_heads,
        query_heads=query_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        scheme=scheme,
    )
    return {name: result[name] for name in ("hbm", "dram")}
