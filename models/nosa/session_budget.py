"""Pure NOSA session allocation bounds for the native shared-serving path.

Each independent device tensor and pinned host tensor is charged through its
allocator bound. CUDA bounds require the native allocator configuration checked
by the caller. They cover PyTorch allocation blocks, including an unsplit device
block's tail and each pinned host allocation's power-of-two bin; they do not
include model weights, ordinary activations, inactive allocator segments or
unobserved CPU malloc overhead. CPU results are storage-role budgets only, not
an allocation-peak guarantee for the PyTorch reference implementation.
"""

import torch

from models.nosa.allocation_budget import allocation_bytes, pinned_allocation_bytes

SCHEMES = ("hbm", "serial_sparse", "dense_prefetch", "overlap")


def _positive(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _align256(size):
    return (size + 255) // 256 * 256


def session_budget_breakdown(
    *, capacity, queries, layers, kv_heads, query_heads, head_dim, dtype, device, scheme
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

    size, group = dtype.itemsize, query_heads // kv_heads

    def charge(*sizes):
        return sum(allocation_bytes(value, device) for value in sizes)

    width = kv_heads * head_dim
    compressed = max(0, capacity // 16 - 1)
    pooled = max(0, (capacity - 16) // 64)
    pages = (capacity + 63) // 64
    kv_record = layers * capacity * width * size
    cis_record = layers * capacity * kv_heads * size
    derived_record_sizes = (
        compressed * width * size,
        compressed * kv_heads * size,
        pooled * kv_heads * size,
    )
    # The allocator cannot round the three records or the L layers together:
    # reserve_layer creates three separate tensors for each participating layer.
    derived = layers * charge(*derived_record_sizes)
    fixed_hbm = charge(cis_record) + derived
    fixed_dram = 0
    if scheme == "hbm":
        fixed_hbm += charge(kv_record, kv_record)
        # native_indexer_host_flag is a session-owned pinned Boolean.
        fixed_dram = int(device.type == "cuda")
    else:
        # K and V own separate pinned host bins on CUDA, with no shared arena.
        # The CPU reference path retains its ordinary unpinned payload ledger.
        fixed_dram = 2 * pinned_allocation_bytes(kv_record, device)
        if device.type == "cuda" and scheme in ("serial_sparse", "overlap"):
            fixed_hbm += charge(2 * torch.int64.itemsize)

    score = queries * kv_heads * pages * size
    if scheme == "hbm":
        # Preparation owns 3072 partial bytes + one device Boolean. The checked
        # path separates that region from scores and a head-wise ranking tail.
        workspace_payload = _align256(_align256(score) + 3328 + kv_heads * 64 * 4)
    else:
        workspace_payload = _align256(score) if pages > 64 else 0
    workspace_slabs = 2 * charge(workspace_payload)
    pending = 0 if scheme == "hbm" else 2 * layers * charge(queries * width * size)

    count = queries * kv_heads
    cis_source = charge(size * count)
    # cis_scores retains the BF16 projection while evaluating FP32 softplus,
    # FP32 A scaling and the final model-dtype conversion. These are successive
    # liveness groups from scoring.py; A.float() is a separate allocation.
    cis_creation = max(
        charge(size * count, 4 * count, 4 * count),
        charge(size * count, 4 * count, 4 * kv_heads, 4 * count),
        charge(size * count, 4 * count, size * count),
    )

    def selection_at(rows, splits):
        elements = rows * kv_heads
        return charge(64 * elements * 8, 64 * elements, splits * elements * group * 2 * 4)

    # Joint/checked selection holds IDs, valid and normalizers together.
    # Separate scores can release normalizers before selection, but conservatively
    # reserving both also covers asynchronous allocation retention. The split
    # transition at q=128 means evaluating only the maximum Q is insufficient.
    selection_candidates = []
    if queries >= 128:
        selection_candidates.append(selection_at(queries, 1))
    small_rows = min(queries, 127)
    small_splits = max(1, min(4, (compressed + 127) // 128))
    selection_candidates.append(selection_at(small_rows, small_splits))
    selection = max(selection_candidates)
    if scheme != "hbm":
        # Resident ranked preparation already provides this storage in W.
        # Offload selection can allocate a separate int32 [KV head, 64] ranking.
        selection += charge(kv_heads * 64 * 4)

    def finite_at(elements):
        # Pinned torch version: isfinite=(x==x)*(abs(x)!=inf). The C++ full
        # expression retains abs and all three Boolean outputs simultaneously.
        expression = charge(size * elements, elements, elements, elements)
        # all() retains its mask and scalar output. Global bool partials have
        # at most one element per input, plus one int semaphore for scalar all.
        # This geometry-derived bound avoids device-property queries in a plan.
        reduction = charge(elements, 1, elements, 4)
        return max(expression, reduction)

    finite = boundary = 0
    if scheme != "hbm":
        # Q validation precedes K/CIS preparation. K.all's scalar survives while
        # CIS is checked; the final bitwise-and can hold three scalar Booleans.
        finite = max(
            finite_at(queries * query_heads * head_dim),
            finite_at(queries * width),
            charge(1) + finite_at(count),
            charge(1, 1, 1),
        )
        boundary_tokens = min(31, capacity)
        boundary = charge(
            boundary_tokens * width * size, (queries + boundary_tokens) * width * size
        )
    # The model-dtype CIS source remains live through indexer and attention even
    # after its contents were copied into the fixed session CIS tensor.
    phase = max(cis_creation, cis_source + max(selection, finite, boundary))

    # workspace's dtype-size probe allocates a CPU scalar. Offload isfinite also
    # constructs a wrapped infinity Scalar(Double) and a later pinned bool for
    # host observation. Sparse request metrics separately copy two int64 device
    # counters to CPU via tolist(); these phases are sequential, so use their max.
    host_temporary = 0
    if device.type == "cuda":
        host_temporary = torch.uint8.itemsize if scheme == "hbm" else torch.float64.itemsize
        if scheme in ("serial_sparse", "overlap"):
            host_temporary = max(host_temporary, 2 * torch.int64.itemsize)
    return {
        "hbm": fixed_hbm + pending + workspace_slabs + phase,
        "dram": fixed_dram + host_temporary,
        "fixed_hbm": fixed_hbm,
        "fixed_dram": fixed_dram,
        "derived_hbm": derived,
        "derived_record_payloads": derived_record_sizes,
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
