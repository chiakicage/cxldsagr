"""Pure resource planning for DeepSeek fixed pools and byte budgets."""

from dataclasses import dataclass, replace
from math import prod

import torch

from cache.capacity import AllocationSpec, ResourcePlan, allocation_footprint
from cache.host_allocation import pinned_allocation_bytes
from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint
from cache.staging import DoubleBufferStaging
from models.deepseek_v32.execution.cache_resources import (
    CACHE_POLICY_REVISION,
    dense_staging_allocation_bytes,
    dense_ticket_reservation,
    execution_reservation,
    padded_tokens,
)


@dataclass(frozen=True)
class ServingResourceConfig:
    scheme: str
    host_arena_tokens: int | None
    max_seq_len: int
    slots: int
    cfg: object
    workspace_query_tokens: int
    extend_chunk_size: int | None
    chunk_size: int
    num_layers: int
    device: torch.device
    enable_compute_graphs: bool = False
    compute_graph_private_limit_bytes: int = 12 * 2**30


def _plan_serving_resources(config, budgets, limits):
    """Plan fixed allocations and workspace; optional budgets limit admission."""
    from cache.sparse_token_pool import SharedSparseTokenPool

    if budgets is None and config.scheme != "hbm" and config.host_arena_tokens is None:
        raise ValueError("fixed-pools admission requires explicit host_arena_tokens")
    shared_pool = config.scheme in ("echo", "serial_sparse") or (
        config.scheme == "dense_prefetch" and budgets is None
    )
    transient = shared_pool or budgets is None
    capacity = limits["max_session_capacity"]
    if capacity > config.max_seq_len:
        raise ValueError("planned session exceeds model context limit")
    if shared_pool and config.slots < min(config.cfg.index_topk, capacity):
        raise CacheBudgetExceeded("sparse pool must fit one query's full exact selection")
    queries = config.workspace_query_tokens
    candidate = limits.get(
        "max_candidate_tokens",
        capacity if config.extend_chunk_size and not transient else min(capacity, queries),
    )
    if type(candidate) is not int or not 0 < candidate <= capacity:
        raise ValueError("candidate limit must satisfy 0 < A <= C")
    history = limits.get("max_history_tokens", capacity)
    if type(history) is not int or not 0 < history <= capacity:
        raise ValueError("history limit must satisfy 0 < H <= context capacity")
    if candidate > queries and (transient or config.extend_chunk_size is None):
        raise CacheBudgetExceeded("candidate limit exceeds whole-batch query workspace")
    if config.extend_chunk_size is not None and config.extend_chunk_size > queries:
        raise CacheBudgetExceeded("extend chunk exceeds query workspace")
    if shared_pool and queries > config.slots:
        raise CacheBudgetExceeded("query workspace exceeds the per-layer device pool")
    if budgets is None and config.scheme in ("hbm", "dense_prefetch") and history > config.slots:
        raise CacheBudgetExceeded("fixed HBM/dense pools must fit one complete history")
    width = config.cfg.kv_lora_rank + config.cfg.qk_rope_head_dim
    execution = execution_reservation(queries, capacity, topk=config.cfg.index_topk, width=width)
    graph_plan = {}
    if config.enable_compute_graphs:
        from models.deepseek_v32.execution.compute_graphs import plan_compute_graphs

        graph_plan = plan_compute_graphs(
            config.cfg,
            config.num_layers,
            config.chunk_size,
            history,
            candidate,
            config.compute_graph_private_limit_bytes,
        )
    graph_reservation = graph_plan.get("compute_graph_reserved_limit_bytes", 0)
    merged_indexer_bytes = capacity * (config.cfg.index_head_dim + 4) if transient else 0
    if not shared_pool:
        staging_bytes = (
            DoubleBufferStaging.estimate_bytes(
                capacity, {"records": (width,)}, dtype=torch.bfloat16
            )
            if config.scheme == "dense_prefetch"
            else 0
        )
        staging_allocation = (
            dense_staging_allocation_bytes(staging_bytes, config.device) if staging_bytes else 0
        )
        return ResourcePlan(
            shared=CacheFootprint(
                execution.hbm + staging_allocation + merged_indexer_bytes + graph_reservation,
                execution.dram,
            ),
            hbm_tokens=config.slots if budgets is None else 0,
            metadata={
                **graph_plan,
                **({"resource_mode": "fixed_pools"} if budgets is None else {}),
                "cache_policy_revision": CACHE_POLICY_REVISION,
                "pool_scope": config.scheme,
                "max_session_capacity": capacity,
                "max_history_tokens": history,
                "candidate_persistence": "gpu_transient" if transient else "committed",
                "candidate_slots": candidate if transient else 0,
                "merged_indexer_bytes": merged_indexer_bytes,
                "sparse_pool_tokens": config.slots,
                "workspace_query_tokens": queries,
                "max_candidate_tokens": candidate,
                "workspace_indexer_bytes": execution.indexer_bytes,
                **execution.attention_workspace_metadata,
                "workspace_copy_source_bytes": execution.copy_source_bytes,
                "dense_staging_bytes": staging_bytes,
                "dense_staging_allocation_bytes": staging_allocation,
                "dense_staging_scope": "backend" if staging_bytes else None,
                "max_inflight_writes": 2,
                **execution.cpu_workspace_metadata,
            },
        )
    minimum = padded_tokens(history)
    # Raw bytes give only an upper bound. The search below checks each
    # layer's independent pinned bin and metadata at every candidate NH.
    upper = config.host_arena_tokens
    if upper is None:
        upper = budgets.dram // (config.num_layers * width * 2) // 64 * 64
    private = SharedSparseTokenPool.estimate_session_bytes(
        history, layers=config.num_layers, device=config.device
    )
    session = CacheFootprint(
        private["hbm"] + config.num_layers * (history * (config.cfg.index_head_dim + 4) + 3 * 64),
        private["dram"],
    )
    transient_indexer_bytes = merged_indexer_bytes
    dense_ticket_bytes = (
        dense_ticket_reservation(history, config.num_layers, config.device)
        if config.scheme == "dense_prefetch"
        else 0
    )

    def estimate(tokens):
        result = SharedSparseTokenPool.estimate_shared_bytes(
            tokens,
            width,
            config.num_layers,
            config.slots,
            dtype=torch.bfloat16,
            device=config.device,
            candidate_slots=candidate,
        )
        metadata_workspace = SharedSparseTokenPool.estimate_execution_workspace_bytes(
            tokens, config.slots
        )
        return CacheFootprint(
            result["hbm"]
            + execution.hbm
            + graph_reservation
            + metadata_workspace
            + dense_ticket_bytes
            + (transient_indexer_bytes if config.device.type == "cuda" else 0),
            result["dram"]
            + execution.dram
            + (transient_indexer_bytes if config.device.type != "cuda" else 0),
        )

    def fits(tokens):
        shared = estimate(tokens)
        # Every page can move from the shared free stack to a session CPU
        # ownership table. Leave that DRAM available for private admission.
        private_page_tables = tokens // 64 * 4
        return (
            shared.hbm + session.hbm <= budgets.hbm
            and shared.dram + max(session.dram, private_page_tables) <= budgets.dram
        )

    # NH is constrained by global maps in HBM as well as allocated host DRAM.
    # Require room for one maximum-size session before allocating or evicting.
    if budgets is None:
        if upper < minimum:
            raise CacheBudgetExceeded("fixed host arena cannot hold one maximum-size session")
        tokens = upper
    elif upper < minimum or not fits(minimum):
        raise CacheBudgetExceeded("shared pool, workspace and one session do not fit budgets")
    elif config.host_arena_tokens is None:
        low, high = minimum // 64, upper // 64
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle * 64):
                low = middle
            else:
                high = middle - 1
        tokens = low * 64
    else:
        tokens = upper
        if not fits(tokens):
            raise CacheBudgetExceeded("configured host arena plus one session exceeds budgets")
    return ResourcePlan(
        shared=estimate(tokens),
        host_pages=tokens // 64,
        metadata={
            **graph_plan,
            "cache_policy_revision": CACHE_POLICY_REVISION,
            **({"resource_mode": "fixed_pools"} if budgets is None else {}),
            "pool_scope": "backend_per_layer",
            "shared_token_pool": True,
            "dense_fetch_policy": "contiguous_history_cuda_memcpy_async_next_layer_v1"
            if config.scheme == "dense_prefetch"
            else None,
            "dense_contiguous": config.scheme == "dense_prefetch",
            "host_arena_tokens": tokens,
            "sparse_pool_tokens": config.slots,
            "max_session_capacity": capacity,
            "max_history_tokens": history,
            "candidate_persistence": "gpu_transient",
            "candidate_slots": candidate,
            "merged_indexer_bytes": merged_indexer_bytes,
            "workspace_query_tokens": queries,
            "max_candidate_tokens": candidate,
            "workspace_indexer_bytes": execution.indexer_bytes,
            **execution.attention_workspace_metadata,
            "workspace_copy_source_bytes": execution.copy_source_bytes,
            **execution.cpu_workspace_metadata,
            "workspace_metadata_bytes": SharedSparseTokenPool.estimate_execution_workspace_bytes(
                tokens, config.slots
            ),
            "workspace_dense_ticket_bytes": dense_ticket_bytes,
            "max_inflight_writes": 2,
        },
    )


def _allocation(
    name, dtype, shape, device, *, owner="shared", charged=None, lifetime="resource", tier=None
):
    size = prod(shape) * getattr(torch, dtype).itemsize
    return AllocationSpec(
        name,
        owner,
        dtype,
        tuple(shape),
        str(device),
        lifetime,
        size,
        size if charged is None else charged,
        accounting_tier=tier,
    )


def _reservation(name, size, device, *, lifetime="execution", tier=None):
    # An upper bound for concurrently live temporaries, not preallocated storage.
    return _allocation(name, "uint8", (size,), device, lifetime=lifetime, tier=tier)


def resource_allocations(config, plan):
    """Describe every existing storage and conservative execution reservation."""
    metadata = plan.metadata
    device, layers, slots = config.device, config.num_layers, config.slots
    width = config.cfg.kv_lora_rank + config.cfg.qk_rope_head_dim
    capacity = metadata["max_session_capacity"]
    candidate = metadata["candidate_slots"]
    shared_pool = metadata.get("shared_token_pool", False)
    allocations = [
        _reservation(
            "indexer_execution_upper_bound", metadata["workspace_indexer_bytes"], device, tier="hbm"
        ),
        _allocation(
            "pending_append_sources",
            "bfloat16",
            (metadata["max_inflight_writes"], metadata["workspace_query_tokens"], width),
            device,
            lifetime="pending_copy",
            tier="hbm",
        ),
    ]
    if metadata.get("workspace_attention_extra_bytes", 0):
        allocations.append(
            _reservation(
                "attention_execution_extra_upper_bound",
                metadata["workspace_attention_extra_bytes"],
                device,
                tier="hbm",
            )
        )
    for key in ("indexer", "scalar", "metrics"):
        allocations.append(
            _reservation(
                "cpu_" + key + "_upper_bound",
                metadata["workspace_cpu_" + key + "_bytes"],
                "cpu",
            )
        )
    if shared_pool:
        host_tokens = metadata["host_arena_tokens"]
        allocations.append(_allocation("host_free_pages", "int32", (host_tokens // 64,), "cpu"))
        for layer in range(layers):
            stem = f"layer_{layer}."
            raw_host = host_tokens * width * 2
            allocations.append(
                _allocation(
                    stem + "host_records",
                    "bfloat16",
                    (host_tokens, width),
                    "cpu",
                    charged=pinned_allocation_bytes(raw_host)
                    if device.type == "cuda"
                    else raw_host,
                )
            )
            for name, dtype, shape in (
                ("records", "bfloat16", (slots + 1 + candidate, width)),
                ("host_to_device", "int32", (host_tokens,)),
                ("device_to_host", "int64", (slots + 1,)),
                ("priority", "int64", (slots + 1,)),
                ("free", "bool", (slots + 1,)),
                ("clock", "int64", (1,)),
                ("append_order", "int64", (slots,)),
            ):
                allocations.append(_allocation(stem + name, dtype, shape, device))
        for name, dtype, shape in (
            ("free_slots", "int32", (slots,)),
            ("allocation_log", "int64", (slots + 1,)),
            ("counter", "uint32", (1,)),
            ("prefetch_stats", "int64", (3,)),
            ("miss_scratch", "int64", (slots,)),
            ("union_bitmap", "uint32", ((slots + 1 + candidate + 31) // 32,)),
            ("union_count", "uint32", (1,)),
        ):
            allocations.append(_allocation(name, dtype, shape, device))
        allocations.append(
            _reservation(
                "pool_metadata_execution_upper_bound",
                metadata["workspace_metadata_bytes"],
                device,
                tier="hbm",
            )
        )
        if metadata.get("workspace_dense_ticket_bytes", 0):
            allocations.append(
                _reservation(
                    "dense_ticket_execution_upper_bound",
                    metadata["workspace_dense_ticket_bytes"],
                    device,
                    tier="hbm",
                )
            )
    elif metadata.get("dense_staging_bytes", 0):
        allocations.append(
            _allocation(
                "dense_double_staging",
                "bfloat16",
                (2, capacity, width),
                device,
                charged=metadata["dense_staging_allocation_bytes"],
                tier="hbm",
            )
        )
    if metadata["merged_indexer_bytes"]:
        tier = None if shared_pool else "hbm"
        allocations.extend(
            (
                _allocation(
                    "merged_index_keys",
                    "float8_e4m3fn",
                    (capacity, config.cfg.index_head_dim),
                    device,
                    tier=tier,
                ),
                _allocation("merged_index_scales", "float32", (capacity,), device, tier=tier),
            )
        )
    if metadata.get("compute_graphs_enabled", False):
        allocations.extend(
            (
                _allocation(
                    "graph_static",
                    "uint8",
                    (metadata["compute_graph_static_storage_bytes"],),
                    device,
                    charged=metadata["compute_graph_static_allocation_limit_bytes"],
                    tier="hbm",
                ),
                _allocation(
                    "graph_private_reserved_limit",
                    "uint8",
                    (0,),
                    device,
                    charged=metadata["compute_graph_private_limit_bytes"],
                    tier="hbm",
                    lifetime="graph_private_pool_upper_bound",
                ),
            )
        )
    result = tuple(allocations)
    if allocation_footprint(result) != plan.shared:
        raise RuntimeError("DeepSeek named shared allocations differ from the resource reservation")
    return result


def plan_serving_resources(config, budgets, limits):
    plan = _plan_serving_resources(config, budgets, limits)
    return replace(plan, allocations=resource_allocations(config, plan))


@dataclass(frozen=True)
class SessionStoragePlan:
    resource_identity: object
    generation: int
    scheme: str
    capacity: int
    shared_pool: bool
    candidate_slots: int
    allocations: tuple[AllocationSpec, ...]
    reservation: CacheFootprint
    host_pages: int
    hbm_tokens: int
    history_tokens: int


def plan_session_storage(
    config,
    capacity,
    *,
    shared_pool,
    candidate_slots,
    resource_identity,
    generation,
    history_tokens=0,
):
    """One retained allocation calculation consumed by session construction."""
    if type(capacity) is not int or not 1 <= capacity <= config.max_seq_len:
        raise ValueError("session capacity exceeds the model context limit")
    device, layers = config.device, config.num_layers
    width = config.cfg.kv_lora_rank + config.cfg.qk_rope_head_dim
    allocations = []
    for layer in range(layers):
        stem = f"layer_{layer}."
        for name, dtype, shape, lifetime in (
            ("index_keys", "float8_e4m3fn", (capacity, config.cfg.index_head_dim), "session"),
            ("index_scales", "float32", (capacity,), "session"),
            ("offset", "float32", (16,), "session"),
            ("prefix_offset", "float32", (16,), "retained_prefix"),
            ("step_offset", "float32", (16,), "execution_backup"),
        ):
            allocations.append(
                _allocation(
                    stem + name,
                    dtype,
                    shape,
                    device,
                    owner="session",
                    lifetime=lifetime,
                )
            )
        if not shared_pool:
            for name, dtype in (
                ("host_to_device", "int32"),
                ("device_to_host", "int64"),
                ("age", "int64"),
            ):
                allocations.append(
                    _allocation(
                        stem + name,
                        dtype,
                        (capacity,),
                        device,
                        owner="session",
                        lifetime="session",
                    )
                )
            if config.scheme == "hbm":
                allocations.append(
                    _allocation(
                        stem + "records",
                        "bfloat16",
                        (capacity + candidate_slots, width),
                        device,
                        owner="session",
                        lifetime="session",
                    )
                )
            elif config.scheme == "dense_prefetch":
                allocations.append(
                    _allocation(
                        stem + "host_records",
                        "bfloat16",
                        (capacity, width),
                        "cpu",
                        owner="session",
                        lifetime="session",
                        charged=pinned_allocation_bytes(capacity * width * 2),
                    )
                )
    pages = padded_tokens(capacity) // 64 if shared_pool else 0
    if shared_pool:
        allocations.append(
            _allocation(
                "page_table",
                "int32",
                (pages,),
                device,
                owner="session",
                lifetime="session",
            )
        )
        if device.type == "cuda":
            allocations.append(
                _allocation(
                    "owned_host_pages",
                    "int32",
                    (pages,),
                    "cpu",
                    owner="session",
                    lifetime="session",
                )
            )
        allocations.append(
            _allocation(
                "layer_counters",
                "int64",
                (layers, 8),
                device,
                owner="session",
                lifetime="session",
            )
        )
    allocations = tuple(allocations)
    return SessionStoragePlan(
        resource_identity,
        generation,
        config.scheme,
        capacity,
        shared_pool,
        candidate_slots,
        allocations,
        allocation_footprint(allocations),
        pages,
        capacity if config.scheme == "hbm" else 0,
        history_tokens,
    )
