"""Conservative capacity planning for a common ECHO GPU memory ceiling.

The resident baseline spends the available cache reservation on whole-user LRU.
Offload modes retain the explicitly configured MLA pool; unused GPU capacity is
reported, not presented as equal allocation or an optimized allocation policy.
"""

from dataclasses import asdict, dataclass

GIB = 1 << 30
PAGE = 64


@dataclass(frozen=True)
class EchoBudgetPlan:
    total_hbm_bytes: int
    non_torch_reserve_bytes: int
    workspace_reserve_bytes: int
    checkpoint_stored_bytes: int
    logical_pool_tokens: int
    retained_users_capacity: int
    device_cache_tokens: int
    estimated_persistent_cache_bytes: int
    dense_context_tokens: int
    policy: str

    @property
    def torch_limit_bytes(self):
        return self.total_hbm_bytes - self.non_torch_reserve_bytes

    def metadata(self):
        return asdict(self) | {
            "torch_allocator_limit_bytes": self.torch_limit_bytes,
            "capacity_estimate_scope": "conservative pool plan, not a measured peak",
            "offload_allocation": "fixed explicit device pool, not maximized to fill HBM",
        }


def plan_echo_budget(
    *,
    mode,
    num_layers,
    users,
    prefix_tokens,
    suffix_tokens,
    checkpoint_stored_bytes,
    total_hbm_bytes=72 * GIB,
    workspace_reserve_bytes=8 * GIB,
    non_torch_reserve_bytes=2 * GIB,
    device_cache_tokens=66624,
):
    if mode not in ("resident", "sparse_sync", "echo_gr_adapted", "dense_prefetch"):
        raise ValueError("unknown budget mode")
    values = (
        num_layers,
        users,
        prefix_tokens,
        suffix_tokens,
        checkpoint_stored_bytes,
        total_hbm_bytes,
        workspace_reserve_bytes,
        non_torch_reserve_bytes,
        device_cache_tokens,
    )
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("budget parameters must be positive integers")
    if num_layers > 5 or prefix_tokens % PAGE or device_cache_tokens % PAGE:
        raise ValueError("expected 1-5 layers and page-aligned capacities")
    reserve = ((suffix_tokens + PAGE - 1) // PAGE + 1) * PAGE
    context = prefix_tokens + reserve
    # Allowance includes unmodeled fixed maps, allocator metadata and page guards.
    fixed_metadata = 256 * (1 << 20)
    available = (
        total_hbm_bytes
        - checkpoint_stored_bytes
        - workspace_reserve_bytes
        - non_torch_reserve_bytes
    )
    if mode == "resident":
        per_token = num_layers * (1152 + 132) + 16
        capacity = (available - fixed_metadata) // per_token
        retained = min(users, (capacity - reserve) // prefix_tokens)
        if retained < 1:
            raise MemoryError("HBM budget cannot hold one complete user and candidate")
        logical = retained * prefix_tokens + reserve
        device = logical
        cache_bytes = fixed_metadata + (logical + PAGE) * per_token
    else:
        if device_cache_tokens < prefix_tokens + suffix_tokens:
            raise MemoryError("offload device pool must hold the complete active context")
        retained = users
        logical = users * prefix_tokens + reserve
        device = device_cache_tokens
        if logical <= device:
            raise MemoryError("ECHO host pool must exceed its device pool")
        # Index + host->device maps + recall marks + host allocator's GPU indices.
        host_side = (logical + PAGE) * (num_layers * (132 + 4) + 1 + 16)
        # MLA + reverse maps/priorities + per-layer allocator + shared free buffers.
        device_side = (device + PAGE) * (num_layers * (1152 + 8 + 4 + 8) + 12)
        dense = (
            2 * (context + PAGE) * 1152 + (logical + 1) * 4 + (context + 1) * 4
            if mode == "dense_prefetch"
            else 0
        )
        cache_bytes = fixed_metadata + host_side + device_side + dense
    if cache_bytes > available:
        raise MemoryError("cache/index/staging estimate plus common reserves exceeds HBM budget")
    return EchoBudgetPlan(
        total_hbm_bytes,
        non_torch_reserve_bytes,
        workspace_reserve_bytes,
        checkpoint_stored_bytes,
        logical,
        retained,
        device,
        cache_bytes,
        context,
        "whole-user HBM LRU" if mode == "resident" else "host user LRU + native ECHO device policy",
    )
