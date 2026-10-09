"""Static capacity planning for a full ECHO fixed-pool session population.

The result is a reservation under the current serving implementation, not a
measured process peak or a physical maximum. It never allocates a model/cache,
changes a live pool, or changes LRU admission. In particular, increasing NH
also reserves every resident session's indexer and the global HBM metadata.
"""

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation

from cache.host_allocation import pinned_allocation_bytes
from models.deepseek_v32.execution.cache_resources import (
    PAGE_SIZE,
    dense_staging_allocation_bytes,
    execution_reservation,
    padded_tokens,
)

INT32_MAX = (1 << 31) - 1
# Native metadata loops increment an int32 index by at most 128 * 256.
# Leave room for that final increment, in addition to the padding sentinel.
MAX_DEVICE_SLOTS = INT32_MAX - 128 * 256
MAX_HOST_TOKENS = (INT32_MAX - 1) // PAGE_SIZE * PAGE_SIZE
DEFAULT_ALLOCATOR_HEADROOM_BYTES = 64 << 20


def _integer(name, value, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _config_value(cfg, name):
    value = cfg[name] if isinstance(cfg, Mapping) else getattr(cfg, name)
    return _integer(f"cfg.{name}", value, minimum=1)


class EchoCapacityPlanner:
    """Plan one variable while holding the other pool capacity fixed.

    The cache envelope is floor(total_hbm_bytes * hbm_fraction) minus
    model_hbm_bytes. The fraction applies to total HBM; model loading is then
    charged in full. A nonpositive envelope cannot admit a session. Explicit
    non-cache headroom is deducted once from the envelope. The fraction's
    unused share is reported separately and is not deducted again.

    Retained sessions own H indexer rows and ceil(H / 64) host pages. Candidate
    KV lives in each layer's backend-owned GPU tail. The whole candidate
    batch executes once; one merged indexer scratch serves the current layer.
    Q = max(chunk_size, A) reserves execution over H + A tokens.

    CUDA allowance includes default native-allocator rounding/tails for all
    enumerated persistent storages, plus configurable scratch/fragmentation
    headroom. The latter defaults to 64 MiB as a planning policy, not a proven
    allocator bound. Non-default allocator settings, ordinary activations,
    library workspaces and whole-process peak memory still require validation.
    """

    def __init__(
        self,
        cfg,
        *,
        num_layers,
        history_tokens,
        candidate_tokens,
        chunk_size,
        total_hbm_bytes,
        model_hbm_bytes,
        dram_budget_bytes,
        hbm_fraction=0.9,
        noncache_headroom_bytes=0,
        allocator_headroom_bytes=DEFAULT_ALLOCATOR_HEADROOM_BYTES,
    ):
        self.num_layers = _integer("num_layers", num_layers, minimum=1)
        self.history_tokens = _integer("history_tokens", history_tokens, minimum=1)
        self.candidate_tokens = _integer("candidate_tokens", candidate_tokens, minimum=1)
        self.chunk_size = _integer("chunk_size", chunk_size, minimum=1)
        self.total_hbm_bytes = _integer("total_hbm_bytes", total_hbm_bytes)
        self.model_hbm_bytes = _integer("model_hbm_bytes", model_hbm_bytes)
        self.dram_budget_bytes = _integer("dram_budget_bytes", dram_budget_bytes)
        self.noncache_headroom_bytes = _integer("noncache_headroom_bytes", noncache_headroom_bytes)
        self.allocator_headroom_bytes = _integer(
            "allocator_headroom_bytes", allocator_headroom_bytes
        )
        try:
            fraction = Decimal(str(hbm_fraction))
        except InvalidOperation as error:
            raise ValueError("hbm_fraction must be finite and satisfy 0 < f <= 1") from error
        if not fraction.is_finite() or not 0 < fraction <= 1:
            raise ValueError("hbm_fraction must be finite and satisfy 0 < f <= 1")
        self.hbm_fraction = float(fraction)
        self.fraction_hbm_limit_bytes = int(self.total_hbm_bytes * fraction)
        self.raw_hbm_envelope_bytes = self.fraction_hbm_limit_bytes - self.model_hbm_bytes
        self.hbm_envelope_bytes = max(0, self.raw_hbm_envelope_bytes)
        self.width = _config_value(cfg, "kv_lora_rank") + _config_value(cfg, "qk_rope_head_dim")
        self.index_head_dim = _config_value(cfg, "index_head_dim")
        self.topk = _config_value(cfg, "index_topk")
        self.context_limit = _config_value(cfg, "max_seq_len")
        self.candidate_persistence = "gpu_transient"
        self.session_capacity_tokens = self.history_tokens
        self.execution_context_tokens = self.history_tokens + self.candidate_tokens
        if self.execution_context_tokens > min(self.context_limit, MAX_DEVICE_SLOTS):
            raise ValueError("H + A exceeds the model context or native indexing limit")
        self.session_page_tokens = padded_tokens(self.session_capacity_tokens)
        self.maximum_device_slots = MAX_DEVICE_SLOTS - self.candidate_tokens
        self.query_tokens = max(self.chunk_size, self.candidate_tokens)
        self.minimum_device_slots = max(
            self.query_tokens, min(self.topk, self.execution_context_tokens)
        )
        if self.minimum_device_slots > self.maximum_device_slots:
            raise ValueError("query workspace exceeds the native device indexing limit")
        self.execution = execution_reservation(
            self.query_tokens,
            self.execution_context_tokens,
            topk=self.topk,
            width=self.width,
        )

    @property
    def budgets(self):
        return {
            "total_hbm_bytes": self.total_hbm_bytes,
            "model_hbm_bytes": self.model_hbm_bytes,
            "hbm_fraction": self.hbm_fraction,
            "fraction_hbm_limit_bytes": self.fraction_hbm_limit_bytes,
            "raw_hbm_envelope_bytes": self.raw_hbm_envelope_bytes,
            "hbm_envelope_bytes": self.hbm_envelope_bytes,
            "fraction_reserve_bytes": self.total_hbm_bytes - self.fraction_hbm_limit_bytes,
            "noncache_headroom_bytes": self.noncache_headroom_bytes,
            "cache_hbm_budget_bytes": self.hbm_envelope_bytes - self.noncache_headroom_bytes,
            "dram_budget_bytes": self.dram_budget_bytes,
            "allocator_scratch_headroom_bytes": self.allocator_headroom_bytes,
        }

    def estimate(self, *, sparse_pool_tokens, host_arena_tokens):
        """Return a JSON-compatible ledger, including failed budget/shape checks.

        Positive, page-aligned capacities may exceed native limits so that the
        next candidate at a search boundary can still show its byte cost.
        ``feasible`` must be checked before using a capacity for execution.
        """
        p = _integer("sparse_pool_tokens", sparse_pool_tokens, minimum=1)
        nh = _integer("host_arena_tokens", host_arena_tokens, minimum=PAGE_SIZE)
        if nh % PAGE_SIZE:
            raise ValueError("host_arena_tokens must be a multiple of 64")
        layers, capacity = self.num_layers, self.session_capacity_tokens
        users, unused_tokens = divmod(nh, self.session_page_tokens)
        page_bytes = self.session_page_tokens // PAGE_SIZE * 4
        record_bytes = self.width * 2
        indexer_key_bytes = capacity * self.index_head_dim
        indexer_scale_bytes = capacity * 4
        resident_bitmap_bytes = ((p + 1 + self.candidate_tokens + 31) // 32) * 4
        # Each (storage bytes, count) remains O(1) in NH and session count.
        shared_storages = [
            ((p + 1 + self.candidate_tokens) * record_bytes, layers),
            (nh * 4, layers),
            ((p + 1) * 8, 2 * layers),
            (p + 1, layers),
            (8, layers),
            (p * 8, layers),
            (p * 4, 1),
            ((p + 1) * 8, 1),
            (4, 1),
            (24, 1),
            (p * 8, 1),
            (resident_bitmap_bytes, 1),
            (4, 1),
            (self.execution_context_tokens * self.index_head_dim, 1),
            (self.execution_context_tokens * 4, 1),
        ]
        private_storages = [
            (indexer_key_bytes, users * layers),
            (indexer_scale_bytes, users * layers),
            (64, users * layers * 3),
            (page_bytes, users),
            # One slab contains prefetch, selection, append and recall counters.
            (layers * 64, users),
        ]
        allocation_allowance = sum(
            (dense_staging_allocation_bytes(size, "cuda") - size) * count
            for size, count in shared_storages + private_storages
        )
        hbm_components = {
            "main_kv_records": layers * (p + 1) * record_bytes,
            "candidate_kv_records": layers * self.candidate_tokens * record_bytes,
            "host_to_device_maps": layers * nh * 4,
            "device_to_host_maps": layers * (p + 1) * 8,
            "priorities": layers * (p + 1) * 8,
            "free_bitmaps": layers * (p + 1),
            "clocks": layers * 8,
            "append_orders": layers * p * 8,
            "shared_scratch": p * 12 + (p + 1) * 8 + 4 + 24,
            "resident_selection_bitmap": resident_bitmap_bytes,
            "resident_selection_count": 4,
            "session_index_keys": users * layers * indexer_key_bytes,
            "session_index_scales": users * layers * indexer_scale_bytes,
            # Match serving admission: current, committed and backup hints.
            "session_hints_reservation": users * layers * 3 * 64,
            "session_page_tables": users * page_bytes,
            "session_prefetch_counters": users * layers * 24,
            "session_native_counters": users * layers * 40,
            "merged_index_keys": self.execution_context_tokens * self.index_head_dim,
            "merged_index_scales": self.execution_context_tokens * 4,
            "execution_indexer_workspace": self.execution.indexer_bytes,
            "execution_append_sources": self.execution.copy_source_bytes,
            "execution_pool_metadata_workspace": 64 * (p + 1) + 64 * nh,
        }
        if self.execution.attention_extra_bytes:
            hbm_components["execution_attention_extra_workspace"] = (
                self.execution.attention_extra_bytes
            )
        reservation = sum(hbm_components.values())
        allowance = allocation_allowance + self.allocator_headroom_bytes
        cache_total = reservation + allowance
        total = cache_total + self.noncache_headroom_bytes
        host_bin = pinned_allocation_bytes(nh * record_bytes)
        dram_components = {
            "host_records_pinned_bins": layers * host_bin,
            "shared_free_page_stack": nh // PAGE_SIZE * 4,
            "session_cpu_page_tables": users * page_bytes,
            "execution_cpu_indexer_workspace": self.execution.cpu_indexer_bytes,
            "execution_cpu_scalar_workspace": self.execution.cpu_scalar_bytes,
            "execution_cpu_metrics_workspace": self.execution.cpu_metrics_bytes,
        }
        dram_total = sum(dram_components.values())
        constraints = []
        if total > self.hbm_envelope_bytes:
            constraints.append("hbm_budget")
        if dram_total > self.dram_budget_bytes:
            constraints.append("dram_budget")
        if p > self.maximum_device_slots:
            constraints.append("native_device_index_limit")
        if nh > MAX_HOST_TOKENS:
            constraints.append("native_host_index_limit")
        if p < self.minimum_device_slots:
            constraints.append("minimum_device_pool")
        if users < 1:
            constraints.append("full_session_fit")
        return {
            "status": "static_reservation_not_measured_peak",
            "feasible": not constraints,
            "violated_constraints": constraints,
            "sparse_pool_tokens": p,
            "host_arena_tokens": nh,
            "full_sessions": users,
            "num_layers": layers,
            "history_tokens": self.history_tokens,
            "candidate_tokens": self.candidate_tokens,
            "candidate_persistence": self.candidate_persistence,
            "chunk_size": self.chunk_size,
            "record_width": self.width,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.topk,
            "session_capacity_tokens": capacity,
            "execution_context_tokens": self.execution_context_tokens,
            "candidate_slots": self.candidate_tokens,
            "candidate_host_tokens": 0,
            "candidate_device_to_host_bytes": 0,
            "session_page_tokens": self.session_page_tokens,
            "unused_host_tokens": unused_tokens,
            "workspace_query_tokens": self.query_tokens,
            "minimum_device_slots": self.minimum_device_slots,
            "maximum_device_slots": self.maximum_device_slots,
            "useful_device_slots": min(p, nh),
            "budgets": self.budgets,
            "hbm": {
                "components": hbm_components,
                "reservation_bytes": reservation,
                "persistent_allocator_allowance_bytes": allocation_allowance,
                "scratch_allocator_headroom_bytes": self.allocator_headroom_bytes,
                "allocator_allowance_bytes": allowance,
                "cache_total_bytes": cache_total,
                "noncache_headroom_bytes": self.noncache_headroom_bytes,
                "total_bytes": total,
                "budget_bytes": self.hbm_envelope_bytes,
                "remaining_bytes": self.hbm_envelope_bytes - total,
            },
            "dram": {
                "components": dram_components,
                "host_record_logical_bytes": layers * nh * record_bytes,
                "host_record_pinned_bin_bytes_per_layer": host_bin,
                "total_bytes": dram_total,
                "budget_bytes": self.dram_budget_bytes,
                "remaining_bytes": self.dram_budget_bytes - dram_total,
            },
        }

    def _search(self, *, variable, fixed, minimum, upper):
        def evaluate(value):
            p, nh = (
                (fixed, value * self.session_page_tokens) if variable == "NH" else (value, fixed)
            )
            return self.estimate(sparse_pool_tokens=p, host_arena_tokens=nh)

        first = evaluate(minimum)
        if not first["feasible"]:
            selected, following = None, first
        else:
            low, high = minimum, max(minimum, upper)
            while low < high:
                middle = (low + high + 1) // 2
                if evaluate(middle)["feasible"]:
                    low = middle
                else:
                    high = middle - 1
            selected, following = evaluate(low), evaluate(low + 1)
        return {
            "status": "static_reservation_not_measured_peak",
            "search_variable": variable,
            "feasible": selected is not None,
            "budgets": self.budgets,
            "selected": selected,
            "next_candidate": following,
            "limiting_constraints": following["violated_constraints"],
            "validation": {
                "gpu_execution_performed": False,
                "physical_peak_proven": False,
                "allocator_assumption": "default native CUDA allocator rounding and split policy",
                "ordinary_activations": (
                    "not measured or bounded; fraction reserve and explicit headroom are provisional"
                ),
                "excluded": [
                    "diagnostic snapshots",
                    "Python objects and non-tensor CPU allocations",
                    "unowned cached allocator blocks",
                ],
                "scope": "fixed history sessions with GPU-transient candidates, serial ECHO/serial_sparse serving",
            },
        }

    def max_host_capacity(self, *, sparse_pool_tokens):
        """Maximize NH in whole-session increments for a fixed P."""
        p = _integer("sparse_pool_tokens", sparse_pool_tokens, minimum=1)
        if not self.minimum_device_slots <= p <= self.maximum_device_slots:
            raise ValueError("fixed P violates the query/top-k or native indexing bound")
        upper = min(
            MAX_HOST_TOKENS // self.session_page_tokens,
            self.dram_budget_bytes // (self.num_layers * self.width * 2 * self.session_page_tokens),
        )
        result = self._search(variable="NH", fixed=p, minimum=1, upper=upper)
        selected = result["selected"]
        result.update(
            fixed_P=p,
            max_host_arena_tokens=selected["host_arena_tokens"] if selected else 0,
            max_full_sessions=selected["full_sessions"] if selected else 0,
            step_tokens=self.session_page_tokens,
        )
        return result

    def max_device_capacity(self, *, host_arena_tokens):
        """Maximize allocated P for a fixed NH; also report min(P, NH).

        P above NH buys no additional distinct host-token residency. It is
        reported as an allocation limit so it cannot be mistaken for useful
        capacity. This search does not simultaneously maximize NH.
        """
        nh = _integer("host_arena_tokens", host_arena_tokens, minimum=PAGE_SIZE)
        if nh % PAGE_SIZE or nh > MAX_HOST_TOKENS:
            raise ValueError("fixed NH must be page aligned and fit the native indexing bound")
        if nh < self.session_page_tokens:
            raise ValueError("fixed NH cannot hold one complete session")
        raw_hbm_upper = max(0, self.hbm_envelope_bytes - self.noncache_headroom_bytes)
        upper = min(self.maximum_device_slots, raw_hbm_upper // (self.num_layers * self.width * 2))
        result = self._search(
            variable="P", fixed=nh, minimum=self.minimum_device_slots, upper=upper
        )
        selected = result["selected"]
        raw = selected["sparse_pool_tokens"] if selected else 0
        useful = min(raw, nh)
        useful_selected = (
            self.estimate(sparse_pool_tokens=useful, host_arena_tokens=nh)
            if useful >= self.minimum_device_slots
            else None
        )
        result.update(
            fixed_NH=nh,
            raw_alloc_max_P=raw,
            useful_max_P=useful,
            useful_selected=useful_selected,
            useful_limiting_constraints=(
                ["host_token_capacity"] if raw > nh else result["limiting_constraints"]
            ),
            step_tokens=1,
        )
        return result
