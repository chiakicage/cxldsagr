"""Memory reports keep allocations, reservations and sampled device use distinct."""

from experiments.cache_management.src import report


def test_memory_table_does_not_add_allocated_and_reserved_or_claim_device_peak():
    row = {
        "scheme": "serial_sparse",
        "cached_users": 16,
        "is_revisit": True,
        "prefix_cache_hit": True,
        "memory_after": {
            "torch_peak_allocated_bytes": 21,
            "torch_peak_reserved_bytes": 30,
            "torch_allocated_bytes": 20,
            "torch_reserved_bytes": 25,
            "cuda_total_bytes": 100,
            "cuda_free_bytes": 71,
        },
    }
    for field in (
        "cache_hbm_bytes",
        "cache_dram_bytes",
        "reserved_hbm_bytes",
        "reserved_dram_bytes",
        "shared_cache_hbm_bytes",
        "shared_cache_dram_bytes",
        "session_reserved_hbm_bytes",
        "session_reserved_dram_bytes",
        "cache_host_pages",
        "cache_hbm_tokens",
    ):
        row[field] = 1
    result = report.memory_rows("deepseek", {"run_id": "measured"}, [row])[0]
    assert result["torch_peak_allocated_bytes"] == 21
    assert result["torch_peak_reserved_bytes"] == 30
    assert result["maximum_after_request_device_used_bytes"] == 29
    assert result["maximum_after_request_reserved_minus_allocated_bytes"] == 5
    assert result["maximum_after_request_device_used_minus_reserved_bytes"] == 4
    assert result["revisit_history_hits"] == result["revisit_requests"] == 1
