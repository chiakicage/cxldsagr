"""Build the cooperative NOSA sparse-fetch/FA3 kernel and its serial control."""

import hashlib
import json
from functools import cache
from pathlib import Path

from operators.nosa.attention.device_only import _fa3 as resident

_PREFILL_HEADER_SHA256 = "0b20b38b91b295202258203cc484d588fc6251b6eacea9da4c0d5f57ea104ad9"
_FETCH_STRIPES = 8


@cache
def build_info():
    info = resident.build_info()
    header = Path(info["flashinfer_include"]) / "flashinfer/attention/hopper/prefill_sm90.cuh"
    if hashlib.sha256(header.read_bytes()).hexdigest() != _PREFILL_HEADER_SHA256:
        raise RuntimeError("NOSA fused body requires its pinned FlashInfer 0.6.18 prefill header")
    source = Path(__file__).with_name("csrc") / "nosa_offload_fused.cu"
    return {
        **info,
        "fused_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fused_wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fused_prefill_header_sha256": _PREFILL_HEADER_SHA256,
        "host_load_cache_policy": "ld.global.cv.v4.u32",
        "main_schedule": "cooperative_fa3_spare_producer_warps_striped_fetch_queue",
        "compute_order": "head1_then_head0_cost_descending_for_256_batches_and_2_kv_heads",
        "fetch_queue": "head_phased_for_matching_compute_geometry_otherwise_block_major",
        "fetch_queue_within_head": "page0_then_descending_blocks",
        "fetch_queue_layout": "int32[page_count,stripe_cursor,logical_page_slots...]",
        "fetch_queue_compactor_threads": 256,
        "fetch_queue_compactor_in_operator_timing": True,
        "fetch_threads_per_cta": 96,
        "fetch_stripes": _FETCH_STRIPES,
        "fetch_completion": "per_page_acq_rel_stripe_count_then_tma_acquire_async_proxy_fence",
        "fused_cuda_flags": [*resident._FLAGS, f"-DNOSA_FETCH_STRIPES={_FETCH_STRIPES}"],
        "warpgroup_register_limits": {"producer": 24, "consumer": 240},
        "cta_register_pool_guard": "round_up(numRegs * 32, 256) * 12 >= 64512",
        "launch_configuration_cache": "thread_local_context_id_device_16_entries",
        "fetch_barrier_hardware_id": 14,
        "fetch_ctas_semantics": "participating_cta_cap_clamped_to_compute_grid",
        "fetch_batch_max_pages": {"overlap": 1, "serialized": 1},
        "fetch_trace_window": "envelope_of_nonempty_stripe_copy_windows",
        "stripe_trace_window": "post_task_decode_barrier_to_stripe_stores_fence_and_barrier",
        "stripe_trace_layout": "after_page_and_softmax_rows_then_page_slot_times_stripes_plus_stripe",
    }


@cache
def _module():
    import tvm_ffi.cpp

    info = build_info()
    source = Path(__file__).with_name("csrc")
    digest = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]
    return tvm_ffi.cpp.load(
        name=f"cxldsagr_nosa_offload_fused_{digest}",
        sources=[str(source / "nosa_offload_fused.cu")],
        extra_include_paths=[
            str(source),
            str(resident._SOURCE),
            str(resident._ROOT / "3rdparty/cutlass/include"),
            str(resident._ROOT / "3rdparty/cutlass/tools/util/include"),
            info["flashinfer_include"],
        ],
        extra_cuda_cflags=info["fused_cuda_flags"],
    )
