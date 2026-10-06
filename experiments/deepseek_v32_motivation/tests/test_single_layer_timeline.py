"""Dense overlap uses the complete next-layer transfer, excluding control work."""

import pytest

from experiments.deepseek_v32_motivation.src.analyze_pipeline import (
    audit_dense_dma,
    is_host_kv_transfer,
)
from experiments.deepseek_v32_motivation.src.plot_prefill_timeline import classify
from experiments.deepseek_v32_motivation.src.plot_single_layer import dense_pair, display_group


@pytest.mark.parametrize("kind", ["kernel", "memcpy"])
def test_dense_overlap_preserves_full_target_and_current_layer_tail(kind):
    def row(layer, group, kind, stream, start, end):
        return {
            "scheme": "dense_prefetch",
            "layer": str(layer),
            "group": group,
            "kind": kind,
            "stream_id": stream,
            "start_ns": start,
            "end_ns": end,
            "start_ms": max(0, start) / 1e6,
            "end_ms": min(400, end) / 1e6,
        }

    capture = {
        "layer": 1,
        "origin_ns": 0,
        "end_ns": 400,
        "next_layer_host_transfers": [
            {
                "start_ns": 150,
                "end_ns": 650,
                "start_ms": 150 / 1e6,
                "end_ms": 650 / 1e6,
                "stream_id": 69,
                "kind": kind,
                "bytes": 72 * 1024**2 if kind == "memcpy" else 0,
            }
        ],
    }
    rows = [
        row(1, "projection", "kernel", 7, 100, 200),
        row(1, "finish", "kernel", 7, 300, 400),
        row(1, "aux", "memcpy", 7, 200, 300),
        row(1, "transfer", kind, 69, -50, 50),
        row(2, "transfer", kind, 69, 150, 650),
    ]
    result = dense_pair(capture, rows)
    assert result["overlap_ms"] == pytest.approx(150 / 1e6)
    assert result["prefetch_full_union_ms"] == pytest.approx(500 / 1e6)
    assert result["overlap_fraction_of_full_prefetch"] == pytest.approx(0.3)
    assert result["current_layer_transfer_tail_ms"] == pytest.approx(50 / 1e6)
    assert result["prefetch_end_ms"] == pytest.approx(650 / 1e6)
    assert result["prefetch_transport"] == (
        "cuda_memcpy_async" if kind == "memcpy" else "mapped_host_gather"
    )
    assert result["prefetch_memcpy_bytes"] == (72 * 1024**2 if kind == "memcpy" else 0)


def test_only_scoped_main_kv_dma_enters_transfer_lane():
    dma = {"kind": "memcpy", "category": "H2D", "scope": {"stage": "host_dma"}}
    control = {"kind": "memcpy", "category": "H2D", "scope": {"stage": "pool_operation"}}
    other = {"kind": "memcpy", "category": "H2D", "scope": {"stage": "model_input"}}
    d2d = {"kind": "memcpy", "category": "other_copy", "scope": {"stage": "host_dma"}}
    assert is_host_kv_transfer(dma)
    assert display_group(dma) == "transfer"
    assert classify(dma) == ("transfer", "kv_h2d_dma")
    for activity in (control, other, d2d):
        assert not is_host_kv_transfer(activity)
        assert display_group(activity) == "aux"
    assert classify(control) == ("aux", "control_h2d")
    assert classify(other) == ("aux", "other_h2d")


def dma_case():
    metadata = {
        "cases": [
            {
                "scheme": "dense_prefetch",
                "resource_plan": {
                    "dense_fetch_policy": "contiguous_history_cuda_memcpy_async_next_layer_v1"
                },
            }
        ]
    }
    capture = {
        "scheme": "dense_prefetch",
        "segment_counters": {
            "candidate": {
                "host_to_device_bytes": 9216,
                "layers": [{"host_to_device_bytes": 9216}],
            }
        },
    }
    copy = {
        "kind": "memcpy",
        "category": "H2D",
        "scope": {"stage": "host_dma", "segment": "candidate", "layer": "0"},
        "api": {"name": "cudaMemcpyAsync_v3020"},
        "start": 150,
        "end": 650,
        "bytes": 9216,
        "device_id": 0,
        "stream_id": 69,
        "process": 17,
        "correlation": 3,
    }
    call = {
        "stage": "host_dma",
        "segment": "candidate",
        "layer": 0,
        "records": 8,
        "record_bytes": 1152,
        "requested_bytes": 9216,
        "host_contiguous": True,
        "device_contiguous": True,
        "host_pinned": True,
        "host_capacity_records": 64,
        "device_capacity_records": 65,
        "host_row_stride_bytes": 1152,
        "device_row_stride_bytes": 1152,
        "host_start": 4,
        "device_start": 1,
        "host_address_start": 10000,
        "host_address_end": 19216,
        "device_address_start": 30000,
        "device_address_end": 39216,
        "transfer_implementation": "cudaMemcpyAsync",
    }
    return metadata, capture, copy, call


def test_dma_audit_requires_async_api_contiguous_span_and_counter_byte_equality():
    metadata, capture, copy, call = dma_case()
    result = audit_dense_dma(metadata, capture, [copy], [call])
    assert result["status"] == "passed"
    assert result["raw_async_copies"][0]["end_ns"] == 650
    assert result["contiguous_address_spans"][0]["host_address_end"] == 19216
    assert result["segments"] == [{"segment": "candidate", "host_to_device_bytes": 9216}]


@pytest.mark.parametrize("problem", ["gather", "api", "bytes", "span", "counter", "layer"])
def test_dma_audit_rejects_invalid_transport_evidence(problem):
    metadata, capture, copy, call = dma_case()
    if problem == "gather":
        copy.update(kind="kernel", category="host_gather")
    elif problem == "api":
        copy["api"]["name"] = "cudaMemcpy_v3020"
    elif problem == "bytes":
        copy["bytes"] -= 1
    elif problem == "span":
        call["device_address_end"] += 1
    elif problem == "counter":
        capture["segment_counters"]["candidate"]["host_to_device_bytes"] += 1
    elif problem == "layer":
        copy["scope"]["layer"] = "7"
    with pytest.raises(ValueError):
        audit_dense_dma(metadata, capture, [copy], [call])
