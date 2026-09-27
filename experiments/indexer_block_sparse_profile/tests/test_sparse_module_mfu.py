"""Synthetic asynchronous launches exercise module attribution, without a GPU."""

import sqlite3
from copy import deepcopy

import pytest

from experiments.indexer_block_sparse_profile.src.module_mfu import (
    attribute_kernels,
    validate_trace_gpu,
)

TIME_SCALE = 100


def trace(prefix=64):
    total = prefix + 64
    workload = {"prefix_tokens": prefix, "new_tokens": 64, "total_tokens": total, "chunk_size": 64}
    scopes, kernels = [], []
    tid = (1 << 24) + 3

    def scope(text, start, end):
        scopes.append(
            {"text": text, "start": start * TIME_SCALE, "end": end * TIME_SCALE, "globalTid": tid}
        )

    def kernel(name, launch, duration=10, grid=(2, 64, 1)):
        kernels.append(
            {
                "name": name,
                "launch_start": launch * TIME_SCALE,
                "launch_end": (launch + 1) * TIME_SCALE,
                # Execution is deliberately outside the launching module's CPU scope.
                "start": (launch + 3000) * TIME_SCALE,
                "end": (launch + 3000 + duration) * TIME_SCALE,
                "globalTid": tid,
                "globalPid": 1 << 24,
                "correlationId": len(kernels) + 1,
                "deviceId": 0,
                "contextId": 1,
                "streamId": 7,
                "gridX": grid[0],
                "gridY": grid[1],
                "gridZ": grid[2],
            }
        )

    phase_width = (total // 64 + 10) * 1000
    for phase, offset, positions in (
        ("full_prefill", 0, list(range(0, total, 64))),
        ("extend", phase_width, [prefix]),
    ):
        scope(f"NOSA/profile/{phase}/0", offset, offset + phase_width - 1000)
        for call, position in enumerate(positions):
            base = offset + 100 + call * 1000

            def stage(name, start, end, *, phase=phase, position=position, base=base):
                scope(f"NOSA/{phase}/layer_0/{name}/q{position}+64", base + start, base + end)

            kernel("nvjet_sm90_tst_144x128_64x6_1x2_h_bz_TNN", base + 10, grid=(2, 66, 1))
            kernel("BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel", base + 50)
            stage("cis_projection", 100, 180)
            kernel("cutlass_bf16_gemm", base + 120, grid=(32, 1, 1))
            kernel("softplus_kernel", base + 140, duration=5)
            stage("indexer_total", 200, 400)
            kernel("indexer_validation", base + 205)
            stage("compression_k", 220, 260)
            kernel("mean_k", base + 230, duration=20)
            stage("compression_cis", 270, 310)
            kernel("mean_cis", base + 280, duration=15)
            kernel("select_short_blocks", base + 320, duration=5)
            if position >= 4096:
                stage("compressed_scores", 325, 345)
                kernel("_scores", base + 330)
                stage("select_from_scores", 345, 390)
                kernel("_select", base + 360)
            stage("block_sparse_attention", 410, 480)
            kernel("_nosa_block_attention", base + 430, duration=40)
            kernel("nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT", base + 500)
            kernel("fused_add_rmsnorm", base + 520)
            kernel("nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT", base + 540, grid=(2, 66, 1))
            kernel("act_and_mul_kernel", base + 570)
            kernel("nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT", base + 600)
    return scopes, kernels, workload


def test_async_attribution_and_inclusive_indexer_children():
    scopes, kernels, workload = trace()
    report = attribute_kernels(scopes, kernels, 1, workload)
    assert len(report["runs"]) == 2
    full, extend = report["runs"]
    assert full["phase"] == "full_prefill"
    assert extend["phase"] == "extend"
    for run, calls in ((full, 2), (extend, 1)):
        modules = run["modules"]
        for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            assert modules[name]["kernel_count"] == calls
            assert modules[name]["kernel_ms"] == pytest.approx(calls * 10 * TIME_SCALE / 1e6)
        assert modules["cis_projection"]["kernel_count"] == calls * 2
        assert modules["cis_projection"]["kernel_ms"] == pytest.approx(
            calls * 15 * TIME_SCALE / 1e6
        )
        assert modules["cis_projection_gemm"]["kernel_ms"] == pytest.approx(
            calls * 10 * TIME_SCALE / 1e6
        )
        assert modules["cis_projection_gemm"]["parent_module"] == "cis_projection"
        assert modules["indexer_total"]["kernel_ms"] == pytest.approx(calls * 50 * TIME_SCALE / 1e6)
        assert modules["compression_k"]["kernel_ms"] == pytest.approx(calls * 20 * TIME_SCALE / 1e6)
        assert modules["compression_k"]["parent_module"] == "indexer_total"
        assert modules["block_sparse_attention"]["kernel_ms"] == pytest.approx(
            calls * 40 * TIME_SCALE / 1e6
        )


def test_input_order_does_not_affect_launch_order_attribution():
    scopes, kernels, workload = trace()
    expected = attribute_kernels(scopes, kernels, 1, workload)
    actual = attribute_kernels(scopes[::-1], kernels[::-1], 1, workload)
    assert actual == expected


def test_compressed_scoring_bypass_and_parent_child_accounting():
    scopes, kernels, workload = trace(prefix=4096)
    result = attribute_kernels(scopes, kernels, 1, workload)
    for run in result["runs"]:
        modules = run["modules"]
        assert modules["compressed_scores"]["kernel_count"] == 1
        assert modules["select_from_scores"]["kernel_count"] == 1
        assert modules["compressed_scores"]["parent_module"] == "indexer_total"
        calls = 65 if run["phase"] == "full_prefill" else 1
        assert modules["indexer_total"]["kernel_ms"] == pytest.approx(
            (calls * 50 + 20) * TIME_SCALE / 1e6
        )


@pytest.mark.parametrize("change", ["missing_gemm", "extra_gemm", "missing_silu", "missing_scope"])
def test_incomplete_or_ambiguous_call_rejected(change):
    scopes, kernels, workload = trace()
    if change == "missing_gemm":
        kernels = kernels[1:]
    elif change == "extra_gemm":
        extra = deepcopy(kernels[0])
        extra.update(
            correlationId=999,
            launch_start=130 * TIME_SCALE,
            start=3130 * TIME_SCALE,
            end=3140 * TIME_SCALE,
        )
        kernels.append(extra)
    elif change == "missing_silu":
        kernels = [kernel for kernel in kernels if "act_and_mul" not in kernel["name"]]
    else:
        scopes = [scope for scope in scopes if "block_sparse_attention" not in scope["text"]]
    with pytest.raises(ValueError):
        attribute_kernels(scopes, kernels, 1, workload)


def test_multiple_streams_rejected_for_serial_reconstruction():
    scopes, kernels, workload = trace()
    kernels[0]["streamId"] = 8
    with pytest.raises(ValueError):
        attribute_kernels(scopes, kernels, 1, workload)


def test_small_timestamp_overlap_preserves_duration_sum():
    scopes, kernels, workload = trace()
    expected = attribute_kernels(scopes, kernels, 1, workload)
    # Nsight can show sub-microsecond timestamp overlap on the same stream.
    kernels[1]["start"] = kernels[0]["end"] - 1
    kernels[1]["end"] = kernels[1]["start"] + 10 * TIME_SCALE
    actual = attribute_kernels(scopes, kernels, 1, workload)
    assert [run["modules"] for run in actual["runs"]] == [
        run["modules"] for run in expected["runs"]
    ]


@pytest.mark.parametrize("mismatch", [None, "uuid", "sm_count", "device"])
def test_sqlite_gpu_identity_matches_kernel_device_and_metadata(tmp_path, mismatch):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE TARGET_INFO_GPU "
            "(id INTEGER,uuid TEXT,name TEXT,smCount INTEGER,computeMajor INTEGER,computeMinor INTEGER)"
        )
        connection.executemany(
            "INSERT INTO TARGET_INFO_GPU VALUES (?,?,?,?,?,?)",
            [(0, "gpu-a", "NVIDIA M403", 132, 9, 0), (1, "gpu-b", "NVIDIA M403", 132, 9, 0)],
        )
    gpu = {"uuid": "GPU-A", "sm_count": 132, "capability": [9, 0]}
    if mismatch == "uuid":
        gpu["uuid"] = "GPU-B"
    elif mismatch == "sm_count":
        gpu["sm_count"] = 100
    device = 2 if mismatch == "device" else 0
    if mismatch:
        with pytest.raises(ValueError):
            validate_trace_gpu(path, device, gpu)
    else:
        assert validate_trace_gpu(path, device, gpu)["id"] == 0
