"""Guard asynchronous attribution, helper accounting and weighted utilization."""

import pytest

from experiments.deepseek_v32_motivation.src.operator_mfu import (
    bind_scopes,
    invocation_row,
    primary_kernel,
    summarize,
)


def call(**updates):
    return {
        "capture_index": 1,
        "scheme": "hbm",
        "phase": "cold",
        "segment": "history",
        "stage": "index_weights_proj",
        "precision": "FP32",
        "chunk": "0",
        "layer": "0",
        "call_id": 1,
        "nvtx": "matrix",
        "useful_flops": 3_350_000,
        "executed_matmul_flops": None,
        "dimensions": {"M": 1, "N": 1, "K": 1},
        **updates,
    }


def test_helper_inherits_matrix_owner_and_gpu_activity_can_execute_after_cpu_scope():
    matrix = call()
    scopes = [
        {"id": 0, "thread": 0, "start": 10, "end": 30, "label": "matrix"},
        {"id": 1, "thread": 0, "start": 11, "end": 20, "label": "helper"},
    ]
    owners = bind_scopes(scopes, [matrix])
    assert owners == {0: matrix, 1: matrix}
    activities = [
        {
            "kind": "kernel",
            "name": "sm80_xmma_gemm_execute_kernel",
            "start": 100,
            "end": 200,
            "device_id": 0,
        },
        {
            "kind": "kernel",
            "name": "cublasLt::splitKreduce_kernel",
            "start": 300,
            "end": 400,
            "device_id": 0,
        },
    ]
    row = invocation_row(matrix, activities, {"fp32": 67})
    assert row["primary_kernel_ns"] == 100
    assert row["operator_gpu_active_ns"] == 200
    assert row["operator_gpu_span_ns"] == 300
    assert row["primary_kernel_count"] == 1


def test_split_k_reduction_never_counts_as_matrix_core_despite_gemm_in_template():
    assert not primary_kernel("execute_split_k_kernel<gemm::Gemm> presented by cublas")
    assert not primary_kernel("cublasLt::splitKreduce_kernel")
    assert not primary_kernel("void deep_gemm::transpose_fp32<512, 64>")
    assert primary_kernel("nvjet_sm90_tst_128x8")
    assert primary_kernel("sm90_fp8_mqa_logits_fuse_prefetch")
    assert primary_kernel("void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm>")
    assert primary_kernel("void gemmSN_TN_kernel<float, (int)128>")


def test_summary_uses_ratio_of_total_work_to_total_time_not_mean_percentages():
    activities = [
        {
            "kind": "kernel",
            "name": "sm80_xmma_gemm_execute_kernel",
            "start": 100,
            "end": 200,
            "device_id": 0,
        }
    ]
    first = invocation_row(call(), activities, {"fp32": 67})
    second = invocation_row(
        call(call_id=2, layer="1"), [{**activities[0], "end": 1000}], {"fp32": 67}
    )
    row = summarize([first, second])[0]
    assert row["primary_kernel_mfu_pct"] == pytest.approx(10)
    assert row["executed_matmul_flops"] is None
    assert row["calls"] == 2
    assert len(summarize([first, second], by_layer=True)) == 2


def test_duplicate_or_missing_call_label_fails_instead_of_partial_mfu():
    with pytest.raises(ValueError, match="unique"):
        bind_scopes([], [call(), call()])
    with pytest.raises(ValueError, match="differ"):
        bind_scopes([], [call()])
