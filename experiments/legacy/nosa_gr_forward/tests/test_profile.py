"""Check disjoint GPU attribution and the FLOP/time units used by MFU reports."""

import json

import pytest

from experiments.legacy.nosa_gr_forward.src.profile import attribute_trace, module_summary


def test_attribution_uses_innermost_cpu_scope_and_unique_gpu_activities(tmp_path):
    events = [
        {
            "cat": "user_annotation",
            "name": "nosa::full_prefill/model_misc/shared",
            "pid": 1,
            "tid": 1,
            "ts": 0,
            "dur": 100,
        },
        {
            "cat": "user_annotation",
            "name": "nosa::full_prefill/q_proj/0",
            "pid": 1,
            "tid": 1,
            "ts": 10,
            "dur": 20,
        },
        {"cat": "cuda_runtime", "pid": 1, "tid": 1, "ts": 15, "args": {"correlation": 7}},
        {"cat": "kernel", "name": "gemm", "ts": 200, "dur": 5, "args": {"correlation": 7}},
        {"cat": "cuda_runtime", "pid": 1, "tid": 1, "ts": 40, "args": {"correlation": 8}},
        {"cat": "gpu_memcpy", "name": "copy", "ts": 210, "dur": 2, "args": {"correlation": 8}},
        # GPU annotation is a duplicate overview, not another kernel.
        {"cat": "gpu_user_annotation", "name": "nosa::full_prefill/q_proj/0", "ts": 200, "dur": 5},
    ]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    times, counts, names = attribute_trace(path)
    assert dict(times) == {"full_prefill/q_proj/0": 5, "full_prefill/model_misc/shared": 2}
    assert sum(counts.values()) == 2
    assert names["full_prefill/q_proj/0"] == {"gemm": 1}


def test_unattributed_gpu_work_is_not_silently_dropped(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "kernel", "name": "unknown", "dur": 10, "args": {"correlation": 99}}
                ]
            }
        )
    )
    with pytest.raises(RuntimeError, match="Unattributed"):
        attribute_trace(path)


def test_mfu_aggregates_flops_over_total_time_and_marks_memory_work_na():
    rows = [
        {
            "phase": "candidate_extend",
            "module": "q_proj",
            "gpu_us": 10,
            "flops": 2e9,
            "gpu_activities": 1,
        },
        {
            "phase": "candidate_extend",
            "module": "q_proj",
            "gpu_us": 30,
            "flops": 2e9,
            "gpu_activities": 1,
        },
        {
            "phase": "candidate_extend",
            "module": "embedding",
            "gpu_us": 4,
            "flops": 0,
            "gpu_activities": 1,
        },
    ]
    gemm, memory = module_summary(rows, 1000)
    assert gemm["tflops"] == 100
    assert gemm["mfu_pct"] == 10
    assert memory["mfu_pct"] is None
