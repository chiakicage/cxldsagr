from experiments.deepseek_v32_motivation.src.report_operator_mfu import render_report


def test_graph_report_uses_current_counts_and_weighted_mixed_indexer_dispatch():
    summary = {
        "profile_run_id": "current_graph_profile",
        "formal_run_id": "current_graph_formal",
        "analysis_run_id": "current_graph_analysis",
        "matrix_calls": 7,
        "captures": [{"graph_attribution": {"replays": 4, "gpu_node_activities": 88}}],
        "dense_peaks_tflops": {"fp8": 1979, "bf16": 989.5, "fp32": 67},
    }
    rows = [
        {
            "scheme": "echo",
            "phase": "cold",
            "segment": "history",
            "stage": stage,
            "precision": "FP8",
            "ideal_ms": ideal,
            "operator_gpu_active_ms": duration,
        }
        for stage, ideal, duration in (("indexer_qk", 1, 2), ("indexer_fused", 1, 8))
    ]
    calls_path = "output/data/current_graph_profile/analysis/operator_mfu/operator_mfu_calls.jsonl"
    report = render_report(summary, rows, calls_path=calls_path)
    assert "current_graph_profile" in report and "7 次矩阵调用" in report
    assert "4 次 CUDA Graph 重放" in report and "88 个 GPU 节点活动" in report
    assert "20.00%" in report
    assert "45,933" not in report and "20261004_01" not in report
    assert calls_path in report
    assert "output/data/current_graph_analysis/operator_mfu_calls.jsonl" not in report
