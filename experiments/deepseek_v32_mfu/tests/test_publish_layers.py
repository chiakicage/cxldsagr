"""Report evidence scope and unmodified runtime exception propagation."""

import pytest

from experiments.deepseek_v32_mfu.src import publish_layers


@pytest.mark.parametrize("error_type", [OSError, ValueError, KeyError, RuntimeError])
def test_cli_preserves_original_runtime_error(monkeypatch, error_type):
    original = error_type("publication could not be verified")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(publish_layers, "generate", fail)
    with pytest.raises(error_type) as raised:
        publish_layers.main(["--run-id", "profile-fixture"])
    assert raised.value is original


def test_cli_preserves_original_error_group(monkeypatch):
    original = ValueError("publication identity differs")
    cleanup = OSError("publication copy cleanup failed")
    group = BaseExceptionGroup("publication and cleanup", [original, cleanup])

    def fail(*args, **kwargs):
        raise group

    monkeypatch.setattr(publish_layers, "generate", fail)
    with pytest.raises(BaseExceptionGroup) as raised:
        publish_layers.main(["--run-id", "profile-fixture"])
    assert raised.value is group
    assert raised.value.exceptions[0] is original
    assert raised.value.exceptions[1] is cleanup


def report_summary(ncu_runs):
    return {
        "run_id": "profile-fixture",
        "prefix_tokens": 65536,
        "extend_tokens": 1024,
        "chunk_size": 1024,
        "slots": 16384,
        "wall_time_denominator": {
            "run_id": "bench-fixture",
            "result_sha256": "0" * 64,
        },
        "measurements": {
            mode: {
                "prefix_median_ms": 2.0,
                "prefix_samples_ms": [2.0],
                "extend_median_ms": 1.0,
                "extend_samples_ms": [1.0],
            }
            for mode in ("resident", "offload")
        },
        "end_to_end_utilization": {
            mode: {
                phase: {"ideal_compute_ms": 0.1, "utilization_at_median_wall_percent": 5.0}
                for phase in ("prefix", "extend")
            }
            for mode in ("resident", "offload")
        },
        "dense_peaks_tflops": {"FP8": 1979.0, "BF16": 989.5, "FP32": 67.0},
        "ncu": [{"run_id": run_id} for run_id in ncu_runs],
    }


def test_report_without_ncu_does_not_claim_ncu_verification():
    report = publish_layers._markdown(report_summary([]), [], [])
    assert "本次未采集 NCU replay" in report
    assert "完整 NCU full/source 的验证记录" not in report
    assert "NCU 是同 run" not in report
    assert "NCU run IDs：" not in report
    assert "--ncu-run-id" not in report
    assert "独立 bench `bench-fixture`" in report


def test_report_with_ncu_names_only_selected_replays():
    report = publish_layers._markdown(report_summary(["ncu-a", "ncu-b"]), [], [])
    assert "本次未采集 NCU replay" not in report
    assert "完整 NCU full/source 的验证记录" in report
    assert "NCU run IDs：`ncu-a`, `ncu-b`" in report
    assert "--ncu-run-id ncu-a --ncu-run-id ncu-b" in report
    assert "独立 bench `bench-fixture`" in report


def test_report_preserves_both_offload_indexer_dispatch_rows():
    rows = [
        {
            "mode": "offload",
            "phase": "prefill_annotated",
            "stage": stage,
            "precision": "FP8",
            "kernel_ms": duration,
            "kernel_mfu_percent": utilization,
        }
        for stage, duration, utilization in (
            ("indexer_qk", 1.25, 25.0),
            ("indexer_fused", 7.5, 75.0),
        )
    ]
    report = publish_layers._markdown(report_summary([]), rows, [])
    assert "| indexer_qk | FP8 | N/A | N/A | 1.250 | 25.00 |" in report
    assert "| indexer_fused | FP8 | N/A | N/A | 7.500 | 75.00 |" in report
