"""End-to-end utilization cannot reuse operator time or extrapolate layer work."""

from copy import deepcopy

import pytest

from experiments.deepseek_v32_echo_prefill.src.execution_utilization import (
    precision_normalized_utilization,
)
from experiments.deepseek_v32_echo_prefill.src.publish_layers import _end_to_end_utilization

PEAKS = {"FP8": 2000.0, "BF16": 1000.0, "FP32": 60.0}
SCOPE = "checkpoint_layers_0_1_2_embedding_final_norm_last_token_lm_head"


def matrix_call(precision="FP8", flops=100_000_000_000, **extra):
    return {
        "mode": "resident",
        "phase": "extend_annotated",
        "layer": "layer_0",
        "stage": "linear",
        "precision": precision,
        "useful_flops": flops,
        "executed_matmul_flops": None if flops is None else flops * 2,
        **extra,
    }


def test_mixed_precision_uses_ideal_time_and_complete_wall_for_every_sample():
    calls = [
        matrix_call(),
        matrix_call("BF16"),
        matrix_call(None, None, formula="N/A (no matrix multiply)", executed_matmul_flops=None),
    ]
    result = precision_normalized_utilization(calls, [1.0, 3.0], peaks_tflops=PEAKS, scope=SCOPE)
    assert result["ideal_compute_ms_by_precision"] == {"FP8": 0.05, "BF16": 0.1}
    assert result["ideal_compute_ms"] == pytest.approx(0.15)
    assert result["utilization_samples_percent"] == pytest.approx([15.0, 5.0])
    # A ratio at median latency differs from the median ratio for an even sample count.
    assert result["wall_median_ms"] == 2.0
    assert result["utilization_at_median_wall_percent"] == pytest.approx(7.5)
    assert result["matrix_call_count"] == 2
    assert result["explicit_nonmatrix_call_count"] == 1
    assert sum(result["useful_flops_by_precision"].values()) == 200_000_000_000


def test_unknown_work_and_missing_precision_peak_are_rejected():
    for call in (
        matrix_call(None, None, executed_matmul_flops=None),
        matrix_call("BF16", None, formula="N/A (no matrix multiply)", executed_matmul_flops=None),
    ):
        with pytest.raises(ValueError, match="Unknown work"):
            precision_normalized_utilization([call], [1.0], peaks_tflops=PEAKS, scope=SCOPE)
    with pytest.raises(ValueError, match="dense peak.*FP64"):
        precision_normalized_utilization(
            [matrix_call("FP64")], [1.0], peaks_tflops=PEAKS, scope=SCOPE
        )


@pytest.mark.parametrize("samples", [[], [0], [-1], [float("nan")], [float("inf")], [True]])
def test_wall_time_must_be_a_positive_measured_sample(samples):
    with pytest.raises(ValueError, match="wall samples"):
        precision_normalized_utilization([matrix_call()], samples, peaks_tflops=PEAKS, scope=SCOPE)


def test_prefix_and_extend_or_cache_modes_cannot_be_summed_into_one_execution():
    for other in (matrix_call(phase="prefill_annotated"), matrix_call(mode="offload")):
        with pytest.raises(ValueError, match="one mode and phase"):
            precision_normalized_utilization(
                [matrix_call(), other], [1.0], peaks_tflops=PEAKS, scope=SCOPE
            )


def complete_fixture():
    result = {"num_layers": 3, "scope": SCOPE, "measurements": {}}
    calls = []
    for mode in ("resident", "offload"):
        result["measurements"][mode] = {"prefix_samples_ms": [10.0], "extend_samples_ms": [1.0]}
        for phase in ("prefill_annotated", "extend_annotated"):
            for layer in range(3):
                for stage in (
                    "q_a_proj",
                    "q_b_proj",
                    "q_absorb",
                    "kv_a_proj",
                    "index_q_proj",
                    "index_k_proj",
                    "index_weights_proj",
                    "mla_qk_pv",
                    "v_expand",
                    "o_proj",
                    "mlp_gate",
                    "mlp_up",
                    "mlp_down",
                    "indexer_qk" if mode == "resident" else "indexer_fused",
                ):
                    calls.append(
                        matrix_call(mode=mode, phase=phase, layer=f"layer_{layer}", stage=stage)
                    )
            calls.append(
                matrix_call("BF16", mode=mode, phase=phase, layer="shared", stage="lm_head")
            )
    return result, {"calls": calls}


def test_publication_binds_each_phase_and_mode_without_extrapolating():
    result, ledger = complete_fixture()
    metrics = _end_to_end_utilization(result, ledger, PEAKS)
    assert metrics["resident"]["extend"]["matrix_call_count"] == 43
    assert metrics["resident"]["extend"]["ideal_compute_ms"] == pytest.approx(2.2)
    assert metrics["resident"]["prefix"]["utilization_at_median_wall_percent"] == pytest.approx(22)
    assert metrics["offload"]["extend"]["scope"] == SCOPE
    result["num_layers"] = 61
    with pytest.raises(ValueError, match="three-layer single-device"):
        _end_to_end_utilization(result, ledger, PEAKS)


def test_publication_rejects_missing_stage_or_non_equivalent_mode_work():
    result, original = complete_fixture()
    ledger = deepcopy(original)
    ledger["calls"].pop(0)
    with pytest.raises(ValueError, match="Incomplete"):
        _end_to_end_utilization(result, ledger, PEAKS)
    ledger = deepcopy(original)
    ledger["calls"][0]["useful_flops"] *= 2
    with pytest.raises(ValueError, match="semantic equivalence"):
        _end_to_end_utilization(result, ledger, PEAKS)


@pytest.mark.parametrize("resident_only", [False, True])
def test_offload_indexer_dispatch_preserves_complete_work(resident_only):
    result, ledger = complete_fixture()
    expected = _end_to_end_utilization(result, ledger, PEAKS)
    for call in list(ledger["calls"]):
        if (
            call["mode"] == "offload"
            and call["phase"] == "prefill_annotated"
            and call["stage"] == "indexer_fused"
        ):
            if resident_only:
                call["stage"] = "indexer_qk"
            else:
                call["useful_flops"] //= 2
                call["executed_matmul_flops"] //= 2
                ledger["calls"].append({**call, "stage": "indexer_qk"})
    actual = _end_to_end_utilization(result, ledger, PEAKS)
    for field in ("ideal_compute_ms", "useful_flops_by_precision", "utilization_samples_percent"):
        assert actual["offload"]["prefix"][field] == expected["offload"]["prefix"][field]


@pytest.mark.parametrize("stage", ["indexer_fused", "unknown_indexer"])
def test_resident_publication_rejects_unsupported_indexer_dispatch(stage):
    result, ledger = complete_fixture()
    call = next(row for row in ledger["calls"] if row["stage"] == "indexer_qk")
    call["stage"] = stage
    with pytest.raises(ValueError, match="Incomplete or unsupported matrix ledger"):
        _end_to_end_utilization(result, ledger, PEAKS)


def test_utilization_retains_independent_benchmark_denominator_identity():
    result, ledger = complete_fixture()
    result["wall_time_denominator"] = {
        "mode": "independent_bench",
        "run_id": "clean-bench",
        "source_sha256": {"model.py": "source-digest"},
        "result_sha256": "bench-digest",
    }
    metrics = _end_to_end_utilization(result, ledger, PEAKS)
    for mode in ("resident", "offload"):
        for phase in ("prefix", "extend"):
            assert metrics[mode][phase]["wall_time_denominator"] == result["wall_time_denominator"]
