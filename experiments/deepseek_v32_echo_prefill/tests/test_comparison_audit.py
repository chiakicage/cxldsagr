"""Saved-run comparison rejects changed contracts and misleading summary arithmetic."""

import copy
import json

import pytest
import torch

from experiments.deepseek_v32_echo_prefill.src.compare_backends import (
    CACHE_FIELDS,
    CORRECTNESS_FIELDS,
    DEFAULT_BOUNDARY,
    DEFAULT_PURPOSE,
    compare,
    digest,
)
from experiments.deepseek_v32_echo_prefill.src.postrun_audit import audit_run


def write_json(path, value):
    path.write_text(json.dumps(value))


def make_run(directory, *, changed=False, cache_contract=True):
    directory.mkdir()
    write_json(directory / "request.json", {"input_ids": list(range(11))})
    tensors = {"hidden": torch.ones(3, 4), "logits": torch.ones(1, 7)}
    if changed:
        tensors["hidden"][2, 3] += 0.5
        tensors["logits"][0, 4] += 0.5
    for name in (
        "resident_control",
        "offload_control",
        "resident_profile_output",
        "offload_profile_output",
    ):
        torch.save(tensors, directory / (name + ".pt"))
    for layer in range(3):
        indices = torch.tensor([[0, 1], [1, 2], [3, 5]], dtype=torch.int32)
        if changed:
            indices[1, 1] = 3
        torch.save(
            {"indices": indices, "source_run_id": directory.name, "layer": layer, "query_start": 8},
            directory / f"kernel_inputs_layer_{layer}.pt",
        )
    run = {
        "run_id": directory.name,
        "accepted": True,
        "scope": "three_layer_fixture",
        "num_layers": 3,
        "prefix_tokens": 8,
        "extend_tokens": 3,
        "chunk_size": 4,
        "slots": 4,
        "warmups": 1,
        "repeats": 2,
        "prefill_repeats": 2,
        "request_sha256": digest(directory / "request.json"),
        "checkpoint_metadata_sha256": {"config.json": "fixture"},
        "source_sha256": {},
        "hardware": {"gpu": {"uuid": "GPU-fixture-5"}},
        "correctness": {
            name: {
                "bitwise_equal": True,
                "max_abs": 0,
                "relative_l2": 0,
                "rtol": 0.01,
                "atol": 0.02,
                "shape": [3, 4] if name.endswith("hidden") else [1, 7],
            }
            for name in CORRECTNESS_FIELDS
        },
        "measurements": {
            mode: {
                "prefix_samples_ms": [5, 7],
                "prefix_median_ms": 6,
                "extend_samples_ms": [2, 4],
                "extend_median_ms": 3,
            }
            for mode in ("resident", "offload")
        },
    }
    if cache_contract:
        run.update({name: "same_fixture_value" for name in CACHE_FIELDS})
        for measurement in run["measurements"].values():
            measurement["cache_resource_plan"] = {"hbm": 2048, "dram": 8192}
            measurement["extend_cache_per_layer"] = [
                {
                    "prefetched_records": 3 if changed else 2,
                    "recalled_records": 5 if changed else 4,
                    "selection_records": 10,
                    "resident_selection_records": 5 if changed else 6,
                    "capacity_splits": 1 if changed else 0,
                }
                for _ in range(3)
            ]
    write_json(directory / "result.json", run)
    return run


def test_comparison_discloses_all_outputs_and_selection_drift(tmp_path):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    make_run(control)
    make_run(candidate, changed=True)
    report = compare(
        control,
        candidate,
        tmp_path / "report",
        purpose="nonmatrix package",
        boundary="same GPU 5, sequential",
        require_same_gpu=True,
    )
    assert (
        report["purpose"] == "nonmatrix package" and report["boundary"] == "same GPU 5, sequential"
    )
    assert report["hardware_comparison"]["same_physical_gpu"]
    assert not report["cache_fields_absent_in_both_historical_runs"]
    for mode in ("resident", "offload"):
        for field, count in (("hidden", 12), ("logits", 7)):
            row = report["cross_backend_numerics_by_mode"][mode][field]
            assert row["elements_compared"] == count
            assert (
                row["elements_outside_original_tolerance"]
                == row["rows_outside_original_tolerance"]
                == 1
            )
            assert row["max_abs"] == 0.5 and not row["allclose_at_original_tolerance"]
    for layer in report["selection_drift"]["layers"]:
        assert layer["intersection_count_by_query"] == [2, 1, 2]
        assert layer["mean_intersection_fraction_of_slots"] == pytest.approx(5 / 6)
        assert layer["min_intersection_fraction_of_slots"] == 0.5
        assert layer["identical_selection_set_row_fraction"] == pytest.approx(2 / 3)
    for mode in ("resident", "offload"):
        metrics = report["extend_cache_metric_comparison"]["modes"][mode]
        assert metrics["available"]
        for layer in metrics["layers"]:
            assert layer["candidate_minus_control"]["prefetched_records"] == 1
            assert layer["candidate_minus_control"]["capacity_splits"] == 1
            assert layer["ensure_resident_selection_fraction"] == {"control": 0.6, "candidate": 0.5}


def test_legacy_comparison_defaults_remain_available(tmp_path):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    make_run(control, cache_contract=False)
    make_run(candidate, cache_contract=False)
    report = compare(control, candidate, tmp_path / "report")
    assert (report["purpose"], report["boundary"]) == (DEFAULT_PURPOSE, DEFAULT_BOUNDARY)
    assert set(report["cache_fields_absent_in_both_historical_runs"]) == set(CACHE_FIELDS)


@pytest.mark.parametrize("field", CACHE_FIELDS)
def test_comparison_rejects_cache_contract_mismatch_or_one_sided_omission(tmp_path, field):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    make_run(control)
    run = make_run(candidate)
    for missing in (False, True):
        changed = copy.deepcopy(run)
        if missing:
            changed.pop(field)
        else:
            changed[field] = "changed"
        write_json(candidate / "result.json", changed)
        with pytest.raises(ValueError, match=field):
            compare(control, candidate, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("failure", ["gpu", "plan", "internal", "repeat"])
def test_comparison_rejects_incompatible_or_unaccepted_runs(tmp_path, failure):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    make_run(control)
    run = make_run(candidate)
    if failure == "gpu":
        run["hardware"]["gpu"]["uuid"] = "GPU-other"
    elif failure == "plan":
        run["measurements"]["offload"]["cache_resource_plan"]["hbm"] += 1
    elif failure == "internal":
        run["correctness"]["resident_vs_offload_hidden"]["bitwise_equal"] = False
    else:
        run["measurements"]["resident"]["prefix_samples_ms"] = [6]
    write_json(candidate / "result.json", run)
    with pytest.raises(ValueError):
        compare(control, candidate, tmp_path / "report", require_same_gpu=True)


def make_analysis(directory, run):
    calls, rows, pooled, captures, inventory = [], [], [], [], []
    for mode in ("resident", "offload"):
        for phase in ("prefill_annotated", "extend_annotated"):
            for stage in ("matrix", "scalar"):
                for layer in range(3):
                    identity = {
                        "mode": mode,
                        "phase": phase,
                        "layer": f"layer_{layer}",
                        "stage": stage,
                    }
                    matrix = stage == "matrix"
                    call = {
                        **identity,
                        "useful_flops": 100_000 if matrix else None,
                        "precision": "FP8" if matrix else None,
                        "formula": "2*M*N*K" if matrix else "N/A (no matrix multiply)",
                    }
                    calls.append(call)
                    row = {
                        **identity,
                        "useful_flops": call["useful_flops"],
                        "precision": call["precision"],
                        "scope_count": 1,
                        "metadata_call_count": 1,
                        "metadata_call_count_matches": True,
                        "kernel_mfu_percent": 20.0 if matrix else None,
                        "dense_peak_tflops": 10 if matrix else None,
                        "api_count": 1,
                    }
                    for kind in ("kernel", "memcpy", "memset"):
                        row.update(
                            {
                                kind + "_count": int(kind == "kernel"),
                                kind + "_ns": 50 if kind == "kernel" else 0,
                                kind + "_ms": 0.00005 if kind == "kernel" else 0,
                            }
                        )
                    rows.append(row)
                    inventory.append(
                        {
                            **identity,
                            "count": 1,
                            "total_ns": 50,
                            "total_ms": 0.00005,
                            "unattributed_reason": None,
                        }
                    )
                aggregate = {
                    **row,
                    "layer": "all_layers",
                    "scope_count": 3,
                    "metadata_call_count": 3,
                    "api_count": 3,
                    "kernel_count": 3,
                    "kernel_ns": 150,
                    "kernel_ms": 0.00015,
                    "useful_flops": 300_000 if matrix else None,
                }
                pooled.append(aggregate)
            totals = {
                kind: {
                    "count": 6 if kind == "kernel" else 0,
                    "attributed_count": 6 if kind == "kernel" else 0,
                    "unattributed_count": 0,
                    "total_ns": 300 if kind == "kernel" else 0,
                    "attributed_ns": 300 if kind == "kernel" else 0,
                    "unattributed_ns": 0,
                }
                for kind in ("kernel", "memcpy", "memset")
            }
            captures.append(
                {
                    "mode": mode,
                    "phase": phase,
                    "audit": {
                        "kernel_count_and_time_conserved": True,
                        "metadata_call_counts_match": True,
                        "metadata_call_count_mismatches": [],
                        "layer_unscoped_kernel_count": 0,
                        "activity_counts_and_ns": totals,
                        "operator_kernel_ns": 300,
                        "inventory_kernel_ns": 300,
                    },
                    "api_summary": {"non_event_api_count": 6},
                    "devices": [
                        {
                            "activity_count": 6,
                            "attributed_activity_count": 6,
                            "activity_duration_sum_ms": 0.0003,
                            "start_ns": 0,
                            "end_ns": 1000,
                            "span_ms": 0.001,
                            "busy_ms": 0.0003,
                            "gap_ms": 0.0007,
                            "overlap_ms": 0,
                            "kernel_busy_ms": 0.0003,
                        }
                    ],
                    "capture_gpu_envelope": {"span_ms": 0.001},
                }
            )
    write_json(directory / "operator_calls.json", {"run_id": run["run_id"], "calls": calls})
    analysis = {
        "ledger_metadata": {"run_id": run["run_id"]},
        "calls_sha256": digest(directory / "operator_calls.json"),
        "calls_outside_selected_captures": 0,
        "dense_peaks_tflops": {"FP8": 10},
        "operators_by_layer": rows,
        "operators": pooled,
        "captures": captures,
        "kernel_inventory": inventory,
    }
    (directory / "analysis").mkdir()
    write_json(directory / "analysis/analysis.json", analysis)
    return analysis


def test_light_postrun_audit_rechecks_saved_tensors_and_summary_arithmetic(tmp_path):
    directory = tmp_path / "run"
    run = make_run(directory)
    make_analysis(directory, run)
    report = audit_run(directory)
    assert report["accepted"] and report["summary_conservation"]["kernel_count"] == 24
    assert len(report["saved_output_checks"]["tensor_checks"]) == 8
    assert len(report["saved_output_checks"]["runtime_only_prefix_checks"]) == 2


@pytest.mark.parametrize("failure", ["inventory", "busy_gap", "pooled", "tensor", "ledger_hash"])
def test_light_audit_rejects_tampering_even_if_analyzer_flags_still_pass(tmp_path, failure):
    directory = tmp_path / "run"
    run = make_run(directory)
    analysis = make_analysis(directory, run)
    if failure == "inventory":
        analysis["kernel_inventory"][0]["count"] += 1
    elif failure == "busy_gap":
        analysis["captures"][0]["devices"][0]["gap_ms"] += 0.1
    elif failure == "pooled":
        analysis["operators"][0]["kernel_count"] += 1
    elif failure == "ledger_hash":
        analysis["calls_sha256"] = "changed"
    else:
        tensors = torch.load(directory / "offload_control.pt", weights_only=True)
        tensors["hidden"][0, 0] += 1
        torch.save(tensors, directory / "offload_control.pt")
    write_json(directory / "analysis/analysis.json", analysis)
    with pytest.raises(ValueError):
        audit_run(directory)
