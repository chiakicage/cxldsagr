"""CPU checks for shape coverage, arithmetic, and immutable publication bindings."""

import copy
import csv
import json
from pathlib import Path

import pytest

from evaluation.validation import write_receipt
from experiments.deepseek_v32_observation.src import contract, publish
from experiments.deepseek_v32_observation.src.aggregate import merge_profile
from experiments.deepseek_v32_observation.src.bandwidth import (
    HBM_FIELDS,
    METRICS,
    from_metrics,
    validate_bandwidth,
)


def _fixture(tmp_path):
    groups = {}
    for group in contract.GROUPS:
        entries = []
        for key in contract.task_keys(group):
            shape = next((shape for shape in contract.matrix() if shape.key == key), None)
            size = shape.S if shape else int(key[1:])
            operators = []
            for name, work in (
                (
                    "indexer" if group == "attention" else "projection",
                    {
                        "precision": "fp8",
                        "useful_flops": 1000000 * size,
                        "formula": "fixture dot products",
                    },
                ),
                (
                    "topk" if group == "attention" else "norm",
                    {"precision": None, "useful_flops": None, "formula": "N/A"},
                ),
            ):
                if group == "moe" and name == "projection":
                    work = {
                        "precision": "mixed",
                        "useful_flops": 1000000 * size,
                        "formula": "fixture FP8 and FP32 dot products",
                        "dimensions": {"fp8_flops": 900000 * size, "fp32_flops": 100000 * size},
                    }
                samples = [8.0, 9.0, 11.0]
                summary = contract.statistics(samples)
                operators.append(
                    {
                        "name": name,
                        "scope": "module_aggregate" if work["precision"] == "mixed" else "operator",
                        "work": work,
                        "metadata": {
                            "precision_boundary": "fixture",
                            "operator_kind": "api_aggregate"
                            if work["precision"] == "mixed"
                            else "gemm"
                            if work["precision"]
                            else "non_gemm",
                            "operator_family": "projection" if work["precision"] else "norm",
                            "parent_api": "fixture_api",
                        },
                        "samples_us": samples,
                        **summary,
                        **contract.mfu(work, summary["time_us"]),
                        "timing_boundary": "external CUDA events around complete API",
                        "profiler_enabled": False,
                    }
                )
            identity = {
                "group": group,
                "key": key,
                "input": {"seed": 42, "shape": key},
                "backend_sha256": "fixture-backend",
                "source_sha256": {"fixture.py": "fixture-source"},
                "operators": [
                    {field: operator[field] for field in ("name", "work", "metadata")}
                    for operator in operators
                ],
            }
            directory = tmp_path / "checks" / group / key
            directory.mkdir(parents=True)
            checks = directory / "checks.json"
            checks.write_text(json.dumps({"passed": True}))
            runtime = directory / "runtime.json"
            runtime.write_text(
                json.dumps(
                    {
                        "backend_sha256": "fixture-backend",
                        "native_runtime": {},
                        "uuid": "fixture-gpu",
                    }
                )
            )
            receipt = write_receipt(
                directory / "receipt.json",
                kind=f"deepseek-v32-observation-{group}",
                identity=identity,
                checks={"passed": True},
                artifacts={"checks": checks, "runtime": runtime},
            )
            entries.append(
                {
                    "key": key,
                    "group": group,
                    "S": size,
                    "B": shape.B if shape else None,
                    "Q": shape.Q if shape else None,
                    "H": shape.H if shape else None,
                    "identity": identity,
                    "runtime": {"backend_sha256": "fixture-backend", "uuid": "fixture-gpu"},
                    "receipt_path": receipt["receipt_path"],
                    "operators": operators,
                }
            )
        groups[group] = entries
    result = {
        "schema": contract.SCHEMA,
        "run_id": "fixture_observation",
        "offload": False,
        "matrix": [shape.as_dict() for shape in contract.matrix()],
        "peaks_tflops": contract.PEAKS_TFLOPS,
        "peak_source": contract.PEAK_SOURCE,
        "groups": groups,
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(result))
    for group, entries in groups.items():
        for entry in entries:
            directory = tmp_path / group / entry["key"]
            directory.mkdir(parents=True)
            (directory / "result.json").write_text(json.dumps(entry))
            profile = {
                field: copy.deepcopy(entry[field])
                for field in ("key", "group", "identity", "receipt_path", "runtime")
            }
            profile["runtime"]["native_runtime"] = {}
            profile["operators"] = []
            for operator in entry["operators"]:
                artifact = directory / f"{operator['name']}.ncu-rep"
                artifact.write_bytes(b"CPU fixture profile bytes")
                bandwidth = from_metrics(
                    {
                        METRICS[0]: 180000.0,
                        METRICS[1]: 20000.0,
                        METRICS[2]: 20000.0,
                        METRICS[3]: 2e12,
                        METRICS[4]: 0.5,
                        "profiler__replayer_passes": 1,
                    },
                    inner=2,
                    profile_path=artifact,
                    profile_sha256=publish.digest(artifact),
                    replay_mode="kernel-graph",
                    time_boundary="NCU application range GPU time per call",
                )
                assert set(bandwidth) == set(HBM_FIELDS)
                profile["operators"].append({"name": operator["name"], **bandwidth})
                operator.update(bandwidth)
            (directory / "hbm.json").write_text(json.dumps(profile))
    results_path.write_text(json.dumps(result))
    return result, results_path


def test_frozen_matrix_distinguishes_sequence_batch_and_query_count():
    shapes = contract.matrix()
    assert len(shapes) == len({shape.key for shape in shapes}) == 52
    assert {shape.H for shape in shapes} == {4096, 16384, 32768, 65536}
    assert {shape.B for shape in shapes if shape.Q == 1} == {1, 32, 64, 128, 256, 512, 1024}
    assert {shape.Q for shape in shapes if shape.B == 1} == {1, 32, 64, 128, 256, 512, 1024}
    assert not any(shape.B > 1 and shape.Q > 1 for shape in shapes)
    assert contract.Shape(128, 1, 4096).S == contract.Shape(1, 128, 4096).S
    assert contract.Shape(128, 1, 4096).key != contract.Shape(1, 128, 4096).key


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_shape_rejects_nonpositive_or_noninteger_dimensions(value):
    with pytest.raises(ValueError):
        contract.Shape(value, 1, 4096)


def test_matrix_mfu_uses_precision_peak_and_nonmatrix_stays_na():
    flops = 1979 * 1000000
    fp8 = contract.mfu({"precision": "fp8", "useful_flops": flops}, 10.0)
    bf16 = contract.mfu({"precision": "bf16", "useful_flops": flops}, 10.0)
    assert fp8["mfu_pct"] == 10.0
    assert bf16["mfu_pct"] == 20.0
    assert contract.mfu({"precision": None, "useful_flops": None}, 10.0)["mfu_pct"] is None
    with pytest.raises(ValueError, match="peak"):
        contract.mfu({"precision": "tf32", "useful_flops": flops}, 10.0)


def test_mixed_mfu_uses_sum_of_precision_specific_ideal_times():
    breakdown = {"fp8_flops": 1979 * 1000000, "fp32_flops": 67 * 1000000}
    work = {"precision": "mixed", "useful_flops": sum(breakdown.values()), "dimensions": breakdown}
    result = contract.mfu(work, 10.0)
    assert result["mfu_pct"] == 20.0
    assert result["peak_tflops"] is None
    assert result["precision_peaks_tflops"] == {"fp8": 1979.0, "fp32": 67.0}
    assert result["precision_flops"] == {
        "fp8": breakdown["fp8_flops"],
        "fp32": breakdown["fp32_flops"],
    }
    with pytest.raises(ValueError, match="breakdown"):
        contract.mfu({**work, "useful_flops": work["useful_flops"] + 1}, 10.0)


def test_conditional_common_inventory_is_supported_only_for_s1(tmp_path):
    result, _ = _fixture(tmp_path)
    entry = result["groups"]["common"][0]
    conditional = copy.deepcopy(entry["operators"][1])
    conditional["name"] = "q1_consumer_layout"
    conditional["metadata"]["supported_condition"] = "S == 1"
    entry["operators"].append(conditional)
    contract.validate_result(result)
    result["groups"]["common"][1]["operators"].append(conditional)
    with pytest.raises(ValueError, match="support condition"):
        contract.validate_result(result)


@pytest.mark.parametrize(
    "change",
    ["missing_shape", "duplicate_shape", "offload", "wrong_S", "wrong_matrix", "wrong_peak"],
)
def test_publication_rejects_incomplete_or_mislabeled_workload(tmp_path, change):
    result, path = _fixture(tmp_path)
    if change == "missing_shape":
        result["groups"]["attention"].pop()
    elif change == "duplicate_shape":
        result["groups"]["attention"][-1] = copy.deepcopy(result["groups"]["attention"][0])
    elif change == "offload":
        result["offload"] = True
    elif change == "wrong_S":
        result["groups"]["attention"][0]["S"] = 128
    elif change == "wrong_matrix":
        result["matrix"][0]["B"] = 128
    elif change == "wrong_peak":
        result["peaks_tflops"] = {**contract.PEAKS_TFLOPS, "fp8": 3958}
    with pytest.raises(ValueError):
        publish.validate_publication(result, path)


@pytest.mark.parametrize(
    "change", ["input", "operator", "runtime", "receipt_evidence", "time", "mfu", "profiler"]
)
def test_publication_rejects_unaccepted_or_changed_evidence(tmp_path, change):
    result, path = _fixture(tmp_path)
    entry = result["groups"]["common"][0]
    if change == "input":
        entry["identity"]["input"]["seed"] = 123
    elif change == "operator":
        entry["operators"][0]["work"] = {**entry["operators"][0]["work"], "formula": "changed"}
    elif change == "runtime":
        entry["runtime"]["backend_sha256"] = "different-backend"
    elif change == "receipt_evidence":
        (tmp_path / "checks/common/s1/checks.json").write_text("changed checks")
    elif change == "time":
        entry["operators"][0]["time_us"] += 1
    elif change == "mfu":
        entry["operators"][0]["mfu_pct"] += 1
    elif change == "profiler":
        entry["operators"][0]["profiler_enabled"] = True
    with pytest.raises(ValueError):
        publish.validate_publication(result, path)


def test_expansion_preserves_shared_measurement_identity(tmp_path):
    result, path = _fixture(tmp_path)
    evidence = publish.validate_publication(result, path)
    direct, expanded = publish.rows(result, evidence)
    assert len(direct["common"]) == len(direct["moe"]) == 14
    assert len(direct["attention"]) == 104
    assert len(expanded) == 52 * 6
    targets = {
        row["target_shape_id"]
        for row in expanded
        if row["group"] == "common"
        and row["source_case_id"] == "s128"
        and row["name"] == "projection"
    }
    assert targets == {
        contract.Shape(b, q, h).key for b, q in ((128, 1), (1, 128)) for h in contract.H_VALUES
    }
    for row in expanded:
        original = next(
            item
            for item in direct[row["group"]]
            if item["source_case_id"] == row["source_case_id"] and item["name"] == row["name"]
        )
        assert row["time_us"] == original["time_us"]
        assert row["identity_sha256"] == original["identity_sha256"]
        assert row["measurement_kind"] == (
            "direct" if row["group"] == "attention" else f"mapped_{row['group']}"
        )


def test_publication_rejects_aggregate_that_changes_clean_samples(tmp_path):
    result, path = _fixture(tmp_path)
    operator = result["groups"]["common"][0]["operators"][0]
    operator["samples_us"] = [8.0, 10.0, 12.0]
    operator.update(contract.statistics(operator["samples_us"]))
    operator.update(contract.mfu(operator["work"], operator["time_us"]))
    with pytest.raises(ValueError, match="independent clean result"):
        publish.validate_publication(result, path)


def test_publication_rejects_mixed_mfu_with_single_peak(tmp_path):
    result, path = _fixture(tmp_path)
    entry = result["groups"]["moe"][0]
    entry["operators"][0]["peak_tflops"] = 1979.0
    child = tmp_path / "moe" / entry["key"] / "result.json"
    child.write_text(json.dumps(publish.clean_entry(entry)))
    with pytest.raises(ValueError, match="single arithmetic peak"):
        publish.validate_publication(result, path)


def test_publish_keeps_na_values_and_hashes_all_selected_artifacts(tmp_path, monkeypatch):
    _, path = _fixture(tmp_path)
    monkeypatch.setattr(publish, "plot_series", lambda *args, **kwargs: [])
    destination = tmp_path / "report"
    evidence = publish.publish(path, destination)
    provenance = publish.read(destination / "provenance.json")
    assert provenance["results_sha256"] == publish.digest(path)
    assert provenance["coverage"]["workloads"] == {"common": 7, "moe": 7, "attention": 52}
    assert provenance["coverage"]["mapped_records_are_new_measurements"] is False
    assert len(provenance["workloads"]) == 66
    for name, expected in provenance["artifacts_sha256"].items():
        assert publish.digest(destination / name) == expected
    with (destination / "common.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    norm = next(row for row in records if row["name"] == "norm")
    assert norm["mfu_pct"] == norm["useful_flops"] == ""
    assert "映射不新增测量" in (destination / "results.md").read_text()
    assert evidence["source_run_id"] == "fixture_observation"
    with pytest.raises(FileExistsError):
        publish.publish(path, destination)


def test_failed_render_does_not_leave_partial_publication(tmp_path, monkeypatch):
    _, path = _fixture(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("figure failure")

    monkeypatch.setattr(publish, "plot_series", fail)
    with pytest.raises(RuntimeError, match="figure failure"):
        publish.publish(path, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_profile_join_preserves_independent_clean_result(tmp_path):
    result, _ = _fixture(tmp_path)
    entry = result["groups"]["common"][0]
    clean = publish.clean_entry(entry)
    original = copy.deepcopy(clean)
    profile = publish.read(tmp_path / "common/s1/hbm.json")
    merged = merge_profile(clean, profile)
    assert clean == original
    assert merged == entry
    assert publish.clean_entry(merged) == original


@pytest.mark.parametrize("change", ["missing", "duplicate", "identity", "metric", "artifact"])
def test_publication_rejects_unbound_or_incomplete_profile(tmp_path, change):
    result, path = _fixture(tmp_path)
    profile_path = tmp_path / "common/s1/hbm.json"
    profile = publish.read(profile_path)
    if change == "missing":
        profile["operators"].pop()
    elif change == "duplicate":
        profile["operators"][1] = copy.deepcopy(profile["operators"][0])
    elif change == "identity":
        profile["identity"]["input"]["seed"] = 9
    elif change == "metric":
        profile["operators"][0]["hbm_bandwidth_gbps"] += 1
    elif change == "artifact":
        Path(profile["operators"][0]["hbm_profile_path"]).write_bytes(b"different profile")
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        publish.validate_publication(result, path)


def test_profile_denominator_stays_distinct_from_clean_time(tmp_path):
    result, path = _fixture(tmp_path)
    evidence = publish.validate_publication(result, path)
    records, expanded = publish.rows(result, evidence)
    row = records["common"][0]
    assert row["time_us"] == 9.0
    assert row["hbm_profile_time_us"] == 10.0
    assert row["hbm_bandwidth_gbps"] == 10.0
    assert json.loads(row["hbm_metrics"])[METRICS[2]] == 20000.0
    report = publish.render_report(result, records, expanded, ["common_projection_1.png"])
    assert "![Projection](figures/common_projection_1.png)" in report
    assert "时间分母与 clean 时间不同" in report
    assert evidence["workloads"][0]["profile_artifacts"]


def test_functional_figures_keep_projection_aggregates_separate(tmp_path):
    result, path = _fixture(tmp_path)
    evidence = publish.validate_publication(result, path)
    records, _ = publish.rows(result, evidence)
    combined = records["common"] + records["moe"]
    grouped = dict(publish.figure_groups(combined))
    assert {row["operator_kind"] for row in grouped["projection"]} == {"gemm"}
    assert {row["operator_kind"] for row in grouped["aggregate_projection"]} == {"api_aggregate"}
    assert {row["operator_kind"] for row in grouped["norm"]} == {"non_gemm"}


def test_report_inventory_lists_actual_names_precision_parent_and_direct_coverage(tmp_path):
    result, path = _fixture(tmp_path)
    evidence = publish.validate_publication(result, path)
    records, expanded = publish.rows(result, evidence)
    inventory = publish.operator_inventory(records)
    row = next(
        item for item in inventory if item["group"] == "common" and item["name"] == "projection"
    )
    assert row["operator_kind"] == "gemm"
    assert row["precision"] == "fp8"
    assert row["parent_api"] == "fixture_api"
    assert row["coverage"] == "S=1,32,64,128,256,512,1024"
    norm = next(item for item in inventory if item["group"] == "common" and item["name"] == "norm")
    assert norm["precision"] == "N/A"
    attention = next(
        item for item in inventory if item["group"] == "attention" and item["name"] == "indexer"
    )
    assert attention["coverage"] == "52 B/Q/H points"
    report = publish.render_report(result, records, expanded, [])
    assert "| common/projection | `projection` | gemm | fp8 | `fixture_api` |" in report
    assert "## 单独测量的 API 聚合" in report


@pytest.mark.parametrize("metric", [*METRICS, "profiler__replayer_passes"])
def test_bandwidth_rejects_raw_hardware_metric_change_with_unchanged_derived_values(
    tmp_path, metric
):
    result, _ = _fixture(tmp_path)
    operator = result["groups"]["common"][0]["operators"][0]
    original = operator["hbm_metrics"][metric]
    operator["hbm_metrics"][metric] += max(1, abs(original) * 0.01)
    with pytest.raises(ValueError):
        validate_bandwidth(operator)


def test_publication_rejects_profile_gpu_mismatch_from_accepted_runtime(tmp_path):
    result, path = _fixture(tmp_path)
    # Keep profile and clean runtime mutually consistent, but differ from the
    # immutable accepted runtime artifact retained in the numerical receipt.
    entry = result["groups"]["common"][0]
    entry["runtime"]["uuid"] = "different-device"
    child = tmp_path / "common/s1/result.json"
    child.write_text(json.dumps(publish.clean_entry(entry)))
    profile_path = tmp_path / "common/s1/hbm.json"
    profile = publish.read(profile_path)
    profile["runtime"]["uuid"] = "different-device"
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="independently accepted runtime"):
        publish.validate_publication(result, path)
