"""MFU publication binds complete useful work, clean timing, and exact graph coverage."""

import csv
import json
from copy import deepcopy
from itertools import chain

import pytest

from experiments.deepseek_v32_mfu.src import report_full_graph_mfu as report
from experiments.deepseek_v32_mfu.src.publish_layers import STAGES


def hardware_fixture():
    return {
        "gpu": {
            "name": "NVIDIA M403",
            "uuid": "GPU-device-physical",
            "pci.device_id": "0x233510DE",
            "compute_cap": "9.0",
        },
        "is_sm90": True,
        "pci_identity": {"vendor_id": "10de", "device_id": "2335"},
        "h200_sxm_reference_matches_pci": True,
        "h200_sxm_reference": {"verified": True, "dense_peaks_tflops": report.PEAKS},
        "torch_device_uuid": "device-physical",
    }


def test_h200_identity_uses_pci_and_runtime_uuid_despite_driver_display_name():
    hardware = hardware_fixture()
    report.audit_hardware(hardware)
    hardware["gpu"]["name"] = "NVIDIA H200"
    hardware["pci_identity"]["device_id"] = "2330"
    with pytest.raises(ValueError, match="PCI"):
        report.audit_hardware(hardware)


@pytest.mark.parametrize("defect", ["uuid", "pci_raw", "sm", "peaks"])
def test_h200_identity_rejects_conflicting_recorded_evidence(defect):
    hardware = hardware_fixture()
    if defect == "uuid":
        hardware["torch_device_uuid"] = "GPU-another-device"
    elif defect == "pci_raw":
        hardware["gpu"]["pci.device_id"] = "0x233010DE"
    elif defect == "sm":
        hardware["is_sm90"] = False
    else:
        hardware["h200_sxm_reference"]["dense_peaks_tflops"] = {**report.PEAKS, "BF16": 1.0}
    with pytest.raises(ValueError):
        report.audit_hardware(hardware)


def graph_fixture():
    result = {
        "prefix_tokens": 16,
        "extend_tokens": 4,
        "chunk_size": 8,
        "compute_graphs": False,
        "scope": "checkpoint_layers_0_1_2_embedding_final_norm_last_token_lm_head",
        "measurements": {},
    }
    calls, templates = [], []
    stages = [stage for stage in STAGES if stage not in {"indexer_fused", "lm_head"}]
    for method_index, method in enumerate(report.METHODS):
        result["measurements"][method] = {
            "prefill_samples_ms": [8.0, 12.0],
            "extend_samples_ms": [1.0 + method_index, 3.0 + method_index],
        }
        for phase in report.PHASES:
            chunks = [(0, 8), (8, 8)] if phase == report.PHASES[0] else [(16, 4)]
            matrix = []
            for start, tokens in chunks:
                for layer in range(3):
                    for stage in stages:
                        precision = "FP32" if stage == "index_weights_proj" else "FP8"
                        matrix.append(
                            {
                                "mode": method,
                                "phase": phase,
                                "layer": f"layer_{layer}",
                                "stage": "indexer_fused"
                                if method == "echo" and stage == "indexer_qk"
                                else stage,
                                "useful_flops": tokens * 1_000_000,
                                "precision": precision,
                                "query_start": start,
                                "query_tokens": tokens,
                            }
                        )
            matrix.append(
                {
                    "mode": method,
                    "phase": phase,
                    "layer": "shared",
                    "stage": "lm_head",
                    "useful_flops": 10_000_000,
                    "precision": "BF16",
                }
            )
            if phase == report.PHASES[0]:
                calls.extend(matrix)
                continue
            nodes = list(range(1, len(matrix) + 2))
            template = {
                "method": method,
                "capture_graph_id": method_index + 10,
                "executable_graph_id": method_index + 100,
                "gpu_node_ids": nodes,
                "node_owners": {
                    str(node): {"stage": "matrix", "layer": "shared"} for node in nodes
                },
                "operators": [],
            }
            replay = {
                "call_id": len(calls),
                "mode": method,
                "phase": phase,
                "layer": "shared",
                "stage": "extend_graph_replay_q_4",
                "graph_replay": True,
                "full_extend_graph": True,
                "graph_capture_id": template["capture_graph_id"],
                "graph_id": template["executable_graph_id"],
                "graph_gpu_node_ids": nodes,
                "graph_node_owners": deepcopy(template["node_owners"]),
                "nvtx": f"echo/{method}/{phase}/shared/extend_graph_replay_q_4/call_0",
                "useful_flops": None,
                "precision": None,
                "formula": "N/A (no matrix multiply)",
            }
            calls.append(replay)
            for index, call in enumerate(matrix):
                operator = {
                    key: value for key, value in call.items() if key not in {"mode", "phase"}
                }
                operator["graph_node_ids"] = [index + 1]
                template["operators"].append(operator)
                calls.append(
                    {
                        **call,
                        "graph_node_ids": operator["graph_node_ids"],
                        "graph_api": True,
                        "graph_replay_nvtx": replay["nvtx"],
                        "parent_call_id": replay["call_id"],
                        "cpu_inclusive_ns": 0,
                    }
                )
            templates.append(template)
    result["full_extend_graph_templates"] = deepcopy(templates)
    result["nsys_capture_order"] = [
        label
        for method in report.METHODS
        for label in (
            f"{method}/prefill_annotated",
            f"{method}/extend_graph_setup",
            f"{method}/extend_annotated",
        )
    ]
    return result, calls, templates


def analysis_fixture(graph):
    return {
        "captures": [
            {
                "mode": method,
                "phase": phase,
                "audit": {
                    "kernel_count_and_time_conserved": True,
                    "metadata_call_counts_match": True,
                    "layer_unscoped_kernel_count": 0,
                    "activity_counts_and_ns": {"kernel": {"unattributed_count": 0}},
                },
                "graph_attribution": None
                if phase == report.PHASES[0]
                else {
                    "full_extend_graph": True,
                    "every_replay_gpu_node_verified": True,
                    "replays": 1,
                    "gpu_node_activities": graph["full_extend"][index]["gpu_nodes"],
                    "matrix_api_count": graph["full_extend"][index]["matrix_api_calls"],
                    "matrix_api_gpu_node_activities": graph["full_extend"][index][
                        "matrix_api_gpu_nodes"
                    ],
                },
            }
            for index, method in enumerate(report.METHODS)
            for phase in report.PHASES
        ],
        "calls_outside_selected_captures": [],
        "dense_peaks_tflops": report.PEAKS,
        "peak_reference": "H200 nominal dense",
        "operators": [{"useful_flops": None, "kernel_mfu_percent": None, "kernel_ns": 4}],
        "operators_by_layer": [{"useful_flops": 10, "kernel_mfu_percent": 1.0, "kernel_ns": 4}],
    }


def test_complete_work_uses_all_prefill_chunks_and_clean_wall_medians():
    result, calls, _ = graph_fixture()
    rows, coverage = report.audit_work(result, calls, report.PEAKS)
    assert len(rows) == 8
    assert len(coverage) == 48
    for row in rows:
        tokens = 16 if row["phase"] == "prefill" else 4
        ideal = (
            3 * 13 * tokens * 1_000_000 / 1979 + 3 * tokens * 1_000_000 / 67 + 10_000_000 / 989.5
        ) / 1e9
        wall = 10 if row["phase"] == "prefill" else 2 + report.METHODS.index(row["method"])
        assert row["ideal_compute_ms"] == pytest.approx(ideal)
        assert row["mfu_percent"] == pytest.approx(100 * ideal / wall)
        assert row["details"]["matrix_call_count"] == (85 if tokens == 16 else 43)
    calls = [
        call for call in calls if not (call.get("query_start") == 0 and call["mode"] == "echo")
    ]
    with pytest.raises(ValueError, match="gap, overlap"):
        report.audit_work(result, calls, report.PEAKS)


def test_full_graph_replay_requires_original_api_metadata_and_exact_nodes():
    result, calls, templates = graph_fixture()
    graph = report.audit_graph_ledgers(result, calls, templates)
    assert [row["matrix_api_calls"] for row in graph["full_extend"]] == [43] * 4
    assert [row["gpu_nodes"] for row in graph["full_extend"]] == [44] * 4
    changed = deepcopy(calls)
    next(call for call in changed if call.get("graph_api"))["useful_flops"] += 1
    with pytest.raises(ValueError, match="capture-time metadata"):
        report.audit_graph_ledgers(result, changed, templates)
    missing = [call for call in calls if not (call["mode"] == "hbm" and call.get("graph_replay"))]
    with pytest.raises(ValueError, match="exactly one full graph"):
        report.audit_graph_ledgers(result, missing, templates)
    changed = deepcopy(templates)
    changed[0]["gpu_node_ids"].append(1)
    with pytest.raises(ValueError, match="differs from profile"):
        report.audit_graph_ledgers(result, calls, changed)


@pytest.mark.parametrize("method", report.METHODS)
def test_isolated_method_helpers_preserve_graph_and_work_checks(tmp_path, method):
    result, calls, templates = graph_fixture()
    complete = report.audit_graph_ledgers(result, calls, templates)
    analysis = analysis_fixture(complete)
    result["full_extend_graph_templates"] = [row for row in templates if row["method"] == method]
    result["nsys_capture_order"] = [
        label for label in result["nsys_capture_order"] if label.startswith(method + "/")
    ]
    selected = [row for row in calls if row["mode"] == method]
    graph = report.audit_graph_ledgers(
        result, selected, result["full_extend_graph_templates"], methods=(method,)
    )
    analysis["captures"] = [row for row in analysis["captures"] if row["mode"] == method]
    report.audit_analysis(result, analysis, graph, methods=(method,))
    rows, coverage = report.audit_work(result, selected, report.PEAKS, methods=(method,))
    assert len(rows) == 2 and len(coverage) == 12
    setup, measured = report.capture_paths(tmp_path, result, methods=(method,))
    assert len(setup) == 1 and len(measured) == 2
    with pytest.raises(ValueError, match="selected methods"):
        report.audit_graph_ledgers(result, selected, result["full_extend_graph_templates"])
    analysis["captures"].append(deepcopy(analysis["captures"][0]))
    with pytest.raises(ValueError, match="method/phase"):
        report.audit_analysis(result, analysis, graph, methods=(method,))


def test_prefill_compute_graph_coverage_includes_earlier_chunks_and_all_layers():
    result, calls, templates = graph_fixture()
    result["compute_graphs"] = True
    compute = {"graphs": []}
    for layer in range(3):
        for part in ("projection", "finish"):
            template = {
                "part": part,
                "layer": layer,
                "queries": 8,
                "capture_graph_id": 1000 + len(compute["graphs"]),
                "executable_graph_id": 2000 + len(compute["graphs"]),
                "gpu_node_ids": [3000 + len(compute["graphs"])],
            }
            compute["graphs"].append(template)
            for method in report.METHODS:
                for _ in range(2):
                    calls.append(
                        {
                            "mode": method,
                            "phase": report.PHASES[0],
                            "layer": f"layer_{layer}",
                            "stage": f"compute_graph_{part}_layer_{layer}_q_8",
                            "graph_replay": True,
                            "graph_capture_id": template["capture_graph_id"],
                            "graph_id": template["executable_graph_id"],
                            "graph_gpu_node_ids": template["gpu_node_ids"],
                        }
                    )
    audit = report.audit_graph_ledgers(result, calls, templates, compute)
    assert audit["prefill_compute_graphs"]["total_replays"] == 48
    with pytest.raises(ValueError, match="every layer and query chunk"):
        report.audit_graph_ledgers(result, calls[:-1], templates, compute)


def test_capture_order_keeps_setup_separate_and_rejects_missing_capture(tmp_path):
    result, _, _ = graph_fixture()
    setup, paths = report.capture_paths(tmp_path, result)
    assert len(setup) == 4 and len(paths) == 8
    assert setup[0].name == "capture_2.sqlite"
    result["compute_graphs"] = True
    result["nsys_capture_order"].insert(0, "graph_setup")
    setup, paths = report.capture_paths(tmp_path, result)
    assert len(setup) == 5 and len(paths) == 8
    result["nsys_capture_order"].pop()
    with pytest.raises(ValueError, match="every independent setup"):
        report.capture_paths(tmp_path, result)


@pytest.mark.parametrize(
    "defect", ["missing_phase", "unattributed", "missing_node", "nonmatrix_mfu", "missing_kernel"]
)
def test_publication_rejects_incomplete_gpu_coverage_or_invented_mfu(defect):
    result, calls, templates = graph_fixture()
    graph = report.audit_graph_ledgers(result, calls, templates)
    analysis = analysis_fixture(graph)
    report.audit_analysis(result, analysis, graph)
    if defect == "missing_phase":
        analysis["captures"].pop()
    elif defect == "unattributed":
        analysis["captures"][0]["audit"]["activity_counts_and_ns"]["kernel"][
            "unattributed_count"
        ] = 1
    elif defect == "missing_node":
        analysis["captures"][1]["graph_attribution"]["gpu_node_activities"] -= 1
    elif defect == "nonmatrix_mfu":
        analysis["operators"][0]["kernel_mfu_percent"] = 0.0
    else:
        analysis["operators_by_layer"][0]["kernel_ns"] = 0
    with pytest.raises(ValueError):
        report.audit_analysis(result, analysis, graph)


def test_existing_destination_is_not_modified(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing accepted artifacts")
    with pytest.raises(FileExistsError, match="already exists"):
        report.generate(tmp_path, output)
    assert marker.read_text() == "existing accepted artifacts"


def _publication_fixture(tmp_path, monkeypatch):
    result, calls, templates = graph_fixture()
    result.update(
        schema_version=3,
        mode="profile",
        accepted=True,
        extend_graph=True,
        extend_graph_policy_revision=report.GRAPH_POLICY,
        profile_detail="matrix_api_node_ownership",
        num_layers=3,
        methods=list(report.METHODS),
        run_id="profile-run",
        hardware=hardware_fixture(),
        execution_identity={"test": "shared identity"},
        source_sha256={},
        dependencies={},
        backend_provenance={},
        compute_precision={},
        extend_chunk_size=None,
        sparse_pool_tokens=20,
        extend_residency="cold",
        warmups=1,
        repeats=2,
        prefill_repeats=2,
        hbm_cache_budget_bytes=1000,
        dram_cache_budget_bytes=1000,
        benchmark={"run_id": "bench-run"},
        validation_receipt={"receipt_sha256": "fixture"},
    )
    directories, runs = {}, {}
    for mode in ("check", "bench", "profile"):
        directory = tmp_path / (mode + "-run")
        directory.mkdir()
        directories[mode] = directory
        (directory / "request.json").write_text("{}")
        run = {**deepcopy(result), "mode": mode, "run_id": directory.name}
        run["request_sha256"] = report.digest(directory / "request.json")
        (directory / "result.json").write_text(json.dumps(run))
        runs[mode] = run
    directory = directories["profile"]
    result = runs["profile"]
    (directory / "full_graph_templates.json").write_text(json.dumps(templates))
    (directory / "operator_calls.json").write_text("fixture read separately")
    receipt_path = directories["check"] / "receipt.json"
    receipt_path.write_text("{}")
    for path in chain.from_iterable(report.capture_paths(directory, result)):
        path.write_text("fixture analyzed separately")
    graph = report.audit_graph_ledgers(result, calls, templates)
    analysis = analysis_fixture(graph)
    monkeypatch.setattr(
        report,
        "benchmark_view",
        lambda *_: {
            **result,
            "wall_time_denominator": {"mode": "independent_bench", "run_id": "bench-run"},
        },
    )
    monkeypatch.setattr(
        report,
        "_independent_runs",
        lambda *_: ({"receipt_path": str(receipt_path)}, directories, runs),
    )
    monkeypatch.setattr(report, "audit_local_native_artifacts", lambda *_: [])
    monkeypatch.setattr(report, "execution_identity", lambda run: run["execution_identity"])
    monkeypatch.setattr(report, "read_calls", lambda *_: (calls, {"run_id": result["run_id"]}))
    monkeypatch.setattr(report, "analyze_captures", lambda *_args, **_kwargs: analysis)
    monkeypatch.setattr(report, "plot_mfu", lambda *_: None)
    return directory


def test_report_emits_eight_rows_clean_samples_and_no_timeline(tmp_path, monkeypatch):
    directory = _publication_fixture(tmp_path, monkeypatch)
    output = tmp_path / "fresh-report"
    summary = report.generate(directory, output)
    assert summary["wall_time_denominator"]["run_id"] == "bench-run"
    assert len(list(csv.DictReader((output / "final_mfu.csv").open()))) == 8
    assert len(list(csv.DictReader((output / "timing_samples.csv").open()))) == 16
    assert not any("timeline" in path.name for path in output.iterdir())
    manifest = json.loads((output / "publication_manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert report.digest(output / name) == expected


def test_report_refuses_missing_independent_binding_before_writing(tmp_path, monkeypatch):
    directory = _publication_fixture(tmp_path, monkeypatch)

    def missing_binding(*_):
        raise ValueError("Profile has no matching independent bench")

    monkeypatch.setattr(report, "benchmark_view", missing_binding)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="no matching independent bench"):
        report.generate(directory, output)
    assert not output.exists()
