"""Acceptance bindings and clean/profile separation in shape-matrix publication."""

import copy
import csv
import json
from pathlib import Path

import pytest

from evaluation.validation import write_receipt
from experiments.deepseek_v32_mfu.src import report_shape_matrix as report
from experiments.deepseek_v32_mfu.src.run_contract import (
    CHECK_FIELDS,
    IDENTITY_FIELDS,
    METHODS,
    RECEIPT_KIND,
    bind_benchmark,
    digest,
    execution_identity,
    receipt_binding,
)


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def fixture_hardware_source(timeout):
    return b"# fixture hardware\n" + report.shape_matrix_sources._PROBE_LINES[timeout]


def make_shape(tmp_path, history=4096, extend=128, *, startup=True, hardware_timeout=None):
    root = tmp_path / f"h{history}_a{extend}"
    root.mkdir()
    paths = {mode: root / mode for mode in ("check", "bench", "profile")}
    for path in paths.values():
        path.mkdir()
        (path / "source").mkdir()
        (path / "source/kernel.py").write_text("fixture source\n")
        (path / "request.json").write_text("fixture request\n")
        if hardware_timeout is not None:
            hardware_path = path / "source" / report.shape_matrix_sources.HARDWARE_SOURCE
            hardware_path.parent.mkdir(parents=True)
            hardware_path.write_bytes(fixture_hardware_source(hardware_timeout))
    result = {key: "fixture" for key in IDENTITY_FIELDS}
    result.update(
        {
            "schema_version": 3,
            "accepted": True,
            "methods": list(METHODS),
            "extend_residency": "cold",
            "compute_graphs": True,
            "extend_graph": True,
            "extend_graph_policy_revision": "fixture-full-graph",
            "num_layers": 3,
            "prefix_tokens": history,
            "extend_tokens": extend,
            "chunk_size": 1024,
            "extend_chunk_size": extend,
            "sparse_pool_tokens": history + extend,
            "host_arena_tokens": history + extend,
            "source_sha256": {"kernel.py": digest(paths["bench"] / "source/kernel.py")},
            "request_sha256": digest(paths["bench"] / "request.json"),
            "hardware": {"gpu": {"uuid": "fixture-gpu"}},
            "warmups": 1,
            "prefill_repeats": 3,
            "repeats": 5,
            "timing": "synchronized independent wall time",
            "measurements": {
                method: {
                    "prefill_samples_ms": [10.0, 11.0, 12.0],
                    "prefill_median_ms": 11.0,
                    "extend_samples_ms": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "extend_median_ms": 3.0,
                }
                for method in METHODS
            },
        }
    )
    if hardware_timeout is not None:
        hardware_name = report.shape_matrix_sources.HARDWARE_SOURCE
        result["source_sha256"][hardware_name] = digest(paths["bench"] / "source" / hardware_name)
        result["hardware"]["hardware_source_sha256"] = result["source_sha256"][hardware_name]
    for method in METHODS:
        row = result["measurements"][method]
        for phase, count, final_key in (
            ("prefill", 3, "prefix_cache_per_layer"),
            ("extend", 5, "extend_cache_per_layer"),
        ):
            samples = []
            for sample in range(count):
                layers = []
                for layer in range(3):
                    prefetched = sample + layer if method == "echo" else 0
                    recalled = 100 - prefetched if method != "hbm" else 0
                    layers.append(
                        {
                            "prefetched_records": prefetched,
                            "recalled_records": recalled,
                            "evicted_records": 0,
                            "prefetch_capacity_failures": 0,
                            "host_to_device_bytes": (prefetched + recalled) * 1152,
                            "device_to_host_bytes": 0 if method == "hbm" else extend * 1152,
                            "record_bytes": 1152,
                            "pool_scope": "session" if method == "hbm" else "shared_per_layer",
                        }
                    )
                samples.append(layers)
            row[phase + "_cache_samples"] = samples
            row[final_key] = samples[-1]
    result["execution_identity"] = execution_identity(result)
    graph_names = {
        "default_graph_logits",
        "default_graph_cache",
        *(
            f"replay_{repeat}_{output}"
            for repeat in range(2)
            for output in ("hidden", "logits", "cache")
        ),
        *(f"changed_input_{output}" for output in ("hidden", "logits", "cache")),
    }
    checks = {
        "passed": True,
        "comparisons": {name: {"passed": True} for name in CHECK_FIELDS},
        "extend_graph": {
            method: {"checks": {name: {"passed": True} for name in graph_names}}
            for method in METHODS
        },
    }
    receipt = write_receipt(
        paths["check"] / "receipt.json",
        kind=RECEIPT_KIND,
        identity=result["execution_identity"],
        checks=checks,
    )
    for mode, path in paths.items():
        current = {
            **copy.deepcopy(result),
            "run_id": f"h{history}_a{extend}_{mode}",
            "mode": mode,
            "correctness": checks if mode == "check" else {},
        }
        if mode != "check":
            current["validation_receipt"] = receipt_binding(receipt)
        if mode == "profile":
            current["benchmark"] = bind_benchmark(paths["bench"], current)
            # A profile's own observations must never become clean wall-time samples.
            current["measurements"] = {method: {"profile_ms": 99.0} for method in METHODS}
        write(path / "result.json", current)
    profile = report.read(paths["profile"] / "result.json")
    write(
        paths["profile"] / "gap_audit.json",
        {
            "run_id": profile["run_id"],
            "input_result_sha256": digest(paths["profile"] / "result.json"),
            "methods": [
                {
                    "method": method,
                    "graph_attribution": {
                        "replays": 1,
                        "full_extend_graph": True,
                        "every_replay_gpu_node_verified": True,
                        "gpu_node_activities": 200,
                    },
                }
                for method in METHODS
            ],
        },
    )
    write(paths["profile"] / "operator_calls.json", [])
    entry = {
        "prefix_tokens": history,
        "extend_tokens": extend,
        **{f"{key}_run": str(value) for key, value in paths.items()},
        "timeline_run": str(root / "timeline"),
        "startup_run": None,
    }
    if startup:
        entry["startup_run"] = str(root / "startup")
    for is_startup, name in ((False, "timeline_run"), (True, "startup_run")):
        if entry[name] is None:
            continue
        directory = Path(entry[name])
        directory.mkdir()
        names = report.STARTUP_FILES if is_startup else report.TIMELINE_FILES
        for filename in (*names, "windows.csv", "annotations.csv"):
            (directory / filename).write_text("fixture timeline\n")
        receipt = {
            "profile_run_id": profile["run_id"],
            "layout": "separate",
            "window_kind": "extend-startup" if is_startup else "three-layers",
            "io_layout": "directions",
            "annotations": "idle-echo",
            "inputs_and_sources_sha256": {
                str(paths["profile"] / filename): digest(paths["profile"] / filename)
                for filename in ("result.json", "gap_audit.json", "operator_calls.json")
            },
            "metrics": {
                f"{phase}/{method}": {
                    "window": {
                        "start_ns": 10,
                        "end_ns": 1000010,
                        "window_ms": 1.0,
                        "gap_ms": 0.1,
                        "gap_no_io_percent": 10.0,
                        "gpu_idle_ms": 0.05,
                    }
                }
                for phase in (("extend",) if is_startup else ("prefill", "extend"))
                for method in METHODS
            },
            "artifacts_sha256": {path.name: digest(path) for path in directory.iterdir()},
        }
        write(directory / "compact_receipt.json", receipt)
    return entry


def manifest(tmp_path, entries):
    path = tmp_path / "manifest.json"
    write(
        path,
        {
            "schema_version": 1,
            "run_id": "fixture_matrix",
            "configuration": {
                "prefix_tokens": sorted({entry["prefix_tokens"] for entry in entries}),
                "extend_tokens": sorted({entry["extend_tokens"] for entry in entries}),
            },
            "shapes": entries,
        },
    )
    return path


def independent_audit(manifest_path):
    matrix = report.read(manifest_path)
    shapes = [
        {
            "prefix_tokens": entry["prefix_tokens"],
            "extend_tokens": entry["extend_tokens"],
            "passed": True,
            "request_prefix": {"prefix_token_ids_sha256": f"history-{entry['prefix_tokens']}"},
        }
        for entry in matrix["shapes"]
    ]
    return {
        "passed": True,
        "run_id": matrix["run_id"],
        "complete_matrix": True,
        "manifest_sha256": digest(manifest_path),
        "input_sha256": {str(manifest_path): digest(manifest_path)},
        "shapes": shapes,
        "history_identity_across_extend_sizes": [
            {
                "prefix_tokens": history,
                "extend_tokens": [
                    row["extend_tokens"] for row in shapes if row["prefix_tokens"] == history
                ],
                "same_prefix_token_ids_sha256": True,
                "prefix_token_ids_sha256_by_extend_tokens": {
                    str(row["extend_tokens"]): row["request_prefix"]["prefix_token_ids_sha256"]
                    for row in shapes
                    if row["prefix_tokens"] == history
                },
            }
            for history in sorted({row["prefix_tokens"] for row in shapes})
        ],
    }


def test_publishes_full_cartesian_matrix_with_independent_times_and_startup(tmp_path):
    entries = [make_shape(tmp_path, history, extend) for history, extend in sorted(report.SHAPES)]
    path = manifest(tmp_path, entries)
    observer = tmp_path / "observer.json"
    write(observer, {"samples": 3, "foreign_gpu_processes": []})
    write(path, {**report.read(path), "observer": str(observer)})
    audit_path = tmp_path / "audit.json"
    write(audit_path, independent_audit(path))
    raw, selected = tmp_path / "raw_report", tmp_path / "selected_report"
    report.publish(path, raw, publish_dir=selected, audit_paths=[audit_path])
    with (selected / "timing.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48
    assert {float(row["extend_median_ms"]) for row in rows} == {3.0}
    assert {float(row["prefill_median_ms"]) for row in rows} == {11.0}
    with (selected / "timing_samples.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 12 * 4 * (3 + 5)
    with (selected / "timeline_windows.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 12 * 4 * 3
    with (selected / "cache_metrics_samples.csv").open() as stream:
        cache_rows = list(csv.DictReader(stream))
    assert len(cache_rows) == 12 * 4 * (3 + 5) * 3
    echo = [row for row in cache_rows if row["method"] == "echo" and row["phase"] == "extend"]
    assert {int(row["cache_prefetched_records"]) for row in echo} == set(range(7))
    assert report.read(selected / "summary.json")["complete_requested_matrix"]
    assert report.read(selected / "summary.json")["execution_batches"] == []
    assert (selected / "matrix_manifest.json").read_bytes() == path.read_bytes()
    assert (selected / "h65536_a1024/with_startup/extend_with_startup.svg").is_file()
    assert report.read(selected / "audit/observer.json") == report.read(observer)
    for name, expected in report.read(selected / "publication_manifest.json")[
        "files_sha256"
    ].items():
        assert digest(selected / name) == expected == digest(raw / name)
    text = (selected / "results.md").read_text()
    assert "| 65,536 | 1,024 | 3.000 | 3.000 | 3.000 | 3.000 |" in text
    assert "intrusive NSYS" in text
    assert "execution batches" not in text
    with pytest.raises(FileExistsError):
        report.publish(path, raw)


def test_preserves_batch_metadata_and_reports_only_recorded_preflight_interruption(tmp_path):
    path = manifest(tmp_path, [make_shape(tmp_path, startup=False)])
    batches = [
        {
            "run_id": "earlier_batch",
            "completed_shapes": [[4096, 128]],
            "preflight_interruption": {
                "stage": "before_model_loading",
                "command": "nvidia-smi",
                "timeout_seconds": 20,
            },
        },
        {"run_id": "later_batch", "completed_shapes": []},
    ]
    matrix = {**report.read(path), "execution_batches": batches}
    # Preserve the input bytes, including formatting, rather than reserializing them.
    path.write_text(json.dumps(matrix, indent=4) + "\n\n")
    output = tmp_path / "batch_report"
    report.publish(path, output, allow_subset=True)
    assert (output / "matrix_manifest.json").read_bytes() == path.read_bytes()
    assert report.read(output / "summary.json")["execution_batches"] == batches
    text = (output / "results.md").read_text()
    assert "2 execution batches with the same frozen measured sources" in text
    assert "before model loading" in text and "20-second timeout" in text
    assert "cause of the query delay is not established" in text
    assert "[exact input matrix manifest](matrix_manifest.json)" in text
    del batches[0]["preflight_interruption"]
    text = report.markdown(matrix, [], [])
    assert "execution batches" in text and "timeout" not in text


def test_accepts_only_verified_deadline_variants_across_shapes(tmp_path, monkeypatch):
    hardware_name = report.shape_matrix_sources.HARDWARE_SOURCE
    current_root = tmp_path / "current"
    current = current_root / hardware_name
    current.parent.mkdir(parents=True)
    current.write_bytes(fixture_hardware_source(120))
    monkeypatch.setattr(report, "ROOT", current_root)
    entries = [
        make_shape(tmp_path, extend=128, hardware_timeout=20),
        make_shape(tmp_path, extend=256, hardware_timeout=120),
    ]
    path = manifest(tmp_path, entries)
    write(path, {**report.read(path), "execution_batches": [{"run_id": "old"}, {"run_id": "new"}]})
    output = tmp_path / "mixed_report"
    report.publish(path, output, allow_subset=True)
    summary = report.read(output / "summary.json")
    compatibility = summary["source_compatibility"]
    assert compatibility["matrix_hardware_probe_deadline_differs"] is True
    assert len(compatibility["original_source_identity_sha256"]) == 2
    assert len({row["source_identity_sha256"] for row in summary["shapes"]}) == 2
    text = (output / "results.md").read_text()
    assert "same model and measurement execution code" in text
    assert "hardware query deadline differs from 20 to 120 seconds" in text
    assert "same frozen measured sources" not in text
    provenance = report.read(output / "provenance.json")
    assert "source_sha256" not in provenance["common_execution_metadata"]
    assert str(current) in provenance["inputs_sha256"]
    assert str(Path(report.shape_matrix_sources.__file__).resolve()) in provenance["inputs_sha256"]
    current.write_bytes(fixture_hardware_source(120) + b"# unrelated change\n")
    with pytest.raises(ValueError, match="differs beyond"):
        report.publish(path, tmp_path / "rejected", allow_subset=True)
    assert not (tmp_path / "rejected").exists()


def test_deadline_compatibility_does_not_relax_identity_within_a_shape(tmp_path):
    entry = make_shape(tmp_path, hardware_timeout=20)
    path = Path(entry["bench_run"]) / "result.json"
    bench = report.read(path)
    name = report.shape_matrix_sources.HARDWARE_SOURCE
    source = path.parent / "source" / name
    source.write_bytes(fixture_hardware_source(120))
    bench["source_sha256"][name] = digest(source)
    bench["hardware"]["hardware_source_sha256"] = digest(source)
    bench["execution_identity"] = execution_identity(bench)
    write(path, bench)
    with pytest.raises(ValueError, match="execution identities differ"):
        report.validate_shape(entry)


def test_rejects_incomplete_duplicate_and_misdeclared_matrix(tmp_path):
    entries = [make_shape(tmp_path)]
    path = manifest(tmp_path, entries)
    with pytest.raises(ValueError, match="complete"):
        report.load_manifest(path)
    report.load_manifest(path, allow_subset=True)
    value = report.read(path)
    value["shapes"].append(entries[0])
    write(path, value)
    with pytest.raises(ValueError, match="duplicate"):
        report.load_manifest(path, allow_subset=True)
    value["shapes"].pop()
    value["configuration"]["extend_tokens"].append(256)
    write(path, value)
    with pytest.raises(ValueError, match="Cartesian product"):
        report.load_manifest(path, allow_subset=True)


def test_rejects_tampered_timeline_and_wrong_shape(tmp_path):
    entry = make_shape(tmp_path)
    report.validate_shape(entry)
    (Path(entry["timeline_run"]) / "extend.svg").write_text("changed\n")
    with pytest.raises(ValueError, match="Timeline artifact changed"):
        report.validate_shape(entry)
    entry["extend_tokens"] = 256
    with pytest.raises(ValueError, match="declared cold full-graph shape"):
        report.validate_shape(entry)


def test_rejects_changed_median_and_profile_benchmark_binding(tmp_path):
    entry = make_shape(tmp_path, startup=False)
    bench_path = Path(entry["bench_run"]) / "result.json"
    bench = report.read(bench_path)
    bench["measurements"]["hbm"]["extend_median_ms"] = 99.0
    write(bench_path, bench)
    with pytest.raises(ValueError, match="median differs"):
        report.validate_shape(entry)
    bench["measurements"]["hbm"]["extend_median_ms"] = 3.0
    write(bench_path, bench)
    entry["bench_run"] = entry["profile_run"]
    with pytest.raises(ValueError, match="independent runs"):
        report.validate_shape(entry)


def test_audit_must_bind_exact_manifest_and_pass_every_shape(tmp_path):
    entry = make_shape(tmp_path, startup=False)
    path = manifest(tmp_path, [entry])
    audit_path = tmp_path / "audit.json"
    audit = independent_audit(path)
    audit["shapes"][0]["passed"] = False
    write(audit_path, audit)
    with pytest.raises(ValueError, match="audit has not passed"):
        report.publish(path, tmp_path / "failed", audit_paths=[audit_path], allow_subset=True)
    assert not (tmp_path / "failed").exists()
    audit["shapes"][0]["passed"] = True
    write(audit_path, audit)
    report.publish(path, tmp_path / "passed", audit_paths=[audit_path], allow_subset=True)
    assert (tmp_path / "passed/audit/0_audit.json").is_file()
    assert not report.read(tmp_path / "passed/summary.json")["complete_requested_matrix"]
    evidence = tmp_path / "audited_capture.sqlite"
    evidence.write_bytes(b"fixture capture")
    audit["input_sha256"][str(evidence)] = digest(evidence)
    write(audit_path, audit)
    evidence.write_bytes(b"changed capture")
    with pytest.raises(ValueError, match="audit input changed"):
        report.publish(path, tmp_path / "stale_audit", audit_paths=[audit_path], allow_subset=True)
    assert not (tmp_path / "stale_audit").exists()


def test_publication_requires_complete_audit_but_preview_does_not(tmp_path):
    path = manifest(tmp_path, [make_shape(tmp_path, startup=False)])
    raw, selected = tmp_path / "raw", tmp_path / "selected"
    with pytest.raises(ValueError, match="requires a complete matching independent audit"):
        report.publish(path, raw, publish_dir=selected, allow_subset=True)
    assert not raw.exists() and not selected.exists()
    audit_path = tmp_path / "audit.json"
    audit = independent_audit(path)
    audit["complete_matrix"] = False
    write(audit_path, audit)
    with pytest.raises(ValueError, match="requires a complete matching independent audit"):
        report.publish(path, raw, publish_dir=selected, audit_paths=[audit_path], allow_subset=True)
    assert not raw.exists() and not selected.exists()
    report.publish(path, raw, allow_subset=True)
    assert raw.exists() and not selected.exists()


@pytest.mark.parametrize("corruption", ["missing", "false", "extends", "hashes", "shape_hash"])
def test_audit_history_assertions_must_cover_and_match_every_shape(tmp_path, corruption):
    path = manifest(tmp_path, [make_shape(tmp_path, extend=extend) for extend in (128, 256)])
    audit_path = tmp_path / "audit.json"
    audit = independent_audit(path)
    assertion = audit["history_identity_across_extend_sizes"][0]
    if corruption == "missing":
        audit["history_identity_across_extend_sizes"].clear()
    elif corruption == "false":
        assertion["same_prefix_token_ids_sha256"] = False
    elif corruption == "extends":
        assertion["extend_tokens"].pop()
    elif corruption == "hashes":
        assertion["prefix_token_ids_sha256_by_extend_tokens"].pop("256")
    else:
        audit["shapes"][1]["request_prefix"]["prefix_token_ids_sha256"] = "changed-history"
        assertion["prefix_token_ids_sha256_by_extend_tokens"]["256"] = "changed-history"
    write(audit_path, audit)
    with pytest.raises(ValueError, match="Independent audit history"):
        report.publish(path, tmp_path / "failed", audit_paths=[audit_path], allow_subset=True)
    assert not (tmp_path / "failed").exists()


def test_requires_every_sample_cache_counts_and_checks_last_compatibility(tmp_path):
    entry = make_shape(tmp_path)
    bench = report.read(Path(entry["bench_run"]) / "result.json")
    report.validate_cache_samples(bench)
    row = bench["measurements"]["echo"]
    dropped = row["extend_cache_samples"].pop()
    with pytest.raises(ValueError, match="Missing per-sample"):
        report.validate_cache_samples(bench)
    row["extend_cache_samples"].append(dropped)
    row["extend_cache_per_layer"][0]["prefetched_records"] += 1
    with pytest.raises(ValueError, match="Final cache counters differ"):
        report.validate_cache_samples(bench)
    row["extend_cache_per_layer"] = copy.deepcopy(row["extend_cache_samples"][-1])
    row["extend_cache_samples"][0][0]["recalled_records"] = -1
    with pytest.raises(ValueError, match="Invalid scalar cache metrics"):
        report.validate_cache_samples(bench)
