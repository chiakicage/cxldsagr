"""CPU fixtures authenticate source chains and conserve overlapping GPU intervals."""

import hashlib
import json
import shutil

import pytest

from experiments.deepseek_v32_echo_official.src import compare_warm_decode_timeline as warm


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def seal(directory, names, schema, *, filename="publication_manifest.json", key="files_sha256"):
    save(
        directory / filename,
        {"schema": schema, key: {name: digest(directory / name) for name in names}},
    )


def refresh(bundle):
    directory, original, raw = bundle["warm"], bundle["original"], bundle["raw"]
    unchanged = [
        "summary.json",
        "window_rows.json",
        "provenance.json",
        "cache_samples.csv",
        "timing_samples.csv",
    ]
    for name in unchanged:
        shutil.copyfile(directory / name, original / name)
    original_names = [*unchanged, "results.md", "timeline.svg", "timeline.png"]
    seal(original, original_names, "q1-warm-artifacts-v1")
    shutil.copyfile(
        original / "publication_manifest.json", directory / "original_publication_manifest.json"
    )
    original_hash = digest(directory / "original_publication_manifest.json")
    display = warm.comparison.read(directory / "display_provenance.json")
    display["source_publication_sha256"] = original_hash
    display["unchanged_data_sha256"] = {name: digest(directory / name) for name in unchanged}
    save(directory / "display_provenance.json", display)
    audit_dir = directory / "input_recall_audit"
    audit = warm.comparison.read(audit_dir / "identity_recall.json")
    audit["warm_cohort"]["publication_manifest_sha256"] = original_hash
    audit["warm_cohort"]["summary_sha256"] = digest(directory / "summary.json")
    save(audit_dir / "identity_recall.json", audit)
    shutil.copyfile(audit_dir / "identity_recall.json", raw / "identity_recall.json")
    seal(raw, ["identity_recall.json", "audit.py"], "deepseek-q1-official-warm-supplement-v1")
    raw_publication = warm.comparison.read(raw / "publication_manifest.json")
    raw_publication.update(
        warm_original_publication_manifest_sha256=original_hash,
        official_observation_sha256="observer-hash",
    )
    save(raw / "publication_manifest.json", raw_publication)
    save(
        audit_dir / "publication_manifest.json",
        {
            "schema": "q1-warm-selected-identity-recall-supplement-v1",
            "files_sha256": {"identity_recall.json": digest(audit_dir / "identity_recall.json")},
            "source_audit": {"path": str(raw / "audit.py"), "sha256": digest(raw / "audit.py")},
            "raw_publication": {
                "path": str(raw / "publication_manifest.json"),
                "sha256": digest(raw / "publication_manifest.json"),
            },
            "warm_original_publication_manifest_sha256": original_hash,
            "official_observation_sha256": "observer-hash",
        },
    )
    names = [
        *original_names,
        "display_provenance.json",
        "original_publication_manifest.json",
        "input_recall_audit/identity_recall.json",
        "input_recall_audit/publication_manifest.json",
    ]
    seal(directory, names, "q1-warm-display-artifacts-v1")


@pytest.fixture
def bundle(tmp_path):
    experiment = tmp_path / "experiment"
    directory = tmp_path / "warm"
    original = tmp_path / "original"
    raw = tmp_path / "raw-audit"
    official = tmp_path / "official"
    official_run = experiment / "output/data/official-profile"
    for path in (directory, original, raw, official, official_run):
        path.mkdir(parents=True)
    b = {
        "experiment": experiment,
        "warm": directory,
        "original": original,
        "raw": raw,
        "official": official,
        "official_run": official_run,
    }
    config = {
        "prefix_tokens": 65536,
        "extend_tokens": 1,
        "num_layers": 3,
        "chunk_size": 1024,
        "extend_chunk_size": 1,
        "extend_residency": "warm",
        "method_isolation": "fresh-process-one-method-v1",
        "append_token": 57841,
        "repeats": 1,
    }
    tokens = [i % 100 for i in range(65536)] + [57841]
    request = tmp_path / "request.json"
    save(request, {"input_ids": tokens})
    history_hash = hashlib.sha256(
        json.dumps(tokens[:65536], separators=(",", ":")).encode()
    ).hexdigest()
    full_hash = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    window = {"start_ns": 100, "end_ns": 600, "window_ms": 0.0005, "gpu_idle_ms": 0.0002}
    layers = [{"layer": i, "start_ns": 100 + i * 200, "end_ns": 200 + i * 200} for i in range(3)]
    profile_dir = "/unmounted/cohort_echo_profile"
    benchmark = {
        "directory": "/unmounted/cohort_echo_bench",
        "run_id": "cohort_echo_bench",
        "result_sha256": "bench-hash",
        "execution_identity_sha256": "execution-hash",
    }
    receipt = {
        "receipt_path": "/unmounted/cohort_echo_check/receipt.json",
        "receipt_sha256": "receipt-payload-hash",
    }
    provenance = {
        "profile_run_id": "cohort_echo_profile",
        "profile_directory": profile_dir,
        "result_sha256": "profile-hash",
        "sqlite": profile_dir + "/capture_4.sqlite",
        "sqlite_sha256": "sqlite-hash",
        "benchmark": benchmark,
        "validation_receipt": receipt,
    }
    panel = {
        "method": "echo",
        "phase": "extend",
        "shape": warm.comparison.compact.profile_shape(config),
        "chunk": None,
        "window": window,
        "layers": layers,
        "provenance": provenance,
        "rows": [
            {
                "kind": "kernel",
                "lane": "Compute",
                "stage": "attention_projection",
                "purpose": "Projection / RoPE",
                "name": "projection",
                "raw_start_ns": 100 + i * 200,
                "raw_end_ns": 200 + i * 200,
            }
            for i in range(3)
        ],
    }
    counts = [
        {
            "layer": i,
            "sample": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 1152,
            "prefetched_records": 0,
            "recalled_records": 0,
        }
        for i in range(3)
    ]
    child = {
        "run_id": "cohort_echo_profile",
        "directory": profile_dir,
        "result_sha256": "profile-hash",
        "files": {"data": {"capture_4.sqlite": "sqlite-hash", "request.json": digest(request)}},
    }
    summary = {
        "schema": "deepseek-v32-q1-warm-report-v1",
        "passed": True,
        "run_id": "cohort",
        "configuration": config,
        "window": window,
        "execution_identity_sha256": "execution-hash",
        "receipts": {"echo": "receipt-payload-hash"},
        "children": {
            "echo": {
                "profile": child,
                "bench": benchmark,
                "check": {"receipt_path": receipt["receipt_path"]},
            }
        },
        "profile_cache_per_layer": counts,
        "cache_samples": counts,
    }
    save(directory / "summary.json", summary)
    save(directory / "window_rows.json", {"extend": [panel]})
    save(
        directory / "provenance.json",
        {
            "profile": provenance,
            "input_sha256": {
                profile_dir + "/result.json": "profile-hash",
                provenance["sqlite"]: "sqlite-hash",
            },
        },
    )
    save(
        directory / "display_provenance.json",
        {
            "schema": "q1-warm-timeline-display-v1",
            "measurement_run_id": "cohort",
            "source_report": str(original),
            "rendered_metrics": {
                "extend/echo": {
                    "window": window,
                    "compute_categories": {"Projection / RoPE": 3},
                    "annotation_intervals_ns": {"GPU idle": [[200, 300], [400, 500]]},
                }
            },
        },
    )
    audit = {
        "schema": "deepseek-q1-official-warm-identity-recall-v1",
        "passed": True,
        "official_numerical_acceptance": False,
        "generator_sha256": "",
        "official_observation_sha256": "observer-hash",
        "official_run_id": "separate-observer",
        "official_residual_recall": [
            {"layer": i, "h2d_records": 0, "d2d_records": 0} for i in range(3)
        ],
        "local_input": {
            "path": str(request),
            "file_sha256": digest(request),
            "history_token_sha256": history_hash,
            "full_token_sha256": full_hash,
            "prefix_tokens": 65536,
            "query_tokens": 1,
            "decode_input_token": 57841,
            "exact_history_arrays_equal": True,
            "exact_decode_token_equal": True,
            "local_sampling_included": False,
        },
        "warm_cohort": {
            "run_id": "cohort",
            "bound_profile_run_id": "cohort_echo_profile",
            "profile_result_sha256": "profile-hash",
            "execution_identity_sha256": "execution-hash",
            "counts": [{**r, "phase": "bench"} for r in counts]
            + [{**r, "phase": "profile"} for r in counts],
        },
    }
    (raw / "audit.py").write_text("# independent input audit\n")
    audit["generator_sha256"] = digest(raw / "audit.py")
    save(directory / "input_recall_audit/identity_recall.json", audit)
    for root in (directory, original):
        for name in [
            "results.md",
            "timeline.svg",
            "timeline.png",
            "cache_samples.csv",
            "timing_samples.csv",
        ]:
            (root / name).write_text(f"fixture {root.name} {name}\n")
    refresh(b)

    protocol = {"history_tokens": 65536, "decode_forwards": 1}
    complete = {
        "status": "completed",
        "case": "echo",
        "mode": "profile",
        "protocol": protocol,
        "prefix_sha256": history_hash,
        "row": {
            "input_sha256": history_hash,
            "output_ids": [57841, 39556],
            "prompt_tokens": 65536,
            "completion_tokens": 2,
        },
    }
    save(official_run / "sample_00/engine/complete.json", complete)
    run = {
        "run_id": "official-profile",
        "status": "completed",
        "case": "echo",
        "mode": "profile",
        "protocol": protocol,
        "prompt_sha256": history_hash,
        "workers": [{"completion": complete}],
    }
    save(official_run / "run.json", run)
    official_windows = []
    official_activities = {}
    sources = {}
    for method in ("hbm", "echo"):
        official_windows.append(
            {
                **window,
                "case": method,
                "phase": "decode",
                "history_tokens": 65536,
                "query_tokens": 1,
                "graph_launches": 1,
                "every_graph_node_verified": True,
                "activity_count": 3,
                "layer_boundaries": layers,
            }
        )
        official_activities[method] = [
            {
                **row,
                "start_ns": row["raw_start_ns"],
                "end_ns": row["raw_end_ns"],
                "category": "Projection / RoPE",
                "potential_io": False,
            }
            for row in panel["rows"]
        ]
        sources[method + "_profile"] = {
            "run_id": "official-profile" if method == "echo" else "hbm-profile",
            "run_sha256": digest(official_run / "run.json"),
        }
    save(official / "windows.json", official_windows)
    save(official / "activities.json", official_activities)
    save(
        official / "report.json",
        {"numerical_acceptance": False, "windows": official_windows, "sources": sources},
    )
    seal(
        official,
        ["windows.json", "activities.json", "report.json"],
        "echo-decode-publication-v1",
        filename="publication.json",
        key="files",
    )
    return b


def test_loader_preserves_original_windows_and_profile_ids(bundle):
    panels, paths, identity = warm.load_comparison(
        bundle["official"], bundle["warm"], official_run_directory=bundle["official_run"]
    )
    original = warm.comparison.read(bundle["warm"] / "window_rows.json")["extend"][0]
    assert panels[1]["window"] == original["window"]
    assert panels[1]["layers"] == original["layers"]
    assert panels[1]["provenance"] == original["provenance"]
    assert panels[1]["rows"][0]["raw_start_ns"] == original["rows"][0]["raw_start_ns"]
    assert [p["profile_run_id"] for p in panels] == ["official-profile", "cohort_echo_profile"]
    assert all(path.is_file() for path in paths)
    assert identity["decode_input_token"] == 57841
    assert identity["separate_recall_observation"]["run_id"] == "separate-observer"


@pytest.mark.parametrize(
    "location,name",
    [
        ("warm", "timeline.svg"),
        ("warm", "summary.json"),
        ("warm", "input_recall_audit/identity_recall.json"),
        ("original", "timeline.png"),
        ("raw", "audit.py"),
    ],
)
def test_every_publication_payload_is_authenticated(bundle, location, name):
    (bundle[location] / name).write_text("corrupt payload")
    with pytest.raises(ValueError, match="Published input changed"):
        warm.validate_warm(bundle["warm"])


@pytest.mark.parametrize(
    "defect",
    [
        "failed",
        "cold",
        "shape",
        "profile",
        "benchmark",
        "receipt",
        "sqlite",
        "window",
        "categories",
        "traffic",
        "source_chain",
    ],
)
def test_resealed_invalid_metadata_is_rejected(bundle, defect):
    directory = bundle["warm"]
    summary = warm.comparison.read(directory / "summary.json")
    rows = warm.comparison.read(directory / "window_rows.json")
    display = warm.comparison.read(directory / "display_provenance.json")
    audit = warm.comparison.read(directory / "input_recall_audit/identity_recall.json")
    if defect == "failed":
        summary["passed"] = False
    elif defect == "cold":
        summary["configuration"]["extend_residency"] = "cold"
    elif defect == "shape":
        summary["configuration"]["extend_tokens"] = 128
    elif defect == "profile":
        rows["extend"][0]["provenance"]["profile_run_id"] = "other-profile"
    elif defect == "benchmark":
        rows["extend"][0]["provenance"]["benchmark"]["result_sha256"] = "other-bench"
    elif defect == "receipt":
        rows["extend"][0]["provenance"]["validation_receipt"]["receipt_sha256"] = "other-receipt"
    elif defect == "sqlite":
        rows["extend"][0]["provenance"]["sqlite_sha256"] = "other-sqlite"
    elif defect == "window":
        rows["extend"][0]["window"]["window_ms"] = 0.003
    elif defect == "categories":
        display["rendered_metrics"]["extend/echo"]["compute_categories"] = {}
    elif defect == "traffic":
        audit["warm_cohort"]["counts"][0]["host_to_device_bytes"] = 1152
    elif defect == "source_chain":
        audit["official_observation_sha256"] = "different-observer"
    save(directory / "summary.json", summary)
    save(directory / "window_rows.json", rows)
    save(directory / "display_provenance.json", display)
    save(directory / "input_recall_audit/identity_recall.json", audit)
    refresh(bundle)
    with pytest.raises(ValueError):
        warm.validate_warm(directory)


@pytest.mark.parametrize("defect", ["history", "token", "official_completion", "official_run_hash"])
def test_input_identity_rejects_mismatch(bundle, defect):
    directory = bundle["warm"]
    audit_path = directory / "input_recall_audit/identity_recall.json"
    audit = warm.comparison.read(audit_path)
    request_path = bundle["experiment"].parent / "request.json"
    if defect in {"history", "token"}:
        request = warm.comparison.read(request_path)
        request["input_ids"][0 if defect == "history" else -1] += 1
        save(request_path, request)
        audit["local_input"]["file_sha256"] = digest(request_path)
        summary = warm.comparison.read(directory / "summary.json")
        summary["children"]["echo"]["profile"]["files"]["data"]["request.json"] = digest(
            request_path
        )
        save(directory / "summary.json", summary)
        save(audit_path, audit)
        refresh(bundle)
    elif defect == "official_completion":
        path = bundle["official_run"] / "sample_00/engine/complete.json"
        complete = warm.comparison.read(path)
        complete["row"]["output_ids"][0] += 1
        save(path, complete)
    else:
        (bundle["official_run"] / "run.json").write_text("{}")
    with pytest.raises(ValueError):
        warm.load_comparison(
            bundle["official"], directory, official_run_directory=bundle["official_run"]
        )


def test_overlapping_control_and_io_do_not_double_count_elapsed_time():
    # Compute 0..40 and 70..100; control 20..60 and 65..80; IO 35..50.
    # Independent expected partition: compute 70, IO-only 10, control-only 15, idle 5.
    rows = [
        {"start_ns": a, "end_ns": b, "lane": lane, "category": lane}
        for a, b, lane in [
            (0, 40, "Compute"),
            (70, 100, "Compute"),
            (20, 60, "GPU control"),
            (65, 80, "GPU control"),
            (35, 50, "D2H"),
        ]
    ]
    metrics = warm.interval_metrics(
        {
            "window": {"start_ns": 0, "end_ns": 100, "window_ms": 0.0001, "gpu_idle_ms": 0.000005},
            "rows": rows,
        }
    )
    assert {
        k: metrics[k + "_ns"]
        for k in ["compute_union", "io_only", "control_only", "idle", "original_control_lane_union"]
    } == {
        "compute_union": 70,
        "io_only": 10,
        "control_only": 15,
        "idle": 5,
        "original_control_lane_union": 55,
    }


def test_unknown_recall_is_control_even_when_displayed_as_h2d():
    panel = {
        "window": {"start_ns": 0, "end_ns": 100, "window_ms": 0.0001, "gpu_idle_ms": 0},
        "rows": [
            {"start_ns": 0, "end_ns": 40, "lane": "Compute", "category": "Attention"},
            {"start_ns": 40, "end_ns": 100, "lane": "H2D", "category": "ECHO recall (IO unknown)"},
        ],
    }
    metrics = warm.interval_metrics(panel)
    assert metrics["control_only_ns"] == 60
    assert metrics["io_only_ns"] == 0


def test_publish_selects_only_display_payloads_and_keeps_evidence_in_output(bundle, monkeypatch):
    monkeypatch.setattr(warm, "EXPERIMENT", bundle["experiment"])

    def draw(panels, output, **kwargs):
        assert kwargs["xlim_ms"] == (0, 2) and kwargs["compact_style"] is True
        assert len(panels) == 2
        for suffix in ("svg", "png"):
            (output / f"timeline_decode.{suffix}").write_text("render fixture")

    monkeypatch.setattr(warm.decode, "draw", draw)
    output = bundle["experiment"] / "output/data/new-comparison"
    published = bundle["experiment"] / "report/warm-comparison"
    result = warm.build_report(bundle["official"], bundle["warm"], output, published)
    assert result["numerical_acceptance_between_implementations"] is False
    assert {path.name for path in published.iterdir()} == set(warm.PUBLISHED_FILES)
    assert (output / "panels.json").is_file()
    assert (output / "source_snapshot").is_dir()
    assert (output / "input_reports/warm/summary.json").is_file()
    assert (output / "input_reports/warm_original/publication_manifest.json").is_file()
    assert (output / "input_reports/token_identity/local_request.json").is_file()
    warm.validate_manifest(published)
    assert (
        warm.comparison.read(published / "summary.json")[
            "official_profile_actual_mapped_host_bytes"
        ]
        is None
    )
    with pytest.raises(FileExistsError):
        warm.build_report(bundle["official"], bundle["warm"], output, published)


def test_publication_path_cannot_escape_report_directory(bundle):
    with pytest.raises(ValueError, match="Output must"):
        warm.build_report(bundle["official"], bundle["warm"], bundle["experiment"] / "bad-output")


def test_manifest_rejects_parent_traversal(tmp_path):
    directory = tmp_path / "publication"
    directory.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    save(
        directory / "publication_manifest.json",
        {"files_sha256": {"../outside.json": digest(outside)}},
    )
    with pytest.raises(ValueError, match="Payload escapes"):
        warm.validate_manifest(directory)
