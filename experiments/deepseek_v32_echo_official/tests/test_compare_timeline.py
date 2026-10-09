"""Presentation readers preserve the original independent local profile bindings."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.deepseek_v32_echo_official.src import compare_decode_timeline as decode
from experiments.deepseek_v32_echo_official.src import compare_timeline as compare
from experiments.deepseek_v32_echo_official.src import diagnose_decode_gap as diagnose


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def seal(directory):
    write_json(
        directory / "publication_manifest.json",
        {
            "schema": "isolated-method-artifacts-v1",
            "files_sha256": {
                name: compare.engine.sha256(directory / name)
                for name in ("summary.json", "window_rows.json", "input_hashes.json")
            },
        },
    )


def panel(method, phase, shape):
    starts, ends = (100, 300, 500), (200, 400, 600)
    return {
        "method": method,
        "phase": phase,
        "shape": shape,
        "chunk": 63 if phase == "prefill" else None,
        "window": {"start_ns": 100, "end_ns": 600, "window_ms": 0.0005, "gpu_idle_ms": 0.0002},
        "layers": [
            {"layer": layer, "start_ns": 100 if layer == 0 else ends[layer - 1], "end_ns": end}
            for layer, end in enumerate(ends)
        ],
        "rows": [
            {
                "kind": "kernel",
                "lane": "Compute",
                "stage": "attention_projection",
                "purpose": "Projection / RoPE",
                "raw_start_ns": start,
                "raw_end_ns": end,
                "start_ns": start,
                "end_ns": end,
            }
            for start, end in zip(starts, ends, strict=True)
        ],
    }


@pytest.fixture
def isolated_report(tmp_path):
    directory = tmp_path / "published"
    directory.mkdir()
    config = {
        "prefix_tokens": 65536,
        "extend_tokens": 1,
        "chunk_size": 1024,
        "extend_chunk_size": 1,
        "num_layers": 3,
        "extend_residency": "cold",
        "method_isolation": "fresh-process-one-method-v1",
    }
    shape = compare.compact.profile_shape(config)
    summary = {
        "schema": "deepseek-v32-isolated-method-report-v1",
        "run_id": "cohort",
        "passed": True,
        "configuration": config,
        "methods": {},
        "children": {},
        "raw_windows": {},
    }
    rows = {"prefill": [], "extend": []}
    hashes = {}
    for method in compare.METHODS:
        children = {}
        for phase in ("check", "bench", "profile", "operators"):
            run_id = f"cohort_{method}_{phase}"
            # Original measurements need not remain mounted to render a saved report.
            original = Path("/unmounted_original_runs") / run_id
            digest = f"{method}-{phase}-result"
            children[phase] = {
                "run_id": run_id,
                "directory": str(original),
                "result_sha256": digest,
            }
            hashes[str(original / "result.json")] = digest
        check = children["check"]
        check.update(
            receipt_path=str(Path(check["directory"]) / "receipt.json"),
            receipt_sha256=f"{method}-receipt-file-hash",
        )
        hashes[check["receipt_path"]] = check["receipt_sha256"]
        summary["children"][method] = children
        summary["methods"][method] = {}
        summary["raw_windows"][method] = {}
        for phase, capture in (("prefill", 2), ("extend", 4)):
            current = panel(method, phase, shape)
            profile = children["profile"]
            sqlite = str(Path(profile["directory"]) / f"capture_{capture}.sqlite")
            hashes[sqlite] = f"{method}-{phase}-sqlite"
            current["provenance"] = {
                "profile_run_id": profile["run_id"],
                "profile_directory": profile["directory"],
                "result_sha256": profile["result_sha256"],
                "sqlite": sqlite,
                "sqlite_sha256": hashes[sqlite],
                "benchmark": children["bench"],
                "validation_receipt": {
                    "receipt_path": check["receipt_path"],
                    "receipt_sha256": "signed-receipt-payload-hash",
                },
            }
            rows[phase].append(current)
            summary["raw_windows"][method][phase] = {
                **current["window"],
                "native_gpu_intervals": len(current["rows"]),
            }
    write_json(directory / "summary.json", summary)
    write_json(directory / "window_rows.json", rows)
    write_json(directory / "input_hashes.json", hashes)
    seal(directory)
    return directory


@pytest.mark.parametrize("phase", ["prefill", "extend"])
def test_isolated_report_keeps_every_method_profile_and_uses_only_saved_artifacts(
    isolated_report, phase
):
    panels, sources = compare.mfu_panels(isolated_report, None, phase, receipt_subdir="h65536_a1")
    assert {panel["profile_run_id"] for panel in panels} == {
        f"cohort_{method}_profile" for method in compare.METHODS
    }
    assert len(sources) == 4 and all(path.is_file() for path in sources)
    assert sources[0].name == "summary.json"
    for current in panels:
        assert current["profile_run_id"] == current["provenance"]["profile_run_id"]
        assert current["idle_intervals_ns"] == [[200, 300], [400, 500]]
        assert {row["category"] for row in current["rows"]} == {"Projection / RoPE"}
    assert (
        compare.mfu_report_shape(isolated_report, receipt_subdir="h65536_a1")["extend_tokens"] == 1
    )


@pytest.mark.parametrize("name", ["summary.json", "window_rows.json", "input_hashes.json"])
def test_isolated_reader_rejects_changed_published_assets(isolated_report, name):
    (isolated_report / name).write_text("{}")
    with pytest.raises(ValueError, match="Published input changed"):
        compare.mfu_panels(isolated_report, None, "extend")


@pytest.mark.parametrize(
    "defect",
    [
        "failed",
        "shape",
        "missing_method",
        "mixed_profile",
        "wrong_benchmark",
        "wrong_receipt",
        "unbound_sqlite",
        "window",
        "idle",
        "unknown_lane",
        "phase",
        "chunk",
    ],
)
def test_isolated_reader_rejects_invalid_resealed_metadata(isolated_report, defect):
    summary = compare.read(isolated_report / "summary.json")
    rows = compare.read(isolated_report / "window_rows.json")
    current = rows["extend"][1]
    if defect == "failed":
        summary["passed"] = False
    elif defect == "shape":
        summary["configuration"]["extend_tokens"] = 128
    elif defect == "missing_method":
        rows["extend"].pop()
    elif defect == "mixed_profile":
        current["provenance"] = deepcopy(rows["extend"][0]["provenance"])
    elif defect == "wrong_benchmark":
        current["provenance"]["benchmark"] = deepcopy(rows["extend"][0]["provenance"]["benchmark"])
    elif defect == "wrong_receipt":
        current["provenance"]["validation_receipt"]["receipt_path"] = "/wrong/receipt.json"
    elif defect == "unbound_sqlite":
        current["provenance"]["sqlite_sha256"] = "unrelated-sqlite"
    elif defect == "window":
        current["window"]["window_ms"] *= 2
    elif defect == "idle":
        current["rows"][0]["raw_end_ns"] = 201
    elif defect == "unknown_lane":
        current["rows"][0]["lane"] = "invented"
    elif defect == "phase":
        current["phase"] = "prefill"
    elif defect == "chunk":
        current["chunk"] = 63
    write_json(isolated_report / "summary.json", summary)
    write_json(isolated_report / "window_rows.json", rows)
    seal(isolated_report)
    with pytest.raises(ValueError):
        compare.mfu_panels(isolated_report, None, "extend")


@pytest.fixture
def legacy_report(tmp_path):
    directory = tmp_path / "legacy"
    directory.mkdir()
    shape = {"prefix_tokens": 65536, "extend_tokens": 128, "chunk_size": 1024}
    rows = {
        phase: [panel(method, phase, shape) for method in compare.METHODS]
        for phase in ("prefill", "extend")
    }
    write_json(directory / "window_rows.json", rows)
    receipt = {
        "profile_run_id": "original_mixed_profile",
        "shape": shape,
        "artifacts_sha256": {
            "window_rows.json": compare.engine.sha256(directory / "window_rows.json")
        },
        "metrics": {
            f"{phase}/{method}": {
                "window": rows[phase][index]["window"],
                "compute_categories": {"Projection / RoPE": 3},
                "annotation_intervals_ns": {"GPU idle": [[200, 300], [400, 500]]},
            }
            for index, method in enumerate(compare.METHODS)
            for phase in rows
        },
    }
    write_json(directory / "compact_receipt.json", receipt)
    write_json(
        directory / "provenance.json",
        {
            "inputs_sha256": {
                str(directory / "window_rows.json"): receipt["artifacts_sha256"]["window_rows.json"]
            }
        },
    )
    write_json(
        directory / "publication_manifest.json",
        {
            "schema": "selected-full-graph-artifacts-v3",
            "files_sha256": {
                name: compare.engine.sha256(directory / name)
                for name in ("compact_receipt.json", "provenance.json")
            },
        },
    )
    return directory


def test_existing_four_method_reader_preserves_receipt_and_shared_run_semantics(legacy_report):
    panels, _ = compare.mfu_panels(legacy_report, None, "extend")
    assert {current["profile_run_id"] for current in panels} == {"original_mixed_profile"}
    assert compare.mfu_report_shape(legacy_report)["extend_tokens"] == 128


def test_existing_reader_still_rejects_changed_classification(legacy_report):
    receipt = compare.read(legacy_report / "compact_receipt.json")
    receipt["metrics"]["extend/hbm"]["compute_categories"] = {"Attention": 3}
    write_json(legacy_report / "compact_receipt.json", receipt)
    publication = compare.read(legacy_report / "publication_manifest.json")
    publication["files_sha256"]["compact_receipt.json"] = compare.engine.sha256(
        legacy_report / "compact_receipt.json"
    )
    write_json(legacy_report / "publication_manifest.json", publication)
    with pytest.raises(ValueError, match="compute categories differ"):
        compare.mfu_panels(legacy_report, None, "extend")


def test_decode_entry_reads_isolated_summary_shape_and_preserves_numerical_boundary(
    isolated_report, tmp_path, monkeypatch
):
    experiment = tmp_path / "official"
    monkeypatch.setattr(decode, "EXPERIMENT", experiment)
    output = experiment / "output/comparison"
    published = experiment / "report/comparison"
    official = []
    for method in ("hbm", "echo"):
        current = panel(method, "decode", {})
        current.update(
            source="Official SGLang",
            condition="normal decode; natural residency",
            profile_run_id=f"official_{method}",
        )
        official.append(current)
    monkeypatch.setattr(decode, "official_panels", lambda _: (official, []))

    def draw(panels, output):
        for suffix in ("svg", "png"):
            (output / f"timeline_decode.{suffix}").write_text("test figure")

    monkeypatch.setattr(decode, "draw", draw)
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_decode_timeline",
            "--mfu-report",
            str(isolated_report),
            "--output-dir",
            str(output),
            "--publish-dir",
            str(published),
        ],
    )
    decode.main()
    provenance = compare.read(output / "provenance.json")
    assert provenance["numerical_acceptance_between_implementations"] is False
    assert provenance["local_profile_run_ids"] == {
        method: f"cohort_{method}_profile" for method in compare.METHODS
    }
    assert "input token 57841" in provenance["conditions"]["official"]
    assert "token 111090" in provenance["conditions"]["local"]
    saved = compare.read(output / "panels.json")
    assert [row["profile_run_id"] for row in saved[2:]] == list(
        provenance["local_profile_run_ids"].values()
    )
    artifacts = {
        path.relative_to(output): compare.engine.sha256(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    assert Path("panels.json") in artifacts
    assert any(path.parts[0] == "source_snapshot" for path in artifacts)
    assert artifacts == {
        path.relative_to(published): compare.engine.sha256(path)
        for path in published.rglob("*")
        if path.is_file()
    }


def test_prefill_entry_records_per_method_ids_without_a_shared_profile_id(
    isolated_report, tmp_path, monkeypatch
):
    experiment = tmp_path / "official"
    monkeypatch.setattr(compare, "EXPERIMENT", experiment)
    output = experiment / "output/prefill"
    monkeypatch.setattr(compare, "engine_panels", lambda *_: ([], []))
    monkeypatch.setattr(compare, "draw", lambda *_: [0.0, 30.0])
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_timeline",
            "--phase",
            "prefill",
            "--mfu-report",
            str(isolated_report),
            "--output-dir",
            str(output),
        ],
    )
    compare.main()
    provenance = compare.read(output / "provenance.json")
    assert "mfu_profile_run_id" not in provenance
    assert provenance["mfu_profile_run_ids"] == {
        method: f"cohort_{method}_profile" for method in compare.METHODS
    }
    assert set(provenance["mfu_panel_provenance"]) == set(compare.METHODS)


def test_engine_extend_does_not_relabel_an_isolated_a1_window_as_a128(
    isolated_report, tmp_path, monkeypatch
):
    experiment = tmp_path / "official"
    monkeypatch.setattr(compare, "EXPERIMENT", experiment)
    output = experiment / "output/wrong_shape"
    monkeypatch.setattr(compare, "engine_panels", lambda *_: ([], []))
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_timeline",
            "--phase",
            "extend",
            "--mfu-report",
            str(isolated_report),
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="use compare_decode_timeline"):
        compare.main()
    assert not output.exists()


@pytest.fixture
def isolated_comparison(isolated_report, tmp_path):
    official = tmp_path / "official_report"
    official.mkdir()
    panels = []
    for method in ("hbm", "echo"):
        current = panel(method, "decode", {})
        current.update(source="Official SGLang", condition="normal decode; natural residency")
        panels.append(current)
    windows = [current["window"] for current in panels]
    write_json(official / "report.json", {"numerical_acceptance": False, "windows": windows})
    write_json(official / "windows.json", windows)
    write_json(official / "activities.json", {p["method"]: p["rows"] for p in panels})
    write_json(
        official / "publication.json",
        {
            "files": {
                name: diagnose.sha256(official / name)
                for name in ("report.json", "windows.json", "activities.json")
            }
        },
    )
    local, _ = compare.mfu_panels(isolated_report, None, "extend")
    for current in local:
        current.update(source="Ours (local MFU)", condition="fixed token")
    directory = tmp_path / "comparison"
    directory.mkdir()
    write_json(directory / "panels.json", panels + local)
    write_json(
        directory / "provenance.json",
        {
            "source_snapshot_sha256": {},
            "inputs_and_sources_sha256": {},
            "artifacts_sha256": {"panels.json": diagnose.sha256(directory / "panels.json")},
        },
    )
    return directory, official, isolated_report


def test_gap_audit_reads_isolated_panels_without_a_mixed_profile_receipt(isolated_comparison):
    sources = {}
    panels, official, context = diagnose.load_panels(*isolated_comparison, sources)
    assert context["format"] == "isolated-method-artifacts-v1"
    assert official["numerical_acceptance"] is False
    assert [current["profile_run_id"] for current in panels[2:]] == [
        f"cohort_{method}_profile" for method in compare.METHODS
    ]
    assert any(Path(name).name == "input_hashes.json" for name in sources)
    assert not any(Path(name).name == "compact_receipt.json" for name in sources)


@pytest.mark.parametrize("defect", ["profile", "category", "timestamp", "layers"])
def test_gap_audit_rejects_resealed_comparison_that_differs_from_method_source(
    isolated_comparison, defect
):
    directory = isolated_comparison[0]
    panels = diagnose.read(directory / "panels.json")
    current = panels[3]
    if defect == "profile":
        current["provenance"] = deepcopy(panels[2]["provenance"])
    elif defect == "category":
        current["rows"][0]["category"] = "Attention"
    elif defect == "timestamp":
        current["rows"][0]["start_ns"] += 1
    elif defect == "layers":
        current["layers"][1]["end_ns"] += 1
    write_json(directory / "panels.json", panels)
    provenance = diagnose.read(directory / "provenance.json")
    provenance["artifacts_sha256"]["panels.json"] = diagnose.sha256(directory / "panels.json")
    write_json(directory / "provenance.json", provenance)
    with pytest.raises(ValueError, match="Local isolated panel differs"):
        diagnose.load_panels(*isolated_comparison, {})
