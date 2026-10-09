"""Complete sample selection and identity checks for private 500-pair analysis."""

import hashlib
import json
import statistics
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.deepseek_v32_echo_official.src import analyze_q1_fused_prepare_model as analysis


def metrics():
    return {
        "record_bytes": 1152,
        "selection_records": 2048,
        "written_records": 1,
        "host_written_records": 1,
        "evicted_records": 0,
        "capacity_splits": 0,
        "transient_written_records": 0,
        "prefetched_records": 64,
        "prefetch_capacity_failures": 400,
        "recalled_records": 1983,
        "resident_selection_records": 65,
        "host_to_device_bytes": 2047 * 1152,
        "device_to_host_bytes": 1152,
    }


def saved_summary(rows):
    by_pair = {(row["pair"], row["arm"]): row["wall_ms"] for row in rows}
    deltas = [by_pair[pair, "candidate"] - by_pair[pair, "baseline"] for pair in range(500)]
    return {
        "median_wall_ms": {
            arm: statistics.median(row["wall_ms"] for row in rows if row["arm"] == arm)
            for arm in analysis.ARMS
        },
        "paired_delta_ms": deltas,
        "median_paired_delta_ms": statistics.median(deltas),
        "candidate_wins": sum(value < 0 for value in deltas),
        "order_median_delta_ms": {
            order: statistics.median(deltas[parity::2]) for parity, order in enumerate(("AB", "BA"))
        },
    }


@pytest.fixture
def bench():
    rows = []
    for pair in range(500):
        delta = (pair // 100 - 2) * 0.010 + (0.003 if pair % 2 == 0 else -0.003)
        for arm in analysis.ARMS if pair % 2 == 0 else analysis.ARMS[::-1]:
            rows.append(
                {
                    "pair": pair,
                    "order": "AB" if pair % 2 == 0 else "BA",
                    "arm": arm,
                    "wall_ms": 2.5 + (delta if arm == "candidate" else 0),
                    "layer_metrics": [metrics() for _ in range(3)],
                }
            )
    return {"pairs": 500, "result": {"samples": rows, "summary": saved_summary(rows)}}


def test_retains_all_pairs_and_exposes_order_and_sequential_block_effects(bench):
    result = analysis.analyze_samples(bench)
    assert len(result["paired_samples"]) == 500
    assert len(result["traffic"]["actual_samples"]) == 3000
    assert result["overall"]["pairs"] == 500
    assert result["overall"]["median_paired_delta_ms"] == pytest.approx(0)
    assert result["order_strata"]["AB"]["median_paired_delta_ms"] == pytest.approx(0.003)
    assert result["order_strata"]["BA"]["median_paired_delta_ms"] == pytest.approx(-0.003)
    blocks = result["sequential_blocks"]
    assert [block["median_paired_delta_ms"] for block in blocks] == pytest.approx(
        [-0.020, -0.010, 0, 0.010, 0.020]
    )
    assert [block["first_pair"] for block in blocks] == [0, 100, 200, 300, 400]
    assert all(block["pairs"] == 100 for block in blocks)
    assert all(
        block["order_strata"][order]["pairs"] == 50 for block in blocks for order in ("AB", "BA")
    )
    assert result["descriptive_block_sensitivity"]["overall"]["resamples"] == 3125
    assert (
        "not an independent-run confidence interval"
        in result["descriptive_block_sensitivity"]["overall"]["interpretation"]
    )


@pytest.mark.parametrize(
    "defect",
    [
        "count",
        "missing",
        "duplicate",
        "shuffle",
        "wrong_order",
        "pair_bool",
        "nan",
        "infinity",
        "negative",
        "zero",
        "wall_bool",
        "counter_bool",
        "negative_counter",
        "missing_layer",
        "bytes",
        "warm_residency",
        "saved_summary",
    ],
)
def test_rejects_incomplete_or_misrepresented_samples(bench, defect):
    rows = bench["result"]["samples"]
    if defect == "count":
        bench["pairs"] = 100
    elif defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif defect == "shuffle":
        rows[0], rows[1] = rows[1], rows[0]
    elif defect == "wrong_order":
        rows[0]["order"] = "BA"
    elif defect == "pair_bool":
        rows[0]["pair"] = False
    elif defect in {"nan", "infinity", "negative", "zero", "wall_bool"}:
        rows[0]["wall_ms"] = {
            "nan": float("nan"),
            "infinity": float("inf"),
            "negative": -1.0,
            "zero": 0.0,
            "wall_bool": True,
        }[defect]
    elif defect == "counter_bool":
        rows[0]["layer_metrics"][0]["written_records"] = True
    elif defect == "negative_counter":
        rows[0]["layer_metrics"][0]["prefetch_capacity_failures"] = -1
    elif defect == "missing_layer":
        rows[0]["layer_metrics"].pop()
    elif defect == "bytes":
        rows[0]["layer_metrics"][0]["host_to_device_bytes"] += 1152
    elif defect == "warm_residency":
        row = rows[0]["layer_metrics"][0]
        row.update(
            resident_selection_records=100,
            recalled_records=1948,
            host_to_device_bytes=(1948 + 64) * 1152,
        )
    elif defect == "saved_summary":
        bench["result"]["summary"]["candidate_wins"] = 500
    with pytest.raises(ValueError):
        analysis.analyze_samples(bench)


def test_descriptive_block_resampling_uses_all_five_prespecified_blocks():
    pairs = [{"block": index // 100, "delta_ms": float(index // 100)} for index in range(500)]
    result = analysis.block_sensitivity(pairs)
    assert result["resamples"] == 5**5
    assert result["median_paired_delta_percentile_band_ms"] == [0, 4]


def test_native_file_verification_rejects_changed_dso(tmp_path):
    library = tmp_path / "candidate.so"
    library.write_bytes(b"accepted native")
    expected = hashlib.sha256(library.read_bytes()).hexdigest()
    library.write_bytes(b"changed native")
    with pytest.raises(ValueError, match="Evidence changed"):
        analysis.verify_runtime_files(
            {"artifact_path": str(library), "artifact_sha256": expected}, analysis.Evidence()
        )


def test_runtime_header_hash_map_is_an_exact_directory_inventory(tmp_path):
    header = tmp_path / "nested/header.cuh"
    header.parent.mkdir()
    header.write_text("original header")
    node = {
        "path": str(tmp_path),
        "sha256": {"nested/header.cuh": hashlib.sha256(header.read_bytes()).hexdigest()},
        "matches_pinned_sources": True,
    }
    evidence = analysis.Evidence()
    analysis.verify_runtime_files(node, evidence)
    assert len(evidence.files) == 1
    (tmp_path / "unrecorded.cuh").write_text("new header")
    with pytest.raises(ValueError, match="directory inventory differs"):
        analysis.verify_runtime_files(node, analysis.Evidence())


@pytest.mark.parametrize("digest", [None, True, {}, "bad"])
def test_runtime_header_hash_map_does_not_relax_per_file_digest_validation(tmp_path, digest):
    (tmp_path / "header.cuh").write_text("header")
    with pytest.raises(ValueError, match="Invalid expected evidence digest"):
        analysis.verify_runtime_files(
            {"path": str(tmp_path), "sha256": {"header.cuh": digest}}, analysis.Evidence()
        )


def test_evidence_rechecks_a_previously_hashed_file_at_completion(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("{}")
    evidence = analysis.Evidence()
    evidence.file(path)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Evidence changed during analysis"):
        evidence.verify()


@pytest.mark.parametrize("digest", [None, True, "", "not-a-digest", "A" * 64])
def test_explicit_invalid_evidence_digest_is_not_an_omitted_constraint(tmp_path, digest):
    path = tmp_path / "file"
    path.write_text("accepted")
    with pytest.raises(ValueError, match="Invalid expected evidence digest"):
        analysis.Evidence().file(path, digest)


@pytest.fixture
def flashinfer_runtime(tmp_path):
    records, archived, observed = {}, {}, []
    for name in sorted(analysis.FLASHINFER_MODULES):
        original_dir = tmp_path / "runtime" / name
        archive_dir = tmp_path / "flashinfer_native" / name
        original_dir.mkdir(parents=True)
        archive_dir.mkdir(parents=True)
        files = {}
        for field, filename in (
            ("library", name + ".so"),
            ("build_metadata", "build.ninja"),
            ("dependency", "source.cuda.o.d"),
        ):
            path = original_dir / filename
            path.write_text("actual bytes " + name + " " + field)
            files[field] = analysis.Evidence().file(path)
        build_identity = {
            "sources": [],
            "request": {"name": name, "extra_cuda_cflags": ["-O3"], "extra_cflags": ["-O3"]},
        }
        record = {
            "schema": analysis.FLASHINFER_SCHEMA,
            "name": name,
            "cache_key": hashlib.sha256(
                json.dumps(build_identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "build_identity": build_identity,
            "library": files["library"],
            "build_metadata": files["build_metadata"],
            "dependencies": [],
            "dependency_metadata": [files["dependency"]],
        }
        path = original_dir / "record.json"
        path.write_text(json.dumps(record))
        record["manifest"] = analysis.Evidence().file(path)
        copies = {}
        for field in ("library", "build_metadata", "manifest"):
            source = Path(record[field]["path"])
            target = archive_dir / source.name
            target.write_bytes(source.read_bytes())
            copies[field] = {
                "original": record[field],
                "archived": analysis.Evidence().file(target),
            }
        dependency = files["dependency"]
        target = archive_dir / Path(dependency["path"]).name
        target.write_bytes(Path(dependency["path"]).read_bytes())
        copies["dependency_metadata"] = [
            {"original": dependency, "archived": analysis.Evidence().file(target)}
        ]
        records[name], archived[name] = record, copies
        observed.append(
            {
                "name": name,
                "loaded_in_this_process": True,
                "library": record["library"],
                "build_metadata": record["build_metadata"],
                "sources": [],
                "cuda_flags": ["-O3"],
                "cxx_flags": ["-O3"],
            }
        )
    manifest = {"schema": analysis.FLASHINFER_SCHEMA, "modules": archived}
    (tmp_path / "flashinfer_native/manifest.json").write_text(json.dumps(manifest))
    return {
        "private_flashinfer_native": records,
        "model": {"loaded_jit": {"native_jit": observed}},
    }


def test_flashinfer_archive_matches_actual_runtime_and_exact_libraries(
    tmp_path, flashinfer_runtime
):
    evidence = analysis.Evidence()
    analysis.audit_flashinfer(tmp_path, flashinfer_runtime, evidence)
    evidence.verify()
    assert len(evidence.files) == 13


@pytest.mark.parametrize(
    "defect",
    ["missing_module", "not_loaded", "wrong_mapping", "changed_elf", "missing_deps", "wrong_key"],
)
def test_flashinfer_archive_rejects_incomplete_or_unbound_artifacts(
    tmp_path, flashinfer_runtime, defect
):
    runtime = flashinfer_runtime
    if defect == "missing_module":
        runtime["private_flashinfer_native"].pop("rope")
    elif defect == "not_loaded":
        runtime["model"]["loaded_jit"]["native_jit"][0]["loaded_in_this_process"] = False
    elif defect == "wrong_mapping":
        runtime["model"]["loaded_jit"]["native_jit"][0]["library"] = {"path": "/other/rope.so"}
    elif defect == "changed_elf":
        (tmp_path / "flashinfer_native/rope/rope.so").write_text("different tmpxft metadata")
    elif defect == "wrong_key":
        runtime["private_flashinfer_native"]["rope"]["cache_key"] = "wrong"
    else:
        path = tmp_path / "flashinfer_native/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["modules"]["rope"]["dependency_metadata"] = []
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        analysis.audit_flashinfer(tmp_path, runtime, analysis.Evidence())


def test_run_rejects_wrong_receipt_before_reading_archives(tmp_path):
    value = {
        "completed": True,
        "mode": "bench",
        "run_id": tmp_path.name,
        "identity": {"test": "identity"},
        "result": {"passed": True, "receipt_sha256": "wrong"},
    }
    (tmp_path / "result.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Clean benchmark lacks"):
        analysis.audit_run(
            tmp_path,
            "bench",
            value["identity"],
            {"receipt_sha256": "expected"},
            analysis.Evidence(),
        )


@pytest.mark.parametrize("mode,completed", [("profile", True), ("bench", False)])
def test_run_rejects_profile_or_incomplete_results(tmp_path, mode, completed):
    value = {
        "completed": completed,
        "mode": mode,
        "run_id": tmp_path.name,
        "result": {"passed": True},
    }
    (tmp_path / "result.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Incomplete or mislabeled"):
        analysis.audit_run(tmp_path, "bench", {}, {}, analysis.Evidence())
