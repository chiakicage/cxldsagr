"""Publish clean timings and separately measured timelines for the H x A matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

from evaluation.validation import identity_digest
from experiments.deepseek_v32_mfu.src import shape_matrix_sources
from experiments.deepseek_v32_mfu.src.run_contract import (
    METHODS,
    benchmark_view,
    digest,
    execution_identity,
)

HISTORY_TOKENS = (4096, 16384, 65536)
EXTEND_TOKENS = (128, 256, 512, 1024)
SHAPES = {(history, extend) for history in HISTORY_TOKENS for extend in EXTEND_TOKENS}
ROOT = Path(__file__).resolve().parents[3]
TIMELINE_FILES = ("prefill.svg", "prefill.png", "extend.svg", "extend.png")
STARTUP_FILES = ("extend_with_startup.svg", "extend_with_startup.png")
COMMON_FIELDS = (
    "scope",
    "num_layers",
    "chunk_size",
    "extend_residency",
    "extend_graph_policy_revision",
    "compute_graphs",
    "compute_precision",
    "torch_precision",
    "checkpoint_identity",
    "dependencies",
    "source_sha256",
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def table(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path, *, allow_subset=False):
    manifest = read(path)
    if manifest.get("schema_version") != 1 or not manifest.get("run_id"):
        raise ValueError("Expected a schema-1 shape matrix manifest with a run_id")
    entries = manifest["shapes"]
    shapes = [(row["prefix_tokens"], row["extend_tokens"]) for row in entries]
    if len(shapes) != len(set(shapes)) or not shapes:
        raise ValueError("Shape matrix must be nonempty and have no duplicate shapes")
    if any(type(value) is not int for shape in shapes for value in shape):
        raise ValueError("Token counts must be integers")
    if not set(shapes) <= SHAPES or (not allow_subset and set(shapes) != SHAPES):
        raise ValueError("Expected the complete [4096,16384,65536] x [128,256,512,1024] matrix")
    configuration = manifest["configuration"]
    requested = {
        (history, extend)
        for history in configuration["prefix_tokens"]
        for extend in configuration["extend_tokens"]
    }
    if requested != set(shapes):
        raise ValueError("Manifest shape entries do not cover its configured Cartesian product")
    return manifest, sorted(entries, key=lambda row: (row["prefix_tokens"], row["extend_tokens"]))


def validate_timeline(directory, profile_directory, profile, *, startup=False):
    directory = Path(directory).resolve(strict=True)
    receipt = read(directory / "compact_receipt.json")
    expected = {
        "profile_run_id": profile["run_id"],
        "layout": "separate",
        "window_kind": "extend-startup" if startup else "three-layers",
        "io_layout": "directions",
        "annotations": "idle-echo",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Timeline profile, windows or layout differ from the requested report")
    if "shape" in receipt and any(
        receipt["shape"][key] != profile[key]
        for key in ("prefix_tokens", "extend_tokens", "chunk_size")
    ):
        raise ValueError("Timeline shape annotation differs from the measured profile")
    required = STARTUP_FILES if startup else TIMELINE_FILES
    required += ("windows.csv", "annotations.csv")
    hashes = receipt["artifacts_sha256"]
    if not set(required) <= hashes.keys():
        raise ValueError("Timeline is missing required figures or window tables")
    for name, expected_hash in hashes.items():
        source = (directory / name).resolve(strict=True)
        if not source.is_relative_to(directory) or digest(source) != expected_hash:
            raise ValueError(f"Timeline artifact changed: {name}")
    inputs = receipt["inputs_and_sources_sha256"]
    for name in ("result.json", "gap_audit.json", "operator_calls.json"):
        path = profile_directory / name
        if inputs.get(str(path)) != digest(path):
            raise ValueError(f"Timeline input differs from the selected profile: {name}")
    phases = ("extend",) if startup else ("prefill", "extend")
    if set(receipt["metrics"]) != {f"{phase}/{method}" for phase in phases for method in METHODS}:
        raise ValueError("Timeline does not cover every requested phase and method")
    selected = [directory / name for name in (*required, "compact_receipt.json")]
    return receipt, selected


def validate_cache_samples(bench):
    """Bind each clean wall sample to its own post-execution cache counters."""
    required = {
        "prefetched_records",
        "recalled_records",
        "evicted_records",
        "prefetch_capacity_failures",
        "host_to_device_bytes",
        "device_to_host_bytes",
        "record_bytes",
    }
    for method in METHODS:
        row = bench["measurements"][method]
        for phase, final_key in (
            ("prefill", "prefix_cache_per_layer"),
            ("extend", "extend_cache_per_layer"),
        ):
            samples = row.get(phase + "_cache_samples")
            if not isinstance(samples, list) or len(samples) != len(row[phase + "_samples_ms"]):
                raise ValueError(f"Missing per-sample cache metrics: {method}/{phase}")
            for layers in samples:
                if not isinstance(layers, list) or len(layers) != bench["num_layers"]:
                    raise ValueError(f"Incomplete per-layer cache metrics: {method}/{phase}")
                for layer in layers:
                    if not isinstance(layer, dict) or not required <= layer.keys():
                        raise ValueError(f"Incomplete cache counter fields: {method}/{phase}")
                    if any(
                        type(value) not in (int, float, str, bool)
                        or type(value) in (int, float)
                        and (not math.isfinite(value) or value < 0)
                        for value in layer.values()
                    ) or any(type(layer[key]) is not int for key in required):
                        raise ValueError(f"Invalid scalar cache metrics: {method}/{phase}")
            if samples[-1] != row[final_key]:
                raise ValueError(f"Final cache counters differ from last sample: {method}/{phase}")


def validate_shape(entry):
    history, extend = entry["prefix_tokens"], entry["extend_tokens"]
    directories = {
        name: Path(entry[f"{name}_run"]).resolve(strict=True)
        for name in ("check", "bench", "profile")
    }
    if len(set(directories.values())) != 3:
        raise ValueError("Check, clean benchmark and intrusive profile must use independent runs")
    runs = {name: read(path / "result.json") for name, path in directories.items()}
    for name, run in runs.items():
        if (
            run.get("mode") != name
            or run.get("accepted") is not True
            or run.get("extend_graph") is not True
            or run.get("num_layers") != 3
            or run.get("extend_residency") != "cold"
            or run.get("methods") != list(METHODS)
            or (run["prefix_tokens"], run["extend_tokens"]) != (history, extend)
            or run["extend_chunk_size"] != extend
            or run["chunk_size"] != 1024
            or run["sparse_pool_tokens"] != history + extend
            or run["host_arena_tokens"] != history + extend
        ):
            raise ValueError(f"Run does not cover the declared cold full-graph shape: {name}")
        if entry.get(f"{name}_run_id", run["run_id"]) != run["run_id"]:
            raise ValueError(f"Manifest run ID differs from the source: {name}")
        if identity_digest(run["execution_identity"]) != identity_digest(execution_identity(run)):
            raise ValueError(f"Stored execution identity differs from source metadata: {name}")
        if (
            shape_matrix_sources.HARDWARE_SOURCE in run["source_sha256"]
            and run["hardware"].get("hardware_source_sha256")
            != run["source_sha256"][shape_matrix_sources.HARDWARE_SOURCE]
        ):
            raise ValueError(f"Hardware evidence source identity differs: {name}")
    identities = {identity_digest(run["execution_identity"]) for run in runs.values()}
    if len(identities) != 1:
        raise ValueError("Check, benchmark and profile execution identities differ")
    profile = runs["profile"]
    benchmark_view(directories["profile"], profile)
    validate_cache_samples(runs["bench"])
    if Path(profile["benchmark"]["directory"]).resolve() != directories["bench"]:
        raise ValueError("Profile binds a different independent benchmark")
    receipt_path = directories["check"] / "receipt.json"
    if Path(profile["validation_receipt"]["receipt_path"]).resolve() != receipt_path:
        raise ValueError("Profile binds a different numerical check")
    gap_path = directories["profile"] / "gap_audit.json"
    gap = read(gap_path)
    if gap["run_id"] != profile["run_id"] or gap["input_result_sha256"] != digest(
        directories["profile"] / "result.json"
    ):
        raise ValueError("Gap audit belongs to a different profile")
    if [row["method"] for row in gap["methods"]] != list(METHODS):
        raise ValueError("Gap audit does not cover the four methods")
    for row in gap["methods"]:
        graph = row["graph_attribution"]
        if (
            graph.get("replays") != 1
            or graph.get("full_extend_graph") is not True
            or graph.get("every_replay_gpu_node_verified") is not True
        ):
            raise ValueError("Profile does not prove one complete graph replay per extend")
    timeline, selected = validate_timeline(entry["timeline_run"], directories["profile"], profile)
    startup = None
    startup_selected = []
    if entry.get("startup_run"):
        startup, startup_selected = validate_timeline(
            entry["startup_run"], directories["profile"], profile, startup=True
        )
    evidence = [path / "result.json" for path in directories.values()]
    evidence += [
        receipt_path,
        gap_path,
        directories["profile"] / "operator_calls.json",
        *selected,
        *startup_selected,
    ]
    return {
        "entry": entry,
        "directories": directories,
        "runs": runs,
        "gap": gap,
        "timeline": timeline,
        "startup": startup,
        "selected": selected,
        "startup_selected": startup_selected,
        "evidence": evidence,
    }


def report_rows(shapes):
    timings, samples, windows = [], [], []
    for shape in shapes:
        entry, bench = shape["entry"], shape["runs"]["bench"]
        key = {"prefix_tokens": entry["prefix_tokens"], "extend_tokens": entry["extend_tokens"]}
        graphs = {row["method"]: row["graph_attribution"] for row in shape["gap"]["methods"]}
        for method in METHODS:
            row = bench["measurements"][method]
            timings.append(
                {
                    **key,
                    "method": method,
                    "prefill_median_ms": row["prefill_median_ms"],
                    "extend_median_ms": row["extend_median_ms"],
                    "warmups": bench["warmups"],
                    "prefill_samples": bench["prefill_repeats"],
                    "extend_samples": bench["repeats"],
                    "extend_graph_replays": graphs[method]["replays"],
                    "profile_graph_gpu_nodes": graphs[method]["gpu_node_activities"],
                    "check_run_id": shape["runs"]["check"]["run_id"],
                    "bench_run_id": bench["run_id"],
                    "profile_run_id": shape["runs"]["profile"]["run_id"],
                }
            )
            for phase in ("prefill", "extend"):
                for index, value in enumerate(row[phase + "_samples_ms"]):
                    samples.append(
                        {
                            **key,
                            "method": method,
                            "phase": phase,
                            "sample": index,
                            "wall_ms": value,
                            "bench_run_id": bench["run_id"],
                        }
                    )
        for view, receipt in (
            ("three-layers", shape["timeline"]),
            ("extend-startup", shape["startup"]),
        ):
            if receipt is None:
                continue
            for label, metrics in receipt["metrics"].items():
                phase, method = label.split("/")
                window = metrics["window"]
                windows.append(
                    {
                        **key,
                        "view": view,
                        "phase": phase,
                        "method": method,
                        **{
                            name: window[name]
                            for name in (
                                "start_ns",
                                "end_ns",
                                "window_ms",
                                "gap_ms",
                                "gap_no_io_percent",
                                "gpu_idle_ms",
                            )
                        },
                        "profile_run_id": receipt["profile_run_id"],
                    }
                )
    return timings, samples, windows


def cache_sample_rows(shapes):
    rows = []
    for shape in shapes:
        entry, bench = shape["entry"], shape["runs"]["bench"]
        for method in METHODS:
            measurement = bench["measurements"][method]
            for phase in ("prefill", "extend"):
                for sample, layers in enumerate(measurement[phase + "_cache_samples"]):
                    for layer, metrics in enumerate(layers):
                        rows.append(
                            {
                                "prefix_tokens": entry["prefix_tokens"],
                                "extend_tokens": entry["extend_tokens"],
                                "method": method,
                                "phase": phase,
                                "sample": sample,
                                "layer": layer,
                                "wall_ms": measurement[phase + "_samples_ms"][sample],
                                "bench_run_id": bench["run_id"],
                                **{f"cache_{key}": value for key, value in metrics.items()},
                            }
                        )
    return rows


def markdown(manifest, shapes, timings, *, source_compatibility=None):
    lines = [
        "# DeepSeek V3.2 shape matrix",
        "",
        f"Matrix run: `{manifest['run_id']}`. Each shape has independent cold correctness, clean timing and NSYS profile runs.",
        "",
        "The real checkpoint's L0–L2 propagate hidden/residual in order, with embedding, three dense MLPs, final norm and last-token LM head. History uses 1,024-token chunks; extend uses one complete A-token batch and one full CUDA Graph replay. P=NH=H+A; ordinary persistent append. This scope does not represent the full 61-layer model or C10 GR serving.",
        "",
        "Times below are medians of synchronized wall-time samples. Input preparation, transactions and required synchronization/commit are included; weight loading, compilation, graph preparation and prefix restoration are excluded. Cold clears offload main-KV HBM residency while retaining DRAM and resident indexer data. Sample counts and all unrounded values are in [timing.csv](timing.csv) and [timing_samples.csv](timing_samples.csv).",
        "",
        "[cache_metrics_samples.csv](cache_metrics_samples.csv) preserves each timed sample's actual per-layer counters and traffic, read after the wall timer stops. ECHO's bounded atomic prefetch can select different eligible records across executions when the cap is saturated, so prefetch/recall counts may differ. Check, clean benchmark and profile counters describe their respective executions and are not substituted for one another.",
        "",
    ]
    batches = manifest.get("execution_batches", [])
    deadline_differs = bool(
        source_compatibility and source_compatibility["matrix_hardware_probe_deadline_differs"]
    )
    if len(batches) > 1:
        source_note = (
            "the same model and measurement execution code; the hardware query deadline differs from 20 to 120 seconds"
            if deadline_differs
            else "the same frozen measured sources"
        )
        batch_note = f"This matrix combines independently completed shape runs from {len(batches)} execution batches with {source_note}. "
        timeouts = {}
        for batch in batches:
            interruption = batch.get("preflight_interruption", {})
            if (
                interruption.get("stage") == "before_model_loading"
                and interruption.get("command") == "nvidia-smi"
                and interruption.get("timeout_seconds") is not None
            ):
                timeout = interruption["timeout_seconds"]
                timeouts[timeout] = timeouts.get(timeout, 0) + 1
        for timeout, count in timeouts.items():
            subject = "A prior preflight" if count == 1 else f"{count} prior preflights"
            batch_note += (
                f"{subject} stopped before model loading after a `nvidia-smi` hardware query "
                f"exceeded the {timeout}-second timeout; the cause of the query delay is not established. "
            )
        lines += [
            batch_note
            + "The [exact input matrix manifest](matrix_manifest.json) preserves detailed batch metadata and run membership.",
            "",
        ]
    if deadline_differs:
        lines += [
            "The source comparison verifies that the archived hardware-probe variants and current source differ only in the single `nvidia-smi` subprocess timeout line. Each shape retains exact check, benchmark and profile execution identities. Original hashes, variant paths and the comparison rule are recorded in [summary.json](summary.json).",
            "",
        ]
    by_shape = {}
    for row in timings:
        by_shape.setdefault((row["prefix_tokens"], row["extend_tokens"]), {})[row["method"]] = row
    for phase in ("prefill", "extend"):
        lines += [
            f"## Complete {phase} wall time (ms)",
            "",
            "| H | A | " + " | ".join(f"`{method}`" for method in METHODS) + " |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for (history, extend), methods in by_shape.items():
            lines.append(
                f"| {history:,} | {extend:,} | "
                + " | ".join(f"{methods[method][phase + '_median_ms']:.3f}" for method in METHODS)
                + " |"
            )
        lines.append("")
    lines += [
        "## Timelines",
        "",
        "Each link compares all four methods from the shape's separate intrusive NSYS capture. Prefill shows only the final 1,024-token chunk's L0–L2; extend shows the complete A-token batch's L0–L2. Main windows end at the last L2 compute kernel and start at first L0 compute, including earlier L0 history H2D for dense extend. Startup views begin at forward entry and retain embedding. Profile windows are distinct from the complete wall-time boundary above.",
        "",
        "Compute phases share one lane. H2D and D2H have separate lanes; red marks GPU idle, and ECHO prepare/finalize/hint have their own lane. Absolute timeline scales are shared across methods within a shape and phase, and may differ across shapes. Exact windows and gap values are in [timeline_windows.csv](timeline_windows.csv).",
        "",
        "| H | A | Final prefill chunk | Complete extend layers | Extend with startup |",
        "| ---: | ---: | --- | --- | --- |",
    ]
    for shape in shapes:
        history, extend = shape["entry"]["prefix_tokens"], shape["entry"]["extend_tokens"]
        directory = f"h{history}_a{extend}"
        startup = (
            f"[SVG]({directory}/with_startup/extend_with_startup.svg)"
            if shape["startup"]
            else "Not captured"
        )
        lines.append(
            f"| {history:,} | {extend:,} | [SVG]({directory}/prefill.svg) | [SVG]({directory}/extend.svg) | {startup} |"
        )
    if manifest.get("observer"):
        lines += [
            "",
            "Discrete GPU/process observations are preserved in the [observer summary](audit/observer.json). These observations do not establish continuous GPU isolation or CPU/DRAM isolation.",
        ]
    lines += [
        "",
        "Run IDs, execution identities, hardware, precision, dependency versions and input hashes are recorded in [summary.json](summary.json) and [provenance.json](provenance.json). [publication_manifest.json](publication_manifest.json) binds the selected report files. This matrix adds timing and timelines; it does not estimate new per-operator MFU values.",
        "",
    ]
    return "\n".join(lines)


def validate_audit_history(audit):
    expected = {}
    for row in audit["shapes"]:
        expected.setdefault(row["prefix_tokens"], {})[str(row["extend_tokens"])] = row.get(
            "request_prefix", {}
        ).get("prefix_token_ids_sha256")
    assertions = audit.get("history_identity_across_extend_sizes")
    if (
        not isinstance(assertions, list)
        or len(assertions) != len(expected)
        or {item.get("prefix_tokens") for item in assertions} != set(expected)
    ):
        raise ValueError("Independent audit history assertion coverage differs")
    for item in assertions:
        hashes = expected[item["prefix_tokens"]]
        extends = item.get("extend_tokens")
        if (
            item.get("same_prefix_token_ids_sha256") is not True
            or not isinstance(extends, list)
            or len(extends) != len(hashes)
            or set(extends) != {int(extend) for extend in hashes}
            or item.get("prefix_token_ids_sha256_by_extend_tokens") != hashes
            or not all(isinstance(value, str) and value for value in hashes.values())
            or len(set(hashes.values())) != 1
        ):
            raise ValueError(
                "Independent audit history prefix identity differs across extend sizes"
            )


def publish(manifest_path, output, *, publish_dir=None, audit_paths=(), allow_subset=False):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest, entries = load_manifest(manifest_path, allow_subset=allow_subset)
    audits = [Path(path).resolve(strict=True) for path in audit_paths]
    if publish_dir is not None and not audits:
        raise ValueError("Publication requires a complete matching independent audit")
    shapes = [validate_shape(entry) for entry in entries]
    source_inputs = {}

    def source_digest(path):
        path = Path(path).resolve(strict=True)
        value = digest(path)
        if path in source_inputs and source_inputs[path] != value:
            raise ValueError(f"Source changed during comparison: {path}")
        source_inputs[path] = value
        return value

    snapshots = [
        shape_matrix_sources.source_snapshot(
            shape["runs"][mode]["source_sha256"],
            shape["directories"][mode] / "source",
            current_hardware=ROOT / shape_matrix_sources.HARDWARE_SOURCE,
            digest_file=source_digest,
        )
        for shape in shapes
        for mode in ("check", "bench", "profile")
    ]
    source_compatibility = shape_matrix_sources.compare_source_snapshots(snapshots)
    first = shapes[0]["runs"]["bench"]
    common = {key: first[key] for key in COMMON_FIELDS}
    common["gpu"] = first["execution_identity"]["gpu"]
    for shape in shapes[1:]:
        bench = shape["runs"]["bench"]
        candidate = {key: bench[key] for key in COMMON_FIELDS}
        candidate["gpu"] = bench["execution_identity"]["gpu"]
        if identity_digest(
            {key: value for key, value in candidate.items() if key != "source_sha256"}
        ) != identity_digest(
            {key: value for key, value in common.items() if key != "source_sha256"}
        ):
            raise ValueError("Matrix mixes implementation, checkpoint, GPU or precision identities")
    if source_compatibility["matrix_hardware_probe_deadline_differs"]:
        del common["source_sha256"]
    audit_inputs = {}
    complete_audit = False
    for path in audits:
        audit = read(path)
        if (
            audit.get("passed") is not True
            or audit.get("run_id") != manifest["run_id"]
            or audit.get("manifest_sha256") != digest(manifest_path)
            or {(row["prefix_tokens"], row["extend_tokens"]) for row in audit["shapes"]}
            != {(row["prefix_tokens"], row["extend_tokens"]) for row in entries}
            or len(audit["shapes"]) != len(entries)
            or not all(row.get("passed") is True for row in audit["shapes"])
            or not audit.get("input_sha256")
        ):
            raise ValueError(f"Independent audit has not passed: {path}")
        validate_audit_history(audit)
        complete_audit |= audit.get("complete_matrix") is True
        for filename, expected_hash in audit["input_sha256"].items():
            filename = str(Path(filename).resolve(strict=True))
            if filename in audit_inputs and audit_inputs[filename] != expected_hash:
                raise ValueError(f"Independent audits disagree on source identity: {filename}")
            audit_inputs[filename] = expected_hash
    if publish_dir is not None and not complete_audit:
        raise ValueError("Publication requires a complete matching independent audit")
    observer = Path(manifest["observer"]).resolve(strict=True) if manifest.get("observer") else None
    evidence = {
        manifest_path,
        Path(__file__).resolve(),
        Path(shape_matrix_sources.__file__).resolve(),
        *audits,
        *source_inputs,
    }
    if observer:
        evidence.add(observer)
    evidence.update(path for shape in shapes for path in shape["evidence"])
    before = {str(path): digest(path) for path in sorted(evidence)}
    if any(before[str(path)] != expected for path, expected in source_inputs.items()):
        raise ValueError("Source changed after compatibility comparison")
    for filename, expected_hash in audit_inputs.items():
        actual = before.get(filename)
        if actual is None:
            actual = digest(filename)
        if actual != expected_hash:
            raise ValueError(f"Independent audit input changed before publication: {filename}")
        before[filename] = actual
    output = Path(output)
    if publish_dir is not None and Path(publish_dir).exists():
        raise FileExistsError(f"Publication destination already exists: {publish_dir}")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, output / "matrix_manifest.json")
    timings, samples, windows = report_rows(shapes)
    table(output / "timing.csv", timings)
    table(output / "timing_samples.csv", samples)
    table(output / "timeline_windows.csv", windows)
    table(output / "cache_metrics_samples.csv", cache_sample_rows(shapes))
    records = []
    for shape in shapes:
        entry = shape["entry"]
        target = output / f"h{entry['prefix_tokens']}_a{entry['extend_tokens']}"
        target.mkdir()
        for source in shape["selected"]:
            shutil.copy2(source, target / source.name)
        if shape["startup_selected"]:
            (target / "with_startup").mkdir()
            for source in shape["startup_selected"]:
                shutil.copy2(source, target / "with_startup" / source.name)
        records.append(
            {
                "prefix_tokens": entry["prefix_tokens"],
                "extend_tokens": entry["extend_tokens"],
                "pool_tokens": entry["prefix_tokens"] + entry["extend_tokens"],
                "run_ids": {name: run["run_id"] for name, run in shape["runs"].items()},
                "execution_identity_sha256": identity_digest(
                    shape["runs"]["bench"]["execution_identity"]
                ),
                "source_identity_sha256": identity_digest(shape["runs"]["bench"]["source_sha256"]),
                "source_directories": {
                    key: str(value) for key, value in shape["directories"].items()
                },
                "timeline_directory": str(Path(entry["timeline_run"]).resolve()),
                "startup_directory": str(Path(entry["startup_run"]).resolve())
                if entry.get("startup_run")
                else None,
                "warmups": shape["runs"]["bench"]["warmups"],
                "prefill_repeats": shape["runs"]["bench"]["prefill_repeats"],
                "extend_repeats": shape["runs"]["bench"]["repeats"],
            }
        )
    if audits or observer:
        (output / "audit").mkdir()
        for index, path in enumerate(audits):
            shutil.copy2(path, output / "audit" / f"{index}_{path.name}")
    if observer:
        shutil.copy2(observer, output / "audit" / "observer.json")
    write(
        output / "summary.json",
        {
            "schema": "deepseek-v32-shape-matrix-report-v1",
            "run_id": manifest["run_id"],
            "complete_requested_matrix": len(shapes) == len(SHAPES),
            "methods": METHODS,
            "configuration": manifest["configuration"],
            "execution_batches": manifest.get("execution_batches", []),
            "source_compatibility": source_compatibility,
            "shapes": records,
            "hardware": first["hardware"],
            "compute_precision": first["compute_precision"],
            "dependencies": first["dependencies"],
            "execution_environment": first["execution_environment"],
            "observer": read(observer) if observer else None,
            "wall_time_boundary": first["timing"],
            "profile_boundary": "Independent intrusive NSYS profile; final prefill chunk and complete extend L0-L2 windows; optional forward-entry startup view.",
        },
    )
    write(
        output / "provenance.json",
        {
            "inputs_sha256": {str(path): before[str(path)] for path in sorted(evidence)},
            "verified_audit_input_files": len(audit_inputs),
            "common_execution_metadata": common,
            "argv": sys.argv,
            "independent_audits": [str(path) for path in audits],
        },
    )
    (output / "results.md").write_text(
        markdown(manifest, shapes, timings, source_compatibility=source_compatibility)
    )
    if before != {filename: digest(filename) for filename in before}:
        raise RuntimeError("Report inputs changed during generation")
    write(
        output / "publication_manifest.json",
        {
            "schema": "selected-shape-matrix-artifacts-v1",
            "run_id": manifest["run_id"],
            "raw_report_directory": str(output.resolve()),
            "files_sha256": {
                str(path.relative_to(output)): digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    if publish_dir is not None:
        shutil.copytree(output, publish_dir)
    return output / "publication_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Runner manifest; paths are absolute or relative to the repository working directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New output/data/<report-run-id> directory"
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        help="New selected report directory, normally report/shape_matrix",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        action="append",
        default=[],
        help="Successful independent audit; a complete matching audit is required for publication",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow a configured subset for explicit partial reports; default requires all 12 shapes",
    )
    args = parser.parse_args()
    print(
        publish(
            args.manifest,
            args.output_dir,
            publish_dir=args.publish_dir,
            audit_paths=args.audit,
            allow_subset=args.allow_subset,
        )
    )


if __name__ == "__main__":
    main()
