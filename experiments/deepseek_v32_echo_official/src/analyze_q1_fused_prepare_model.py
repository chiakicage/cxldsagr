"""Audit all 500 private model pairs after timing, without importing model code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import statistics
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt
from experiments.deepseek_v32_echo_official.src import analyze_q1_hint_model as common

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
KIND = "deepseek-private-q1-fused-prepare-model-v1"
COMPONENT_KIND = "deepseek-private-q1-fused-prepare-v1"
FLASHINFER_SCHEMA = "private-fused-prepare-flashinfer-immutable-v1"
FLASHINFER_MODULES = {"rope", "silu_and_mul", "topk"}
_UNSET = object()
ARMS = ("baseline", "candidate")
TOKENS = (111090, 111091, 111092)
PAIR_COUNT, BLOCK_SIZE = 500, 100
SOURCE_CLOSURE = (*common.SOURCE_CLOSURE, str(Path(__file__).resolve().relative_to(ROOT)))
REQUIRED_SOURCES = {
    "experiments/deepseek_v32_echo_official/src/" + name
    for name in (
        "q1_fused_prepare_model.py",
        "q1_fused_prepare.py",
        "q1_fused_prepare.cu",
        "q1_fused_prepare_run.py",
        "q1_fused_prepare_native.py",
        "q1_fused_prepare_model_native.py",
        "q1_fused_prepare_model_pinned_run.py",
        "q1_fused_prepare_flashinfer.py",
        "q1_hint_model.py",
    )
}
COUNTERS = (
    "record_bytes",
    "selection_records",
    "written_records",
    "host_written_records",
    "evicted_records",
    "capacity_splits",
    "transient_written_records",
    "prefetched_records",
    "prefetch_capacity_failures",
    "recalled_records",
    "resident_selection_records",
    "host_to_device_bytes",
    "device_to_host_bytes",
)
require = common.require


class Evidence:
    def __init__(self):
        self.files = {}

    def file(self, path, expected=_UNSET, *, size=None):
        if expected is not _UNSET:
            require(
                isinstance(expected, str)
                and len(expected) == 64
                and all(character in "0123456789abcdef" for character in expected),
                "Invalid expected evidence digest",
            )
        path = Path(path).resolve(strict=True)
        require(path.is_file(), "Evidence is not a file: " + str(path))
        if str(path) not in self.files:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            self.files[str(path)] = {"sha256": digest, "bytes": path.stat().st_size}
        record = self.files[str(path)]
        require(
            expected is _UNSET or record["sha256"] == expected, "Evidence changed: " + str(path)
        )
        require(size is None or record["bytes"] == size, "Evidence size changed: " + str(path))
        return {"path": str(path), **record}

    def read(self, path):
        self.file(path)
        return json.loads(Path(path).read_text())

    def receipt(self, path, *, kind, identity):
        self.file(path)
        receipt = require_receipt(path, kind=kind, identity=identity)
        for name, record in receipt["artifacts"].items():
            resolved = receipt["artifact_paths"][name]
            self.files[resolved] = {key: record[key] for key in ("sha256", "bytes")}
        return receipt

    def verify(self):
        for name, expected in self.files.items():
            with Path(name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            require(
                digest == expected["sha256"] and Path(name).stat().st_size == expected["bytes"],
                "Evidence changed during analysis: " + name,
            )


def safe_relative(value):
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe archive path")
    return path


def verify_runtime_files(node, evidence):
    """Verify native libraries, JIT artifacts and recorded compiler/source files."""
    if isinstance(node, dict):
        if "matches_pinned_sources" in node:
            require(node["matches_pinned_sources"] is True, "Native source pinning differs")
        for path_key, hash_key in (
            ("path", "sha256"),
            ("artifact_path", "artifact_sha256"),
            ("executable", "sha256"),
        ):
            if path_key in node and hash_key in node:
                expected = node[hash_key]
                if path_key == "path" and isinstance(expected, dict):
                    root = Path(node[path_key]).resolve(strict=True)
                    require(root.is_dir() and expected, "Invalid runtime directory inventory")
                    for relative, digest in expected.items():
                        evidence.file(root / safe_relative(relative), digest)
                    actual = {
                        str(path.relative_to(root))
                        for path in root.rglob("*")
                        if path.is_file() and "__pycache__" not in path.parts
                    }
                    require(actual == set(expected), "Runtime directory inventory differs")
                else:
                    evidence.file(node[path_key], expected, size=node.get("bytes"))
        for value in node.values():
            verify_runtime_files(value, evidence)
    elif isinstance(node, list):
        for value in node:
            verify_runtime_files(value, evidence)


def audit_contract(identity):
    expected = {
        "layers": [0, 1, 2],
        "history": 65536,
        "append": 1,
        "tokens": list(TOKENS),
        "capacity": 65537,
        "slots": 65600,
        "host_arena_tokens": 65600,
        "chunk_size": 1024,
        "extend_chunk_size": 1,
        "workspace_query_tokens": 1024,
        "compute_graphs": [1024, 1],
        "warmups": 5,
        "candidate": "q1-fused-page-and-stage-prepare-v1",
        "timing": "Complete forward(return_hidden=True) plus synchronize; cold prefix/hint restore, binding, metrics and all diagnostics outside timing",
    }
    contract = identity["source"]["contract"]
    require(
        all(contract.get(key) == value for key, value in expected.items()), "Wrong model contract"
    )
    require(
        REQUIRED_SOURCES <= set(identity["source"]["sources"]), "Missing execution source coverage"
    )
    for name in identity["source"]["sources"]:
        safe_relative(name)
    source = identity["source"]
    require(
        source.get("private_flashinfer_policy")
        == "private-fused-prepare-model-immutable-flashinfer-v1",
        "Missing immutable FlashInfer execution policy",
    )
    require(
        source["gpu"]["capability"] == [9, 0]
        and source["affinity"] == list(range(8, 16))
        and source["environment"].get("CUDA_VISIBLE_DEVICES") == "1"
        and source["threads"] == 8
        and source["precision"] == "ieee",
        "Unexpected benchmark device, affinity or precision",
    )


def audit_flashinfer(directory, runtime, evidence):
    records = runtime.get("private_flashinfer_native", {})
    require(set(records) == FLASHINFER_MODULES, "Incomplete immutable FlashInfer inventory")
    observed = runtime["model"]["loaded_jit"]["native_jit"]
    rows = {row["name"]: row for row in observed if row["name"] in FLASHINFER_MODULES}
    require(len(rows) == len(observed) == 3, "Unexpected FlashInfer runtime inventory")
    archive_root = directory / "flashinfer_native"
    manifest = evidence.read(archive_root / "manifest.json")
    require(
        manifest.get("schema") == FLASHINFER_SCHEMA
        and set(manifest.get("modules", {})) == FLASHINFER_MODULES,
        "Incomplete FlashInfer archive manifest",
    )
    for name, record in records.items():
        request = record["build_identity"]["request"]
        row = rows[name]
        key = hashlib.sha256(
            json.dumps(record["build_identity"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        require(
            record.get("schema") == FLASHINFER_SCHEMA
            and record.get("name") == name
            and record.get("cache_key") == key
            and request["name"] == name
            and row.get("loaded_in_this_process") is True
            and row["library"] == record["library"]
            and row["build_metadata"] == record["build_metadata"]
            and row["sources"] == record["build_identity"]["sources"]
            and row["cuda_flags"] == request["extra_cuda_cflags"]
            and row["cxx_flags"] == request["extra_cflags"],
            "FlashInfer declared identity differs from the actual runtime: " + name,
        )
        copies = manifest["modules"][name]
        require(
            set(copies) == {"library", "build_metadata", "manifest", "dependency_metadata"},
            "Incomplete FlashInfer artifact archival",
        )

        def verify_copy(copy, original, *, name=name):
            require(copy["original"] == original, "FlashInfer archive source binding differs")
            target = archive_root / name / Path(original["path"]).name
            require(
                copy["archived"]["path"] == str(target.resolve())
                and copy["archived"]["sha256"] == original["sha256"]
                and copy["archived"]["bytes"] == original["bytes"],
                "FlashInfer archive location or identity differs",
            )
            evidence.file(target, original["sha256"], size=original["bytes"])

        for field in ("library", "build_metadata", "manifest"):
            verify_copy(copies[field], record[field])
        require(
            record["dependency_metadata"]
            and len(copies["dependency_metadata"]) == len(record["dependency_metadata"]),
            "Incomplete FlashInfer compiler dependency archival",
        )
        for copy, original in zip(
            copies["dependency_metadata"], record["dependency_metadata"], strict=True
        ):
            verify_copy(copy, original)
        saved = evidence.read(archive_root / name / Path(record["manifest"]["path"]).name)
        require(
            saved == {key: value for key, value in record.items() if key != "manifest"},
            "Archived FlashInfer build record differs from runtime",
        )


def audit_run(directory, mode, identity, receipt, evidence):
    result = evidence.read(directory / "result.json")
    require(
        result.get("completed") is True
        and result.get("mode") == mode
        and result.get("run_id") == directory.name
        and result.get("result", {}).get("passed") is True,
        "Incomplete or mislabeled " + mode + " execution",
    )
    require(result["identity"] == identity, "Check and benchmark identities differ")
    if mode == "bench":
        require(
            result["result"].get("receipt_sha256") == receipt["receipt_sha256"]
            and "capture_templates" not in result["result"],
            "Clean benchmark lacks its receipt or contains profile templates",
        )
    for relative, digest in identity["source"]["sources"].items():
        evidence.file(directory / "source" / safe_relative(relative), digest)
    native = identity["timing_runtime"]["mapped_native"]
    require(
        native and len({entry["name"] for entry in native}) == len(native),
        "Invalid native inventory",
    )
    for entry in native:
        evidence.file(
            directory / "native" / safe_relative(entry["name"]),
            entry["library"]["sha256"],
            size=entry["library"]["bytes"],
        )
    audit_flashinfer(directory, identity["timing_runtime"], evidence)
    require(set(result["memory"]) == set(ARMS), "Missing model memory record")
    for arm in ARMS:
        graph = result["memory"][arm]["graph"]
        require(
            graph["history_tokens"] == 65536
            and graph["query_tokens"] == graph["graph_count"] == 1
            and graph["cache_method"] == "echo"
            and graph["return_hidden"] is True
            and graph["enabled"] is True,
            "Unexpected complete model graph",
        )
    return result


def audit_component(component, runtime, evidence):
    identity = component["identity"]
    receipt_record, bench_record = component["receipt"], component["benchmark"]
    evidence.file(receipt_record["path"], receipt_record["sha256"])
    receipt = evidence.receipt(receipt_record["path"], kind=COMPONENT_KIND, identity=identity)
    require(
        receipt["checks"].get("complete_cases") == 210
        and receipt["checks"].get("byte_cases") == 100,
        "Incomplete component correctness coverage",
    )
    for key in ("private_generic_native", "candidate_native", "official_native"):
        require(runtime[key] == identity[key], "Model/component native differs: " + key)
    libraries = {row["name"]: row["library"] for row in runtime["mapped_native"]}
    for row in identity["mapped_local_native"]:
        require(libraries.get(row["name"]) == row["library"], "Accepted component native is absent")
    for field in ("sources", "runtime_files", "inputs"):
        for name, digest in identity[field].items():
            evidence.file(name, digest)
    evidence.file(bench_record["path"], bench_record["sha256"])
    bench = evidence.read(bench_record["path"])
    require(
        bench.get("mode") == "bench"
        and bench.get("identity") == identity
        and bench.get("receipt_sha256") == receipt["receipt_sha256"],
        "Component benchmark binding differs",
    )
    return {"receipt_signature": receipt["receipt_sha256"], "checks": receipt["checks"]}


def audit_bindings(receipt_path, bench_dir, evidence):
    raw = evidence.read(receipt_path)
    identity = raw["identity"]
    audit_contract(identity)
    receipt = evidence.receipt(receipt_path, kind=KIND, identity=identity)
    require(
        {"result.json", "outputs.pt"} <= set(receipt["artifacts"]), "Missing saved model evidence"
    )
    checks = receipt["checks"]
    require(
        checks.get("tokens") == list(TOKENS)
        and checks.get("prepared_cap_both_arms") == 64
        and all(
            checks.get(key) is True
            for key in (
                "passed",
                "eager_graph_both_arms",
                "all_offsets_bitwise",
                "actual_stage_proofs",
                "clean_graph_checked",
                "public_observer_preserved",
            )
        ),
        "Model receipt lacks numerical, state or dispatch coverage",
    )
    check = audit_run(receipt_path.parent, "check", identity, receipt, evidence)
    result = check["result"]
    require(
        result.get("tokens") == list(TOKENS)
        and result.get("cloned_outputs_all_offsets_and_actual_stage_proofs") is True
        and set(result["cross_arm"]) == set(map(str, TOKENS)),
        "Model check result lost saved output/state coverage",
    )
    for token in TOKENS:
        cross = result["cross_arm"][str(token)]
        require(set(cross) == {"eager", "graph"}, "Missing eager/graph comparison")
        for value in cross.values():
            require(
                value.get("layers") == 3
                and all(
                    value.get(key) is True
                    for key in (
                        "equal",
                        "exact_topk_equal",
                        "logical_slot_order",
                        "bounded_prefetch_transitions_validated",
                    )
                ),
                "Failed or incomplete per-execution state comparison",
            )
            require(
                set(value.get("schedule_dependent_metrics", ()))
                == {
                    "host_to_device_bytes",
                    "prefetch_capacity_failures",
                    "prefetched_records",
                    "recalled_records",
                    "resident_selection_records",
                }
                and set(value.get("schedule_dependent_state_fields", ()))
                == {
                    "free_count",
                    "logical_priority",
                    "resident",
                },
                "Unexpected relaxation of exact state comparisons",
            )
    bench = audit_run(bench_dir, "bench", identity, receipt, evidence)
    envelope = evidence.read(bench_dir / "pre_gate_identity.json")
    require(
        envelope.get("schema") == "private-fused-prepare-model-immutable-flashinfer-v1"
        and envelope.get("mode") == "bench"
        and type(envelope.get("pairs")) is int
        and envelope["pairs"] == PAIR_COUNT
        and envelope.get("identity") == identity
        and envelope.get("receipt_path") == str(receipt_path.resolve()),
        "Benchmark identity differs from its saved receipt-gate envelope",
    )
    source, runtime = identity["source"], identity["timing_runtime"]
    component = audit_component(source["component"], runtime, evidence)
    request = source["request"]
    evidence.file(request["path"], request["sha256"])
    ids = evidence.read(request["path"])["input_ids"]
    require(len(ids) == 65537 and ids[-1] == TOKENS[0], "Wrong fixed H64K/A1 request")
    for relative, digest in source["sources"].items():
        evidence.file(ROOT / safe_relative(relative), digest)
    verify_runtime_files(runtime, evidence)
    return bench, {
        "receipt": evidence.file(receipt_path),
        "receipt_signature": receipt["receipt_sha256"],
        "identity_sha256": identity_digest(identity),
        "check": evidence.file(receipt_path.parent / "result.json"),
        "bench": evidence.file(bench_dir / "result.json"),
        "checks": checks,
        "component": component,
        "state_validation_boundary": "Signed numerical/state evidence and all artifact hashes are verified. Cross-arm eager/graph proof flags are checked for all three tokens. This analysis does not rerun GPU transitions or deserialize tensor payloads; timing counters are independently checked below.",
    }


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (position - left) * (ordered[right] - ordered[left])


def describe(pairs):
    deltas = [row["delta_ms"] for row in pairs]
    medians = {arm: statistics.median(row[arm + "_ms"] for row in pairs) for arm in ARMS}
    return {
        "pairs": len(pairs),
        "median_wall_ms": medians,
        "median_paired_delta_ms": statistics.median(deltas),
        "mean_paired_delta_ms": statistics.mean(deltas),
        "paired_delta_sample_stdev_ms": statistics.stdev(deltas) if len(deltas) > 1 else None,
        "paired_delta_quantiles_ms": {
            str(p): quantile(deltas, p) for p in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)
        },
        "median_paired_relative_delta_percent": statistics.median(
            row["relative_delta_percent"] for row in pairs
        ),
        "ratio_of_arm_medians": medians["candidate"] / medians["baseline"],
        "candidate_faster_pairs": sum(value < 0 for value in deltas),
        "candidate_slower_pairs": sum(value > 0 for value in deltas),
        "equal_pairs": sum(value == 0 for value in deltas),
    }


def block_sensitivity(pairs):
    """Enumerate the empirical five-block resamples; this is not a promotion gate."""
    blocks = [[row["delta_ms"] for row in pairs if row["block"] == block] for block in range(5)]
    require(all(blocks), "Missing prespecified block")
    medians = [
        statistics.median(value for block in choice for value in blocks[block])
        for choice in itertools.product(range(5), repeat=5)
    ]
    return {
        "blocks": 5,
        "resamples": len(medians),
        "median_paired_delta_percentile_band_ms": [
            quantile(medians, 0.025),
            quantile(medians, 0.975),
        ],
        "method": "Exact enumeration of 5^5 ordered resamples of the five prespecified consecutive blocks; 2.5/97.5 percentiles use linear interpolation.",
        "interpretation": "Descriptive sensitivity to the five observed time blocks, not an independent-run confidence interval. One process with two fixed model instances is measured; cross-block dependence, drift and shared machine effects are not identified by this resampling.",
    }


def analyze_samples(bench):
    require(
        type(bench.get("pairs")) is int and bench["pairs"] == PAIR_COUNT,
        "Expected exactly 500 predeclared pairs",
    )
    rows = bench["result"]["samples"]
    for row in rows:
        require(type(row.get("pair")) is int, "Noninteger pair ID")
        require(
            type(row.get("wall_ms")) in (int, float)
            and math.isfinite(row["wall_ms"])
            and row["wall_ms"] > 0,
            "Invalid model latency",
        )
        require(len(row["layer_metrics"]) == 3, "Missing per-layer state counters")
        for metrics in row["layer_metrics"]:
            require(
                all(type(metrics.get(key)) is int and metrics[key] >= 0 for key in COUNTERS),
                "Invalid state counter",
            )
            require(
                metrics["prefetched_records"] <= 64
                and metrics["resident_selection_records"]
                <= metrics["prefetched_records"] + metrics["written_records"],
                "Cold candidate state differs",
            )
    # This reader independently derives the unchanged q1_hint_model.summarize schema.
    clean = common.clean_timing(bench)
    by_pair = {(row["pair"], row["arm"]): row for row in rows}
    pairs = []
    for index in range(PAIR_COUNT):
        baseline, candidate = (by_pair[index, arm]["wall_ms"] for arm in ARMS)
        pairs.append(
            {
                "pair": index,
                "block": index // BLOCK_SIZE,
                "order": "AB" if index % 2 == 0 else "BA",
                "baseline_ms": baseline,
                "candidate_ms": candidate,
                "delta_ms": candidate - baseline,
                "relative_delta_percent": 100 * (candidate - baseline) / baseline,
            }
        )
    strata = {order: [row for row in pairs if row["order"] == order] for order in ("AB", "BA")}
    return {
        "predeclared_analysis": {
            "pairs": PAIR_COUNT,
            "warmups_per_arm": 5,
            "orders": {"AB": 250, "BA": 250},
            "consecutive_blocks": 5,
            "block_size_pairs": BLOCK_SIZE,
            "sample_selection": "All 1000 arm measurements retained in original execution order; no outlier removal, early/late trimming, favorable block selection or replacement samples.",
            "delta_convention": "candidate minus baseline; negative paired deltas mean lower observed candidate latency",
        },
        "summary": clean["summary"],
        "overall": describe(pairs),
        "order_strata": {order: describe(selected) for order, selected in strata.items()},
        "sequential_blocks": [
            {
                "block": block,
                "first_pair": block * BLOCK_SIZE,
                "last_pair": (block + 1) * BLOCK_SIZE - 1,
                **describe([row for row in pairs if row["block"] == block]),
                "order_strata": {
                    order: describe([row for row in selected if row["block"] == block])
                    for order, selected in strata.items()
                },
            }
            for block in range(5)
        ],
        "descriptive_block_sensitivity": {
            "overall": block_sensitivity(pairs),
            **{order: block_sensitivity(selected) for order, selected in strata.items()},
        },
        "paired_samples": pairs,
        "traffic": clean["traffic"],
        "boundary": clean["boundary"],
        "promotion": "No statistical significance threshold or automatic promotion rule is introduced. Profile/work equivalence and parent review remain separate requirements. This timing audit alone does not establish deployment or serving gains.",
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt, bench_dir, output = (
        path.resolve() for path in (args.receipt, args.bench_dir, args.output_dir)
    )
    require(output.is_relative_to(EXPERIMENT / "output"), "Output must remain in the experiment")
    require(not output.exists(), "Use a new analysis run ID")
    evidence = Evidence()
    analysis_sources = {name: evidence.file(ROOT / name)["sha256"] for name in SOURCE_CLOSURE}
    bench, bindings = audit_bindings(receipt, bench_dir, evidence)
    timing = analyze_samples(bench)
    result = {
        "schema": "deepseek-private-q1-fused-prepare-model-analysis-v1",
        "run_id": output.name,
        "passed": True,
        "passed_meaning": "Input identity, acceptance evidence and complete-sample accounting passed; this flag does not accept a performance claim.",
        "bindings": bindings,
        "clean_timing": timing,
        "memory": bench["memory"],
        "memory_boundary": "Graph private storage is per arm. Allocated/reserved/device-used snapshots include the process state at sequential preparation points, not isolated per-model footprints or continuous peaks.",
        "analysis_sources": analysis_sources,
    }
    evidence.verify()
    output.mkdir(parents=True)
    write_json(output / "result.json", result)
    with (output / "paired_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing["paired_samples"][0]))
        writer.writeheader()
        writer.writerows(timing["paired_samples"])
    for relative, expected in analysis_sources.items():
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        evidence.file(target, expected)
    write_json(output / "input_hashes.json", evidence.files)
    write_json(
        output / "publication_manifest.json",
        {
            "schema": "private-q1-model-analysis-artifacts-v1",
            "files_sha256": {
                str(path.relative_to(output)): common.digest(path)
                for path in output.rglob("*")
                if path.is_file()
            },
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "integrity_passed": True,
                "overall": timing["overall"],
                "order_strata": timing["order_strata"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
