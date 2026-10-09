"""CPU reread of shape-matrix tensors, timing, source identities and native trace intervals.

The numerical receipt is checked with the experiment contract. Saved tensors,
SQLite GPU inventories and timeline interval arithmetic are read independently.
Stage/lane labels retain the source-bound renderer's classification. Cold-ECHO
compact stage arrays are reread; omitted score eligibility and KV content remain
runtime acceptance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt
from experiments.deepseek_v32_mfu.src import shape_matrix_sources
from experiments.deepseek_v32_mfu.src.run_contract import (
    CHECK_FIELDS,
    METHODS,
    RECEIPT_KIND,
    benchmark_view,
    execution_identity,
)

ROOT = Path(__file__).resolve().parents[3]
PROCESS_MASK = 0xFFFFFFFFFF000000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(
        type(actual) in (int, float)
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
        f"{label}: {actual} differs from {expected}",
    )


class Evidence:
    """Hash each input once and reject any mutation through the end of the audit."""

    def __init__(self):
        self.hashes = {}
        self.stats = {}

    @staticmethod
    def stat(path):
        value = path.stat()
        return tuple(
            getattr(value, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        )

    def digest(self, path):
        path = Path(path).resolve(strict=True)
        key, before = str(path), self.stat(path)
        if key in self.hashes:
            require(before == self.stats[key], f"Audit input changed: {path}")
            return self.hashes[key]
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        require(before == self.stat(path), f"File changed while hashing: {path}")
        self.hashes[key], self.stats[key] = value, before
        return value

    def read(self, path):
        self.digest(path)
        return json.loads(Path(path).read_text())

    def verify(self, *, rehash=True):
        for name, expected in self.stats.items():
            require(self.stat(Path(name)) == expected, f"Audit input changed: {name}")
            if rehash:
                with Path(name).open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                require(actual == self.hashes[name], f"Audit input changed: {name}")


def source_audit(directory, result, evidence):
    manifest = evidence.read(directory / "sources.json")
    require(manifest == result["source_sha256"], "Execution source manifest differs")
    snapshot = shape_matrix_sources.source_snapshot(
        manifest,
        directory / "source",
        current_hardware=ROOT / shape_matrix_sources.HARDWARE_SOURCE,
        digest_file=evidence.digest,
    )
    if shape_matrix_sources.HARDWARE_SOURCE in manifest:
        require(
            result["hardware"]["hardware_source_sha256"]
            == manifest[shape_matrix_sources.HARDWARE_SOURCE],
            "Hardware evidence source identity differs",
        )
    for name, expected in manifest.items():
        if name == shape_matrix_sources.HARDWARE_SOURCE:
            continue
        require(
            evidence.digest(ROOT / name) == expected, f"Current execution source changed: {name}"
        )
    measurement = result.get("measurement_identity", {}).get("source_sha256", {})
    if measurement:
        require(
            evidence.read(directory / "measurement_sources.json") == measurement,
            "Measurement source manifest differs",
        )
        for name, expected in measurement.items():
            require(
                evidence.digest(directory / "measurement_source" / name)
                == evidence.digest(ROOT / name)
                == expected,
                f"Measurement source changed: {name}",
            )
    return {
        "execution_files": len(manifest),
        "measurement_files": len(measurement),
        "snapshot": snapshot,
    }


def runtime_records(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for child in value.values():
            yield from runtime_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from runtime_records(child)


def q1_hint_audit(result, evidence):
    """Reread the bounded exact-mean build without importing a CUDA provider."""
    name = "cxldsagr_q1_hint_exact_mean_sm90"
    runtime = result["execution_runtime_artifacts"]
    observed = runtime.get("q1_hint_native")
    mapped = [item for item in runtime["local_native_jit"] if item["name"] == f"{name}.so"]
    backend = result["backend_provenance"]
    if "q1_exact_prefetch_hint" not in backend:
        require(observed is None and not mapped, "Q1 hint runtime has no declared build identity")
        return None
    declared = backend["q1_exact_prefetch_hint"]
    require(
        isinstance(declared, dict)
        and declared.get("policy") == "q1-exact-torch-tree-65537-512x4-v1",
        "Unknown Q1 hint source policy",
    )
    files = {}

    def verify_files(records):
        require(isinstance(records, dict) and bool(records), "Empty Q1 hint file closure")
        for path, expected in records.items():
            require(
                isinstance(path, str)
                and Path(path).is_absolute()
                and isinstance(expected, str)
                and re.fullmatch("[0-9a-f]{64}", expected) is not None,
                "Invalid Q1 hint file identity",
            )
            require(evidence.digest(path) == expected, f"Q1 hint dependency changed: {path}")
            require(files.get(path, expected) == expected, "Conflicting Q1 hint file identities")
            files[path] = expected

    sources = declared.get("source_sha256")
    verify_files(sources)
    project = {
        str((ROOT / relative).resolve())
        for relative in (
            "operators/deepseek_v32/indexer/q1_hint_exact.py",
            "operators/deepseek_v32/indexer/csrc/q1_hint_exact.cu",
            "operators/deepseek_v32/indexer/_native_cache.py",
        )
    }
    require(project <= sources.keys(), "Q1 hint project source closure is incomplete")
    required = (
        result.get("prefix_tokens") == 65536
        and result.get("extend_tokens") == 1
        and "echo" in result.get("methods", ())
    )
    if observed is None:
        require(not required and not mapped, "Expected an observed Q1 hint production artifact")
        return {"loaded": False, "source_files": len(sources)}
    require(isinstance(observed, dict), "Invalid Q1 hint native record")
    build = observed["build_identity"]
    require(build["source_identity"] == declared, "Q1 hint native/source identity differs")
    require(
        build["cuda_flags"] == declared["flags"]
        and build["include_paths"] == []
        and build["link_flags"] == [],
        "Q1 hint native build flags differ",
    )
    translation = str((ROOT / "operators/deepseek_v32/indexer/csrc/q1_hint_exact.cu").resolve())
    loader = str((ROOT / "operators/deepseek_v32/indexer/_native_cache.py").resolve())
    require(
        build["sources_sha256"] == {translation: sources[translation]}
        and build["loader_sha256"] == sources[loader],
        "Q1 hint translation unit or loader differs",
    )
    toolchain = build["toolchain"]
    for compiler in ("cc", "cxx", "nvcc"):
        item = toolchain[compiler]
        verify_files({item["executable"]: item["sha256"]})
    verify_files(toolchain["cuda_compiler_sha256"])
    headers = Path(toolchain["nvcc"]["executable"]).parent.parent / "include"
    require(headers.is_dir(), "Q1 hint CUDA include directory missing")
    require(
        set(sources) == project | {str(p.resolve()) for p in headers.rglob("*") if p.is_file()},
        "Q1 hint CUDA source closure inventory differs",
    )
    ffi_root = Path(toolchain["tvm_ffi_root"])
    require(ffi_root.is_absolute() and ffi_root.is_dir(), "Invalid Q1 hint TVM-FFI root")
    ffi_files = toolchain["tvm_ffi_sha256"]
    require(
        set(ffi_files)
        == {
            str(p.relative_to(ffi_root))
            for p in ffi_root.rglob("*")
            if p.is_file() and p.suffix in {".py", ".so", ".h", ".hpp"}
        },
        "Q1 hint TVM-FFI closure inventory differs",
    )
    verify_files({str(ffi_root / path): expected for path, expected in ffi_files.items()})
    fingerprint = hashlib.sha256(json.dumps(build, sort_keys=True).encode()).hexdigest()
    key = f"{name}_{fingerprint}"
    artifact = Path(observed["artifact_path"])
    require(
        observed.get("schema") == 1
        and observed.get("cache_key") == key
        and observed.get("artifact_name") == f"{name}.so"
        and artifact.is_absolute()
        and artifact.name == f"{name}.so"
        and artifact.parent.name == key,
        "Q1 hint immutable artifact identity differs",
    )
    require(
        evidence.read(artifact.parent / "record.json")
        == {key: value for key, value in observed.items() if key != "artifact_path"},
        "Q1 hint immutable cache record differs",
    )
    verify_files({str(artifact): observed["artifact_sha256"]})
    require(
        len(mapped) == 1
        and mapped[0]["library"]
        == {
            "path": str(artifact),
            "sha256": observed["artifact_sha256"],
            "bytes": artifact.stat().st_size,
        },
        "Q1 hint artifact does not match the mapped production DSO",
    )
    return {
        "loaded": True,
        "source_files": len(sources),
        "source_and_toolchain_files": len(files),
        "tvm_ffi_files": len(ffi_files),
        "cache_key": key,
        "artifact_sha256": observed["artifact_sha256"],
        "mapped_artifact_matches": True,
        "boundary": "Declared project/CUDA, compiler and TVM-FFI closure plus the exact mapped "
        "DSO. System C/C++ headers outside that declaration are not build-time evidence.",
    }


def runtime_audit(result, evidence, *, require_offload=True):
    files = {}
    for group in ("backend_provenance", "execution_runtime_artifacts"):
        for item in runtime_records(result[group]):
            path = Path(item["path"])
            if path.is_absolute():
                require(
                    evidence.digest(path) == item["sha256"], f"Runtime artifact changed: {path}"
                )
                if "bytes" in item:
                    require(path.stat().st_size == item["bytes"], "Runtime artifact size changed")
                files[str(path)] = item["sha256"]
    if not require_offload:
        require(
            result.get("schema_version") == 4 and result.get("methods") == ["hbm"],
            "Only isolated HBM can omit the loaded recall bridge",
        )
        return {"runtime_and_dependency_files": len(files), "bridge_fingerprint": None}
    bridge = result["backend_provenance"]["recall_dispatch"]
    fingerprint = hashlib.sha256(
        json.dumps(bridge["identity"], sort_keys=True).encode()
    ).hexdigest()
    require(fingerprint == bridge["fingerprint"], "Native bridge fingerprint differs")
    for name, expected in bridge["identity"]["source_and_dependency_sha256"].items():
        require(evidence.digest(name) == expected, f"Native bridge dependency changed: {name}")
        files[name] = expected
    native = result["execution_runtime_artifacts"]["local_native_jit"]
    require(
        any(fingerprint[:16] in item["name"] for item in native), "Loaded native bridge differs"
    )
    hint = q1_hint_audit(result, evidence)
    return {
        "runtime_and_dependency_files": len(files),
        "bridge_fingerprint": fingerprint,
        **({"q1_exact_prefetch_hint": hint} if hint is not None else {}),
    }


def tensor_record(tensor, name, extend):
    import torch

    require(
        isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu", f"Non-CPU tensor: {name}"
    )
    expected = [extend, 7168] if name.endswith("/hidden") else [1, 129280]
    require(list(tensor.shape) == expected, f"Saved tensor shape differs: {name}")
    require(bool(torch.isfinite(tensor).all()), f"Nonfinite saved output: {name}")
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "raw_bytes_sha256": hashlib.sha256(
            tensor.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def compare(actual, expected, name):
    import torch

    require(
        actual.shape == expected.shape and actual.dtype == expected.dtype,
        f"Output ABI differs: {name}",
    )
    actual32, expected32 = actual.float(), expected.float()
    difference = (actual32 - expected32).abs()
    require(
        bool((difference <= 0.02 + 0.01 * expected32.abs()).all()), f"Numerical mismatch: {name}"
    )
    return {
        "name": name,
        "bitwise_equal": bool(
            torch.equal(
                actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
            )
        ),
        "max_abs": float(difference.max()),
        "rtol": 0.01,
        "atol": 0.02,
        "passed": True,
    }


def saved_tensors(check, profile, extend, evidence):
    import torch

    def load(path):
        evidence.digest(path)
        return torch.load(path, map_location="cpu", weights_only=True)

    loaded, inventory, comparisons = {}, [], []
    for method in METHODS:
        check_directory = check[method] if isinstance(check, dict) else check
        control = load(check_directory / f"{method}_control.pt")
        prefix = load(check_directory / f"{method}_prefix_logits.pt")
        graph = load(check_directory / f"{method}_extend_graph_check.pt")
        require(set(control) == {"hidden", "logits"}, "Saved control schema differs")
        require(
            set(graph) == {"baseline", "changed_baseline", "changed_graph"},
            "Saved graph schema differs",
        )
        bundles = {"control": control, "prefix": {"logits": prefix}, **graph}
        for group, bundle in bundles.items():
            expected = {"logits"} if group == "prefix" else {"hidden", "logits"}
            require(set(bundle) == expected, "Saved output schema differs")
            for key, value in bundle.items():
                inventory.append(tensor_record(value, f"{method}/{group}/{key}", extend))
        for key in ("hidden", "logits"):
            comparisons.append(
                compare(
                    control[key],
                    graph["baseline"][key],
                    f"{method}/standard_graph_vs_baseline/{key}",
                )
            )
            comparisons.append(
                compare(
                    graph["changed_graph"][key],
                    graph["changed_baseline"][key],
                    f"{method}/changed_graph_vs_baseline/{key}",
                )
            )
            require(
                not torch.equal(control[key], graph["changed_graph"][key]),
                "Changed input did not affect saved outputs",
            )
        loaded[method] = bundles
    for method in METHODS[1:]:
        for group, keys in (
            ("control", ("hidden", "logits")),
            ("prefix", ("logits",)),
            ("changed_graph", ("hidden", "logits")),
        ):
            for key in keys:
                comparisons.append(
                    compare(
                        loaded[method][group][key],
                        loaded["hbm"][group][key],
                        f"hbm_vs_{method}/{group}/{key}",
                    )
                )
    profile_comparisons = []
    for method in METHODS:
        profile_directory = profile[method] if isinstance(profile, dict) else profile
        actual = load(profile_directory / f"{method}_profile_output.pt")
        require(set(actual) == {"hidden", "logits"}, "Saved profile output schema differs")
        for key in ("hidden", "logits"):
            inventory.append(tensor_record(actual[key], f"{method}/profile/{key}", extend))
            profile_comparisons.append(
                compare(
                    actual[key], loaded[method]["control"][key], f"{method}/profile_vs_check/{key}"
                )
            )
    return {
        "check_tensor_count": 36,
        "profile_tensor_count": 8,
        "check_comparisons": comparisons,
        "profile_comparisons": profile_comparisons,
        "all_bitwise_equal": all(row["bitwise_equal"] for row in comparisons + profile_comparisons),
        "inventory": inventory,
        "boundary": "Saved outputs reread on CPU. Cache states and unsaved earlier default/repeat calls retain runtime-only evidence.",
    }


PREFETCH_LABELS = (
    "baseline",
    "default_graph",
    "replay_0",
    "replay_1",
    "changed_baseline",
    "changed_graph",
)


def prefetch_audit(directory, result, receipt, evidence):
    """Replay the compact state proof, without claiming to reread omitted logits/KV."""
    import torch

    from experiments.deepseek_v32_mfu.src import prefetch_transition_audit as transitions

    evidence.digest(transitions.__file__)
    checks = result["extend_graph_checks"]["echo"]
    proofs = checks.get("bounded_prefetch", {})
    require(set(proofs) == set(PREFETCH_LABELS), "Incomplete bounded-prefetch execution coverage")
    history, append, slots = result["prefix_tokens"], result["extend_tokens"], result["slots"]
    baseline_scope = proofs["baseline"].get("scope", {})
    prepared_cap = transitions.preparation_capacity(baseline_scope)
    expected_scope = {
        "method": "echo",
        "residency": "cold",
        "single_session": True,
        "num_layers": 3,
        "H": history,
        "A": append,
        "slots": slots,
        "max_prefetch": prepared_cap,
    }
    policy = baseline_scope.get("prefetch_policy", transitions.COARSE_POLICY)
    require(
        policy in (transitions.COARSE_POLICY, transitions.OFFICIAL_POLICY),
        "Unknown prefetch policy in matrix acceptance",
    )
    if policy == transitions.OFFICIAL_POLICY:
        require(append == 1 and slots - append >= 64, "Unsupported official Q1 headroom")
        expected_scope.update(
            prefetch_policy=transitions.OFFICIAL_POLICY,
            max_prefetch=64,
            prepared_max_prefetch=prepared_cap,
            hint_index=1,
        )
    if "preparation" in baseline_scope:
        expected_scope.update(
            preparation=dict(baseline_scope["preparation"]), record_bytes=1152, topk=2048
        )
    require(
        result["extend_residency"] == "cold" and history + append <= slots,
        "Unsupported bounded-prefetch audit scope",
    )
    expected_files = {
        f"echo_{label}_prefetch_{kind}.{suffix}"
        for label in PREFETCH_LABELS
        for kind, suffix in (("evidence", "pt"), ("receipt", "json"))
    }
    found = {path.name for path in directory.glob("*_prefetch_evidence.pt")}
    found.update(path.name for path in directory.glob("*_prefetch_receipt.json"))
    require(found == expected_files, "Prefetch evidence file coverage differs")
    artifacts = receipt["artifacts"]
    require(expected_files <= artifacts.keys(), "Numerical receipt does not bind prefetch evidence")
    rows, states = [], {}
    for label in PREFETCH_LABELS:
        accepted = proofs[label]
        receipt_name = f"echo_{label}_prefetch_receipt.json"
        evidence_name = f"echo_{label}_prefetch_evidence.pt"
        require(
            evidence.read(directory / receipt_name) == accepted,
            "Saved prefetch receipt differs from runtime check",
        )
        for name in (receipt_name, evidence_name):
            record = artifacts[name]
            path = directory / name
            require(
                (directory / record["path"]).resolve() == path.resolve()
                and record["sha256"] == evidence.digest(path)
                and record["bytes"] == path.stat().st_size,
                "Prefetch artifact differs from numerical receipt",
            )
        require(
            accepted.get("schema") == "cold-echo-stage-acceptance-v1"
            and accepted.get("passed") is True
            and accepted.get("scope") == expected_scope
            and accepted.get("evidence_file") == evidence_name
            and accepted.get("evidence_sha256") == evidence.digest(directory / evidence_name),
            "Prefetch acceptance binding or scope differs",
        )
        compact = torch.load(directory / evidence_name, map_location="cpu", weights_only=True)
        require(
            all("scores" not in layer for layer in compact["layers"]),
            "Expected explicitly compact prefetch evidence",
        )
        reconstructed = transitions.validate_execution(compact)
        require(
            reconstructed == accepted["proof"] and reconstructed["passed"] is True,
            "Compact prefetch proof differs from runtime acceptance",
        )
        require(
            all(reconstructed["scope"].get(key) == value for key, value in expected_scope.items()),
            "Reconstructed prefetch proof covers a different execution scope",
        )
        indices = [transitions.tensor_identity(layer["indices"]) for layer in compact["layers"]]
        hints = [transitions.tensor_identity(layer["initial_hint"]) for layer in compact["layers"]]
        scores = [layer["eligibility"]["score_identity"] for layer in compact["layers"]]
        require(
            indices == accepted["indices"]
            and hints == accepted["initial_hints"]
            and scores == accepted["scores"],
            "Prefetch index/score/hint identity differs",
        )
        final = {
            "length": history + append,
            "layers": [layer["final_cache_state"] for layer in compact["layers"]],
        }
        require(
            identity_digest(final) == accepted["final_cache_state_sha256"],
            "Prefetch proof does not bind final cache state",
        )
        states[label] = final
        rows.append(
            {
                "execution": label,
                "passed": True,
                "evidence_file": evidence_name,
                "evidence_sha256": accepted["evidence_sha256"],
                "proof": reconstructed,
                "final_cache_state_sha256": identity_digest(final),
            }
        )
    require(
        states["baseline"] == checks["baseline_cache"]
        and states["changed_baseline"] == checks["changed_baseline_cache"],
        "Saved prefetch baseline differs from the original cache check",
    )
    schedule_fields = {
        "prefetched_records",
        "prefetch_capacity_failures",
        "recalled_records",
        "resident_selection_records",
        "host_to_device_bytes",
    }
    schedule_state_fields = {"logical_priority", "metrics"}
    if policy == transitions.OFFICIAL_POLICY:
        schedule_state_fields.update(("resident", "free_count"))
    for baseline, other in (
        ("baseline", "default_graph"),
        ("baseline", "replay_0"),
        ("baseline", "replay_1"),
        ("changed_baseline", "changed_graph"),
    ):
        for field in ("indices", "scores", "initial_hints"):
            require(
                proofs[baseline][field] == proofs[other][field],
                f"Graph changed prefetch {field}: {other}",
            )
        for left, right in zip(states[baseline]["layers"], states[other]["layers"], strict=True):
            left_fixed = {
                key: value for key, value in left.items() if key not in schedule_state_fields
            }
            right_fixed = {
                key: value for key, value in right.items() if key not in schedule_state_fields
            }
            require(left_fixed == right_fixed, f"Graph changed deterministic cache state: {other}")
            left_metrics = {
                key: value for key, value in left["metrics"].items() if key not in schedule_fields
            }
            right_metrics = {
                key: value for key, value in right["metrics"].items() if key not in schedule_fields
            }
            require(
                left_metrics == right_metrics, f"Graph changed deterministic cache metrics: {other}"
            )
    return {
        "passed": True,
        "executions": rows,
        "compact_evidence_files": len(PREFETCH_LABELS),
        "score_index_hint_identities_equal_across_matching_executions": True,
        "boundary": "Saved compact maps, priorities, free bitmaps, clocks, selection indices and counters are reread and stage transitions recomputed. Eligibility derivation from raw scores, exact top-k score validity and final KV byte equality were checked at runtime; raw scores and KV payloads are not retained in this compact evidence. The saved eligibility bitmap and score/hint identities bind that runtime check, not a new score-to-eligibility recomputation.",
    }


def cache_sample_audit(result, method, phase):
    row = result["measurements"][method]
    samples = row.get(phase + "_cache_samples")
    count = result["prefill_repeats" if phase == "prefill" else "repeats"]
    require(
        isinstance(samples, list) and len(samples) == count, "Cache metric sample coverage differs"
    )
    required = {
        "prefetched_records",
        "recalled_records",
        "evicted_records",
        "prefetch_capacity_failures",
        "host_to_device_bytes",
        "device_to_host_bytes",
        "host_written_records",
        "record_bytes",
    }
    for sample in samples:
        require(
            isinstance(sample, list) and len(sample) == result["num_layers"],
            "Cache metric layer coverage differs",
        )
        for layer in sample:
            require(
                isinstance(layer, dict) and required <= layer.keys(),
                "Cache metric fields are incomplete",
            )
            require(
                all(type(layer[key]) is int and layer[key] >= 0 for key in required),
                "Invalid cache metric counter",
            )
            require(
                layer["record_bytes"] > 0
                and layer["host_to_device_bytes"]
                == (layer["prefetched_records"] + layer["recalled_records"]) * layer["record_bytes"]
                and layer["device_to_host_bytes"]
                == layer["host_written_records"] * layer["record_bytes"],
                "Sample cache traffic differs from actual record counters",
            )
    final_key = "prefix_cache_per_layer" if phase == "prefill" else "extend_cache_per_layer"
    require(
        samples[-1] == row[final_key], "Last sample cache metrics differ from compatibility field"
    )
    return samples


def timing_audit(result):
    require(not result["correctness"], "Clean timing contains inline numerical comparisons")
    rows = []
    for method in METHODS:
        for phase in ("prefill", "extend"):
            samples = result["measurements"][method][f"{phase}_samples_ms"]
            count = result["prefill_repeats" if phase == "prefill" else "repeats"]
            require(len(samples) == count and count > 0, "Timing sample count differs")
            require(
                all(
                    type(value) in (int, float) and math.isfinite(value) and value > 0
                    for value in samples
                ),
                "Invalid timing sample",
            )
            median = statistics.median(samples)
            require(
                median == result["measurements"][method][f"{phase}_median_ms"],
                "Timing median differs",
            )
            rows.append(
                {
                    "method": method,
                    "phase": phase,
                    "samples_ms": samples,
                    "median_ms": median,
                    "cache_samples": cache_sample_audit(result, method, phase),
                }
            )
    return rows


def request_prefix(directory, result, evidence):
    request = evidence.read(directory / "request.json")
    prefix, extend = result["prefix_tokens"], result["extend_tokens"]
    require(
        request["stable_prefix_tokens"] == prefix and request["candidate_suffix_tokens"] == extend,
        "Saved request semantic lengths differ",
    )
    ids = request["input_ids"]
    require(
        len(ids) == prefix + extend and all(type(value) is int for value in ids),
        "Saved token ID coverage differs",
    )
    return {
        "prefix_token_count": prefix,
        "prefix_token_ids_sha256": identity_digest(ids[:prefix]),
        "declared_history_text_sha256": request["history_sha256"],
        "history_token_span": request["history_token_span"],
        "candidate_token_span": request["candidate_token_span"],
        "boundary": "Hash of all fixed prefix token IDs, including the instruction; history text hash is the request's separate declaration.",
    }


def manifest_receipt_binding(shape, receipt_path, evidence):
    """The sweep manifest hashes the receipt file, not its signed JSON payload."""
    if "validation_receipt_sha256" in shape:
        require(
            shape["validation_receipt_sha256"] == evidence.digest(receipt_path),
            "Manifest check receipt SHA differs",
        )


def read_native(path, evidence):
    """Read native NSYS tables directly, independently of the rendering parser."""
    evidence.digest(path)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        copies = (
            dict(connection.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER"))
            if "ENUM_CUDA_MEMCPY_OPER" in tables
            else {}
        )
        gpu = []
        for kind, alternatives in (
            ("kernel", ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")),
            ("memcpy", ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMCPY2")),
            ("memset", ("CUPTI_ACTIVITY_KIND_MEMSET",)),
        ):
            table = next((name for name in alternatives if name in tables), None)
            if table is None:
                continue
            for item in connection.execute("SELECT * FROM " + table):
                row = dict(item)
                require(row["end"] > row["start"], "Nonpositive native GPU interval")
                name = (
                    strings[row["demangledName"]]
                    if kind == "kernel"
                    else copies[row["copyKind"]]
                    if kind == "memcpy"
                    else "memset"
                )
                gpu.append(
                    {
                        "kind": kind,
                        "name": name,
                        "start": row["start"],
                        "end": row["end"],
                        "process": row["globalPid"],
                        "device": row["deviceId"],
                        "stream": row["streamId"],
                        "correlation": row["correlationId"],
                        "bytes": row.get("bytes"),
                        "graph_id": row.get("graphId", 0),
                        "graph_node_id": row.get("graphNodeId", 0),
                    }
                )
        scopes = []
        for item in connection.execute("SELECT * FROM NVTX_EVENTS"):
            row = dict(item)
            label = row.get("text") or strings.get(row.get("textId"), "")
            if label.startswith("echo/") and row.get("end") is not None:
                scopes.append(
                    {
                        "label": label,
                        "start": row["start"],
                        "end": row["end"],
                        "thread": row["globalTid"],
                    }
                )
        launches = []
        for family in ("RUNTIME", "DRIVER"):
            table = "CUPTI_ACTIVITY_KIND_" + family
            if table in tables:
                for item in connection.execute("SELECT * FROM " + table):
                    row = dict(item)
                    if "GraphLaunch" in strings[row["nameId"]]:
                        launches.append(
                            {
                                "start": row["start"],
                                "end": row["end"],
                                "thread": row["globalTid"],
                                "correlation": row["correlationId"],
                            }
                        )
        lineage = {}
        if "CUDA_GRAPH_NODE_EVENTS" in tables:
            for node, parent in connection.execute(
                "SELECT graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS"
            ):
                if parent is not None:
                    require(
                        node not in lineage or lineage[node] == parent, "Ambiguous graph lineage"
                    )
                    lineage[node] = parent
    return {"gpu": gpu, "scopes": scopes, "launches": launches, "lineage": lineage}


def original_node(node, lineage):
    visited = set()
    while node in lineage:
        require(node not in visited, "Cyclic native graph lineage")
        visited.add(node)
        node = lineage[node]
    return node


def native_key(row):
    return tuple(
        row[key]
        for key in (
            "kind",
            "name",
            "start",
            "end",
            "process",
            "device",
            "stream",
            "correlation",
            "bytes",
            "graph_id",
            "graph_node_id",
        )
    )


def plotted_key(row):
    return native_key(
        {**row, "start": row["raw_start_ns"], "end": row["raw_end_ns"], "device": row["device_id"]}
    )


def union_ns(intervals):
    total, end = 0, None
    for left, right in sorted((left, right) for left, right in intervals if right > left):
        total += max(0, right - max(left, end if end is not None else left))
        end = max(right, end if end is not None else right)
    return total


def window_arithmetic(panel):
    window = panel["window"]
    start, end = window["start_ns"], window["end_ns"]
    require(end > start, "Empty timeline window")
    lanes = {}
    for row in panel["rows"]:
        if row["kind"] == "api":
            continue
        left, right = max(start, row["raw_start_ns"]), min(end, row["raw_end_ns"])
        require(
            row["start_ns"] == left and row["end_ns"] == right and right > left,
            "Timeline clipping differs",
        )
        lanes.setdefault(row["lane"], []).append((left, right))
    require(set(lanes) <= {"Compute", "Compute + IO", "IO", "GPU control"}, "Unknown GPU lane")
    compute, fused, io = (lanes.get(name, []) for name in ("Compute", "Compute + IO", "IO"))
    productive = union_ns(compute + fused + io)
    compute_with_fused = union_ns(compute + fused)
    busy = union_ns([span for spans in lanes.values() for span in spans])
    duration, gap, idle = end - start, end - start - productive, end - start - busy
    pure_io = productive - compute_with_fused
    expected = {
        "window_ms": duration / 1e6,
        "compute_union_ms": union_ns(compute) / 1e6,
        "fused_union_ms": union_ns(fused) / 1e6,
        "io_union_ms": union_ns(io) / 1e6,
        "compute_io_union_ms": productive / 1e6,
        "gpu_idle_ms": idle / 1e6,
        "control_only_ms": (gap - idle) / 1e6,
        "gap_ms": gap / 1e6,
        "gap_percent": 100 * gap / duration,
        "pure_io_only_ms": pure_io / 1e6,
        "non_io_window_ms": (duration - pure_io) / 1e6,
        "gap_no_io_percent": 100 * gap / (duration - pure_io),
    }
    for name, value in expected.items():
        close(window[name], value, f"Timeline {name}")
    require(
        window["unresolved_gather_count"] == 0 and window["gate_certifiable"],
        "Timeline has unresolved transport classification",
    )
    return {"start_ns": start, "end_ns": end, **expected}


def read_setup_lineage(path, evidence):
    """Read capture/clone lineage; setup need not contain measured NVTX scopes."""
    evidence.digest(path)
    lineage = {}
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as connection:
        for node, parent in connection.execute(
            "SELECT graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS"
        ):
            if parent is not None:
                require(node not in lineage or lineage[node] == parent, "Ambiguous graph lineage")
                lineage[node] = parent
    return lineage


def graph_inventory(profile, result, method, native, evidence):
    setup_index = result["nsys_capture_order"].index(f"{method}/extend_graph_setup") + 1
    lineage = read_setup_lineage(profile / f"capture_{setup_index}.sqlite", evidence)
    templates = evidence.read(profile / "full_graph_templates.json")
    require(
        templates == result["full_extend_graph_templates"], "Full graph template identity differs"
    )
    matches = [row for row in templates if row["method"] == method]
    require(len(matches) == 1, "Missing or duplicate graph template")
    template = matches[0]
    forwards = [
        scope
        for scope in native["scopes"]
        if scope["label"].startswith(f"echo/{method}/extend_annotated/shared/forward_misc/")
    ]
    require(len(forwards) == 1, "Expected exactly one measured forward scope")
    forward = forwards[0]
    selected = [
        row
        for row in native["gpu"]
        if forward["start"] <= row["start"] < row["end"] <= forward["end"]
    ]
    require(selected, "No measured forward GPU activities")
    launches = [
        row
        for row in native["launches"]
        if row["thread"] == forward["thread"] and forward["start"] <= row["start"] < forward["end"]
    ]
    require(len(launches) == 1, "Extend does not execute exactly one native graph launch")
    nodes, rows = [], []
    for row in selected:
        if not row["graph_node_id"] and not row["graph_id"]:
            continue
        require(
            not row["graph_id"] or row["graph_id"] == template["executable_graph_id"],
            "Unexpected executable graph",
        )
        node = original_node(row["graph_node_id"], lineage)
        require(str(node) in template["node_owners"], "Native graph contains an unknown node")
        require(
            template["node_types"][str(node)]
            == {"kernel": 0, "memcpy": 1, "memset": 2}[row["kind"]],
            "Native node type differs",
        )
        owner = template["node_owners"][str(node)]
        layer = owner["layer"]
        match = re.fullmatch(r"dense_history_prefetch_layer_(\d+)", owner["stage"])
        if match:
            layer = "layer_" + match[1]
        rows.append({**row, "layer": layer, "stage": owner["stage"]})
        nodes.append(node)
    require(
        len(nodes) == len(set(nodes)) and set(nodes) == set(template["gpu_node_ids"]),
        "Missing or duplicate native graph nodes",
    )
    io, dense_dependencies = [], []
    for layer, counts in enumerate(result["measurements"][method]["extend_cache_per_layer"]):
        current = [row for row in rows if row["layer"] == f"layer_{layer}"]
        h2d = [
            row for row in current if row["kind"] == "memcpy" and row["name"] == "Host-to-Device"
        ]
        d2h = [
            row for row in current if row["kind"] == "memcpy" and row["name"] == "Device-to-Host"
        ]
        h2d_bytes, d2h_bytes = sum(row["bytes"] for row in h2d), sum(row["bytes"] for row in d2h)
        require(
            d2h_bytes
            == counts["device_to_host_bytes"]
            == (0 if method == "hbm" else result["extend_tokens"] * counts["record_bytes"]),
            "Native append D2H bytes differ",
        )
        require(
            counts["host_to_device_bytes"]
            == (counts["prefetched_records"] + counts["recalled_records"]) * counts["record_bytes"],
            "H2D record/byte accounting differs",
        )
        if method == "dense_prefetch":
            require(
                h2d_bytes
                == counts["host_to_device_bytes"]
                == result["prefix_tokens"] * counts["record_bytes"],
                "Native dense H2D bytes differ",
            )
            publish = [row for row in current if "dense_history_publish_kernel" in row["name"]]
            consumers = [
                row
                for row in current
                if row["stage"]
                in {"cache_write", "offload_exact_recall", "sparse_mla", "mla_qk_pv"}
            ]
            clear = [row for row in current if "dense_history_clear_kernel" in row["name"]]
            require(
                len(h2d) == len(publish) == len(clear) == 1
                and {"cache_write", "offload_exact_recall", "sparse_mla"}
                <= {row["stage"] for row in consumers},
                "Dense dependency nodes missing",
            )
            require(
                clear[0]["end"] <= h2d[0]["start"]
                and h2d[0]["end"] <= publish[0]["start"]
                and publish[0]["end"] <= min(row["start"] for row in consumers),
                "Dense consumers precede H2D/publication completion",
            )
            compute = [
                row
                for row in current
                if row["stage"] in {"attention_projection", "indexer_qk", "exact_topk"}
                and row["kind"] == "kernel"
            ]
            overlap = union_ns(
                [
                    (max(row["start"], h2d[0]["start"]), min(row["end"], h2d[0]["end"]))
                    for row in compute
                ]
            )
            dense_dependencies.append(
                {
                    "layer": layer,
                    "h2d_start_ns": h2d[0]["start"],
                    "h2d_end_ns": h2d[0]["end"],
                    "publish_end_ns": publish[0]["end"],
                    "first_consumer_ns": min(row["start"] for row in consumers),
                    "projection_indexer_topk_h2d_overlap_ns": overlap,
                }
            )
        else:
            require(h2d_bytes == 0, "Unexpected explicit layer H2D")
        io.append(
            {
                "layer": layer,
                "native_h2d_bytes": h2d_bytes,
                "native_d2h_bytes": d2h_bytes,
                "runtime_h2d_bytes": counts["host_to_device_bytes"],
            }
        )
    return {
        "method": method,
        "native_graph_launch_count": 1,
        "graph_gpu_node_count": len(nodes),
        "exact_gpu_node_membership": True,
        "io": io,
        "dense_dependencies": dense_dependencies,
    }


def window_inventory(native, panel):
    window = window_arithmetic(panel)
    selected = [
        row
        for row in native["gpu"]
        if row["start"] < window["end_ns"] and row["end"] > window["start_ns"]
    ]
    plotted = [row for row in panel["rows"] if row["kind"] != "api"]
    require(
        Counter(native_key(row) for row in selected)
        == Counter(plotted_key(row) for row in plotted),
        "Timeline omits, duplicates or alters a native GPU interval",
    )
    return {"native_gpu_intervals": len(selected), **window}


def timeline_audit(directory, profile, result, evidence, *, startup=False, native_cache=None):
    receipt = evidence.read(directory / "compact_receipt.json")
    require(receipt["profile_run_id"] == result["run_id"], "Timeline profile binding differs")
    require(
        receipt["window_kind"] == ("extend-startup" if startup else "three-layers"),
        "Timeline boundary differs",
    )
    for name, expected in receipt["inputs_and_sources_sha256"].items():
        require(evidence.digest(name) == expected, f"Timeline input/source changed: {name}")
    for name, expected in receipt["artifacts_sha256"].items():
        require(evidence.digest(directory / name) == expected, f"Timeline artifact changed: {name}")
    data = evidence.read(directory / "window_rows.json")
    require(
        set(data) == ({"extend"} if startup else {"prefill", "extend"}),
        "Timeline phase coverage differs",
    )
    native_cache = {} if native_cache is None else native_cache
    rows, graphs = [], []
    for phase, panels in data.items():
        require(
            len(panels) == len(METHODS) and {panel["method"] for panel in panels} == set(METHODS),
            "Timeline method coverage differs",
        )
        for panel in panels:
            method = panel["method"]
            index = result["nsys_capture_order"].index(f"{method}/{phase}_annotated") + 1
            path = profile / f"capture_{index}.sqlite"
            if index not in native_cache:
                native_cache[index] = read_native(path, evidence)
            native = native_cache[index]
            window = window_inventory(native, panel)
            if phase == "prefill":
                require(
                    panel["chunk"] == math.ceil(result["prefix_tokens"] / result["chunk_size"]) - 1,
                    "Prefill does not select the final history chunk",
                )
            elif not startup:
                graphs.append(graph_inventory(profile, result, method, native, evidence))
            rows.append({"phase": phase, "method": method, **window})
    return {"windows": rows, "extend_graphs": graphs, "native_gpu_coverage_exact": True}


def audit_shape(row, configuration, evidence):
    prefix, extend = row["prefix_tokens"], row["extend_tokens"]
    directories = {
        name: Path(row[name + "_run"]).resolve(strict=True)
        for name in ("check", "bench", "profile", "timeline")
    }
    results = {
        name: evidence.read(directories[name] / "result.json")
        for name in ("check", "bench", "profile")
    }
    check, bench, profile = (results[name] for name in ("check", "bench", "profile"))
    for mode, result in results.items():
        require(result["accepted"] is True and result["mode"] == mode, f"Unaccepted {mode} run")
        require(result["run_id"] == row[mode + "_run_id"], "Manifest run ID differs")
        require(
            result["methods"] == list(METHODS) and result["num_layers"] == 3,
            "Execution method/layer coverage differs",
        )
        require(
            (
                result["prefix_tokens"],
                result["extend_tokens"],
                result["chunk_size"],
                result["extend_chunk_size"],
            )
            == (prefix, extend, configuration["history_chunk_tokens"], extend),
            "Execution shape differs",
        )
        require(
            result["extend_graph"] is True
            and result["extend_residency"] == configuration["extend_residency"],
            "Execution graph/residency differs",
        )
        require(
            result["slots"] == result["sparse_pool_tokens"] == row["pool_tokens"]
            and result["host_arena_tokens"] == row["host_tokens"],
            "Execution pool configuration differs",
        )
        require(
            result["execution_identity"]
            == execution_identity(result)
            == check["execution_identity"],
            "Check/bench/profile identity differs",
        )
        require(
            evidence.digest(directories[mode] / "request.json") == result["request_sha256"],
            "Request SHA differs",
        )
    receipt_path = directories["check"] / "receipt.json"
    receipt = require_receipt(receipt_path, kind=RECEIPT_KIND, identity=check["execution_identity"])
    manifest_receipt_binding(row, receipt_path, evidence)
    evidence.digest(receipt_path)
    require(
        receipt["checks"]["comparisons"] == check["correctness"]
        and set(check["correctness"]) == CHECK_FIELDS,
        "Runtime comparison coverage differs",
    )
    require(
        receipt["checks"]["extend_graph"] == check["extend_graph_checks"],
        "Runtime graph checks differ",
    )
    for item in receipt["artifacts"].values():
        require(
            evidence.digest(directories["check"] / item["path"]) == item["sha256"],
            "Numerical receipt artifact changed",
        )
    benchmark_view(directories["bench"], bench)
    benchmark_view(directories["profile"], profile)
    require(
        Path(profile["benchmark"]["directory"]).resolve() == directories["bench"],
        "Profile binds another benchmark",
    )
    for mode in ("bench", "profile"):
        require(
            Path(results[mode]["validation_receipt"]["receipt_path"]).resolve() == receipt_path,
            "Run binds another check",
        )
    for bundle in check["extend_graph_checks"].values():
        runtime = bundle["runtime"]
        require(
            runtime["allocated"]
            and runtime["graph_count"] == 1
            and (runtime["history_tokens"], runtime["query_tokens"]) == (prefix, extend),
            "Runtime graph shape/count differs",
        )
        require(
            runtime["policy_revision"] == check["extend_graph_policy_revision"],
            "Runtime graph policy differs",
        )
        require(
            0 < runtime["private_reserved_bytes"] <= runtime["chosen_private_limit_bytes"]
            and runtime["static_allocated_bytes"] + runtime["private_reserved_bytes"]
            <= runtime["reservation_bytes"],
            "Graph reservation exceeded",
        )
        for name, item in bundle["checks"].items():
            if name.endswith("_cache"):
                require(item["equal"] and item["layers"] == 3, "Runtime cache comparison failed")
            else:
                require(
                    item["rtol"] == 0.01 and item["atol"] == 0.02,
                    "Runtime numerical tolerance differs",
                )
    gap = evidence.read(directories["profile"] / "gap_audit.json")
    require(
        gap["run_id"] == profile["run_id"]
        and gap["input_result_sha256"] == evidence.digest(directories["profile"] / "result.json"),
        "Gap analysis binding differs",
    )
    for phase in ("methods", "prefill_methods"):
        require(
            len(gap[phase]) == len(METHODS)
            and {item["method"] for item in gap[phase]} == set(METHODS),
            "Gap analysis coverage differs",
        )
        for item in gap[phase]:
            require(
                evidence.digest(directories["profile"] / item["sqlite"]) == item["sqlite_sha256"],
                "Gap analysis SQLite changed",
            )
    sources = {
        name: source_audit(directories[name], result, evidence) for name, result in results.items()
    }
    runtimes = {name: runtime_audit(result, evidence) for name, result in results.items()}
    native_cache = {}
    main_timeline = timeline_audit(
        directories["timeline"],
        directories["profile"],
        profile,
        evidence,
        native_cache=native_cache,
    )
    startup = None
    if row.get("startup_run"):
        startup = timeline_audit(
            Path(row["startup_run"]).resolve(strict=True),
            directories["profile"],
            profile,
            evidence,
            startup=True,
            native_cache=native_cache,
        )
    require(
        bool(startup) == bool(configuration.get("startup_timeline")),
        "Startup timeline coverage differs",
    )
    result = {
        "prefix_tokens": prefix,
        "extend_tokens": extend,
        "passed": True,
        "execution_identity_sha256": identity_digest(check["execution_identity"]),
        "source_identity_sha256": identity_digest(check["source_sha256"]),
        "sources": sources,
        "runtime": runtimes,
        "request_prefix": request_prefix(directories["check"], check, evidence),
        "timing": timing_audit(bench),
        "tensors": saved_tensors(directories["check"], directories["profile"], extend, evidence),
        "bounded_prefetch": prefetch_audit(directories["check"], check, receipt, evidence),
        "timeline": main_timeline,
        "startup_timeline": startup,
    }
    evidence.verify(rehash=False)
    return result


def history_identity_audit(rows):
    identities = []
    for prefix in sorted({row["prefix_tokens"] for row in rows}):
        by_extend = {
            str(row["extend_tokens"]): row["request_prefix"]["prefix_token_ids_sha256"]
            for row in rows
            if row["prefix_tokens"] == prefix
        }
        require(
            len(set(by_extend.values())) == 1,
            f"History prefix token IDs differ across extend sizes for H={prefix}",
        )
        identities.append(
            {
                "prefix_tokens": prefix,
                "extend_tokens": [int(extend) for extend in by_extend],
                "same_prefix_token_ids_sha256": True,
                "prefix_token_ids_sha256_by_extend_tokens": by_extend,
            }
        )
    return identities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", action="append", help="Audit only H:A; repeat for a subset")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Audit output already exists")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    require(not torch.cuda.is_initialized(), "Audit must start without CUDA")
    torch.set_num_threads(1)
    evidence = Evidence()
    evidence.digest(__file__)
    evidence.digest(shape_matrix_sources.__file__)
    manifest = evidence.read(args.manifest)
    require(manifest["schema_version"] == 1, "Unsupported matrix manifest")
    runner = manifest.get("runner")
    if runner:
        require(
            evidence.digest(ROOT / runner["source"]) == runner["sha256"],
            "Matrix runner source changed",
        )
    configuration = manifest["configuration"]
    expected = {
        (h, a) for h in configuration["prefix_tokens"] for a in configuration["extend_tokens"]
    }
    available = [(row["prefix_tokens"], row["extend_tokens"]) for row in manifest["shapes"]]
    require(
        len(available) == len(set(available)) and set(available) <= expected,
        "Matrix has unexpected or duplicate shapes",
    )
    selected = (
        {tuple(map(int, value.split(":"))) for value in args.shape} if args.shape else expected
    )
    require(selected and selected <= set(available), "Selected shapes are missing from manifest")
    rows = []
    for row in manifest["shapes"]:
        if (row["prefix_tokens"], row["extend_tokens"]) in selected:
            rows.append(audit_shape(row, configuration, evidence))
            print(
                json.dumps(
                    {
                        "prefix_tokens": row["prefix_tokens"],
                        "extend_tokens": row["extend_tokens"],
                        "passed": True,
                    }
                ),
                flush=True,
            )
    source_compatibility = shape_matrix_sources.compare_source_snapshots(
        [row["sources"]["check"]["snapshot"] for row in rows]
    )
    history_identities = history_identity_audit(rows)
    evidence.verify()
    require(not torch.cuda.is_initialized(), "CPU audit initialized CUDA")
    output = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "passed": True,
        "complete_matrix": selected == expected,
        "manifest_sha256": evidence.digest(args.manifest),
        "shapes": rows,
        "source_compatibility": source_compatibility,
        "history_identity_across_extend_sizes": history_identities,
        "input_sha256": evidence.hashes,
        "stable_before_after": True,
        "cuda_initialized": False,
        "boundary": "CPU saved-tensor, compact cold-ECHO state-transition and native SQLite reread. Stage/lane labels inherit the source-bound renderer; raw interval identities, completeness, graph nodes, dense IO bytes/dependencies and arithmetic are checked independently. Eligibility derivation and top-k score validity were checked against raw scores at runtime; compact reread verifies stored eligibility identities and transitions, not omitted raw scores. Final KV byte comparison and unsaved outputs retain runtime-only evidence. Actual clean per-sample cache counters are preserved separately from check/profile executions. No new GPU timing or continuous isolation claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
