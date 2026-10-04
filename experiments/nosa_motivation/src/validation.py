"""Separate correctness receipts from uninstrumented motivation measurements."""

from __future__ import annotations

import json
import os
from pathlib import Path


def add_mode_arguments(command):
    command.add_argument(
        "--mode",
        choices=("check", "bench"),
        default="bench",
        help="check saves numerical/lifecycle evidence; bench requires an independent receipt",
    )
    command.add_argument(
        "--validation-receipt",
        type=Path,
        help="receipt.json from a compatible --mode check run (required for bench)",
    )


def validate_mode_arguments(args):
    if args.mode == "bench" and args.validation_receipt is None:
        raise ValueError("bench requires --validation-receipt from an independent check run")
    if args.mode == "check" and args.validation_receipt is not None:
        raise ValueError("check creates its own receipt; do not pass --validation-receipt")
    if args.validation_receipt is not None and not args.validation_receipt.is_file():
        raise FileNotFoundError(args.validation_receipt)


def without_performance(value):
    """The runner still executes its real guards, but checks publish no timing/MFU claims."""
    if isinstance(value, dict):
        return {
            key: item
            for key, item in value.items()
            if not key.endswith(("_ms", "_seconds", "_mfu_pct")) and key != "effective_work"
        }
    if isinstance(value, (tuple, list)):
        return [without_performance(item) for item in value]
    return value


def _stable_runtime(value):
    if isinstance(value, dict):
        return {
            key: _stable_runtime(item)
            for key, item in value.items()
            if not key.endswith(("_replays", "_seconds"))
            and key not in {"git_revision", "git_status", "eager_fallbacks", "memory_at_allocation"}
        }
    if isinstance(value, (tuple, list)):
        return [_stable_runtime(item) for item in value]
    return value


def base_identity(metadata, directory):
    manifest = json.loads((Path(directory) / "source_manifest.json").read_text())
    # Plotting and report edits do not invalidate numerical acceptance. Execution
    # orchestration, source dependencies and the receipt implementation do.
    execution_names = {"measure.py", "config.py", "validation.py", "workload.py"}
    sources = {
        name: sha
        for name, sha in manifest.items()
        if not name.startswith("experiments/") or Path(name).name in execution_names
    }
    return {
        "schema": "motivation-validation-identity-v1",
        "sources": sources,
        "config": {
            key: value for key, value in metadata["config"].items() if key != "peak_bf16_tflops"
        },
        "workload_sha256": metadata["workload_sha256"],
        "checkpoint": metadata["checkpoint"],
        "hardware": _stable_runtime(metadata["hardware"]),
        "precision_settings": metadata["precision_settings"],
        "native_build": metadata.get("native_provenance", {}).get("build_before"),
        "backend_provenance": metadata.get("backend_provenance"),
        "execution_environment": metadata["execution_environment"],
    }


def execution_environment():
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("CXLDSAGR_", "CUDA_", "TRITON_", "FLASHINFER_", "DEEPGEMM_"))
    }


def begin_validation(metadata, directory, receipt_path, kind):
    metadata["validation_identity"] = {"base": base_identity(metadata, directory), "methods": {}}
    if metadata["mode"] == "check":
        return
    from evaluation.validation import require_receipt

    path = Path(receipt_path).resolve()
    candidate = json.loads(path.read_text())
    _require_method_coverage(metadata, candidate["identity"]["methods"])
    # Verify the complete receipt/artifact manifest once. Every saved per-method
    # identity is replaced with freshly observed runtime evidence before use.
    expected = {
        "base": metadata["validation_identity"]["base"],
        "methods": candidate["identity"]["methods"],
    }
    receipt = require_receipt(path, kind=kind, identity=expected)
    from experiments.nosa_motivation.src.provenance import digest

    metadata["correctness_receipt"] = {
        "path": str(path),
        "sha256": digest(path),
        "kind": kind,
        "identity": receipt["identity"],
    }


def record_runtime(metadata, method, backend, native, token_validation, allocator=None):
    identity = _stable_runtime(
        {
            "backend_type": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "backend": backend.describe(),
            "native_artifacts": native,
            "token_validation": token_validation,
            "allocator_snapshot": allocator,
        }
    )
    if metadata.get("mode") == "bench":
        expected = metadata["correctness_receipt"]["identity"]["methods"].get(method)
        if identity != expected:
            raise ValueError(f"{method}: backend/cache/graph/native identity differs from check")
    metadata["validation_identity"]["methods"][method] = identity


def audit_receipt(metadata, directory, kind):
    from evaluation.validation import require_receipt
    from experiments.nosa_motivation.src.provenance import digest

    evidence = metadata["correctness_receipt"]
    path = Path(evidence["path"])
    if digest(path) != evidence["sha256"]:
        raise ValueError("external correctness receipt changed")
    actual = {
        "base": base_identity(metadata, directory),
        "methods": metadata["validation_identity"]["methods"],
    }
    _require_method_coverage(metadata, actual["methods"])
    if actual != metadata["validation_identity"] or actual != evidence["identity"]:
        raise ValueError("bench identity differs from independently checked execution")
    receipt = require_receipt(path, kind=kind, identity=actual)
    return {
        "kind": kind,
        "path": str(path),
        "sha256": evidence["sha256"],
        "checks": receipt["checks"],
    }


def publish_receipt(metadata, directory, kind, audit):
    from evaluation.validation import write_receipt

    directory = Path(directory)
    _require_method_coverage(metadata, metadata["validation_identity"]["methods"])
    artifacts = {
        "metadata": directory / "metadata.json",
        "requests": directory / "workload/requests.jsonl",
        "workload": directory / "workload/workload.json",
        "checks": directory / "measurements.jsonl",
    }
    artifacts.update(
        {
            str(path.relative_to(directory)): path
            for path in sorted((directory / "numerical").rglob("*.pt"))
        }
    )
    return write_receipt(
        directory / "receipt.json",
        kind=kind,
        identity=metadata["validation_identity"],
        checks={
            "passed": True,
            "coverage": "complete four-method multi-request trace",
            "numerical_and_lifecycle": audit,
            "does_not_establish": "failure injection, task quality, or performance",
        },
        artifacts=artifacts,
    )


def _require_method_coverage(metadata, methods):
    expected = metadata["config"].get("methods", metadata["config"].get("schemes"))
    if (
        not expected
        or set(methods) != set(expected)
        or any(not value for value in methods.values())
    ):
        raise ValueError("correctness receipt lacks complete backend/cache/graph method coverage")


def reference_directory(directory, *, bench_schema, kind):
    """Resolve a clean bench to its independent check data for diagnostic replay."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema") != bench_schema:
        return directory
    audit_receipt(metadata, directory, kind)
    from evaluation.validation import require_receipt

    receipt = require_receipt(
        metadata["correctness_receipt"]["path"], kind=kind, identity=metadata["validation_identity"]
    )
    return Path(receipt["artifact_paths"]["metadata"]).parent
