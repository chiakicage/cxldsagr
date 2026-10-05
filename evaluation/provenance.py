"""Shared experiment source snapshots and numerical-evidence helpers.

This module contains no benchmark orchestration. Numerical helpers belong to
independent acceptance; clean benchmarks use their compatible saved receipts.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from evaluation import pool_scan_provenance as pool_scan

ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_snapshot(output, *, include_official=False):
    """Save actual source bytes, including uncommitted and untracked implementation."""
    extensions = {".py", ".sh", ".cu", ".cuh", ".c", ".cpp", ".h", ".hpp"}
    manifest = {}
    for directory in (
        "models",
        "operators",
        "cache",
        "executor",
        "serving",
        "GR",
        "evaluation",
    ):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or (
                path.suffix not in extensions and path != ROOT / pool_scan.POOL_SCAN_ABI
            ):
                continue
            if any(part in ("__pycache__", "output", "generated", "build") for part in path.parts):
                continue
            relative = path.relative_to(ROOT)
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for experiment in ("cache_management",):
        for path in sorted((ROOT / "experiments" / experiment).rglob("*")):
            if not path.is_file() or path.suffix not in (".py", ".sh") or "output" in path.parts:
                continue
            relative = path.relative_to(ROOT)
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if include_official:
        from experiments.deepseek_v32_echo_prefill.src.backend_provenance import source_files

        for path in sorted(source_files()):
            relative = path.relative_to(ROOT)
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(output / "source_manifest.json", manifest)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def snapshot_report_helpers(output) -> dict:
    """Save current source files for repository Python modules already loaded."""
    output = Path(output)
    manifest_path = output / "report_helper_sources.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    directories = {
        "GR",
        "cache",
        "evaluation",
        "executor",
        "experiments",
        "models",
        "operators",
        "scripts",
        "serving",
        "tests",
    }
    excluded = {
        ".cache",
        ".git",
        ".pytest_cache",
        ".venv",
        "3rdparty",
        "__pycache__",
        "build",
        "dist",
        "generated",
        "output",
        "report",
        "reports",
        "weights",
    }
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(ROOT):
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] not in directories or excluded.intersection(relative.parts):
            continue
        loaded.setdefault(relative, []).append(name)
    files = {relative: {"modules": sorted(names)} for relative, names in loaded.items()}
    for name in ("pyproject.toml", "uv.lock"):
        files[Path(name)] = {"kind": "environment specification"}
    snapshot = output / "source" / "report_helpers"
    snapshot.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "report-helper-sources-v1",
        "boundary": (
            "Current loaded repository Python modules only, plus pyproject.toml and uv.lock; "
            "not captured runtime/native identity, not every possible source. Saved bytes are "
            "the files on disk at report generation, not the loaded Python bytecode."
        ),
        "repository_root": str(ROOT),
        "included_directories": sorted(directories),
        "excluded_path_components": sorted(excluded),
        "invocation": {
            "orig_argv": list(sys.orig_argv),
            "cwd": str(Path.cwd()),
            "python": {"executable": sys.executable, "version": sys.version},
        },
        "files": {},
    }
    for relative, entry in sorted(files.items()):
        content = (ROOT / relative).read_bytes()
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)
        manifest["files"][str(relative)] = {
            **entry,
            "snapshot_path": str(destination.relative_to(output)),
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    with manifest_path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def backend_provenance():
    from experiments.deepseek_v32_echo_prefill.src.backend_provenance import (
        collect_backend_provenance,
        digest,
    )

    result = collect_backend_provenance()
    if "flashinfer" in result["installed"]:
        return result
    distribution = importlib.metadata.distribution("flashinfer-python")
    files = {}
    for relative in distribution.files or ():
        path = Path(distribution.locate_file(relative)).resolve()
        if path.is_file() and path.suffix in {".py", ".so", ".cuh", ".h", ".hpp", ".cu", ".cpp"}:
            files[str(path)] = digest(path)
    if not files:
        raise RuntimeError("FlashInfer installed source provenance is unavailable")
    result["flashinfer"] = {"version": distribution.version, "files_sha256": files}
    return result


def save_numerical_evidence(output, *, model, scheme, users, request_id, hidden, logits):
    relative = Path("numerical") / model / scheme / str(users) / f"{request_id:06d}.pt"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    torch.save({"hidden": hidden, "logits": logits}, destination)
    with destination.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(relative), "sha256": digest}


def numerical_comparison(actual, expected, *, atol, rtol):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("all candidate hidden states must match shape and dtype")
    a, b = actual.float(), expected.float()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise AssertionError("nonfinite output in serving correctness gate")
    difference = a - b
    metrics = {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.norm() / b.norm().clamp_min(1e-20)),
        "atol": atol,
        "rtol": rtol,
    }
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return metrics


def verify_source_snapshot(output):
    manifest = json.loads((output / "source_manifest.json").read_text())
    changed = [
        name
        for name, digest in manifest.items()
        if not (ROOT / name).is_file()
        or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise RuntimeError(f"implementation changed while measuring: {changed}")


def verify_access_trace(workload, path):
    """Bind generated model inputs to an already selected complete access trace."""
    with Path(path).open(newline="") as source:
        expected = list(csv.DictReader(source))
    if len(expected) != len(workload.requests):
        raise ValueError("saved access trace and generated workload have different lengths")
    for request, row in zip(workload.requests, expected, strict=True):
        identity = {name: int(row[name]) for name in ("request_id", "user_id", "visit_index")}
        identity.update(
            is_revisit=bool(int(row["is_revisit"])),
            previous_request_id=(
                int(row["previous_request_id"]) if row["previous_request_id"] else None
            ),
            timestamp=float(row["synthetic_timestamp"]),
        )
        if any(request[name] != value for name, value in identity.items()):
            raise ValueError(f"request {request['request_id']}: differs from saved access trace")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
