"""Reusable numerical acceptance, separate from clean performance measurements.

Callers own the numerical checks and the exact execution identity. This module
binds their successful result to immutable evidence; it does not cache runtime
allocator, input, transaction, or finite-value checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "cxldsagr-numerical-acceptance-v1"


def identity_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_record(path, base):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Acceptance evidence is not a file: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    try:
        location = str(path.relative_to(base))
    except ValueError:
        location = str(path)
    return {"path": location, "bytes": path.stat().st_size, "sha256": digest}


def write_receipt(path, *, kind, identity, checks, artifacts=None):
    """Publish acceptance only after the caller completes all required checks.

    A receipt is never overwritten. Store it with independent check evidence
    outside experiment performance results (a system temporary directory is a
    suitable default). ``identity`` must include the caller's actual supported
    path/configuration, input, runtime source, and native/backend identities.
    """
    path = Path(path).resolve()
    if not isinstance(kind, str) or not kind:
        raise ValueError("Acceptance kind must be a nonempty string")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("Acceptance requires a nonempty execution identity")
    if not isinstance(checks, dict) or checks.get("passed") is not True:
        raise ValueError("Only completed successful checks can issue acceptance")
    record = {
        "schema": SCHEMA,
        "kind": kind,
        "identity": identity,
        "checks": checks,
        "artifacts": {
            name: _file_record(artifact, path.parent)
            for name, artifact in (artifacts or {}).items()
        },
    }
    record["receipt_sha256"] = identity_digest(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(record, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
    return require_receipt(path, kind=kind, identity=identity)


def require_receipt(path, *, kind, identity):
    """Verify exact compatibility and evidence before any benchmark samples.

    Returns the stored record plus ``receipt_path`` and resolved
    ``artifact_paths``. These convenience fields are not part of the signature.
    """
    if path is None:
        raise ValueError("A compatible independent numerical acceptance receipt is required")
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError("Invalid numerical acceptance record")  # noqa: TRY004 - malformed file
    signature = record.get("receipt_sha256")
    payload = {key: value for key, value in record.items() if key != "receipt_sha256"}
    if signature != identity_digest(payload):
        raise ValueError("Numerical acceptance receipt integrity check failed")
    if record.get("schema") != SCHEMA or record.get("kind") != kind:
        raise ValueError("Numerical acceptance schema or experiment kind differs")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("Expected execution identity must be nonempty")
    if identity_digest(record.get("identity")) != identity_digest(identity):
        raise ValueError("Numerical acceptance does not cover this execution identity")
    if record.get("checks", {}).get("passed") is not True:
        raise ValueError("Numerical acceptance did not pass")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Invalid numerical evidence inventory")  # noqa: TRY004 - malformed file
    resolved = {}
    for name, expected in artifacts.items():
        if not isinstance(expected, dict) or set(expected) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid numerical evidence record")
        source = (path.parent / expected["path"]).resolve(strict=True)
        actual = _file_record(source, path.parent)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError(f"Numerical acceptance evidence changed: {name}")
        resolved[name] = str(source)
    return {**record, "receipt_path": str(path), "artifact_paths": resolved}
