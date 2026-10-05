"""Bind independent resident-model checks to benchmark execution identities."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt, write_receipt

SOURCE_ROOT = Path(__file__).resolve().parents[3]
KIND = "nosa-resident-baseline"
IDENTITY_FIELDS = (
    "model_config",
    "checkpoint_path",
    "checkpoint_files",
    "checkpoint_config_sha256",
    "request_sha256",
    "source_sha256",
    "torch",
    "cuda",
    "flashinfer",
    "gpu",
    "runtime_settings",
    "native_artifacts",
)


def execution_identity(metadata, attention_mode):
    """Repetition counts and profiling do not change the accepted model path."""
    args = metadata["args"]
    return {
        "attention_mode": attention_mode,
        "dtype": "bfloat16",
        "prefix_tokens": args["prefix_tokens"],
        "new_tokens": args["new_tokens"],
        "chunk_size": args["chunk_size"],
        "kernel_backend": args.get("kernel_backend"),
        **{name: metadata[name] for name in IDENTITY_FIELDS},
        **{
            name: metadata[name]
            for name in ("triton", "tvm_ffi", "native_build")
            if name in metadata
        },
    }


def check_directory(path):
    path = Path(path).resolve()
    experiments = Path(__file__).resolve().parents[2]
    if path.is_relative_to(experiments):
        raise ValueError("Numerical check evidence belongs outside experiments (use /tmp)")
    return path


def publish_check(data_dir, metadata, attention_mode, checks, audit_name):
    data_dir = check_directory(data_dir)
    completed_metadata = data_dir / "metadata.json"
    recorded = json.loads(completed_metadata.read_text())
    if identity_digest(recorded) != identity_digest(metadata):
        raise ValueError("Final check metadata was not written successfully")
    if metadata.get("validation") != checks:
        raise ValueError("Final metadata does not record these completed numerical checks")
    for name, expected in metadata["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Check source must be relative to the repository")
        live = SOURCE_ROOT / relative
        snapshot = (
            data_dir / ("source_" + "_".join(relative.parts))
            if attention_mode == "dense"
            else data_dir / "sources" / relative
        )
        for source in (live, snapshot):
            with source.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Final check source or snapshot changed: {source}")
    (data_dir / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    return write_receipt(
        data_dir / "receipt.json",
        kind=KIND,
        identity=execution_identity(metadata, attention_mode),
        checks={"passed": True, **checks},
        artifacts={
            "validation": data_dir / "validation.json",
            "metadata": completed_metadata,
            "attention_audit": data_dir / audit_name,
        },
    )


def attach_acceptance(receipt, data_dir, metadata, attention_mode):
    record = require_receipt(
        receipt, kind=KIND, identity=execution_identity(metadata, attention_mode)
    )
    destination = Path(data_dir) / "numerical_acceptance"
    destination.mkdir(exist_ok=True)
    # Check writers use flat, relative artifact paths so the receipt remains verifiable after a move.
    for name, source in record["artifact_paths"].items():
        relative = Path(record["artifacts"][name]["path"])
        if relative.is_absolute() or relative.parent != Path("."):
            raise ValueError("Baseline check artifacts must be next to their receipt")
        shutil.copy2(source, destination / relative)
    shutil.copy2(record["receipt_path"], destination / "receipt.json")
    verified = require_receipt(
        destination / "receipt.json",
        kind=KIND,
        identity=execution_identity(metadata, attention_mode),
    )
    metadata["numerical_acceptance"] = {
        "receipt": "numerical_acceptance/receipt.json",
        "receipt_sha256": verified["receipt_sha256"],
        "execution": "independent_check_reused",
    }
    return verified


def verify_acceptance(data_dir, metadata, attention_mode):
    reference = metadata.get("numerical_acceptance")
    if (
        not isinstance(reference, dict)
        or reference.get("receipt") != "numerical_acceptance/receipt.json"
    ):
        raise ValueError("Missing independent numerical acceptance")
    record = require_receipt(
        Path(data_dir) / reference["receipt"],
        kind=KIND,
        identity=execution_identity(metadata, attention_mode),
    )
    if record["receipt_sha256"] != reference.get("receipt_sha256"):
        raise ValueError("Numerical acceptance reference differs from verified evidence")
    return record
