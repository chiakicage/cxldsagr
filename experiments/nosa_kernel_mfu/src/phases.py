"""Independent correctness receipts for resident and offload microbenchmarks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.validation import require_receipt, write_receipt


def offload_config(args):
    return {
        "layers": args.layers,
        "prefix": args.prefix if args.synthetic else None,
        "queries": args.queries,
        "seed": args.seed if args.synthetic else None,
        "synthetic": args.synthetic,
        "tile_size": args.tile_size,
        "fetch_ctas": args.fetch_ctas,
        "backend": "native",
        "q_heads": 32,
        "kv_heads": 2,
        "head_dim": 128,
        "block_budget": 64,
    }


def add_phase_arguments(parser):
    parser.allow_abbrev = False
    parser.add_argument(
        "--mode",
        choices=("check", "bench", "profile"),
        default="bench",
        help="Independent correctness, clean performance (default), or diagnostic profiling",
    )
    parser.add_argument(
        "--validation-receipt",
        type=Path,
        help="Matching check receipt required by bench/profile; optional destination for check",
    )


def validate_phase_arguments(parser, args):
    if args.mode != "check" and args.validation_receipt is None:
        parser.error("bench/profile requires --validation-receipt from an independent --mode check")
    if getattr(args, "reference_all", False) and args.mode != "check":
        parser.error("--reference-all belongs to --mode check; bench/profile reuse its receipt")
    if args.mode == "check" and hasattr(args, "reference_all"):
        args.reference_all = True
    if (
        args.mode == "check"
        and args.validation_receipt is not None
        and args.validation_receipt.exists()
    ):
        parser.error("validation receipt already exists; choose a new check destination")


def validation_identity(metadata, *, config, inputs, cache):
    """Bind acceptance to actual code, binary recipe, inputs and execution contract."""
    return {
        "source_sha256": metadata["source_sha256"],
        "native_build": metadata["native_build"],
        "gpu": metadata["gpu"],
        "torch": metadata["torch"],
        "cuda": metadata["cuda"],
        "dependencies": metadata["dependencies"],
        "environment": metadata.get("environment", metadata.get("kernel_environment", {})),
        "config": config,
        "inputs": inputs,
        "cache": cache,
        "dtype": "torch.bfloat16",
    }


def capture_identity(directory):
    if directory is None:
        return None
    content = (directory / "metadata.json").read_bytes()
    capture = json.loads(content)
    files = {}
    for entry in capture["layers"]:
        path = directory / entry["file"]
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Capture file escapes input directory")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["file_sha256"]:
            raise ValueError(f"Capture file hash mismatch: {path}")
        files[entry["file"]] = digest
    return {"metadata_sha256": hashlib.sha256(content).hexdigest(), "files": files}


def open_validation(args, metadata, *, kind, config, cache):
    identity = validation_identity(
        metadata, config=config, inputs=capture_identity(args.input_dir), cache=cache
    )
    metadata["mode"] = args.mode
    metadata["validation_identity"] = identity
    args.validation_cases = {}
    if args.mode == "check":
        return identity
    receipt = require_receipt(args.validation_receipt, kind=kind, identity=identity)
    args.validation_cases = receipt["checks"]["cases"]
    # Store the receipt beside the result so report generation needs no mutable
    # external path, and preserve its independent check provenance.
    content = args.validation_receipt.read_bytes()
    destination = args.output_dir / "validation_receipt.json"
    if destination.resolve() != args.validation_receipt.resolve():
        destination.write_bytes(content)
    metadata["numerical_validation"] = {
        "kind": kind,
        "file": destination.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "source": "independent_check",
    }
    return identity


def check_case_identity(args, label, tensors):
    """Verify real tensor layout/content before executing any performance samples."""
    if args.mode == "check":
        return None
    checked = args.validation_cases.get(label)
    if checked is None or checked.get("tensors") != tensors:
        raise ValueError(f"Validation receipt input/layout mismatch: {label}")
    return checked


def finish_validation(args, metadata, *, kind, identity, cases):
    if args.mode != "check":
        return
    destination = args.validation_receipt or args.output_dir / "validation_receipt.json"
    write_receipt(
        destination,
        kind=kind,
        identity=identity,
        checks={
            "passed": True,
            "run_id": args.run_id,
            "cases": cases,
            "scope": "independent numerical and lifecycle checks; no performance measurement",
        },
    )
    metadata["numerical_validation"] = {
        "kind": kind,
        "file": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source": "independent_check",
    }
