"""Identity and immutable links between independent three-layer experiment runs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt

METHODS = ("hbm", "echo", "serial_sparse", "dense_prefetch")
RECEIPT_KIND = "deepseek-v32-checkpoint-layers-0-2-four-method-v3"
CHECK_FIELDS = {
    *(f"{method}_default_extend_logits" for method in METHODS),
    *(
        f"hbm_vs_{method}_{output}"
        for method in METHODS[1:]
        for output in ("prefill_logits", "hidden", "logits")
    ),
}
IDENTITY_FIELDS = (
    "methods",
    "extend_residency",
    "compute_graphs",
    "scope",
    "num_layers",
    "checkpoint_num_layers",
    "prefix_tokens",
    "extend_tokens",
    "chunk_size",
    "slots",
    "cache_policy_revision",
    "pool_scope",
    "sparse_pool_tokens",
    "host_arena_tokens",
    "workspace_query_tokens",
    "hbm_cache_budget_bytes",
    "dram_cache_budget_bytes",
    "extend_chunk_size",
    "snapshot_schema",
    "snapshot_scope",
    "prefetch_cap",
    "prefetch_flags",
    "timed_output",
    "source_sha256",
    "backend_provenance",
    "compute_precision",
    "torch_precision",
    "seed",
    "indexer_build",
    "request_sha256",
    "checkpoint_identity",
    "dependencies",
    "execution_environment",
    "execution_runtime_artifacts",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_identity(directory):
    """Hash metadata and stat every checkpoint shard, without reading weight payloads."""
    directory = Path(directory).resolve(strict=True)
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        path = directory / name
        stat = path.stat()
        shards[name] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino,
            "device": stat.st_dev,
        }
    return {
        "directory": str(directory),
        "metadata_sha256": {
            name: digest(directory / name)
            for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")
        },
        "shard_stat_inventory": shards,
        "boundary": "Metadata hashes and all-shard filesystem identity; no full weight hashing",
    }


def execution_identity(result):
    fields = (
        IDENTITY_FIELDS
        if result.get("schema_version") == 3
        else tuple(
            key
            for key in IDENTITY_FIELDS
            if key not in {"methods", "extend_residency", "compute_graphs"}
        )
    )
    # Older accepted runs predate the explicit dense transport declaration.
    # New runs bind it alongside the full execution source snapshot.
    if "dense_history_transport" in result:
        fields += ("dense_history_transport",)
    if "extend_graph" in result:
        fields += ("extend_graph", "extend_graph_policy_revision")
    return {
        **{key: result[key] for key in fields},
        "gpu": {
            key: value
            for key, value in result["hardware"]["gpu"].items()
            if key
            in {
                "uuid",
                "name",
                "pci.device_id",
                "compute_cap",
                "memory.total",
                "driver_version",
                "clocks.max.sm",
                "power.limit",
            }
        },
        "numerical_policy": {
            "rtol": 0.01,
            "atol": 0.02,
            "hidden": "all extend rows",
            "logits": "last token",
            "finite": True,
        },
    }


def receipt_binding(receipt):
    return {key: receipt[key] for key in ("receipt_path", "receipt_sha256")}


def validated_receipt(result):
    identity = result["execution_identity"]
    if identity_digest(identity) != identity_digest(execution_identity(result)):
        raise ValueError("Stored execution identity differs from run metadata")
    binding = result["validation_receipt"]
    kind = (
        RECEIPT_KIND
        if result.get("schema_version") == 3
        else "deepseek-v32-checkpoint-layers-0-2-v2"
    )
    receipt = require_receipt(binding["receipt_path"], kind=kind, identity=identity)
    if receipt["receipt_sha256"] != binding["receipt_sha256"]:
        raise ValueError("Independent check receipt changed")
    checks = receipt["checks"].get("comparisons", {})
    expected = (
        CHECK_FIELDS
        if result.get("schema_version") == 3
        else {
            "resident_default_extend_logits",
            "offload_default_extend_logits",
            "resident_vs_offload_prefix_logits",
            "resident_vs_offload_hidden",
            "resident_vs_offload_logits",
        }
    )
    if set(checks) != expected:
        raise ValueError("Independent check lacks complete output comparisons")
    if result.get("extend_graph"):
        graph_checks = receipt["checks"].get("extend_graph", {})
        required = {
            "default_graph_logits",
            "default_graph_cache",
            *(
                f"replay_{repeat}_{output}"
                for repeat in range(2)
                for output in ("hidden", "logits", "cache")
            ),
            *(f"changed_input_{output}" for output in ("hidden", "logits", "cache")),
        }
        if set(graph_checks) != set(METHODS) or any(
            set(row.get("checks", {})) != required for row in graph_checks.values()
        ):
            raise ValueError("Independent check lacks full graph baseline/cache/replay validation")
    return receipt


def validate_samples(result):
    for mode in result.get("methods", ("resident", "offload")):
        for phase in (
            ("prefill", "extend") if result.get("schema_version") == 3 else ("prefix", "extend")
        ):
            row = result["measurements"][mode]
            samples = row[phase + "_samples_ms"]
            count = result["prefill_repeats" if phase in ("prefix", "prefill") else "repeats"]
            if (
                len(samples) != count
                or not samples
                or not all(
                    type(value) in (int, float) and math.isfinite(value) and value > 0
                    for value in samples
                )
            ):
                raise ValueError("Invalid clean timing samples or repeat count")
            if statistics.median(samples) != row[phase + "_median_ms"]:
                raise ValueError("Stored clean median differs from samples")


def validate_benchmark(directory, result):
    directory = Path(directory)
    if (
        result.get("schema_version") not in (2, 3)
        or result.get("mode") != "bench"
        or not result["accepted"]
    ):
        raise ValueError("A successful independent schema-3 bench is required")
    if result["num_layers"] != 3 or result["correctness"]:
        raise ValueError("Clean bench cannot contain inline numerical checks")
    if digest(directory / "request.json") != result["request_sha256"]:
        raise ValueError("Benchmark request SHA mismatch")
    validated_receipt(result)
    validate_samples(result)
    for name, expected in result["source_sha256"].items():
        if digest(directory / "source" / name) != expected:
            raise ValueError(f"Benchmark source snapshot changed: {name}")


def bind_benchmark(directory, profile):
    directory = Path(directory).resolve(strict=True)
    path = directory / "result.json"
    result = json.loads(path.read_text())
    validate_benchmark(directory, result)
    if identity_digest(result["execution_identity"]) != identity_digest(
        profile["execution_identity"]
    ):
        raise ValueError("Independent bench does not cover this profile execution")
    if result["validation_receipt"] != profile["validation_receipt"]:
        raise ValueError("Profile and bench must bind the same independent check")
    return {
        "directory": str(directory),
        "run_id": result["run_id"],
        "result_sha256": digest(path),
        "source_sha256": result["source_sha256"],
        "execution_identity_sha256": identity_digest(result["execution_identity"]),
    }


def benchmark_view(directory, result, *, required=True):
    """Join clean wall samples for consumers while preserving the profile's own identity."""
    if result.get("schema_version", 1) == 1:
        return result
    validated_receipt(result)
    if result["mode"] == "bench":
        validate_benchmark(directory, result)
        return result
    if result["mode"] != "profile":
        raise ValueError("Performance consumers require a bench or profile run")
    binding = result.get("benchmark")
    if binding is None:
        if required:
            raise ValueError("Profile has no matching independent bench; formal report is pending")
        return result
    actual = bind_benchmark(binding["directory"], result)
    if actual != binding:
        raise ValueError("Bound independent bench changed")
    bench = json.loads((Path(binding["directory"]) / "result.json").read_text())
    return {
        **result,
        "measurements": bench["measurements"],
        "warmups": bench["warmups"],
        "repeats": bench["repeats"],
        "prefill_repeats": bench["prefill_repeats"],
        "wall_time_denominator": {"mode": "independent_bench", **binding},
    }


def control_directory(directory, result):
    if result.get("schema_version", 1) == 1:
        return Path(directory)
    receipt = validated_receipt(result)
    return Path(
        receipt["artifact_paths"][
            "hbm_control.pt" if result.get("schema_version") == 3 else "resident_control.pt"
        ]
    ).parent
