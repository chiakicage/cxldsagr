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
ISOLATED_RECEIPT_KIND = "deepseek-v32-checkpoint-layers-0-2-single-method-v4"
METHOD_ISOLATION = "fresh-process-one-method-v1"
Q1_PREFETCH_PREPARATION = "official-q1-current-page64-and-staging-v1"
GRAPH_CHECK_FIELDS = {
    "default_graph_logits",
    "default_graph_cache",
    *(
        f"replay_{repeat}_{output}"
        for repeat in range(2)
        for output in ("hidden", "logits", "cache")
    ),
    *(f"changed_input_{output}" for output in ("hidden", "logits", "cache")),
}
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


def result_methods(result):
    """Return a validated execution selection, never silently treating v4 as v2."""
    schema = result.get("schema_version")
    if schema == 4:
        method = result.get("selected_method")
        if method not in METHODS or result.get("methods") != [method]:
            raise ValueError("Single-method run requires exactly its selected method")
        if result.get("method_isolation") != METHOD_ISOLATION:
            raise ValueError("Single-method run has an unsupported isolation policy")
        return (method,)
    if schema == 3:
        if tuple(result.get("methods", ())) != METHODS:
            raise ValueError("Schema-3 run must retain all four methods in their original order")
        return METHODS
    if schema == 2:
        return ("resident", "offload")
    raise ValueError(f"Unsupported experiment schema: {schema}")


methods_for_result = result_methods


def receipt_kind(result):
    return {
        2: "deepseek-v32-checkpoint-layers-0-2-v2",
        3: RECEIPT_KIND,
        4: ISOLATED_RECEIPT_KIND,
    }[result["schema_version"]]


def preparation_contract(method, warmups, compute_graphs):
    return {
        "compute_bank": "prepare_once_before_method_warmup" if compute_graphs else "disabled",
        "warmup_methods": [method] * warmups,
        "each_warmup": [
            "fresh_selected_cache",
            "complete_history_prefill",
            "snapshot_prefix",
            "restore_declared_residency",
            "prepare_extend_graph_if_enabled",
            "complete_default_extend",
            "release_snapshot",
        ],
        "runtime_binding": "after_all_selected_method_warmups_before_check_bench_or_profile",
    }


def validate_isolation(result):
    methods = result_methods(result)
    if result.get("schema_version") != 4:
        return
    warmups = result.get("warmups")
    if type(warmups) is not int or warmups < 1:
        raise ValueError("Single-method preparation requires positive bound warmups")
    if result.get("preparation_contract") != preparation_contract(
        methods[0], warmups, result["compute_graphs"]
    ):
        raise ValueError("Single-method ordered preparation contract changed")


def expected_check_fields(result, *, include_profile=False):
    methods = result_methods(result)
    if result.get("schema_version") == 2:
        fields = {
            "resident_default_extend_logits",
            "offload_default_extend_logits",
            "resident_vs_offload_prefix_logits",
            "resident_vs_offload_hidden",
            "resident_vs_offload_logits",
        }
    else:
        fields = {
            *(f"{method}_default_extend_logits" for method in methods),
            *(
                f"hbm_vs_{method}_{output}"
                for method in methods
                if method != "hbm"
                for output in ("prefill_logits", "hidden", "logits")
            ),
        }
    if include_profile:
        fields |= {
            *(f"{method}_profile_prefix_logits" for method in methods),
            *(
                f"{method}_profile_extend_{output}"
                for method in methods
                for output in ("hidden", "logits")
            ),
        }
    return fields


def common_workload_identity(result_or_identity):
    """Explicit cross-method projection; full method identities stay independently bound."""
    identity = (
        execution_identity(result_or_identity)
        if "schema_version" in result_or_identity
        else result_or_identity
    )
    if identity.get("method_isolation") != METHOD_ISOLATION:
        raise ValueError("Common workload comparison requires isolated method identities")
    excluded = {
        "methods",
        "selected_method",
        "dense_history_transport",
        "preparation_contract",
        "execution_runtime_artifacts",
    }
    return {key: value for key, value in identity.items() if key not in excluded}


def _validate_fused_preparation(runtime, declared):
    """Bind actual current-key participation to its declared immutable mapped ELF."""
    observed = runtime.get("q1_prefetch_preparation")
    if not isinstance(observed, dict) or (
        observed.get("preparation") != Q1_PREFETCH_PREPARATION
        or observed.get("entry_point") != "logits_from_keys"
    ):
        raise ValueError("Selected ECHO Q1 method is missing observed fused preparation")
    native = observed.get("native") or {}
    if (native.get("build_identity") or {}).get("source_identity") != declared:
        raise ValueError("Fused preparation native source identity differs from declaration")
    name, digest, path = (
        native.get("artifact_name"),
        native.get("artifact_sha256"),
        native.get("artifact_path"),
    )
    if (
        not isinstance(name, str)
        or not name.startswith("cxldsagr_official_prefetch_")
        or not name.endswith(".so")
        or not digest
        or not path
        or sum(
            row.get("name") == name
            and row.get("library", {}).get("sha256") == digest
            and row.get("library", {}).get("path") == path
            for row in runtime.get("local_native_jit", ())
        )
        != 1
    ):
        raise ValueError("Fused preparation is missing its actual immutable mapped adapter")


def validate_runtime_participation(result, runtime=None):
    """Require dispatched adapters from observed records without loading unused providers."""
    if result.get("schema_version") != 4:
        return
    (method,) = result_methods(result)
    runtime = result["execution_runtime_artifacts"] if runtime is None else runtime
    loaded = {
        row["name"]
        for row in runtime.get("native_jit", ())
        if row.get("loaded_in_this_process") and row.get("library")
    }
    if not {"rope", "silu_and_mul", "topk"} <= loaded:
        raise ValueError("Selected method is missing required FlashInfer runtime adapters")
    if not runtime.get("cute_jit") or not (runtime.get("linear_quantization_triton") or {}).get(
        "specializations"
    ):
        raise ValueError("Selected method is missing observed norm/linear runtime adapters")
    local = runtime.get("local_native_jit", ())
    if method != "hbm":
        for category in ("echo_indexer", "record_transfer"):
            if sum(row.get("category") == category for row in local) != 1:
                raise ValueError(f"Selected offload method is missing required {category} adapter")
    declared_prefetch = result.get("backend_provenance", {}).get(
        "official_echo_prefetch_adapter", {}
    )
    preparation = declared_prefetch.get("preparation")
    if preparation not in (None, Q1_PREFETCH_PREPARATION):
        raise ValueError("Unsupported official Q1 preparation identity")
    observed_preparation = runtime.get("q1_prefetch_preparation")
    if observed_preparation is not None:
        if preparation is None or method != "echo":
            raise ValueError("Observed fused preparation does not match the selected source path")
        _validate_fused_preparation(runtime, declared_prefetch)
    extend_has_q1 = (
        result["extend_chunk_size"] == 1
        or result["extend_tokens"] % result["extend_chunk_size"] == 1
    )
    current_q1_extend = (
        preparation == Q1_PREFETCH_PREPARATION
        and method == "echo"
        and extend_has_q1
        and result["prefix_tokens"] + result["extend_tokens"] >= 32768
    )
    if (
        current_q1_extend
        and result["extend_tokens"] == 1
        and result.get("extend_residency") == "cold"
        and result["slots"] - 1 >= 64
    ):
        # A cold A1 uses official Q1 through bounded-free or full preparation.
        # Warming an unused resident packer cannot replace its evidence.
        _validate_fused_preparation(runtime, declared_prefetch)
    # Warm or chunked requests can also execute official Q1. This marker proves
    # process participation, not every call; profile nodes retain that boundary.
    # Only the observed entry grants an exemption, never a residency label.
    fused_extend = current_q1_extend and observed_preparation is not None
    q1_ends = []
    if extend_has_q1:
        q1_ends.append(result["prefix_tokens"] + result["extend_tokens"])
    if result["chunk_size"] == 1 or result["prefix_tokens"] % result["chunk_size"] == 1:
        q1_ends.append(result["prefix_tokens"])
    if any(end >= 2048 for end in q1_ends) and not (
        runtime.get("attention_decode_triton") or {}
    ).get("specializations"):
        raise ValueError("Selected Q1 method is missing its attention decode specialization")
    if any(end >= 32768 for end in q1_ends):
        topk = runtime.get("q1_topk_cub_native")
        if not topk or not any(
            row.get("library", {}).get("sha256") == topk.get("artifact_sha256")
            and row.get("name") == topk.get("artifact_name")
            for row in local
        ):
            raise ValueError("Selected Q1 method is missing its mapped CUB top-k adapter")
        prefill_q1 = (
            result["chunk_size"] == 1 or result["prefix_tokens"] % result["chunk_size"] == 1
        ) and result["prefix_tokens"] >= 32768
        if (not fused_extend or prefill_q1) and not runtime.get(
            "indexer_adaptation_triton", {}
        ).get("page64"):
            raise ValueError("Selected Q1 method is missing its page64 specialization")
    if (
        method == "echo"
        and result["extend_tokens"] == 1
        and result["prefix_tokens"] + 1 >= 32768
        and result["slots"] - result["prefix_tokens"] >= 64
    ):
        names = [row.get("name", "") for row in local]
        for prefix in ("cxldsagr_official_echo_decode_", "cxldsagr_official_prefetch_"):
            if sum(name.startswith(prefix) for name in names) != 1:
                raise ValueError("Selected ECHO Q1 method is missing its official mapped adapter")
        if not runtime.get("indexer_adaptation_triton", {}).get("decode_hint"):
            raise ValueError("Selected ECHO Q1 method is missing its decode hint specialization")
        if result["prefix_tokens"] + 1 == 65537:
            hint = runtime.get("q1_hint_native")
            if not hint or not any(
                row.get("library", {}).get("sha256") == hint.get("artifact_sha256")
                and row.get("name") == hint.get("artifact_name")
                for row in local
            ):
                raise ValueError("Selected ECHO Q1 method is missing its mapped exact hint adapter")


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
    if result.get("schema_version") == 4:
        validate_isolation(result)
    fields = (
        IDENTITY_FIELDS
        if result.get("schema_version") in (3, 4)
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
    if result.get("schema_version") == 4:
        fields += ("selected_method", "method_isolation", "warmups", "preparation_contract")
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


def _validate_check_contents(result, receipt):
    if set(receipt["checks"].get("comparisons", {})) != expected_check_fields(result):
        raise ValueError("Independent check lacks complete output comparisons")
    if result.get("extend_graph"):
        graph_checks = receipt["checks"].get("extend_graph", {})
        if set(graph_checks) != set(result_methods(result)) or any(
            set(row.get("checks", {})) != GRAPH_CHECK_FIELDS for row in graph_checks.values()
        ):
            raise ValueError("Independent check lacks full graph baseline/cache/replay validation")


def require_hbm_reference(path, result):
    """Authenticate an isolated HBM check and the shared workload before CPU output use."""
    if result.get("schema_version") != 4:
        raise ValueError("HBM reference requires a schema-4 isolated method")
    raw = json.loads(Path(path).read_text())
    identity = raw.get("identity", {})
    if identity.get("methods") != ["hbm"] or identity.get("selected_method") != "hbm":
        raise ValueError("Cross-method reference must be an isolated HBM check")
    receipt = require_receipt(path, kind=ISOLATED_RECEIPT_KIND, identity=identity)
    reference_path = receipt["artifact_paths"].get("result.json")
    if reference_path is None:
        raise ValueError("HBM reference lacks its signed check result")
    reference = json.loads(Path(reference_path).read_text())
    if (
        reference.get("schema_version") != 4
        or reference.get("mode") != "check"
        or reference.get("accepted") is not True
        or result_methods(reference) != ("hbm",)
    ):
        raise ValueError("Cross-method reference is not an accepted isolated HBM check")
    if identity_digest(execution_identity(reference)) != identity_digest(identity):
        raise ValueError("HBM reference result and receipt identities differ")
    _validate_check_contents(reference, receipt)
    validate_runtime_participation(reference)
    common = common_workload_identity(result)
    if identity_digest(common_workload_identity(reference)) != identity_digest(common):
        raise ValueError("HBM reference does not cover this common workload")
    for name in ("hbm_control.pt", "hbm_prefix_logits.pt"):
        if name not in receipt["artifact_paths"]:
            raise ValueError(f"HBM reference lacks required saved output: {name}")
    return receipt


def hbm_reference_binding(receipt, result):
    return {
        **receipt_binding(receipt),
        "execution_identity": receipt["identity"],
        "common_workload_sha256": identity_digest(common_workload_identity(result)),
    }


def validated_receipt(result):
    identity = result["execution_identity"]
    if identity_digest(identity) != identity_digest(execution_identity(result)):
        raise ValueError("Stored execution identity differs from run metadata")
    binding = result["validation_receipt"]
    kind = receipt_kind(result)
    receipt = require_receipt(binding["receipt_path"], kind=kind, identity=identity)
    if receipt["receipt_sha256"] != binding["receipt_sha256"]:
        raise ValueError("Independent check receipt changed")
    _validate_check_contents(result, receipt)
    if result.get("schema_version") == 4:
        validate_runtime_participation(result)
        reference = receipt["checks"].get("hbm_reference")
        if result_methods(result) == ("hbm",):
            if reference is not None:
                raise ValueError("HBM check cannot itself bind another method reference")
        else:
            if not isinstance(reference, dict):
                raise ValueError("Isolated offload check lacks its independent HBM reference")
            hbm = require_hbm_reference(reference["receipt_path"], result)
            if reference != hbm_reference_binding(hbm, result):
                raise ValueError("Independent HBM reference binding changed")
            if result.get("hbm_reference") != reference:
                raise ValueError("Result and receipt HBM reference bindings differ")
    return receipt


def validate_samples(result):
    for mode in result.get("methods", ("resident", "offload")):
        for phase in (
            ("prefill", "extend")
            if result.get("schema_version") in (3, 4)
            else ("prefix", "extend")
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
        result.get("schema_version") not in (2, 3, 4)
        or result.get("mode") != "bench"
        or not result["accepted"]
    ):
        raise ValueError("A successful independent schema-2/3/4 bench is required")
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
            (result_methods(result)[0] + "_control.pt")
            if result.get("schema_version") == 4
            else "hbm_control.pt"
            if result.get("schema_version") == 3
            else "resident_control.pt"
        ]
    ).parent


def validate_completed_result(result_path, *, expected_mode, expected_profile_detail=None):
    """Strict child completion gate shared by runners and CPU publication consumers."""
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    if (
        result.get("schema_version") not in (3, 4)
        or result.get("accepted") is not True
        or result.get("mode") != expected_mode
        or result.get("num_layers") != 3
    ):
        raise ValueError("Child did not complete the requested accepted three-layer run")
    methods = result_methods(result)
    validate_isolation(result)
    if identity_digest(result["execution_identity"]) != identity_digest(execution_identity(result)):
        raise ValueError("Completed child execution identity differs from metadata")
    if expected_mode == "check":
        receipt = require_receipt(
            result_path.with_name("receipt.json"),
            kind=receipt_kind(result),
            identity=result["execution_identity"],
        )
        # Reuse the exact same receipt validation used before performance samples.
        validated_receipt({**result, "validation_receipt": receipt_binding(receipt)})
        expected = expected_check_fields(result)
    else:
        validated_receipt(result)
        expected = (
            set()
            if expected_mode == "bench"
            else expected_check_fields(result, include_profile=True)
        )
    if set(result.get("correctness", {})) != expected:
        raise ValueError("Completed child lacks the exact required comparison set")
    if expected_mode == "bench":
        validate_benchmark(result_path.parent, result)
    if expected_mode == "profile":
        if (
            expected_profile_detail is not None
            and result.get("profile_detail") != expected_profile_detail
        ):
            raise ValueError("Profile child used the wrong observation boundary")
        if (
            result.get("profile_detail") == "minimal_node_model_scopes"
            and result.get("measurement_identity", {}).get("operator_wrappers_during_forward")
            is not False
        ):
            raise ValueError("Minimal profile unexpectedly installed operator wrappers")
        captures = ["graph_setup"] if result["compute_graphs"] else []
        for method in methods:
            captures.append(f"{method}/prefill_annotated")
            if result.get("extend_graph"):
                captures.append(f"{method}/extend_graph_setup")
            captures.append(f"{method}/extend_annotated")
        if result.get("nsys_capture_order") != captures:
            raise ValueError("Profile child has incomplete or foreign-method capture order")
        if result.get("extend_graph") and len(result.get("full_extend_graph_templates", ())) != len(
            methods
        ):
            raise ValueError("Profile child lacks the selected method full-extend templates")
    if expected_mode in ("bench", "profile") and set(result.get("measurements", {})) != set(
        methods
    ):
        raise ValueError("Child measurements contain missing or foreign methods")
    if result.get("schema_version") == 4:
        provenance = result.get("process_provenance", {})
        if (
            type(provenance.get("pid")) is not int
            or not provenance.get("start_ticks")
            or not provenance.get("driver_started_utc")
            or provenance.get("completed_method_warmups") != [methods[0]] * result["warmups"]
            or not provenance.get("cache_method_selections")
            or set(provenance["cache_method_selections"]) != set(methods)
        ):
            raise ValueError("Child process provenance does not prove selected-method preparation")
    return result


def validate_operator_analysis(result_path, analysis_path):
    result = validate_completed_result(
        result_path, expected_mode="profile", expected_profile_detail="matrix_api_node_ownership"
    )
    report = json.loads(Path(analysis_path).read_text())
    captures = report.get("captures", ())
    if (
        len(captures) != 2 * len(result_methods(result))
        or report.get("calls_outside_selected_captures") != 0
    ):
        raise ValueError("Operator analysis lacks exactly the selected measured captures")
    for capture in captures:
        audit = capture.get("audit", {})
        if not audit.get("kernel_count_and_time_conserved") or not audit.get(
            "metadata_call_counts_match"
        ):
            raise ValueError("Operator capture failed activity/call conservation")
    return report
