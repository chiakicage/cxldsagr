"""Compare exclusive non-matrix scope costs in two accepted three-layer runs."""

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from experiments.deepseek_v32_mfu.src.compare_backends import (
    COMMON_FIELDS,
    OPTIONAL_COMMON_FIELDS,
    validate_result,
)

GROUPS = {
    "hadamard": ("normalized_hadamard",),
    "index_quantization": ("quantize_index",),
    "rotary": ("apply_rope", "apply_rope_pair", "prepare_rotary_cache"),
    "norm": (
        "rms_norm",
        "residual_rms_norm",
        "input_residual_norm",
        "post_attention_residual_norm",
    ),
    "mlp_activation": ("dense_mlp", "silu_mul"),
    "projection_helpers": ("attention_projection",),
    "exact_topk": ("exact_topk",),
    "cache": (
        "cache_write",
        "index_cache_write",
        "offload_prepare",
        "offload_exact_recall",
        "offload_finalize",
        "offload_source_reservation",
        "indexer_prefetch_aux",
    ),
}
LABELS = {
    "hadamard": "Hadamard",
    "index_quantization": "Indexer quantization",
    "rotary": "RoPE + trig table",
    "norm": "Norm + residual",
    "mlp_activation": "MLP activation + packing",
    "projection_helpers": "Projection helpers",
    "exact_topk": "Exact top-k + helpers",
    "cache": "Cache work",
    "other_nonmatrix": "Other non-matrix scopes",
    "matrix_call_scopes": "Matrix-call scopes",
    "nonmatrix_excluding_cache": "Non-cache non-matrix total",
    "all_nonmatrix_scopes": "Non-matrix total including cache",
    "all_kernel_scopes": "All kernel scopes",
}
MODES = ("resident", "offload")
PHASES = ("prefill_annotated", "extend_annotated")
CACHE_METRICS = (
    "written_records",
    "recalled_records",
    "evicted_records",
    "capacity_splits",
    "selection_records",
    "resident_selection_records",
    "prefetched_records",
    "host_to_device_bytes",
    "device_to_host_bytes",
)
BOUNDARY = (
    "Every scope contributes only its exclusive kernel count/ns. Inclusive host/NVTX "
    "durations, memcpy and memset durations are excluded. Cache scopes are reported "
    "separately from nonmatrix_excluding_cache. Matrix-call scopes retain their nested "
    "activation quantization/helpers; this is not a separation of every non-GEMM kernel. "
    "Prefetch fused inside indexer_fused remains in matrix_call_scopes, outside the cache "
    "group. Kernel-duration sums are not GPU-busy unions or wall latency. "
    "Hadamard removal changes FP8 values and sparse selections, so cache work may change "
    "despite identical cache code/budgets. These results do not establish a cache optimization."
)
TYPED_NORM_FACTORY = "operators.deepseek_v32.norm._compile._get_compiled_typed_norm_kernel"
TYPED_NORM_NAMESPACE = "operatorsdeepseek_v32norm_"
TYPED_NORM_SIGNATURE = "bf16_fp32_bf16_align128_weight16_sm90"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def classify(stage, matrix, groups=GROUPS):
    matches = [name for name, stages in groups.items() if stage in stages]
    if len(matches) > 1:
        raise ValueError(f"Overlapping scope mapping: {stage}")
    if matches:
        if matrix:
            raise ValueError(f"Matrix scope incorrectly mapped as non-matrix: {stage}")
        return matches[0]
    if matrix:
        return "matrix_call_scopes"
    if stage.startswith(("offload_", "cache_", "index_cache_")):
        raise ValueError(f"Unmapped cache scope: {stage}")
    return "other_nonmatrix"


def aggregate(operators):
    """Partition the exclusive ledger, rejecting duplicate or overlapping entries."""
    totals = defaultdict(lambda: {"kernel_count": 0, "kernel_ns": 0, "stages": set()})
    seen = set()
    for row in operators:
        key = (row["mode"], row["phase"], row["stage"])
        if key in seen or row["layer"] != "all_layers":
            raise ValueError(f"Duplicate or non-aggregate scope row: {key}")
        seen.add(key)
        if row["mode"] not in MODES or row["phase"] not in PHASES:
            raise ValueError(f"Unexpected capture: {key}")
        if row["metadata_call_count_matches"] is not True:
            raise ValueError(f"Scope/call metadata mismatch: {key}")
        group = classify(row["stage"], row["useful_flops"] is not None)
        target = totals[(row["mode"], row["phase"], group)]
        for field in ("kernel_count", "kernel_ns"):
            value = row[field]
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid exclusive {field}: {key}")
            target[field] += value
        target["stages"].add(row["stage"])
    all_groups = (*GROUPS, "other_nonmatrix", "matrix_call_scopes")
    for mode in MODES:
        for phase in PHASES:
            for group in all_groups:
                totals[(mode, phase, group)]
            for total_name, selected in (
                (
                    "nonmatrix_excluding_cache",
                    [g for g in all_groups if g not in ("cache", "matrix_call_scopes")],
                ),
                ("all_nonmatrix_scopes", [g for g in all_groups if g != "matrix_call_scopes"]),
                ("all_kernel_scopes", all_groups),
            ):
                target = totals[(mode, phase, total_name)]
                for group in selected:
                    source = totals[(mode, phase, group)]
                    for field in ("kernel_count", "kernel_ns"):
                        target[field] += source[field]
                    target["stages"].update(source["stages"])
    return dict(totals)


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _typed_norm_fingerprint(result):
    identity = result.get("backend_provenance", {}).get("typed_norm")
    if (
        not isinstance(identity, dict)
        or identity.get("loaded_process_identity") is not True
        or not isinstance(identity.get("versions"), dict)
        or not identity["versions"]
        or not isinstance(identity.get("sources"), dict)
    ):
        raise ValueError("Typed norm requires recorded source and toolchain identity")
    sources = identity["sources"]
    required = {
        "local/" + name
        for name in (
            "api.py",
            "_compile.py",
            "_fingerprint.py",
            "_layout.py",
            "_plain.py",
            "_fused.py",
        )
    }
    if not required <= sources.keys() or any(
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or not record["path"]
        or not _is_sha256(record.get("sha256"))
        for record in sources.values()
    ):
        raise ValueError("Typed norm source manifest is missing or malformed")
    canonical = {
        "versions": identity["versions"],
        "sources": {name: record["sha256"] for name, record in sources.items()},
    }
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    if identity.get("fingerprint") != fingerprint:
        raise ValueError("Typed norm source fingerprint differs from its recorded manifest")
    return fingerprint


def _typed_norm_scopes(module, fingerprint):
    key = module.get("compile_key")
    if not isinstance(key, list) or len(key) != 7:
        raise ValueError("Typed norm compile key must contain seven fields")
    kind, width, device, pdl, compiled_fingerprint, signature, options = key
    if (
        kind not in ("plain", "fused")
        or type(width) is not int
        or width not in (512, 1536, 7168)
        or (kind == "fused" and width != 7168)
        or type(device) is not int
        or device < 0
        or type(pdl) is not bool
        or signature != TYPED_NORM_SIGNATURE
        or options != "--enable-tvm-ffi"
    ):
        raise ValueError(
            "Typed norm compile key differs from the validated BF16/FP32 SM90 contract"
        )
    if compiled_fingerprint != fingerprint:
        raise ValueError("Typed norm compiled source fingerprint differs from backend identity")
    ir = module.get("mlir_bytecode")
    if (
        not isinstance(ir, dict)
        or not _is_sha256(ir.get("sha256"))
        or type(ir.get("bytes")) is not int
        or ir["bytes"] <= 0
    ):
        raise ValueError("Typed norm requires recorded live compiled MLIR identity")
    names = module.get("kernel_names")
    prefix = "kernel_cutlass_kernel_" + TYPED_NORM_NAMESPACE + kind
    prefix += "Local" + kind.title() + "RMSNorm_"
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name.startswith(prefix) for name in names)
    ):
        raise ValueError("Typed norm kernel identity does not match its compiled factory kind")
    allowed = {"residual_rms_norm"} if kind == "fused" else {"rms_norm", "residual_rms_norm"}
    return {name: allowed for name in names}


def verify_runtime(result, inventory):
    runtime = result.get("flashinfer_runtime_artifacts", {})
    checked, unavailable, registered = [], [], set()
    typed_fingerprint, typed_scopes = None, {}
    for module in runtime.get("native_jit", []):
        if module["loaded_in_this_process"] is not True:
            raise ValueError("A recorded native module was not loaded in the measured process")
        for record in (module["library"], module["build_metadata"], *module["sources"]):
            path = Path(record["path"])
            if not path.exists():
                unavailable.append(str(path))
            elif digest(path) != record["sha256"]:
                raise ValueError(f"Live native artifact differs from recorded identity: {path}")
            else:
                checked.append(record)
    for module in runtime.get("cute_jit", []):
        factory = module.get("factory", "")
        if factory == TYPED_NORM_FACTORY:
            if typed_fingerprint is None:
                typed_fingerprint = _typed_norm_fingerprint(result)
            typed_scopes.update(_typed_norm_scopes(module, typed_fingerprint))
        else:
            if factory.startswith("operators.deepseek_v32.norm.") or any(
                TYPED_NORM_NAMESPACE in name for name in module.get("kernel_names", [])
            ):
                raise ValueError("Typed norm kernel requires its recorded typed compiler factory")
            # Preserve the vendor adapter contract and legacy metadata shape.
            if not module.get("compile_key") or module["compile_key"][0] != "float32":
                raise ValueError("DeepSeek norm runtime did not retain FP32 inputs/weights")
        registered.update(module["kernel_names"])
    observed = set()
    for row in inventory:
        name, stage = row["kernel_name"], row["stage"]
        if row["unattributed_reason"]:
            raise ValueError("Unattributed kernel in inventory")
        allowed = None
        if TYPED_NORM_NAMESPACE in name:
            observed.add(name)
            if name not in typed_scopes:
                raise ValueError("Typed CuTe profile kernel lacks a recorded live JIT identity")
            allowed = typed_scopes[name]
        elif "flashinfernormkernels" in name:
            observed.add(name)
            if name not in registered:
                raise ValueError("CuTe profile kernel lacks a recorded live JIT identity")
            allowed = (
                {"residual_rms_norm"}
                if "FusedAddRMSNorm" in name
                else {"rms_norm", "residual_rms_norm"}
            )
        elif "BatchQKApplyRotary" in name:
            allowed = {"apply_rope_pair"}
        elif "act_and_mul_kernel" in name:
            allowed = {"silu_mul"}
        elif name == "quantize_kernel":
            allowed = {"quantize_index"}
        if allowed is not None and stage not in allowed:
            raise ValueError(f"Kernel assigned to wrong non-matrix scope: {name}, {stage}")
    return {
        "recorded_runtime": runtime,
        "live_native_files_verified": checked,
        "live_native_files_unavailable": unavailable,
        "observed_cute_kernel_names": sorted(observed),
        "registered_cute_kernel_names": sorted(registered),
        "typed_norm_source_fingerprint": typed_fingerprint,
        "runtime_scope_mapping_verified": True,
        "note": "MLIR hashes describe objects recorded in the measured process; absent live "
        "cache files are disclosed and are not reconstructed from unrelated files.",
    }


def audit_run(directory, result, analysis, grouped):
    manifest = result["source_sha256"]
    snapshot_verified, current_verified, current_changed = [], [], []
    root = directory.parents[4]
    for name, expected in manifest.items():
        snapshot = directory / "source" / name
        if not snapshot.is_file() or digest(snapshot) != expected:
            raise ValueError(f"Recorded source snapshot mismatch: {name}")
        snapshot_verified.append(name)
        current = root / name
        if current.is_file() and digest(current) == expected:
            current_verified.append(name)
        else:
            current_changed.append(name)
    if analysis["ledger_metadata"]["run_id"] != result["run_id"]:
        raise ValueError("Analysis/run ID mismatch")
    if analysis["calls_sha256"] != digest(directory / "operator_calls.json"):
        raise ValueError("Analysis call-ledger identity mismatch")
    if analysis["calls_outside_selected_captures"] != 0:
        raise ValueError("Calls exist outside selected captures")
    captures = []
    seen = set()
    for capture in analysis["captures"]:
        key = capture["mode"], capture["phase"]
        if key in seen:
            raise ValueError("Duplicate capture")
        seen.add(key)
        stored = capture["audit"]
        if (
            not stored["kernel_count_and_time_conserved"]
            or not stored["metadata_call_counts_match"]
            or stored["layer_unscoped_kernel_count"]
            or stored["unattributed_reasons"]
            or capture["api_summary"]["outside_echo_scope_thread_union_ms"]
        ):
            raise ValueError("Capture attribution audit failed")
        raw = directory / Path(capture["sqlite"]).name
        if digest(raw) != capture["sqlite_sha256"]:
            raise ValueError("Capture SQLite identity mismatch")
        with sqlite3.connect(raw.as_uri() + "?mode=ro", uri=True) as connection:
            count, ns = connection.execute(
                'SELECT COUNT(*), SUM("end" - "start") FROM CUPTI_ACTIVITY_KIND_KERNEL'
            ).fetchone()
        expected = grouped[(*key, "all_kernel_scopes")]
        inventory = [
            row for row in analysis["kernel_inventory"] if (row["mode"], row["phase"]) == key
        ]
        if (count, ns) != (expected["kernel_count"], expected["kernel_ns"]) or (
            count != sum(row["count"] for row in inventory)
            or ns != sum(row["total_ns"] for row in inventory)
        ):
            raise ValueError("Exclusive groups/inventory do not conserve raw SQLite kernels")
        captures.append(
            {
                "mode": key[0],
                "phase": key[1],
                "raw_kernel_count": count,
                "raw_kernel_ns": ns,
                "sqlite_sha256": capture["sqlite_sha256"],
                "attribution_audit": stored,
            }
        )
    if seen != {(mode, phase) for mode in MODES for phase in PHASES}:
        raise ValueError("Expected four captures")
    return {
        "run_id": result["run_id"],
        "source_sha256": manifest,
        "snapshot_verified_count": len(snapshot_verified),
        "current_verified_count": len(current_verified),
        "current_changed_or_missing": current_changed,
        "correctness": result["correctness"],
        "captures": captures,
        "runtime": verify_runtime(result, analysis["kernel_inventory"]),
    }


def compare(control_directory, candidate_directory):
    directories = {"control": control_directory, "candidate": candidate_directory}
    results, analyses, grouped, audits = {}, {}, {}, {}
    for label, directory in directories.items():
        result = results[label] = read_json(directory / "result.json")
        validate_result(directory, result)
        analysis = analyses[label] = read_json(directory / "analysis/analysis.json")
        grouped[label] = aggregate(analysis["operators"])
        audits[label] = audit_run(directory, result, analysis, grouped[label])
    for name in (*COMMON_FIELDS, *OPTIONAL_COMMON_FIELDS, "dependencies"):
        if results["control"].get(name) != results["candidate"].get(name):
            raise ValueError(f"Comparison workload/budget mismatch: {name}")
    if (
        results["control"]["hardware"]["torch_device_uuid"]
        != results["candidate"]["hardware"]["torch_device_uuid"]
    ):
        raise ValueError("This paired comparison requires the same physical GPU")
    records = []
    for mode, phase, group in grouped["control"]:
        row = {"mode": mode, "phase": phase, "group": group, "label": LABELS[group]}
        row["row_kind"] = (
            "total"
            if group in ("nonmatrix_excluding_cache", "all_nonmatrix_scopes", "all_kernel_scopes")
            else "group"
        )
        for label in directories:
            values = grouped[label][(mode, phase, group)]
            for field in ("kernel_count", "kernel_ns"):
                row[f"{label}_{field}"] = values[field]
            row[f"{label}_kernel_ms"] = values["kernel_ns"] / 1e6
            row[f"{label}_stages"] = sorted(values["stages"])
        row["kernel_count_reduction"] = row["control_kernel_count"] - row["candidate_kernel_count"]
        row["kernel_ms_reduction"] = row["control_kernel_ms"] - row["candidate_kernel_ms"]
        records.append(row)
    cache_work = []
    for phase in ("prefix", "extend"):
        for metric in CACHE_METRICS:
            row = {"phase": phase, "metric": metric}
            for label, result in results.items():
                values = [
                    layer[metric]
                    for layer in result["measurements"]["offload"][phase + "_cache_per_layer"]
                ]
                row[label + "_per_layer"] = values
                row[label] = sum(values)
            row["candidate_minus_control"] = row["candidate"] - row["control"]
            cache_work.append(row)
    old, new = (results[label]["source_sha256"] for label in directories)
    source_differences = {
        name: {"control": old.get(name), "candidate": new.get(name)}
        for name in sorted(old.keys() | new.keys())
        if old.get(name) != new.get(name)
    }
    document = {
        "schema_version": 1,
        "boundary": BOUNDARY,
        "comparison_source_sha256": digest(__file__),
        "comparison_support_source_sha256": digest(validate_result.__code__.co_filename),
        "group_mapping": GROUPS,
        "dynamic_group_rules": {
            "matrix_call_scopes": "useful_flops is not None; no explicit non-matrix group",
            "other_nonmatrix": "Remaining unmapped non-matrix scopes; resolved names in rows",
        },
        "runs": {
            label: {
                "run_id": results[label]["run_id"],
                "directory": str(directory),
                "input_sha256": {
                    name: digest(directory / name)
                    for name in ("result.json", "analysis/analysis.json", "operator_calls.json")
                },
                "compute_precision": results[label]["compute_precision"],
                "hardware": results[label]["hardware"],
            }
            for label, directory in directories.items()
        },
        "source_differences": source_differences,
        "cache_sources_equal": not any(name.startswith("cache/") for name in source_differences),
        "rows": records,
        "cache_work": cache_work,
    }
    return document, {"schema_version": 1, "runs": audits}


def write_svg(path, document):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = (*GROUPS, "other_nonmatrix")
    rows = {
        (r["mode"], r["group"]): r for r in document["rows"] if r["phase"] == "extend_annotated"
    }
    with plt.rc_context({"svg.fonttype": "none", "font.size": 10}):
        figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.6), sharex=True, sharey=True)
        for axis, mode in zip(axes, MODES):
            for offset, label, color in (
                (-0.17, "control", "#5479a8"),
                (0.17, "candidate", "#e89642"),
            ):
                values = [rows[(mode, group)][label + "_kernel_ms"] for group in groups]
                bars = axis.barh(
                    [i + offset for i in range(len(groups))],
                    values,
                    height=0.30,
                    label=label.title(),
                    color=color,
                )
                axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
            axis.set_title(mode.title())
            axis.set_xlabel("Exclusive CUDA kernel time (ms)")
            axis.set_yticks(range(len(groups)), [LABELS[group] for group in groups])
            axis.grid(axis="x", alpha=0.2)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(x=0.16)
        axes[0].invert_yaxis()
        axes[1].legend(loc="lower right")
        figure.suptitle("Extend1024 non-matrix scope comparison", fontsize=16, y=0.98)
        figure.text(0.02, 0.075, "Control: " + document["runs"]["control"]["run_id"], fontsize=8)
        figure.text(
            0.02, 0.050, "Candidate: " + document["runs"]["candidate"]["run_id"], fontsize=8
        )
        figure.text(
            0.02,
            0.025,
            "Packing, casts and trig tables are included; cache is separate. "
            "Matrix-call scopes are excluded.",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.10, 1, 0.95))
        figure.savefig(path)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Default: candidate run/nonmatrix_comparison")
    parser.add_argument("--svg", action="store_true", help="Draw both extend modes")
    args = parser.parse_args()
    output = args.output or args.candidate_run / "nonmatrix_comparison"
    document, audit = compare(args.control_run.resolve(), args.candidate_run.resolve())
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("comparison.json", document), ("source_runtime_audit.json", audit)):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    rows = [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in r.items()}
        for r in document["rows"]
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison_source.py").write_text(Path(__file__).read_text())
    (output / "comparison_support_source.py").write_text(
        Path(validate_result.__code__.co_filename).read_text()
    )
    if args.svg:
        write_svg(output / "extend_nonmatrix.svg", document)
    print(
        json.dumps(
            {
                "output": str(output),
                "runs": {k: v["run_id"] for k, v in document["runs"].items()},
                "rows": len(rows),
                "raw_kernel_conservation_verified": True,
            }
        )
    )


if __name__ == "__main__":
    main()
