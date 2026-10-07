"""Audit and report the complete fixed-capacity C10 H/A matrix on CPU."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

from experiments.deepseek_v32_motivation.src import report
from experiments.deepseek_v32_motivation.src.measure import BENCH_SCHEMA, CHECK_SCHEMA
from experiments.nosa_motivation.src.cpu_environment import validate_cpu_environment_record

HISTORIES = (4096, 16384, 65536)
CANDIDATES = (128, 256, 512, 1024)
GRAPH_POLICY = "deepseek-compute-islands-v4-bound-inputs"
COUNTER_COMPARISON_POLICY = "echo-cap8192-rejected-reservation-attempts-v1"
# This closed exception applies only to the reviewed implementation snapshots.
# A changed implementation needs a new review before reusing the exception.
COUNTER_SOURCE_SHA256 = {
    "models/deepseek_v32/attention.py": "929e7afcc0b3c9abdddce464aca23a0c1a50dc19e656275bdf5031fd4dc4fc7b",
    "models/deepseek_v32/cache/session.py": "50e11d659389aeafc2ed8a608915f4eca2202e953529a4ea7180d2dd0edaaad9",
    "cache/sparse_token_cache.py": "1b3da2acfe5dc22c403b5794d6aedcebd1daaccaa9e82e54741fb0afd69c3cdf",
    "operators/deepseek_v32/indexer/echo.py": "de20ebc3b4905b3dddb646167fb770d27a92146e76934be3160ad1b5b149c562",
    "operators/deepseek_v32/indexer/csrc/echo_cache.cuh": "409be4a062180a8c8e73c1fa783d3980cf3e9926bf53ef34ddbaefe46e2dcd12",
    "operators/deepseek_v32/indexer/csrc/echo_logits.cuh": "562b10fbca2082948fef14a1ac1c40b8918e1c2c81717a965cad040f5293e61f",
    "operators/deepseek_v32/indexer/csrc/echo_indexer.cu": "c39ee55cb67ef452d6ee920c84c5852002ec8d2efd7c3115dfcb3564e9c8a255",
    "operators/deepseek_v32/indexer/csrc/echo_sparse_recall.cuh": "a7c33da4c60eb3354e93fc77cac39faf19810d4a5b52c03283242fa61e52ff44",
    "models/deepseek_v32/execution/adapter.py": "bd20895dca304db0923f1489ed093094f3d199574112035e4745d518d20e4989",
}
ECHO_FLAGS = {
    "fused_logits_recall_extend": True,
    "early_evict": False,
    "offset_policy": "mean_of_last_up_to_four_rows_with_finite_guard",
    "exact_union_overflow": "local_query_consumption_split_without_selection_clipping",
}
FIXED_CONFIG = {
    "num_users": 16,
    "rounds": 2,
    "requests_per_scheme": 32,
    "layers": 10,
    "chunk_size": 1024,
    "workspace_query_tokens": 1024,
    "sparse_pool_tokens": 65536,
    "host_arena_tokens": 16777216,
    "seed": 42,
    "sampling": "sequential",
    "enable_compute_graphs": True,
    "byte_subbudgets": None,
    "noncache_headroom_bytes": None,
}


def read(path):
    return json.loads(Path(path).read_text())


def strict_equal(left, right):
    """JSON equality that does not equate booleans, integers and floats."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            strict_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def echo_counter_proof(directory, metadata, check_dir, check_metadata):
    """Bind the derived cap and diagnostic meaning to saved execution identities."""
    methods = []
    for path, meta in ((directory, metadata), (check_dir, check_metadata)):
        config = meta["config"]
        report.require(
            strict_equal(config["sparse_pool_tokens"], 65536)
            and strict_equal(config["layers"], 10),
            "ECHO counter proof requires P=65536 and ten layers",
        )
        method = meta["validation_identity"]["methods"]["echo"]
        cases = [case for case in meta["cases"] if case["scheme"] == "echo"]
        report.require(len(cases) == 1, "ECHO counter proof requires one ECHO case")
        case, backend = cases[0], method["backend"]
        expected = {
            "scheme": "echo",
            "physical_layers": 10,
            "sparse_pool_tokens": 65536,
            "pool_scope": "backend_per_layer",
            "cache_policy_revision": "echo-global-pages-fifo-v1",
            "echo_flags": ECHO_FLAGS,
        }
        report.require(
            method["backend_type"] == "models.deepseek_v32.execution.adapter.DeepSeekServingBackend"
            and all(strict_equal(backend.get(key), value) for key, value in expected.items())
            and strict_equal(
                {key: value for key, value in case["backend"].items() if key != "compute_graphs"},
                {key: value for key, value in backend.items() if key != "compute_graphs"},
            ),
            "ECHO counter proof backend or flags differ",
        )
        plan = backend["resource_plan"]
        report.require(
            all(
                strict_equal(plan.get(key), value)
                for key, value in {
                    "resource_mode": "fixed_pools",
                    "pool_scope": "backend_per_layer",
                    "cache_policy_revision": "echo-global-pages-fifo-v1",
                    "shared_token_pool": True,
                    "sparse_pool_tokens": 65536,
                    "candidate_persistence": "gpu_transient",
                    "candidate_slots": config["candidate_tokens"],
                }.items()
            )
            and strict_equal(case["resource_plan"], plan),
            "ECHO counter proof resource plan differs",
        )
        native = method["native_artifacts"]
        echo_native = {
            name: identity
            for name, identity in native.items()
            if Path(name).name.startswith("cxldsagr_echo_indexer_") and name.endswith(".so")
        }
        report.require(
            len(echo_native) == 1
            and strict_equal(case["native_artifacts_before"], native)
            and strict_equal(case["native_artifacts_after"], native),
            "ECHO counter proof native identities differ or are missing",
        )
        for identity in echo_native.values():
            report.require(
                type(identity.get("size")) is int
                and identity["size"] > 0
                and isinstance(identity.get("sha256"), str)
                and len(identity["sha256"]) == 64
                and all(char in "0123456789abcdef" for char in identity["sha256"]),
                "ECHO counter proof has invalid native identity",
            )
        manifest = read(path / "source_manifest.json")
        execution_sources = meta["validation_identity"]["base"]["sources"]
        for name, expected_sha in COUNTER_SOURCE_SHA256.items():
            report.require(
                manifest.get(name) == execution_sources.get(name) == expected_sha
                and report.digest(path / "source" / name) == expected_sha,
                f"ECHO counter proof source differs: {name}",
            )
        methods.append(method)
    report.require(strict_equal(*methods), "ECHO counter proof check/bench identities differ")
    return {
        "status": "source_reviewed",
        "derived_prefetch_cap": 8192,
        "candidate_slots": metadata["config"]["candidate_tokens"],
        "source_sha256": COUNTER_SOURCE_SHA256,
        "backend": expected,
        "resource_plan": plan,
        "native_artifacts": native,
        "echo_indexer_native_artifacts": echo_native,
        "cap_derivation": (
            "attention.py:329 and cache/session.py:58-60 pass no limit override; "
            "cache/sparse_token_cache.py:741,805-807 defaults/clamps to 8192, with "
            "P=65536 and gpu_transient candidate storage; echo_logits.cuh:407,471 "
            "also caps prefetch tasks at 8192. This cap is derived, not recorded as a flag."
        ),
        "counter_semantics": (
            "echo_cache.cuh:19-49 counts successful temporary CAS claims rejected by "
            "the reservation cap, then restores those claims. echo_logits.cuh:568 "
            "permits schedule-dependent in-flight attempts. echo_indexer.cu:128-157 "
            "finalizes from the allocation log, not stats[2]; echo_sparse_recall.cuh "
            "resets the reused counter. sparse_token_cache.py:813-825,890-917 reports "
            "stats[2] without including it in actual transfer bytes."
        ),
        "boundary": (
            "Only rejected-reservation diagnostic counts may differ. Exact observed "
            "aggregate metrics do not prove identical logical prefetch sets or priority "
            "trajectories; temporary claims can affect concurrent claim opportunities."
        ),
    }


def compare_cache_diagnostics(bench, check, scheme, request_id, prove_fixed_cap):
    """Keep every field exact except the proven, saturated ECHO diagnostic leaf."""
    differences = []

    def compare(actual, reference, path=()):
        field = "cache_diagnostics" + "".join(
            f"[{part}]" if type(part) is int else f".{part}" for part in path
        )
        error = f"check/bench cache_diagnostics differs at {(scheme, request_id)}: {field}"
        eligible = (
            len(path) == 3
            and path[0] == "layers"
            and type(path[1]) is int
            and path[2] == "prefetch_capacity_failures"
        )
        if eligible:
            report.require(
                type(actual) is int and actual >= 0 and type(reference) is int and reference >= 0,
                error + " requires nonnegative integer counts",
            )
        report.require(type(actual) is type(reference), error + " (type)")
        if isinstance(actual, dict):
            report.require(actual.keys() == reference.keys(), error + " (keys)")
            for key, value in actual.items():
                compare(value, reference[key], (*path, key))
        elif isinstance(actual, list):
            report.require(len(actual) == len(reference), error + " (length)")
            for index, (value, checked) in enumerate(zip(actual, reference, strict=True)):
                compare(value, checked, (*path, index))
        elif actual != reference:
            report.require(eligible and scheme == "echo", error)
            layer = path[1]
            report.require(len(bench["layers"]) == len(check["layers"]) == 10, error)
            for record in (bench["layers"][layer], check["layers"][layer]):
                report.require(
                    strict_equal(record.get("prefetched_records"), 8192)
                    and strict_equal(record.get("device_slots"), 65536)
                    and record.get("pool_scope") == "shared_per_layer",
                    error + " requires saturated fixed-cap prefetch",
                )
            proof = prove_fixed_cap()
            report.require(
                strict_equal(proof["derived_prefetch_cap"], 8192)
                and all(
                    strict_equal(
                        records["layers"][layer].get("candidate_slots"), proof["candidate_slots"]
                    )
                    for records in (bench, check)
                ),
                error + " candidate storage differs from proof",
            )
            differences.append(
                {
                    "scheme": scheme,
                    "request_id": request_id,
                    "layer": layer,
                    "field": field,
                    "bench": actual,
                    "check": reference,
                    "derived_prefetch_cap": 8192,
                    "prefetched_records": 8192,
                }
            )

    compare(bench, check)
    return differences


def point_key(metadata):
    report.require(
        metadata.get("status") == "accepted" and metadata.get("schema") == BENCH_SCHEMA,
        "matrix inputs must be accepted independent-check benchmarks",
    )
    config = metadata["config"]
    for name, expected in FIXED_CONFIG.items():
        report.require(name in config and config[name] == expected, f"mixed matrix config: {name}")
    history, candidates = config["history_tokens"], config["candidate_tokens"]
    report.require(
        type(history) is int
        and type(candidates) is int
        and history in HISTORIES
        and candidates in CANDIDATES,
        "unexpected H/A matrix point",
    )
    report.require(config["padded_history_tokens"] == history, "incorrect aligned history length")
    report.require(config["schemes"] == list(report.SCHEMES), "incomplete scheme matrix")
    return history, candidates


def common_identity(metadata, directory):
    workload = read(directory / "workload/workload.json")
    config = workload["config"]
    for name in ("history_tokens", "candidate_tokens"):
        report.require(config[name] == metadata["config"][name], "workload shape differs")
    report.require(
        config["context_limit"] == config["history_tokens"] + config["candidate_tokens"],
        "workload context limit differs",
    )
    report.require(
        validate_cpu_environment_record(metadata)["status"] == "recorded_and_equal",
        "matrix requires matching recorded CPU affinity, threads and NUMA policy",
    )
    hardware = metadata["hardware"]
    cpu = hardware["cpu_environment"]
    hardware_identity = {
        name: hardware[name]
        for name in (
            "hostname",
            "python",
            "torch",
            "cuda",
            "device",
            "name",
            "uuid",
            "total_memory",
            "sm_count",
        )
    }
    hardware_identity["cpu_environment"] = {
        name: cpu[name]
        for name in (
            "affinity",
            "allowed",
            "torch_num_threads",
            "torch_num_interop_threads",
            "numa_policy",
            "environment",
        )
    }
    return {
        **{
            name: metadata[name]
            for name in (
                "source_sha256",
                "checkpoint",
                "precision_settings",
                "backend_provenance",
                "execution_environment",
                "model_dimensions",
                "model_boundary",
                "diagnostics_boundary",
                "memory_boundary",
            )
        },
        "hardware": hardware_identity,
        "execution_sources": metadata["validation_identity"]["base"]["sources"],
        "warmup_policy": metadata["config"]["warmup_policy"],
        "indexer_dispatch_policy": metadata["config"]["indexer_dispatch_policy"],
        "input_rules": {
            name: value
            for name, value in config.items()
            if name not in {"history_tokens", "candidate_tokens", "context_limit"}
        },
        "tokenizer_sha256": workload["tokenizer_sha256"],
        "heat_sha256": workload["heat_sha256"],
    }


def audit_memory(directory, metadata, rows):
    ledger = read(directory / "memory.json")
    by_stage = {sample["stage"]: sample for sample in ledger}
    report.require(len(by_stage) == len(ledger), "duplicate memory ledger stage")
    for row in rows:
        for side in ("before", "after"):
            sample = row[f"memory_{side}"]
            key = f"{row['scheme']}/{side}_request_{row['request_id']}"
            report.require(sample == by_stage.get(key), "request memory differs from ledger")
            allocated, reserved = sample["torch_allocated_bytes"], sample["torch_reserved_bytes"]
            report.require(0 <= allocated <= reserved, "invalid allocator memory sample")
            report.require(
                allocated <= sample["torch_peak_allocated_bytes"]
                and reserved <= sample["torch_peak_reserved_bytes"]
                and 0 <= sample["cuda_free_bytes"] <= sample["cuda_total_bytes"],
                "invalid peak or device memory sample",
            )
    result = {}
    queries = {metadata["config"]["candidate_tokens"], 1024}
    for case in metadata["cases"]:
        scheme = case["scheme"]
        selected = [row for row in rows if row["scheme"] == scheme]
        bank = case["backend"]["compute_graphs"]
        report.require(
            bank["enabled"]
            and bank["allocated"]
            and bank["policy_revision"] == GRAPH_POLICY
            and bank["eager_fallbacks"] == 0,
            "matrix requires the prepared v4 graph path without eager fallback",
        )
        graph_keys = [(entry["layer"], entry["queries"]) for entry in bank["graphs"]]
        report.require(
            len(graph_keys) == 10 * len(queries)
            and set(graph_keys) == {(layer, query) for layer in range(10) for query in queries},
            "incomplete graph layer/shape inventory",
        )
        item = {
            "max_cached_users": max(row["cached_users"] for row in selected),
            "maximum_after_request_device_used_bytes": max(
                row["memory_after"]["cuda_total_bytes"] - row["memory_after"]["cuda_free_bytes"]
                for row in selected
            ),
            **{
                name: max(row["memory_after"][name] for row in selected)
                for name in ("torch_peak_allocated_bytes", "torch_peak_reserved_bytes")
            },
            **{
                name: max(row[name] for row in selected)
                for name in ("reserved_hbm_bytes", "reserved_dram_bytes")
            },
            **{
                f"graph_{name}": bank[name]
                for name in (
                    "static_storage_bytes",
                    "static_allocated_bytes",
                    "private_reserved_bytes",
                )
            },
        }
        for name in ("torch_peak_allocated_bytes", "torch_peak_reserved_bytes"):
            report.require(item[name] == case[name], "case and request memory peaks differ")
        result[scheme] = item
    return result, len(ledger)


def audit_point(directory):
    metadata, rows, bench_audit = report.audit_run(directory)
    history, candidates = point_key(metadata)
    expected_keys = {(scheme, index) for scheme in report.SCHEMES for index in range(32)}
    keyed = {(row["scheme"], row["request_id"]): row for row in rows}
    report.require(
        len(rows) == len(keyed) == 128 and set(keyed) == expected_keys, "incomplete request matrix"
    )
    receipt_path = Path(metadata["correctness_receipt"]["path"]).resolve()
    receipt = read(receipt_path)
    check_dir = (receipt_path.parent / receipt["artifacts"]["metadata"]["path"]).parent.resolve()
    check_metadata, check_rows, check_audit = report.audit_run(check_dir)
    report.require(
        check_metadata["schema"] == CHECK_SCHEMA
        and check_metadata["config"] == metadata["config"]
        and check_metadata["validation_identity"] == metadata["validation_identity"],
        "check does not match the matrix benchmark",
    )
    checked = {(row["scheme"], row["request_id"]): row for row in check_rows}
    report.require(
        len(check_rows) == len(checked) == 128 and set(checked) == expected_keys,
        "incomplete check matrix",
    )
    comparison = {"policy": COUNTER_COMPARISON_POLICY, "differences": [], "proof": None}

    def prove_fixed_cap():
        if comparison["proof"] is None:
            comparison["proof"] = echo_counter_proof(directory, metadata, check_dir, check_metadata)
        return comparison["proof"]

    for key, row in keyed.items():
        reference = checked[key]
        comparison["differences"].extend(
            compare_cache_diagnostics(
                row["cache_diagnostics"],
                reference["cache_diagnostics"],
                row["scheme"],
                row["request_id"],
                prove_fixed_cap,
            )
        )
        for name in (
            "input_sha256",
            "is_revisit",
            "prefix_cache_hit",
            "prefix_hit_tier",
            "cached_users",
            "evicted_users",
        ):
            report.require(
                strict_equal(row[name], reference[name]), f"check/bench {name} differs at {key}"
            )
    memory, memory_samples = audit_memory(directory, metadata, rows)
    audit_memory(check_dir, check_metadata, check_rows)
    summary = report.summarize(rows)
    report.require(len(summary) == 8, "incomplete visit summary")
    for item in summary:
        expected_hits = (
            16
            if item["visit_kind"] == "revisit"
            and (item["scheme"] != "hbm" or 65536 // history >= 16)
            else 0
        )
        report.require(
            item["requests"] == 16 and item["prefix_hits"] == expected_hits,
            "visit count or history hits differ from fixed quotas",
        )
        item.update(
            history_tokens=history,
            candidate_tokens=candidates,
            run_id=metadata["run_id"],
            check_run_id=check_metadata["run_id"],
            receipt_sha256=metadata["correctness_receipt"]["sha256"],
            source_sha256=metadata["source_sha256"],
            trace_latency_sum_ms=sum(
                row["latency_ms"] for row in rows if row["scheme"] == item["scheme"]
            ),
            **memory[item["scheme"]],
        )
    source = {
        "history_tokens": history,
        "candidate_tokens": candidates,
        "run_id": metadata["run_id"],
        "directory": str(directory),
        "check_run_id": check_metadata["run_id"],
        "check_directory": str(check_dir),
        "receipt_path": str(receipt_path),
        "receipt_sha256": report.digest(receipt_path),
        "source_sha256": metadata["source_sha256"],
        "workload_sha256": metadata["workload_sha256"],
        "hardware": metadata["hardware"],
        "measurement_boundary": metadata["measurement_boundary"],
        "memory_samples": memory_samples,
        "input_sha256": {
            name: report.digest(directory / name)
            for name in (
                "metadata.json",
                "measurements.jsonl",
                "memory.json",
                "source_manifest.json",
            )
        },
        "bench_audit": bench_audit,
        "check_audit": check_audit,
        "cache_diagnostics_comparison": comparison,
    }
    return metadata, summary, source


def collect_matrix(run_dirs):
    directories = [Path(path).resolve() for path in run_dirs]
    report.require(
        len(directories) == 12 and len(set(directories)) == 12,
        "matrix requires 12 unique run directories",
    )
    indexed = {}
    run_ids = set()
    for directory in directories:
        metadata = read(directory / "metadata.json")
        key = point_key(metadata)
        report.require(key not in indexed, f"duplicate H/A matrix point: {key}")
        report.require(metadata["run_id"] not in run_ids, "duplicate benchmark run ID")
        indexed[key] = directory
        run_ids.add(metadata["run_id"])
    report.require(
        set(indexed) == {(h, a) for h in HISTORIES for a in CANDIDATES}, "incomplete H/A coverage"
    )
    common, summary, sources = None, [], []
    for key, directory in sorted(indexed.items()):
        metadata, point_summary, source = audit_point(directory)
        identity = common_identity(metadata, directory)
        if common is None:
            common = identity
        else:
            report.require(
                identity == common,
                f"mixed implementation, hardware, precision or input rules at {key}",
            )
        summary.extend(point_summary)
        sources.append(source)
    return {
        "schema": "deepseek-c10-fixed-capacity-matrix-v1",
        "status": "accepted",
        "histories": list(HISTORIES),
        "candidates": list(CANDIDATES),
        "fixed_config": FIXED_CONFIG,
        "common_identity": common,
        "points": sources,
        "summary": summary,
        "boundary": (
            "One sequential 32-request trace per scheme and H/A point; P95 describes its "
            "16 visits, not repeated-run uncertainty. Transfer counts are candidate payload, "
            "not PCIe bus traffic. PyTorch peaks include model/execution; device-used values "
            "are after-request samples, not continuous physical peaks. HBM retains 16, 4, "
            "or 1 histories at H=4096, 16384, or 65536; all offload histories fit NH. "
            "The synthetic C10 checkpoint workload does not establish real GR quality."
        ),
    }


def plot_matrix(summary, output):
    from experiments.deepseek_v32_motivation.src.plot import COLORS, LABELS, plt, save_figure

    for visit, field, title, name in (
        ("revisit", "latency_mean_ms", "Mean revisit latency (ms)", "revisit_latency"),
        ("first", "latency_mean_ms", "Mean first-visit latency (ms)", "first_visit_latency"),
        ("revisit", "trace_latency_sum_ms", "32-request trace latency (s)", "trace_latency"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        for axis, history in zip(axes, HISTORIES, strict=True):
            selected = [
                row
                for row in summary
                if row["history_tokens"] == history and row["visit_kind"] == visit
            ]
            for scheme in report.SCHEMES:
                group = sorted(
                    (row for row in selected if row["scheme"] == scheme),
                    key=lambda row: row["candidate_tokens"],
                )
                report.require(
                    [row["candidate_tokens"] for row in group] == list(CANDIDATES),
                    "incomplete plotted series",
                )
                values = [
                    row[field] / (1000 if field == "trace_latency_sum_ms" else 1) for row in group
                ]
                report.require(
                    all(math.isfinite(value) and value > 0 for value in values),
                    "invalid plotted latency",
                )
                axis.plot(
                    CANDIDATES, values, marker="o", color=COLORS[scheme], label=LABELS[scheme]
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(CANDIDATES, [str(value) for value in CANDIDATES])
            if visit == "revisit" and field == "latency_mean_ms":
                axis.set_yscale("log")
            else:
                axis.set_ylim(bottom=0)
            axis.set_title(f"H = {history:,}\nHBM quota: {65536 // history} histories")
            axis.set_xlabel("Candidate tokens A")
            axis.set_ylabel(title + ("; log scale" if axis.get_yscale() == "log" else ""))
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        figure.suptitle("DeepSeek C10: P=65,536, NH=16,777,216; 16 users, two rounds")
        figure.text(
            0.5,
            0.07,
            "One observation per request. Panel scales may differ.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.12, 1, 0.94))
        save_figure(figure, output, name)


def write_matrix(run_dirs, output_dir):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = collect_matrix(run_dirs)
    temporary = Path(tempfile.mkdtemp(prefix="deepseek-motivation-matrix-report-"))
    try:
        for point in result["points"]:
            destination = (
                temporary / "points" / f"h{point['history_tokens']}_a{point['candidate_tokens']}"
            )
            report.write_report(point["directory"], destination)
            point["report"] = str(destination.relative_to(temporary))
        plot_matrix(result["summary"], temporary)
        report.write_csv(temporary / "summary.csv", result["summary"])
        result["report_generator_sha256"] = report.digest(__file__)
        report.write_json(temporary / "summary.json", result)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        shutil.move(str(temporary), str(output))
    except BaseException as error:
        if temporary.exists():
            try:
                shutil.rmtree(temporary)
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve both failures
                raise BaseExceptionGroup(
                    "matrix reporting and cleanup failed", [error, cleanup_error]
                ) from None
        raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = write_matrix(args.run_dir, args.output_dir)
    print(
        f"verified matrix: {len(result['points'])} points, {len(result['summary'])} summary rows; {args.output_dir}"
    )


if __name__ == "__main__":
    main()
