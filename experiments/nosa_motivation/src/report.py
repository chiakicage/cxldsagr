"""Revalidate saved inputs, capacities and full hidden tensors before reporting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import tempfile
from pathlib import Path
from types import SimpleNamespace

from experiments.nosa_motivation.src.config import (
    BENCH_SCHEMA,
    CHECK_SCHEMA,
    METHODS,
    SCHEMA,
    configuration,
)
from experiments.nosa_motivation.src.cpu_environment import validate_cpu_environment_record
from experiments.nosa_motivation.src.measure import (
    RECEIPT_KIND,
    check_graph_replays,
    check_request,
    check_resource_plan,
    compare_hidden,
    effective_work,
    validate_warmup,
)
from experiments.nosa_motivation.src.provenance import (
    audit_native_identity,
    digest,
    publish_directories,
    verify_source_snapshot,
    write_json,
)
from experiments.nosa_motivation.src.validation import audit_receipt


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def graph_plan_metadata(state):
    return {
        key: value
        for key, value in state.items()
        if key not in ("project_replays", "finish_replays", "eager_fallbacks")
    }


def audit_shared_resources(case, config):
    """Join base cache storage with separately charged graph allocator capacity."""
    plan, backend = case["resource_plan"], case["backend"]
    base = backend["resource_plan"]
    require(
        base == {key: value for key, value in plan.items() if key != "compute_graphs"},
        "backend base and runner resource plans disagree",
    )
    graph = backend.get("compute_graphs", {"enabled": False})
    charge = 0
    if config["enable_compute_graphs"]:
        require(
            graph.get("enabled") is True
            and plan.get("compute_graphs") == graph_plan_metadata(graph),
            "compute graph plan differs from backend graph storage",
        )
        names = (
            "static_storage_bytes",
            "static_allocated_bytes",
            "private_reserved_bytes",
            "chosen_private_limit_bytes",
            "static_allocation_limit_bytes",
        )
        require(
            all(type(graph.get(name)) is int and graph[name] >= 0 for name in names),
            "invalid compute graph storage accounting",
        )
        require(
            graph["static_storage_bytes"]
            <= graph["static_allocated_bytes"]
            <= graph["static_allocation_limit_bytes"]
            and graph["private_reserved_bytes"] <= graph["chosen_private_limit_bytes"],
            "compute graph storage exceeds its allocation limits",
        )
        charge = graph["static_allocated_bytes"] + graph["private_reserved_bytes"]
    else:
        require(
            graph.get("enabled") is False and "compute_graphs" not in plan,
            "disabled compute graphs have graph storage in the resource plan",
        )
    shared, reservation = backend["shared_cache_bytes"], case["shared_reservation"]
    logical = base["logical_shared_hbm_bytes"]
    require(type(logical) is int and logical >= 0, "invalid base shared storage")
    require(
        shared == {"hbm": logical + charge, "dram": 0},
        "actual shared storage differs from base cache plus compute graphs",
    )
    require(
        set(reservation) == {"hbm", "dram"}
        and all(type(value) is int and value >= 0 for value in reservation.values())
        and reservation["hbm"] - charge >= logical
        and reservation["dram"] == 0,
        "shared storage exceeds base allocator plus compute graph reservation",
    )
    return {
        "base_logical_hbm_bytes": logical,
        "graph_allocator_charge_bytes": charge,
        "base_reserved_hbm_bytes_derived": reservation["hbm"] - charge,
    }


def audit_request_graph_storage(states, case, config):
    if config["enable_compute_graphs"]:
        expected = case["resource_plan"]["compute_graphs"]
        require(
            all(graph_plan_metadata(states[name]) == expected for name in ("before", "after")),
            "request compute graph storage differs from its resource plan",
        )


def audit_observations(directory, metadata, rows):
    """Check independent copies of allocator, device, plan and request observations."""
    samples = json.loads((directory / "memory.json").read_text())
    by_stage = {sample["stage"]: sample for sample in samples}
    require(len(by_stage) == len(samples), "duplicate memory observation stage")
    total = metadata["hardware"]["total_memory"]
    for sample in samples:
        values = {name: value for name, value in sample.items() if name != "stage"}
        require(
            all(type(value) is int and value >= 0 for value in values.values()),
            "invalid memory observation",
        )
        require(sample["device_total_bytes"] == total, "device memory capacity changed")
        require(
            sample["device_free_bytes"] + sample["device_used_bytes"] == total,
            "device used/free observations disagree",
        )
        require(
            sample["torch_allocated_bytes"]
            <= sample["torch_reserved_bytes"]
            <= sample["device_used_bytes"]
            <= total,
            "allocated/reserved/device-used observations disagree",
        )
        require(
            sample["torch_allocated_bytes"]
            <= sample["torch_peak_allocated_bytes"]
            <= sample["torch_peak_reserved_bytes"]
            <= total
            and sample["torch_reserved_bytes"] <= sample["torch_peak_reserved_bytes"],
            "allocator peak observations disagree",
        )
    require(
        metadata["model_loaded_memory"] == by_stage["after_model_loading"],
        "model-loaded memory differs from the observation log",
    )
    for case in metadata["cases"]:
        method = case["method"]
        backend = case["backend"]
        shared = backend["shared_cache_bytes"]
        audit_shared_resources(case, metadata["config"])
        group = [row for row in rows if row["method"] == method]
        for row in group:
            audit_request_graph_storage(row.get("compute_graphs", {}), case, metadata["config"])
            sample = by_stage[f"{method}/after_request_{row['request_id']}"]
            require(row["memory_after"] == sample, "request memory differs from observation log")
            for tier in ("hbm", "dram"):
                require(
                    row[f"shared_cache_{tier}_bytes"] == shared[tier]
                    and row[f"shared_reserved_{tier}_bytes"] == case["shared_reservation"][tier],
                    "request shared storage/reservation differs from case observations",
                )
        for name in ("torch_peak_allocated_bytes", "torch_peak_reserved_bytes"):
            require(
                case[name] == max(row["memory_after"][name] for row in group),
                "case memory peak differs from request observations",
            )
    return {"samples_checked": len(samples), "device_observations_are_peaks": False}


def audit_workload(directory, metadata):
    from GR.workload import token_sha256

    config = metadata["config"]
    manifest = json.loads((directory / "workload/workload.json").read_text())
    expected_config = {
        "model": "nosa",
        "num_users": config["num_users"],
        "requests": config["requests_per_method"],
        "history_tokens": config["history_tokens"],
        "candidate_tokens": config["candidate_tokens"],
        "sampling": "sequential",
        "seed": config["seed"],
        "context_limit": config["history_tokens"] + config["candidate_tokens"],
    }
    require(
        all(manifest["config"].get(key) == value for key, value in expected_config.items()),
        "workload configuration differs from measurement configuration",
    )
    requests = read_jsonl(directory / "workload/requests.jsonl")
    require(len(requests) == config["requests_per_method"], "incomplete workload")
    identity = {
        name: manifest[name] for name in ("config", "heat_sha256", "tokenizer_sha256", "requests")
    }
    import hashlib

    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    require(
        identity_hash == metadata["workload_sha256"] == manifest["workload_sha256"],
        "workload identity changed",
    )
    require(len(manifest["requests"]) == len(requests), "workload identity inventory is incomplete")
    users, prefixes, candidates = [], {}, {}
    for index, (request, identity) in enumerate(zip(requests, manifest["requests"], strict=True)):
        require(request["request_id"] == index, "workload order changed")
        require(
            all(request.get(key) == value for key, value in identity.items()),
            "workload identity inventory disagrees with request",
        )
        ids, user = request["input_ids"], request["user_id"]
        history, candidate = ids[: config["history_tokens"]], ids[config["history_tokens"] :]
        require(
            len(history) == config["history_tokens"]
            and len(candidate) == config["candidate_tokens"],
            "workload lengths changed",
        )
        for tokens, name in (
            (ids, "input_sha256"),
            (history, "prefix_sha256"),
            (candidate, "candidate_sha256"),
        ):
            require(token_sha256(tokens) == request[name], f"workload tokens changed: {name}")
        if index < config["num_users"]:
            users.append(user)
        require(user == users[index % config["num_users"]], "workload is not cyclic")
        require(prefixes.setdefault(user, history) == history, "user history changed on revisit")
        require(candidates.get(user) != candidate, "candidate must change between user visits")
        candidates[user] = candidate
    require(len(set(users)) == config["num_users"], "first round repeats users")
    return requests


def audit_run(directory):
    import torch

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    require(
        metadata.get("schema") in (SCHEMA, CHECK_SCHEMA, BENCH_SCHEMA), "unknown measurement schema"
    )
    checking = metadata["schema"] == CHECK_SCHEMA
    benchmark = metadata["schema"] == BENCH_SCHEMA
    receipt_audit = audit_receipt(metadata, directory, RECEIPT_KIND) if benchmark else None
    require(metadata.get("status") == "accepted", "run is incomplete or failed")
    validate_cpu_environment_record(metadata)
    config = metadata["config"]
    require(
        config
        == configuration(
            SimpleNamespace(
                **config, run_id=metadata["run_id"], compute_graphs=config["enable_compute_graphs"]
            )
        ),
        "configuration differs from the fixed-capacity contract",
    )
    require(metadata["model_config"]["num_hidden_layers"] == 32, "incomplete NOSA checkpoint")
    source_id = verify_source_snapshot(directory)
    require(source_id == metadata["source_sha256"], "source manifest identity changed")
    from evaluation.pool_scan_provenance import POOL_SCAN_ABI, requires_provenance

    saved_sources = json.loads((directory / "source_manifest.json").read_text())
    native_id = audit_native_identity(
        metadata,
        require_pool_referrers=requires_provenance(saved_sources),
        expected_abi_sha256=saved_sources.get(POOL_SCAN_ABI),
    )
    requests = audit_workload(directory, metadata)
    cases = metadata["cases"]
    require([case["method"] for case in cases] == list(METHODS), "incomplete method matrix")
    for case in cases:
        require(
            case["requests"] == config["requests_per_method"]
            and case["started_empty"]
            and case["warmup_requests"] == 3,
            "incomplete method or independent-cache initialization",
        )
        check_resource_plan(
            case["method"],
            SimpleNamespace(
                metadata=case["resource_plan"],
                hbm_tokens=case["admission_hbm_tokens"],
                host_pages=case["admission_host_pages"],
            ),
            config,
        )
        validate_warmup(case["method"], metadata["warmup_traces"][case["method"]], config, requests)
    rows = read_jsonl(directory / "measurements.jsonl")
    require(len(rows) == len(METHODS) * len(requests), "incomplete request matrix")
    keyed = {(row["method"], row["request_id"]): row for row in rows}
    require(len(keyed) == len(rows), "duplicate request measurement")
    observations = audit_observations(directory, metadata, rows)
    numerical_hashes, exact_count, largest_error = {}, 0, 0.0
    for request in requests:
        index, reference = request["request_id"], None
        for method in METHODS:
            row = keyed.get((method, index))
            require(row is not None, "missing method/request measurement")
            require(row["run_id"] == metadata["run_id"], "row run ID mismatch")
            require(row["workload_sha256"] == metadata["workload_sha256"], "row workload mismatch")
            check_request(request, row, config)
            graphs = row.get("compute_graphs", {})
            check_graph_replays(graphs.get("before"), graphs.get("after"), config, row)
            if not checking:
                stages = ("admission_ms", "prefix_ms", "extend_ms", "cleanup_ms")
                for name in (*stages, "latency_ms"):
                    require(math.isfinite(row[name]) and row[name] >= 0, f"invalid timing {name}")
                require(row["latency_ms"] > 0, "latency must be positive")
                require(
                    row["effective_work"] == effective_work(metadata["model_config"], config, row),
                    "matrix work or wall MFU arithmetic differs",
                )
                require(
                    math.isclose(
                        sum(row[name] for name in stages), row["latency_ms"], abs_tol=1e-5
                    ),
                    "request stages do not sum to wall latency",
                )
            if benchmark:
                require(
                    not {"numerical", "output_file", "output_sha256"}.intersection(row),
                    "bench must not contain per-request numerical output evidence",
                )
                continue
            relative = f"numerical/{method}/{index:06d}.pt"
            require(row["output_file"] == relative, "unexpected numerical output path")
            require(digest(directory / relative) == row["output_sha256"], "saved output changed")
            payload = torch.load(directory / relative, map_location="cpu", weights_only=True)
            require(payload["request_id"] == index, "output request identity changed")
            require(
                payload["input_sha256"] == request["input_sha256"], "output input identity changed"
            )
            require(
                tuple(payload["hidden"].shape)
                == (config["candidate_tokens"], metadata["model_config"]["hidden_size"]),
                "saved output does not contain every candidate hidden state",
            )
            if reference is None:
                reference = payload["hidden"]
            comparison = compare_hidden(payload["hidden"], reference, config)
            require(comparison == row["numerical"], "recorded numerical comparison differs")
            exact_count += comparison["exact"]
            largest_error = max(largest_error, comparison["max_abs"])
            numerical_hashes[relative] = row["output_sha256"]
    return (
        metadata,
        rows,
        {
            "passed": True,
            "checked_requests": len(rows),
            "offload_requests_compared_to_independent_hbm": 0 if benchmark else len(requests) * 3,
            "checked_warmup_requests": len(METHODS) * 3,
            "all_candidate_hidden_checked": not benchmark,
            **({"independent_correctness_receipt": receipt_audit} if benchmark else {}),
            "exact_requests": exact_count,
            "hidden_max_abs": largest_error,
            "numerical_atol": config["numerical_atol"],
            "numerical_rtol": config["numerical_rtol"],
            "source_sha256": source_id,
            "native_build_sha256": native_id,
            "observations": observations,
            "numerical_sha256": numerical_hashes,
            "scope": "latency, declared capacities, workload, and full hidden numerical validation",
            "does_not_establish": "matrix API MFU proximity, internal overlap, or compute/IO dominance",
        },
    )


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(rows):
    summary = []
    for method in METHODS:
        for revisit in (False, True):
            group = [
                row for row in rows if row["method"] == method and row["is_revisit"] == revisit
            ]
            require(group, f"missing first/revisit group for {method}")
            latencies = [row["latency_ms"] for row in group]
            summary.append(
                {
                    "method": method,
                    "visit_kind": "revisit" if revisit else "first",
                    "requests": len(group),
                    "prefix_hits": sum(row["prefix_cache_hit"] for row in group),
                    "evicted_users": sum(len(row["evicted_users"]) for row in group),
                    "latency_mean_ms": statistics.mean(latencies),
                    "latency_median_ms": statistics.median(latencies),
                    "latency_p95_ms": percentile(latencies, 0.95),
                    "latency_sum_ms": sum(latencies),
                    "request_effective_matrix_flops": sum(
                        sum(row["effective_work"]["request_flops"].values()) for row in group
                    ),
                    "request_wall_mfu_pct": 100
                    * sum(sum(row["effective_work"]["request_flops"].values()) for row in group)
                    / (sum(latencies) * group[0]["effective_work"]["peak_bf16_tflops"] * 1e9),
                    "candidate_wall_mfu_pct": 100
                    * sum(sum(row["effective_work"]["candidate_flops"].values()) for row in group)
                    / (
                        sum(row["extend_ms"] for row in group)
                        * group[0]["effective_work"]["peak_bf16_tflops"]
                        * 1e9
                    ),
                    **{
                        name.replace("_ms", "_mean_ms"): statistics.mean(row[name] for row in group)
                        for name in ("admission_ms", "prefix_ms", "extend_ms", "cleanup_ms")
                    },
                    **{
                        name: sum(row["cache_diagnostics"][name] for row in group)
                        for name in (
                            "prefix_host_to_device_bytes",
                            "prefix_device_to_host_bytes",
                            "candidate_host_to_device_bytes",
                            "candidate_device_to_host_bytes",
                        )
                    },
                    "max_cache_hbm_bytes": max(row["cache_hbm_bytes"] for row in group),
                    "max_cache_dram_bytes": max(row["cache_dram_bytes"] for row in group),
                }
            )
    baseline = {
        row["visit_kind"]: row["latency_mean_ms"] for row in summary if row["method"] == "hbm"
    }
    for row in summary:
        row["mean_latency_speedup_vs_hbm"] = baseline[row["visit_kind"]] / row["latency_mean_ms"]
    return summary


def async_speedup_gate(rows):
    """Use only uninstrumented synchronized request times for the speedup gate."""
    groups = {}
    for visit in (False, True):
        totals = {
            method: sum(
                row["latency_ms"]
                for row in rows
                if row["method"] == method and row["is_revisit"] == visit
            )
            for method in ("sync_sparse", "async_sparse")
        }
        if any(value <= 0 or not math.isfinite(value) for value in totals.values()):
            raise ValueError("async speedup requires both methods and both visit groups")
        groups["revisit" if visit else "first"] = {
            "sync_over_async": totals["sync_sparse"] / totals["async_sparse"],
            "passed": totals["async_sparse"] < totals["sync_sparse"],
            "total_request_ms": totals,
        }
    totals = {
        method: sum(group["total_request_ms"][method] for group in groups.values())
        for method in ("sync_sparse", "async_sparse")
    }
    return {
        "status": "passed" if totals["async_sparse"] < totals["sync_sparse"] else "failed",
        "groups": groups,
        "total_request_ms": totals,
        "sync_over_async": totals["sync_sparse"] / totals["async_sparse"],
        "rule": "strictly lower total uninstrumented request latency over the complete trace; visit groups reported separately",
        "boundary": "Observed trace comparison; no statistical-confidence or repeatability claim",
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(directory, destination):
    metadata, rows, audit = audit_run(directory)
    require(metadata["schema"] != CHECK_SCHEMA, "check runs do not publish performance reports")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(prefix="nosa-motivation-report-"))
    try:
        _write_report(directory, staging, metadata, rows, audit)
        publish_directories({"report": staging}, {"report": destination})
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return audit


def _write_report(directory, destination, metadata, rows, audit):
    summary = summarize(rows)
    write_json(destination / "summary.json", summary)
    write_json(destination / "audit.json", audit)
    write_csv(destination / "summary.csv", summary)
    gates = {
        "measurement_validity": "passed",
        "async_end_to_end_speedup": async_speedup_gate(rows),
        "async_every_sample_page_and_stripe_90pct": {
            "status": "pending",
            "reason": "separate profile required",
        },
        "raw_matrix_api_mfu_proximity": "pending",
        "compute_io_dominance": "pending",
    }
    write_json(destination / "acceptance.json", gates)
    fields = (
        "method",
        "request_id",
        "user_id",
        "visit_index",
        "is_revisit",
        "prefix_cache_hit",
        "latency_ms",
        "admission_ms",
        "prefix_ms",
        "extend_ms",
        "cleanup_ms",
    )
    write_csv(destination / "per_request.csv", [{key: row[key] for key in fields} for row in rows])
    text = [
        f"# NOSA fixed-capacity motivation: {metadata['run_id']}",
        "",
        metadata["model_boundary"],
        "",
        metadata["measurement_boundary"],
        "",
        "| Method | Visit | Requests | Prefix hits | Mean ms | P95 ms | Speedup vs HBM | Request MFU % | Candidate MFU % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    text.extend(
        f"| {row['method']} | {row['visit_kind']} | {row['requests']} | {row['prefix_hits']} | "
        f"{row['latency_mean_ms']:.3f} | {row['latency_p95_ms']:.3f} | "
        f"{row['mean_latency_speedup_vs_hbm']:.3f} | {row['request_wall_mfu_pct']:.3f} | "
        f"{row['candidate_wall_mfu_pct']:.3f} |"
        for row in summary
    )
    text.extend(
        [
            "",
            (
                f"Independent correctness receipt: `{audit['independent_correctness_receipt']['path']}`. "
                "This benchmark did not copy, compare or save full request outputs."
                if metadata["schema"] == BENCH_SCHEMA
                else f"All {audit['checked_requests']} candidate hidden tensors passed independent-cache "
                f"comparison (atol={audit['numerical_atol']}, rtol={audit['numerical_rtol']}); "
                f"{audit['exact_requests']} were exact. Maximum absolute error: "
                f"{audit['hidden_max_abs']:.9g}."
            ),
            "",
            (
                "Latency and cache admission alone do not prove MFU proximity, CPU launch "
                "efficiency, internal copy/compute overlap, or compute/IO dominance. "
                "Those claims require the separate complete API and timeline analysis."
            ),
            "",
            f"Source identity: `{metadata['source_sha256']}`.",
            f"Workload identity: `{metadata['workload_sha256']}`.",
            f"Native build/dependency identity: `{audit['native_build_sha256']}`.",
            (
                "Async versus synchronous sparse request-latency gate: "
                f"**{gates['async_end_to_end_speedup']['status']}**. "
                "Every-sample internal overlap, matrix API proximity and compute/IO "
                "dominance remain pending separate diagnostic evidence."
            ),
        ]
    )
    (destination / "results.md").write_text("\n".join(text) + "\n")
    write_json(
        destination / "report_provenance.json",
        {
            "run_id": metadata["run_id"],
            "source_sha256": metadata["source_sha256"],
            "workload_sha256": metadata["workload_sha256"],
            "metadata_sha256": digest(Path(directory) / "metadata.json"),
            "measurements_sha256": digest(Path(directory) / "measurements.jsonl"),
            "generated_by": "python -m experiments.nosa_motivation.src.report",
        },
    )


def main(argv=None):
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("data_dir", type=Path)
    command.add_argument("--output-dir", required=True, type=Path)
    args = command.parse_args(argv)
    print(json.dumps(write_report(args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
