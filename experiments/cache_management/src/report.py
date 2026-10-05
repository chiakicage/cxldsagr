"""Join verified NOSA/DeepSeek allocation plans and complete-request observations.

Static plans never acquire a physical-fit verdict from unrelated motivation
traces. The two models retain their own source, workload and accounting scopes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_nosa_plan(directory):
    from cache.capacity import AllocationSpec
    from experiments.cache_management.src.nosa_plan import _summarize

    directory = Path(directory)
    value = read_json(directory / "plan.json")
    require(
        value.get("schema") == "nosa-offline-allocation-plan-v1"
        and value.get("status") == "static_plan_not_execution",
        "Expected an offline NOSA allocation plan",
    )
    require(bool(value["sources"]), "NOSA source inventory is empty")
    for name, expected in value["sources"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid source path")
        require(digest(directory / "source" / relative) == expected, f"Source differs: {name}")
    require(
        digest(directory / "checkpoint_config.json") == value["config_source"]["sha256"],
        "NOSA checkpoint configuration differs",
    )
    require(
        not any(
            value["assumptions"][name]
            for name in (
                "live_hardware_validated",
                "live_allocator_validated",
                "model_weights_loaded",
                "cuda_initialized_by_planner",
            )
        ),
        "Offline NOSA planning must not claim execution validation",
    )
    require(
        set(value["cases"]) == {"hbm", "dense_prefetch", "serial_sparse", "overlap"},
        "NOSA plan must contain all four supported schemes",
    )
    for scheme, case in value["cases"].items():
        for scope in ("shared", "one_session"):
            allocations = tuple(
                AllocationSpec(**{**item, "shape": tuple(item["shape"])})
                for item in case[scope]["allocations"]
            )
            recomputed = _summarize(allocations)
            for field in (
                "storage_payload_bound",
                "allocator_reservation_bound",
                "categories",
            ):
                require(recomputed[field] == case[scope][field], f"NOSA {scope} {field} differs")
        for population in case["populations"].values():
            count = population["sessions"]
            require(type(count) is int and count >= 0, "Invalid session population")
            require(population["physical_fit_validated"] is False, "Static population claims fit")
            require(
                population["admissible_under_token_quota"]
                == (count <= case["quota"]["full_history_sessions"]),
                f"NOSA {scheme} quota verdict differs",
            )
            for field in ("storage_payload_bound", "allocator_reservation_bound"):
                expected = {
                    tier: case["shared"][field][tier] + count * case["one_session"][field][tier]
                    for tier in ("hbm", "dram")
                }
                require(
                    population[field + "_excluding_graphs"] == expected,
                    f"NOSA {scheme} population accounting differs",
                )
    return value


def memory_rows(model, metadata, rows):
    """Keep allocator peaks, boundary samples and cache reservations distinct."""
    result = []
    for scheme in dict.fromkeys(row["scheme"] for row in rows):
        selected = [row for row in rows if row["scheme"] == scheme]
        samples = [row["memory_after"] for row in selected]

        def device_used(sample):
            if model == "nosa":
                return sample["device_used_bytes"]
            return sample["cuda_total_bytes"] - sample["cuda_free_bytes"]

        item = {
            "model": model,
            "scheme": scheme,
            "run_id": metadata["run_id"],
            "requests": len(selected),
            "max_retained_users": max(row["cached_users"] for row in selected),
            "revisit_requests": sum(row["is_revisit"] for row in selected),
            "revisit_history_hits": sum(
                row["is_revisit"] and row["prefix_cache_hit"] for row in selected
            ),
            "torch_peak_allocated_bytes": max(s["torch_peak_allocated_bytes"] for s in samples),
            "torch_peak_reserved_bytes": max(s["torch_peak_reserved_bytes"] for s in samples),
            "maximum_after_request_device_used_bytes": max(map(device_used, samples)),
            "maximum_after_request_reserved_minus_allocated_bytes": max(
                s["torch_reserved_bytes"] - s["torch_allocated_bytes"] for s in samples
            ),
            "maximum_after_request_device_used_minus_reserved_bytes": max(
                device_used(s) - s["torch_reserved_bytes"] for s in samples
            ),
        }
        for field in (
            "cache_hbm_bytes",
            "cache_dram_bytes",
            "reserved_hbm_bytes",
            "reserved_dram_bytes",
            "shared_cache_hbm_bytes",
            "shared_cache_dram_bytes",
            "session_reserved_hbm_bytes",
            "session_reserved_dram_bytes",
            "cache_host_pages",
            "cache_hbm_tokens",
        ):
            item["maximum_" + field] = max(row[field] for row in selected)
        result.append(item)
    return result


def read_motivation(model, directory, receipt_override=None):
    if model == "nosa":
        from experiments.nosa_motivation.src.report import audit_run
    else:
        from experiments.deepseek_v32_motivation.src.report import audit_run

    directory = Path(directory)
    options = {} if receipt_override is None else {"receipt_override": receipt_override}
    metadata, rows, audit = audit_run(directory, **options)
    require(metadata.get("mode") == "bench", "Cache observations require a clean accepted bench")
    require(
        metadata["status"] == "accepted" and metadata["config"]["num_users"] >= 2,
        "Expected an accepted multi-user trace",
    )
    return {
        "run_id": metadata["run_id"],
        "directory": str(directory.resolve()),
        "config": metadata["config"],
        "checkpoint": metadata["checkpoint"],
        "model_config": metadata.get("model_config"),
        "hardware": metadata["hardware"],
        "source_sha256": metadata["source_sha256"],
        "source_manifest": read_json(directory / "source_manifest.json"),
        "workload_sha256": metadata["workload_sha256"],
        "memory_boundary": metadata["memory_boundary"],
        "measurement_boundary": metadata["measurement_boundary"],
        "model_boundary": metadata["model_boundary"],
        "model_loaded_memory": metadata["model_loaded_memory"],
        "ordinary_activation_peak_bytes": None,
        "activation_boundary": (
            "Ordinary activation peaks were not isolated from cache/workspace execution. "
            "Do not infer them by subtracting reservations from process memory peaks."
        ),
        "cases": metadata["cases"],
        "audit": audit,
        "artifacts": {
            name: digest(directory / name)
            for name in (
                "metadata.json",
                "measurements.jsonl",
                "memory.json",
                "source_manifest.json",
            )
        },
        "memory": memory_rows(model, metadata, rows),
        "capacity_sweep_performed": False,
    }


def validate_nosa_observation(plan, trace):
    """Only describe graph accounting as matched when layout and source match."""
    fields = {
        "history_tokens": "history_tokens",
        "candidate_tokens": "candidate_tokens",
        "chunk_size": "chunk_size",
        "sparse_pool_tokens": "sparse_pool_tokens",
        "host_arena_tokens": "host_arena_tokens",
        "requested_users": "num_users",
    }
    for static, measured in fields.items():
        require(
            plan["parameters"][static] == trace["config"][measured],
            f"NOSA static/observed {static} differs",
        )
    require(plan["model_config"] == trace["model_config"], "NOSA model configuration differs")
    require(
        plan["config_source"]["sha256"] == trace["checkpoint"]["metadata_sha256"]["config.json"],
        "NOSA checkpoint configuration source differs",
    )
    for name, expected in plan["sources"].items():
        if name != "experiments/cache_management/src/nosa_plan.py":
            require(
                trace["source_manifest"].get(name) == expected,
                f"NOSA planning/execution source differs: {name}",
            )
    require(
        {case["scheme"] for case in trace["cases"]} == set(plan["cases"]),
        "NOSA static/observed schemes differ",
    )


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(result):
    lines = [
        f"# Cache management evidence: `{result['run_id']}`",
        "",
        (
            "Static allocations and complete motivation traces have separate provenance. "
            "No static maximum below is a measured physical-capacity limit."
        ),
        "",
        "## DeepSeek static P/NH plans",
        "",
        "| Plan | P | NH | Users | Extra headroom GiB | HBM reservation GiB | DRAM reservation GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for plan in result["deepseek_static"]["plans"]:
        lines.append(
            f"| `{plan['run_id']}` | {plan['P']:,} | {plan['NH']:,} | {plan['U']} | "
            f"{plan['noncache_headroom_bytes'] / 2**30:g} | "
            f"{plan['hbm_total_bytes'] / 2**30:.6f} | {plan['dram_total_bytes'] / 2**30:.6f} |"
        )
    lines += [
        "",
        "## NOSA static allocation declarations",
        "",
        (
            "Graph allocations are excluded here. Shared graph storage and private reservation "
            "are already included in the matched request observations; do not add them again."
        ),
        "",
        (
            "| Scheme | Quota users | Shared HBM bound GiB | One-session HBM bound GiB | "
            "One-session host payload GiB | One-session host reservation GiB |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scheme, case in result["nosa_static"]["cases"].items():
        shared, session = case["shared"], case["one_session"]
        lines.append(
            f"| {scheme} | {case['quota']['full_history_sessions']} | "
            f"{shared['allocator_reservation_bound']['hbm'] / 2**30:.6f} | "
            f"{session['allocator_reservation_bound']['hbm'] / 2**30:.6f} | "
            f"{session['storage_payload_bound']['dram'] / 2**30:.6f} | "
            f"{session['allocator_reservation_bound']['dram'] / 2**30:.6f} |"
        )
    lines += [
        "",
        "## Complete-request memory observations",
        "",
        (
            "Allocator peaks include model weights and ordinary activations. Device-used values "
            "are maxima of after-request samples, not continuous process peaks. Cache charges "
            "and reservations are reported independently in memory.csv; charges include graph "
            "private reservations and are not a tensor-payload sum. These traces do not fill "
            "the offload NH quota and do not validate any offline maximum above."
        ),
        "",
        (
            "Memory after model loading is recorded separately in summary.json. It is not "
            "a tensor-only weight inventory. Ordinary activation peaks were not measured "
            "independently and are null; subtracting cache reservations from process peaks "
            "does not establish them."
        ),
        "",
        (
            "| Model | Scheme | Retained users | Allocated peak GiB | Reserved peak GiB | "
            "Max sampled device used GiB | Cache HBM GiB | Cache DRAM GiB |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("nosa", "deepseek"):
        trace = result["observed"][model]
        for row in trace["memory"]:
            values = [
                row[name] / 2**30
                for name in (
                    "torch_peak_allocated_bytes",
                    "torch_peak_reserved_bytes",
                    "maximum_after_request_device_used_bytes",
                    "maximum_cache_hbm_bytes",
                    "maximum_cache_dram_bytes",
                )
            ]
            lines.append(
                f"| {model} | {row['scheme']} | {row['max_retained_users']} | "
                + " | ".join(f"{value:.6f}" for value in values)
                + " |"
            )
    for model in ("nosa", "deepseek"):
        trace = result["observed"][model]
        lines += ["", f"{model} observation source: `{trace['run_id']}`."]
    lines += [
        "",
        (
            "Both traces use fixed token quotas, not equal HBM/DRAM byte budgets. NOSA's NH "
            "is a lazy per-session host quota; DeepSeek allocates a global arena. The workloads "
            "also differ: full 32-layer NOSA hidden outputs versus a ten-block DeepSeek source-input "
            "replay with a last-token LM head. The table does not rank cross-model efficiency."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--nosa-plan", type=Path, required=True)
    parser.add_argument("--deepseek-plans", type=Path, nargs="+", required=True)
    parser.add_argument("--nosa-run", type=Path, required=True)
    parser.add_argument("--deepseek-run", type=Path, required=True)
    parser.add_argument("--nosa-receipt", type=Path)
    parser.add_argument("--deepseek-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from experiments.cache_management.src.capacity_report import read_plan

    plans, provenance = zip(*(read_plan(path, []) for path in args.deepseek_plans), strict=True)
    require(len({item["run_id"] for item in plans}) == len(plans), "Duplicate DeepSeek plan")
    result = {
        "schema": "unified-cache-management-report-v1",
        "run_id": args.run_id,
        "nosa_static": read_nosa_plan(args.nosa_plan),
        "deepseek_static": {"plans": plans, "provenance": provenance, "complete_capacity_runs": 0},
        "observed": {
            "nosa": read_motivation("nosa", args.nosa_run, args.nosa_receipt),
            "deepseek": read_motivation("deepseek", args.deepseek_run, args.deepseek_receipt),
        },
        "generator_sha256": digest(Path(__file__)),
    }
    validate_nosa_observation(result["nosa_static"], result["observed"]["nosa"])
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    write_csv(args.output_dir / "deepseek_plans.csv", plans)
    write_csv(
        args.output_dir / "memory.csv",
        [row for trace in result["observed"].values() for row in trace["memory"]],
    )
    (args.output_dir / "results.md").write_text(markdown(result))
    from evaluation.provenance import snapshot_report_helpers

    snapshot_report_helpers(args.output_dir)
    print(f"Verified unified cache report: {args.output_dir}")


if __name__ == "__main__":
    main()
