"""Reopen diagnostic evidence and keep validity, overlap and speedup independent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.nosa_motivation.src.config import CHECK_SCHEMA, METHODS
from experiments.nosa_motivation.src.cpu_environment import (
    require_matching_cpu_environment,
    validate_cpu_environment_record,
)
from experiments.nosa_motivation.src.flops import observe_selection
from experiments.nosa_motivation.src.matrix_comparison import (
    audit_matrix_native_provenance,
    audit_matrix_reference,
    compare_request_apis,
)
from experiments.nosa_motivation.src.measure import (
    check_graph_replays,
    check_request,
    check_resource_plan,
    compare_hidden,
)
from experiments.nosa_motivation.src.profile_hbm import analyze_trace as analyze_api_trace
from experiments.nosa_motivation.src.provenance import (
    audit_native_identity,
    digest,
    verify_source_snapshot,
)
from experiments.nosa_motivation.src.report import (
    async_speedup_gate,
    audit_request_graph_storage,
    audit_run,
    audit_shared_resources,
    require,
)
from experiments.nosa_motivation.src.timeline import analyze_trace
from experiments.nosa_motivation.src.work_intervals import expected_misses
from experiments.nosa_offload_overlap.src.analyze import (
    stripe_interval_metrics,
    work_interval_metrics,
)


def evidence_path(directory, filename):
    if not filename or Path(filename).name != filename:
        raise ValueError("diagnostic evidence must use a plain filename")
    return Path(directory) / filename


def recompute_layer(data, layer, config, method):
    import torch

    from models.attention_contracts import BlockSelection

    path = evidence_path(data, layer["selection_file"])
    require(digest(path) == layer["selection_sha256"], "saved selection evidence changed")
    evidence = torch.load(path, map_location="cpu", weights_only=True)
    ids, valid = evidence["block_ids"], evidence["valid_mask"]
    require(
        tuple(ids.shape) == (config["candidate_tokens"], 2, 64)
        and valid.shape == ids.shape
        and evidence["query_heads"] == 32
        and evidence["head_dim"] == 128
        and evidence["element_size"] == 2,
        "selection evidence differs from complete checkpoint geometry",
    )
    expected = expected_misses(
        ids, valid, config["history_tokens"], evidence["tags"], evidence["owner"], 128, 2
    )
    geometry = {
        "prefix": config["history_tokens"],
        "queries": config["candidate_tokens"],
        "kv_heads": 2,
        "expected_fetch_rows": expected,
        "expected_prefix_bytes": sum(row["bytes"] for row in expected),
    }
    require(layer["geometry"] == geometry, "recorded misses differ from selected pages and owners")
    observation = observe_selection(
        BlockSelection(ids, block_size=64, valid_mask=valid),
        query_start=config["history_tokens"],
        query_heads=32,
    )
    require(
        layer["selection_observation"] == observation, "recorded causal-work observation differs"
    )
    require(
        layer["mode"] == ("overlap" if method == "async_sparse" else "serialized"),
        "internal diagnostic execution mode differs",
    )
    require(layer["fetch_stripes"] == 8, "unexpected stripe geometry")
    page = work_interval_metrics(geometry, layer)
    stripe = stripe_interval_metrics(geometry, layer, 8) if expected else None
    if not expected:
        require(
            not layer["stripe_intervals"]
            and not any(row["kind"] == 1 for row in layer["intervals"]),
            "no-fetch layer emitted copy intervals",
        )
    require(page == layer["page_metrics"], "stored page metrics differ from intervals")
    require(stripe == layer["stripe_metrics"], "stored stripe metrics differ from intervals")
    passed = (
        page["fetch_math_overlap_fraction"] >= 0.9
        and stripe["fetch_stripe_math_overlap_fraction"] >= 0.9
        if expected and method == "async_sparse"
        else None
    )
    require(layer["page_and_stripe_90pct"] is passed, "stored overlap verdict differs")
    return passed


def overlap_gate(records, config, repeats):
    """A single below-threshold layer/sample fails, regardless of any median."""
    selected = (0, config["num_users"])
    required = {(request, sample) for request in selected for sample in range(repeats)}
    work = [
        row for row in records if row["method"] == "async_sparse" and row["mode"] == "internal_work"
    ]
    present = {(row["request_id"], row["sample"]) for row in work}
    if present != required or len(work) != len(required):
        return {
            "status": "pending",
            "reason": "complete asynchronous first/revisit samples required",
        }
    checks, no_fetch = [], 0
    for record in work:
        if [layer["layer"] for layer in record["layers"]] != list(range(config["layers"])):
            return {"status": "pending", "reason": "every candidate layer requires evidence"}
        for layer in record["layers"]:
            if layer["page_and_stripe_90pct"] is None:
                no_fetch += 1
                continue
            checks.append(
                {
                    "request_id": record["request_id"],
                    "sample": record["sample"],
                    "layer": layer["layer"],
                    "page_ratio": layer["page_metrics"]["fetch_math_overlap_fraction"],
                    "stripe_ratio": layer["stripe_metrics"]["fetch_stripe_math_overlap_fraction"],
                }
            )
    failed = [row for row in checks if min(row["page_ratio"], row["stripe_ratio"]) < 0.9]
    return {
        "status": "failed" if failed else "passed" if checks else "pending",
        "rule": "both page-envelope and nonempty-stripe-copy ratios >= 0.9 for every applicable layer/sample",
        "checked_layer_samples": len(checks),
        "no_fetch_layer_samples": no_fetch,
        "failed_layer_samples": failed,
        "minimum_page_ratio": min((row["page_ratio"] for row in checks), default=None),
        "minimum_stripe_ratio": min((row["stripe_ratio"] for row in checks), default=None),
        "boundary": "Actual device copy intervals intersect softmax-only work; separate intrusive replay",
    }


def audit_profile(data, profiles, reference_dir):
    from experiments.nosa_motivation.src.config import BENCH_SCHEMA
    from experiments.nosa_motivation.src.measure import RECEIPT_KIND
    from experiments.nosa_motivation.src.validation import reference_directory

    reference_dir = reference_directory(reference_dir, bench_schema=BENCH_SCHEMA, kind=RECEIPT_KIND)
    import torch

    data, profiles = Path(data), Path(profiles)
    metadata = json.loads((data / "metadata.json").read_text())
    require(metadata.get("schema") == "nosa-motivation-profile-v2", "unknown diagnostic schema")
    require(metadata["status"] in ("validating", "diagnostic_valid"), "profile is incomplete")
    validate_cpu_environment_record(metadata)
    require(
        type(metadata["repeats"]) is int
        and metadata["repeats"] > 0
        and metadata["methods"]
        and len(set(metadata["methods"])) == len(metadata["methods"])
        and all(method in METHODS for method in metadata["methods"]),
        "diagnostic requires declared methods and positive sample coverage",
    )
    reference, reference_rows, _ = audit_run(reference_dir)
    config = reference["config"]
    require(metadata["config"] == config, "profile configuration differs from reference")
    require(metadata["reference_run_id"] == reference["run_id"], "profile reference run differs")
    require(metadata["workload_sha256"] == reference["workload_sha256"], "profile workload differs")
    for name in ("checkpoint", "precision_settings"):
        require(metadata[name] == reference[name], f"profile {name} differs from reference")
    require_matching_cpu_environment(metadata["hardware"], reference["hardware"])
    for name in (
        "uuid",
        "packages",
        "torch_cuda",
        "compute_capability",
        "total_memory",
        "sm_count",
    ):
        require(
            metadata["hardware"][name] == reference["hardware"][name],
            f"profile hardware/runtime {name} differs",
        )
    require(
        verify_source_snapshot(data) == metadata["source_sha256"], "profile source identity differs"
    )
    from experiments.nosa_motivation.src.profile import runtime_sources

    require(
        runtime_sources(json.loads((data / "source_manifest.json").read_text()))
        == runtime_sources(json.loads((Path(reference_dir) / "source_manifest.json").read_text())),
        "profile runtime/orchestration source differs from reference",
    )
    from evaluation.pool_scan_provenance import requires_provenance, source_abi_sha256

    saved_sources = json.loads((data / "source_manifest.json").read_text())
    native_id = audit_native_identity(
        metadata,
        require_pool_referrers=requires_provenance(saved_sources),
        expected_abi_sha256=source_abi_sha256(saved_sources),
    )
    require(
        native_id == reference["native_provenance"]["build_after"]["sha256"],
        "profile native build differs from reference",
    )
    reference_artifacts = reference["native_provenance"]["artifacts_final"]
    require(
        all(
            reference_artifacts.get(path) == value
            for path, value in metadata["native_provenance"]["artifacts_final"].items()
        ),
        "profile loaded binaries differ from reference",
    )
    matrix_reference = metadata["matrix_reference"]
    raw_path = evidence_path(data, matrix_reference["file"])
    require(digest(raw_path) == matrix_reference["sha256"], "raw matrix reference changed")
    raw_reference = json.loads(raw_path.read_text())
    raw_summary = audit_matrix_reference(raw_reference, reference["model_config"], config)
    raw_native = audit_matrix_native_provenance(
        matrix_reference["native_provenance"], metadata["native_provenance"]
    )
    keyed = {(row["method"], row["request_id"]): row for row in reference_rows}
    records = metadata["records"]
    expected = {
        (method, request, sample, mode)
        for method in metadata["methods"]
        for request in (0, config["num_users"])
        for sample in range(metadata["repeats"])
        for mode in (
            ("timeline", "internal_work")
            if method in ("sync_sparse", "async_sparse")
            else ("timeline",)
        )
    }
    present = {(row["method"], row["request_id"], row["sample"], row["mode"]) for row in records}
    require(
        present == expected and len(records) == len(expected), "incomplete diagnostic replay matrix"
    )
    expected_cases = {(method, sample, mode) for method, _, sample, mode in expected}
    cases = {(case["method"], case["sample"], case["mode"]): case for case in metadata["cases"]}
    require(
        set(cases) == expected_cases and len(cases) == len(metadata["cases"]),
        "incomplete replay provenance cases",
    )
    for case in cases.values():
        plan = case["resource_plan"]
        check_resource_plan(
            case["method"],
            SimpleNamespace(
                metadata=plan,
                hbm_tokens=case["admission_hbm_tokens"],
                host_pages=case["admission_host_pages"],
            ),
            config,
        )
        audit_shared_resources(case, config)
    api_comparisons = []
    for record in records:
        require(
            json.loads(evidence_path(data, record["record_file"]).read_text()) == record,
            "profile record differs from metadata inventory",
        )
        method, request_id = record["method"], record["request_id"]
        row = keyed[(method, request_id)]
        metrics = {**record["instrumented_runner_metrics"], "method": method}
        check_request(row, metrics, config)
        require(
            metrics["cache_diagnostics"] == row["cache_diagnostics"],
            "replay transfer counters differ from reference",
        )
        graphs = record["compute_graphs"]
        check_graph_replays(graphs["before"], graphs["after"], config, metrics)
        case = cases[(method, record["sample"], record["mode"])]
        audit_request_graph_storage(graphs, case, config)
        for tier in ("hbm", "dram"):
            require(
                metrics[f"shared_cache_{tier}_bytes"] == case["backend"]["shared_cache_bytes"][tier]
                and metrics[f"shared_reserved_{tier}_bytes"] == case["shared_reservation"][tier],
                "profile shared storage observations differ",
            )
        output = evidence_path(data, record["output_file"])
        require(digest(output) == record["output_sha256"], "profile hidden evidence changed")
        hidden = torch.load(output, map_location="cpu", weights_only=True)
        original = torch.load(
            Path(reference_dir) / row["output_file"], map_location="cpu", weights_only=True
        )
        comparison = compare_hidden(hidden, original["hidden"], config)
        require(
            comparison == record["numerical"] and comparison["exact"],
            "profile numerical evidence differs",
        )
        if record["mode"] == "timeline":
            path = evidence_path(profiles, record["trace_file"])
            require(digest(path) == record["trace_sha256"], "profile timeline changed")
            require(
                analyze_trace(json.loads(path.read_text()), record["root_name"])
                == record["analysis"],
                "stored timeline analysis differs",
            )
            api_analysis = analyze_api_trace(path)
            require(api_analysis == record["api_analysis"], "stored compute API analysis differs")
            api_comparisons.append(
                {
                    "sample": record["sample"],
                    **compare_request_apis(
                        row, reference["model_config"], config, api_analysis, raw_reference
                    ),
                }
            )
        else:
            require(
                record["extra_shared_trace_bytes"] == case["resource_plan"]["trace_capacity"] * 32,
                "trace buffer accounting differs from resource plan",
            )
            require(
                [layer["layer"] for layer in record["layers"]] == list(range(config["layers"])),
                "incomplete layer evidence",
            )
            for layer in record["layers"]:
                recompute_layer(data, layer, config, method)
            require(
                sum(layer["recorded_transfer_bytes"] for layer in record["layers"])
                == metrics["cache_diagnostics"]["candidate_main_kv_host_to_device_bytes"],
                "layer transfers disagree with candidate transfer total",
            )
    gates = {
        "diagnostic_validity": "passed",
        "async_every_sample_page_and_stripe_90pct": overlap_gate(
            records, config, metadata["repeats"]
        ),
        "async_end_to_end_speedup": (
            {
                "status": "pending",
                "reason": "independent check publishes no wall timings; run clean bench",
            }
            if reference["schema"] == CHECK_SCHEMA
            else async_speedup_gate(reference_rows)
        ),
        "raw_matrix_api_mfu_proximity": "pending",
        "compute_io_dominance": "pending interpretation of complete API and timeline evidence",
        "formal_performance_acceptance": "pending",
        "matrix_api_evidence": {
            "raw_reference": raw_summary,
            "native_provenance": raw_native,
            "request_comparisons": api_comparisons,
        },
    }
    if metadata["status"] == "diagnostic_valid":
        require(
            metadata["acceptance"] == gates,
            "saved diagnostic acceptance differs from recomputed evidence",
        )
    return gates


def main(argv=None):
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("data_dir", type=Path)
    command.add_argument("--profile-dir", required=True, type=Path)
    command.add_argument("--reference-run", required=True, type=Path)
    args = command.parse_args(argv)
    print(json.dumps(audit_profile(args.data_dir, args.profile_dir, args.reference_run), indent=2))


if __name__ == "__main__":
    main()
