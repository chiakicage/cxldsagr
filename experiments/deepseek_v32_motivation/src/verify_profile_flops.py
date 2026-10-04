"""Reconcile eager and graph calls with the independently expanded workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.deepseek_v32_motivation.src.analyze_pipeline import sha, write_json
from experiments.deepseek_v32_motivation.src.flops import phase_ledger


def canonical_operator(name):
    return {
        "indexer_prefetch": "indexer",
        "indexer_fused": "indexer",
        "indexer_qk": "indexer",
        "mla_qk_pv": "sparse_mla",
    }.get(name, name)


def verify_calls(calls, captures, work):
    """Check chunk/independent-layer/operator groups while allowing MLA splits."""
    captures_by_id = {row["capture_index"]: row for row in captures}
    if len(captures_by_id) != len(captures):
        raise ValueError("duplicate request capture identity")
    if len({(r["capture_index"], r["call_id"]) for r in calls}) != len(calls):
        raise ValueError("duplicate instrumented call identity")
    matrix = [row for row in calls if row.get("useful_flops") is not None]
    if not matrix:
        raise ValueError("profile contains no matrix work")
    observed = defaultdict(lambda: {"useful_flops": 0, "calls": 0})
    segments = defaultdict(set)
    for row in matrix:
        capture = captures_by_id.get(row["capture_index"])
        if capture is None or (row["scheme"], row["phase"]) != (
            capture["scheme"],
            capture["phase"],
        ):
            raise ValueError("matrix call has an unknown request capture")
        segment = row["segment"]
        if segment not in ("history", "candidate") or row["useful_flops"] < 0:
            raise ValueError("matrix work is outside serving compute segments")
        operator = canonical_operator(row.get("operator_work_name", row["stage"]))
        layer = -1 if operator == "lm_head" else int(row["layer"])
        key = (
            row["capture_index"],
            segment,
            int(row["chunk"]),
            layer,
            operator,
            row["precision"].lower(),
        )
        observed[key]["useful_flops"] += row["useful_flops"]
        observed[key]["calls"] += 1
        segments[row["capture_index"]].add(segment)
    precisions = {int(key): value for key, value in work["linear_precisions_by_source"].items()}
    expected = {}
    for capture in captures:
        index = capture["capture_index"]
        metrics, config = capture.get("runner_metrics"), work["config"]
        history = (
            (not metrics["prefix_cache_hit"])
            if metrics is not None
            else (
                capture["phase"] == "cold"
                or (
                    capture["scheme"] == "hbm"
                    and config["sparse_pool_tokens"] // config["padded_history_tokens"]
                    < config["num_users"]
                )
            )
        )
        required = {"candidate", "history"} if history else {"candidate"}
        if segments[index] != required:
            raise ValueError("captured segments differ from the admission trajectory")
        for segment in required:
            ledger = phase_ledger(
                work["model_config"], config, precisions, scheme=capture["scheme"], phase=segment
            )
            for item in ledger:
                if item["useful_flops"] is None:
                    continue
                layers = (
                    [-1]
                    if item["source_layer"] == -1
                    else [
                        layer
                        for layer in range(config["layers"])
                        if layer % 3 == item["source_layer"]
                    ]
                )
                if len(layers) != item["layer_copies"] or item["useful_flops"] % len(layers):
                    raise ValueError("planned independent-layer multiplicity is inconsistent")
                for layer in layers:
                    key = (
                        index,
                        segment,
                        item["chunk"],
                        layer,
                        canonical_operator(item["name"]),
                        item["precision"].lower(),
                    )
                    if key in expected:
                        raise ValueError("planned work has duplicate operator groups")
                    expected[key] = {
                        "useful_flops": item["useful_flops"] // len(layers),
                        "calls": 1,
                    }
    if observed.keys() != expected.keys():
        raise ValueError("captured operator groups differ from the planned workload")
    mismatches = []
    for key, planned in expected.items():
        actual = observed[key]
        # Exact sparse-union pressure may split MLA, conserving useful work.
        call_count_ok = (
            actual["calls"] >= planned["calls"]
            if key[4] == "sparse_mla"
            else actual["calls"] == planned["calls"]
        )
        if actual["useful_flops"] != planned["useful_flops"] or not call_count_ok:
            mismatches.append({"identity": key, "expected": planned, "actual": actual})
    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "matrix_calls": len(matrix),
        "matrix_calls_by_precision": dict(Counter(row["precision"] for row in matrix)),
        "graph_matrix_calls": sum(bool(row.get("graph_api")) for row in matrix),
        "verified_planned_chunk_layer_operator_groups": len(expected),
    }


def verify(profile, flops, output):
    metadata = json.loads((profile / "metadata.json").read_text())
    result = json.loads((profile / "result.json").read_text())
    work = json.loads(flops.read_text())
    if metadata["status"] != "accepted" or not result["accepted"]:
        raise ValueError("FLOP verification requires an accepted numerical profile")
    if (
        work["source_run_id"] != metadata["reference_run_id"]
        or work["config"] != metadata["config"]
    ):
        raise ValueError("profile and planned FLOPs refer to different workloads")
    if metadata["precision_settings"]["cuda_matmul_allow_tf32"] is not False:
        raise ValueError("FP32 work requires the recorded no-TF32 policy")
    manifest_path = profile / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if digest != metadata["source_sha256"] or digest != result["source_sha256"]:
        raise ValueError("profile source identity differs from its manifest")
    changed = [name for name, value in manifest.items() if sha(profile / "source" / name) != value]
    if changed:
        raise ValueError(f"profile source snapshot changed: {changed}")
    calls_path = profile / "operator_calls.json"
    verified = verify_calls(json.loads(calls_path.read_text()), metadata["captures"], work)
    verified.update(
        schema="deepseek-v32-motivation-profile-flops-verification-v2",
        profile_run_id=metadata["run_id"],
        reference_run_id=metadata["reference_run_id"],
        input_sha256={
            str(path): sha(path)
            for path in (
                profile / "metadata.json",
                profile / "result.json",
                manifest_path,
                calls_path,
                flops,
            )
        },
        analysis_source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        boundaries=[
            "Useful work and API counts are checked by chunk, independent layer, operator and precision against an expanded workload.",
            "Graph templates and eager calls share one ledger. Physical nodes and GPU timing require the separate Nsight clone-lineage audit.",
            "Sparse MLA can split under exact-union pressure. Its useful FLOPs must be conserved while API count can increase.",
            "This is a shape/formula cross-check, not a hardware instruction counter or a numerical rerun.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, verified)
    if not verified["passed"]:
        raise ValueError(f"matrix work verification failed: {verified['mismatches'][:3]}")
    return verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--flops", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.profile_run / "analysis/flops_verification.json"
    verified = verify(args.profile_run, args.flops, output)
    print(f"Verified {verified['matrix_calls']} actual matrix calls: {output}")


if __name__ == "__main__":
    main()
