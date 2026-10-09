"""Reuse the matched audit for the distinct formal-prelude driver and receipt."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from evaluation.validation import require_receipt
from experiments.deepseek_v32_mfu.src import analyze_replay_matched as matched
from experiments.deepseek_v32_mfu.src.q1_replay_matched import digest, require

KIND = "deepseek-q1-replay-prelude-v1"
DRIVER = "experiments/deepseek_v32_mfu/src/q1_replay_prelude.py"
BASE_WRAPPER = "experiments/deepseek_v32_mfu/src/q1_replay_matched.py"
BASE_KIND = "deepseek-q1-replay-matched-v1"
METHODS = ["hbm", "echo", "serial_sparse", "dense_prefetch"]
COMPLETED = {
    "methods": METHODS,
    "compute_bank_preparations": 1,
    "cache_generation_increase": 5,
    "same_compute_bank": True,
    "fresh_empty_hbm_cache": True,
    "offload_pools_sessions_helpers_released": True,
    "prelude_extend_graphs_closed": True,
}


def validate_prelude(result, baseline):
    source = result["identity"]["source"]
    previous = baseline["identity"]["source"]
    require(
        source["sources"][BASE_WRAPPER] == digest(matched.ROOT / BASE_WRAPPER),
        "Base matched wrapper changed",
    )
    require(
        {name: value for name, value in source["sources"].items() if name != DRIVER}
        == previous["sources"],
        "Source changes exceed the explicit prelude driver",
    )
    contract = source["contract"]["four_method_prelude"]
    require(
        contract["kind"] == KIND
        and contract["methods"] == METHODS
        and contract["warmups_per_method"] == 1
        and contract["extend_residency"] == "cold"
        and contract["position"]
        == "after the one compute bank; before the original reduced HBM prefix"
        and contract["final_transition"]
        == "select_cache_method(hbm) creates an empty resident cache"
        and contract["collection"] == "stopped by the unchanged timer; no added profiler ranges",
        "Prelude execution contract differs",
    )
    expected_contract = {
        **previous["contract"],
        "matched_environment": KIND,
        "four_method_prelude": contract,
    }
    require(source["contract"] == expected_contract, "Other reduced timer conditions changed")
    require(
        {key: value for key, value in source.items() if key not in ("sources", "contract")}
        == {key: value for key, value in previous.items() if key not in ("sources", "contract")},
        "Matched baseline environment/input/source reference differs",
    )
    runtime = result["identity"]["runtime"]
    require(runtime["four_method_prelude"] == COMPLETED, "Prelude completion gates differ")
    require(
        runtime["backend"] == baseline["identity"]["runtime"]["backend"],
        "Prelude/baseline provider identities differ",
    )
    require(
        all(not rows for rows in runtime["formal_match"]["unobserved_formal_entries"].values()),
        "The formal prelude did not observe all expected runtime entries",
    )


def baseline_audit(directory):
    result = matched.read(directory / "result.json")
    require(result["accepted"] is True, "Matched baseline audit not accepted")
    require(result["matched_control"]["receipt_kind"] == BASE_KIND, "Wrong baseline receipt kind")
    for group in ("inputs_sha256", "analysis_sources_sha256"):
        for path, expected in result[group].items():
            require(digest(path) == expected, "Matched baseline evidence changed")
    profiles = [
        Path(path)
        for path in result["inputs_sha256"]
        if Path(path).name == "result.json" and matched.read(path).get("mode") == "profile"
    ]
    require(len(profiles) == 1, "Expected one matched baseline profile result")
    profile = matched.read(profiles[0])
    require(
        digest(profile["receipt"]["path"]) == profile["receipt"]["sha256"],
        "Baseline check receipt changed",
    )
    require_receipt(profile["receipt"]["path"], kind=BASE_KIND, identity=profile["identity"])
    return result, profile


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--matched-audit-dir",
        type=Path,
        default=matched.EXPERIMENT / "output/data/q1_replay_matched_audit_20261008_01",
    )
    options, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        sys.argv = [sys.argv[0], *remaining]
        matched.main()
        return
    context_parser = argparse.ArgumentParser(add_help=False)
    context_parser.add_argument("--output-dir", type=Path, required=True)
    context, _ = context_parser.parse_known_args(remaining)
    baseline_dir = options.matched_audit_dir.resolve()
    baseline, baseline_profile = baseline_audit(baseline_dir)
    original_validator = matched.validate_execution

    def validate_execution(directory, mode, formal_dir):
        result = original_validator(directory, mode, formal_dir)
        validate_prelude(result, baseline_profile)
        return result

    previous = (matched.KIND, matched.WRAPPER, matched.validate_execution, sys.argv)
    try:
        matched.KIND, matched.WRAPPER = KIND, DRIVER
        matched.validate_execution = validate_execution
        sys.argv = [sys.argv[0], *remaining]
        matched.main()
    finally:
        matched.KIND, matched.WRAPPER, matched.validate_execution, sys.argv = previous
    output = context.output_dir.resolve()
    result = matched.read(output / "result.json")
    require(
        [scope["label"] for scope in result["scopes"]]
        == [scope["label"] for scope in baseline["scopes"]],
        "Scope sequence differs",
    )
    comparison = []
    for current, reference in zip(result["scopes"], baseline["scopes"], strict=True):
        require(
            [matched.timer_audit.signature(row) for row in current["nodes"]]
            == [matched.timer_audit.signature(row) for row in reference["nodes"]],
            "Prelude/baseline raw GPU signatures differ",
        )
        require(
            [row["owner"] for row in current["nodes"]]
            == [row["owner"] for row in reference["nodes"]],
            "Native owners differ",
        )
        keys = ("span_us", "busy_us", "idle_us", "gap_median_ns", "gap_max_ns")
        comparison.append(
            {
                "label": current["label"],
                "matched_baseline": {key: reference["statistics"][key] for key in keys},
                "formal_prelude": {key: current["statistics"][key] for key in keys},
            }
        )
    for path in (Path(__file__).resolve(), matched.ROOT / BASE_WRAPPER):
        shutil.copy2(path, output / path.name)
        result["analysis_sources_sha256"][str(path)] = digest(path)
    shutil.copy2(baseline_dir / "result.json", output / "matched_baseline_audit.json")
    result["inputs_sha256"][str(baseline_dir / "result.json")] = digest(
        baseline_dir / "result.json"
    )
    result["formal_prelude_control"] = {
        "receipt_kind": KIND,
        "driver_sha256": digest(matched.ROOT / DRIVER),
        "completed_state": COMPLETED,
        "matched_baseline_audit_run_id": baseline_dir.name,
        "profile_comparison": comparison,
        "clean_wall_comparison": {
            "matched_baseline": baseline["clean_benchmark"],
            "formal_prelude": result["clean_benchmark"],
        },
        "boundary": "Two separately accepted process histories, each with within-process "
        "plain/timed AB/BA pairs. The prelude treatment itself is not process-balanced; "
        "no cross-process causal speedup or specific allocator/stream mechanism follows.",
    }
    result["boundary"] += " Formal-prelude driver/receipt and completed-state gates also passed."
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
