"""Audit four HBM warmups against accepted matched and four-method histories."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from evaluation.validation import require_receipt
from experiments.deepseek_v32_mfu.src import analyze_replay_matched as matched
from experiments.deepseek_v32_mfu.src import analyze_replay_prelude as four_method
from experiments.deepseek_v32_mfu.src.q1_replay_matched import digest, require

KIND = "deepseek-q1-replay-hbm-prelude-v1"
DRIVER = "experiments/deepseek_v32_mfu/src/q1_replay_hbm_prelude.py"
BASE_WRAPPER = "experiments/deepseek_v32_mfu/src/q1_replay_matched.py"
METHODS = ["hbm"] * 4
COMPLETED = {
    "methods": METHODS,
    "compute_bank_preparations": 1,
    "cache_generation_increase": 5,
    "same_compute_bank": True,
    "fresh_empty_hbm_cache": True,
    "shared_pools_sessions_helpers_empty": True,
    "prelude_extend_graphs_closed": True,
}


def validate_hbm_prelude(result, baseline):
    source = result["identity"]["source"]
    previous = baseline["identity"]["source"]
    require(
        source["sources"][BASE_WRAPPER] == digest(matched.ROOT / BASE_WRAPPER),
        "Base matched wrapper changed",
    )
    require(
        {name: value for name, value in source["sources"].items() if name != DRIVER}
        == previous["sources"],
        "Source changes exceed the explicit HBM-prelude driver",
    )
    contract = source["contract"]["hbm_prelude"]
    require(
        contract["kind"] == KIND
        and contract["methods"] == METHODS
        and contract["warmup_count"] == 4
        and contract["extend_residency"] == "cold"
        and contract["position"]
        == "after the one compute bank; before the original reduced HBM prefix"
        and contract["final_transition"]
        == "select_cache_method(hbm) creates an empty resident cache"
        and contract["collection"] == "stopped by the unchanged timer; no added profiler ranges",
        "HBM-prelude execution contract differs",
    )
    require(
        source["contract"]
        == {**previous["contract"], "matched_environment": KIND, "hbm_prelude": contract},
        "Other reduced timer conditions changed",
    )
    require(
        {key: value for key, value in source.items() if key not in ("sources", "contract")}
        == {key: value for key, value in previous.items() if key not in ("sources", "contract")},
        "Matched baseline environment/input/source reference differs",
    )
    runtime = result["identity"]["runtime"]
    require(runtime["hbm_prelude"] == COMPLETED, "HBM-prelude completion gates differ")
    require(
        runtime["backend"] == baseline["identity"]["runtime"]["backend"],
        "HBM-prelude/baseline provider identities differ",
    )
    # matched.validate_execution already checks exact participating formal entries.
    # HBM-only warmups legitimately leave offload entries unobserved.


def reference_audit(directory, *, kind, control):
    result = matched.read(directory / "result.json")
    require(result["accepted"] is True, "Reference audit not accepted")
    require(result[control]["receipt_kind"] == kind, "Wrong reference receipt kind")
    require(result["matched_control"]["receipt_kind"] == kind, "Reference wrapper kind differs")
    for group in ("inputs_sha256", "analysis_sources_sha256"):
        for path, expected in result[group].items():
            require(digest(path) == expected, "Reference audit evidence changed")
    profiles = [
        Path(path)
        for path in result["inputs_sha256"]
        if Path(path).name == "result.json" and matched.read(path).get("mode") == "profile"
    ]
    require(len(profiles) == 1, "Expected one reference profile result")
    profile = matched.read(profiles[0])
    require(
        digest(profile["receipt"]["path"]) == profile["receipt"]["sha256"],
        "Reference check receipt changed",
    )
    require_receipt(profile["receipt"]["path"], kind=kind, identity=profile["identity"])
    return result, profile


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--matched-audit-dir",
        type=Path,
        default=matched.EXPERIMENT / "output/data/q1_replay_matched_audit_20261008_01",
    )
    parser.add_argument(
        "--four-method-audit-dir",
        type=Path,
        default=matched.EXPERIMENT / "output/data/q1_replay_prelude_audit_20261008_01",
    )
    options, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        sys.argv = [sys.argv[0], *remaining]
        matched.main()
        return
    context_parser = argparse.ArgumentParser(add_help=False)
    context_parser.add_argument("--output-dir", type=Path, required=True)
    context, _ = context_parser.parse_known_args(remaining)
    reference_dirs = {
        "matched": options.matched_audit_dir.resolve(),
        "four_method": options.four_method_audit_dir.resolve(),
    }
    baseline, baseline_profile = reference_audit(
        reference_dirs["matched"], kind=four_method.BASE_KIND, control="matched_control"
    )
    prelude, prelude_profile = reference_audit(
        reference_dirs["four_method"], kind=four_method.KIND, control="formal_prelude_control"
    )
    four_method.validate_prelude(prelude_profile, baseline_profile)
    original_validator = matched.validate_execution
    validated = {}

    def validate_execution(directory, mode, formal_dir):
        result = original_validator(directory, mode, formal_dir)
        validate_hbm_prelude(result, baseline_profile)
        validated[mode] = result
        return result

    previous = (matched.KIND, matched.WRAPPER, matched.validate_execution, sys.argv)
    try:
        matched.KIND, matched.WRAPPER = KIND, DRIVER
        matched.validate_execution = validate_execution
        sys.argv = [sys.argv[0], *remaining]
        matched.main()
    finally:
        matched.KIND, matched.WRAPPER, matched.validate_execution, sys.argv = previous
    require(set(validated) == {"profile", "bench"}, "Missing accepted HBM-prelude execution")
    output = context.output_dir.resolve()
    result = matched.read(output / "result.json")
    references = {"matched": baseline, "four_method": prelude}
    comparison = []
    for reference in references.values():
        require(
            [scope["label"] for scope in result["scopes"]]
            == [scope["label"] for scope in reference["scopes"]],
            "Reference scope sequence differs",
        )
    keys = ("span_us", "busy_us", "idle_us", "gap_median_ns", "gap_max_ns")
    for index, current in enumerate(result["scopes"]):
        row = {
            "label": current["label"],
            "hbm_prelude": {key: current["statistics"][key] for key in keys},
        }
        for name, reference in references.items():
            scope = reference["scopes"][index]
            require(
                [matched.timer_audit.signature(node) for node in current["nodes"]]
                == [matched.timer_audit.signature(node) for node in scope["nodes"]],
                "HBM-prelude/reference raw GPU signatures differ",
            )
            require(
                [node["owner"] for node in current["nodes"]]
                == [node["owner"] for node in scope["nodes"]],
                "HBM-prelude/reference native owners differ",
            )
            row[name] = {key: scope["statistics"][key] for key in keys}
        comparison.append(row)
    for path in (
        Path(__file__).resolve(),
        Path(four_method.__file__).resolve(),
        matched.ROOT / BASE_WRAPPER,
    ):
        shutil.copy2(path, output / path.name)
        result["analysis_sources_sha256"][str(path)] = digest(path)
    for name, directory in reference_dirs.items():
        path = directory / "result.json"
        shutil.copy2(path, output / f"{name}_reference_audit.json")
        result["inputs_sha256"][str(path)] = digest(path)
    result["hbm_prelude_control"] = {
        "receipt_kind": KIND,
        "driver_sha256": digest(matched.ROOT / DRIVER),
        "completed_state": COMPLETED,
        "reference_audit_run_ids": {name: path.name for name, path in reference_dirs.items()},
        "profile_comparison": comparison,
        "clean_wall_comparison": {
            **{name: reference["clean_benchmark"] for name, reference in references.items()},
            "hbm_prelude": result["clean_benchmark"],
        },
        "unobserved_formal_runtime_entries": {
            name: profile["identity"]["runtime"]["formal_match"]["unobserved_formal_entries"]
            for name, profile in (
                ("matched", baseline_profile),
                ("four_method", prelude_profile),
                ("hbm_prelude", validated["profile"]),
            )
        },
        "boundary": "Three separately accepted process histories. Four HBM warmups preserve "
        "the four-method prelude's warmup count, one compute bank and five cache transitions. "
        "Actual participating runtime entries must match formal identities; unused offload "
        "entries may remain unloaded. This audit alone does not balance the new HBM-prelude "
        "treatment across processes or isolate allocation, streams or graph history.",
    }
    result["boundary"] += " HBM-prelude driver/receipt and actual completion gates also passed."
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
