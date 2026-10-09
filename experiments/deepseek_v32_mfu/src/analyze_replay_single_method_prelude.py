"""Audit one selected warmup against accepted matched and preparation histories."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from experiments.deepseek_v32_mfu.src import analyze_replay_cache_construction as construction
from experiments.deepseek_v32_mfu.src import analyze_replay_hbm_prelude as references
from experiments.deepseek_v32_mfu.src import analyze_replay_matched as matched
from experiments.deepseek_v32_mfu.src import analyze_replay_prelude as four_method
from experiments.deepseek_v32_mfu.src.q1_replay_matched import digest, require

KIND = "deepseek-q1-replay-single-method-prelude-v1"
DRIVER = "experiments/deepseek_v32_mfu/src/q1_replay_single_method_prelude.py"
METHODS = ("hbm", "echo", "serial_sparse", "dense_prefetch")


def completed(method):
    return {
        "method": method,
        "warmup_count": 1,
        "compute_bank_preparations": 1,
        "cache_generation_increase": 2,
        "same_compute_bank": True,
        "fresh_empty_hbm_cache": True,
        "shared_pools_sessions_helpers_empty": True,
        "prelude_extend_graphs_closed": True,
    }


def contract(method):
    return {
        "kind": KIND,
        "method": method,
        "warmup_count": 1,
        "extend_residency": "cold",
        "position": "after the one compute bank; before the original reduced HBM prefix",
        "final_transition": "select_cache_method(hbm) creates an empty resident cache",
        "collection": "stopped by the unchanged timer; no added profiler ranges",
        "boundary": "Run one selected method's complete formal warmup, then return "
        "to HBM. This varies a method-specific initialization/execution package, "
        "not an isolated DMA, allocation, stream or fence cause. The measured "
        "reduced prefix, capture and replay remain unchanged. No added flush or statistics.",
    }


def validate_single_method(result, baseline, method):
    require(method in METHODS, "Unexpected selected prelude method")
    source, previous = result["identity"]["source"], baseline["identity"]["source"]
    require(
        source["sources"][DRIVER] == digest(matched.ROOT / DRIVER)
        and source["sources"][references.BASE_WRAPPER]
        == digest(matched.ROOT / references.BASE_WRAPPER),
        "Frozen single-method driver or matched wrapper changed",
    )
    require(
        {name: value for name, value in source["sources"].items() if name != DRIVER}
        == previous["sources"],
        "Source changes exceed the explicit single-method driver",
    )
    require(
        source["contract"]
        == {
            **previous["contract"],
            "matched_environment": KIND,
            "single_method_prelude": contract(method),
        },
        "Selected method or other reduced timer conditions changed",
    )
    require(
        {key: value for key, value in source.items() if key not in ("sources", "contract")}
        == {key: value for key, value in previous.items() if key not in ("sources", "contract")},
        "Matched baseline environment/input/reference differs",
    )
    runtime = result["identity"]["runtime"]
    require(
        set(runtime) == {"backend", "loaded_jit", "formal_match", "single_method_prelude"}
        and runtime["single_method_prelude"] == completed(method)
        and runtime["backend"] == baseline["identity"]["runtime"]["backend"],
        "Selected method completion or provider identities differ",
    )
    require(
        {
            key: value
            for key, value in result["identity"].items()
            if key not in ("source", "runtime")
        }
        == {
            key: value
            for key, value in baseline["identity"].items()
            if key not in ("source", "runtime")
        },
        "Other reduced graph identity changed",
    )
    # The unchanged matched validator checks exact participating formal entries.
    # Each chosen method may leave different unused offload entries unobserved.


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prelude-method", choices=METHODS)
    for name, run_id in (
        ("matched", "q1_replay_matched_audit_20261008_01"),
        ("four-method", "q1_replay_prelude_audit_20261008_01"),
        ("hbm-prelude", "q1_replay_hbm_prelude_audit_20261008_01"),
        ("construction", "q1_replay_cache_construction_audit_20261008_01"),
    ):
        parser.add_argument(
            f"--{name}-audit-dir", type=Path, default=matched.EXPERIMENT / "output/data" / run_id
        )
    options, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
        sys.argv = [sys.argv[0], *remaining]
        matched.main()
        return
    require(options.prelude_method in METHODS, "Specify the expected --prelude-method")
    method = options.prelude_method
    context_parser = argparse.ArgumentParser(add_help=False)
    context_parser.add_argument("--output-dir", type=Path, required=True)
    context, _ = context_parser.parse_known_args(remaining)
    directories = {
        "matched": options.matched_audit_dir.resolve(),
        "four_method": options.four_method_audit_dir.resolve(),
        "hbm_prelude": options.hbm_prelude_audit_dir.resolve(),
        "cache_construction": options.construction_audit_dir.resolve(),
    }
    audits, profiles = {}, {}
    for name, kind, control in (
        ("matched", four_method.BASE_KIND, "matched_control"),
        ("four_method", four_method.KIND, "formal_prelude_control"),
        ("hbm_prelude", references.KIND, "hbm_prelude_control"),
        ("cache_construction", construction.KIND, "cache_construction_control"),
    ):
        audits[name], profiles[name] = references.reference_audit(
            directories[name], kind=kind, control=control
        )
    four_method.validate_prelude(profiles["four_method"], profiles["matched"])
    references.validate_hbm_prelude(profiles["hbm_prelude"], profiles["matched"])
    construction.validate_construction(profiles["cache_construction"], profiles["matched"])
    original_validator = matched.validate_execution
    validated = {}

    def validate_execution(directory, mode, formal_dir):
        result = original_validator(directory, mode, formal_dir)
        validate_single_method(result, profiles["matched"], method)
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
    require(set(validated) == {"profile", "bench"}, "Missing accepted single-method execution")
    output = context.output_dir.resolve()
    result = matched.read(output / "result.json")
    comparison = []
    keys = ("span_us", "busy_us", "idle_us", "gap_median_ns", "gap_max_ns")
    for name, audit in audits.items():
        require(
            [scope["label"] for scope in result["scopes"]]
            == [scope["label"] for scope in audit["scopes"]],
            "Reference scope sequence differs",
        )
        for actual, expected in zip(result["scopes"], audit["scopes"], strict=True):
            require(
                [matched.timer_audit.signature(node) for node in actual["nodes"]]
                == [matched.timer_audit.signature(node) for node in expected["nodes"]]
                and [node["owner"] for node in actual["nodes"]]
                == [node["owner"] for node in expected["nodes"]],
                "Single-method/reference GPU signatures or native owners differ",
            )
            comparison.append(
                {
                    "reference": name,
                    "label": actual["label"],
                    "single_method": {key: actual["statistics"][key] for key in keys},
                    "reference_statistics": {key: expected["statistics"][key] for key in keys},
                }
            )
    for path in (
        Path(__file__).resolve(),
        Path(construction.__file__).resolve(),
        Path(references.__file__).resolve(),
        Path(four_method.__file__).resolve(),
        matched.ROOT / references.BASE_WRAPPER,
    ):
        shutil.copy2(path, output / path.name)
        result["analysis_sources_sha256"][str(path)] = digest(path)
    for name, directory in directories.items():
        path = directory / "result.json"
        shutil.copy2(path, output / f"{name}_reference_audit.json")
        result["inputs_sha256"][str(path)] = digest(path)
    result["single_method_prelude_control"] = {
        "receipt_kind": KIND,
        "method": method,
        "driver_sha256": digest(matched.ROOT / DRIVER),
        "completed_state": completed(method),
        "reference_audit_run_ids": {name: path.name for name, path in directories.items()},
        "profile_comparison": comparison,
        "clean_wall_comparison": {
            **{name: audit["clean_benchmark"] for name, audit in audits.items()},
            "single_method": result["clean_benchmark"],
        },
        "unobserved_formal_runtime_entries": {
            name: profile["identity"]["runtime"]["formal_match"]["unobserved_formal_entries"]
            for name, profile in {**profiles, "single_method": validated["profile"]}.items()
        },
        "boundary": "One selected method's complete formal warmup follows one compute bank; "
        "the second cache selection returns to fresh HBM. Receipt, contract and completion "
        "bind the selected method. Participating runtime entries match formal identities "
        "exactly; unused offload entries may remain unloaded. This is a method-specific "
        "initialization/execution package, not an isolated DMA, allocation, stream or "
        "fence mechanism. This audit alone does not establish fresh-process treatment "
        "replication. Clean wall and intrusive GPU windows remain separate.",
    }
    result["boundary"] += " Selected-method driver/receipt and completion gates also passed."
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
