"""Audit cache construction against accepted matched and warmup histories."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from experiments.deepseek_v32_mfu.src import analyze_replay_hbm_prelude as references
from experiments.deepseek_v32_mfu.src import analyze_replay_matched as matched
from experiments.deepseek_v32_mfu.src import analyze_replay_prelude as four_method
from experiments.deepseek_v32_mfu.src.q1_replay_matched import digest, require

KIND = "deepseek-q1-replay-cache-construction-v1"
DRIVER = "experiments/deepseek_v32_mfu/src/q1_replay_cache_construction.py"
METHODS = ["hbm", "echo", "serial_sparse", "dense_prefetch"]
COMPLETED = {
    "selected_methods": [*METHODS, "hbm"],
    "compute_bank_preparations": 1,
    "cache_generation_increase": 5,
    "same_compute_bank": True,
    "all_selected_caches_empty": True,
    "fresh_empty_hbm_cache": True,
    "shared_pools_sessions_helpers_empty": True,
    "extend_graphs_empty_after_each_selection": True,
}
CONTRACT = {
    "kind": KIND,
    "methods": METHODS,
    "final_method": "hbm",
    "position": "after the one compute bank; before the original reduced HBM prefix",
    "model_forwards": 0,
    "prefix_snapshots_or_restores": 0,
    "extend_graph_preparations": 0,
    "collection": "stopped by the unchanged timer; no added profiler ranges",
    "boundary": "Construct and release the four fresh cache methods, then return "
    "to HBM without prelude prefix or forward execution. Cache/native/GPU-buffer/"
    "stream initialization is one package, not an isolated allocation cause. "
    "The measured reduced prefix, capture and replay remain unchanged. "
    "No added flush, statistics query or statistics reset.",
}


def validate_construction(result, baseline):
    source, previous = result["identity"]["source"], baseline["identity"]["source"]
    require(
        source["sources"][DRIVER] == digest(matched.ROOT / DRIVER)
        and source["sources"][references.BASE_WRAPPER]
        == digest(matched.ROOT / references.BASE_WRAPPER),
        "Frozen cache-construction driver or matched wrapper changed",
    )
    require(
        {name: value for name, value in source["sources"].items() if name != DRIVER}
        == previous["sources"],
        "Source changes exceed the explicit cache-construction driver",
    )
    require(
        source["contract"]
        == {**previous["contract"], "matched_environment": KIND, "cache_construction": CONTRACT},
        "Cache construction or other reduced timer conditions changed",
    )
    require(
        {key: value for key, value in source.items() if key not in ("sources", "contract")}
        == {key: value for key, value in previous.items() if key not in ("sources", "contract")},
        "Matched baseline environment/input/reference differs",
    )
    runtime = result["identity"]["runtime"]
    require(
        set(runtime) == {"backend", "loaded_jit", "formal_match", "cache_construction"}
        and runtime["cache_construction"] == COMPLETED
        and runtime["backend"] == baseline["identity"]["runtime"]["backend"],
        "Cache-construction completion or provider identities differ",
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
    # Cache construction need not execute or load all offload kernels.


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name, run_id in (
        ("matched", "q1_replay_matched_audit_20261008_01"),
        ("four-method", "q1_replay_prelude_audit_20261008_01"),
        ("hbm-prelude", "q1_replay_hbm_prelude_audit_20261008_01"),
    ):
        parser.add_argument(
            f"--{name}-audit-dir", type=Path, default=matched.EXPERIMENT / "output/data" / run_id
        )
    options, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        sys.argv = [sys.argv[0], *remaining]
        matched.main()
        return
    context_parser = argparse.ArgumentParser(add_help=False)
    context_parser.add_argument("--output-dir", type=Path, required=True)
    context, _ = context_parser.parse_known_args(remaining)
    directories = {
        "matched": options.matched_audit_dir.resolve(),
        "four_method": options.four_method_audit_dir.resolve(),
        "hbm_prelude": options.hbm_prelude_audit_dir.resolve(),
    }
    audits, profiles = {}, {}
    for name, kind, control in (
        ("matched", four_method.BASE_KIND, "matched_control"),
        ("four_method", four_method.KIND, "formal_prelude_control"),
        ("hbm_prelude", references.KIND, "hbm_prelude_control"),
    ):
        audits[name], profiles[name] = references.reference_audit(
            directories[name], kind=kind, control=control
        )
    four_method.validate_prelude(profiles["four_method"], profiles["matched"])
    references.validate_hbm_prelude(profiles["hbm_prelude"], profiles["matched"])
    original_validator = matched.validate_execution
    validated = {}

    def validate_execution(directory, mode, formal_dir):
        result = original_validator(directory, mode, formal_dir)
        validate_construction(result, profiles["matched"])
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
    require(set(validated) == {"profile", "bench"}, "Missing cache-construction execution")
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
                "Cache-construction/reference GPU signatures or native owners differ",
            )
            comparison.append(
                {
                    "reference": name,
                    "label": actual["label"],
                    "cache_construction": {key: actual["statistics"][key] for key in keys},
                    "reference_statistics": {key: expected["statistics"][key] for key in keys},
                }
            )
    for path in (
        Path(__file__).resolve(),
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
    result["cache_construction_control"] = {
        "receipt_kind": KIND,
        "driver_sha256": digest(matched.ROOT / DRIVER),
        "completed_state": COMPLETED,
        "reference_audit_run_ids": {name: path.name for name, path in directories.items()},
        "profile_comparison": comparison,
        "clean_wall_comparison": {
            **{name: audit["clean_benchmark"] for name, audit in audits.items()},
            "cache_construction": result["clean_benchmark"],
        },
        "unobserved_formal_runtime_entries": {
            name: profile["identity"]["runtime"]["formal_match"]["unobserved_formal_entries"]
            for name, profile in {**profiles, "cache_construction": validated["profile"]}.items()
        },
        "boundary": "Five cache selections follow one compute-bank construction without "
        "prelude prefix, forward, snapshot/restore or extend-graph preparation. This varies "
        "the cache/native/GPU-buffer/stream initialization package, not an individual "
        "allocation cause. Participating runtime entries match formal identities exactly; "
        "unused offload entries may remain unloaded. No host statistics are added. This "
        "audit alone does not provide fresh-process treatment replication or establish "
        "a hardware mechanism. Clean wall and intrusive GPU windows remain separate.",
    }
    result["boundary"] += " Cache-construction driver/receipt and completion gates also passed."
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
