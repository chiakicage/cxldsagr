"""Audit allocation-only history, separate host evidence and unchanged GPU replay."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from evaluation.validation import require_receipt
from experiments.deepseek_v32_mfu.src import analyze_replay_hbm_prelude as hbm
from experiments.deepseek_v32_mfu.src import analyze_replay_matched as matched
from experiments.deepseek_v32_mfu.src.q1_replay_matched import digest, require

KIND = "deepseek-q1-replay-host-allocation-v1"
DRIVER = "experiments/deepseek_v32_mfu/src/q1_replay_host_allocation.py"
HELPER = "cache/host_allocation.py"
EVIDENCE = "host_allocation_observations.json"
COMPLETED = {
    "lifetimes": 3,
    "simultaneous_backings": 3,
    "allocation_requests": 9,
    "logical_bytes_per_request": 75_571_200,
    "storage_bytes_per_request": 134_217_728,
    "view_and_backing_owners_dropped": True,
    "compute_bank_preparations": 1,
    "same_compute_bank": True,
    "cache_generation_increase": 0,
    "original_empty_hbm_cache_unchanged": True,
}
REFERENCES = {
    "matched": (
        "q1_replay_matched_audit_20261008_01",
        hbm.four_method.BASE_KIND,
        "matched_control",
    ),
    "four_method": (
        "q1_replay_prelude_audit_20261008_01",
        hbm.four_method.KIND,
        "formal_prelude_control",
    ),
    "hbm_prelude": ("q1_replay_hbm_prelude_audit_20261008_01", hbm.KIND, "hbm_prelude_control"),
}


def validate_contract(result, baseline):
    source, previous = result["identity"]["source"], baseline["identity"]["source"]
    require(
        {name: value for name, value in source["sources"].items() if name != DRIVER}
        == previous["sources"],
        "Source changes exceed the allocation-only driver",
    )
    helper = {"path": HELPER, "sha256": digest(matched.ROOT / HELPER)}
    require(source["sources"][HELPER] == helper["sha256"], "Allocation helper source differs")
    contract = source["contract"]["host_allocation"]
    expected = {
        "kind": KIND,
        "helper": helper,
        "shape": [65600, 576],
        "dtype": "bfloat16",
        "pin_memory": True,
        "lifetimes": 3,
        "simultaneous_backings": 3,
        "allocation_requests": 9,
        "logical_bytes_per_request": 75_571_200,
        "storage_bytes_per_request": 134_217_728,
        "position": "after the one compute bank; before the original reduced HBM prefix",
        "cache_transitions": 0,
        "collection": "stopped by the unchanged timer; no added profiler ranges",
        "diagnostic_evidence": EVIDENCE,
    }
    require(
        all(contract[key] == value for key, value in expected.items()),
        "Allocation contract differs",
    )
    require(
        source["contract"]
        == {**previous["contract"], "matched_environment": KIND, "host_allocation": contract},
        "Other reduced timer conditions changed",
    )
    require(
        {key: value for key, value in source.items() if key not in ("sources", "contract")}
        == {key: value for key, value in previous.items() if key not in ("sources", "contract")},
        "Baseline environment/input/source reference differs",
    )
    runtime = result["identity"]["runtime"]
    require(runtime["host_allocation"] == COMPLETED, "Allocation completion gates differ")
    require(runtime["backend"] == baseline["identity"]["runtime"]["backend"], "Provider differs")


def validate_observations(result, directory):
    binding = result["host_allocation_observations"]
    require(binding["path"] == EVIDENCE, "Unexpected observation path")
    path = directory / EVIDENCE
    require(
        digest(path) == binding["sha256"] and path.stat().st_size == binding["bytes"],
        "Observation evidence changed",
    )
    evidence = matched.read(path)
    sources = result["identity"]["source"]["sources"]
    require(
        evidence["schema"] == "q1-host-allocation-observations-v1"
        and evidence["run_id"] == result["run_id"]
        and evidence["mode"] == result["mode"]
        and evidence["driver_sha256"] == sources[DRIVER]
        and evidence["helper_sha256"] == sources[HELPER],
        "Observation source or run identity differs",
    )
    phases = [{"phase": "before_allocation_lifetimes"}]
    for lifetime in range(3):
        phases.extend(
            (
                {
                    "phase": "three_backings_live",
                    "lifetime": lifetime,
                    "logical_bytes": [75_571_200] * 3,
                    "storage_bytes": [134_217_728] * 3,
                },
                {"phase": "view_and_backing_owners_dropped", "lifetime": lifetime},
            )
        )
    phases.extend(
        (
            {"phase": "runtime_after_graph_construction_and_five_warmups", "runtime_call": 1},
            {"phase": "runtime_after_check_or_samples", "runtime_call": 2},
        )
    )
    rows = evidence["observations"]
    require(
        [
            {key: value for key, value in row.items() if key != "reported_host_allocator_stats"}
            for row in rows
        ]
        == phases,
        "Expected nine exact observation boundaries and allocation sizes",
    )
    for row in rows:
        stats = row["reported_host_allocator_stats"]
        require(
            all(
                type(stats[key]) is int and stats[key] >= 0
                for key in ("allocated_bytes.current", "num_host_alloc", "num_host_free")
            ),
            "Invalid reported owned-byte or allocation/free counter",
        )
    return {"path": str(path), "sha256": binding["sha256"], "bytes": binding["bytes"], **evidence}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name, (run_id, _, _) in REFERENCES.items():
        parser.add_argument(
            "--" + name.replace("_", "-") + "-audit-dir",
            type=Path,
            default=matched.EXPERIMENT / "output/data" / run_id,
        )
    options, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        sys.argv = [sys.argv[0], *remaining]
        matched.main()
        return
    context_parser = argparse.ArgumentParser(add_help=False)
    context_parser.add_argument("--output-dir", type=Path, required=True)
    context, _ = context_parser.parse_known_args(remaining)
    references, profiles, directories = {}, {}, {}
    for name, (_, kind, control) in REFERENCES.items():
        directories[name] = getattr(options, name + "_audit_dir").resolve()
        references[name], profiles[name] = hbm.reference_audit(
            directories[name], kind=kind, control=control
        )
    baseline = profiles["matched"]
    hbm.four_method.validate_prelude(profiles["four_method"], baseline)
    hbm.validate_hbm_prelude(profiles["hbm_prelude"], baseline)
    original_validator = matched.validate_execution
    observations = {}

    def validate_execution(directory, mode, formal_dir):
        result = original_validator(directory, mode, formal_dir)
        validate_contract(result, baseline)
        observations[mode] = validate_observations(result, directory)
        if "check" not in observations:
            receipt = require_receipt(
                result["receipt"]["path"], kind=KIND, identity=result["identity"]
            )
            check_path = Path(receipt["artifact_paths"]["result"])
            check = matched.read(check_path)
            require(
                check["accepted"] is True
                and check["mode"] == "check"
                and check["identity"] == result["identity"],
                "Independent check result differs",
            )
            validate_contract(check, baseline)
            observations["check"] = validate_observations(check, check_path.parent)
            require(
                Path(receipt["artifact_paths"]["host_allocation_observations"])
                == Path(observations["check"]["path"]),
                "Check receipt does not bind its allocation observations",
            )
        return result

    previous = (matched.KIND, matched.WRAPPER, matched.validate_execution, sys.argv)
    try:
        matched.KIND, matched.WRAPPER = KIND, DRIVER
        matched.validate_execution = validate_execution
        sys.argv = [sys.argv[0], *remaining]
        matched.main()
    finally:
        matched.KIND, matched.WRAPPER, matched.validate_execution, sys.argv = previous
    require(set(observations) == {"check", "bench", "profile"}, "Missing observation evidence")
    output = context.output_dir.resolve()
    result = matched.read(output / "result.json")
    comparisons = []
    for name, reference in references.items():
        require(
            [scope["label"] for scope in result["scopes"]]
            == [scope["label"] for scope in reference["scopes"]],
            "Reference scope sequence differs",
        )
        for current, expected in zip(result["scopes"], reference["scopes"], strict=True):
            require(
                [matched.timer_audit.signature(node) for node in current["nodes"]]
                == [matched.timer_audit.signature(node) for node in expected["nodes"]]
                and [node["owner"] for node in current["nodes"]]
                == [node["owner"] for node in expected["nodes"]],
                "Allocation/reference GPU signatures or native owners differ",
            )
            keys = ("span_us", "busy_us", "idle_us", "gap_median_ns", "gap_max_ns")
            comparisons.append(
                {
                    "reference": name,
                    "label": current["label"],
                    "allocation": {key: current["statistics"][key] for key in keys},
                    "reference_statistics": {key: expected["statistics"][key] for key in keys},
                }
            )
    for path in (
        Path(__file__).resolve(),
        Path(hbm.__file__).resolve(),
        Path(hbm.four_method.__file__).resolve(),
        matched.ROOT / hbm.BASE_WRAPPER,
    ):
        shutil.copy2(path, output / path.name)
        result["analysis_sources_sha256"][str(path)] = digest(path)
    for name, directory in directories.items():
        path = directory / "result.json"
        shutil.copy2(path, output / f"{name}_reference_audit.json")
        result["inputs_sha256"][str(path)] = digest(path)
    for mode, evidence in observations.items():
        shutil.copy2(evidence["path"], output / f"{mode}_{EVIDENCE}")
        result["inputs_sha256"][evidence["path"]] = evidence["sha256"]
    result["host_allocation_control"] = {
        "receipt_kind": KIND,
        "driver_sha256": digest(matched.ROOT / DRIVER),
        "completed_state": COMPLETED,
        "reference_audit_run_ids": {name: path.name for name, path in directories.items()},
        "observations": observations,
        "profile_comparison": comparisons,
        "clean_wall_comparison": {
            **{name: reference["clean_benchmark"] for name, reference in references.items()},
            "allocation": result["clean_benchmark"],
        },
        "boundary": "Nine allocation requests in three owner lifetimes; allocator growth/free "
        "counters and owned bytes are reported exactly as saved and may differ across check, "
        "bench and profile. They are outside exact execution identity. No nine-CUDA-allocation "
        "claim, active-minus-cached inference or allocation API trace is implied. The normal "
        "graph-entry host-cache flush remains. This audit alone neither balances this new "
        "treatment across processes nor establishes an allocation/visibility mechanism.",
    }
    result["boundary"] += (
        " Allocation-only contract, completion and separate observation evidence verified."
    )
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
