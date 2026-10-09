"""Audit the observed four-method prelude without changing the raw replay audit."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from evaluation.validation import require_receipt
from experiments.deepseek_v32_mfu.src import analyze_replay_hbm_prelude as references
from experiments.deepseek_v32_mfu.src import analyze_replay_matched as matched
from experiments.deepseek_v32_mfu.src import analyze_replay_prelude as four_method
from experiments.deepseek_v32_mfu.src.q1_replay_matched import digest, require
from experiments.deepseek_v32_mfu.src.q1_replay_prelude_host_stats import (
    EVIDENCE,
    KIND,
    PARENT_KIND,
    PHASES,
)

DRIVER = "experiments/deepseek_v32_mfu/src/q1_replay_prelude_host_stats.py"
SOURCE_INPUTS = (
    DRIVER,
    four_method.DRIVER,
    four_method.BASE_WRAPPER,
    "experiments/deepseek_v32_mfu/src/q1_replay_timer.py",
)


def validate_contract(result, parent):
    source, previous = result["identity"]["source"], parent["identity"]["source"]
    require(
        {name: value for name, value in source["sources"].items() if name != DRIVER}
        == previous["sources"],
        "Source changes exceed the host-statistics observer",
    )
    for name in SOURCE_INPUTS:
        require(source["sources"][name] == digest(matched.ROOT / name), "Frozen driver changed")
    observer = source["contract"]["prelude_host_statistics"]
    expected = {
        "kind": KIND,
        "parent_kind": PARENT_KIND,
        "prelude_source": four_method.DRIVER,
        "callbacks": list(PHASES),
        "observation_count": 2,
        "diagnostic_evidence": EVIDENCE,
    }
    require(
        set(observer) == {*expected, "boundary"}
        and all(observer[key] == value for key, value in expected.items()),
        "Observer contract differs",
    )
    require(
        source["contract"]
        == {
            **previous["contract"],
            "matched_environment": KIND,
            "four_method_prelude": {**previous["contract"]["four_method_prelude"], "kind": KIND},
            "prelude_host_statistics": observer,
        },
        "Four-method prelude or other timer conditions changed",
    )
    require(
        {key: value for key, value in source.items() if key not in ("sources", "contract")}
        == {key: value for key, value in previous.items() if key not in ("sources", "contract")},
        "Prelude environment/input/reference differs",
    )
    require(
        {key: value for key, value in result["identity"].items() if key != "source"}
        == {key: value for key, value in parent["identity"].items() if key != "source"},
        "Prelude runtime or graph identity changed, or variable observations entered identity",
    )
    require(
        result["identity"]["runtime"]["four_method_prelude"] == four_method.COMPLETED,
        "Four-method completion gates differ",
    )


def validate_observations(result, directory):
    binding = result["prelude_host_allocator_observations"]
    require(
        set(binding) == {"path", "sha256", "bytes"}
        and binding["path"] == EVIDENCE
        and type(binding["bytes"]) is int,
        "Unexpected observer evidence binding",
    )
    path = directory / EVIDENCE
    require(
        digest(path) == binding["sha256"] and path.stat().st_size == binding["bytes"],
        "Host allocator observations changed",
    )
    evidence = matched.read(path)
    sources = result["identity"]["source"]["sources"]
    require(
        evidence["schema"] == "q1-prelude-host-allocator-observations-v1"
        and evidence["kind"] == KIND
        and evidence["run_id"] == result["run_id"] == directory.name
        and evidence["mode"] == result["mode"]
        and evidence["source_sha256"] == {name: sources[name] for name in SOURCE_INPUTS},
        "Observer evidence source or run differs",
    )
    rows = evidence["observations"]
    require(len(rows) == 2, "Expected two existing runtime observations")
    for index, (row, phase) in enumerate(zip(rows, PHASES, strict=True), 1):
        require(
            set(row) == {"phase", "runtime_call", "reported_host_allocator_stats"}
            and row["phase"] == phase
            and type(row["runtime_call"]) is int
            and row["runtime_call"] == index,
            "Observer callback boundary differs",
        )
        stats = row["reported_host_allocator_stats"]
        require(
            all(
                type(stats[key]) is int and stats[key] >= 0
                for key in ("allocated_bytes.current", "num_host_alloc", "num_host_free")
            ),
            "Invalid reported owned bytes or allocation/free counters",
        )
    return {"path": str(path), "sha256": binding["sha256"], "bytes": binding["bytes"], **evidence}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name, run_id in (
        ("matched", "q1_replay_matched_audit_20261008_01"),
        ("four-method", "q1_replay_prelude_audit_20261008_01"),
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
    }
    audits, profiles = {}, {}
    for name, kind, control in (
        ("matched", four_method.BASE_KIND, "matched_control"),
        ("four_method", PARENT_KIND, "formal_prelude_control"),
    ):
        audits[name], profiles[name] = references.reference_audit(
            directories[name], kind=kind, control=control
        )
    four_method.validate_prelude(profiles["four_method"], profiles["matched"])
    original_validator = matched.validate_execution
    observations = {}

    def validate_execution(directory, mode, formal_dir):
        result = original_validator(directory, mode, formal_dir)
        validate_contract(result, profiles["four_method"])
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
                "Independent observer check differs",
            )
            validate_contract(check, profiles["four_method"])
            observations["check"] = validate_observations(check, check_path.parent)
            require(
                Path(receipt["artifact_paths"]["prelude_host_allocator_observations"])
                == Path(observations["check"]["path"]),
                "Check receipt does not bind its host observations",
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
    require(set(observations) == {"check", "bench", "profile"}, "Missing host observations")
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
                "Observer/reference GPU signatures or native owners differ",
            )
            comparison.append(
                {
                    "reference": name,
                    "label": actual["label"],
                    "observed_prelude": {key: actual["statistics"][key] for key in keys},
                    "reference_statistics": {key: expected["statistics"][key] for key in keys},
                }
            )
    for path in (
        Path(__file__).resolve(),
        Path(references.__file__).resolve(),
        Path(four_method.__file__).resolve(),
        matched.ROOT / four_method.DRIVER,
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
    result["prelude_host_statistics_control"] = {
        "receipt_kind": KIND,
        "driver_sha256": digest(matched.ROOT / DRIVER),
        "parent_driver_sha256": digest(matched.ROOT / four_method.DRIVER),
        "completed_state": four_method.COMPLETED,
        "reference_audit_run_ids": {name: path.name for name, path in directories.items()},
        "observations": observations,
        "profile_comparison": comparison,
        "clean_wall_comparison": {
            **{name: audit["clean_benchmark"] for name, audit in audits.items()},
            "observed_prelude": result["clean_benchmark"],
        },
        "boundary": "The only source delta from accepted B1 is the observer. Its two saved "
        "PyTorch host-allocator observations stay outside execution identity and may vary "
        "across phases. Allocator-owned rounded bytes do not identify individual owners or "
        "cover other allocators. No active/cached subtraction or fence mechanism is inferred. "
        "Raw GPU audits, B1/A1 signature/owner comparisons and clean wall timings remain separate.",
    }
    result["boundary"] += " Observer-only source delta and both external host snapshots verified."
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
