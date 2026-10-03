"""Read-only recovery audit; no CUDA imports, allocation, replay, or publication."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from experiments.gr_cache_serving.src.fixed_budget import PROFILE, validate_trace
from experiments.gr_cache_serving.src.host_memory import (
    inspect_host_availability,
    inspect_pinned_capacity,
    plan_host_cache,
)
from experiments.gr_cache_serving.src.replay import MODES, source_fingerprints
from experiments.gr_cache_serving.src.report_capacity import load_capacity_runs

POPULATIONS = (64, 128, 256, 384, 512)


def inspect_trace(directory, population, profile=None):
    metadata = json.loads((directory / "metadata.json").read_text())
    if (
        metadata["artifact_type"] != "prepared_gr_workload"
        or metadata["stable_prefix_tokens"] != 65536
        or metadata["candidate_suffix_tokens"] != 1024
        or metadata["stats"]["population_users"] != population
        or metadata["stats"]["requests"] != (512 if profile == PROFILE else 4 * population)
    ):
        raise ValueError(f"unexpected fixed-sweep input: {directory}")
    if profile == PROFILE:
        validate_trace(metadata)
    for name in ("requests", "users"):
        with (directory / f"{name}.jsonl").open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != metadata[f"{name}_sha256"]:
            raise ValueError(f"input checksum mismatch: {directory}/{name}.jsonl")
    return metadata


def audit_sweep(base, group, populations=POPULATIONS, modes=MODES, profile=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        raise ValueError("invalid run group")
    if (
        not populations
        or len(set(populations)) != len(populations)
        or any(value not in POPULATIONS for value in populations)
        or not modes
        or len(set(modes)) != len(modes)
        or any(value not in MODES for value in modes)
    ):
        raise ValueError("invalid or repeated population/mode")
    if profile not in (None, PROFILE):
        raise ValueError("unknown sweep profile")
    current_source = source_fingerprints()
    completed_by_version, entries = {}, []
    for population in populations:
        trace = inspect_trace(base / f"{group}_input_u{population}", population, profile)
        for mode in modes:
            path = base / f"{group}_u{population}_{mode}_r1"
            entry = {"run_id": path.name, "population": population, "mode": mode}
            if path.exists():
                # A partially published directory is an error, never a completed run.
                _, summary, _ = load_capacity_runs([path], allow_partial=True)[0]
                if (
                    summary["mode"] != mode
                    or summary["host_users_capacity"] != (128 if profile else population)
                    or summary.get("experiment_profile") != profile
                    or summary["trace_metadata"] != trace
                    or summary["budget_plan"]["total_hbm_bytes"] != 72 * 2**30
                ):
                    raise ValueError(f"run does not match the requested sweep: {path}")
                measured = summary["source_sha256"]
                version = hashlib.sha256(json.dumps(measured, sort_keys=True).encode()).hexdigest()
                entry.update(
                    status="complete",
                    requests=summary["requests"],
                    measured_source_version=version,
                    source_changes_since_run=sorted(
                        name
                        for name in measured.keys() | current_source.keys()
                        if measured.get(name) != current_source.get(name)
                    ),
                )
                completed_by_version.setdefault(version, []).append(path)
            else:
                entry["status"] = "pending"
                if profile:
                    plan = plan_host_cache(
                        users=128, layers=5, prefix=65536, suffix=1024, budget_bytes=66 * 2**30
                    )
                    try:
                        entry["host_preflight"] = inspect_host_availability(plan)
                    except MemoryError as error:
                        entry.update(status="blocked_host", reason=str(error))
                elif mode != "resident":
                    payload = (population * 65536 + 1088) * 5 * 1152
                    try:
                        entry["host_preflight"] = inspect_pinned_capacity(payload)
                    except MemoryError as error:
                        entry.update(status="blocked_host", reason=str(error))
            entries.append(entry)
    for completed in completed_by_version.values():
        # Validate each measured version without silently pooling distinct versions.
        load_capacity_runs(completed, allow_partial=True)
    return {
        "run_group": group,
        "profile": profile,
        "scope": "read-only integrity and necessary Host capacity checks; no GPU readiness claim",
        "counts": {
            status: sum(entry["status"] == status for entry in entries)
            for status in ("complete", "pending", "blocked_host")
        },
        "runs": entries,
        "measured_source_versions": {
            version: [path.name for path in paths]
            for version, paths in completed_by_version.items()
        },
        "resume_note": (
            "Never overwrite completed runs. Source changes are not silently accepted as "
            "measurement-compatible; the report still requires identical measured fingerprints. "
            "Pending means no published result; a replay may already be running. It does not "
            "guarantee free pinned memory, GPU health or runtime feasibility."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group")
    parser.add_argument(
        "--base", type=Path, default=Path("experiments/gr_cache_serving/output/data")
    )
    parser.add_argument("--populations", type=int, nargs="+", default=POPULATIONS)
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--profile", choices=(PROFILE,))
    args = parser.parse_args(argv)
    print(
        json.dumps(
            audit_sweep(args.base, args.group, args.populations, args.modes, args.profile), indent=2
        )
    )


if __name__ == "__main__":
    main()
