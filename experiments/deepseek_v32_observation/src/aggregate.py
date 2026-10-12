"""Aggregate independent clean measurements, rejecting incomplete matrices."""

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.deepseek_v32_observation.src.bandwidth import HBM_FIELDS, validate_bandwidth
from experiments.deepseek_v32_observation.src.contract import (
    GROUPS,
    PEAK_CONVENTION,
    PEAK_SOURCE,
    PEAKS_TFLOPS,
    SCHEMA,
    matrix,
    task_keys,
    validate_result,
)
from experiments.deepseek_v32_observation.src.provenance import save_json


def merge_profile(clean, profile):
    """Join independent NCU observations without altering the clean result."""
    for field in ("group", "key", "identity", "receipt_path"):
        if clean.get(field) != profile.get(field):
            raise ValueError(f"HBM profile differs from clean workload {field}")
    profiles = {operator["name"]: operator for operator in profile["operators"]}
    clean_names = {operator["name"] for operator in clean["operators"]}
    if len(profiles) != len(profile["operators"]) or set(profiles) != clean_names:
        raise ValueError("HBM profile must cover the exact clean operator inventory")
    merged = copy.deepcopy(clean)
    for operator in merged["operators"]:
        observed = profiles[operator["name"]]
        validate_bandwidth(observed)
        if any(field in operator for field in HBM_FIELDS):
            raise ValueError("Independent clean results must not contain NCU profile metrics")
        operator.update({field: observed[field] for field in HBM_FIELDS})
    return merged


def aggregate(directory, run_id):
    directory = Path(directory)
    groups = {}
    for group in GROUPS:
        groups[group] = [
            merge_profile(
                json.loads((directory / group / key / "result.json").read_text()),
                json.loads((directory / group / key / "hbm.json").read_text()),
            )
            for key in task_keys(group)
        ]
    result = {
        "schema": SCHEMA,
        "run_id": run_id,
        "offload": False,
        "created_at": datetime.now(UTC).isoformat(),
        "matrix": [shape.as_dict() for shape in matrix()],
        "groups": groups,
        "peaks_tflops": PEAKS_TFLOPS,
        "peak_source": PEAK_SOURCE,
        "peak_convention": PEAK_CONVENTION,
        "scope": "Independent device-resident operator API observation; not full-model performance",
        "common_mapping": "Common and MoE measured once per S, mapped to matching B/Q/H cases",
    }
    validate_result(result)
    save_json(directory / "results.json", result)
    print(f"Aggregated {sum(len(entries) for entries in groups.values())} workloads", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.data_dir, args.run_id)


if __name__ == "__main__":
    main()
