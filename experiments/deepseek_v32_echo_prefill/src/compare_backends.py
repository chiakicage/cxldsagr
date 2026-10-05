"""Compare fresh three-layer implementation runs with identical workload inputs."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from experiments.deepseek_v32_echo_prefill.src.run_contract import (
    benchmark_view,
    control_directory,
    validated_receipt,
)

DEFAULT_PURPOSE = (
    "Original implementation versus official compute backends, same pre-shared-cache semantics"
)
DEFAULT_BOUNDARY = (
    "Different physical H200 devices, default clocks, one request. Speedups are ratios of "
    "measured medians, without a confidence interval. Cross-backend closeness is reported, "
    "not assumed; official MLA is accepted separately against FP32. These runs do not "
    "validate the concurrent shared-cache implementation or GR serving."
)
COMMON_FIELDS = (
    "scope",
    "num_layers",
    "prefix_tokens",
    "extend_tokens",
    "chunk_size",
    "slots",
    "warmups",
    "repeats",
    "prefill_repeats",
    "request_sha256",
    "checkpoint_metadata_sha256",
)
CACHE_FIELDS = (
    "cache_policy_revision",
    "pool_scope",
    "sparse_pool_tokens",
    "host_arena_tokens",
    "workspace_query_tokens",
    "hbm_cache_budget_bytes",
    "dram_cache_budget_bytes",
    "extend_chunk_size",
    "snapshot_schema",
    "snapshot_scope",
    "prefetch_cap",
    "prefetch_flags",
)
OPTIONAL_COMMON_FIELDS = CACHE_FIELDS + ("checkpoint_num_layers", "timed_output", "timing")
CORRECTNESS_FIELDS = {"resident_vs_offload_hidden", "resident_vs_offload_logits"} | {
    mode + "_" + suffix
    for mode in ("resident", "offload")
    for suffix in ("profile_prefix_logits", "profile_extend_logits", "profile_extend_hidden")
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_result(directory, run, *, require_benchmark=True):
    """Keep the accepted three-layer run's internal numerical gate unchanged."""
    if run["accepted"] is not True or run["num_layers"] != 3:
        raise ValueError("Both runs must be accepted three-layer measurements")
    if run.get("schema_version", 1) == 2:
        original = run
        run = benchmark_view(directory, run, required=require_benchmark)
        checks = validated_receipt(original)["checks"]["comparisons"]
        if not all(row["bitwise_equal"] is True for row in checks.values()):
            raise ValueError("Independent check did not pass the internal bitwise audit")
        if original["mode"] == "bench":
            return run
    if set(run["correctness"]) != CORRECTNESS_FIELDS or not all(
        row["bitwise_equal"] is True
        and row["max_abs"] == 0
        and row["relative_l2"] == 0
        and row["rtol"] == 0.01
        and row["atol"] == 0.02
        for row in run["correctness"].values()
    ):
        raise ValueError("Each implementation must pass its eight resident/offload checks")
    if digest(directory / "request.json") != run["request_sha256"]:
        raise ValueError("Request SHA mismatch")
    if (
        run.get("schema_version", 1) == 2
        and run.get("mode") == "profile"
        and not run.get("benchmark")
    ):
        return run
    for mode in ("resident", "offload"):
        for phase in ("prefix", "extend"):
            samples = run["measurements"][mode][phase + "_samples_ms"]
            count = run["prefill_repeats" if phase == "prefix" else "repeats"]
            if (
                len(samples) != count
                or not samples
                or not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples)
            ):
                raise ValueError("Invalid timing samples or repeat count")
            if statistics.median(samples) != run["measurements"][mode][phase + "_median_ms"]:
                raise ValueError("Stored median differs from samples")
    return run


def compare_tensors(actual, previous):
    """Report every output element, without making closeness an acceptance gate."""
    import torch

    if actual.shape != previous.shape or actual.dtype != previous.dtype or not actual.numel():
        raise ValueError("Output shape/dtype mismatch or empty output")
    if not torch.isfinite(actual).all() or not torch.isfinite(previous).all():
        raise ValueError("Nonfinite output in accepted run")
    shape, dtype = list(actual.shape), str(actual.dtype)
    bitwise_equal = torch.equal(actual, previous)
    actual, previous = actual.float(), previous.float()
    difference = actual - previous
    outside = difference.abs() > 0.02 + 0.01 * previous.abs()
    return {
        "shape": shape,
        "dtype": dtype,
        "elements_compared": actual.numel(),
        "bitwise_equal": bitwise_equal,
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (difference.norm() / previous.norm().clamp_min(1e-20)).item(),
        "allclose_at_original_tolerance": not bool(outside.any()),
        "elements_outside_original_tolerance": int(outside.sum()),
        "rows_outside_original_tolerance": int(outside.reshape(len(actual), -1).any(-1).sum()),
        "rtol": 0.01,
        "atol": 0.02,
    }


def compare_selections(directories, runs):
    """Compare saved logical selections from each implementation's real propagation."""
    import torch

    layers = []
    for layer in range(runs["official"]["num_layers"]):
        filename = f"kernel_inputs_layer_{layer}.pt"
        payloads = {
            name: torch.load(path / filename, map_location="cpu", weights_only=True, mmap=True)
            for name, path in directories.items()
        }
        indices = {}
        for name, data in payloads.items():
            ids = data["indices"]
            if (
                data["source_run_id"] != runs[name]["run_id"]
                or data["layer"] != layer
                or data["query_start"] != runs[name]["prefix_tokens"]
                or ids.ndim != 2
                or ids.shape[0] != runs[name]["extend_tokens"]
                or ids.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("Captured selection identity/coverage mismatch")
            ends = torch.arange(len(ids))[:, None] + data["query_start"] + 1
            if not ((ids == -1) | ((ids >= 0) & (ids < ends))).all():
                raise ValueError("Captured selection contains invalid or noncausal IDs")
            ordered = ids.sort(-1).values
            if ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any():
                raise ValueError("Captured selection repeats a logical token")
            expected_valid = ends.squeeze(1).clamp_max(ids.shape[1])
            if not torch.equal((ids >= 0).sum(-1), expected_valid):
                raise ValueError("Captured selection violates exact causal top-k capacity")
            indices[name] = ids
        old, new = indices["control"], indices["official"]
        if old.shape != new.shape or old.shape[1] < 1:
            raise ValueError("Captured selection shape mismatch")
        ordered = old.sort(-1).values.contiguous()
        location = torch.searchsorted(ordered, new.contiguous())
        matches = (
            (new >= 0)
            & (location < old.shape[1])
            & (ordered.gather(1, location.clamp_max(old.shape[1] - 1)) == new)
        )
        intersection = matches.sum(-1)
        old_valid, new_valid = (old >= 0).sum(-1), (new >= 0).sum(-1)
        fractions = intersection.double() / old.shape[1]
        same_sets = (intersection == old_valid) & (intersection == new_valid)
        layers.append(
            {
                "layer": layer,
                "shape": list(old.shape),
                "query_start": runs["control"]["prefix_tokens"],
                "input_sha256": {
                    name: digest(path / filename) for name, path in directories.items()
                },
                "fraction_denominator_slots": old.shape[1],
                "intersection_count_by_query": intersection.tolist(),
                "intersection_fraction_of_slots_by_query": fractions.tolist(),
                "mean_intersection_fraction_of_slots": fractions.mean().item(),
                "min_intersection_fraction_of_slots": fractions.min().item(),
                "identical_selection_set_rows": int(same_sets.sum()),
                "identical_selection_set_row_fraction": same_sets.double().mean().item(),
                "identical_ordered_ids_row_fraction": (old == new).all(-1).double().mean().item(),
                "control_valid_count_by_query": old_valid.tolist(),
                "candidate_valid_count_by_query": new_valid.tolist(),
            }
        )
    return {
        "boundary": "Each run's saved logical selections follow its own actual propagated activations. Drift describes the complete nonmatrix change set; it is not attributed to an individual operation and is not a task-quality or accuracy gate.",
        "layers": layers,
    }


def compare_cache_metrics(runs):
    """Preserve actual per-layer counters without attributing drift to one operation."""
    modes = {}
    for mode in ("resident", "offload"):
        measured = {name: run["measurements"][mode] for name, run in runs.items()}
        field = "extend_cache_per_layer"
        if not any(field in row for row in measured.values()):
            modes[mode] = {"available": False, "layers": []}
            continue
        if any(field not in row for row in measured.values()) or any(
            len(row[field]) != runs[name]["num_layers"] for name, row in measured.items()
        ):
            raise ValueError("Extend cache metric coverage mismatch")
        layers = []
        for layer, (old, new) in enumerate(
            zip(measured["control"][field], measured["official"][field])
        ):
            numeric_delta = {
                key: new[key] - old[key]
                for key in old.keys() & new.keys()
                if type(old[key]) in (int, float) and type(new[key]) in (int, float)
            }
            fractions = {}
            for name, metrics in (("control", old), ("candidate", new)):
                selected = metrics.get("selection_records", 0)
                resident = metrics.get("resident_selection_records")
                fractions[name] = resident / selected if selected and resident is not None else None
            layers.append(
                {
                    "layer": layer,
                    "control": old,
                    "candidate": new,
                    "candidate_minus_control": numeric_delta,
                    "ensure_resident_selection_fraction": fractions,
                }
            )
        modes[mode] = {"available": True, "layers": layers}
    return {
        "boundary": "Actual extend_cache_per_layer counters from each run are retained, including any prefetched/recalled/hit/split fields supplied by that implementation. The derived resident fraction divides resident_selection_records by selection_records over ensure consumption groups after prefetch; it is not a prefetch-free request-level cache-hit rate. Selection and cache-access drift can contribute to offload wall-time changes, so overall speedup cannot be assigned wholly to faster nonmatrix kernels.",
        "modes": modes,
    }


def compare(
    control,
    official,
    output,
    *,
    purpose=DEFAULT_PURPOSE,
    boundary=DEFAULT_BOUNDARY,
    require_same_gpu=False,
):
    import torch

    if not all(isinstance(text, str) and text.strip() for text in (purpose, boundary)):
        raise ValueError("Comparison purpose and boundary must be nonempty")
    directories = {"control": control, "official": official}
    runs = {
        name: json.loads((path / "result.json").read_text()) for name, path in directories.items()
    }
    runs = {name: validate_result(directories[name], run) for name, run in runs.items()}
    controls = {name: control_directory(directories[name], run) for name, run in runs.items()}
    # Historical runs may omit the whole newer cache contract. A one-sided
    # omission is a mismatch, so a new cache cannot silently reuse old results.
    optional = tuple(
        key for key in OPTIONAL_COMMON_FIELDS if any(key in run for run in runs.values())
    )
    common = COMMON_FIELDS + optional
    for key in common:
        if any(key not in run for run in runs.values()) or (
            runs["control"][key] != runs["official"][key]
        ):
            raise ValueError(f"Workload mismatch: {key}")
    plans = {}
    for mode in ("resident", "offload"):
        measured = {name: run["measurements"][mode] for name, run in runs.items()}
        if any("cache_resource_plan" in row for row in measured.values()):
            if any("cache_resource_plan" not in row for row in measured.values()) or (
                measured["control"]["cache_resource_plan"]
                != measured["official"]["cache_resource_plan"]
            ):
                raise ValueError(f"Workload mismatch: {mode}/cache_resource_plan")
            plans[mode] = measured["official"]["cache_resource_plan"]
    gpu_uuids = {name: run["hardware"]["gpu"]["uuid"] for name, run in runs.items()}
    same_gpu = len({uuid.removeprefix("GPU-") for uuid in gpu_uuids.values()}) == 1
    if require_same_gpu and not same_gpu:
        raise ValueError("Comparison requires the same physical GPU UUID")
    rows = []
    for mode in ("resident", "offload"):
        for phase in ("prefix", "extend"):
            row = {"mode": mode, "phase": phase}
            for name, run in runs.items():
                samples = run["measurements"][mode][phase + "_samples_ms"]
                median = statistics.median(samples)
                row[name + "_median_ms"] = median
                row[name + "_samples_ms"] = samples
            row["speedup"] = row["control_median_ms"] / row["official_median_ms"]
            rows.append(row)
    numerical, output_hashes = {}, {}
    for mode in ("resident", "offload"):
        filename = mode + "_control.pt"
        outputs = {
            name: torch.load(path / filename, map_location="cpu", weights_only=True)
            for name, path in controls.items()
        }
        output_hashes[mode] = {name: digest(path / filename) for name, path in controls.items()}
        numerical[mode] = {}
        for key in ("hidden", "logits"):
            actual, previous = (outputs[name][key] for name in ("official", "control"))
            expected_rows = runs["official"]["extend_tokens"] if key == "hidden" else 1
            if actual.ndim != 2 or actual.shape[0] != expected_rows:
                raise ValueError(f"Incomplete {mode}/{key} output coverage")
            numerical[mode][key] = compare_tensors(actual, previous)
            if key == "logits":
                numerical[mode][key]["same_argmax"] = bool(
                    (actual.argmax(-1) == previous.argmax(-1)).all()
                )
    if all(run.get("mode") != "bench" for run in runs.values()):
        selection_drift = compare_selections(directories, runs)
    else:
        selection_drift = {
            "available": False,
            "boundary": "Selection comparison requires profile runs bound to these independent benches",
            "layers": [],
        }
    cache_metrics = compare_cache_metrics(runs)
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "purpose": purpose,
        "runs": {
            name: {
                "run_id": run["run_id"],
                "result_sha256": digest(directories[name] / "result.json"),
                "hardware": run["hardware"],
                "source_sha256": run["source_sha256"],
                "wall_time_denominator": run.get("wall_time_denominator"),
                "validation_receipt": run.get("validation_receipt"),
            }
            for name, run in runs.items()
        },
        "common_workload": {key: runs["official"][key] for key in common},
        "common_cache_resource_plans": plans,
        "cache_fields_absent_in_both_historical_runs": [
            key for key in CACHE_FIELDS if key not in optional
        ],
        "hardware_comparison": {"same_physical_gpu": same_gpu, "gpu_uuids": gpu_uuids},
        "latencies": rows,
        "cross_backend_numerics": numerical["resident"],
        "cross_backend_numerics_by_mode": numerical,
        "compared_output_sha256": output_hashes,
        "selection_drift": selection_drift,
        "extend_cache_metric_comparison": cache_metrics,
        "comparison_source_sha256": digest(Path(__file__)),
        "boundary": boundary,
    }
    (output / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "latency_comparison.csv").open("w") as stream:
        fields = ["mode", "phase", "control_median_ms", "official_median_ms", "speedup"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "latencies": rows}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--official", "--candidate", dest="official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", default=DEFAULT_PURPOSE)
    parser.add_argument("--boundary", default=DEFAULT_BOUNDARY)
    parser.add_argument("--require-same-gpu", action="store_true")
    args = parser.parse_args()
    compare(
        args.control,
        args.official,
        args.output,
        purpose=args.purpose,
        boundary=args.boundary,
        require_same_gpu=args.require_same_gpu,
    )


if __name__ == "__main__":
    main()
