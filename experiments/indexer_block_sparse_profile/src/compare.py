"""Compare completed native and Triton runs captured from the same source tree."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.indexer_block_sparse_profile.src.analyze import PHASES, _duration, _integer
from experiments.indexer_block_sparse_profile.src.launch_report import IDENTITY_FIELDS
from experiments.indexer_block_sparse_profile.src.mfu import build_report

MODULES = ("pooled_scores", "block_sparse_attention", "indexer_total")
TIMING_ARGS = (
    "prefix_tokens",
    "new_tokens",
    "chunk_size",
    "warmup",
    "repeats",
    "profile_repeats",
    "device",
)
INPUTS = ("metadata.json", "summary.json", "mfu.json", "module_mfu.json")


def _load_run(data_dir, backend):
    data_dir = Path(data_dir)
    raw = {name: (data_dir / name).read_bytes() for name in INPUTS}
    metadata, summary, mfu, modules = [json.loads(raw[name]) for name in INPUTS]
    if summary["workload"].get("kernel_backend") != backend:
        raise ValueError(f"Expected kernel_backend={backend} in {data_dir}")
    if hashlib.sha256((data_dir / "request.json").read_bytes()).hexdigest() != metadata.get(
        "request_sha256"
    ):
        raise ValueError(f"Request content differs from its captured hash in {data_dir}")
    for key in TIMING_ARGS[:-1]:
        if metadata["args"].get(key) != summary["workload"].get(key):
            raise ValueError(f"Capture arguments and summary disagree on {key}")
    expected = build_report(metadata, summary, peak_tflops=mfu["peak_tflops"])
    for key in ("schema_version", "run_id", "gpu", "workload", "dimensions", "implementation"):
        if mfu.get(key) != expected[key] or modules.get(key) != expected[key]:
            raise ValueError(f"Derived reports disagree with captured {key} in {data_dir}")
    if mfu.get("phases") != expected["phases"]:
        raise ValueError(
            f"Stored MFU disagrees with benchmark timings or useful FLOPs in {data_dir}"
        )
    if modules.get("peak_tflops") != mfu["peak_tflops"]:
        raise ValueError("Module MFU and benchmark MFU must use the same peak")
    provenance = modules["provenance"]
    if provenance.get("captured_source_sha256") != metadata["source_sha256"]:
        raise ValueError("Module report source provenance differs from its capture")
    for label, digests, required in (
        ("mfu", mfu["input_sha256"], {"metadata.json", "summary.json"}),
        (
            "module_mfu",
            provenance["input_sha256"],
            {"metadata.json", "summary.json", "profile_metadata.json", "attention_audit.json"},
        ),
    ):
        if set(digests) != required:
            raise ValueError(f"Incomplete {label} input provenance in {data_dir}")
        for name, digest in digests.items():
            content = raw[name] if name in raw else (data_dir / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Stale {label} report: {name} hash mismatch in {data_dir}")
    repeats = _integer(summary["workload"]["profile_repeats"], "profile_repeats", minimum=1)
    for phase in PHASES:
        if modules["phases"][phase]["end_to_end"] != mfu["phases"][phase]:
            raise ValueError("Module report has different end-to-end measurements")
        for name in MODULES:
            row = modules["phases"][phase]["modules"][name]
            _duration(row["kernel_ms_median"], f"{phase}.{name}.kernel_ms_median")
            if row["sample_count"] != repeats or len(row["kernel_count_per_run"]) != repeats:
                raise ValueError("Incomplete module timing samples")
            for count in row["kernel_count_per_run"]:
                _integer(count, f"{phase}.{name}.kernel_count", minimum=1)
    return {
        "metadata": metadata,
        "summary": summary,
        "mfu": mfu,
        "modules": modules,
        "input_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in raw.items()
        },
    }


def compare(native_data_dir, triton_data_dir):
    """Validate identities before computing ratios of independently measured medians."""
    native = _load_run(native_data_dir, "cuda_tvm_ffi")
    triton = _load_run(triton_data_dir, "triton")
    left, right = native["metadata"], triton["metadata"]
    if left["run_id"] == right["run_id"]:
        raise ValueError("Native and Triton measurements require distinct run IDs")
    for key in set(IDENTITY_FIELDS) - {"native_build"}:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Native and Triton captures differ in {key}")
    for key in TIMING_ARGS:
        if key not in left["args"] or left["args"][key] != right["args"].get(key):
            raise ValueError(f"Native and Triton captures differ in argument {key}")
    builds = [
        {key: value for key, value in meta["native_build"].items() if key != "selected_backend"}
        for meta in (left, right)
    ]
    if builds[0] != builds[1]:
        raise ValueError("Native and Triton captures differ in native build inputs")
    workloads = [
        {key: value for key, value in run["summary"]["workload"].items() if key != "kernel_backend"}
        for run in (native, triton)
    ]
    if workloads[0] != workloads[1]:
        raise ValueError("Native and Triton captures differ in workload")
    for key in ("peak_tflops", "dimensions", "analysis_source_sha256"):
        if not native["mfu"].get(key) or native["mfu"][key] != triton["mfu"].get(key):
            raise ValueError(f"MFU reports differ in {key}")
    if (
        native["modules"]["provenance"]["analysis_source_sha256"]
        != triton["modules"]["provenance"]["analysis_source_sha256"]
    ):
        raise ValueError("Module reports use different analysis sources")
    rows = []
    for phase in PHASES:
        for name in ("end_to_end", *MODULES):
            if name == "end_to_end":
                values = [run["mfu"]["phases"][phase]["wall_ms"] for run in (native, triton)]
                boundary = "unprofiled wall time"
            else:
                values = [
                    run["modules"]["phases"][phase]["modules"][name]["kernel_ms_median"]
                    for run in (native, triton)
                ]
                boundary = "nsys correlated kernel duration sum"
            rows.append(
                {
                    "phase": phase,
                    "metric": name,
                    "timing_boundary": boundary,
                    "native_ms": values[0],
                    "triton_ms": values[1],
                    "speedup_triton_over_native": values[1] / values[0],
                    "native_run_id": left["run_id"],
                    "triton_run_id": right["run_id"],
                }
            )
    return {
        "schema_version": 1,
        "native_run_id": left["run_id"],
        "triton_run_id": right["run_id"],
        "workload": workloads[0],
        "gpu": left["gpu"],
        "model_config": left["model_config"],
        "rows": rows,
        "definitions": {
            "speedup_triton_over_native": "Triton median milliseconds / native median milliseconds; greater than 1 means native is faster",
            "boundaries": "End-to-end uses independent unprofiled wall timings; module rows use separate nsys captures and cannot be added to or subtracted from wall time",
            "inclusive_modules": "indexer_total includes pooled_scores, validation, cache updates and selection; do not sum parent and child rows",
        },
        "provenance": {
            "request_sha256": left["request_sha256"],
            "source_sha256": left["source_sha256"],
            "native_build_inputs": builds[0],
            "input_sha256": {"native": native["input_sha256"], "triton": triton["input_sha256"]},
            "comparison_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


def write_report(native_data_dir, triton_data_dir, output_dir):
    report = compare(native_data_dir, triton_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-data-dir", type=Path, required=True)
    parser.add_argument("--triton-data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New comparison data directory"
    )
    args = parser.parse_args(argv)
    try:
        write_report(args.native_data_dir, args.triton_data_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
