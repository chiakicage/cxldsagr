"""Select verified latency, MFU and pipeline evidence for the motivation report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from statistics import mean

from experiments.deepseek_v32_motivation.src.analyze_pipeline import write_csv, write_json

SCHEMES = ("hbm", "echo", "serial_sparse", "dense_prefetch")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def publish(base, profile, flops, output):
    result = read_json(profile / "result.json")
    metadata = read_json(profile / "metadata.json")
    pipeline = read_json(profile / "analysis/pipeline.json")
    verification = read_json(profile / "analysis/flops_verification.json")
    work = read_json(flops / "flops.json")
    if not result["accepted"] or result["captures"] != 8 or result["exact_output_count"] != 80:
        raise ValueError("expected eight accepted captures and eighty exact-output checks")
    if not verification["passed"] or verification["mismatches"]:
        raise ValueError("FLOPs verification must pass before publication")
    if metadata["reference_run_id"] != base.name or work["source_run_id"] != base.name:
        raise ValueError("profile and MFU must refer to the same formal run")
    graph_enabled = metadata["config"].get("enable_compute_graphs", False)
    graph_setups = metadata.get("graph_setup_captures", [])
    expected_captures = {(row["scheme"], row["phase"]) for row in metadata["captures"]}
    if (
        len(pipeline["captures"]) != len(metadata["captures"])
        or {(row["scheme"], row["phase"]) for row in pipeline["captures"]} != expected_captures
    ):
        raise ValueError("pipeline analysis must contain every profiled request exactly once")
    if graph_enabled and (
        len(graph_setups) != len(SCHEMES)
        or result.get("setup_captures") != len(SCHEMES)
        or result.get("nsys_captures") != 3 * len(SCHEMES)
    ):
        raise ValueError("graph publication requires one setup and two request captures per scheme")
    if graph_enabled and read_json(profile / "graph_capture_ledger.json") != graph_setups:
        raise ValueError("standalone graph ledger differs from profile metadata")
    for capture in pipeline["captures"]:
        if (
            capture["unattributed_gpu_count"]
            or not capture["stage_totals_conserve_gpu_count"]
            or not capture["stage_totals_conserve_gpu_ns"]
            or not capture["cpu_exclusive_conserves_thread_unions"]
        ):
            raise ValueError("incomplete trace attribution")
        if digest(profile / Path(capture["sqlite"]).name) != capture["sqlite_sha256"]:
            raise ValueError("trace changed after analysis")
        if graph_enabled:
            audit = capture.get("graph_attribution") or {}
            if not audit.get("every_replay_gpu_node_verified") or not audit.get("replays"):
                raise ValueError("graph replay attribution must pass before publication")
            expected = {
                "metadata.json",
                "operator_calls.json",
                *(item["sqlite"] for item in graph_setups),
            }
            recorded = {
                Path(name).name: value for name, value in capture["graph_input_sha256"].items()
            }
            if recorded.keys() != expected:
                raise ValueError("graph analysis inputs are incomplete")
            for name, checksum in recorded.items():
                if digest(profile / name) != checksum:
                    raise ValueError("graph inputs changed after analysis")
    requests = [json.loads(line) for line in (base / "measurements.jsonl").read_text().splitlines()]
    mfu = {(r["scheme"], r["visit_kind"], r["stage"]): r for r in work["stage_mfu"]}
    hbm = [r for r in requests if r["scheme"] == "hbm" and not r["is_revisit"]]
    hbm_total = mean(r["latency_ms"] for r in hbm)
    hbm_history = mean(r["prefix_ms"] for r in hbm)
    formal = []
    for scheme in SCHEMES:
        first = [r for r in requests if r["scheme"] == scheme and not r["is_revisit"]]
        if len(first) != 16:
            raise ValueError("formal run must contain sixteen first visits per scheme")
        row = {"scheme": scheme}
        for name in ("admission", "prefix", "extend", "cleanup", "latency"):
            row[f"first_{name}_mean_ms"] = mean(r[f"{name}_ms"] for r in first)
        extra = row["first_latency_mean_ms"] - hbm_total
        row["history_fraction_of_extra_latency"] = (
            (row["first_prefix_mean_ms"] - hbm_history) / extra if extra else None
        )
        for visit, stage in (
            ("first", "e2e"),
            ("first", "prefix"),
            ("first", "extend"),
            ("revisit", "extend"),
        ):
            row[f"{visit}_{stage}_mfu_pct"] = mfu[(scheme, visit, stage)]["effective_mfu_pct"]
        formal.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "formal_latency_mfu.csv", formal)
    selected = {
        **{name: flops / name for name in ("flops.json", "stage_mfu.csv", "operator_flops.csv")},
        **{
            name: profile / "analysis" / name
            for name in (
                "pipeline.json",
                "stage_costs.csv",
                "cuda_apis.csv",
                "flops_verification.json",
                "revisit_pipeline_evidence.json",
                "revisit_layer_pipeline.csv",
                "revisit_fetch_windows.csv",
                "revisit_dependency_edges.csv",
            )
        },
    }
    if graph_enabled:
        selected["graph_capture_ledger.json"] = profile / "graph_capture_ledger.json"
    for path in sorted((profile / "analysis").glob("pipeline_*.*")):
        if path.suffix in (".png", ".svg"):
            selected[path.name] = path
    for path in sorted((profile / "analysis").glob("cold_*.*")):
        if path.suffix in (".json", ".csv"):
            selected[path.name] = path
    for name, path in selected.items():
        shutil.copyfile(path, output / name)
    captures = []
    for capture in metadata["captures"]:
        captures.append(
            {
                k: capture[k]
                for k in (
                    "capture_index",
                    "scheme",
                    "phase",
                    "request_id",
                    "diagnostic_runner_latency_ms",
                )
            }
            | {
                "segment_counters": {
                    phase: {
                        key: value
                        for key, value in counters.items()
                        if key not in ("layers", "layer_diagnostics")
                    }
                    for phase, counters in capture["segment_counters"].items()
                }
            }
        )
    write_json(
        output / "profile_metadata.json",
        {
            "result": result,
            "hardware": metadata["hardware"],
            "precision_settings": metadata["precision_settings"],
            "captures": captures,
            "graph_setup_captures": graph_setups,
            "graph_attribution_boundary": metadata.get("graph_attribution_boundary"),
            "instrumentation_boundary": metadata["instrumentation_boundary"],
            "counter_boundary": metadata["counter_boundary"],
            "source_verified_after_execution": metadata["source_verified_after_execution"],
        },
    )
    write_json(
        output / "publication.json",
        {
            "base_run_id": base.name,
            "profile_run_id": profile.name,
            "flops_analysis_run_id": flops.name,
            "result": result,
            "input_sha256": {
                str(path): digest(path)
                for path in (
                    base / "measurements.jsonl",
                    profile / "metadata.json",
                    profile / "result.json",
                    flops / "flops.json",
                    *(profile / item["sqlite"] for item in graph_setups),
                )
            },
            "analysis_sources": pipeline["analysis_sources"]
            | {str(Path(__file__).resolve()): digest(__file__)},
            "published_sha256": {
                path.name: digest(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "publication.json"
            },
            "boundaries": [
                f"Formal latency comes from uninstrumented {base.name}; diagnostic captures do not replace it.",
                "The profile uses the formal run's configuration and verifies exact outputs against its independent HBM reference. Instrumentation has a separate source identity.",
                "Graph matrix intervals, when enabled, use capture-time API node sets and Nsight's explicit executable clone lineage. Setup captures do not contribute request latency.",
                "Effective MFU is matrix work divided by reference dense peaks and formal wall time, not a hardware counter.",
                "Nsight provides whole-kernel and copy intervals; internal fused ECHO fetch timing was not measured.",
            ],
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--flops-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    publish(args.base_run, args.profile_run, args.flops_dir, args.output_dir)
    print(f"Selected verified diagnosis evidence: {args.output_dir}")


if __name__ == "__main__":
    main()
