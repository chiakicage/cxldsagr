"""Publish the accepted fixed-prefix full-extend graph measurements and figures."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from evaluation.validation import identity_digest
from experiments.deepseek_v32_mfu.src.run_contract import METHODS, digest, validate_benchmark


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def table(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("check-run", "warm-check-run", "bench-run", "profile-run", "timeline-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--gap-source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directories = {
        name: getattr(args, name + "_run").resolve()
        for name in ("check", "warm_check", "bench", "profile")
    }
    runs = {name: read(path / "result.json") for name, path in directories.items()}
    for name, run in runs.items():
        if not run["accepted"] or not run["extend_graph"] or run["num_layers"] != 3:
            raise ValueError(f"Unaccepted full graph run: {name}")
    validate_benchmark(directories["bench"], runs["bench"])
    if any(
        identity_digest(runs[name]["execution_identity"])
        != identity_digest(runs["check"]["execution_identity"])
        for name in ("bench", "profile")
    ):
        raise ValueError("Cold check, bench and profile identities differ")
    warm = dict(runs["warm_check"]["execution_identity"])
    warm["extend_residency"] = "cold"
    if identity_digest(warm) != identity_digest(runs["check"]["execution_identity"]):
        raise ValueError("Warm check differs beyond residency")
    gap_path = directories["profile"] / "gap_audit.json"
    gap = read(gap_path)
    if gap["input_result_sha256"] != digest(directories["profile"] / "result.json"):
        raise ValueError("Gap analysis belongs to a different capture")
    timeline = read(args.timeline_dir / "compact_receipt.json")
    if (
        timeline["profile_run_id"] != runs["profile"]["run_id"]
        or timeline["window_kind"] != "three-layers"
        or timeline["layout"] != "separate"
        or timeline["annotations"] != "idle-echo"
        or timeline["io_layout"] != "directions"
    ):
        raise ValueError("Timeline does not use the requested final windows/layout")
    for name, expected in timeline["artifacts_sha256"].items():
        if digest(args.timeline_dir / name) != expected:
            raise ValueError(f"Timeline artifact changed: {name}")
    evidence = [*args.audit, *[path / "result.json" for path in directories.values()], gap_path]
    evidence.extend(args.timeline_dir / name for name in timeline["artifacts_sha256"])
    evidence.append(args.timeline_dir / "compact_receipt.json")
    gap_source_names = (
        "summary.csv",
        "control_sources.csv",
        "sources.json",
        "independent_audit.json",
    )
    if read(args.gap_source_dir / "sources.json")["profile_run_id"] != runs["profile"]["run_id"]:
        raise ValueError("Gap sources belong to a different profile")
    evidence.extend(args.gap_source_dir / name for name in gap_source_names)
    source = Path(__file__).resolve()
    evidence.append(source)
    before = {str(path.resolve()): digest(path) for path in evidence}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    copied = [
        "prefill.svg",
        "prefill.png",
        "extend.svg",
        "extend.png",
        "windows.csv",
        "annotations.csv",
        "compact_receipt.json",
    ]
    for name in copied:
        shutil.copy2(args.timeline_dir / name, out / name)
    (out / "audit").mkdir()
    for path in args.audit:
        shutil.copy2(path, out / "audit" / path.name)
    (out / "gap_sources").mkdir()
    for name in gap_source_names:
        shutil.copy2(args.gap_source_dir / name, out / "gap_sources" / name)
    timing, gates = [], []
    for method in METHODS:
        row = runs["bench"]["measurements"][method]
        for phase in ("prefill", "extend"):
            for index, value in enumerate(row[phase + "_samples_ms"]):
                timing.append({"method": method, "phase": phase, "sample": index, "wall_ms": value})
    for method in gap["methods"]:
        for label, row in [
            ("complete_extend", method["full_extend"]),
            *[("layer_" + str(row["layer"]), row) for row in method["layers"]],
        ]:
            gates.append(
                {
                    "method": method["method"],
                    "window": label,
                    "window_ms": row["window_ms"],
                    "gap_ms": row["gap_ms"],
                    "gap_percent": row["gap_no_io_percent"],
                    "pass_below_10_percent": row["gap_below_threshold"],
                }
            )
    table(out / "timing_samples.csv", timing)
    table(out / "extend_gate.csv", gates)
    summary = {
        "schema": "deepseek-full-extend-graph-report-v1",
        "run_ids": {name: run["run_id"] for name, run in runs.items()},
        "execution_identity_sha256": identity_digest(runs["check"]["execution_identity"]),
        "graph_policy": runs["bench"]["extend_graph_policy_revision"],
        "measurements": runs["bench"]["measurements"],
        "timeline_metrics": timeline["metrics"],
        "extend_gate_pass": gap["extend_gate_pass"],
        "all_layers_gate_pass": gap["all_layers_gate_pass"],
        "graph_membership": {row["method"]: row["graph_attribution"] for row in gap["methods"]},
        "full_extend": {row["method"]: row["full_extend"] for row in gap["methods"]},
        "layers": {row["method"]: row["layers"] for row in gap["methods"]},
        "scope": runs["bench"]["scope"],
        "hardware": runs["bench"]["hardware"],
        "dependencies": runs["bench"]["dependencies"],
        "execution_environment": runs["bench"]["execution_environment"],
        "numerical_boundary": "Cold and warm independent checks; runtime cache/replay checks and saved-tensor reread are distinguished in audit evidence.",
        "performance_boundary": "Cold only. Clean synchronized wall-time benchmark and intrusive node profile are separate. No new per-operator MFU.",
        "graph_boundary": "Fixed restored prefix, shape, output mode, cache storage and clocks. GPU body in one graph; validation/input staging and host begin/sync/commit remain outside.",
        "gap_definition": gap["gap_definition"],
        "gap_denominator": gap["no_io_definition"],
        "gate_definition": gap["gate_definition"],
        "postprocess": (
            read(directories["profile"] / "postprocess.json")
            if (directories["profile"] / "postprocess.json").is_file()
            else None
        ),
    }
    write(out / "summary.json", summary)
    write(
        out / "provenance.json",
        {
            "inputs_sha256": before,
            "source_sha256": digest(source),
            "argv": sys.argv,
            "directories": {key: str(path) for key, path in directories.items()},
            "timeline_directory": str(args.timeline_dir.resolve()),
            "publication_scope": "Selected artifacts from independently accepted execution and CPU-audited native captures; no GPU work during report generation.",
        },
    )
    lines = [
        "# Complete extend CUDA Graph",
        "",
        "The real checkpoint's L0–L2 run in one full CUDA Graph per extend, including cache operations and actual IO. Cold and warm correctness passed. This file records the original timeline-source measurement; the experiment README contains the current timing, MFU and selected-window gap results.",
        "",
        "H=65,536; A=128; prefill chunk=1,024; P=NH=65,664; ordinary persistent append. One matched warmup per method, 3 clean prefill samples and 5 clean extend samples. Graph preparation and prefix restoration are outside timing; begin/synchronization/commit are included.",
        "",
        "| Method | Prefill median ms | Extend median ms | Full-forward diagnostic gap % | Graph GPU nodes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = runs["bench"]["measurements"][method]
        lines.append(
            f"| {method} | {row['prefill_median_ms']:.3f} | {row['extend_median_ms']:.3f} | {summary['full_extend'][method]['gap_no_io_percent']:.2f} | {summary['graph_membership'][method]['gpu_node_activities']} |"
        )
    lines.extend(
        [
            "",
            "Figures are displayed only in the experiment README. Prefill shows only the final chunk (64/64). Main windows end at L2 final compute and start at L0 first compute, extended for dense extend to include its L0 history H2D if earlier. Compute phases use color; IO directions are separate; red denotes GPU idle only. ECHO prepare/finalize/hint are separate annotations.",
            "",
            "The full-forward diagnostic includes host startup and commit, and assigns startup to L0. The README uses the selected main window and its corresponding layer intervals, including L0 compute and history H2D for dense extend. Gap is the complement of compute and actual IO; fused ECHO is entirely productive and stays in the denominator.",
            "",
            "The graph is valid only for its restored fixed prefix, cache/storage/clock state, query shape and output mode. Outputs are borrowed. The result does not establish arbitrary growing-history replay, C10/NOSA full graphs, full 61-layer performance, or serving capacity limits.",
            "",
            "GPU execution, exported native traces and interval accounting are covered by the independent audit files. Any postprocessing repair is recorded in summary.postprocess; a null value means none was recorded. GPU ownership observations and their limits are recorded separately in the audit evidence. Discrete observations do not prove machine or CPU exclusivity.",
            "",
            "Run IDs and unrounded memory/latency data: [summary](summary.json). Raw sample rows: [timing](timing_samples.csv). Complete and layer gates: [gate table](extend_gate.csv). Cropped windows: [windows](windows.csv). Input hashes: [provenance](provenance.json). Independent evidence: `audit/`.",
            "",
        ]
    )
    (out / "results.md").write_text("\n".join(lines))
    if before != {str(path.resolve()): digest(path) for path in evidence}:
        raise RuntimeError("Report inputs changed during generation")
    write(
        out / "publication_manifest.json",
        {
            "schema": "selected-full-graph-artifacts-v1",
            "files_sha256": {
                str(path.relative_to(out)): digest(path)
                for path in sorted(out.rglob("*"))
                if path.is_file()
            },
        },
    )
    print(out / "publication_manifest.json")


if __name__ == "__main__":
    main()
