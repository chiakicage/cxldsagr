"""Compare existing cold/warm core reports without assuming instance availability."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.deepseek_v32_echo_official.src.analyze_q1_official_core import digest, require

METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_active.avg",
    "sm__cycles_active.min",
    "sm__cycles_active.max",
    "smsp__inst_executed_op_generic_atom_dot_alu.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum",
    "lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum",
    "dram__bytes_read.sum",
    "sass__inst_executed_register_spilling_mem_local",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
)
INSTANCE_METRICS = (
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "inst_executed",
    "thread_inst_executed_true",
    "pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg",
)


def capture(directory):
    def load(name):
        return json.loads((directory / name).read_text())

    metrics, instances, pcs = load("metrics.json"), load("instances.json"), load("pc_evidence.json")
    analysis = load("summary.json")
    scalars = {name: metrics[name] for name in METRICS}
    require(all(row["has_value"] for row in scalars.values()), "A cited aggregate is unavailable")
    validity = {}
    for name in INSTANCE_METRICS:
        rows = instances[name]
        nonzero = [row for row in rows if row["value"]]
        validity[name] = {
            "aggregate_has_value": metrics[name]["has_value"],
            "aggregate_value": metrics[name]["value"],
            "instances": len(rows),
            "nonzero_instances": len(nonzero),
            "nonzero_has_value_true": sum(row["has_value"] for row in nonzero),
            "nonzero_has_value_false": sum(not row["has_value"] for row in nonzero),
        }
    lookup = {row["absolute_pc"]: row for row in pcs}
    grouped = defaultdict(lambda: defaultdict(int))
    for row in instances["smsp__pcsamp_warps_issue_stalled_long_scoreboard"]:
        if not row["value"]:
            continue
        source = lookup[row["correlation"]]["source"]
        key = f"{Path(source['file']).name}:{source['line']}" if source else "<unmapped>"
        group = grouped[key]
        group["reported_samples"] += row["value"]
        group["nonzero_instances"] += 1
        group["instance_has_value_true" if row["has_value"] else "instance_has_value_false"] += 1
    executed = {row["correlation"]: row for row in instances["inst_executed"]}
    atoms = [
        {
            "source": row["source"],
            "sass": row["sass"],
            "absolute_pc": row["absolute_pc"],
            "raw_inst_executed": executed[row["absolute_pc"]],
        }
        for row in pcs
        if "ATOMG" in row["sass"]
    ]
    return {
        "report": analysis["report"],
        "report_sha256": analysis["report_sha256"],
        "official_elf_sha256": analysis["official_elf_sha256"],
        "receipt_sha256": analysis["receipt_sha256"],
        "observation": analysis["observation"],
        "valid_aggregate_metrics": scalars,
        "instance_validity": validity,
        "raw_source_sample_groups": dict(grouped),
        "raw_atomic_pc_diagnostics": atoms,
        "analysis_source_hashes": {
            name: digest(directory / name)
            for name in ("metrics.json", "instances.json", "pc_evidence.json", "summary.json")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "run_id": args.run_dir.name,
        "source_sha256": digest(Path(__file__)),
        "boundary": "Cited aggregate metrics require has_value=true. Per-PC and PM numbers with individual false flags remain raw diagnostics, not valid source-count percentages or timelines.",
        "captures": {
            tag: capture(args.run_dir / tag / "analysis")
            for tag in ("layer_2_zero_full", "layer_2_warm_full")
        },
    }
    cold, warm = result["captures"].values()
    for key in ("official_elf_sha256", "receipt_sha256"):
        require(cold[key] == warm[key], "Cold/warm binding differs")
    require(
        cold["observation"]["threshold_fp32_bits"]
        == warm["observation"]["threshold_fp32_bits"]
        == 0,
        "Expected the same positive-zero threshold",
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
