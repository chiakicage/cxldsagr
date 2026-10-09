"""Extract source-bound NCU evidence for the unchanged official fused core."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from evaluation.validation import require_receipt

EXPERIMENT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def metric_record(metric, index=None):
    args = () if index is None else (index,)
    return {"has_value": metric.has_value(*args), "value": metric.value(index)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    data = EXPERIMENT / "output/data" / args.run_id / args.tag
    report_path = EXPERIMENT / "output/profile" / args.run_id / (args.tag + ".ncu-rep")
    passes = sorted((data / "passes").glob("pass-*"))
    require(len(passes) == 1, "Expected one application with kernel replay")
    component_dir = passes[0] / "component"
    component = json.loads((component_dir / "summary.json").read_text())
    receipt = require_receipt(
        args.receipt, kind="deepseek-private-q1-fused-prepare-v1", identity=component["identity"]
    )
    require(component["receipt_sha256"] == receipt["receipt_sha256"], "Receipt signature differs")
    require(component["variant"] == "baseline", "Expected the original baseline preparation")
    observation = json.loads((passes[0] / "observation.json").read_text())
    require(component["observation"] == observation, "Observation differs from completion")
    for name, expected in component["runtime_archives"].items():
        require(digest(component_dir / name) == expected, "Archived runtime bytes changed")
    observer = passes[0] / "q1_official_core_profile.py"
    require(digest(observer) == observation["observer_sha256"], "Archived observer changed")
    sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
    import ncu_report

    report = ncu_report.load_report(str(report_path))
    require(report.num_ranges() == 1, "Expected one NCU range")
    region = report.range_by_idx(0)
    require(region.num_actions() == 1, "Expected exactly one profiled kernel")
    require(
        tuple(region.actions_by_nvtx(["q1_fused_prepare_complete_baseline/"], [])) == (0,),
        "Profile is outside the accepted baseline call",
    )
    action = region.action_by_idx(0)
    require("sm90_fp8_paged_mqa_logits_fused_v2" in action.name(), "Unexpected core kernel")
    scalars, instances, locations = {}, {}, defaultdict(dict)
    for name in sorted(action.metric_names()):
        metric = action[name]
        scalars[name] = {
            **metric_record(metric),
            "unit": metric.unit(),
            "description": metric.description(),
            "instances": metric.num_instances(),
        }
        if not metric.num_instances():
            continue
        correlation = metric.correlation_ids() if metric.has_correlation_ids() else None
        require(
            correlation is None or correlation.num_instances() == metric.num_instances(),
            "Metric correlation length differs",
        )
        rows = []
        for index in range(metric.num_instances()):
            row = metric_record(metric, index)
            row["correlation"] = None if correlation is None else correlation.value(index)
            rows.append(row)
            if (
                correlation is not None
                and row["value"]
                and (
                    name.startswith("smsp__pcsamp_warps_issue_stalled")
                    or name
                    in ("inst_executed", "thread_inst_executed", "thread_inst_executed_true")
                )
            ):
                locations[row["correlation"]][name] = row["value"]
        instances[name] = rows
    for name, expected in (("launch__grid_size", 132), ("launch__block_size", 768)):
        require(scalars[name]["value"] == expected, "Unexpected launch geometry")
    pc_rows = []
    for pc, counters in sorted(locations.items()):
        source = action.source_info(pc)
        pc_rows.append(
            {
                "absolute_pc": pc,
                "sass": action.sass_by_pc(pc),
                "source": None
                if source is None
                else {"file": source.file_name(), "line": source.line()},
                "counters": counters,
            }
        )
    destination = data / "analysis"
    destination.mkdir(exist_ok=False)
    write(destination / "metrics.json", scalars)
    write(destination / "instances.json", instances)
    write(destination / "pc_evidence.json", pc_rows)
    write(destination / "rules.json", action.rule_results_as_dicts())
    summary = {
        "report": str(report_path),
        "report_sha256": digest(report_path),
        "kernel": action.name(),
        "metric_count": len(scalars),
        "pc_locations": len(pc_rows),
        "observation": observation,
        "receipt_sha256": receipt["receipt_sha256"],
        "official_elf_sha256": component["identity"]["official_native"]["artifact_sha256"],
        "parser": str(Path(ncu_report.__file__).resolve()),
        "analyzer_sha256": digest(Path(__file__)),
        "boundary": "Invasive core-only NCU, policy-restored kernel replay with cache flush and base clock control; not clean timing. The observation describes the application-visible completion, not every internal replay pass.",
        "selected_metrics": {
            name: row
            for name, row in scalars.items()
            if name.startswith(
                (
                    "gpu__time_duration",
                    "l1tex__t_requests_pipe_lsu_mem_global_op_atom",
                    "lts__t_sectors_op_atom",
                    "lts__t_sectors_aperture_sysmem_op_read",
                    "smsp__sass_inst_executed_op_global_atom",
                    "sm__cycles_active",
                    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
                )
            )
        },
    }
    write(destination / "summary.json", summary)
    (destination / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"analysis": str(destination), "metrics": len(scalars), "pcs": len(pc_rows)}))


if __name__ == "__main__":
    main()
