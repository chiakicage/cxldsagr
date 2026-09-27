"""Select compact, report-ready evidence from launch analysis and CIS counters."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

METRICS = {
    "Duration": ("duration_us", "us"),
    "Grid Size": ("ctas", ""),
    "Waves Per SM": ("waves_per_sm", ""),
    "DRAM Throughput": ("dram_pct", "%"),
    "L2 Cache Throughput": ("l2_pct", "%"),
    "Compute (SM) Throughput": ("sm_pct", "%"),
    "Achieved Occupancy": ("occupancy_pct", "%"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument(
        "--cis-only", action="store_true", help="Export CIS evidence without a model timeline"
    )
    args = parser.parse_args(argv)
    data = args.data_dir
    cis = json.loads((data / "cis_bottleneck.json").read_text())
    scoring = args.source_data_dir / "sources/models/nosa/scoring.py"
    if (
        hashlib.sha256(scoring.read_bytes()).hexdigest()
        != cis["source_sha256"]["models/nosa/scoring.py"]
    ):
        raise ValueError("CIS implementation differs from the source run")
    kernels = {}
    with (data / "ncu.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["Metric Name"] not in METRICS:
                continue
            key, unit = METRICS[row["Metric Name"]]
            if row["Metric Unit"] != unit:
                raise ValueError(f"Unexpected Nsight unit for {key}")
            kernel = kernels.setdefault(
                row["ID"], {"id": int(row["ID"]), "name": row["Kernel Name"]}
            )
            kernel[key] = float(row["Metric Value"].replace(",", ""))
    if len(kernels) != 6 or any(len(row) != len(METRICS) + 2 for row in kernels.values()):
        raise ValueError("Expected all selected metrics for six original CIS kernels")
    evidence = {
        "analysis_run_id": cis["run_id"],
        "cis": cis,
        "ncu": list(kernels.values()),
        "ncu_scope": "One original CIS call, 10 kernel-replay passes, cache-control=none, clock-control=none; counters are not unprofiled latency",
    }
    if args.cis_only:
        output_name = "cis_evidence.json"
    else:
        launch = json.loads((data / "launch_analysis.json").read_text())
        evidence.update(source_run_id=launch["source_run_id"], timeline=launch)
        output_name = "bottleneck_evidence.json"
    (data / output_name).write_text(json.dumps(evidence, indent=2) + "\n")
    with (data / "cis_counters.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(kernels.values()))))
        writer.writeheader()
        writer.writerows(kernels.values())
    print(f"Evidence: {data / output_name}")


if __name__ == "__main__":
    main()
