"""Compare aggregate sampled matrix-API MFU with matching formal stage latency.

Reads operator_mfu.csv/operator_mfu_summary.json and stage_mfu.csv/flops.json.
Per-call GPU activity unions are summed; this is not total pipeline GPU time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

PHASE_VISITS = {"cold": "first", "revisit": "revisit"}
SEGMENT_STAGES = {"history": "prefix", "candidate": "extend", "request": "e2e"}
SUM_FIELDS = ("calls", "useful_flops", "ideal_ms", "operator_gpu_active_ms")


def number(row, key, *, positive=False):
    value = float(row[key])
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {key}: {row[key]}")
    return value


def totals(rows):
    result = {key: sum(number(row, key, positive=True) for row in rows) for key in SUM_FIELDS}
    result["calls"] = int(result["calls"])
    result["useful_flops"] = int(result["useful_flops"])
    result["operator_gpu_active_mfu_pct"] = (
        100 * result["ideal_ms"] / result["operator_gpu_active_ms"]
    )
    return result


def aggregate(operator_rows, formal_rows):
    """Keep real operator stages; never force ECHO calls into a fused category."""
    if not operator_rows or not formal_rows:
        raise ValueError("operator and formal stage rows must be nonempty")
    formal = {}
    for row in formal_rows:
        key = row["scheme"], row["visit_kind"], row["stage"]
        if key in formal:
            raise ValueError(f"duplicate formal stage: {key}")
        formal[key] = row
    captures, operators = defaultdict(list), defaultdict(list)
    identities = set()
    for row in operator_rows:
        if row["phase"] not in PHASE_VISITS or row["segment"] not in ("history", "candidate"):
            raise ValueError("unknown profile phase or segment")
        identity = tuple(
            row[key]
            for key in ("capture_index", "scheme", "phase", "segment", "stage", "precision")
        )
        if identity in identities:
            raise ValueError(f"duplicate operator group: {identity}")
        identities.add(identity)
        for key in ("calls", "useful_flops"):
            value = number(row, key, positive=True)
            if not value.is_integer():
                raise ValueError(f"noninteger {key}")
        capture = tuple(row[key] for key in ("scheme", "phase", "capture_index"))
        captures[*capture, row["segment"]].append(row)
        captures[*capture, "request"].append(row)
        operators[
            tuple(row[key] for key in ("scheme", "phase", "segment", "stage", "precision"))
        ].append(row)

    stage_totals = [
        {
            **dict(zip(("scheme", "phase", "segment", "stage", "precision"), key, strict=True)),
            "captures": len({row["capture_index"] for row in rows}),
            **totals(rows),
        }
        for key, rows in sorted(operators.items())
    ]
    groups = defaultdict(list)
    for (scheme, phase, capture_index, segment), rows in captures.items():
        groups[scheme, phase, segment].append({"capture_index": capture_index, **totals(rows)})
    comparison = []
    for (scheme, phase, segment), rows in sorted(groups.items()):
        key = scheme, PHASE_VISITS[phase], SEGMENT_STAGES[segment]
        if key not in formal:
            raise ValueError(f"missing matching formal latency: {key}")
        measured = formal[key]
        ideal = number(measured, "ideal_compute_mean_ms", positive=True)
        # Equal work is required in every capture, not only on average. Missing
        # history/candidate matrix rows must not produce an artificially low API total.
        if any(
            not math.isclose(row["ideal_ms"], ideal, rel_tol=1e-8, abs_tol=1e-9) for row in rows
        ):
            raise ValueError(f"profile/formal matrix work differs: {key}")
        total = totals(rows)
        api_mean = total["operator_gpu_active_ms"] / len(rows)
        wall_mean = number(measured, "wall_mean_ms", positive=True)
        formal_mfu = 100 * ideal / wall_mean
        if not math.isclose(number(measured, "effective_mfu_pct"), formal_mfu, rel_tol=1e-8):
            raise ValueError(f"formal MFU arithmetic differs: {key}")
        comparison.append(
            {
                "scheme": scheme,
                "phase": phase,
                "segment": segment,
                "formal_stage": key[2],
                "captures": len(rows),
                "formal_requests": int(number(measured, "requests", positive=True)),
                **total,
                "ideal_mean_ms": ideal,
                "operator_gpu_active_mean_ms": api_mean,
                "formal_wall_mean_ms": wall_mean,
                "formal_mfu_pct": formal_mfu,
                "formal_wall_to_operator_api_ratio": wall_mean / api_mean,
                "formal_mfu_to_operator_mfu_ratio": api_mean / wall_mean,
                "formal_wall_minus_operator_api_ms": wall_mean - api_mean,
            }
        )
    return comparison, stage_totals


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(operator_dir, flops_dir, output_dir):
    operator_dir, flops_dir, output = map(Path, (operator_dir, flops_dir, output_dir))
    if output.exists():
        raise FileExistsError(output)
    operator_summary = json.loads((operator_dir / "operator_mfu_summary.json").read_text())
    flops = json.loads((flops_dir / "flops.json").read_text())
    if operator_summary["formal_run_id"] != flops["source_run_id"]:
        raise ValueError("operator and formal analyses refer to different runs")
    if operator_summary["dense_peaks_tflops"] != flops["dense_peaks_tflops"]:
        raise ValueError("operator and formal analyses use different reference peaks")
    operator_rows = read_csv(operator_dir / "operator_mfu.csv")
    if sum(int(row["calls"]) for row in operator_rows) != operator_summary["matrix_calls"]:
        raise ValueError("operator CSV does not conserve declared matrix calls")
    comparison, stages = aggregate(operator_rows, read_csv(flops_dir / "stage_mfu.csv"))
    payload = {
        "schema": "deepseek-v32-motivation-aggregate-mfu-v1",
        "analysis_run_id": output.name,
        "formal_run_id": flops["source_run_id"],
        "profile_run_id": operator_summary["profile_run_id"],
        "operator_analysis_run_id": operator_summary["analysis_run_id"],
        "dense_peaks_tflops": flops["dense_peaks_tflops"],
        "precision_evidence": flops.get("precision_evidence"),
        "comparison": comparison,
        "operator_stages": stages,
        "definitions": {
            "operator_mfu": "100 * sum(ideal_ms) / sum(per-call GPU activity union ms)",
            "formal_wall_to_operator_api_ratio": "formal mean wall ms / sampled mean matrix API GPU-active ms",
            "formal_mfu_to_operator_mfu_ratio": "formal MFU / sampled aggregate matrix API MFU",
            "formal_wall_minus_operator_api_ms": "Difference between separate formal and instrumented measurements; not isolated CPU cost.",
        },
        "boundaries": [
            "API activity includes attributed matrix helpers, excluding API-external selection, norms, copies, cache control and launch gaps.",
            "Sampled per-call unions are summed, not unioned across APIs; concurrent APIs can overlap.",
            "Profile captures and formal request means are different measurements. Their gap is diagnostic, not a guaranteed removable cost or achievable speedup.",
            "Every captured segment and full request must match the formal mean matrix work. Operator stages and schemes are taken from data.",
            "Input analysis identities, reference peaks, work and arithmetic are checked; this aggregation does not rerun numerical or profiler-attribution acceptance.",
        ],
        "input_sha256": {
            str(path.resolve()): digest(path)
            for path in (
                operator_dir / "operator_mfu.csv",
                operator_dir / "operator_mfu_summary.json",
                flops_dir / "stage_mfu.csv",
                flops_dir / "flops.json",
            )
        },
        "analysis_source_sha256": {str(Path(__file__).resolve()): digest(__file__)},
    }
    output.mkdir(parents=True)
    write_csv(output / "aggregate_mfu.csv", comparison)
    write_csv(output / "operator_stage_totals.csv", stages)
    (output / "aggregate_mfu.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-dir", type=Path, required=True)
    parser.add_argument("--flops-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.operator_dir, args.flops_dir, args.output_dir)
    print(f"Aggregate matrix API / formal MFU comparison: {args.output_dir}")


if __name__ == "__main__":
    main()
