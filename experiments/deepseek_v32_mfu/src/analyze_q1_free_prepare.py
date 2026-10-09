"""Audit accepted preparation component samples, source archives and profiler coverage."""

import argparse
import json
import sqlite3
import statistics
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt
from experiments.deepseek_v32_mfu.src.q1_free_prepare import KIND
from experiments.deepseek_v32_mfu.src.q1_prepare_baseline import digest, require, write


def audit_run(path, identity):
    result = json.loads((path / "result.json").read_text())
    require(result["passed"] is True, "Run did not pass: " + str(path))
    require(result["identity"] == identity, "Source/native/input identity differs: " + str(path))
    sources = identity["source"]["files"]
    for filename, expected in sources.items():
        archived = path / "source" / Path(filename).relative_to("/")
        require(digest(archived) == expected, "Archived source differs: " + str(archived))
    require(
        digest(path / "input_evidence.pt") == identity["input_sha256"], "Archived input differs"
    )
    require(
        digest(path / "candidate.so") == identity["native"]["candidate"]["artifact_sha256"],
        "Archived candidate differs",
    )
    require(
        digest(path / "baseline.so") == identity["native"]["baseline"][0]["library"]["sha256"],
        "Archived baseline differs",
    )
    return {
        "path": str(path.resolve()),
        "mode": result["mode"],
        "source_files_verified": len(sources),
        "result_sha256": digest(path / "result.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--nsys-data", type=Path, required=True)
    parser.add_argument("--nsys-report", type=Path, required=True)
    parser.add_argument("--ncu-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    identity = json.loads(args.receipt.read_text())["identity"]
    receipt = require_receipt(args.receipt, kind=KIND, identity=identity)
    runs = [args.receipt.parent, args.bench, args.nsys_data / "candidate_cold"]
    runs += [
        args.ncu_data / case / collection
        for case in ("cold", "scattered")
        for collection in ("full", "source")
    ]
    audit = [audit_run(path, identity) for path in runs]
    bench = json.loads((args.bench / "result.json").read_text())
    rows = []
    for case in bench["cases"]:
        samples = case["samples"]
        require(len(samples) == 100, "Expected exactly 100 pairs")
        for index, sample in enumerate(samples):
            require(sample["pair"] == index, "Nonsequential pairs")
            require(
                sample["order"]
                == (["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]),
                "Unbalanced AB/BA",
            )
            require(all(value > 0 for value in sample["us"].values()), "Invalid latency")
        medians = {
            variant: statistics.median(sample["us"][variant] for sample in samples)
            for variant in ("baseline", "candidate")
        }
        wins = sum(sample["us"]["candidate"] < sample["us"]["baseline"] for sample in samples)
        require(
            medians == case["median_us"] and wins == case["paired_wins"], "Derived timing mismatch"
        )
        rows.append(
            {
                "state": case["state"],
                "P": case["P"],
                "H": case["H"],
                "median_us": medians,
                "candidate_paired_wins": wins,
                "speedup": medians["baseline"] / medians["candidate"],
                "median_paired_saving_us": statistics.median(
                    sample["us"]["baseline"] - sample["us"]["candidate"] for sample in samples
                ),
                "actual_dispatch_shape": case["H"] + 1 >= 32768,
            }
        )
    sql = args.nsys_data / "candidate_cold.sqlite"
    with sqlite3.connect(f"file:{sql.resolve()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        kernels = [
            dict(row)
            for row in connection.execute(
                "SELECT k.start, k.end, k.streamId, k.correlationId, s.value AS name "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"
            )
        ]
        expected = ("prepare_masks", "select_slots", "reset_publish")
        require(len(kernels) == len(expected), "NSYS must contain exactly three kernels")
        for row, fragment in zip(kernels, expected, strict=True):
            require(fragment in row["name"], "NSYS kernel order mismatch")
        require(len({row["streamId"] for row in kernels}) == 1, "Preparation streams differ")
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        transfers = {}
        for name in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
            transfers[name] = (
                connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                if name in tables
                else 0
            )
            require(
                transfers[name] == 0, "Unexpected memory activity in complete candidate profile"
            )
        ranges = [
            dict(row)
            for row in connection.execute(
                "SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'q1_free_prepare/%'"
            )
        ]
        require(
            len(ranges) == 1 and ranges[0]["text"].endswith("candidate/saved_cold"),
            "Unexpected complete call NVTX",
        )
        require(
            all(
                ranges[0]["start"] <= row["start"] < row["end"] <= ranges[0]["end"]
                for row in kernels
            ),
            "Kernel falls outside complete call range",
        )
    ncu = {}
    for case in ("cold", "scattered"):
        summary_path = args.ncu_data / case / "analysis/summary.json"
        summary = json.loads(summary_path.read_text())
        require(summary["variant"] == "candidate", "Unexpected NCU variant")
        for report in summary["reports"].values():
            require(digest(report["path"]) == report["sha256"], "NCU report differs")
            require(len(report["kernels"]) == 3, "NCU kernel count differs")
        ncu[case] = {
            "summary": str(summary_path.resolve()),
            "summary_sha256": digest(summary_path),
            "full_kernel_us": [
                {
                    "name": row["name"],
                    "us": row["metrics"]["gpu__time_duration.sum"]["value"] / 1000,
                }
                for row in summary["reports"]["full"]["kernels"]
            ],
        }
    write(
        args.output / "audit.json",
        {
            "passed": True,
            "analyzer_sha256": digest(__file__),
            "receipt_sha256": receipt["receipt_sha256"],
            "execution_identity_sha256": identity_digest(identity),
            "runs": audit,
            "timing": rows,
            "nsys": {
                "report": str(args.nsys_report.resolve()),
                "report_sha256": digest(args.nsys_report),
                "sqlite_sha256": digest(sql),
                "kernels": kernels,
                "memory_activities": transfers,
                "nvtx": ranges,
                "kernel_sum_us": sum(row["end"] - row["start"] for row in kernels) / 1000,
                "kernel_span_us": (kernels[-1]["end"] - kernels[0]["start"]) / 1000,
            },
            "ncu": ncu,
            "boundary": "Private component only. CUDA graph event timing covers all prepare GPU work; Python dispatch and restoration are outside. NSYS and NCU are independent intrusive profiles; their kernel durations are not substituted for clean timing. Minimum64 is component edge coverage, outside the intended production dispatch shape.",
        },
    )
    print(args.output / "audit.json")


if __name__ == "__main__":
    main()
