"""Verify Q1 layout traces by NVTX/CUDA correlation, without kernel-position guesses."""

import argparse
import json
import sqlite3
from pathlib import Path

from experiments.deepseek_v32_mfu.src.q1_control import digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        ranges = [
            dict(row)
            for row in db.execute("SELECT * FROM NVTX_EVENTS")
            if (row["text"] or strings.get(row["textId"], "")).startswith("q1_control/")
        ]
        require(len(ranges) == 32, "Expected two APIs and two arms for eight shapes")
        rows = []
        for scope in ranges:
            label = scope["text"] or strings[scope["textId"]]
            _, arm, mode, shape = label.split("/")
            apis = list(
                db.execute(
                    "SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                    "WHERE start>=? AND end<=? AND globalTid=?",
                    (scope["start"], scope["end"], scope["globalTid"]),
                )
            )
            correlations = {row[0] for row in apis}
            kernels = [
                dict(row)
                for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL")
                if row["correlationId"] in correlations
            ]
            for row in kernels:
                row["name"] = strings[row["demangledName"]]
            require(len(kernels) == (3 if arm == "baseline" else 2), label + ": wrong kernel count")
            gemm = [row for row in kernels if "sm90_fp8_gemm_1d2d_impl" in row["name"]]
            transpose = [row for row in kernels if "transpose_fp32" in row["name"]]
            require(len(gemm) == 1, label + ": missing original GEMM")
            require(
                len(transpose) == (1 if arm == "baseline" else 0), label + ": wrong transpose count"
            )
            rows.append(
                {
                    "arm": arm,
                    "mode": mode,
                    "shape": shape,
                    "kernel_count": len(kernels),
                    "transpose_count": len(transpose),
                    "kernel_ns": sum(row["end"] - row["start"] for row in kernels),
                    "window_ns": max(row["end"] for row in kernels)
                    - min(row["start"] for row in kernels),
                    "kernels": [
                        {
                            key: row[key]
                            for key in (
                                "name",
                                "start",
                                "end",
                                "graphId",
                                "graphNodeId",
                                "correlationId",
                            )
                        }
                        for row in kernels
                    ],
                }
            )
        for mode in ("eager", "graph"):
            for shape in {row["shape"] for row in rows}:
                selected = [row for row in rows if row["mode"] == mode and row["shape"] == shape]
                symbols = [
                    next(
                        kernel["name"]
                        for kernel in row["kernels"]
                        if "sm90_fp8_gemm_1d2d_impl" in kernel["name"]
                    )
                    for row in selected
                ]
                require(len(symbols) == 2 and symbols[0] == symbols[1], "GEMM changed between arms")
    result = {
        "passed": True,
        "sqlite": {"path": str(args.sqlite), "sha256": digest(args.sqlite)},
        "analyzer": {"path": __file__, "sha256": digest(__file__)},
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "scopes": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
