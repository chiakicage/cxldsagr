"""Independent check / clean benchmark for one observation workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_observation.src.contract import GROUPS, mfu
from experiments.deepseek_v32_observation.src.graph_validation import validate_graph
from experiments.deepseek_v32_observation.src.provenance import (
    execution_identity,
    require_runtime_match,
    runtime_evidence,
    save_json,
)
from experiments.deepseek_v32_observation.src.timing import benchmark
from experiments.deepseek_v32_observation.src.workload_entry import planned_inner, prepare


def execute(args):
    owner, cases, identity, shape, key = prepare(args)
    input_identity = identity["input"]
    receipt_kind = f"deepseek-v32-observation-{args.group}"
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "check":
        if args.group == "attention":
            checks = owner.validate()
        else:
            checks = {"passed": True, "operators": {}}
            for case in cases:
                if case.get("before_call") is not None:
                    case["before_call"]()
                actual = case["call"]()
                torch.cuda.synchronize()
                checks["operators"][case["name"]] = case["verify"](actual)
                del actual
        checks["static_graph"] = {}
        for case in cases:
            if args.group == "attention":
                owner.run()
                torch.cuda.synchronize()
            transform = owner.graph_transform(case["name"]) if args.group == "attention" else None
            if args.group == "attention" and case["name"] == "indexer":
                # Native score tail bytes are undefined before the explicit
                # causal cleanup. Compare every visible score, not padding.
                positions = torch.arange(owner.score_width, device=owner.device)
                visible = positions[None] < owner.ends[:, None]
                transform = lambda scores, visible=visible: scores[visible]
            checks["static_graph"][case["name"]] = validate_graph(
                case["call"], transform=transform, before_call=case.get("before_call")
            )
        save_json(args.output / "checks.json", checks)
        runtime = runtime_evidence()
        save_json(args.output / "runtime.json", runtime)
        write_receipt(
            args.output / "receipt.json",
            kind=receipt_kind,
            identity=identity,
            checks=checks,
            artifacts={
                "checks": args.output / "checks.json",
                "runtime": args.output / "runtime.json",
            },
        )
        print(f"CHECK {args.group}/{key}: {len(cases)} operators passed", flush=True)
    else:
        receipt = require_receipt(args.receipt, kind=receipt_kind, identity=identity)
        checked_runtime = json.loads(Path(receipt["artifact_paths"]["runtime"]).read_text())
        observed = []
        for case in cases:
            if args.group == "attention":
                # Standalone operation graph capture changes owner-held output
                # references; reestablish all dependencies before the next API.
                owner.run()
                torch.cuda.synchronize()
            # Packing duplicates the full resident indexer cache. Plan one API
            # per graph for this operation before allocation, avoiding B1024
            # graph temporaries that would exceed H200 capacity at inner=20.
            case_inner = planned_inner(case, args.inner)
            timing = benchmark(
                case["call"],
                warmup=args.warmup,
                repeats=args.repeats,
                inner=case_inner,
                before_call=case.get("before_call"),
            )
            row = {
                "name": case["name"],
                "work": case["work"],
                "metadata": case["metadata"],
                "scope": case["metadata"].get("scope", "operator"),
                **timing,
                **mfu(case["work"], timing["time_us"]),
                "requested_inner": args.inner,
                "inner_policy": "indexer_prepare/page64_pack and mutable input-reset leaves=1; others=requested_inner",
            }
            observed.append(row)
            print(f"BENCH {args.group}/{key}/{case['name']}: {row['time_us']:.3f} us", flush=True)
        after = execution_identity(
            group=args.group,
            key=key,
            seed=args.seed,
            checkpoint=args.checkpoint,
            operators=identity["operators"],
            workload_identity=input_identity,
        )
        if after != identity:
            raise RuntimeError("Execution source or checkpoint identity changed during measurement")
        runtime = runtime_evidence()
        runtime["acceptance_match"] = require_runtime_match(checked_runtime, runtime)
        record = {
            "key": key,
            "group": args.group,
            "S": shape.S if args.group == "attention" else args.S,
            "B": shape.B if args.group == "attention" else None,
            "Q": shape.Q if args.group == "attention" else None,
            "H": shape.H if args.group == "attention" else None,
            "receipt_path": receipt["receipt_path"],
            "identity": identity,
            "runtime": runtime,
            "operators": observed,
        }
        save_json(args.output / "result.json", record)
    owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench"), required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--S", type=int, default=1)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--Q", type=int, default=1)
    parser.add_argument("--H", type=int, default=4096)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("/preset-models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        execute(args)


if __name__ == "__main__":
    main()
