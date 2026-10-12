"""Receipt-bound invasive HBM profiling; never supplies clean timing samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from evaluation.validation import require_receipt
from experiments.deepseek_v32_observation.src.bandwidth import METRICS, from_metrics

PROFILE_PREFIX = "observation_hbm__"


def capture(call, *, inner, warmup, before_call=None):
    import torch

    device = torch.cuda.current_device()
    if before_call is not None and inner != 1:
        raise ValueError("Mutable input reset requires one profiled call per graph")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            if before_call is not None:
                before_call()
            output = call()
    stream.synchronize()
    if before_call is not None:
        before_call()
        torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(inner):
            output = call()
    stream.synchronize()
    for _ in range(warmup):
        if before_call is not None:
            before_call()
            torch.cuda.synchronize(device)
        graph.replay()
        torch.cuda.synchronize(device)
    return graph, output, stream


def execute(args):
    import torch

    from experiments.deepseek_v32_observation.src.provenance import (
        execution_identity,
        require_runtime_match,
        runtime_evidence,
        save_json,
    )
    from experiments.deepseek_v32_observation.src.workload_entry import (
        planned_inner,
        prepare,
        refresh_attention,
    )

    owner, cases, identity, _, key = prepare(args)
    receipt = require_receipt(
        args.receipt,
        kind=f"deepseek-v32-observation-{args.group}",
        identity=identity,
    )
    accepted_runtime = json.loads(Path(receipt["artifact_paths"]["runtime"]).read_text())
    records = []
    for case in cases:
        refresh_attention(owner, args.group)
        inner = planned_inner(case, args.inner)
        before_call = case.get("before_call")
        graph, output, stream = capture(
            case["call"],
            inner=inner,
            warmup=args.warmup,
            before_call=before_call,
        )
        if before_call is not None:
            before_call()
            torch.cuda.synchronize()
        range_name = PROFILE_PREFIX + case["name"]
        torch.cuda.nvtx.range_push(range_name)
        graph.replay()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        records.append({"name": case["name"], "range_name": range_name, "inner": inner})
        print(f"PROFILE {args.group}/{key}/{case['name']}: inner={inner}", flush=True)
        del graph, output
        stream.synchronize()
    after = execution_identity(
        group=args.group,
        key=key,
        seed=args.seed,
        checkpoint=args.checkpoint,
        operators=identity["operators"],
        workload_identity=identity["input"],
    )
    if after != identity:
        raise RuntimeError("Execution source or checkpoint identity changed during profiling")
    runtime = runtime_evidence()
    runtime["acceptance_match"] = require_runtime_match(accepted_runtime, runtime)
    version = subprocess.run(
        ["/usr/local/cuda/bin/ncu", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    save_json(
        args.output,
        {
            "key": key,
            "group": args.group,
            "identity": identity,
            "receipt_path": receipt["receipt_path"],
            "runtime": runtime,
            "operators": records,
            "warmup": args.warmup,
            "profiler": {
                "executable": "/usr/local/cuda/bin/ncu",
                "version": version.stdout,
                "metrics": list(METRICS),
                "replay_mode": "kernel",
                "graph_profiling": "graph",
                "cache_control": "none",
                "clock_control": "none",
                "profile_graph_external_events": False,
            },
            "boundary": "Independent intrusive CUDA Graph profile; no numerical checks in profile ranges",
        },
    )
    owner.close()


def _ncu_module(directory):
    # Explicit version path: failure is propagated rather than trying a second
    # provider or loading an arbitrary ABI from an unrelated CUDA installation.
    sys.path.insert(0, str(directory))
    import ncu_report

    return ncu_report


def _range_name(action):
    state = action.nvtx_state()
    if state is None:
        raise ValueError("Profile action has no operator NVTX identity")
    names = []
    for domain_id in state.domains():
        domain = state.domain_by_id(domain_id)
        names.extend(name for name in domain.push_pop_ranges() if name.startswith(PROFILE_PREFIX))
    if len(names) != 1:
        raise ValueError(f"Ambiguous operator profile NVTX ranges: {names}")
    return names[0]


def parse_report(args):
    from experiments.deepseek_v32_observation.src.provenance import save_json

    execution = json.loads(args.execution.read_text())
    expected = {record["range_name"]: record for record in execution["operators"]}
    module = _ncu_module(args.ncu_python)
    report = module.load_report(str(args.report))
    actual = {}
    with args.report.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    for range_index in range(report.num_ranges()):
        report_range = report.range_by_idx(range_index)
        for action_index in range(report_range.num_actions()):
            action = report_range.action_by_idx(action_index)
            if action.name() != "graph":
                raise ValueError("HBM observation requires one whole-graph action per operator")
            name = _range_name(action)
            if name not in expected or name in actual:
                raise ValueError(f"Duplicate or unexpected HBM operator: {name}")
            metrics = {
                name: action[name].value() for name in (*METRICS, "profiler__replayer_passes")
            }
            units = ("byte", "byte", "ns", "byte/s", "%", "pass")
            for metric, unit in zip((*METRICS, "profiler__replayer_passes"), units, strict=True):
                if action[metric].unit() != unit:
                    raise ValueError(f"Unexpected raw NCU metric unit for {metric}")
            actual[name] = {
                "name": expected[name]["name"],
                **from_metrics(
                    metrics,
                    inner=expected[name]["inner"],
                    profile_path=args.report.resolve(),
                    profile_sha256=sha256,
                    replay_mode=args.replay_mode,
                    time_boundary=args.time_boundary,
                ),
            }
    if set(actual) != set(expected):
        raise ValueError(f"Missing HBM operators: {sorted(set(expected) - set(actual))}")
    result = {
        key: execution[key] for key in ("key", "group", "identity", "receipt_path", "runtime")
    }
    result.update(
        operators=[actual[name] for name in expected],
        execution_path=str(args.execution.resolve()),
        ncu_report_path=str(args.report.resolve()),
        ncu_report_sha256=sha256,
        profiler=execution["profiler"],
    )
    save_json(args.output, result)
    print(f"PARSED {len(actual)} actual HBM operator profiles", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--group", choices=("common", "moe", "attention"), required=True)
    for name in ("S", "B", "Q"):
        capture_parser.add_argument(f"--{name}", type=int, default=1)
    capture_parser.add_argument("--H", type=int, default=4096)
    capture_parser.add_argument("--device", type=int, default=0)
    capture_parser.add_argument("--seed", type=int, default=42)
    capture_parser.add_argument("--checkpoint", type=Path, default=Path("/preset-models"))
    capture_parser.add_argument("--warmup", type=int, default=5)
    capture_parser.add_argument("--inner", type=int, default=20)
    capture_parser.add_argument("--receipt", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    parse_parser = sub.add_parser("parse")
    parse_parser.add_argument("--execution", type=Path, required=True)
    parse_parser.add_argument("--report", type=Path, required=True)
    parse_parser.add_argument("--output", type=Path, required=True)
    parse_parser.add_argument(
        "--ncu-python", type=Path, default=Path("/opt/nvidia/nsight-compute/2026.1.1/extras/python")
    )
    parse_parser.add_argument("--replay-mode", required=True)
    parse_parser.add_argument("--time-boundary", required=True)
    args = parser.parse_args()
    if args.mode == "capture":
        import torch

        with torch.inference_mode():
            execute(args)
    else:
        parse_report(args)


if __name__ == "__main__":
    main()
