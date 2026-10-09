"""Check and measure complete Q1 projections with an explicit private scheduler."""

import argparse
import importlib.util
import json
import os
import statistics
import time
from dataclasses import fields
from pathlib import Path

import torch

from experiments.deepseek_v32_mfu.src.q1_control import digest, exact, graph, require
from models.deepseek_v32.checkpoint import CheckpointReader
from models.deepseek_v32.projections import CheckpointAttention

ROOT = Path(__file__).resolve().parents[3]


def identity(args):
    roots = [ROOT / "operators/deepseek_v32/linear", ROOT / "operators/deepseek_v32/norm"]
    paths = {path for root in roots for path in root.rglob("*.py") if "tests" not in path.parts}
    paths.update(
        ROOT / name
        for name in (
            "models/deepseek_v32/projections.py",
            "models/deepseek_v32/rotary.py",
            "models/deepseek_v32/nonmatrix.py",
            "models/deepseek_v32/checkpoint.py",
            "operators/deepseek_v32/indexer/quantization.py",
            "operators/flashinfer.py",
        )
    )
    paths.update((Path(__file__), Path(__file__).with_name("q1_control.py"), args.candidate))
    return {
        "sources": {str(path.resolve()): digest(path) for path in sorted(paths)},
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "affinity": sorted(os.sched_getaffinity(0)),
        "precision": torch.backends.cuda.matmul.fp32_precision,
    }


def same(a, b):
    for field in fields(a):
        exact(getattr(a, field.name), getattr(b, field.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), required=True)
    parser.add_argument("--model", type=Path, default=Path("/preset-models"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from triton.backends.nvidia.compiler import get_ptxas

    os.environ.setdefault("TRITON_PTXAS_PATH", get_ptxas(90).path)
    os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/usr/local/cuda/bin/ptxas")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.set_num_threads(8)
    spec = importlib.util.spec_from_file_location("_q1_projection_candidate", args.candidate)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    before = identity(args)
    if args.mode != "check":
        require(args.check is not None, "Missing independent check")
        receipt = json.loads(args.check.read_text())
        require(receipt["mode"] == "check" and receipt["passed"], "Invalid check")
        require(receipt["identity"] == before, "Source/input/runtime identity differs")
    reader = CheckpointReader(args.model)
    results = []
    with torch.inference_mode():
        for layer in range(3):
            print(f"Loading layer {layer}", flush=True)
            attention = CheckpointAttention(
                args.model, layer, device="cuda", reader=reader, linear_backend="fp8"
            )
            side = torch.cuda.Stream()
            torch.manual_seed(4200 + layer)
            x = torch.randn((1, attention.cfg.dim), device="cuda", dtype=torch.bfloat16)
            position = torch.full((1,), 65536, device="cuda", dtype=torch.float32)
            calls = {
                "baseline": lambda source, attn=attention, p=position: attn.project_positions(
                    source, p, normalized=True
                ),
                "overlap": lambda source, attn=attention, p=position, stream=side: (
                    candidate.project(attn, source, p, stream, normalized=True)
                ),
            }
            captures = {arm: graph(function, x) for arm, function in calls.items()}
            if args.mode == "check":
                for iteration in range(6):
                    x.mul_(0.75).add_(0.125)
                    position.fill_(65536 - iteration * 127)
                    for capture, _ in captures.values():
                        capture.replay()
                    torch.cuda.synchronize()
                    same(captures["baseline"][1], captures["overlap"][1])
                    same(captures["baseline"][1], calls["baseline"](x))
                path = args.output / f"layer_{layer}.pt"
                torch.save(
                    {
                        arm: {f.name: getattr(value, f.name).cpu() for f in fields(value)}
                        for arm, (_, value) in captures.items()
                    },
                    path,
                )
                results.append(
                    {
                        "layer": layer,
                        "exact_projected_fields": 6,
                        "changed_hidden_and_position_replays": 6,
                        "output": {"path": str(path), "sha256": digest(path)},
                    }
                )
            else:
                for capture, _ in captures.values():
                    for _ in range(3):
                        capture.replay()
                torch.cuda.synchronize()
                if args.mode == "profile":
                    torch.cuda.cudart().cudaProfilerStart()
                    # Node tracing is activated only at profiler start. Warm its
                    # graph instrumentation before selecting the measured replay.
                    for capture, _ in captures.values():
                        for _ in range(3):
                            capture.replay()
                    torch.cuda.synchronize()
                    for arm, (capture, _) in captures.items():
                        torch.cuda.nvtx.range_push(f"q1_projection/{arm}/layer{layer}")
                        capture.replay()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
                else:
                    samples = []
                    for pair in range(args.pairs):
                        for arm in (
                            ("baseline", "overlap") if pair % 2 == 0 else ("overlap", "baseline")
                        ):
                            torch.cuda.synchronize()
                            start, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            now = time.perf_counter_ns()
                            start.record()
                            for _ in range(args.iterations):
                                captures[arm][0].replay()
                            end.record()
                            end.synchronize()
                            samples.append(
                                {
                                    "pair": pair,
                                    "arm": arm,
                                    "event_us": start.elapsed_time(end) * 1000 / args.iterations,
                                    "wall_us": (time.perf_counter_ns() - now)
                                    / 1000
                                    / args.iterations,
                                }
                            )
                    summary = {
                        arm: statistics.median(r["event_us"] for r in samples if r["arm"] == arm)
                        for arm in captures
                    }
                    results.append({"layer": layer, "samples": samples, "median_us": summary})
                    print(summary, flush=True)
            torch.cuda.synchronize()
            del captures, calls, attention, x, position, side
    require(identity(args) == before, "Source/runtime changed")
    result = {
        "mode": args.mode,
        "passed": True,
        "identity": before,
        "results": results,
        "boundary": "Capture/replay of complete checkpoint projection, all six outputs, both stream branches and joins included; normalized BF16 Q1 synthetic hidden and real L0-L2 weights.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "mode": args.mode, "output": str(args.output)}))


if __name__ == "__main__":
    main()
