"""Independent acceptance, full-API timing, and single-call Q1 attention profile.

The prefill and decode production APIs are called separately. Saved model
inputs came from an extra eager diagnostic forward, outside the original
profile capture; this experiment binds their bytes independently.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from experiments.deepseek_v32_echo_official.src.q1_attention_dense import metrics
from experiments.deepseek_v32_echo_official.src.q1_indexer import (
    DEFAULT_INPUT,
    digest,
    receipt_matches,
    require,
)

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]


def configure_environment():
    for name, directory in (
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("TMPDIR", "tmp"),
    ):
        path = EXPERIMENT / "output/runtime/q1-attention" / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["TRITON_DISABLE_LINE_INFO"] = "0"


def source_paths():
    attention = ROOT / "operators/deepseek_v32/attention"
    return [
        Path(__file__).resolve(),
        Path(__file__).with_name("q1_attention_dense.py").resolve(),
        Path(__file__).with_name("q1_indexer.py").resolve(),
        attention / "device_only/mla.py",
        attention / "device_only/decode.py",
        attention / "reference/torch.py",
        attention / "_config.py",
        attention / "_validation.py",
    ]


def runtime_identity():
    decode = importlib.import_module("operators.deepseek_v32.attention.device_only.decode")
    runtime = decode.runtime_info()
    require(runtime is not None and runtime.get("specializations"), "Decode was not compiled")
    return json.loads(json.dumps(runtime))


def identity(args, torch, mla):
    import triton

    return json.loads(
        json.dumps(
            {
                "inputs_sha256": {str(path.resolve()): digest(path) for path in args.inputs},
                "source_sha256": {
                    str(path.relative_to(ROOT)): digest(path) for path in source_paths()
                },
                "attention_backend": mla.build_info(),
                "compiled_attention": runtime_identity(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "gpu_uuid": subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        str(args.physical_device),
                        "--query-gpu=uuid",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip(),
                "physical_device": args.physical_device,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "environment": {
                    name: os.environ.get(name)
                    for name in (
                        "CUDA_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "TRITON_CACHE_DIR",
                        "TRITON_DISABLE_LINE_INFO",
                        "CUDA_CACHE_PATH",
                    )
                },
                "numerical_policy": {"atol": 4e-3, "rtol": 2e-2},
                "input_provenance": "Extra eager diagnostic forward after restored model prefix, "
                "outside the original profiler capture. The old numerical receipt did not bind "
                "these saved input bytes. This experiment independently binds their hashes.",
                "boundary": "Complete production sparse_mla or sparse_mla_decode call, including "
                "selection/layout adaptation, native sparse attention, and decode merge. "
                "No oracle, comparison, file I/O or source hashing inside timed/profile ranges.",
            }
        )
    )


class Case:
    def __init__(self, path, torch, mla):
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.path = path
        self.layer = data["layer"]
        self.source_run_id = data["source_run_id"]
        self.q = data["q"].cuda()
        self.kv = data["kv"].cuda()
        self.indices = data["indices"].cuda()
        self.scale = float(data["attention_scale"])
        self.functions = {"prefill": mla.sparse_mla, "decode": mla.sparse_mla_decode}
        require(self.q.shape[0] == 1 and self.indices.shape[0] == 1, "Q1 input required")

    def call(self, method):
        return self.functions[method](self.q, self.kv, self.indices, self.scale)

    def description(self):
        return {
            "path": str(self.path.resolve()),
            "layer": self.layer,
            "source_run_id": self.source_run_id,
            "scale": self.scale,
            "q": {"shape": list(self.q.shape), "dtype": str(self.q.dtype)},
            "kv": {"shape": list(self.kv.shape), "dtype": str(self.kv.dtype)},
            "indices": {"shape": list(self.indices.shape), "dtype": str(self.indices.dtype)},
        }


def compare(actual, expected):
    import torch

    result = metrics(actual, expected)
    result["elements"] = actual.numel()
    result["different_bf16_bits"] = (
        (actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16))
        .sum()
        .item()
    )
    return result


def validate(case, output_dir, number):
    import torch

    from operators.deepseek_v32.attention.reference.torch import reference_sparse_mla

    oracle = reference_sparse_mla(case.q, case.kv, case.indices, case.scale)
    outputs = {method: case.call(method) for method in ("prefill", "decode")}
    for output in outputs.values():
        torch.testing.assert_close(output, oracle, atol=4e-3, rtol=2e-2)
    torch.testing.assert_close(outputs["decode"], outputs["prefill"], atol=4e-3, rtol=2e-2)
    graph_checks = {}
    for method in ("prefill", "decode"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = case.call(method)
        for _ in range(4):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                captured.view(torch.int16), outputs[method].view(torch.int16), atol=0, rtol=0
            )
        original = case.indices.clone()
        case.indices.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        require(captured.eq(0).all().item(), f"{method} all-invalid graph output is nonzero")
        case.indices.copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            captured.view(torch.int16), outputs[method].view(torch.int16), atol=0, rtol=0
        )
        graph_checks[method] = {
            "fixed_input_bitwise_replays": 4,
            "all_invalid_then_restored_selection_checked": True,
        }
        del graph, captured, original
    saved = output_dir / f"accepted_outputs_{number}_layer_{case.layer}.pt"
    torch.save(
        {**{name: value.cpu() for name, value in outputs.items()}, "oracle": oracle.cpu()}, saved
    )
    return {
        "case": case.description(),
        "prefill_vs_oracle": compare(outputs["prefill"], oracle),
        "decode_vs_oracle": compare(outputs["decode"], oracle),
        "decode_vs_prefill": compare(outputs["decode"], outputs["prefill"]),
        "graph": graph_checks,
        "outputs": {"path": saved.name, "sha256": digest(saved)},
        "oracle_boundary": "Independent FP32 attention arithmetic, rounded to BF16 output.",
    }


def memory_snapshot(torch):
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "device_used": total - free,
    }


def benchmark(case, args):
    import torch

    rows = []
    for method in ("prefill", "decode"):
        for execution in ("eager", "graph"):
            graph = None
            memory = None
            if execution == "graph":
                torch.cuda.synchronize()
                before = memory_snapshot(torch)
                torch.cuda.reset_peak_memory_stats()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = case.call(method)
                torch.cuda.synchronize()
                memory = {
                    "before_capture": before,
                    "after_capture": memory_snapshot(torch),
                    "capture_peak_allocated": torch.cuda.max_memory_allocated(),
                    "boundary": "Observed process/driver memory around capture, not a graph "
                    "storage reservation; allocator cache release can lower reserved bytes.",
                }
                call = graph.replay
            else:
                call = lambda method=method: case.call(method)
            for _ in range(args.warmups):
                call()
            torch.cuda.synchronize()
            events, walls = [], []
            for _ in range(args.repeats):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                begin.record()
                for _ in range(args.iterations):
                    call()
                end.record()
                end.synchronize()
                events.append(begin.elapsed_time(end) * 1000 / args.iterations)
                walls.append((time.perf_counter_ns() - start) / 1000 / args.iterations)
            rows.append(
                {
                    "layer": case.layer,
                    "input": str(case.path.resolve()),
                    "method": method,
                    "execution": execution,
                    "gpu_us_samples": events,
                    "gpu_us_median": statistics.median(events),
                    "wall_us_samples": walls,
                    "wall_us_median": statistics.median(walls),
                    "memory": memory,
                    "warmups": args.warmups,
                    "iterations_per_sample": args.iterations,
                }
            )
            del call
            if graph is not None:
                del graph, output
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("check", "bench", "profile"))
    parser.add_argument("--method", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--inputs", nargs="+", type=Path, default=[DEFAULT_INPUT])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    require(
        args.output_dir.resolve().is_relative_to(EXPERIMENT / "output"), "Use experiment output"
    )
    require(min(args.warmups, args.repeats, args.iterations) > 0, "Counts must be positive")
    require(args.mode == "check" or args.receipt is not None, "Bench/profile require a receipt")
    require(args.mode != "profile" or len(args.inputs) == 1, "Profile requires exactly one input")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configure_environment()
    import torch

    from operators.deepseek_v32.attention.device_only import mla

    require(
        torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0),
        "One SM90 GPU required",
    )
    with torch.inference_mode():
        cases = [Case(path, torch, mla) for path in args.inputs]
        # Warm both explicit APIs before collecting their common compiled identity.
        for case in cases:
            for method in ("prefill", "decode"):
                for _ in range(args.warmups):
                    case.call(method)
        torch.cuda.synchronize()
        binding = identity(args, torch, mla)
        receipt = None
        if args.mode != "check":
            receipt = json.loads(args.receipt.read_text())
            require(receipt["accepted"] is True, "Independent check did not pass")
            receipt_matches(receipt["identity"], binding)
        for path in source_paths():
            destination = args.output_dir / "source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        result = {
            "run_id": args.output_dir.name,
            "mode": args.mode,
            "argv": sys.argv,
            "identity": binding,
            "cases": [case.description() for case in cases],
        }
        if receipt is not None:
            result["receipt"] = {
                "path": str(args.receipt.resolve()),
                "sha256": digest(args.receipt),
            }
        if args.mode == "check":
            result["checks"] = [
                validate(case, args.output_dir, index) for index, case in enumerate(cases)
            ]
            result["accepted"] = True
        elif args.mode == "bench":
            result["samples"] = [row for case in cases for row in benchmark(case, args)]
        else:
            case = cases[0]
            torch.cuda.profiler.start()
            output = case.call(args.method)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            result["profile_method"] = args.method
            result["profile_calls"] = 1
            result["output_shape"] = list(output.shape)
        torch.cuda.synchronize()
        require(
            identity(args, torch, mla) == binding,
            "Source/input/native/compiled identity changed during execution",
        )
        result["identity_stable_after_execution"] = True
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "result.json"),
                "mode": args.mode,
                "accepted": result.get("accepted"),
            }
        )
    )


if __name__ == "__main__":
    main()
