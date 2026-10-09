"""Independent check, clean timing, and trace for Q1 activation-scale layout."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch

from operators.deepseek_v32.linear import fp8 as baseline

ROOT = Path(__file__).resolve().parents[3]
SHAPES = (
    (7168, 1536),
    (1536, 24576),
    (7168, 576),
    (1536, 8192),
    (7168, 128),
    (16384, 7168),
    (7168, 18432),
    (18432, 7168),
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def exact(left, right):
    require(left.shape == right.shape and left.dtype == right.dtype, "Tensor ABI differs")
    a = left.clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)
    b = right.clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)
    require(torch.equal(a, b), "Tensor bytes differ")


def inputs(k, n):
    torch.manual_seed(42000 + k + n)
    x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scales = torch.full(((n + 127) // 128, (k + 127) // 128), 0.125, device="cuda")
    return x, weight, scales


def functions(candidate, weight, scales):
    return {
        "baseline": lambda x: baseline.fp8_linear(x, weight, scales),
        "layout": lambda x: candidate.fp8_linear(x, weight, scales, _consumer_scale_layout=True),
    }


def graph(function, x):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        out = function(x)
    return capture, out


def identity(candidate, source_dir):
    files = [
        Path(__file__),
        *Path(baseline.__file__).parent.glob("*.py"),
        *source_dir.glob("*.py"),
        ROOT / "3rdparty/DeepGEMM/csrc/jit_kernels/impls/smxx_layout.hpp",
        ROOT / "3rdparty/DeepGEMM/deep_gemm/utils/math.py",
        Path(baseline.__file__).parent / "tests/test_linear_quantization.py",
    ]
    import deep_gemm

    files.append(Path(deep_gemm.__file__))
    files.extend(
        Path(module.__file__)
        for name, module in list(sys.modules.items())
        if name.startswith("deep_gemm") and getattr(module, "__file__", "").endswith(".so")
    )
    return {
        "source_sha256": {str(path.resolve()): digest(path) for path in sorted(set(files))},
        "baseline": baseline.quantization.build_info(),
        "layout": candidate.quantization.build_info(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "capability": list(torch.cuda.get_device_capability()),
        "affinity": sorted(os.sched_getaffinity(0)),
        "shapes": [list(shape) for shape in SHAPES],
    }


def check(candidate, output):
    from deep_gemm import per_token_cast_to_fp8

    from operators.deepseek_v32.linear.tests.test_linear_quantization import _edge_values

    # The fixed dtype/shape/stride matrix intentionally compiles more than
    # Dynamo's default eight variants of the independent official helper.
    torch._dynamo.config.recompile_limit = 128
    oracle = torch.compile(
        lambda x: per_token_cast_to_fp8(x, True),
        fullgraph=True,
        dynamic=True,
        options={"triton.cudagraphs": False},
    )
    cases = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for columns in (1, 127, 128, 129, 1536, 7168, 16384, 18432):
            for stride in (1, 2):
                owner = torch.zeros((1, columns * stride + 16), dtype=dtype, device="cuda")
                x = owner[:, 8 : 8 + columns * stride : stride]
                x.copy_(_edge_values(dtype, 1, columns))
                before = owner.clone()
                a = baseline.quantize_fp8_activation(x)
                b = candidate.quantization.quantize(x, consumer_scale_layout=True)
                reference = oracle(x)
                for left, right, truth in zip(a, b, reference, strict=True):
                    exact(left, right)
                    exact(left, truth)
                exact(owner, before)
                require(b[1].stride() == (1, 4), "Wrong Q1 scale strides")
                cases.append(
                    {
                        "dtype": str(dtype),
                        "columns": columns,
                        "stride": stride,
                        "exact": True,
                        "scale_storage_bytes": b[1].untyped_storage().nbytes(),
                    }
                )
    linear = []
    for k, n in SHAPES:
        x, weight, scales = inputs(k, n)
        apis = functions(candidate, weight, scales)
        a, b = apis["baseline"](x), apis["layout"](x)
        exact(a, b)
        captures = {name: graph(function, x) for name, function in apis.items()}
        for replay in range(4):
            x.mul_(0.75).add_(0.125)
            for capture, value in captures.values():
                capture.replay()
            torch.cuda.synchronize()
            exact(captures["baseline"][1], captures["layout"][1])
            exact(captures["baseline"][1], apis["baseline"](x))
        path = output / f"linear_k{k}_n{n}.pt"
        torch.save(
            {
                "input": x.cpu(),
                "baseline": captures["baseline"][1].cpu(),
                "layout": captures["layout"][1].cpu(),
            },
            path,
        )
        linear.append(
            {
                "shape": [1, k, n],
                "exact": True,
                "changed_graph_replays": 4,
                "outputs": {"path": str(path), "sha256": digest(path)},
            }
        )
        del captures, apis, x, weight, scales, a, b
    return {"quantizer_cases": cases, "linears": linear}


def measure(candidate, pairs, iterations, profile):
    samples = []
    for k, n in SHAPES:
        x, weight, scales = inputs(k, n)
        apis = functions(candidate, weight, scales)
        captures = {name: graph(function, x) for name, function in apis.items()}
        for mode in ("eager", "graph"):
            for pair in range(1 if profile else pairs):
                order = ("baseline", "layout") if pair % 2 == 0 else ("layout", "baseline")
                for name in order:
                    function = (
                        (lambda api=apis[name], source=x: api(source))
                        if mode == "eager"
                        else captures[name][0].replay
                    )
                    torch.cuda.synchronize()
                    if profile:
                        torch.cuda.nvtx.range_push(f"q1_control/{name}/{mode}/k{k}_n{n}")
                        _value = function()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                        continue
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    count = 1 if mode == "eager" else iterations
                    t0 = time.perf_counter_ns()
                    start.record()
                    for _ in range(count):
                        _value = function()
                    end.record()
                    end.synchronize()
                    samples.append(
                        {
                            "shape": [1, k, n],
                            "mode": mode,
                            "pair": pair,
                            "order": "AB" if pair % 2 == 0 else "BA",
                            "arm": name,
                            "iterations": count,
                            "event_us": start.elapsed_time(end) * 1000 / count,
                            "wall_us": (time.perf_counter_ns() - t0) / 1000 / count,
                        }
                    )
        del apis, captures, x, weight, scales
    summary = []
    for k, n in SHAPES:
        for mode in ("eager", "graph"):
            for arm in ("baseline", "layout"):
                chosen = [
                    row
                    for row in samples
                    if row["shape"] == [1, k, n] and row["mode"] == mode and row["arm"] == arm
                ]
                if chosen:
                    summary.append(
                        {
                            "shape": [1, k, n],
                            "mode": mode,
                            "arm": arm,
                            "event_us": statistics.median(row["event_us"] for row in chosen),
                            "wall_us": statistics.median(row["wall_us"] for row in chosen),
                        }
                    )
    return {
        "samples": samples,
        "summary": summary,
        "boundary": "Eager includes validation, allocation, quantization, original GEMM and sync. Graph contains complete quantization and original GEMM on capture-owned input/output storage; capture and replay submission are separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    require(args.pairs > 0 and args.iterations > 0, "Repetition counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    package = types.ModuleType("_q1_control_candidate")
    package.__path__ = [str(args.candidate_root.resolve())]
    sys.modules[package.__name__] = package
    candidate = importlib.import_module("_q1_control_candidate.fp8")
    # Declare compiler selection before either wrapper freezes its identity.
    from torch._inductor.runtime.compile_tasks import (
        _set_triton_libdevice_path,
        _set_triton_ptxas_path,
    )
    from triton.backends.nvidia.compiler import get_ptxas

    os.environ.setdefault("TRITON_PTXAS_PATH", get_ptxas(90).path)
    _set_triton_ptxas_path()
    _set_triton_libdevice_path()
    torch.set_num_threads(8)
    with torch.inference_mode():
        warm_x, warm_w, warm_s = inputs(128, 64)
        for function in functions(candidate, warm_w, warm_s).values():
            function(warm_x)
        torch.cuda.synchronize()
        before = identity(candidate, args.candidate_root)
        if args.mode != "check":
            require(args.check is not None, "A successful matching check is required")
            receipt = json.loads(args.check.read_text())
            require(receipt["passed"] is True and receipt["mode"] == "check", "Invalid check")
            require(receipt["identity"] == before, "Check identity differs")
        if args.mode == "profile":
            torch.cuda.cudart().cudaProfilerStart()
        result = (
            check(candidate, args.output)
            if args.mode == "check"
            else measure(candidate, args.pairs, args.iterations, args.mode == "profile")
        )
        torch.cuda.synchronize()
        if args.mode == "profile":
            torch.cuda.cudart().cudaProfilerStop()
        require(identity(candidate, args.candidate_root) == before, "Identity changed")
        payload = {
            "mode": args.mode,
            "passed": True,
            "identity": before,
            "result": result,
            "check": None
            if args.check is None
            else {"path": str(args.check), "sha256": digest(args.check)},
            "runtime": {
                "baseline": baseline.quantization.runtime_info(),
                "layout": candidate.quantization.runtime_info(),
            },
        }
        (args.output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "passed": True,
                    "mode": args.mode,
                    "output": str(args.output),
                    "summary": result.get("summary", []),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
