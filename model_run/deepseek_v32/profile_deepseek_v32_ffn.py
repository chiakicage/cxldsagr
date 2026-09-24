"""Measure synthetic dense SwiGLU with the extend runner's MXFP8 projection path."""

import argparse
import json
from collections import defaultdict
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model_run.deepseek_v32.deepseek_v32_decode import QuantizedLinear, bench_cuda, deep_gemm
from model_run.deepseek_v32.deepseek_v32_extend_kernels import quantize_activation


def profile(layers, tokens, repeats, output_dir):
    x = torch.randn(tokens, 7168, device="cuda", dtype=torch.bfloat16)
    enabled = False

    def scope(name):
        return torch.profiler.record_function("ffn::" + name) if enabled else nullcontext()

    def run():
        with scope("gate"):
            gate = layers[0](x)
        with scope("up"):
            up = layers[1](x)
        with scope("swiglu"):
            intermediate = (F.silu(gate.float()) * up.float()).bfloat16()
        with scope("down"):
            return layers[2](intermediate)

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    e2e = bench_cuda(run, 1, repeats)

    def wrapped(name, fn):
        def call(*args, **kwargs):
            with scope(name):
                return fn(*args, **kwargs)

        return call

    with ExitStack() as patches:
        for name, layer in zip(("gate", "up", "down"), layers, strict=True):
            patches.enter_context(
                patch.object(
                    layer, "activation_quantizer", wrapped(name + "/quantize", quantize_activation)
                )
            )
        patches.enter_context(
            patch.object(deep_gemm, "fp8_gemm_nt", wrapped("gemm", deep_gemm.fp8_gemm_nt))
        )
        enabled = True
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            out = run()
            torch.cuda.synchronize()
    assert out.shape == x.shape and torch.isfinite(out).all().item()
    trace = output_dir / f"n{tokens}.trace.json"
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    scopes = [
        e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("ffn::")
    ]
    launches = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})
    }
    times = defaultdict(float)
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches[e["args"]["correlation"]]
        parents = sorted(
            (
                s
                for s in scopes
                if (s["pid"], s["tid"]) == (launch["pid"], launch["tid"])
                and s["ts"] <= launch["ts"] <= s["ts"] + s["dur"]
            ),
            key=lambda s: s["dur"],
        )
        name = parents[0]["name"][5:]
        if name == "gemm":
            name = parents[1]["name"][5:] + "/gemm"
        times[name] += e["dur"] / 1000
    return {
        "tokens": tokens,
        "e2e_ms": e2e,
        "gpu_total_ms": sum(times.values()),
        "steps_ms": dict(times),
        "finite_output": True,
        "trace": str(trace),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-tokens", default="1024,4096")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/extend_step_profile/dense_ffn")
    )
    args = parser.parse_args()
    tokens = [int(t) for t in args.new_tokens.split(",")]
    if args.iters < 1 or any(t < 1 for t in tokens):
        parser.error("token counts and iters must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    layers = [
        QuantizedLinear(n, k, mode="fp8") for n, k in ((18432, 7168), (18432, 7168), (7168, 18432))
    ]
    for layer in layers:
        layer.activation_quantizer = quantize_activation
    reports = [profile(layers, t, args.iters, args.output_dir) for t in tokens]
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "mode": "fp8",
        "synthetic": True,
        "dim": 7168,
        "inter_dim": 18432,
        "repeats": args.iters,
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
