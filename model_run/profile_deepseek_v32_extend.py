"""Profile the actual extend path with nested operation scopes, without changing kernels."""

import argparse
import json
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import torch

if __package__:
    from . import deepseek_v32_decode as decode
    from . import deepseek_v32_extend as extend
else:
    import deepseek_v32_decode as decode
    import deepseek_v32_extend as extend


class Scopes:
    def __init__(self):
        self.stack = []
        self.events = []
        self.enabled = False

    def wrap(self, name, fn):
        @wraps(fn)
        def call(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)
            label = name(*args, **kwargs) if callable(name) else name
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.profiler.record_function("step::" + label):
                begin.record()
                self.stack.append(label)
                try:
                    result = fn(*args, **kwargs)
                finally:
                    self.stack.pop()
                    end.record()
                    self.events.append((label, begin, end))
                return result

        return call


def instrument(runner, scopes, patches):
    for name in (
        "wq_a",
        "wq_b",
        "wkv_a",
        "wk_b",
        "index_wqi",
        "index_wki",
        "index_weights",
        "wv_b",
        "wo",
    ):
        layer = getattr(runner, name)
        if hasattr(layer, "activation_quantizer"):
            patches.enter_context(
                patch.object(
                    layer,
                    "activation_quantizer",
                    scopes.wrap(name + "/quantize", layer.activation_quantizer),
                )
            )
        patches.enter_context(patch.object(runner, name, scopes.wrap(name, layer)))
    for name in (
        "fp8_gemm_nt",
        "fp8_fp4_gemm_nt",
        "m_grouped_fp8_gemm_nt_masked",
        "m_grouped_fp8_fp4_gemm_nt_masked",
    ):
        fn = getattr(decode.deep_gemm, name)
        patches.enter_context(
            patch.object(
                decode.deep_gemm, name, scopes.wrap(lambda *a, **kw: scopes.stack[-1] + "/gemm", fn)
            )
        )
    for name in ("q_norm", "kv_norm", "index_norm"):
        patches.enter_context(
            patch.object(runner.ops, name, scopes.wrap(name, getattr(runner.ops, name)))
        )
    patches.enter_context(
        patch.object(
            runner.ops,
            "apply_rope",
            scopes.wrap(
                lambda *a, **kw: "index/rope" if kw["is_neox"] else "mla/rope",
                runner.ops.apply_rope,
            ),
        )
    )
    patches.enter_context(
        patch.object(
            extend, "quantize_index", scopes.wrap("index/q_quantize", extend.quantize_index)
        )
    )
    patches.enter_context(
        patch.object(
            decode.deep_gemm,
            "fp8_fp4_mqa_logits",
            scopes.wrap("index/mqa_logits", decode.deep_gemm.fp8_fp4_mqa_logits),
        )
    )
    for attr, label in (
        ("project_chunk", "projection/layout"),
        ("write_chunk", "cache/append"),
        ("index_logits", "index/weights_bounds"),
        ("select_indices", "index/topk"),
        ("sparse_prefill", "mla/prefill"),
        ("post_wv_b", "wv_b/layout"),
        ("post_wo", "wo/layout"),
        ("run_once", "loop/output_copy"),
    ):
        patches.enter_context(patch.object(runner, attr, scopes.wrap(label, getattr(runner, attr))))


def profile_case(runner, case, repeats, trace):
    for _ in range(2):
        runner.run_once(case)
    torch.cuda.synchronize()
    e2e = decode.bench_cuda(lambda: runner.run_once(case), 1, repeats)
    scopes = Scopes()
    with ExitStack() as patches:
        instrument(runner, scopes, patches)
        scopes.enabled = True
        # Event timing is collected separately from CUPTI to limit profiler distortion.
        for _ in range(repeats):
            runner.run_once(case)
        torch.cuda.synchronize()
        event_ms = defaultdict(float)
        for name, begin, end in scopes.events:
            event_ms[name] += begin.elapsed_time(end) / repeats
        scopes.events.clear()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            out = runner.run_once(case)
            torch.cuda.synchronize()
        assert torch.isfinite(out).all().item()
    trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace))
    # Direct driver launches (Triton/DeepGEMM/MLA) may lack a PyTorch CPU op.
    # Use CUPTI correlation IDs to find the launch and its innermost named scope.
    events = json.loads(trace.read_text())["traceEvents"]
    annotations = [
        e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("step::")
    ]
    launches = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})
    }
    gpu_ms, calls, kernels = defaultdict(float), defaultdict(int), defaultdict(int)
    for annotation in annotations:
        calls[annotation["name"][6:]] += 1
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches.get(event.get("args", {}).get("correlation"))
        candidates = (
            []
            if launch is None
            else [
                a
                for a in annotations
                if (a["pid"], a["tid"]) == (launch["pid"], launch["tid"])
                and a["ts"] <= launch["ts"] <= a["ts"] + a["dur"]
            ]
        )
        if not candidates:
            raise RuntimeError(f"Unattributed GPU activity: {event['name']}")
        name = min(candidates, key=lambda a: a["dur"])["name"][6:]
        gpu_ms[name] += event["dur"] / 1000
        kernels[name] += 1
    rows = [
        {
            "step": name,
            "calls": calls[name],
            "event_inclusive_ms": event_ms[name],
            "gpu_exclusive_ms": gpu_ms[name],
            "kernels": kernels[name],
        }
        for name in event_ms
    ]
    return {
        "history": case.history_len,
        "new_tokens": case.new_tokens,
        "chunk": runner.chunk_size,
        "e2e_ms": e2e,
        "gpu_total_ms": sum(gpu_ms.values()),
        "steps": rows,
        "trace": str(trace),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=decode.CONFIG_PATH)
    parser.add_argument("--history-lens", default="4096,65536")
    parser.add_argument("--new-tokens", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/extend_step_profile"))
    args = parser.parse_args()
    if args.iters < 1:
        parser.error("iters must be positive")
    torch.manual_seed(0)
    runner = extend.V32ExtendRunner(
        decode.load_config(args.config), chunk_size=args.chunk_size
    )
    reports = []
    for history in decode.parse_ints(args.history_lens):
        case = runner.make_case(history, args.new_tokens)
        report = profile_case(
            runner, case, args.iters, args.output_dir / f"h{history}_n{args.new_tokens}.trace.json"
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
        del case
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "repeats": args.iters,
                "mode": "fp8",
                "bf16_qk": False,
                "synthetic": True,
                "reports": reports,
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Extend step profile",
        "",
        "FP8 synthetic attention, new tokens="
        + str(args.new_tokens)
        + ", chunk="
        + str(args.chunk_size)
        + ".",
        "",
        "GPU times below are exclusive and summed across chunks. Event times in summary.json are inclusive (parents include children); do not sum them. E2E is measured separately without instrumentation. CUPTI kernel times are from one warmed iteration.",
        "",
    ]
    for report in reports:
        lines += [
            f"## History {report['history']}",
            "",
            f"E2E: {report['e2e_ms']:.3f} ms; attributed GPU: {report['gpu_total_ms']:.3f} ms.",
            "",
            "| Step | Calls | GPU ms (exclusive) |",
            "|---|---:|---:|",
        ]
        lines += [
            f"| {row['step']} | {row['calls']} | {row['gpu_exclusive_ms']:.4f} |"
            for row in report["steps"]
        ]
        lines.append("")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
