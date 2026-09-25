"""Attribute NOSA GPU work and useful matmul FLOPs to modules on saved GR inputs."""

from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch

from executor.model_executor import run_chunks
from experiments.legacy.nosa_gr_forward.src.measure import measure_request
from experiments.nosa_gr_65536_1024.src.instrumentation import ModuleScopes
from experiments.nosa_gr_65536_1024.src.sources import source_hashes
from models.nosa import model as nosa


def attribute_trace(path):
    """Use CUPTI correlation IDs, counting each GPU activity exactly once.

    FunctionEvent.kernels can duplicate a kernel on a parent and a child in
    PyTorch 2.10, so use the actual Chrome GPU activities instead.
    """
    events = json.loads(path.read_text())["traceEvents"]
    scopes = defaultdict(list)
    launches = {}
    for event in events:
        if event.get("cat") == "user_annotation" and event["name"].startswith("nosa::"):
            scopes[event["pid"], event["tid"]].append(event)
        if event.get("cat") in ("cuda_runtime", "cuda_driver"):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation] = event
    starts = {}
    for key, values in scopes.items():
        values.sort(key=lambda event: (event["ts"], -event["dur"]))
        starts[key] = [event["ts"] for event in values]
    times = Counter()
    counts = Counter()
    kernel_names = defaultdict(Counter)
    total_us = 0.0
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        total_us += event["dur"]
        launch = launches.get(event.get("args", {}).get("correlation"))
        name = "unattributed"
        if launch is not None:
            key = launch["pid"], launch["tid"]
            index = bisect.bisect_right(starts.get(key, []), launch["ts"]) - 1
            while index >= 0:
                scope = scopes[key][index]
                if launch["ts"] <= scope["ts"] + scope["dur"]:
                    name = scope["name"][6:]
                    break
                index -= 1
        times[name] += event["dur"]
        counts[name] += 1
        kernel_names[name][event["name"]] += 1
    if times["unattributed"]:
        raise RuntimeError(f"Unattributed GPU activity: {times['unattributed']} us")
    assert abs(sum(times.values()) - total_us) < 1e-5
    return times, counts, kernel_names


@torch.inference_mode()
def profile_request(model, ids, prefix_tokens, chunk_size, trace_path):
    cache = model.new_cache(ids.numel())
    scopes = ModuleScopes()
    torch.cuda.synchronize(ids.device)
    outputs = {}
    with ExitStack() as stack:
        stack.callback(model.cache_manager.release, cache)
        scopes.install(model, stack)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for phase, tokens in (
                ("full_prefill", ids),
                ("prefix_prefill", ids[:prefix_tokens]),
                ("candidate_extend", ids[prefix_tokens:]),
            ):
                if phase == "prefix_prefill":
                    cache.reset()
                scopes.phase = phase
                with scopes.scope("model_misc"):
                    hidden = run_chunks(model, tokens, cache, chunk_size)
                    if phase != "prefix_prefill":
                        outputs[phase] = hidden[-1]
            torch.cuda.synchronize(ids.device)
    full, extend = outputs["full_prefill"].float(), outputs["candidate_extend"].float()
    validation = {
        "finite": bool(torch.isfinite(full).all() and torch.isfinite(extend).all()),
        "full_vs_split_last_hidden_max_abs": float((full - extend).abs().max()),
        "full_vs_split_last_hidden_cosine": float(
            torch.nn.functional.cosine_similarity(full, extend, dim=0)
        ),
    }
    assert validation["finite"]
    prof.export_chrome_trace(str(trace_path))
    del prof
    gc.collect()
    times, counts, names = attribute_trace(trace_path)
    rows = []
    for key, us in times.items():
        phase, category, layer = key.split("/")
        rows.append(
            {
                "phase": phase,
                "module": category,
                "layer": layer,
                "gpu_us": us,
                "flops": scopes.flops[key],
                "calls": scopes.calls[key],
                "gpu_activities": counts[key],
                "kernel_names": dict(names[key]),
            }
        )
    assert sum(row["flops"] for row in rows) == sum(scopes.flops.values())
    return rows, validation


def module_summary(rows, peak_tflops):
    grouped = {}
    for row in rows:
        key = row["phase"], row["module"]
        item = grouped.setdefault(
            key, {"phase": key[0], "module": key[1], "gpu_us": 0, "flops": 0, "gpu_activities": 0}
        )
        for field in ("gpu_us", "flops", "gpu_activities"):
            item[field] += row[field]
    for item in grouped.values():
        item["tflops"] = item["flops"] / item["gpu_us"] / 1e6 if item["flops"] else None
        item["mfu_pct"] = item["tflops"] / peak_tflops * 100 if item["flops"] else None
    return list(grouped.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/legacy/nosa_gr_forward/output/data/20260925/measure"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=989.0,
        help="Reference BF16 dense Tensor Core peak; default nominal H200 SXM",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 profiles all input shapes")
    args = parser.parse_args()
    if args.peak_tflops <= 0 or args.limit < 0:
        parser.error("peak must be positive, limit nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.profile_dir.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.input_dir / "metadata.json").read_text())
    requests = [
        json.loads(line) for line in (args.input_dir / "requests.jsonl").read_text().splitlines()
    ]
    requests = [row for row in requests if row["request_key"].endswith("_r1")]
    if args.limit:
        requests = requests[: args.limit]
    baseline = json.loads((args.input_dir / "summary.json").read_text())
    device = torch.device(meta["args"]["device"])
    torch.cuda.set_device(device)
    model = nosa.NosaForCausalLM.from_pretrained(
        meta["args"]["model_path"], device=device, dtype=torch.bfloat16
    )
    chunk = meta["args"]["prefill_chunk_size"]
    results = []
    for request in requests:
        key = request["request_key"]
        ids = torch.tensor(request["input_ids"], device=device, dtype=torch.long)
        warmup = measure_request(
            model, ids, prefix_tokens=request["stable_prefix_tokens"], chunk_size=chunk
        )
        rows, validation = profile_request(
            model,
            ids,
            request["stable_prefix_tokens"],
            chunk,
            args.profile_dir / f"{key}.trace.json",
        )
        assert (
            validation["full_vs_split_last_hidden_max_abs"]
            == warmup["full_vs_split_last_hidden_max_abs"]
        )
        summary = module_summary(rows, args.peak_tflops)
        result = {
            "request_key": key,
            "input_tokens": ids.numel(),
            "prefix_tokens": request["stable_prefix_tokens"],
            "candidate_tokens": request["candidate_suffix_tokens"],
            "modules": summary,
            "per_layer": rows,
            "validation": validation,
        }
        old = next(r for r in baseline if r["input_tokens"] == ids.numel())
        result["e2e"] = {}
        config = model.config
        linear_per_token = (
            2
            * config.num_hidden_layers
            * (
                2 * config.hidden_size**2
                + 2 * config.hidden_size * config.num_key_value_heads * config.head_dim
                + 3 * config.hidden_size * config.intermediate_size
            )
        )
        for phase, tokens, prefix in (
            ("full_prefill", ids.numel(), 0),
            ("prefix_prefill", request["stable_prefix_tokens"], 0),
            (
                "candidate_extend",
                request["candidate_suffix_tokens"],
                request["stable_prefix_tokens"],
            ),
        ):
            flops = sum(r["flops"] for r in summary if r["phase"] == phase)
            expected = (
                linear_per_token * tokens
                + 4
                * config.num_hidden_layers
                * config.num_attention_heads
                * config.head_dim
                * (tokens * prefix + tokens * (tokens + 1) // 2)
            )
            assert flops == expected, (phase, flops, expected)
            ms = old["metrics"][phase + "_ms"]["median"]
            result["e2e"][phase] = {
                "flops": flops,
                "wall_ms": ms,
                "mfu_pct": flops / (ms * 1e9) / args.peak_tflops * 100,
            }
        results.append(result)
        output = {
            "peak_tflops": args.peak_tflops,
            "peak_basis": "nominal H200 SXM BF16 dense Tensor Core reference, not measured; no 2:4 sparsity multiplier",
            "input_dir": str(args.input_dir),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime_source_sha256": source_hashes(*Path(__file__).parent.glob("*.py")),
            "gpu": str(torch.cuda.get_device_properties(device)),
            "torch": torch.__version__,
            "results": results,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
        print(key, json.dumps(result["e2e"]), flush=True)
        del rows, summary
        gc.collect()


if __name__ == "__main__":
    main()
