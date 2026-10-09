"""Independent complete-projection trial of a private official QKV-A fusion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
from dataclasses import fields
from pathlib import Path

import torch

from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_mfu.src.backend_provenance import (
    collect_backend_provenance,
    collect_flashinfer_runtime_artifacts,
)
from experiments.deepseek_v32_mfu.src.q1_control import digest, exact, graph, require
from experiments.deepseek_v32_mfu.src.q1_projection import identity as projection_identity
from experiments.deepseek_v32_mfu.src.run_contract import checkpoint_identity
from models.deepseek_v32.checkpoint import CheckpointReader
from models.deepseek_v32.projections import CheckpointAttention
from operators.deepseek_v32.linear import quantization

KIND = "deepseek-q1-fused-qkv-projection-v1"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_digest(tensor):
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def weight_identity(attention):
    result = {}
    for name, value in vars(attention).items():
        if isinstance(value, torch.Tensor):
            result[name] = tensor_digest(value)
        elif hasattr(value, "weight"):
            result[name + ".weight"] = tensor_digest(value.weight)
            if value.scales is not None:
                result[name + ".scales"] = tensor_digest(value.scales)
    return result


def same(left, right):
    for field in fields(left):
        exact(getattr(left, field.name), getattr(right, field.name))


def runtime():
    return {
        "backend": collect_backend_provenance(),
        "loaded_jit": collect_flashinfer_runtime_artifacts(),
        "quantization": quantization.runtime_info(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), required=True)
    parser.add_argument("--model", type=Path, default=Path("/preset-models"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--pairs", type=int, default=30)
    args = parser.parse_args()
    require(args.pairs > 0, "Positive pair count required")
    args.output.mkdir(parents=True, exist_ok=False)
    from triton.backends.nvidia.compiler import get_ptxas

    os.environ.setdefault("TRITON_PTXAS_PATH", get_ptxas(90).path)
    os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/usr/local/cuda/bin/ptxas")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    spec = importlib.util.spec_from_file_location("_q1_qkv_trial", args.candidate)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    source = projection_identity(args)
    source["sources"].update(
        {
            str(path.resolve()): digest(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("q1_projection.py"),
                Path(__file__).with_name("backend_provenance.py"),
            )
        }
    )
    source["checkpoint"] = checkpoint_identity(args.model)
    archive = args.output / "source"
    archive.mkdir()
    for number, path in enumerate(source["sources"]):
        shutil.copy2(path, archive / (str(number) + "_" + Path(path).name))
    reader = CheckpointReader(args.model)
    cases, identities = [], []
    with torch.inference_mode():
        for layer in range(3):
            print(f"Preparing layer {layer}", flush=True)
            attn = CheckpointAttention(
                args.model, layer, device="cuda", reader=reader, linear_backend="fp8"
            )
            prepared = candidate.prepare(attn)
            torch.manual_seed(4200 + layer)
            x = torch.randn((1, attn.cfg.dim), device="cuda", dtype=torch.bfloat16)
            position = torch.full((1,), 65536, device="cuda", dtype=torch.float32)
            identity = {
                "layer": layer,
                "weights": weight_identity(attn),
                "prepared_weights": {k: tensor_digest(v) for k, v in prepared.items()},
                "input": tensor_digest(x),
                "position": tensor_digest(position),
            }
            calls = {
                "baseline": lambda data, attn=attn, position=position: attn.project_positions(
                    data, position, normalized=True
                ),
                "fused": lambda data, attn=attn, prepared=prepared, position=position: (
                    candidate.project(attn, prepared, data, position)
                ),
            }
            memory_before = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "device_used": torch.cuda.device_memory_used(),
            }
            captures = {arm: graph(function, x) for arm, function in calls.items()}
            torch.cuda.synchronize()
            identity["runtime"] = runtime()
            identities.append(identity)
            case_identity = {"source": source, "case": identity}
            check_path = args.check / f"layer_{layer}/receipt.json" if args.check else None
            receipt = None
            if args.mode != "check":
                receipt = require_receipt(check_path, kind=KIND, identity=case_identity)
            result = {
                "layer": layer,
                "identity": case_identity,
                "memory_before": memory_before,
                "memory_after": {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "device_used": torch.cuda.device_memory_used(),
                },
            }
            if args.mode == "check":
                for iteration in range(8):
                    if iteration == 6:
                        x.zero_()
                    elif iteration == 7:
                        x.copy_(torch.randn_like(x).neg_())
                    else:
                        x.mul_(0.75).add_(0.125)
                    position.fill_(65536 - iteration * 127)
                    for capture, _ in captures.values():
                        capture.replay()
                    torch.cuda.synchronize()
                    same(captures["baseline"][1], captures["fused"][1])
                    same(captures["baseline"][1], calls["baseline"](x))
                    same(captures["fused"][1], calls["fused"](x))
                result["checks"] = {
                    "passed": True,
                    "bitwise_projected_fields": 6,
                    "changed_hidden_position_replays": 8,
                    "eager_graph_both_arms": True,
                }
            else:
                result["receipt_sha256"] = receipt["receipt_sha256"]
                for capture, _ in captures.values():
                    for _ in range(10):
                        capture.replay()
                torch.cuda.synchronize()
                if args.mode == "profile":
                    torch.cuda.cudart().cudaProfilerStart()
                    for capture, _ in captures.values():
                        for _ in range(3):
                            capture.replay()
                    torch.cuda.synchronize()
                    for arm, (capture, _) in captures.items():
                        torch.cuda.nvtx.range_push(f"q1_qkv/{arm}/layer{layer}")
                        capture.replay()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
                else:
                    samples = []
                    for pair in range(args.pairs):
                        for arm in (
                            ("baseline", "fused") if pair % 2 == 0 else ("fused", "baseline")
                        ):
                            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                            torch.cuda.synchronize()
                            start.record()
                            captures[arm][0].replay()
                            end.record()
                            end.synchronize()
                            samples.append(
                                {"pair": pair, "arm": arm, "gpu_us": start.elapsed_time(end) * 1000}
                            )
                    result["samples"] = samples
                    result["medians_us"] = {
                        arm: statistics.median(r["gpu_us"] for r in samples if r["arm"] == arm)
                        for arm in captures
                    }
                    print(result["medians_us"], flush=True)
            require(runtime() == identity["runtime"], "Loaded backend/JIT identity changed")
            directory = args.output / f"layer_{layer}"
            directory.mkdir()
            save(directory / "result.json", result)
            if args.mode == "check":
                write_receipt(
                    directory / "receipt.json",
                    kind=KIND,
                    identity=case_identity,
                    checks=result["checks"],
                    artifacts={"result": directory / "result.json"},
                )
            cases.append(result)
            del calls, captures, x, position, prepared, attn
    for path, expected in source["sources"].items():
        require(digest(Path(path)) == expected, "Source changed during trial")
    require(checkpoint_identity(args.model) == source["checkpoint"], "Checkpoint changed")
    save(
        args.output / "result.json",
        {
            "run_id": args.output.name,
            "mode": args.mode,
            "accepted": True,
            "source": source,
            "cases": cases,
            "boundary": "Complete normalized Q1 projection Graph replay, real L0-L2 weights, deterministic synthetic BF16 hidden. All six outputs, quantization and graph joins included; weight packing and graph preparation excluded. Separate check/bench/profile processes.",
        },
    )


if __name__ == "__main__":
    main()
