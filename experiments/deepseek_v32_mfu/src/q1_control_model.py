"""Private full HBM Q1 model trial; independent instances and source-bound phases."""

import argparse
import importlib
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch

from experiments.deepseek_v32_mfu.src.profile_layers import sources
from experiments.deepseek_v32_mfu.src.q1_control import digest, exact, require
from models.deepseek_v32.model import DeepSeekEchoModel


def signature(args):
    return {
        "model_sources": sources(),
        "trial_sources": {
            str(path.resolve()): digest(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("q1_control.py"),
                *args.candidate_root.rglob("*.py"),
            )
        },
        "request": {"path": str(args.request.resolve()), "sha256": digest(args.request)},
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "precision": torch.backends.cuda.matmul.fp32_precision,
    }


def finite_exact(a, b):
    require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), "Nonfinite output")
    exact(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench"), required=True)
    parser.add_argument("--model", type=Path, default=Path("/preset-models"))
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--pairs", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from triton.backends.nvidia.compiler import get_ptxas

    os.environ.setdefault("TRITON_PTXAS_PATH", get_ptxas(90).path)
    os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/usr/local/cuda/bin/ptxas")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    package = types.ModuleType("_q1_control_model_candidate")
    package.__path__ = [str(args.candidate_root.resolve())]
    sys.modules[package.__name__] = package
    adapter = importlib.import_module(package.__name__ + ".adapter")
    candidate = importlib.import_module(package.__name__ + ".linear.fp8")
    ids = json.loads(args.request.read_text())["input_ids"]
    require(len(ids) == 65537 and ids[-1] == 111090, "Expected exact H64K/A1 input")
    before = signature(args)
    if args.mode == "bench":
        require(args.check is not None, "Benchmark needs independent check")
        receipt = json.loads(args.check.read_text())
        require(receipt["passed"] is True and receipt["mode"] == "check", "Invalid check")
        require(receipt["signature"] == before, "Check source or input differs")
    models, snapshots, graphs, expected, memory = {}, {}, {}, {}, {}
    result = {}
    try:
        with torch.inference_mode():
            for arm in ("baseline", "layout"):
                print("Loading " + arm, flush=True)
                model = DeepSeekEchoModel(
                    args.model,
                    devices=[0],
                    num_layers=3,
                    capacity=65537,
                    slots=65600,
                    chunk_size=1024,
                    extend_chunk_size=1,
                    host_arena_tokens=65600,
                    workspace_query_tokens=1024,
                    hbm_cache_budget_bytes=24 * 2**30,
                    dram_cache_budget_bytes=64 * 2**30,
                )
                models[arm] = model
                if arm == "layout":
                    binding = adapter.bind_model(model, candidate.fp8_linear)
                    require(len(binding.replacements) == 27, "Model linear target set differs")
                model.prepare_compute_graphs([1024, 1])
                prefix = model.forward(ids[:65536]).cpu()
                snapshots[arm] = model.snapshot_prefix()
                if args.mode == "check":
                    eager = {
                        name: value.cpu()
                        for name, value in model.forward(
                            ids[65536:], return_hidden=True, use_extend_graph=False
                        ).items()
                    }
                    model.restore_prefix(snapshots[arm])
                graphs[arm] = model.prepare_extend_graph(ids[65536:], return_hidden=True)
                for _ in range(3):
                    model.restore_prefix(snapshots[arm])
                    model.forward(ids[65536:], return_hidden=True)
                if args.mode == "check":
                    model.restore_prefix(snapshots[arm])
                    graph_output = {
                        name: value.cpu()
                        for name, value in model.forward(ids[65536:], return_hidden=True).items()
                    }
                    for name, value in eager.items():
                        finite_exact(value, graph_output[name])
                    expected[arm] = {"prefix": prefix, **graph_output}
                memory[arm] = graphs[arm].describe()
                print("Prepared " + arm, flush=True)
            if args.mode == "check":
                for name in expected["baseline"]:
                    finite_exact(expected["baseline"][name], expected["layout"][name])
                for changed in ([111091], [111092]):
                    outputs = {}
                    for arm, model in models.items():
                        model.restore_prefix(snapshots[arm])
                        outputs[arm] = {
                            name: value.cpu()
                            for name, value in model.forward(changed, return_hidden=True).items()
                        }
                    for name in outputs["baseline"]:
                        finite_exact(outputs["baseline"][name], outputs["layout"][name])
                path = args.output / "outputs.pt"
                torch.save(expected, path)
                result = {
                    "prefix_last_logits_and_all_extend_hidden_logits_exact": True,
                    "changed_graph_inputs": 2,
                    "outputs": {"path": str(path), "sha256": digest(path)},
                }
            else:
                samples = []
                for pair in range(args.pairs):
                    for arm in ("baseline", "layout") if pair % 2 == 0 else ("layout", "baseline"):
                        model = models[arm]
                        model.restore_prefix(snapshots[arm])
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        _value = model.forward(ids[65536:], return_hidden=True)
                        torch.cuda.synchronize()
                        samples.append(
                            {
                                "pair": pair,
                                "order": "AB" if pair % 2 == 0 else "BA",
                                "arm": arm,
                                "wall_ms": (time.perf_counter_ns() - start) / 1e6,
                            }
                        )
                summary = {
                    arm: statistics.median(row["wall_ms"] for row in samples if row["arm"] == arm)
                    for arm in models
                }
                result = {
                    "samples": samples,
                    "median_wall_ms": summary,
                    "boundary": "Complete synchronized HBM model.forward, embedding/L0-L2/final norm/LM head, graph includes all original work; prefix restoration outside timer; independent model+graph instances; hidden return requested in both arms.",
                }
            require(signature(args) == before, "Model source or runtime signature changed")
            payload = {
                "passed": True,
                "mode": args.mode,
                "signature": before,
                "result": result,
                "graph_storage": memory,
                "candidate_runtime": candidate.quantization.runtime_info(),
                "check": None
                if args.check is None
                else {"path": str(args.check), "sha256": digest(args.check)},
            }
    finally:
        errors = []
        primary = sys.exception()
        for model in models.values():
            try:
                model.close()
            except BaseException as error:  # noqa: BLE001 - retain all cleanup failures.
                errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "Q1 model trial cleanup failed", ([primary] if primary else []) + errors
            )
    (args.output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"passed": True, "mode": args.mode, "summary": result.get("median_wall_ms")}))


if __name__ == "__main__":
    main()
