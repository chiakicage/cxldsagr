"""Independent Q1 exact-selection A/B with full-graph and L0-L2 CUDA events."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_mfu.src.profile_layers import sources
from experiments.deepseek_v32_mfu.src.q1_control import digest, require
from experiments.deepseek_v32_mfu.src.q1_projection_model import finite_exact
from experiments.deepseek_v32_mfu.src.q1_qkv import runtime, save, tensor_digest, weight_identity
from experiments.deepseek_v32_mfu.src.run_contract import checkpoint_identity
from models.deepseek_v32.model import DeepSeekEchoModel

ROOT = Path(__file__).resolve().parents[3]
KIND = "deepseek-q1-exact-selection-full-model-v1"


def source_identity(args):
    files = {str((ROOT / path).resolve()): sha for path, sha in sources().items()}
    for path in (
        Path(__file__),
        Path(__file__).with_name("q1_qkv.py"),
        Path(__file__).with_name("q1_projection_model.py"),
        Path(__file__).with_name("q1_control.py"),
        *args.candidate.parent.glob("*.py"),
    ):
        files[str(path.resolve())] = digest(path)
    return {
        "sources": files,
        "checkpoint": checkpoint_identity(args.model),
        "request": {"path": str(args.request.resolve()), "sha256": digest(args.request)},
        "gpu": torch.cuda.get_device_name(),
        "uuid": str(torch.cuda.get_device_properties(0).uuid),
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "precision": torch.backends.cuda.matmul.fp32_precision,
    }


def parameters(model):
    return {
        str(block.layer_idx): {
            "original": weight_identity(block.attention_layer),
            "prepared": {
                name: tensor_digest(value)
                for name, value in getattr(
                    block.attention_layer, "_q1_qkv_trial_weights", {}
                ).items()
            },
        }
        for block in model.blocks
    }


def cpu_outputs(model, token_ids, **kwargs):
    return {
        name: value.cpu()
        for name, value in model.forward(token_ids, return_hidden=True, **kwargs).items()
    }


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), required=True)
    parser.add_argument("--model", type=Path, default=Path("/preset-models"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
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
    spec = importlib.util.spec_from_file_location("_q1_selection_model_trial", args.candidate)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    ids = json.loads(args.request.read_text())["input_ids"]
    require(len(ids) == 65537 and ids[-1] == 111090, "Expected supplied H64K/A1 input")
    source = source_identity(args)
    archive = args.output / "source"
    archive.mkdir()
    for number, path in enumerate(source["sources"]):
        shutil.copy2(path, archive / f"{number}_{Path(path).name}")
    models, snapshots, graphs, expected, weights, memory, events = {}, {}, {}, {}, {}, {}, {}
    layer_events = {}
    result = {}
    try:
        with torch.inference_mode():
            for arm in ("baseline", "fused"):
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
                if arm == "fused":
                    candidate.bind_model(model)
                weights[arm] = parameters(model)
                model.prepare_compute_graphs([1024, 1])
                prefix = model.forward(ids[:65536]).cpu()
                snapshots[arm] = model.snapshot_prefix()
                if args.mode == "check":
                    eager = cpu_outputs(model, ids[65536:], use_extend_graph=False)
                    model.restore_prefix(snapshots[arm])
                begin_event = torch.cuda.Event(enable_timing=True, external=True)
                end_event = torch.cuda.Event(enable_timing=True, external=True)
                events[arm] = begin_event, end_event
                layer_begin = torch.cuda.Event(enable_timing=True, external=True)
                layer_end = torch.cuda.Event(enable_timing=True, external=True)
                layer_events[arm] = layer_begin, layer_end

                @contextmanager
                def graph_timing(
                    stage, begin=begin_event, end=end_event, first=layer_begin, last=layer_end
                ):
                    if stage == "extend_graph_body":
                        begin.record()
                        yield
                        end.record()
                    elif stage == "layer_0":
                        first.record()
                        yield
                    elif stage == "layer_2":
                        yield
                        last.record()
                    else:
                        yield

                graphs[arm] = model.prepare_extend_graph(
                    ids[65536:], return_hidden=True, capture_scope=graph_timing
                )
                for _ in range(5):
                    model.restore_prefix(snapshots[arm])
                    model.forward(ids[65536:], return_hidden=True)
                if args.mode == "check":
                    model.restore_prefix(snapshots[arm])
                    actual = cpu_outputs(model, ids[65536:])
                    for name, value in eager.items():
                        finite_exact(value, actual[name])
                    expected[arm] = {"prefix": prefix, **actual}
                memory[arm] = {
                    "graph": graphs[arm].describe(),
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "device_used": torch.cuda.device_memory_used(),
                    "additional_persistent_weight_bytes": sum(
                        getattr(
                            getattr(block.attention_layer, "_q1_input_projection", None),
                            "padding_weight_bytes",
                            0,
                        )
                        for block in model.blocks
                    ),
                }
                print("Prepared " + arm, flush=True)
            candidate_identity = candidate.source_identity()
            identity = {
                "source": source,
                "weights": weights,
                "runtime": runtime(),
                "candidate": candidate_identity,
            }
            if args.mode == "check":
                for name in expected["baseline"]:
                    finite_exact(expected["baseline"][name], expected["fused"][name])
                for changed in ([111091], [111092]):
                    outputs = {}
                    for arm, model in models.items():
                        model.restore_prefix(snapshots[arm])
                        outputs[arm] = cpu_outputs(model, changed)
                    for name in outputs["baseline"]:
                        finite_exact(outputs["baseline"][name], outputs["fused"][name])
                torch.save(expected, args.output / "outputs.pt")
                result = {
                    "passed": True,
                    "prefix_last_logits_and_all_extend_hidden_logits_exact": True,
                    "eager_graph_both_arms": True,
                    "changed_graph_inputs": 2,
                }
            else:
                receipt = require_receipt(args.check, kind=KIND, identity=identity)
                result["receipt_sha256"] = receipt["receipt_sha256"]
                if args.mode == "profile":
                    torch.cuda.cudart().cudaProfilerStart()
                    for arm, model in models.items():
                        for _ in range(3):
                            model.restore_prefix(snapshots[arm])
                            model.forward(ids[65536:], return_hidden=True)
                        model.restore_prefix(snapshots[arm])
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_push(f"q1_qkv_model/{arm}")
                        model.forward(ids[65536:], return_hidden=True)
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
                else:
                    samples = []
                    for pair in range(args.pairs):
                        for arm in (
                            ("baseline", "fused") if pair % 2 == 0 else ("fused", "baseline")
                        ):
                            model = models[arm]
                            model.restore_prefix(snapshots[arm])
                            torch.cuda.synchronize()
                            begin = time.perf_counter_ns()
                            model.forward(ids[65536:], return_hidden=True)
                            torch.cuda.synchronize()
                            elapsed_wall_ms = (time.perf_counter_ns() - begin) / 1e6
                            samples.append(
                                {
                                    "pair": pair,
                                    "arm": arm,
                                    "wall_ms": elapsed_wall_ms,
                                    "graph_ms": events[arm][0].elapsed_time(events[arm][1]),
                                    "layers_ms": layer_events[arm][0].elapsed_time(
                                        layer_events[arm][1]
                                    ),
                                }
                            )
                    result["samples"] = samples
                    result["median_wall_ms"] = {
                        arm: statistics.median(r["wall_ms"] for r in samples if r["arm"] == arm)
                        for arm in models
                    }
                    result["median_graph_ms"] = {
                        arm: statistics.median(r["graph_ms"] for r in samples if r["arm"] == arm)
                        for arm in models
                    }
                    result["median_layers_ms"] = {
                        arm: statistics.median(r["layers_ms"] for r in samples if r["arm"] == arm)
                        for arm in models
                    }
            require(source_identity(args) == source, "Source/checkpoint identity changed")
            require(runtime() == identity["runtime"], "Loaded backend identity changed")
            require(
                candidate.source_identity() == candidate_identity,
                "Candidate source/dispatch changed",
            )
            for arm, model in models.items():
                require(parameters(model) == weights[arm], "Weight bytes changed")
            payload = {
                "run_id": args.output.name,
                "mode": args.mode,
                "identity": identity,
                "memory": memory,
                "result": result,
                "boundary": "Independent HBM models, ordinary persistent append, complete synchronized model.forward with embedding/L0-L2/final norm/LM head, return_hidden=True; identical prefix restored outside every timed/profiled call; five warmups. Only Q1/k2048/N>=32768 selection changes: unchanged FlashInfer SMALL Filtered core followed by official CUB composite-key sort and nonfinite masking; all other model/cache work retained. Four external CUDA event nodes time the full graph body and L0-L2 in both arms; GPU sample collection is outside the synchronized wall timer.",
            }
    finally:
        primary = sys.exception()
        errors = []
        for model in models.values():
            try:
                model.close()
            except BaseException as error:  # noqa: BLE001 -- retain all cleanup exceptions
                errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "QKV model trial cleanup failed", ([primary] if primary else []) + errors
            )
    save(args.output / "result.json", payload)
    if args.mode == "check":
        write_receipt(
            args.output / "receipt.json",
            kind=KIND,
            identity=identity,
            checks=result,
            artifacts={
                "result": args.output / "result.json",
                "outputs": args.output / "outputs.pt",
            },
        )
    print(
        {
            "mode": args.mode,
            "accepted": True,
            "wall": result.get("median_wall_ms"),
            "graph": result.get("median_graph_ms"),
            "layers": result.get("median_layers_ms"),
        }
    )


if __name__ == "__main__":
    main()
