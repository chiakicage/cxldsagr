"""Control collection during the first five replays of identical HBM Q1 graphs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_mfu.src.backend_provenance import (
    collect_backend_provenance,
    collect_flashinfer_runtime_artifacts,
)
from experiments.deepseek_v32_mfu.src.full_graph_profile import FullExtendGraphCapture
from experiments.deepseek_v32_mfu.src.profile_layers import sources
from experiments.deepseek_v32_mfu.src.run_contract import checkpoint_identity
from models.deepseek_v32.model import DeepSeekEchoModel

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
KIND = "deepseek-q1-replay-boundary-v1"
ARMS = ("first5_outside", "first5_inside")
CUPTI = Path("/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/libcupti.so.13.2")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def finite_exact(left, right):
    require(torch.isfinite(left).all().item() and torch.isfinite(right).all().item(), "Nonfinite")
    require(
        torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)),
        "Output bits differ",
    )


def profiler_call(name):
    code = getattr(torch.cuda.cudart(), name)()
    require(int(code) == 0, f"{name} failed: {code}")


@contextmanager
def collecting():
    profiler_call("cudaProfilerStart")
    try:
        yield
    except BaseException as original:
        try:
            profiler_call("cudaProfilerStop")
        except BaseException as cleanup:  # noqa: BLE001 -- preserve both failures
            raise BaseExceptionGroup("Profiler execution and stop failed", [original, cleanup])
        raise
    else:
        profiler_call("cudaProfilerStop")


def environment():
    root = EXPERIMENT / "output/runtime/q1-replay-boundary"
    for variable, suffix in (
        ("TRITON_CACHE_DIR", "triton"),
        ("DG_JIT_CACHE_DIR", "deep-gemm"),
        ("TVM_FFI_CACHE_DIR", "tvm"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("TORCH_EXTENSIONS_DIR", "torch"),
        ("TMPDIR", "tmp"),
    ):
        path = root / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    from triton.backends.nvidia.compiler import get_ptxas

    os.environ.setdefault("TRITON_PTXAS_PATH", get_ptxas(90).path)
    os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/usr/local/cuda/bin/ptxas")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.fp32_precision = "ieee"


def source_identity(args):
    files = sources()
    files[str(Path(__file__).resolve().relative_to(ROOT))] = digest(__file__)
    return {
        "sources": files,
        "checkpoint": checkpoint_identity(args.model),
        "request": {"path": str(args.request.resolve()), "sha256": digest(args.request)},
        "gpu": torch.cuda.get_device_name(),
        "uuid": str(torch.cuda.get_device_properties(0).uuid),
        "physical_device": args.physical_device,
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "precision": torch.backends.cuda.matmul.fp32_precision,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TRITON_CACHE_DIR",
                "DG_JIT_CACHE_DIR",
                "TVM_FFI_CACHE_DIR",
                "FLASHINFER_WORKSPACE_BASE",
                "FLASHINFER_CUBIN_DIR",
            )
        },
        "contract": {
            "history": 65536,
            "append": [111090],
            "layers": 3,
            "return_hidden": False,
            "cache": "hbm",
            "graph_warmups": 5,
            "first_replay_collection": "first5_outside off / first5_inside on",
            "in_trace_replay_warmups": 1,
            "inspector": "same FullExtendGraphCapture in both arms; both constructions collected",
            "cupti": {"path": str(CUPTI), "sha256": digest(CUPTI)},
            "events": "zero internal graph nodes; same ordinary CUDA events outside capture around replay submission",
            "variable": "collection off/on during the first five full graph replays",
        },
    }


def runtime():
    return {
        "backend": collect_backend_provenance(),
        "loaded_jit": collect_flashinfer_runtime_artifacts(),
    }


def restore(model, snapshot):
    model.restore_prefix(snapshot)
    require(model.length == 65536, "Prefix restoration length changed")
    torch.cuda.synchronize()


def summarize(samples, pairs):
    result = {}
    for metric in ("gpu_replay_ms", "wall_ms"):
        deltas = [
            next(r[metric] for r in samples if r["pair"] == pair and r["arm"] == "first5_inside")
            - next(r[metric] for r in samples if r["pair"] == pair and r["arm"] == "first5_outside")
            for pair in range(pairs)
        ]
        result[metric] = {
            "median": {
                arm: statistics.median(r[metric] for r in samples if r["arm"] == arm)
                for arm in ARMS
            },
            "paired_delta_inside_minus_outside": deltas,
            "paired_median_delta": statistics.median(deltas),
            "inside_faster_pairs": sum(value < 0 for value in deltas),
            "order_median_delta": {
                order: statistics.median(
                    deltas[p] for p in range(pairs) if (p % 2 == 0) == (order == "AB")
                )
                for order in ("AB", "BA")
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "bench", "profile"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, default=Path("/preset-models"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--pairs", type=int, default=50)
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id and args.pairs >= 2, "Invalid run or pairs")
    environment()
    cupti_provider = ctypes.CDLL(str(CUPTI))
    require(cupti_provider is not None, "Explicit CUPTI provider failed")
    require(torch.cuda.get_device_capability() == (9, 0), "Hopper is required")
    source = source_identity(args)
    ids = json.loads(args.request.read_text())["input_ids"]
    require(len(ids) == 65537 and ids[-1] == 111090, "Expected exact H64K/A1 input")
    base = (
        Path("/tmp/cxldsagr-checks/q1-replay-boundary")
        if args.mode == "check"
        else EXPERIMENT / "output/data"
    )
    destination = base / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for relative in source["sources"]:
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        artifacts["source/" + relative] = target
    shutil.copy2(args.request, destination / "request.json")
    artifacts["request"] = destination / "request.json"
    models, snapshots, graphs, events, memory, expected = {}, {}, {}, {}, {}, {}
    templates, replay_scopes, first_warmups, lifecycle = {}, {}, {}, {}
    profiler_call("cudaProfilerStop")
    try:
        with torch.inference_mode():
            for arm in ARMS:
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
                require(not model.offload, "HBM-only model expected")
                model.prepare_compute_graphs([1024, 1])
                prefix = model.forward(ids[:65536]).cpu()
                snapshots[arm] = model.snapshot_prefix()
                if args.mode == "check":
                    eager = model.forward(
                        ids[65536:], return_hidden=False, use_extend_graph=False
                    ).cpu()
                    restore(model, snapshots[arm])
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                events[arm] = begin, end

                @contextmanager
                def replay_timing(stage, start=begin, stop=end):
                    if stage == "extend_graph_replay_q_1":
                        require(
                            not torch.cuda.is_current_stream_capturing(),
                            "Replay timer entered capture",
                        )
                        start.record()
                        yield
                        stop.record()
                    else:
                        yield

                replay_scopes[arm] = replay_timing
                observer = FullExtendGraphCapture("hbm")
                with collecting():
                    torch.cuda.nvtx.range_push(f"q1_replay_boundary|construct={arm}")
                    graphs[arm] = model.prepare_extend_graph(
                        ids[65536:], return_hidden=False, capture_scope=observer
                    )
                    templates[arm] = observer.finalize(graphs[arm])
                    torch.cuda.nvtx.range_pop()
                require(graphs[arm].replays == 0, "Full graph replayed before assigned warmups")
                first_warmups[arm] = []
                with collecting() if arm == "first5_inside" else nullcontext():
                    for index in range(5):
                        require(graphs[arm].replays == index, "Unexpected full graph replay count")
                        restore(model, snapshots[arm])
                        torch.cuda.nvtx.range_push(
                            f"q1_replay_boundary|first_warmup={index}|arm={arm}"
                        )
                        started = time.perf_counter_ns()
                        model.forward(ids[65536:], return_hidden=False, scope=replay_scopes[arm])
                        torch.cuda.synchronize()
                        wall_ms = (time.perf_counter_ns() - started) / 1e6
                        torch.cuda.nvtx.range_pop()
                        first_warmups[arm].append(
                            {
                                "index": index,
                                "collection": arm == "first5_inside",
                                "gpu_replay_ms": begin.elapsed_time(end),
                                "wall_ms": wall_ms,
                            }
                        )
                require(graphs[arm].replays == 5, "Expected exactly five assigned warmups")
                lifecycle[arm] = {
                    "after_construction": 0,
                    "after_first_warmups": graphs[arm].replays,
                }
                if args.mode == "check":
                    restore(model, snapshots[arm])
                    actual = model.forward(
                        ids[65536:], return_hidden=False, scope=replay_scopes[arm]
                    ).cpu()
                    finite_exact(actual, eager)
                    expected[arm] = {"prefix": prefix, "eager": eager, "graph": actual}
                memory[arm] = {
                    "graph": graphs[arm].describe(),
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "device_used": torch.cuda.device_memory_used(),
                }
                print("Prepared " + arm, flush=True)
            node_counts = {}
            for arm, template in templates.items():
                kinds = list(template["node_types"].values())
                require(len(template["gpu_node_ids"]) == 197, "Expected 197 GPU nodes")
                require(kinds.count(7) == 0, "Internal event nodes are forbidden")
                require(all(kind in (0, 1, 2) for kind in kinds), "Unexpected graph node type")
                require(
                    sum(
                        owner["layer"] in ("layer_0", "layer_1", "layer_2")
                        for owner in template["node_owners"].values()
                    )
                    == 192,
                    "Expected 192 layer nodes",
                )
                node_counts[arm] = {str(kind): kinds.count(kind) for kind in sorted(set(kinds))}
            identity = {"source": source, "runtime": runtime(), "node_type_counts": node_counts}
            receipt = None
            if args.mode == "check":
                for key in expected["first5_outside"]:
                    finite_exact(expected["first5_outside"][key], expected["first5_inside"][key])
                changed = []
                for token in (111091, 111092):
                    outputs = {}
                    for arm in ARMS:
                        restore(models[arm], snapshots[arm])
                        outputs[arm] = (
                            models[arm]
                            .forward([token], return_hidden=False, scope=replay_scopes[arm])
                            .cpu()
                        )
                    finite_exact(outputs["first5_outside"], outputs["first5_inside"])
                    changed.append({"token": token, "outputs": outputs})
                torch.save({"expected": expected, "changed": changed}, destination / "outputs.pt")
                result = {
                    "passed": True,
                    "eager_graph_both_arms_bitwise": True,
                    "prefix_and_both_arms_bitwise": True,
                    "changed_graph_inputs": 2,
                }
            else:
                receipt = require_receipt(args.receipt, kind=KIND, identity=identity)
                samples = []
                pairs = 2 if args.mode == "profile" else args.pairs
                with collecting():
                    for arm in ARMS:
                        restore(models[arm], snapshots[arm])
                        torch.cuda.nvtx.range_push(f"q1_replay_boundary|warmup={arm}")
                        models[arm].forward(
                            ids[65536:], return_hidden=False, scope=replay_scopes[arm]
                        )
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                    for pair in range(pairs):
                        order = ARMS if pair % 2 == 0 else ARMS[::-1]
                        for arm in order:
                            restore(models[arm], snapshots[arm])
                            torch.cuda.nvtx.range_push(f"q1_replay_boundary|pair={pair}|arm={arm}")
                            start = time.perf_counter_ns()
                            models[arm].forward(
                                ids[65536:], return_hidden=False, scope=replay_scopes[arm]
                            )
                            torch.cuda.synchronize()
                            wall_ms = (time.perf_counter_ns() - start) / 1e6
                            torch.cuda.nvtx.range_pop()
                            samples.append(
                                {
                                    "pair": pair,
                                    "arm": arm,
                                    "order": "AB" if pair % 2 == 0 else "BA",
                                    "wall_ms": wall_ms,
                                    "gpu_replay_ms": events[arm][0].elapsed_time(events[arm][1]),
                                }
                            )
                result = {"samples": samples, "summary": summarize(samples, pairs)}
            require(source_identity(args) == source, "Production source or environment changed")
            require(runtime() == identity["runtime"], "Loaded native/backend identity changed")
            payload = {
                "mode": args.mode,
                "run_id": args.run_id,
                "accepted": True,
                "identity": identity,
                "memory": memory,
                "templates": templates,
                "first_warmups": first_warmups,
                "lifecycle": lifecycle,
                "result": result,
                "receipt": None
                if receipt is None
                else {"path": str(args.receipt), "sha256": digest(args.receipt)},
                "boundary": "Same current HBM production, independent model/cache/graph arms, H65536/A1/token111090, return_hidden=False; both use FullExtendGraphCapture and construct under collection, with zero internal event nodes. Only collection during the first five restored-prefix full graph replays differs: off for first5_outside and on for first5_inside. Zero replays after construction and exactly five assigned first warmups are asserted. One matched in-trace warmup per arm precedes formal AB/BA timing. Both arms use identical ordinary CUDA events around extend_graph_replay_q_1 outside capture, excluding input staging and post-replay host commit; wall time includes synchronized complete forward. Same explicit NSYS CUPTI provider in all phases/arms. This diagnoses first replay/upload collection state, not a production optimization.",
            }
    finally:
        original = sys.exception()
        errors = []
        for model in models.values():
            try:
                model.close()
            except BaseException as error:  # noqa: BLE001 -- retain all cleanup exceptions
                errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "Capture boundary cleanup failed", ([original] if original else []) + errors
            )
    save(destination / "result.json", payload)
    if args.mode == "check":
        artifacts.update(
            {"result": destination / "result.json", "outputs": destination / "outputs.pt"}
        )
        write_receipt(
            destination / "receipt.json",
            kind=KIND,
            identity=identity,
            checks=result,
            artifacts=artifacts,
        )
    print(destination / "result.json", flush=True)
    print(result.get("summary", result), flush=True)


if __name__ == "__main__":
    main()
