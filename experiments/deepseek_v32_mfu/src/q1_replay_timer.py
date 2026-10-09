"""Compare zero/two ordinary replay timing events on the same HBM Q1 graph."""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_mfu.src.full_graph_profile import FullExtendGraphCapture
from experiments.deepseek_v32_mfu.src.q1_event_boundary import (
    CUPTI,
    EXPERIMENT,
    ROOT,
    collecting,
    digest,
    environment,
    finite_exact,
    profiler_call,
    require,
    restore,
    runtime,
    save,
)
from experiments.deepseek_v32_mfu.src.q1_event_boundary import (
    source_identity as base_source_identity,
)
from models.deepseek_v32.model import DeepSeekEchoModel

KIND = "deepseek-q1-replay-timer-v1"
ARMS = ("plain", "timed")


def source_identity(args):
    result = base_source_identity(args)
    result["sources"][str(Path(__file__).resolve().relative_to(ROOT))] = digest(__file__)
    result["contract"] = {
        "history": 65536,
        "append": [111090],
        "layers": 3,
        "return_hidden": False,
        "cache": "hbm",
        "models_and_graphs": 1,
        "graph_warmups": 5,
        "warmup_scope": "plain; collection stopped",
        "in_trace_replay_warmups_per_arm": 1,
        "inspector": "FullExtendGraphCapture; construct under collection; zero internal events",
        "cupti": {"path": str(CUPTI), "sha256": digest(CUPTI)},
        "variable": "zero/two ordinary CUDA event records outside capture around replay submission",
        "timing": "synchronized complete forward wall only for both arms",
    }
    return result


def summarize(samples, pairs):
    delta = [
        next(row["wall_ms"] for row in samples if row["pair"] == pair and row["arm"] == "timed")
        - next(row["wall_ms"] for row in samples if row["pair"] == pair and row["arm"] == "plain")
        for pair in range(pairs)
    ]
    return {
        "wall_ms": {
            "median": {
                arm: statistics.median(row["wall_ms"] for row in samples if row["arm"] == arm)
                for arm in ARMS
            },
            "paired_delta_timed_minus_plain": delta,
            "paired_median_delta": statistics.median(delta),
            "timed_faster_pairs": sum(value < 0 for value in delta),
            "order_median_delta": {
                order: statistics.median(
                    delta[pair] for pair in range(pairs) if (pair % 2 == 0) == (order == "AB")
                )
                for order in ("AB", "BA")
            },
        }
    }


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
    provider = ctypes.CDLL(str(CUPTI))
    require(provider is not None, "Explicit CUPTI provider failed")
    require(torch.cuda.get_device_capability() == (9, 0), "Hopper is required")
    source = source_identity(args)
    ids = json.loads(args.request.read_text())["input_ids"]
    require(len(ids) == 65537 and ids[-1] == 111090, "Expected exact H64K/A1 input")
    base = (
        Path("/tmp/cxldsagr-checks/q1-replay-timer")
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
    model = None
    profiler_call("cudaProfilerStop")
    try:
        with torch.inference_mode():
            print("Loading shared model", flush=True)
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
            require(not model.offload, "HBM-only model expected")
            model.prepare_compute_graphs([1024, 1])
            prefix = model.forward(ids[:65536]).cpu()
            snapshot = model.snapshot_prefix()
            if args.mode == "check":
                eager = model.forward(
                    ids[65536:], return_hidden=False, use_extend_graph=False
                ).cpu()
                restore(model, snapshot)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            scopes = {}
            for arm in ARMS:

                @contextmanager
                def replay_scope(stage, enabled=arm == "timed"):
                    if stage == "extend_graph_replay_q_1":
                        require(
                            not torch.cuda.is_current_stream_capturing(), "Timer entered capture"
                        )
                        if enabled:
                            begin.record()
                        yield
                        if enabled:
                            end.record()
                    else:
                        yield

                scopes[arm] = replay_scope
            observer = FullExtendGraphCapture("hbm")
            with collecting():
                torch.cuda.nvtx.range_push("q1_replay_timer|construct=shared")
                graph = model.prepare_extend_graph(
                    ids[65536:], return_hidden=False, capture_scope=observer
                )
                template = observer.finalize(graph)
                torch.cuda.nvtx.range_pop()
            require(graph.replays == 0, "Full graph replayed before assigned warmups")
            native_graph = graph.graph
            cache_identity = graph.cache_identity
            kinds = list(template["node_types"].values())
            require(len(template["gpu_node_ids"]) == 197, "Expected 197 GPU nodes")
            require(all(kind in (0, 1, 2) for kind in kinds), "Internal event node or unknown kind")
            require(
                sum(
                    owner["layer"] in ("layer_0", "layer_1", "layer_2")
                    for owner in template["node_owners"].values()
                )
                == 192,
                "Expected 192 layer nodes",
            )

            def same_graph():
                require(graph.graph is native_graph, "Native graph object changed")
                require(graph.cache_identity == cache_identity, "Graph cache identity changed")

            for index in range(5):
                require(graph.replays == index, "Unexpected initial graph replay count")
                restore(model, snapshot)
                model.forward(ids[65536:], return_hidden=False, scope=scopes["plain"])
                torch.cuda.synchronize()
                same_graph()
            require(graph.replays == 5, "Expected five shared warmups")
            memory = {
                "graph": graph.describe(),
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "device_used": torch.cuda.device_memory_used(),
            }
            identity = {
                "source": source,
                "runtime": runtime(),
                "node_type_counts": {str(kind): kinds.count(kind) for kind in sorted(set(kinds))},
            }
            receipt = None
            if args.mode == "check":
                outputs = {}
                for arm in ARMS:
                    restore(model, snapshot)
                    outputs[arm] = model.forward(
                        ids[65536:], return_hidden=False, scope=scopes[arm]
                    ).cpu()
                    finite_exact(outputs[arm], eager)
                    same_graph()
                finite_exact(outputs["plain"], outputs["timed"])
                changed = []
                for token in (111091, 111092):
                    restore(model, snapshot)
                    reference = model.forward(
                        [token], return_hidden=False, use_extend_graph=False
                    ).cpu()
                    actual = {}
                    for arm in ARMS:
                        restore(model, snapshot)
                        actual[arm] = model.forward(
                            [token], return_hidden=False, scope=scopes[arm]
                        ).cpu()
                        finite_exact(actual[arm], reference)
                        same_graph()
                    finite_exact(actual["plain"], actual["timed"])
                    changed.append({"token": token, "eager": reference, "outputs": actual})
                torch.save(
                    {"prefix": prefix, "eager": eager, "outputs": outputs, "changed": changed},
                    destination / "outputs.pt",
                )
                result = {
                    "passed": True,
                    "eager_both_scopes_bitwise": True,
                    "changed_inputs_eager_both_scopes_bitwise": 2,
                    "same_native_graph_and_cache": True,
                }
            else:
                receipt = require_receipt(args.receipt, kind=KIND, identity=identity)
                samples = []
                pairs = 2 if args.mode == "profile" else args.pairs
                with collecting():
                    for arm in ARMS:
                        restore(model, snapshot)
                        torch.cuda.nvtx.range_push(f"q1_replay_timer|warmup={arm}")
                        model.forward(ids[65536:], return_hidden=False, scope=scopes[arm])
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                        same_graph()
                    for pair in range(pairs):
                        order = ARMS if pair % 2 == 0 else ARMS[::-1]
                        for arm in order:
                            restore(model, snapshot)
                            torch.cuda.nvtx.range_push(f"q1_replay_timer|pair={pair}|arm={arm}")
                            started = time.perf_counter_ns()
                            model.forward(ids[65536:], return_hidden=False, scope=scopes[arm])
                            torch.cuda.synchronize()
                            wall_ms = (time.perf_counter_ns() - started) / 1e6
                            torch.cuda.nvtx.range_pop()
                            same_graph()
                            samples.append(
                                {
                                    "pair": pair,
                                    "arm": arm,
                                    "order": "AB" if pair % 2 == 0 else "BA",
                                    "wall_ms": wall_ms,
                                }
                            )
                result = {"samples": samples, "summary": summarize(samples, pairs)}
            same_graph()
            require(source_identity(args) == source, "Source or environment changed")
            require(runtime() == identity["runtime"], "Loaded native/backend identity changed")
            payload = {
                "mode": args.mode,
                "run_id": args.run_id,
                "accepted": True,
                "identity": identity,
                "memory": memory,
                "template": template,
                "result": result,
                "receipt": None
                if receipt is None
                else {"path": str(args.receipt), "sha256": digest(args.receipt)},
                "boundary": "One current production HBM model/cache/full graph, H65536/A1/token111090, return_hidden=False. FullExtendGraphCapture under collection; zero internal event nodes; five shared plain-scope warmups with collection stopped. Plain/timed scopes replay the same native graph and cache after restored prefix, adding zero/two ordinary CUDA event records only around replay submission outside capture. One matched trace warmup per scope precedes ABBA pairs. Both use synchronized complete forward wall timing only; no GPU-event timing is compared. Same fixed explicit CUPTI provider in all phases. This diagnoses outside timing instrumentation, not production optimization.",
            }
    finally:
        original = sys.exception()
        if model is not None:
            try:
                model.close()
            except BaseException as cleanup:
                if original is not None:
                    raise BaseExceptionGroup(
                        "Execution and model cleanup failed", [original, cleanup]
                    )
                raise
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
