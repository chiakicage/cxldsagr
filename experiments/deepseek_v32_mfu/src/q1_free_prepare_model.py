"""Private complete-model A/B gate for bounded-free Q1 preparation mirrors."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from evaluation.local_native import collect_local_native_artifacts
from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_mfu.src.extend_graph_validation import cache_state
from experiments.deepseek_v32_mfu.src.prefetch_validation import ColdPrefetchObserver
from experiments.deepseek_v32_mfu.src.profile_layers import sources
from experiments.deepseek_v32_mfu.src.q1_free_prepare import allocate, inspect, make_states, restore
from experiments.deepseek_v32_mfu.src.q1_free_prepare_binding import DEFAULT, Binding
from experiments.deepseek_v32_mfu.src.q1_free_prepare_validation import (
    candidate_validation,
    compare_cache_state,
    validation_identity,
)
from experiments.deepseek_v32_mfu.src.q1_prepare_baseline import (
    DEFAULT_INPUT,
    digest,
    exact,
    require,
    write,
)
from experiments.deepseek_v32_mfu.src.q1_qkv import runtime
from experiments.deepseek_v32_mfu.src.run_contract import checkpoint_identity
from models.deepseek_v32.model import DeepSeekEchoModel
from operators.deepseek_v32.indexer import cache_ops

ROOT = Path(__file__).resolve().parents[3]
KIND = "deepseek-q1-free-prepare-private-model-v1"
ARMS = ("baseline", "candidate")


def source_identity(args, binding):
    files = {str((ROOT / name).resolve()): sha for name, sha in sources().items()}
    for name in (
        "q1_free_prepare_model.py",
        "q1_free_prepare_binding.py",
        "q1_free_prepare_validation.py",
        "q1_free_prepare.py",
        "q1_prepare_baseline.py",
        "q1_qkv.py",
    ):
        path = Path(__file__).with_name(name)
        files[str(path)] = digest(path)
    files.update(
        {str(binding.source / name): sha for name, sha in binding.before["candidate"].items()}
    )
    return {
        "files": files,
        "binding": binding.before,
        "validation": validation_identity(),
        "checkpoint": checkpoint_identity(args.model),
        "request": {"path": str(args.request.resolve()), "sha256": digest(args.request)},
        "component_input_sha256": digest(DEFAULT_INPUT),
        "gpu": torch.cuda.get_device_name(),
        "uuid": str(torch.cuda.get_device_properties(0).uuid),
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precision": torch.backends.cuda.matmul.fp32_precision,
    }


def restore_cold(model, snapshot):
    model.restore_prefix(snapshot)
    model.evict_prefix_residency()


def outputs(model, ids, *, eager=False):
    return {
        name: value.cpu()
        for name, value in model.forward(
            ids, return_hidden=True, use_extend_graph=not eager
        ).items()
    }


def exact_outputs(left, right):
    require(left.keys() == right.keys(), "Output fields differ")
    for name in left:
        require(
            bool(torch.isfinite(left[name]).all()) and bool(torch.isfinite(right[name]).all()),
            "Nonfinite output",
        )
        exact(left[name], right[name], "Complete model " + name)


def check_component(binding):
    rows = []
    with binding.installed():
        for state in make_states(DEFAULT_INPUT):
            case = allocate(state)
            restore(case, state)
            token = cache_ops.prepare_prefetch_free(
                case["inputs"]["priority"],
                case["inputs"]["bitmap"],
                case["inputs"]["reverse"],
                **case["buffers"],
                timestamp=state["clock"],
            )
            torch.cuda.synchronize()
            inspect("candidate", case, state, token)
            token.consume(case["buffers"], 1, consumer_limit=64)
            rows.append({"state": state["name"], "P": state["P"], "H": state["H"], "passed": True})
    return rows


def check_model(args, binding, model, snapshot, arm):
    saved = {}
    folder = args.output / arm
    folder.mkdir()
    restore_cold(model, snapshot)
    context = binding.installed() if arm == "candidate" else nullcontext()
    validation = candidate_validation() if arm == "candidate" else nullcontext()
    with context, validation, ColdPrefetchObserver(model, 65536, 1, folder) as observer:
        if arm == "candidate":
            # Declare the actually selected private contract; receipts retain
            # the intentional 64-vs-8192 preparation boundary separately.
            for row in observer.metadata:
                row["max_prefetch"] = 64
        for index, token_id in enumerate((111090, 111091, 111092)):
            restore_cold(model, snapshot)
            eager = outputs(model, [token_id], eager=True)
            eager_state = cache_state(model)
            eager_proof = observer.audit(f"input_{index}_eager", eager_state)
            restore_cold(model, snapshot)
            if index == 0:
                model.prepare_extend_graph([token_id], return_hidden=True)
            actual = outputs(model, [token_id])
            graph_state = cache_state(model)
            graph_proof = observer.audit(f"input_{index}_graph", graph_state, captured=True)
            exact_outputs(eager, actual)
            cache_comparison = compare_cache_state(
                graph_state,
                eager_state,
                actual_prefetch=graph_proof,
                expected_prefetch=eager_proof,
                prepared_cap=64 if arm == "candidate" else 8192,
            )
            saved[str(token_id)] = {
                "eager": eager,
                "graph": actual,
                "cache": graph_state,
                "proof": graph_proof,
                "cache_comparison": cache_comparison,
            }
    return saved


def check_clean_graph(model, snapshot, diagnostic):
    """Check the actual timing graph after diagnostic graphs have been destroyed."""
    saved = {}
    for token_id in (111090, 111091, 111092):
        restore_cold(model, snapshot)
        actual = outputs(model, [token_id])
        expected = diagnostic[str(token_id)]
        exact_outputs(actual, expected["eager"])
        # This validates both map directions, the free partition, every resident
        # record against host backing, and committed lengths on the clean graph.
        state = cache_state(model)
        for observed, reference in zip(state["layers"], expected["cache"]["layers"], strict=True):
            for key in (
                "length",
                "written",
                "indexer_visible_end",
                "records",
                "index_keys",
                "index_scales",
                "hint",
                "clock",
            ):
                require(observed[key] == reference[key], "Clean graph changed " + key)
            require(observed["map_invariants_passed"] is True, "Clean graph map check failed")
        saved[str(token_id)] = {
            "output": actual,
            "cache": state,
            "bitwise_equal_to_eager": True,
            "boundary": "Actual clean timing graph: outputs and immutable cache data match diagnostic eager execution; maps/free partition/resident record bytes independently checked. Actual-stage prediction oracle is supplied by the separate diagnostic graph only.",
        }
    return saved


def compare_arms(accepted):
    comparisons = []
    # These differences already require each independent stage's exact CPU
    # oracle proof. Capped official predictions may differ with CTA scheduling.
    schedule_keys = {"resident", "logical_priority", "free_count"}
    schedule_metrics = {
        "prefetched_records",
        "prefetch_capacity_failures",
        "recalled_records",
        "resident_selection_records",
        "host_to_device_bytes",
    }
    for token_id in accepted["baseline"]:
        left, right = accepted["baseline"][token_id], accepted["candidate"][token_id]
        exact_outputs(left["graph"], right["graph"])
        for name in ("indices", "scores", "initial_hints"):
            require(left["proof"][name] == right["proof"][name], "Cross-arm " + name + " differs")
        require(
            left["proof"]["scope"]["prepared_max_prefetch"] == 8192,
            "Baseline preparation cap changed",
        )
        require(
            right["proof"]["scope"]["prepared_max_prefetch"] == 64,
            "Candidate preparation cap changed",
        )
        require(
            left["proof"]["scope"]["max_prefetch"] == right["proof"]["scope"]["max_prefetch"] == 64,
            "Official effective cap differs",
        )
        differences = []
        for index, (a, b) in enumerate(
            zip(left["cache"]["layers"], right["cache"]["layers"], strict=True)
        ):
            for field in a.keys() | b.keys():
                if a.get(field) == b.get(field):
                    continue
                differences.append(f"layer_{index}.{field}")
                if field == "metrics":
                    require(
                        {k: v for k, v in a[field].items() if k not in schedule_metrics}
                        == {k: v for k, v in b[field].items() if k not in schedule_metrics},
                        "Nonscheduling cache metrics differ",
                    )
                else:
                    require(field in schedule_keys, "Nonscheduling cache state differs: " + field)
        comparisons.append(
            {
                "token": token_id,
                "exact_outputs_scores_topk_hints": True,
                "both_actual_stage_oracles_passed": True,
                "schedule_dependent_differences": differences,
                "prepared_cap": {"baseline": 8192, "candidate": 64},
            }
        )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), required=True)
    parser.add_argument("--model", type=Path, default=Path("/preset-models"))
    parser.add_argument("--mirrors", type=Path, default=DEFAULT)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--pairs", type=int, default=100)
    args = parser.parse_args()
    require(args.pairs >= 100 and args.pairs % 2 == 0, "At least 100 balanced pairs required")
    require(torch.cuda.get_device_capability() == (9, 0), "SM90 required")
    from triton.backends.nvidia.compiler import get_ptxas

    os.environ.setdefault("TRITON_PTXAS_PATH", get_ptxas(90).path)
    os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/usr/local/cuda/bin/ptxas")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    ids = json.loads(args.request.read_text())["input_ids"]
    require(len(ids) == 65537 and ids[-1] == 111090, "Expected supplied H64K/A1 request")
    args.output.mkdir(parents=True, exist_ok=False)
    binding = Binding(args.mirrors)
    source = source_identity(args, binding)
    for filename in source["files"]:
        path = Path(filename)
        target = args.output / "source" / path.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    binding.load_cuda()
    write(args.output / "private_native.json", binding.native)
    shutil.copyfile(binding.native["artifact_path"], args.output / "private_native.so")
    models, snapshots, accepted, prefix, memory, clean = {}, {}, {}, {}, {}, {}
    result = {"passed": True}
    try:
        with torch.inference_mode():
            if args.mode == "check":
                result["new_integration_native_component_cases"] = check_component(binding)
            for arm in ARMS:
                print("Loading " + arm, flush=True)
                model = DeepSeekEchoModel(
                    args.model,
                    devices=[0],
                    num_layers=3,
                    capacity=65537,
                    offload=True,
                    slots=65600,
                    chunk_size=1024,
                    extend_chunk_size=1,
                    host_arena_tokens=65600,
                    workspace_query_tokens=1024,
                    hbm_cache_budget_bytes=24 * 2**30,
                    dram_cache_budget_bytes=64 * 2**30,
                )
                models[arm] = model
                model.prepare_compute_graphs([1024, 1])
                prefix_value = model.forward(ids[:65536])
                if args.mode == "check":
                    prefix[arm] = prefix_value.cpu()
                snapshots[arm] = model.snapshot_prefix()
                if args.mode == "check":
                    accepted[arm] = check_model(args, binding, model, snapshots[arm], arm)
                restore_cold(model, snapshots[arm])
                before_calls = binding.calls
                with binding.installed() if arm == "candidate" else nullcontext():
                    graph = model.prepare_extend_graph(ids[65536:], return_hidden=True)
                    for _ in range(5):
                        restore_cold(model, snapshots[arm])
                        model.forward(ids[65536:], return_hidden=True)
                    if args.mode == "check":
                        clean[arm] = check_clean_graph(model, snapshots[arm], accepted[arm])
                require(
                    (binding.calls > before_calls) is (arm == "candidate"),
                    "Wrong arm preparation dispatch",
                )
                memory[arm] = {
                    "graph": graph.describe(),
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "device_used": torch.cuda.device_memory_used(),
                    "candidate_prepare_capture_calls": binding.calls - before_calls,
                }
                print("Prepared " + arm, flush=True)
            native = collect_local_native_artifacts(required=True)
            identity = {
                "source": source,
                "private_native": binding.native,
                "mapped_native": native,
                "runtime": runtime(),
                "boundary": "Independent ECHO models, cold H64K/A1, ordinary persistent append, complete graph without observer or event nodes; return_hidden=True",
            }
            if args.mode == "check":
                exact(prefix["baseline"], prefix["candidate"], "Prefix logits")
                result["cross_arm"] = compare_arms(accepted)
                for token_id in clean["baseline"]:
                    exact_outputs(
                        clean["baseline"][token_id]["output"],
                        clean["candidate"][token_id]["output"],
                    )
                result["clean_graph_inputs_per_arm"] = 3
                result["clean_graph_outputs_bitwise_and_cache_invariants_checked"] = True
                torch.save(
                    {"prefix": prefix, "accepted": accepted, "clean": clean},
                    args.output / "outputs.pt",
                )
            else:
                receipt = require_receipt(args.receipt, kind=KIND, identity=identity)
                result["receipt_sha256"] = receipt["receipt_sha256"]
                samples = []
                if args.mode == "profile":
                    torch.cuda.cudart().cudaProfilerStart()
                for pair in range(args.pairs if args.mode == "bench" else 1):
                    order = ARMS if pair % 2 == 0 else ARMS[::-1]
                    for arm in order:
                        model = models[arm]
                        restore_cold(model, snapshots[arm])
                        torch.cuda.synchronize()
                        context = binding.installed() if arm == "candidate" else nullcontext()
                        with context:
                            if args.mode == "profile":
                                torch.cuda.nvtx.range_push("q1_free_prepare_model/" + arm)
                            begin = time.perf_counter_ns()
                            model.forward(ids[65536:], return_hidden=True)
                            torch.cuda.synchronize()
                            elapsed = (time.perf_counter_ns() - begin) / 1e6
                            if args.mode == "profile":
                                torch.cuda.nvtx.range_pop()
                        samples.append(
                            {"pair": pair, "order": list(order), "arm": arm, "wall_ms": elapsed}
                        )
                if args.mode == "profile":
                    torch.cuda.cudart().cudaProfilerStop()
                else:
                    result["median_wall_ms"] = {
                        arm: statistics.median(
                            row["wall_ms"] for row in samples if row["arm"] == arm
                        )
                        for arm in ARMS
                    }
                result["samples"] = samples
            binding.verify()
            require(
                source_identity(args, binding) == source, "Model/source/checkpoint identity changed"
            )
            require(
                collect_local_native_artifacts(required=True) == native, "Mapped native changed"
            )
            require(runtime() == identity["runtime"], "Runtime changed")
            for entry in native:
                shutil.copyfile(entry["library"]["path"], args.output / entry["name"])
            payload = {
                "passed": True,
                "run_id": args.output.name,
                "mode": args.mode,
                "identity": identity,
                "result": result,
                "memory": memory,
                "boundary": "Wall time surrounds complete model.forward+synchronize. Cold prefix restore and process-local binding changes are outside the interval. Timing graphs contain no observer/event nodes. Check uses separate diagnostic graphs with independent actual-stage CPU proofs; those graphs are replaced before timing graph setup.",
            }
    finally:
        primary, errors = sys.exception(), []
        for model in models.values():
            try:
                with torch.inference_mode():
                    model.close()
            except BaseException as error:  # noqa: BLE001 -- preserve primary and all cleanup failures
                errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "Private model cleanup failed", ([primary] if primary else []) + errors
            )
    write(args.output / "result.json", payload)
    if args.mode == "check":
        write_receipt(
            args.output / "receipt.json",
            kind=KIND,
            identity=identity,
            checks=result,
            artifacts={
                str(path.relative_to(args.output)): path
                for path in sorted(args.output.rglob("*"))
                if path.is_file()
            },
        )
    print(
        json.dumps(
            {"mode": args.mode, "passed": True, "median_wall_ms": result.get("median_wall_ms")}
        )
    )


if __name__ == "__main__":
    main()
