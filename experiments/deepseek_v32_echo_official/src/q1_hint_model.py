"""Private exact-hint gate on complete cold H64K/A1 checkpoint L0-L2 forwards."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.local_native import collect_local_native_artifacts
from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_echo_official.src import q1_hint_baseline as base
from experiments.deepseek_v32_echo_official.src import q1_hint_candidate as component
from experiments.deepseek_v32_echo_official.src import q1_hint_exact as candidate
from experiments.deepseek_v32_mfu.src.extend_graph_validation import (
    cache_state,
    compare_cache_state,
    tensor_identity,
)
from experiments.deepseek_v32_mfu.src.full_graph_profile import FullExtendGraphCapture
from experiments.deepseek_v32_mfu.src.prefetch_validation import ColdPrefetchObserver
from experiments.deepseek_v32_mfu.src.profile_layers import sources
from experiments.deepseek_v32_mfu.src.q1_qkv import runtime as model_runtime
from experiments.deepseek_v32_mfu.src.run_contract import checkpoint_identity
from models.deepseek_v32 import attention as attention_module
from models.deepseek_v32.model import DeepSeekEchoModel
from operators.deepseek_v32.indexer import prefetch_hint

KIND = "deepseek-q1-hint-exact-private-model-v1"
ARMS = ("baseline", "candidate")
TOKENS = (111090, 111091, 111092)
HISTORY = 65536
DEFAULT_COMPONENT = Path(
    "/tmp/cxldsagr-checks/q1-hint-candidate/q1_hint_candidate_check_20261008_01/receipt.json"
)
DEFAULT_BENCH = base.EXPERIMENT / "output/data/q1_hint_candidate_bench_20261008_01/result.json"
EXTRA_SOURCES = (
    "experiments/deepseek_v32_echo_official/src/q1_hint_model.py",
    "experiments/deepseek_v32_mfu/src/q1_qkv.py",
    "experiments/deepseek_v32_mfu/src/q1_control.py",
    "experiments/deepseek_v32_mfu/src/q1_projection.py",
)
CONTRACT = {
    "layers": [0, 1, 2],
    "history": HISTORY,
    "append": 1,
    "tokens": list(TOKENS),
    "capacity": 65537,
    "slots": 65600,
    "host_arena_tokens": 65600,
    "chunk_size": 1024,
    "extend_chunk_size": 1,
    "workspace_query_tokens": 1024,
    "hbm_cache_budget_bytes": 24 * 2**30,
    "dram_cache_budget_bytes": 64 * 2**30,
    "compute_graphs": [1024, 1],
    "warmups": 5,
    "minimum_balanced_pairs": 100,
    "prefetch": "Official ECHO Q1, actual bounded preparation and effective cap both 64",
    "candidate_binding": "Only models.deepseek_v32.attention.update_prefetch_hint -> q1_hint_exact.update",
    "timing": "Complete forward(return_hidden=True) plus synchronize; cold prefix/hint restore, binding, metrics and all diagnostics outside timing",
    "consumer_check": "Independent capacity65539/extend_chunk2 models; actual Q1 then A2, residency eviction only between calls, no restore/truncate",
}


def cleanup(actions):
    """Run every cleanup and retain the active exception without replacing it."""
    primary, errors = sys.exception(), []
    for action in actions:
        try:
            action()
        except BaseException as error:  # noqa: BLE001 -- retain every cleanup failure
            errors.append(error)
    if errors:
        raise BaseExceptionGroup(
            "Execution and cleanup failed", ([primary] if primary is not None else []) + errors
        )


@contextmanager
def binding(arm):
    base.require(arm in ARMS, "Unknown arm")
    original = prefetch_hint.update_prefetch_hint
    base.require(
        attention_module.update_prefetch_hint is original, "Production hint binding changed"
    )
    context = (
        patch.object(attention_module, "update_prefetch_hint", candidate.update)
        if arm == "candidate"
        else nullcontext()
    )
    context.__enter__()
    try:
        yield
    finally:
        primary = sys.exception()
        cleanup(
            (
                lambda: context.__exit__(
                    type(primary) if primary is not None else None,
                    primary,
                    primary.__traceback__ if primary is not None else None,
                ),
                lambda: base.require(
                    attention_module.update_prefetch_hint is original,
                    "Hint binding was not restored",
                ),
            )
        )


def profiler_api(start):
    result = (
        torch.cuda.cudart().cudaProfilerStart() if start else torch.cuda.cudart().cudaProfilerStop()
    )
    base.require(int(result) == 0, "CUDA profiler API failed")


@contextmanager
def profiler_range(enabled):
    if not enabled:
        yield
        return
    profiler_api(True)
    try:
        yield
    finally:
        cleanup((lambda: profiler_api(False),))


@contextmanager
def nvtx_range(enabled, label):
    if not enabled:
        yield
        return
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        cleanup((torch.cuda.nvtx.range_pop,))


def exact(left, right, label):
    base.require(
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(base.bits(left), base.bits(right)),
        "Bitwise mismatch: " + label,
    )


def exact_outputs(left, right):
    base.require(left.keys() == right.keys(), "Output fields differ")
    for key in left:
        base.require(
            bool(torch.isfinite(left[key]).all()) and bool(torch.isfinite(right[key]).all()),
            "Nonfinite model output",
        )
        exact(left[key], right[key], "model " + key)


def offsets(model):
    return [block.attention.offset.detach().cpu().clone() for block in model.blocks]


def exact_offsets(left, right):
    base.require(len(left) == len(right) == 3, "Expected three offset vectors")
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        base.require(a.shape == b.shape == (16,), "Expected all 16 offsets")
        exact(a, b, f"layer {index} all offsets")


def outputs(model, ids, *, eager=False):
    result = model.forward(ids, return_hidden=True, use_extend_graph=not eager)
    return {name: value.detach().cpu().clone() for name, value in result.items()}


def restore_cold(model, snapshot):
    model.restore_prefix(snapshot)
    model.evict_prefix_residency()


def component_binding(args):
    raw = json.loads(args.component_receipt.read_text())
    receipt = require_receipt(args.component_receipt, kind=component.KIND, identity=raw["identity"])
    checks = receipt["checks"]
    base.require(
        checks.get("comparisons") == 78
        and checks.get("bitwise_offset_owner_values") is True
        and checks.get("changed_graph_inputs") == 3
        and checks.get("nondefault_stream_per_arm") is True,
        "Component acceptance is incomplete",
    )
    expected = receipt["identity"]
    for relative, digest in {
        **expected["baseline"]["sources"],
        **expected["sources"],
    }.items():
        base.require(
            base.digest(base.ROOT / relative) == digest, "Component source changed: " + relative
        )
    base.require(candidate.build_info() == expected["candidate_build"], "Candidate build changed")
    accepted = json.loads(Path(receipt["artifact_paths"]["result.json"]).read_text())
    bench = json.loads(args.component_bench.read_text())
    base.require(
        bench.get("completed") is True
        and bench.get("mode") == "bench"
        and bench.get("identity") == expected
        and bench.get("pairs", 0) >= 100
        and bench.get("pairs", 1) % 2 == 0
        and bench.get("receipt", {}).get("sha256") == base.digest(args.component_receipt),
        "Component benchmark does not bind the accepted candidate",
    )
    component.require_runtime(bench["runtime"], accepted["runtime"])
    summaries = bench["payload"]["summary"]
    base.require(
        len(summaries) == 6
        and all(
            row["median_paired_delta_gpu_us"] < 0
            and all(value < 0 for value in row["order_median_delta_gpu_us"].values())
            for row in summaries
        ),
        "Component benchmark did not improve every layer/execution/order stratum",
    )
    return {
        "receipt": {
            "path": str(args.component_receipt.resolve()),
            "sha256": base.digest(args.component_receipt),
        },
        "benchmark": {
            "path": str(args.component_bench.resolve()),
            "sha256": base.digest(args.component_bench),
        },
        "identity": expected,
        "native": accepted["runtime"]["native"],
        "component_runtime": accepted["runtime"],
    }


def source_identity(args, accepted):
    files = dict(sources())
    for relative in (*base.SOURCE_PATHS, *component.EXTRA_SOURCES, *EXTRA_SOURCES):
        files[relative] = base.digest(base.ROOT / relative)
    props = torch.cuda.get_device_properties(0)
    return {
        "sources": files,
        "component": accepted,
        "checkpoint": checkpoint_identity(args.model),
        "request": {"path": str(args.request.resolve()), "sha256": base.digest(args.request)},
        "contract": CONTRACT,
        "gpu": {
            "name": props.name,
            "uuid": str(props.uuid),
            "capability": [props.major, props.minor],
        },
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "precision": torch.backends.cuda.matmul.fp32_precision,
        "threads": torch.get_num_threads(),
        "environment": {
            name: os.environ.get(name)
            for name in (*base.ENV_KEYS, "TVM_FFI_CACHE_DIR", "TVM_FFI_CUDA_ARCH_LIST")
        },
    }


def runtime(destination=None):
    return {
        "model": model_runtime(),
        "hint": component.runtime(destination),
        "mapped_native": collect_local_native_artifacts(required=True),
    }


def require_component_native(observed, accepted):
    base.require(
        observed["hint"]["native"] == accepted["native"], "Private DSO differs from component check"
    )
    for name, digest in accepted["component_runtime"]["loaded_libraries"].items():
        base.require(
            observed["hint"]["loaded_libraries"].get(name) == digest,
            "Loaded component library changed: " + name,
        )


def observed_record(model, observer, label, value, *, captured=False):
    state = cache_state(model)
    proof = observer.audit(label, state, captured=captured)
    targets = observer.captured if captured else observer.eager
    return {
        "output": value,
        "offsets": offsets(model),
        "cache": state,
        "proof": proof,
        "observed": [
            {
                key: target[key].detach().cpu().clone()
                for key in ("scores", "indices", "initial_hint")
            }
            for target in targets
        ],
    }


def require_q1_proof(record):
    scope = record["proof"]["scope"]
    base.require(
        scope.get("A") == 1
        and scope.get("prepared_max_prefetch") == scope.get("max_prefetch") == 64
        and scope.get("hint_index") == 1
        and scope.get("preparation", {}).get("prepared_limit") == 64,
        "Actual official Q1 preparation/policy differs",
    )


def compare_records(left, right):
    exact_outputs(left["output"], right["output"])
    exact_offsets(left["offsets"], right["offsets"])
    for index, (a, b) in enumerate(zip(left["observed"], right["observed"], strict=True)):
        for key in a:
            exact(a[key], b[key], f"observed layer {index} {key}")
    return compare_cache_state(
        left["cache"],
        right["cache"],
        actual_prefetch=left["proof"],
        expected_prefetch=right["proof"],
    )


def check_model(model, snapshot, arm, folder):
    folder.mkdir()
    saved = {}
    restore_cold(model, snapshot)
    with (
        binding(arm),
        component.audited_dispatch() as counter,
        ColdPrefetchObserver(model, HISTORY, 1, folder) as observer,
    ):
        for index, token in enumerate(TOKENS):
            restore_cold(model, snapshot)
            before = counter[0]
            value = outputs(model, [token], eager=True)
            base.require(
                counter[0] - before == (3 if arm == "candidate" else 0), "Wrong eager hint dispatch"
            )
            eager = observed_record(model, observer, f"{token}_eager", value)
            restore_cold(model, snapshot)
            if index == 0:
                before = counter[0]
                model.prepare_extend_graph([token], return_hidden=True)
                base.require(
                    counter[0] - before == (12 if arm == "candidate" else 0),
                    "Wrong captured hint dispatch",
                )
            before = counter[0]
            value = outputs(model, [token])
            base.require(counter[0] == before, "Replay unexpectedly invoked a host hint function")
            graph = observed_record(model, observer, f"{token}_graph", value, captured=True)
            require_q1_proof(eager)
            require_q1_proof(graph)
            comparison = compare_records(eager, graph)
            saved[str(token)] = {"eager": eager, "graph": graph, "comparison": comparison}
    base.require(
        not getattr(model, "_extend_graphs", {}), "Diagnostic graph survived observer cleanup"
    )
    base.require(
        getattr(model, "_prefetch_validation_owner", None) is None,
        "Diagnostic observer survived cleanup",
    )
    return saved


def check_clean_graph(model, snapshot, diagnostic):
    saved = {}
    for token in TOKENS:
        restore_cold(model, snapshot)
        value = outputs(model, [token])
        reference = diagnostic[str(token)]["eager"]
        exact_outputs(value, reference["output"])
        actual_offsets = offsets(model)
        exact_offsets(actual_offsets, reference["offsets"])
        state = cache_state(model)
        for actual, expected in zip(state["layers"], reference["cache"]["layers"], strict=True):
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
                base.require(actual[key] == expected[key], "Clean graph changed " + key)
            base.require(
                actual["map_invariants_passed"] is True, "Clean graph cache invariants failed"
            )
        saved[str(token)] = {"output": value, "offsets": actual_offsets, "cache": state}
    return saved


def new_model(args, *, consumer=False):
    return DeepSeekEchoModel(
        args.model,
        devices=[0],
        num_layers=3,
        capacity=65539 if consumer else 65537,
        offload=True,
        slots=65600,
        chunk_size=1024,
        extend_chunk_size=2 if consumer else 1,
        host_arena_tokens=65600,
        workspace_query_tokens=1024,
        hbm_cache_budget_bytes=24 * 2**30,
        dram_cache_budget_bytes=64 * 2**30,
    )


def close_models(models):
    primary, errors = sys.exception(), []
    for model in models.values():
        try:
            with torch.inference_mode():
                model.close()
        except BaseException as error:  # noqa: BLE001 -- preserve primary and every cleanup error
            errors.append(error)
    if errors:
        raise BaseExceptionGroup(
            "Model execution and cleanup failed", ([primary] if primary else []) + errors
        )


def check_consumer(args, ids):
    saved = {}
    for arm in ARMS:
        model = new_model(args, consumer=True)
        try:
            model.prepare_compute_graphs([1024, 1, 2])
            prefix = model.forward(ids[:HISTORY]).detach().cpu().clone()
            model.evict_prefix_residency()
            folder = args.output_dir / ("consumer_" + arm)
            folder.mkdir()
            with binding(arm), ColdPrefetchObserver(model, HISTORY, 1, folder) as observer:
                q1 = observed_record(model, observer, "q1", outputs(model, [TOKENS[0]], eager=True))
                require_q1_proof(q1)
            # Keep the actual committed Q1 history and all Q1-produced hint bits.
            # Only main-KV residency changes; no restore/truncate occurs here.
            produced = offsets(model)
            base.require(model.length == HISTORY + 1, "Q1 did not commit its token")
            model.evict_prefix_residency()
            base.require(model.length == HISTORY + 1, "Residency eviction changed history")
            exact_offsets(offsets(model), produced)
            with ColdPrefetchObserver(model, HISTORY + 1, 2, folder) as observer:
                a2 = observed_record(
                    model, observer, "a2", outputs(model, list(TOKENS[1:]), eager=True)
                )
            base.require(model.length == HISTORY + 3, "A2 did not commit both tokens")
            for index, (hint, row) in enumerate(zip(produced, a2["observed"], strict=True)):
                exact(row["initial_hint"], hint, f"Q1-to-A2 initial offsets layer {index}")
                base.require(
                    a2["proof"]["initial_hints"][index] == tensor_identity(hint[:1]),
                    "A2 consumed a different prefill mean",
                )
                exact(
                    a2["offsets"][index][1:],
                    hint[1:],
                    "A2 must preserve decode EMA and unused slots",
                )
            saved[arm] = {"prefix": prefix, "q1": q1, "a2": a2, "produced_offsets": produced}
        finally:
            close_models({arm: model})
    exact(saved["baseline"]["prefix"], saved["candidate"]["prefix"], "Consumer prefix")
    for step in ("q1", "a2"):
        compare_records(saved["baseline"][step], saved["candidate"][step])
    return saved


def samples(models, snapshots, pairs, *, profile=False):
    rows = []
    with profiler_range(profile):
        for pair in range(1 if profile else pairs):
            order = ARMS if pair % 2 == 0 else ARMS[::-1]
            for arm in order:
                model = models[arm]
                restore_cold(model, snapshots[arm])
                torch.cuda.synchronize()
                with binding(arm), nvtx_range(profile, "q1_hint_model/" + arm):
                    begin = time.perf_counter_ns()
                    model.forward([TOKENS[0]], return_hidden=True)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter_ns() - begin) / 1e6
                rows.append(
                    {
                        "pair": pair,
                        "order": "AB" if pair % 2 == 0 else "BA",
                        "arm": arm,
                        "wall_ms": elapsed,
                        "layer_metrics": [block.cache.metrics() for block in model.blocks],
                    }
                )
    return rows


def summarize(rows, pairs):
    base.require(len(rows) == 2 * pairs, "Incomplete paired model samples")
    by_pair = {(row["pair"], row["arm"]): row for row in rows}
    base.require(len(by_pair) == 2 * pairs, "Duplicate paired model samples")
    delta = [
        by_pair[(pair, "candidate")]["wall_ms"] - by_pair[(pair, "baseline")]["wall_ms"]
        for pair in range(pairs)
    ]
    return {
        "median_wall_ms": {
            arm: statistics.median(row["wall_ms"] for row in rows if row["arm"] == arm)
            for arm in ARMS
        },
        "paired_delta_ms": delta,
        "median_paired_delta_ms": statistics.median(delta),
        "candidate_wins": sum(value < 0 for value in delta),
        "order_median_delta_ms": {
            order: statistics.median(delta[i] for i in range(pairs) if i % 2 == parity)
            for parity, order in enumerate(("AB", "BA"))
        },
    }


def archive(source, observed, destination):
    for relative, digest in source["sources"].items():
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base.ROOT / relative, target)
        base.require(base.digest(target) == digest, "Source archive differs")
    for entry in observed["mapped_native"]:
        target = destination / "native" / entry["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["library"]["path"], target)
        base.require(base.digest(target) == entry["library"]["sha256"], "Native archive differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "bench", "profile"))
    parser.add_argument("--model", type=Path, default=Path("/preset-models"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component-receipt", type=Path, default=DEFAULT_COMPONENT)
    parser.add_argument("--component-bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--pairs", type=int, default=100)
    args = parser.parse_args()
    base.require(args.pairs >= 100 and args.pairs % 2 == 0, "At least 100 balanced pairs required")
    args.output_dir = args.output_dir.resolve()
    base.require(
        args.mode != "check" or not args.output_dir.is_relative_to(base.EXPERIMENT / "output"),
        "Checks belong outside experiment output",
    )
    for name, directory in (
        ("TRITON_CACHE_DIR", "/tmp/cxldsagr-q1-hint-cache/triton"),
        ("CUDA_CACHE_PATH", "/tmp/cxldsagr-q1-hint-cache/cuda"),
        (
            "TVM_FFI_CACHE_DIR",
            os.environ.get("TVM_FFI_CACHE_DIR", str(Path.home() / ".cache/tvm-ffi")),
        ),
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
        os.environ[name] = directory
    base.require(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0), "SM90 required"
    )
    # This normal backend import initializes Triton's compiler environment.
    # Freeze identities only after that setup, before any model execution.
    importlib.import_module("flashinfer.triton")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    ids = json.loads(args.request.read_text())["input_ids"]
    base.require(len(ids) == 65537 and ids[-1] == TOKENS[0], "Expected supplied H64K/A1 request")
    accepted_component = component_binding(args)
    source = source_identity(args, accepted_component)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    models, snapshots, diagnostic, clean, prefix, memory = {}, {}, {}, {}, {}, {}
    result = {"passed": True}
    with torch.inference_mode():
        candidate.module()
        try:
            for arm in ARMS:
                print("Loading " + arm, flush=True)
                model = new_model(args)
                models[arm] = model
                model.prepare_compute_graphs(CONTRACT["compute_graphs"])
                value = model.forward(ids[:HISTORY])
                prefix[arm] = value.detach().cpu().clone()
                snapshots[arm] = model.snapshot_prefix()
                if args.mode == "check":
                    diagnostic[arm] = check_model(model, snapshots[arm], arm, args.output_dir / arm)
                restore_cold(model, snapshots[arm])
                with binding(arm):
                    capture = FullExtendGraphCapture(arm) if args.mode == "profile" else None
                    with (
                        profiler_range(capture is not None),
                        nvtx_range(capture is not None, "q1_hint_model_capture/" + arm),
                    ):
                        graph = model.prepare_extend_graph(
                            [TOKENS[0]], return_hidden=True, capture_scope=capture
                        )
                        if capture is not None:
                            template = capture.finalize(graph)
                            result.setdefault("capture_templates", {})[arm] = template
                            base.write(args.output_dir / (arm + "_capture_template.json"), template)
                    for _ in range(CONTRACT["warmups"]):
                        restore_cold(model, snapshots[arm])
                        model.forward([TOKENS[0]], return_hidden=True)
                    if args.mode == "check":
                        clean[arm] = check_clean_graph(model, snapshots[arm], diagnostic[arm])
                memory[arm] = {
                    "graph": graph.describe(),
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "device_used": torch.cuda.device_memory_used(),
                }
                print("Prepared " + arm, flush=True)
            observed = runtime(args.output_dir)
            require_component_native(observed, accepted_component)
            identity = {"source": source, "timing_runtime": observed}
            if args.mode == "check":
                exact(prefix["baseline"], prefix["candidate"], "Independent prefix logits")
                result["cross_arm"] = {}
                for token in TOKENS:
                    result["cross_arm"][str(token)] = {
                        execution: compare_records(
                            diagnostic["baseline"][str(token)][execution],
                            diagnostic["candidate"][str(token)][execution],
                        )
                        for execution in ("eager", "graph")
                    }
                    exact_outputs(
                        clean["baseline"][str(token)]["output"],
                        clean["candidate"][str(token)]["output"],
                    )
                    exact_offsets(
                        clean["baseline"][str(token)]["offsets"],
                        clean["candidate"][str(token)]["offsets"],
                    )
                result["clean_graph_inputs_per_arm"] = 3
                result["clean_graph_outputs_all_offsets_and_cache_invariants_checked"] = True
            else:
                receipt = require_receipt(args.receipt, kind=KIND, identity=identity)
                result["receipt_sha256"] = receipt["receipt_sha256"]
                rows = samples(models, snapshots, args.pairs, profile=args.mode == "profile")
                result["samples"] = rows
                if args.mode == "bench":
                    result["summary"] = summarize(rows, args.pairs)
            base.require(runtime() == observed, "Timing runtime changed")
            archive(source, observed, args.output_dir)
        finally:
            close_models(models)
        models.clear()
        snapshots.clear()
        model = graph = None
        # Q2 adds check-only specializations. Keep them separate from the exact
        # pre-consumer runtime identity required by timing/profile processes.
        if args.mode == "check":
            consumer = check_consumer(args, ids)
            result["actual_q1_to_a2_consumer_per_arm"] = True
            result["consumer_runtime"] = runtime(args.output_dir / "consumer_runtime")
            torch.save(
                {"prefix": prefix, "diagnostic": diagnostic, "clean": clean, "consumer": consumer},
                args.output_dir / "outputs.pt",
            )
        base.require(
            source_identity(args, component_binding(args)) == source,
            "Sources, checkpoint, request or component evidence changed",
        )
    payload = {
        "completed": True,
        "mode": args.mode,
        "run_id": args.output_dir.name,
        "identity": identity,
        "result": result,
        "memory": memory,
        "pairs": args.pairs,
        "boundary": "Real checkpoint L0-L2, independent prefixes/models, official ECHO and identical 64-slot Q1 preparation. Diagnostic graphs are destroyed and replaced before clean graph checks/timing. Consumer check uses separate capacity65539 models. Profile activity durations are not clean latency samples.",
    }
    base.write(args.output_dir / "result.json", payload)
    if args.mode == "check":
        write_receipt(
            args.output_dir / "receipt.json",
            kind=KIND,
            identity=identity,
            checks={
                "passed": True,
                "tokens": list(TOKENS),
                "eager_graph_both_arms": True,
                "all_offsets_bitwise": True,
                "actual_stage_proofs": True,
                "prepared_cap_both_arms": 64,
                "clean_graph_checked": True,
                "actual_q1_to_a2_consumer": True,
            },
            artifacts={
                str(path.relative_to(args.output_dir)): path
                for path in args.output_dir.rglob("*")
                if path.is_file()
            },
        )
    print(
        json.dumps({"run_id": args.output_dir.name, "mode": args.mode, "completed": True}),
        flush=True,
    )


if __name__ == "__main__":
    main()
