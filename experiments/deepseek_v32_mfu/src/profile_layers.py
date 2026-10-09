"""Measure four cache methods on the real first three DeepSeek checkpoint layers."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import statistics
import time
from contextlib import ExitStack, nullcontext
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_mfu.src.measure import make_request, source_manifest
from experiments.deepseek_v32_mfu.src.operator_instrumentation import (
    InstrumentOperators,
    OperatorScopes,
)
from experiments.deepseek_v32_mfu.src.profile_hardware import gather_hardware
from experiments.deepseek_v32_mfu.src.run_contract import (
    METHOD_ISOLATION,
    METHODS,
    bind_benchmark,
    checkpoint_identity,
    execution_identity,
    hbm_reference_binding,
    preparation_contract,
    receipt_binding,
    receipt_kind,
    require_hbm_reference,
    validate_runtime_participation,
    validated_receipt,
)
from models.deepseek_v32.execution.extend_graph import EXTEND_GRAPH_POLICY_REVISION
from models.deepseek_v32.model import DeepSeekEchoModel
from models.deepseek_v32.nonmatrix import rms_norm
from operators.deepseek_v32.indexer.echo import build_info

ROOT = Path(__file__).resolve().parents[3]
DENSE_TRANSPORT = "cuda_memcpy_async_contiguous"


def select_cache_method(model, method):
    """Bind the declared dense transport to the actual allocated pool layout."""
    model.set_cache_method(method)
    pools = model._shared_pools
    dense = method == "dense_prefetch"
    if bool(pools) != (method != "hbm") or any(
        pool.dense_contiguous is not dense for pool in pools.values()
    ):
        raise RuntimeError("allocated cache layout differs from the declared method transport")
    sequence = getattr(model, "_experiment_cache_method_selections", None)
    if sequence is not None:
        sequence.append(method)


def selected_methods(args):
    method = getattr(args, "method", None)
    return METHODS if method is None else (method,)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def differing_identity_paths(before, after, path="backend_provenance"):
    """Describe exact identity differences without printing environment values."""
    if type(before) is not type(after):
        return [path]
    if isinstance(before, dict):
        return [
            item
            for key in sorted(before.keys() | after.keys())
            for item in (
                differing_identity_paths(before[key], after[key], f"{path}.{key}")
                if key in before and key in after
                else [f"{path}.{key}"]
            )
        ]
    if isinstance(before, list):
        if len(before) != len(after):
            return [path]
        return [
            item
            for index, (left, right) in enumerate(zip(before, after, strict=True))
            for item in differing_identity_paths(left, right, f"{path}[{index}]")
        ]
    return [] if before == after else [path]


def sources():
    result = source_manifest()
    for name in (
        "profile_layers.py",
        "extend_graph_validation.py",
        "prefetch_validation.py",
        "prefetch_transition_audit.py",
        "full_graph_profile.py",
        "operator_instrumentation.py",
        "operator_flops.py",
        "profile_hardware.py",
        "run_contract.py",
        "gap_profile.py",
    ):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("graph_instrumentation.py", "graph_attribution.py"):
        path = ROOT / "experiments/deepseek_v32_motivation/src" / name
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in (
        "scripts/profile_layers.sh",
        "scripts/run.sh",
        "scripts/gap_profile.sh",
        "scripts/runner_common.sh",
        "scripts/method_isolation.sh",
    ):
        path = Path(__file__).parents[1] / name
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def tensor_storage_bytes(value):
    """Count diagnostic snapshot storage by identity, including nested pool state."""
    seen = {}

    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            seen[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return sum(seen.values())


def comparison(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError("Nonfinite checkpoint output")
    error = actual.float() - expected.float()
    return {
        "shape": list(actual.shape),
        "bitwise_equal": torch.equal(actual, expected),
        "max_abs": error.abs().max().item(),
        "relative_l2": (error.norm() / expected.float().norm().clamp_min(1e-20)).item(),
        "rtol": 0.01,
        "atol": 0.02,
    }


def timed(model, ids):
    model.synchronize()
    begin = time.perf_counter()
    output = model.forward(ids)
    return output, (time.perf_counter() - begin) * 1000


@torch.inference_mode()
def annotate(model, ids, mode, phase, nsys):
    scopes = OperatorScopes(mode, phase)
    block_outputs = []
    original = model.blocks[-1].forward

    def remember(*args, **kwargs):
        output = original(*args, **kwargs)
        if phase == "extend_annotated":
            # Retain existing tensors, with no additional CUDA work inside capture.
            block_outputs.append(output)
        return output

    model.synchronize()
    if nsys:
        torch.cuda.cudart().cudaProfilerStart()
    operators = InstrumentOperators(model, scopes)
    full_graph = getattr(model, "_extend_graph", None) if phase == "extend_annotated" else None
    bank = getattr(model, "_compute_graphs", None) if full_graph is None else None
    if bank is not None:
        from experiments.deepseek_v32_motivation.src.graph_instrumentation import (
            instrument_graph_queries,
        )

        query_context = instrument_graph_queries(bank, operators)
    else:
        query_context = nullcontext()
    with operators, query_context, ExitStack() as hooks:
        if bank is None and full_graph is None:
            hooks.enter_context(patch.object(model.blocks[-1], "forward", remember))
        elif bank is not None:
            original_graph = bank.forward_block

            def remember_graph(layer, *args, **kwargs):
                value = original_graph(layer, *args, **kwargs)
                if layer == model.num_layers - 1 and phase == "extend_annotated":
                    block_outputs.append(value)
                return value

            hooks.enter_context(patch.object(bank, "forward_block", remember_graph))
        begin = time.perf_counter()
        with scopes("forward_misc"):
            output = model.forward(ids, scope=scopes)
        wall = (time.perf_counter() - begin) * 1000
    if nsys:
        torch.cuda.cudart().cudaProfilerStop()
    if full_graph is not None:
        block_outputs = [(full_graph.last_hidden, full_graph.last_residual)]
    normalized = None
    if block_outputs:
        normalized = torch.cat(
            [
                rms_norm(
                    hidden.float() + residual.float(), model.final_norm, model.cfg.norm_eps
                ).bfloat16()
                for hidden, residual in block_outputs
            ]
        ).cpu()
    for capture in (
        getattr(model, "_profile_graph_capture", None),
        getattr(model, "_profile_full_graph_capture", None),
    ):
        if capture is None:
            continue
        expanded = []
        next_id = len(scopes.calls)
        for call in scopes.calls:
            children = capture.expand_replay(call, next_id)
            expanded.extend(children)
            next_id += len(children)
        scopes.calls.extend(expanded)
    return output.cpu(), normalized, scopes.calls, wall


def cache_metrics(model, snapshot, prefix_metrics):
    return {
        "snapshot_cpu_tensor_bytes": tensor_storage_bytes(snapshot),
        "compute_graph_runtime": model._compute_graphs.describe()
        if getattr(model, "_compute_graphs", None) is not None
        else None,
        "cache_resource_plan": getattr(model, "_cache_resource_plan", None),
        "prefix_cache_per_layer": prefix_metrics,
        "extend_cache_per_layer": [block.cache.metrics() for block in model.blocks],
        "extend_graph_runtime": getattr(model, "_extend_graph", None).describe()
        if getattr(model, "_extend_graph", None) is not None
        else None,
    }


def restore_extend_prefix(model, snapshot, args):
    model.restore_prefix(snapshot)
    if args.extend_residency == "cold" and model.offload:
        model.evict_prefix_residency()


def prepare_extend_graph(model, ids, args, *, return_hidden=False, capture_scope=None):
    """Prepare after restoration and before any measured forward scope."""
    if not getattr(args, "extend_graph", False):
        return None
    return model.prepare_extend_graph(ids, return_hidden=return_hidden, capture_scope=capture_scope)


def profile_extend_graph(model, ids, args, result, method, *, record_operators=False):
    """Record graph construction separately from the measured extend replay."""
    if not getattr(args, "extend_graph", False):
        return None
    from experiments.deepseek_v32_mfu.src.full_graph_profile import FullExtendGraphCapture
    from experiments.deepseek_v32_mfu.src.gap_profile import profiler_capture

    with profiler_capture():
        capture = FullExtendGraphCapture(method, record_operators=record_operators)
        with InstrumentOperators(model, capture) if record_operators else nullcontext():
            graph = prepare_extend_graph(model, ids, args, capture_scope=capture)
        template = capture.finalize(graph)
    model._profile_full_graph_capture = capture
    result["nsys_capture_order"].append(f"{method}/extend_graph_setup")
    result.setdefault("full_extend_graph_templates", []).append(template)
    write_json(args.output / "full_graph_templates.json", result["full_extend_graph_templates"])
    return capture


def run_benchmark(model, ids, args, result):
    """Only default model outputs and wall timers; no output copies or comparisons."""
    for mode in selected_methods(args):
        prefix_times = []
        prefix_cache_samples = []
        for _ in range(args.prefill_repeats):
            select_cache_method(model, mode)
            _, elapsed = timed(model, ids[: args.prefix])
            prefix_times.append(elapsed)
            prefix_cache_samples.append([block.cache.metrics() for block in model.blocks])
        prefix_metrics = [block.cache.metrics() for block in model.blocks]
        snapshot = model.snapshot_prefix()
        restore_extend_prefix(model, snapshot, args)
        prepare_extend_graph(model, ids[args.prefix :], args)
        extend_times = []
        extend_cache_samples = []
        for _ in range(args.repeats):
            restore_extend_prefix(model, snapshot, args)
            _, elapsed = timed(model, ids[args.prefix :])
            extend_times.append(elapsed)
            extend_cache_samples.append([block.cache.metrics() for block in model.blocks])
        result["measurements"][mode] = {
            **cache_metrics(model, snapshot, prefix_metrics),
            "prefill_samples_ms": prefix_times,
            "extend_samples_ms": extend_times,
            "prefill_cache_samples": prefix_cache_samples,
            "extend_cache_samples": extend_cache_samples,
            "prefill_median_ms": statistics.median(prefix_times),
            "extend_median_ms": statistics.median(extend_times),
        }
        del snapshot


def run_check(model, ids, args, result):
    controls, prefixes = {}, {}
    methods = selected_methods(args)
    if getattr(args, "hbm_check_receipt", None) is not None:
        reference = require_hbm_reference(args.hbm_check_receipt, result)
        result["hbm_reference"] = hbm_reference_binding(reference, result)
        controls["hbm"] = torch.load(
            reference["artifact_paths"]["hbm_control.pt"], map_location="cpu", weights_only=True
        )
        prefixes["hbm"] = torch.load(
            reference["artifact_paths"]["hbm_prefix_logits.pt"],
            map_location="cpu",
            weights_only=True,
        )
    for mode in methods:
        select_cache_method(model, mode)
        prefixes[mode] = model.forward(ids[: args.prefix]).cpu()
        snapshot = model.snapshot_prefix()
        if getattr(args, "extend_graph", False):
            _check_extend_graph(model, ids[args.prefix :], snapshot, args, result, mode)
        restore_extend_prefix(model, snapshot, args)
        prepare_extend_graph(model, ids[args.prefix :], args)
        default = model.forward(ids[args.prefix :]).cpu()
        restore_extend_prefix(model, snapshot, args)
        prepare_extend_graph(model, ids[args.prefix :], args, return_hidden=True)
        control = {
            key: value.cpu()
            for key, value in model.forward(ids[args.prefix :], return_hidden=True).items()
        }
        if control["hidden"].ndim != 2 or control["hidden"].shape[0] != args.extend:
            raise ValueError("Check must cover every extend hidden row")
        if control["logits"].ndim != 2 or control["logits"].shape[0] != 1:
            raise ValueError("Check must cover the last-token logits")
        result["correctness"][mode + "_default_extend_logits"] = comparison(
            control["logits"], default
        )
        controls[mode] = control
        torch.save(control, args.output / f"{mode}_control.pt")
        torch.save(prefixes[mode], args.output / f"{mode}_prefix_logits.pt")
        del snapshot
    for mode in methods:
        if mode == "hbm":
            continue
        for key in ("hidden", "logits"):
            result["correctness"][f"hbm_vs_{mode}_{key}"] = comparison(
                controls[mode][key], controls["hbm"][key]
            )
        result["correctness"][f"hbm_vs_{mode}_prefill_logits"] = comparison(
            prefixes[mode], prefixes["hbm"]
        )


def _check_extend_graph(model, ids, snapshot, args, result, mode):
    if mode == "echo" and args.extend_residency == "cold":
        from experiments.deepseek_v32_mfu.src.prefetch_validation import ColdPrefetchObserver

        restore_extend_prefix(model, snapshot, args)
        with ColdPrefetchObserver(model, args.prefix, args.extend, args.output) as observer:
            _check_extend_graph_impl(model, ids, snapshot, args, result, mode, observer=observer)
    else:
        _check_extend_graph_impl(model, ids, snapshot, args, result, mode)


def _check_extend_graph_impl(model, ids, snapshot, args, result, mode, *, observer=None):
    from experiments.deepseek_v32_mfu.src.extend_graph_validation import (
        cache_state,
        compare_cache_state,
    )

    checks, prefetch_proofs = {}, {}

    def observed_state(label, *, captured=False):
        state = cache_state(model)
        proof = None
        if observer is not None:
            proof = observer.audit(label, state, captured=captured)
            prefetch_proofs[label] = proof
        return state, proof

    def compare_observed(label, baseline_state, baseline_proof, *, captured=True):
        state, proof = observed_state(label, captured=captured)
        return compare_cache_state(
            state,
            baseline_state,
            actual_prefetch=proof,
            expected_prefetch=baseline_proof,
        )

    restore_extend_prefix(model, snapshot, args)
    expected = {
        key: value.cpu()
        for key, value in model.forward(ids, return_hidden=True, use_extend_graph=False).items()
    }
    expected_cache, expected_proof = observed_state("baseline")
    restore_extend_prefix(model, snapshot, args)
    prepare_extend_graph(model, ids, args)
    default = model.forward(ids).cpu()
    checks["default_graph_logits"] = comparison(default, expected["logits"])
    checks["default_graph_cache"] = compare_observed(
        "default_graph", expected_cache, expected_proof
    )
    restore_extend_prefix(model, snapshot, args)
    graph = prepare_extend_graph(model, ids, args, return_hidden=True)
    for repetition in range(2):
        restore_extend_prefix(model, snapshot, args)
        actual = {key: value.cpu() for key, value in model.forward(ids, return_hidden=True).items()}
        for key in ("hidden", "logits"):
            checks[f"replay_{repetition}_{key}"] = comparison(actual[key], expected[key])
        checks[f"replay_{repetition}_cache"] = compare_observed(
            f"replay_{repetition}", expected_cache, expected_proof
        )
    changed = list(ids)
    changed[-1] = (int(changed[-1]) + 1) % model.cfg.vocab_size
    restore_extend_prefix(model, snapshot, args)
    changed_expected = {
        key: value.cpu()
        for key, value in model.forward(changed, return_hidden=True, use_extend_graph=False).items()
    }
    changed_cache, changed_proof = observed_state("changed_baseline")
    restore_extend_prefix(model, snapshot, args)
    changed_actual = {
        key: value.cpu() for key, value in model.forward(changed, return_hidden=True).items()
    }
    for key in ("hidden", "logits"):
        checks[f"changed_input_{key}"] = comparison(changed_actual[key], changed_expected[key])
    checks["changed_input_cache"] = compare_observed("changed_graph", changed_cache, changed_proof)
    torch.save(
        {
            "baseline": expected,
            "changed_baseline": changed_expected,
            "changed_graph": changed_actual,
        },
        args.output / f"{mode}_extend_graph_check.pt",
    )
    result.setdefault("extend_graph_checks", {})[mode] = {
        "checks": checks,
        "baseline_cache": expected_cache,
        "changed_baseline_cache": changed_cache,
        "runtime": graph.describe(),
        "baseline": "same model.forward(use_extend_graph=False), same restored prefix and residency",
        **({"bounded_prefetch": prefetch_proofs} if observer is not None else {}),
    }


def run_profile(model, ids, args, result, receipt):
    calls = []
    result["nsys_capture_order"] = ["graph_setup"] if args.compute_graphs else []
    result["profile_detail"] = "matrix_api_node_ownership"
    result["correctness"].update(receipt["checks"]["comparisons"])
    for mode in selected_methods(args):
        control = torch.load(
            receipt["artifact_paths"][mode + "_control.pt"], map_location="cpu", weights_only=True
        )
        prefix_control = torch.load(
            receipt["artifact_paths"][mode + "_prefix_logits.pt"],
            map_location="cpu",
            weights_only=True,
        )
        select_cache_method(model, mode)
        annotated_prefix, _, records, prefix_wall = annotate(
            model, ids[: args.prefix], mode, "prefill_annotated", args.nsys
        )
        result["nsys_capture_order"].append(f"{mode}/prefill_annotated")
        calls.extend(records)
        result["correctness"][mode + "_profile_prefix_logits"] = comparison(
            annotated_prefix, prefix_control
        )
        prefix_metrics = [block.cache.metrics() for block in model.blocks]
        snapshot = model.snapshot_prefix()
        restore_extend_prefix(model, snapshot, args)
        profile_extend_graph(model, ids[args.prefix :], args, result, mode, record_operators=True)
        annotated, hidden, records, extend_wall = annotate(
            model, ids[args.prefix :], mode, "extend_annotated", args.nsys
        )
        result["nsys_capture_order"].append(f"{mode}/extend_annotated")
        calls.extend(records)
        for key, actual in (("logits", annotated), ("hidden", hidden)):
            result["correctness"][mode + "_profile_extend_" + key] = comparison(
                actual, control[key]
            )
        torch.save(
            {"hidden": hidden, "logits": annotated}, args.output / f"{mode}_profile_output.pt"
        )
        result["measurements"][mode] = {
            **cache_metrics(model, snapshot, prefix_metrics),
            "annotated_prefix_wall_ms": prefix_wall,
            "annotated_extend_wall_ms": extend_wall,
        }
        write_json(
            args.output / "operator_calls.json",
            {"schema_version": result["schema_version"], "run_id": args.run_id, "calls": calls},
        )
        if mode == "echo":
            capture_kernel_inputs(model, ids[args.prefix :], snapshot, args)
        del snapshot


def capture_kernel_inputs(model, ids, snapshot, args):
    restore_extend_prefix(model, snapshot, args)
    for layer, block in enumerate(model.blocks):

        def capture(p, keys, scales, indices, cache, position, layer=layer):
            torch.save(
                {
                    "layer": layer,
                    "query_start": position,
                    "attention_scale": model.cfg.attention_scale,
                    "q": p.q.cpu(),
                    "index_q": p.index_q.cpu(),
                    "index_weights": p.index_weights.cpu(),
                    "index_keys": keys.cpu(),
                    "index_scales": scales.cpu(),
                    "indices": indices.cpu(),
                    "kv": cache.host_records(),
                    "source_run_id": args.run_id,
                },
                args.output / f"kernel_inputs_layer_{layer}.pt",
            )

        block.attention.capture_hook = capture
    try:
        if getattr(args, "extend_graph", False):
            # Saved operator inputs need the real Python hooks to run. This
            # explicit diagnostic is outside every measured profiler capture.
            model.forward(ids, use_extend_graph=False)
        else:
            model.forward(ids)
    finally:
        for block in model.blocks:
            block.attention.capture_hook = None
    if not all(
        (args.output / f"kernel_inputs_layer_{layer}.pt").is_file()
        for layer in range(model.num_layers)
    ):
        raise RuntimeError("Profile did not capture every checkpoint layer's kernel inputs")


def prepare_graphs(model, args, mode):
    if not args.compute_graphs:
        return None
    query_sizes = {min(args.prefix, args.chunk_size), args.extend_chunk_size or args.extend}
    if args.prefix % args.chunk_size:
        query_sizes.add(args.prefix % args.chunk_size)
    extend_chunk = args.extend_chunk_size or args.extend
    if args.extend % extend_chunk:
        query_sizes.add(args.extend % extend_chunk)
    if mode != "profile" or not args.nsys:
        model.prepare_compute_graphs(sorted(query_sizes))
        if mode == "profile":
            raise ValueError("graph operator profiling requires --nsys for exact node attribution")
        return None
    from experiments.deepseek_v32_motivation.src.graph_instrumentation import CaptureGraphOperators

    owner = SimpleNamespace(
        attentions=[block.attention.attention for block in model.blocks],
        blocks=model.blocks,
        head_weight=model.head_weight,
        scheme="four_method_checkpoint",
    )
    torch.cuda.cudart().cudaProfilerStart()
    try:
        with CaptureGraphOperators(owner) as capture:
            model.prepare_compute_graphs(sorted(query_sizes))
        owner._compute_graphs = model._compute_graphs
        capture.finalize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    return capture


def parse_run_args(mode=None, argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__
        if mode == "profile"
        else "Independent numerical check or clean three-layer checkpoint benchmark.",
        allow_abbrev=False,
    )
    parser.add_argument("--model", type=Path, default=Path("/preset-models"))
    parser.add_argument(
        "--method",
        choices=METHODS,
        help="isolate one method in this process; omitted keeps the four-method run",
    )
    parser.add_argument(
        "--hbm-check-receipt",
        type=Path,
        help="check-only isolated HBM reference for an isolated offload check",
    )
    parser.add_argument("--request", type=Path)
    parser.add_argument("--prefix", type=int, default=65536)
    parser.add_argument("--extend", type=int, default=128)
    parser.add_argument(
        "--sparse-pool-tokens",
        "--slots",
        dest="slots",
        type=int,
        default=None,
        help="usable tokens per shared model layer pool; excludes sentinel row",
    )
    parser.add_argument("--host-arena-tokens", type=int)
    parser.add_argument("--workspace-query-tokens", type=int)
    parser.add_argument("--hbm-cache-budget-gib", type=float, default=24)
    parser.add_argument("--dram-cache-budget-gib", type=float, default=64)
    parser.add_argument(
        "--extend-chunk-size",
        type=int,
        help="explicit extend chunk; default is the complete extend batch",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--extend-residency", choices=("cold", "warm"), default="cold")
    parser.add_argument("--compute-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extend-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prefill-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--physical-device", default="1", help="nvidia-smi index or UUID; CUDA device is 0"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    if mode == "profile":
        parser.add_argument("--nsys", action="store_true")
        parser.add_argument("--benchmark-run", type=Path)
    else:
        parser.add_argument("--mode", choices=("check", "bench"), default="bench")
    parser.add_argument("--validation-receipt", type=Path)
    args = parser.parse_args(argv)
    mode = mode or args.mode
    if args.hbm_check_receipt is not None and not (mode == "check" and args.method in METHODS[1:]):
        parser.error("--hbm-check-receipt is only for isolated offload checks")
    if mode == "check" and args.method in METHODS[1:] and args.hbm_check_receipt is None:
        parser.error("isolated offload check requires --hbm-check-receipt")
    if args.extend_chunk_size is None:
        args.extend_chunk_size = args.extend
    if args.extend_graph and args.extend_chunk_size != args.extend:
        parser.error("full extend graph requires one complete extend chunk")
    if (
        mode == "profile"
        and args.compute_graphs
        and args.extend_chunk_size is not None
        and args.extend_chunk_size < args.extend
    ):
        parser.error("graph profile output verification requires one complete extend chunk")
    if mode != "check" and args.validation_receipt is None:
        parser.error("--validation-receipt is required for bench/profile")
    if mode == "check" and args.validation_receipt is not None:
        parser.error("check writes its own receipt; do not supply --validation-receipt")
    if (ROOT / "experiments").resolve() in args.output.resolve().parents:
        parser.error("Execute outside experiments; scripts publish accepted bench/profile staging")
    if args.hbm_cache_budget_gib <= 0 or args.dram_cache_budget_gib <= 0:
        parser.error("cache budgets must be positive")
    if args.slots is None:
        args.slots = (args.prefix + args.extend + 63) // 64 * 64
    if args.slots < args.prefix + args.extend:
        parser.error("four-method dense prefetch requires slots >= prefix + extend")
    required_queries = max(args.chunk_size, args.extend_chunk_size or args.extend)
    if args.workspace_query_tokens is None:
        args.workspace_query_tokens = required_queries
    if args.workspace_query_tokens < required_queries:
        parser.error("workspace-query-tokens must cover prefill and extend query batches")
    if args.extend_chunk_size is not None and args.extend_chunk_size < 1:
        parser.error("extend-chunk-size must be positive")
    if args.host_arena_tokens is not None and (
        args.host_arena_tokens % 64
        or args.host_arena_tokens < (args.prefix + args.extend + 63) // 64 * 64
    ):
        parser.error("host-arena-tokens must contain whole pages covering prefix plus extend")
    if (
        min(
            args.prefix,
            args.extend,
            args.slots,
            args.chunk_size,
            args.warmups,
            args.repeats,
            args.prefill_repeats,
        )
        < 1
    ):
        parser.error("All dimensions and repetition counts must be positive")
    return mode, args


def run(mode=None, argv=None, *, profile_runner=None):
    mode, args = parse_run_args(mode, argv)
    methods = selected_methods(args)
    # Linux process start ticks distinguish a reused PID without entering a CUDA runtime.
    process_provenance = {
        "pid": os.getpid(),
        "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "driver_started_utc": datetime.now(UTC).isoformat(),
        "completed_method_warmups": [],
        "cache_method_selections": [],
    }
    if mode == "profile" and args.extend_graph and not args.nsys:
        raise ValueError("full extend graph operator profiling requires --nsys node tracing")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    request = (
        json.loads(args.request.read_text())
        if args.request
        else make_request(args.model, args.prefix, args.extend, args.seed)
    )
    if (
        request["stable_prefix_tokens"] != args.prefix
        or request["candidate_suffix_tokens"] != args.extend
        or len(request["input_ids"]) != args.prefix + args.extend
    ):
        raise ValueError("Saved request does not match the exact prefix/extend boundary")
    write_json(args.output / "request.json", request)
    manifest = sources()
    write_json(args.output / "sources.json", manifest)
    for relative in manifest:
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    hardware = gather_hardware(args.physical_device)
    hardware["torch_properties"] = str(torch.cuda.get_device_properties(0))
    hardware["torch_device_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
    if hardware["torch_device_uuid"].removeprefix("GPU-") != hardware["gpu"]["uuid"].removeprefix(
        "GPU-"
    ):
        raise ValueError(
            "CUDA device 0 differs from --physical-device; set CUDA_VISIBLE_DEVICES explicitly"
        )
    if not hardware["is_sm90"] or hardware["pci_identity"]["device_id"] != "2335":
        raise ValueError("MFU reference peaks require the verified H200 SXM device")
    write_json(args.output / "hardware.json", hardware)
    print(f"Loading real checkpoint layers 0–2; input {args.prefix}+{args.extend}", flush=True)
    model = DeepSeekEchoModel(
        args.model,
        devices=[0],
        num_layers=3,
        capacity=args.prefix + args.extend,
        slots=args.slots,
        chunk_size=args.chunk_size,
        extend_chunk_size=args.extend_chunk_size,
        host_arena_tokens=args.host_arena_tokens,
        workspace_query_tokens=args.workspace_query_tokens,
        hbm_cache_budget_bytes=int(args.hbm_cache_budget_gib * 2**30),
        dram_cache_budget_bytes=int(args.dram_cache_budget_gib * 2**30),
    )
    if args.method is not None:
        model._experiment_cache_method_selections = process_provenance["cache_method_selections"]
    if any(block.is_moe for block in model.blocks):
        raise ValueError("This diagnostic requires the first three dense checkpoint blocks")
    from experiments.deepseek_v32_mfu.src.backend_provenance import (
        collect_backend_provenance,
        collect_flashinfer_runtime_artifacts,
    )

    backend_identity = collect_backend_provenance()
    write_json(args.output / "backend_provenance_before.json", backend_identity)
    ids = request["input_ids"]
    result = {
        "schema_version": 3 if args.method is None else 4,
        "mode": mode,
        "run_id": args.run_id,
        "accepted": False,
        "scope": "checkpoint_layers_0_1_2_embedding_final_norm_last_token_lm_head",
        "model": str(args.model.resolve()),
        "num_layers": model.num_layers,
        "checkpoint_num_layers": model.cfg.num_hidden_layers,
        "seed": args.seed,
        "prefix_tokens": args.prefix,
        "extend_tokens": args.extend,
        "chunk_size": args.chunk_size,
        "slots": args.slots,
        "cache_policy_revision": "four-method-checkpoint-persistent-v3",
        "dense_history_transport": {
            method: DENSE_TRANSPORT if method == "dense_prefetch" else None for method in methods
        },
        "methods": list(methods),
        "extend_residency": args.extend_residency,
        "compute_graphs": args.compute_graphs,
        "extend_graph": args.extend_graph,
        "extend_graph_policy_revision": EXTEND_GRAPH_POLICY_REVISION if args.extend_graph else None,
        "pool_scope": "model_device_per_layer",
        "sparse_pool_tokens": args.slots,
        "host_arena_tokens": args.host_arena_tokens or (args.prefix + args.extend + 63) // 64 * 64,
        "workspace_query_tokens": args.workspace_query_tokens,
        "hbm_cache_budget_bytes": int(args.hbm_cache_budget_gib * 2**30),
        "dram_cache_budget_bytes": int(args.dram_cache_budget_gib * 2**30),
        "extend_chunk_size": args.extend_chunk_size,
        "snapshot_schema": "echo-shared-prefix-v1",
        "snapshot_scope": "shared pools once plus session metadata; diagnostic CPU storage excluded from serving capacity",
        "prefetch_cap": (
            "ECHO preparation: 64 free slots for eligible sole-session persistent official Q1 "
            "with P-H>=64; otherwise min(8192, P-Q). Official Q1 effective prefetch cap: 64. "
            "The independent transition proof records the actual preparation dispatch."
        ),
        "prefetch_flags": {"fused_extend_equivalent": True, "early_evict": False},
        "warmups": args.warmups,
        "repeats": args.repeats,
        "prefill_repeats": args.prefill_repeats,
        "timed_output": "last-token logits; all extend hidden verified by independent check",
        "nsys_capture_order": (
            (["graph_setup"] if args.compute_graphs else [])
            + [
                f"{method}/{phase}_annotated"
                for method in methods
                for phase in ("prefill", "extend")
            ]
        )
        if mode == "profile"
        else [],
        "timing": "synchronized wall; independent check, clean bench, and NVTX profile processes",
        "source_sha256": manifest,
        "backend_provenance": backend_identity,
        "compute_precision": dict(model.blocks[0].attention.attention.precision),
        "torch_precision": {
            "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "indexer_build": build_info(),
        "request_sha256": hashlib.sha256((args.output / "request.json").read_bytes()).hexdigest(),
        "checkpoint_metadata_sha256": {
            name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
            for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")
            if (args.model / name).exists()
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "safetensors", "apache-tvm-ffi")
        },
        "hardware": hardware,
        "measurements": {},
        "correctness": {},
    }
    if args.method is not None:
        result.update(
            {
                "selected_method": args.method,
                "method_isolation": METHOD_ISOLATION,
                "preparation_contract": preparation_contract(
                    args.method, args.warmups, args.compute_graphs
                ),
                "process_provenance": process_provenance,
            }
        )
    result["checkpoint_identity"] = checkpoint_identity(args.model)
    result["execution_environment"] = {
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_num_threads": torch.get_num_threads(),
        "variables": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(
                (
                    "CXLDSAGR_",
                    "DG_",
                    "DJ_",
                    "FLASHINFER_",
                    "CUTE_DSL_",
                    "TRITON_",
                    "PYTORCH_",
                    "OMP_",
                    "MKL_",
                    "OPENBLAS_",
                )
            )
            or key in {"CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING"}
        },
    }
    graph_capture = prepare_graphs(model, args, mode)
    if graph_capture is not None:
        model._profile_graph_capture = graph_capture
        write_json(args.output / "graph_templates.json", graph_capture.finalize())
    # Exercise exactly the default execution path before binding loaded JIT identities.
    # This also completes compilation before any formal sample or profiler capture.
    for method in methods:
        for _ in range(args.warmups):
            select_cache_method(model, method)
            model.forward(ids[: args.prefix])
            snapshot = model.snapshot_prefix()
            restore_extend_prefix(model, snapshot, args)
            prepare_extend_graph(model, ids[args.prefix :], args)
            model.forward(ids[args.prefix :])
            del snapshot
            process_provenance["completed_method_warmups"].append(method)
    result["execution_runtime_artifacts"] = collect_flashinfer_runtime_artifacts(
        require_local_native=args.method != "hbm"
    )
    validate_runtime_participation(result)
    result["execution_identity"] = execution_identity(result)
    receipt = None
    if mode != "check":
        receipt = require_receipt(
            args.validation_receipt,
            kind=receipt_kind(result),
            identity=result["execution_identity"],
        )
        result["validation_receipt"] = receipt_binding(receipt)
        if args.method not in (None, "hbm"):
            result["hbm_reference"] = receipt["checks"].get("hbm_reference")
        receipt = validated_receipt(result)
    if mode == "profile" and args.benchmark_run is not None:
        result["benchmark"] = bind_benchmark(args.benchmark_run, result)
    if mode == "bench":
        run_benchmark(model, ids, args, result)
    elif mode == "check":
        run_check(model, ids, args, result)
    else:
        (run_profile if profile_runner is None else profile_runner)(
            model, ids, args, result, receipt
        )
    if checkpoint_identity(args.model) != result["checkpoint_identity"]:
        raise RuntimeError("Checkpoint identity changed during execution")
    if sources() != manifest:
        raise RuntimeError("Implementation changed during measurement")
    backend_after = collect_backend_provenance()
    write_json(args.output / "backend_provenance_after.json", backend_after)
    if backend_after != backend_identity:
        differences = differing_identity_paths(backend_identity, backend_after)
        write_json(args.output / "backend_provenance_differences.json", differences)
        raise RuntimeError(f"Official backend libraries changed during measurement: {differences}")
    result["flashinfer_runtime_artifacts"] = collect_flashinfer_runtime_artifacts(
        require_local_native=args.method != "hbm"
    )
    validate_runtime_participation(result, result["flashinfer_runtime_artifacts"])
    if (
        result["execution_runtime_artifacts"]["local_native_jit"]
        != result["flashinfer_runtime_artifacts"]["local_native_jit"]
    ):
        raise RuntimeError("Loaded local native libraries changed during measurement")
    result["accepted"] = True
    write_json(args.output / "result.json", result)
    if mode == "check":
        write_receipt(
            args.output / "receipt.json",
            kind=receipt_kind(result),
            identity=result["execution_identity"],
            checks={
                "passed": True,
                "comparisons": result["correctness"],
                **({"hbm_reference": result["hbm_reference"]} if "hbm_reference" in result else {}),
                **({"extend_graph": result["extend_graph_checks"]} if args.extend_graph else {}),
            },
            artifacts={path.name: path for path in args.output.iterdir() if path.is_file()},
        )
    print(
        json.dumps({"run_id": args.run_id, "accepted": True, "correctness": result["correctness"]}),
        flush=True,
    )


def main(argv=None):
    run("profile", argv)


if __name__ == "__main__":
    main()
