"""Private complete-model gate for Q1 preparation fusion; production files stay unchanged."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.local_native import collect_local_native_artifacts
from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_echo_official.src import q1_fused_prepare as candidate
from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_model_native as transfer
from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_native as native
from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_run as component
from experiments.deepseek_v32_echo_official.src import q1_hint_model as common
from experiments.deepseek_v32_mfu.src.full_graph_profile import FullExtendGraphCapture
from experiments.deepseek_v32_mfu.src.prefetch_validation import ColdPrefetchObserver
from experiments.deepseek_v32_mfu.src.profile_layers import sources
from experiments.deepseek_v32_mfu.src.run_contract import checkpoint_identity
from operators.deepseek_v32.indexer import echo, official_decode

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
KIND = "deepseek-private-q1-fused-prepare-model-v1"
ARMS, TOKENS, HISTORY = common.ARMS, common.TOKENS, common.HISTORY
require, digest, write = common.base.require, common.base.digest, common.base.write
DEFAULT_COMPONENT = Path("/tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_02/receipt.json")
DEFAULT_BENCH = EXPERIMENT / "output/data/q1_fused_prepare_bench_20261008_02/summary.json"
CONTRACT = {
    **{
        key: value
        for key, value in common.CONTRACT.items()
        if key not in {"candidate_binding", "consumer_check"}
    },
    "candidate_binding": "Harness-owned stable public trampoline before observer; private arm state routes only existing echo._uses_official_q1 support to candidate.logits. All other calls delegate to the original function unchanged.",
    "dispatch_proof": "CPU delegation/restoration tests plus actual eager/capture/replay route counts; public observer identity remains installed during all diagnostic calls.",
    "candidate": candidate.POLICY,
    "scope": "Private real L0-L2 complete model gate, not full61-layer or serving capacity acceptance",
}


class Binding:
    """Keep one public observer entry while selecting a private dispatch target."""

    def __init__(self):
        self.original = echo.logits
        self.arm = "baseline"
        self.count = 0
        self.installed = False

        def entry(q, k, weights, kscale, query_start, prefetch=None, **kwargs):
            supported = (
                self.arm == "candidate"
                and isinstance(q, torch.Tensor)
                and isinstance(k, torch.Tensor)
                and q.ndim > 0
                and k.ndim > 0
                and echo._uses_official_q1(prefetch, len(q), len(k), query_start)
            )
            if supported:
                self.count += 1
                return candidate.logits(q, k, weights, kscale, query_start, prefetch, **kwargs)
            return self.original(q, k, weights, kscale, query_start, prefetch=prefetch, **kwargs)

        self.entry = entry

    @contextmanager
    def public(self):
        require(not self.installed and echo.logits is self.original, "Unexpected public binding")
        self.installed = True
        context = patch.object(echo, "logits", self.entry)
        context.__enter__()
        try:
            yield self
        finally:
            self.installed = False
            self.arm = "baseline"
            primary = sys.exception()
            common.cleanup(
                (
                    lambda: context.__exit__(
                        type(primary) if primary is not None else None,
                        primary,
                        primary.__traceback__ if primary else None,
                    ),
                    lambda: require(
                        echo.logits is self.original, "Original public indexer was not restored"
                    ),
                )
            )

    @contextmanager
    def bound(self, arm):
        require(self.installed and arm in ARMS, "Invalid private arm binding")
        previous = self.arm
        self.arm = arm
        try:
            yield
        finally:
            self.arm = previous

    def observer_entry(self):
        current = echo.logits
        require(current is not self.entry, "ColdPrefetchObserver did not wrap public entry")
        closure = tuple(cell.cell_contents for cell in (current.__closure__ or ()))
        require(
            any(value is self.entry for value in closure), "Observer lost stable dispatch entry"
        )
        return current


def component_binding(args):
    raw = json.loads(args.component_receipt.read_text())
    receipt = require_receipt(args.component_receipt, kind=component.KIND, identity=raw["identity"])
    expected = receipt["identity"]
    require(
        receipt["checks"].get("complete_cases") == 210
        and receipt["checks"].get("byte_cases") == 100,
        "Incomplete component acceptance",
    )
    for name, sha in {
        **expected["sources"],
        **expected["runtime_files"],
        **expected["inputs"],
    }.items():
        require(digest(name) == sha, "Component dependency changed: " + name)
    bench = json.loads(args.component_bench.read_text())
    require(
        bench.get("mode") == "bench"
        and bench.get("identity") == expected
        and bench.get("receipt_sha256") == receipt["receipt_sha256"],
        "Component timing binding differs",
    )
    summary_rows = bench.get("rows", [])
    groups = {(row["case"], row["policy"]): row for row in summary_rows}
    require(
        len(summary_rows) == len(groups) == len(expected["inputs"]) * len(component.POLICIES) == 15,
        "Incomplete or duplicate component timing cases",
    )
    require(
        set(groups)
        == {
            (Path(name).stem, policy)
            for name in expected["inputs"]
            for policy in component.POLICIES
        },
        "Component case identities differ",
    )
    samples = bench.get("samples", [])
    expected_total = 0
    for (case, policy), summary in groups.items():
        require(policy in component.POLICIES, "Unknown component policy")
        require(summary["warmups"] >= 20, "Insufficient component warmups")
        pairs = summary["pairs"]
        require(pairs >= 100 and pairs % 2 == 0, "Component pairs must be balanced")
        subset = [row for row in samples if row["case"] == case and row["policy"] == policy]
        by_pair = {(row["pair"], row["variant"]): row for row in subset}
        require(len(subset) == len(by_pair) == 2 * pairs, "Missing/duplicate component samples")
        deltas = []
        for pair in range(pairs):
            order = list(ARMS if pair % 2 == 0 else ARMS[::-1])
            for arm in ARMS:
                require(by_pair[pair, arm]["order"] == order, "Component AB/BA order differs")
                elapsed = by_pair[pair, arm]["gpu_us"]
                require(math.isfinite(elapsed) and elapsed > 0, "Invalid component timing")
            deltas.append(
                by_pair[pair, "candidate"]["gpu_us"] - by_pair[pair, "baseline"]["gpu_us"]
            )
        require(
            statistics.median(deltas) == summary["paired_delta_us"], "Component summary mismatch"
        )
        require(
            all(statistics.median(deltas[parity::2]) < 0 for parity in (0, 1)),
            "Component timing did not improve both order strata",
        )
        expected_total += 2 * pairs
    require(len(samples) == expected_total, "Unexpected component timing samples")
    return {
        "receipt": {
            "path": str(args.component_receipt.resolve()),
            "sha256": digest(args.component_receipt),
        },
        "benchmark": {
            "path": str(args.component_bench.resolve()),
            "sha256": digest(args.component_bench),
        },
        "identity": expected,
    }


def source_identity(args, accepted):
    files = dict(sources())
    for module in (common, common.base, component, candidate, native, transfer):
        path = Path(module.__file__).resolve()
        files[str(path.relative_to(ROOT))] = digest(path)
    for relative in (
        "experiments/deepseek_v32_echo_official/src/q1_fused_prepare_model.py",
        "experiments/deepseek_v32_echo_official/src/q1_fused_prepare.cu",
        "experiments/deepseek_v32_mfu/src/q1_qkv.py",
        "experiments/deepseek_v32_mfu/src/q1_control.py",
        "experiments/deepseek_v32_mfu/src/q1_projection.py",
    ):
        files[relative] = digest(ROOT / relative)
    props = torch.cuda.get_device_properties(0)
    return {
        "sources": files,
        "component": accepted,
        "contract": CONTRACT,
        "checkpoint": checkpoint_identity(args.model),
        "request": {"path": str(args.request.resolve()), "sha256": digest(args.request)},
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
            for name in (*common.base.ENV_KEYS, "TVM_FFI_CACHE_DIR", "TVM_FFI_CUDA_ARCH_LIST")
        },
    }


def runtime():
    return {
        "model": common.model_runtime(),
        "private_generic_native": native.native_info(),
        "private_transfer_native": transfer.native_info(),
        "candidate_native": candidate.native_info(),
        "official_native": official_decode.native_info(
            torch.cuda.get_device_properties(0).multi_processor_count
        ),
        "mapped_native": collect_local_native_artifacts(required=True),
    }


def require_component_native(observed, accepted):
    expected = accepted["identity"]
    require(
        observed["private_generic_native"] == expected["private_generic_native"],
        "Generic ECHO native changed",
    )
    require(
        observed["candidate_native"] == expected["candidate_native"], "Candidate native changed"
    )
    require(observed["official_native"] == expected["official_native"], "Official native changed")
    libraries = {row["name"]: row["library"] for row in observed["mapped_native"]}
    for row in expected["mapped_local_native"]:
        require(libraries.get(row["name"]) == row["library"], "Component native missing or changed")


def check_model(model, snapshot, arm, folder, binding):
    folder.mkdir()
    saved = {}
    common.restore_cold(model, snapshot)
    with binding.bound(arm), ColdPrefetchObserver(model, HISTORY, 1, folder) as observer:
        public = binding.observer_entry()
        for index, token in enumerate(TOKENS):
            common.restore_cold(model, snapshot)
            before = binding.count
            value = common.outputs(model, [token], eager=True)
            require(
                binding.count - before == (3 if arm == "candidate" else 0),
                "Wrong eager dispatch count",
            )
            require(echo.logits is public, "Arm binding replaced diagnostic observer")
            eager = common.observed_record(model, observer, f"{token}_eager", value)
            common.restore_cold(model, snapshot)
            if index == 0:
                before = binding.count
                model.prepare_extend_graph([token], return_hidden=True)
                require(
                    binding.count - before == (12 if arm == "candidate" else 0),
                    "Wrong graph preparation dispatch count",
                )
            before = binding.count
            value = common.outputs(model, [token])
            require(binding.count == before, "Graph replay invoked a Python indexer")
            require(echo.logits is public, "Replay replaced diagnostic observer")
            graph = common.observed_record(model, observer, f"{token}_graph", value, captured=True)
            common.require_q1_proof(eager)
            common.require_q1_proof(graph)
            saved[str(token)] = {
                "eager": eager,
                "graph": graph,
                "comparison": common.compare_records(eager, graph),
            }
    require(not getattr(model, "_extend_graphs", {}), "Diagnostic graph survived cleanup")
    require(getattr(model, "_prefetch_validation_owner", None) is None, "Observer survived cleanup")
    require(echo.logits is binding.entry, "Public trampoline not restored after observer")
    return saved


def sample(models, snapshots, binding, pairs, *, profile):
    rows = []
    with common.profiler_range(profile):
        for pair in range(1 if profile else pairs):
            order = ARMS if pair % 2 == 0 else ARMS[::-1]
            for arm in order:
                model = models[arm]
                common.restore_cold(model, snapshots[arm])
                torch.cuda.synchronize()
                with (
                    binding.bound(arm),
                    common.nvtx_range(profile, "q1_fused_prepare_model/" + arm),
                ):
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


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("mode", choices=("check", "bench", "profile"))
    value.add_argument("--model", type=Path, default=Path("/preset-models"))
    value.add_argument("--request", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--component-receipt", type=Path, default=DEFAULT_COMPONENT)
    value.add_argument("--component-bench", type=Path, default=DEFAULT_BENCH)
    value.add_argument("--receipt", type=Path)
    value.add_argument("--pairs", type=int, default=100)
    return value


def _main():
    args = parser().parse_args()
    require(args.pairs >= 100 and args.pairs % 2 == 0, "At least 100 balanced pairs required")
    args.output_dir = args.output_dir.resolve()
    allowed = Path("/tmp/cxldsagr-checks") if args.mode == "check" else EXPERIMENT / "output"
    require(args.output_dir.is_relative_to(allowed), "Output directory outside required boundary")
    component.raw.environment()
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and torch.cuda.get_device_capability() == (9, 0),
        "Expose one SM90 GPU",
    )
    importlib.import_module("flashinfer.triton")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    ids = json.loads(args.request.read_text())["input_ids"]
    require(len(ids) == HISTORY + 1 and ids[-1] == TOKENS[0], "Expected fixed H64K/A1 request")
    accepted = component_binding(args)
    source = source_identity(args, accepted)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    models, snapshots, diagnostic, clean, prefix, memory = {}, {}, {}, {}, {}, {}
    result = {"passed": True}
    binding = Binding()
    with torch.inference_mode(), binding.public():
        candidate.module()
        try:
            for arm in ARMS:
                print("Loading " + arm, flush=True)
                model = common.new_model(args)
                models[arm] = model
                model.prepare_compute_graphs(CONTRACT["compute_graphs"])
                prefix[arm] = model.forward(ids[:HISTORY]).detach().cpu().clone()
                snapshots[arm] = model.snapshot_prefix()
                if args.mode == "check":
                    diagnostic[arm] = check_model(
                        model, snapshots[arm], arm, args.output_dir / arm, binding
                    )
                common.restore_cold(model, snapshots[arm])
                with binding.bound(arm):
                    capture = FullExtendGraphCapture(arm) if args.mode == "profile" else None
                    with (
                        common.profiler_range(capture is not None),
                        common.nvtx_range(
                            capture is not None, "q1_fused_prepare_model_capture/" + arm
                        ),
                    ):
                        graph = model.prepare_extend_graph(
                            [TOKENS[0]], return_hidden=True, capture_scope=capture
                        )
                        if capture is not None:
                            template = capture.finalize(graph)
                            result.setdefault("capture_templates", {})[arm] = template
                            write(args.output_dir / (arm + "_capture_template.json"), template)
                    for _ in range(CONTRACT["warmups"]):
                        common.restore_cold(model, snapshots[arm])
                        model.forward([TOKENS[0]], return_hidden=True)
                    if args.mode == "check":
                        clean[arm] = common.check_clean_graph(
                            model, snapshots[arm], diagnostic[arm]
                        )
                memory[arm] = {
                    "graph": graph.describe(),
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "device_used": torch.cuda.device_memory_used(),
                }
                print("Prepared " + arm, flush=True)
            observed = runtime()
            require_component_native(observed, accepted)
            identity = {"source": source, "timing_runtime": observed}
            if args.mode == "check":
                common.exact(prefix["baseline"], prefix["candidate"], "Independent prefix logits")
                result["cross_arm"] = {}
                for token in TOKENS:
                    result["cross_arm"][str(token)] = {
                        execution: common.compare_records(
                            diagnostic["baseline"][str(token)][execution],
                            diagnostic["candidate"][str(token)][execution],
                        )
                        for execution in ("eager", "graph")
                    }
                    common.exact_outputs(
                        clean["baseline"][str(token)]["output"],
                        clean["candidate"][str(token)]["output"],
                    )
                    common.exact_offsets(
                        clean["baseline"][str(token)]["offsets"],
                        clean["candidate"][str(token)]["offsets"],
                    )
                torch.save(
                    {"prefix": prefix, "diagnostic": diagnostic, "clean": clean},
                    args.output_dir / "outputs.pt",
                )
                result["cloned_outputs_all_offsets_and_actual_stage_proofs"] = True
                result["tokens"] = list(TOKENS)
            else:
                receipt = require_receipt(args.receipt, kind=KIND, identity=identity)
                result["receipt_sha256"] = receipt["receipt_sha256"]
                rows = sample(
                    models, snapshots, binding, args.pairs, profile=args.mode == "profile"
                )
                result["samples"] = rows
                if args.mode == "bench":
                    result["summary"] = common.summarize(rows, args.pairs)
            require(runtime() == observed, "Model runtime changed")
            common.archive(source, observed, args.output_dir)
        finally:
            common.close_models(models)
    require(source_identity(args, component_binding(args)) == source, "Source/evidence changed")
    payload = {
        "completed": True,
        "mode": args.mode,
        "run_id": args.output_dir.name,
        "identity": identity,
        "result": result,
        "memory": memory,
        "pairs": args.pairs,
        "boundary": "Independent cold real L0-L2 complete forwards; diagnostics removed before clean graphs. Cloned outputs, exact original unsupported delegation, unchanged official core. Profile node durations are not clean latency samples.",
    }
    write(args.output_dir / "result.json", payload)
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
                "public_observer_preserved": True,
            },
            artifacts={
                str(path.relative_to(args.output_dir)): path
                for path in args.output_dir.rglob("*")
                if path.is_file()
            },
        )
    print(
        json.dumps({"completed": True, "run_id": args.output_dir.name, "mode": args.mode}),
        flush=True,
    )


def main():
    parser().parse_args()
    component.raw.environment()
    with native.pinned(), transfer.pinned():
        _main()


if __name__ == "__main__":
    main()
