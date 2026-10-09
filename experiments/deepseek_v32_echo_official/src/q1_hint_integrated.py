"""Check the relocated production exact mean against the installed Torch expression."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from evaluation.local_native import collect_local_native_artifacts
from evaluation.validation import write_receipt
from experiments.deepseek_v32_echo_official.src import q1_hint_baseline as base
from experiments.deepseek_v32_echo_official.src import q1_hint_candidate as component
from experiments.deepseek_v32_echo_official.src import q1_hint_model as model_check
from experiments.deepseek_v32_mfu.src.run_contract import checkpoint_identity
from models.deepseek_v32 import attention as attention_module
from operators.deepseek_v32.indexer import decode_hint, prefetch_hint

KIND = "deepseek-q1-hint-exact-integrated-v1"
PRODUCTION_MODULE = "operators.deepseek_v32.indexer.q1_hint_exact"
PRODUCTION_NAME = "cxldsagr_q1_hint_exact_mean_sm90"
EXTRA_SOURCES = (
    "experiments/deepseek_v32_echo_official/src/q1_hint_integrated.py",
    *base.SOURCE_PATHS,
    *component.EXTRA_SOURCES,
    *model_check.EXTRA_SOURCES,
)


@contextmanager
def replacements(changes):
    """Restore every explicit check-helper adaptation, preserving all failures."""
    previous = []
    try:
        for target, name, value in changes:
            previous.append((target, name, getattr(target, name)))
            setattr(target, name, value)
        yield
    finally:
        model_check.cleanup(
            tuple(
                lambda target=target, name=name, value=value: setattr(target, name, value)
                for target, name, value in reversed(previous)
            )
        )


@contextmanager
def model_binding(arm):
    production = prefetch_hint.update_prefetch_hint
    base.require(arm in model_check.ARMS, "Unknown integration arm")
    base.require(
        attention_module.update_prefetch_hint is production, "Production model binding changed"
    )
    try:
        changes = (
            ((attention_module, "update_prefetch_hint", prefetch_hint._reference_update),)
            if arm == "baseline"
            else ()
        )
        with replacements(changes):
            yield
    finally:
        model_check.cleanup(
            (
                lambda: base.require(
                    attention_module.update_prefetch_hint is production,
                    "Production binding was not restored",
                ),
            )
        )


def component_call(arm, case):
    base.require(arm in model_check.ARMS, "Unknown component arm")
    update = (
        prefetch_hint._reference_update if arm == "baseline" else prefetch_hint.update_prefetch_hint
    )
    update(case["scores"], case["offset"])
    if case["scores"].shape[0] == 1:
        decode_hint.update_decode_hint(case["values"], case["offset"])


@contextmanager
def audited_dispatch(adapter):
    original, counter = adapter.module, [0]

    def observed():
        counter[0] += 1
        return original()

    observed.cache_info = original.cache_info
    with replacements(((adapter, "module", observed),)):
        yield counter


@contextmanager
def check_helpers(adapter):
    """Adapt only check callbacks; the candidate production dispatcher is untouched."""
    with replacements(
        (
            (component, "call", component_call),
            (component, "audited_dispatch", lambda: audited_dispatch(adapter)),
            (model_check, "binding", model_binding),
        )
    ):
        yield


def no_private_native():
    base.require(
        model_check.candidate.module.cache_info().currsize == 0, "Private candidate DSO was loaded"
    )


def runtime(adapter, destination):
    # The frozen exporter reads candidate.runtime_info only. Its explicit alias
    # here selects the production adapter and never invokes a private function.
    with replacements(((component, "candidate", adapter),)):
        hint = component.runtime(destination)
    native = collect_local_native_artifacts(required=True)
    base.require(hint["native"] == adapter.runtime_info(), "Production native identity differs")
    base.require(
        sum(entry["library"]["path"] == hint["native"]["artifact_path"] for entry in native) == 1,
        "Production hint DSO is not uniquely mapped",
    )
    no_private_native()
    return {"hint": hint, "model": model_check.model_runtime(), "mapped_native": native}


def source_identity(args, adapter, cases):
    files = model_check.sources()
    files.update({relative: base.digest(base.ROOT / relative) for relative in EXTRA_SOURCES})
    properties = torch.cuda.get_device_properties(0)
    return {
        "kind": KIND,
        "sources": files,
        "production_module": PRODUCTION_MODULE,
        "production_name": adapter.NAME,
        "production_build": adapter.build_info(),
        "checkpoint": checkpoint_identity(args.model),
        "request": {"path": str(args.request.resolve()), "sha256": base.digest(args.request)},
        "corpus_receipt": {
            "path": str(args.corpus_receipt.resolve()),
            "sha256": base.digest(args.corpus_receipt),
        },
        "cases": component.case_identities(cases),
        "gpu": {
            "name": properties.name,
            "uuid": str(properties.uuid),
            "capability": [properties.major, properties.minor],
        },
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "optimize": sys.flags.optimize,
        },
        "precision": torch.backends.cuda.matmul.fp32_precision,
        "threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            name: os.environ.get(name)
            for name in (
                *base.ENV_KEYS,
                "TVM_FFI_CACHE_DIR",
                "TVM_FFI_CUDA_ARCH_LIST",
                "CXX",
                "CC",
                "CUDA_HOME",
                "CUDA_PATH",
                "TRITON_PTXAS_BLACKWELL_PATH",
                "PYTHONPATH",
                "PYTHONOPTIMIZE",
            )
        },
        "contract": {
            "component_baseline": "Installed Torch finite-mask, sum and count expression plus unchanged decode EMA",
            "component_candidate": "Unpatched production update_prefetch_hint plus unchanged decode EMA",
            "model": "Independent checkpoint L0-L2 models, H65536/A1/capacity65537, cold actual-stage proofs, tokens111090/111091/111092, eager and replaced clean graphs",
            "consumer": "Separate capacity65539 models; reference-versus-production Q1 then actual production A2 consumer without restore/truncate between calls",
            "boundary": "Correctness only. The new relocated production DSO receives a new receipt; private native acceptance is not reused as integrated acceptance.",
        },
    }


def check_models(args, ids, adapter):
    saved, clean, prefixes = {}, {}, {}
    for arm in model_check.ARMS:
        model = model_check.new_model(args)
        try:
            print("Checking integrated model " + arm, flush=True)
            model.prepare_compute_graphs([1024, 1])
            prefixes[arm] = model.forward(ids[:65536]).detach().cpu().clone()
            snapshot = model.snapshot_prefix()
            with check_helpers(adapter):
                saved[arm] = model_check.check_model(model, snapshot, arm, args.output_dir / arm)
            model_check.restore_cold(model, snapshot)
            with model_binding(arm), audited_dispatch(adapter) as counter:
                model.prepare_extend_graph([model_check.TOKENS[0]], return_hidden=True)
                base.require(
                    counter[0] == (12 if arm == "candidate" else 0),
                    "Clean graph captured wrong production dispatch",
                )
                before = counter[0]
                clean[arm] = model_check.check_clean_graph(model, snapshot, saved[arm])
                base.require(
                    counter[0] == before, "Clean replay unexpectedly called a host hint function"
                )
        finally:
            model_check.close_models({arm: model})
        model = snapshot = None
    model_check.exact(
        prefixes["baseline"], prefixes["candidate"], "Independent integrated prefixes"
    )
    comparisons = {}
    for token in map(str, model_check.TOKENS):
        comparisons[token] = {
            execution: model_check.compare_records(
                saved["baseline"][token][execution], saved["candidate"][token][execution]
            )
            for execution in ("eager", "graph")
        }
        model_check.exact_outputs(
            clean["baseline"][token]["output"], clean["candidate"][token]["output"]
        )
        model_check.exact_offsets(
            clean["baseline"][token]["offsets"], clean["candidate"][token]["offsets"]
        )
    with check_helpers(adapter), audited_dispatch(adapter) as counter:
        consumer = model_check.check_consumer(args, ids)
        base.require(
            counter[0] == 3,
            "Sequential consumer did not execute exactly three production Q1 hint calls",
        )
    torch.save(
        {"prefix": prefixes, "diagnostic": saved, "clean": clean, "consumer": consumer},
        args.output_dir / "model_outputs.pt",
    )
    return {
        "passed": True,
        "cross_arm": comparisons,
        "tokens": list(model_check.TOKENS),
        "all_offsets_bitwise": True,
        "actual_prepared_cap_both_arms": 64,
        "clean_graph_checked": True,
        "actual_q1_to_a2_consumer": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/preset-models"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--corpus-receipt", type=Path, default=base.DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    base.require(
        not args.output_dir.is_relative_to(base.EXPERIMENT / "output"),
        "Integrated correctness evidence belongs outside experiment output",
    )
    for name, directory in (
        ("TRITON_CACHE_DIR", "/tmp/cxldsagr-q1-hint-integrated-cache/triton"),
        ("CUDA_CACHE_PATH", "/tmp/cxldsagr-q1-hint-integrated-cache/cuda"),
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
    importlib.import_module("flashinfer.triton")
    adapter = importlib.import_module(PRODUCTION_MODULE)
    base.require(adapter.NAME == PRODUCTION_NAME, "Expected relocated production native module")
    no_private_native()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    ids = json.loads(args.request.read_text())["input_ids"]
    base.require(len(ids) == 65537 and ids[-1] == model_check.TOKENS[0], "Expected H64K/A1 request")
    cases = component.make_cases(base.load_corpus(args.corpus_receipt))
    source = source_identity(args, adapter, cases)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    component_dir = args.output_dir / "component"
    component_dir.mkdir()
    with torch.inference_mode():
        # Compile and exercise the real production dispatcher before installing
        # any check-only callback adapters or recording CUDA Graph captures.
        adapter.module()
        for cpu in cases.values():
            case = component.device_case(cpu)
            with torch.set_grad_enabled(cpu["grad"]):
                component_call("candidate", case)
        torch.cuda.synchronize()
        with check_helpers(adapter):
            component_result = component.run_check(cases, component_dir)
        model_result = check_models(args, ids, adapter)
        observed = runtime(adapter, args.output_dir)
        base.require(
            observed["hint"]["native"]["build_identity"]["source_identity"]
            == source["production_build"],
            "Loaded production build differs from checked source",
        )
        base.require(
            source_identity(args, adapter, cases) == source,
            "Integrated execution source/input identity changed",
        )
        model_check.archive(source, observed, args.output_dir)
        no_private_native()
    torch.save(cases, component_dir / "inputs.pt")
    identity = {"source": source, "runtime": observed}
    checks = {
        "passed": True,
        "component": component_result,
        "model": model_result,
        "private_dso_loaded": False,
        "unpatched_production_candidate": True,
    }
    base.write(
        args.output_dir / "result.json",
        {
            "completed": True,
            "mode": "check",
            "run_id": args.output_dir.name,
            "identity": identity,
            "checks": checks,
        },
    )
    write_receipt(
        args.output_dir / "receipt.json",
        kind=KIND,
        identity=identity,
        checks=checks,
        artifacts={
            str(path.relative_to(args.output_dir)): path
            for path in args.output_dir.rglob("*")
            if path.is_file()
        },
    )
    print(
        json.dumps(
            {
                "completed": True,
                "run_id": args.output_dir.name,
                "component_comparisons": component_result["comparisons"],
                "model_passed": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
