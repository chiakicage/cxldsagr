"""Independent validation, paired timing and profiling of bounded FIFO preparation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from evaluation.local_native import collect_local_native_artifacts
from evaluation.validation import identity_digest, require_receipt, write_receipt
from experiments.deepseek_v32_mfu.src import q1_prepare_baseline as baseline
from operators.deepseek_v32.indexer import _native_cache, cache_ops, echo

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT / "experiments/deepseek_v32_mfu/output/data/q1_free_prepare_candidate_20261008_01/source"
)
KIND = "deepseek-q1-bounded-free-prepare-v1"
VARIANTS = ("baseline", "candidate")
MISSING = 2**31 - 1
digest, require, exact, write = baseline.digest, baseline.require, baseline.exact, baseline.write


def sources(source_dir):
    identity = baseline.source_identity()
    for path in (
        Path(__file__),
        Path(cache_ops.__file__),
        Path(_native_cache.__file__),
        ROOT / "evaluation/validation.py",
        source_dir / "adapter.py",
        source_dir / "free_prepare.cu",
    ):
        identity["files"][str(path.resolve())] = digest(path)
    return identity


def load_candidate(source_dir, source):
    native_files = {
        name: value for name, value in source["files"].items() if Path(name).suffix != ".py"
    }
    native_identity = {"source_sha256": native_files}
    name = "cxldsagr_q1_free_prepare_" + identity_digest(native_identity)[:16]
    previous = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    try:
        module = _native_cache.load(
            name=name,
            sources=[str(source_dir / "free_prepare.cu")],
            extra_include_paths=[
                str(ROOT / "3rdparty/cutlass/include"),
                str(ROOT / "operators/deepseek_v32/indexer/csrc"),
            ],
            extra_cuda_cflags=echo._FLAGS,
            extra_ldflags=["-lcuda"],
            source_identity=native_identity,
        )
    finally:
        if previous is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous
    spec = importlib.util.spec_from_file_location("private_free_prepare", source_dir / "adapter.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return module, adapter, _native_cache.native_info(name)


def make_states(path):
    layer, initial = baseline.load_input(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for other in payload["layers"]:
        for field in ("free", "reverse"):
            exact(other["stages"]["initial"][field], initial[field], "Saved layer " + field)
    state = {
        "name": "saved_cold",
        "H": layer["H"],
        "P": layer["slots"],
        "clock": int(initial["clock"][0]),
        "host_capacity": layer["host_arena_tokens"],
        "priority": initial["priority"].clone(),
        "bitmap": initial["free"].clone(),
        "reverse": initial["reverse"].clone(),
    }
    states = [state]
    generator = torch.Generator().manual_seed(730108)
    for name, history, slots, mode in (
        ("partial_prefix", 65536, 65600, "prefix"),
        ("partial_random", 65536, 65600, "random"),
        ("last64_free", 65536, 65600, "last"),
        ("scattered64_free", 65536, 65600, "scattered"),
        ("tight_small", 32768, 32832, "last"),
        ("partial_mask_tail", 32768, 32833, "last"),
        ("minimum64", 0, 64, "last"),
    ):
        if mode == "prefix":
            occupied = torch.arange(1, 32769)
        elif mode == "random":
            occupied = torch.randperm(slots, generator=generator)[:50000] + 1
        elif mode == "scattered":
            occupied = torch.randperm(slots, generator=generator)[:history] + 1
        else:
            occupied = torch.arange(1, history + 1)
        priority = torch.full((slots + 1,), -1, dtype=torch.int64)
        priority[0] = MISSING
        priority[occupied] = torch.arange(len(occupied), dtype=torch.int64) % 257
        bitmap = torch.ones(slots + 1, dtype=torch.bool)
        bitmap[0] = False
        bitmap[occupied] = False
        reverse = torch.full((slots + 1,), MISSING, dtype=torch.int64)
        reverse[occupied] = torch.arange(len(occupied), dtype=torch.int64)
        states.append(
            {
                "name": name,
                "H": history,
                "P": slots,
                "clock": 256,
                "host_capacity": max(slots, 64),
                "priority": priority,
                "bitmap": bitmap,
                "reverse": reverse,
            }
        )
    return states


def allocate(state):
    slots = state["P"]
    return {
        "inputs": {name: state[name].cuda() for name in ("priority", "bitmap", "reverse")},
        "buffers": {
            "free_slots": torch.empty(slots, dtype=torch.int32, device="cuda"),
            "allocation_log": torch.empty(slots + 1, dtype=torch.int64, device="cuda"),
            "counter": torch.empty(1, dtype=torch.uint32, device="cuda"),
            "prefetch_stats": torch.empty(3, dtype=torch.int64, device="cuda"),
            "keys": torch.empty(slots, dtype=torch.int64, device="cuda"),
        },
    }


def restore(case, state):
    for name, tensor in case["inputs"].items():
        tensor.copy_(state[name])
    for value in case["buffers"].values():
        value.fill_(123)


def call(variant, case, state, module, adapter):
    inputs, buffers = case["inputs"], case["buffers"]
    if variant == "baseline":
        return cache_ops.prepare_prefetch(
            inputs["priority"],
            **buffers,
            timestamp=state["clock"],
            host_capacity=state["host_capacity"],
            query_count=1,
        )
    return adapter.prepare(module, **inputs, buffers=buffers, timestamp=state["clock"])


def inspect(variant, case, state, token):
    slots, buffers = state["P"], case["buffers"]
    for name, tensor in case["inputs"].items():
        exact(tensor.cpu(), state[name], "Unchanged " + name)
    expected = (torch.argsort(state["priority"][1:], stable=True) + 1).int()
    if variant == "candidate":
        require(
            bool((state["priority"][expected[:64].long()] == -1).all()), "Expected 64 free slots"
        )
        expected[64:] = MISSING
        require(token.prepared_limit == 64, "Explicit prepared bound")
    else:
        require(token.rows == 1, "Single request metadata")
    exact(buffers["free_slots"].cpu(), expected, "Stable slots and unused tail")
    exact(
        buffers["allocation_log"].cpu(),
        torch.full((slots + 1,), MISSING, dtype=torch.int64),
        "Complete journal reset",
    )
    exact(buffers["counter"].cpu(), torch.zeros(1, dtype=torch.uint32), "Counter reset")
    exact(buffers["prefetch_stats"].cpu(), torch.zeros(3, dtype=torch.int64), "Stats reset")
    exact(token.metadata[0].cpu(), torch.ones(1, dtype=torch.int32), "Request length")
    exact(token.metadata[1].cpu(), torch.zeros(1, dtype=torch.int32), "Request index")
    return {
        "slots": buffers["free_slots"].cpu(),
        "journal": buffers["allocation_log"].cpu(),
        "counter": buffers["counter"].cpu(),
        "stats": buffers["prefetch_stats"].cpu(),
        "metadata": torch.cat(token.metadata).cpu(),
    }


def expect_value_error(action, label):
    try:
        action()
    except ValueError:
        return label
    raise RuntimeError("Expected validation failure: " + label)


def token_checks(state, module, adapter):
    checks = []
    for limit in (-1, 65, 8192, True, 1.0, None):
        case = allocate(state)
        token = call("candidate", case, state, module, adapter)
        torch.cuda.synchronize()
        checks.append(
            expect_value_error(
                lambda token=token, case=case, limit=limit: token.consume(
                    case["buffers"], consumer_limit=limit
                ),
                "consumer_limit=" + str(limit),
            )
        )
    for limit in (0, 1, 64):
        case = allocate(state)
        token = call("candidate", case, state, module, adapter)
        token.consume(case["buffers"], consumer_limit=limit)
        torch.cuda.synchronize()
        checks.append(
            expect_value_error(
                lambda token=token, case=case, limit=limit: token.consume(
                    case["buffers"], consumer_limit=limit
                ),
                "second consumption limit=" + str(limit),
            )
        )
    for name in adapter.NAMES:
        for change in ("replace", "resize", "set_storage", "shape"):
            case = allocate(state)
            token = call("candidate", case, state, module, adapter)
            torch.cuda.synchronize()
            buffers = case["buffers"]
            tensor = buffers[name]
            if change == "replace":
                buffers[name] = tensor.clone()
            elif change == "resize":
                tensor.resize_(tensor.numel() + 1)
            elif change == "set_storage":
                tensor.set_(tensor.clone())
            else:
                tensor.unsqueeze_(0)
            checks.append(
                expect_value_error(
                    lambda token=token, buffers=buffers: token.consume(buffers, consumer_limit=64),
                    name + ":" + change,
                )
            )
    case = allocate(state)
    token = call("candidate", case, state, module, adapter)
    torch.cuda.synchronize()
    other = torch.cuda.Stream()
    with torch.cuda.stream(other):
        checks.append(
            expect_value_error(
                lambda: token.consume(case["buffers"], consumer_limit=64), "wrong_stream"
            )
        )
    case = allocate(state)
    case["buffers"]["keys"] = case["inputs"]["priority"][1:]
    checks.append(
        expect_value_error(
            lambda: call("candidate", case, state, module, adapter), "scratch_input_alias"
        )
    )
    return checks


def invalid_call(args, state, module, adapter):
    state = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }
    if args.invalid_case == "priority_low":
        state["priority"][100] = -2
    elif args.invalid_case == "priority_high":
        state["priority"][100] = state["clock"] + 1
    elif args.invalid_case == "insufficient_free":
        state["priority"][64:] = 0
        state["bitmap"][64:] = False
        state["reverse"][64:] = torch.arange(state["P"] - 63)
    elif args.invalid_case == "bitmap_mismatch":
        state["bitmap"][1] = False
    elif args.invalid_case == "reverse_mismatch":
        state["reverse"][1] = 0
    else:
        raise ValueError("Unknown invalid case")
    case = allocate(state)
    call(args.variant, case, state, module, adapter)
    torch.cuda.synchronize()
    raise RuntimeError("Invalid input unexpectedly completed")


def run_check(args, states, module, adapter):
    rows, evidence = [], {}
    for state in states:
        outputs = {}
        for variant in VARIANTS:
            case = allocate(state)
            restore(case, state)
            token = call(variant, case, state, module, adapter)
            torch.cuda.synchronize()
            outputs[variant] = inspect(variant, case, state, token)
            other = torch.cuda.Stream()
            other.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(other):
                restore(case, state)
                token = call(variant, case, state, module, adapter)
                other.synchronize()
                inspect(variant, case, state, token)
                if variant == "candidate":
                    token.consume(case["buffers"], consumer_limit=64)
            torch.cuda.current_stream().wait_stream(other)
        exact(
            outputs["baseline"]["slots"][:64],
            outputs["candidate"]["slots"][:64],
            "Baseline/candidate prefix",
        )
        rows.append(
            {
                "state": state["name"],
                "P": state["P"],
                "H": state["H"],
                "eager": True,
                "nondefault_stream": True,
            }
        )
        evidence[state["name"]] = {"inputs": state, "outputs": outputs}
    # Replay the same captured pointers with five different valid residency states.
    replay_states = [state for state in states if state["P"] == 65600]
    graph_outputs = {}
    for variant in VARIANTS:
        case = allocate(replay_states[0])
        call(variant, case, replay_states[0], module, adapter)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            token = call(variant, case, replay_states[0], module, adapter)
        graph_outputs[variant] = {}
        for state in replay_states:
            restore(case, state)
            graph.replay()
            torch.cuda.synchronize()
            graph_outputs[variant][state["name"]] = inspect(variant, case, state, token)
    negative = token_checks(states[0], module, adapter)
    children = []
    for variant in VARIANTS:
        invalids = ("priority_low", "priority_high")
        if variant == "candidate":
            invalids += ("insufficient_free", "bitmap_mismatch", "reverse_mismatch")
        for invalid in invalids:
            command = [
                sys.executable,
                "-B",
                "-m",
                "experiments.deepseek_v32_mfu.src.q1_free_prepare",
                "--mode",
                "invalid",
                "--source-dir",
                str(args.source_dir),
                "--input",
                str(args.input),
                "--output",
                str(args.output),
                "--variant",
                variant,
                "--invalid-case",
                invalid,
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            stderr = result.stderr.lower()
            require(
                result.returncode != 0
                and any(
                    marker in stderr
                    for marker in (
                        "device-side assert",
                        "unspecified launch failure",
                        "cudaerrorlaunchfailure",
                    )
                ),
                "Invalid child did not fail at CUDA boundary: "
                + variant
                + ":"
                + invalid
                + "\n"
                + result.stderr,
            )
            log = args.output / (variant + "_" + invalid + ".log")
            log.write_text(result.stdout + "\n" + result.stderr)
            children.append(
                {
                    "variant": variant,
                    "invalid": invalid,
                    "returncode": result.returncode,
                    "log": log.name,
                }
            )
    torch.save({"cases": evidence, "graph_outputs": graph_outputs}, args.output / "outputs.pt")
    result = {
        "passed": True,
        "cases": rows,
        "graph_replays_per_variant": len(replay_states),
        "token_negative_checks": negative,
        "invalid_children": children,
        "boundary": "Private preparation component only. No complete-model or cache integration acceptance.",
    }
    write(args.output / "check.json", result)
    return result


def capture(variant, case, state, module, adapter):
    restore(case, state)
    call(variant, case, state, module, adapter)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(graph):
        start.record()
        call(variant, case, state, module, adapter)
        end.record()
    torch.cuda.synchronize()
    return graph, start, end


def run_bench(args, states, module, adapter):
    rows = []
    for state in states:
        cases = {variant: allocate(state) for variant in VARIANTS}
        graphs = {
            variant: capture(variant, cases[variant], state, module, adapter)
            for variant in VARIANTS
        }
        for _ in range(args.warmups):
            for variant in VARIANTS:
                restore(cases[variant], state)
                graphs[variant][0].replay()
        torch.cuda.synchronize()
        samples = []
        for pair in range(args.repeats):
            order = VARIANTS if pair % 2 == 0 else VARIANTS[::-1]
            sample = {"pair": pair, "order": list(order), "us": {}}
            for variant in order:
                restore(cases[variant], state)
                graph, start, end = graphs[variant]
                graph.replay()
                end.synchronize()
                sample["us"][variant] = start.elapsed_time(end) * 1000
            samples.append(sample)
        medians = {
            variant: statistics.median(sample["us"][variant] for sample in samples)
            for variant in VARIANTS
        }
        rows.append(
            {
                "state": state["name"],
                "P": state["P"],
                "H": state["H"],
                "samples": samples,
                "median_us": medians,
                "speedup": medians["baseline"] / medians["candidate"],
                "paired_wins": sum(
                    sample["us"]["candidate"] < sample["us"]["baseline"] for sample in samples
                ),
                "order_medians_us": {
                    order: {
                        variant: statistics.median(
                            sample["us"][variant]
                            for sample in samples
                            if sample["order"][0] == order
                        )
                        for variant in VARIANTS
                    }
                    for order in VARIANTS
                },
            }
        )
    return {
        "passed": True,
        "warmups": args.warmups,
        "pairs_per_state": args.repeats,
        "cases": rows,
        "timing": "External CUDA event nodes immediately surround the entire prepare call in separate CUDA graphs and buffers. All input/scratch restores, allocation during capture, and host replay dispatch are outside the interval. All preparation kernels, CUB memsets and inter-kernel gaps are included. Alternating AB/BA; no cache flush.",
        "boundary": "Private component GPU latency; not complete-model speedup or Python API wall time.",
    }


def run_profile(args, states, module, adapter):
    state = next(state for state in states if state["name"] == args.state)
    case = allocate(state)
    for _ in range(args.warmups):
        restore(case, state)
        call(args.variant, case, state, module, adapter)
    restore(case, state)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range("q1_free_prepare/" + args.variant + "/" + args.state):
        call(args.variant, case, state, module, adapter)
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    return {
        "passed": True,
        "variant": args.variant,
        "state": args.state,
        "warmups": args.warmups,
        "profiled_complete_calls": 1,
        "boundary": "Intrusive direct-call profile, not clean component latency.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile", "invalid"), required=True)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--input", type=Path, default=baseline.DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="candidate")
    parser.add_argument("--state", default="saved_cold")
    parser.add_argument("--invalid-case")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    require(
        args.warmups > 0 and args.repeats >= 100 and args.repeats % 2 == 0,
        "At least 100 balanced pairs required",
    )
    require(torch.cuda.get_device_capability() == (9, 0), "SM90 required")
    torch.set_num_threads(8)
    args.source_dir, args.input, args.output = (
        args.source_dir.resolve(),
        args.input.resolve(),
        args.output.resolve(),
    )
    source = sources(args.source_dir)
    module, adapter, candidate_native = load_candidate(args.source_dir, source)
    states = make_states(args.input)
    if args.mode == "invalid":
        invalid_call(args, states[0], module, adapter)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    # Load the production baseline DSO without modifying module dispatch.
    warm_case = allocate(states[0])
    call("baseline", warm_case, states[0], module, adapter)
    torch.cuda.synchronize()
    mapped_native = collect_local_native_artifacts(required=False)
    production_native = [entry for entry in mapped_native if entry["category"] == "echo_indexer"]
    require(
        len(production_native) == 1 and production_native[0]["category"] == "echo_indexer",
        "Unexpected production native libraries",
    )
    require(
        len(mapped_native) == 2
        and any(
            entry["library"]["path"] == candidate_native["artifact_path"]
            and entry["library"]["sha256"] == candidate_native["artifact_sha256"]
            for entry in mapped_native
        ),
        "Candidate DSO is not the actually mapped artifact",
    )
    native = {"baseline": production_native, "candidate": candidate_native}
    identity = {
        "schema": KIND,
        "source": source,
        "input_sha256": digest(args.input),
        "native": native,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "states": [
            {key: value for key, value in state.items() if not isinstance(value, torch.Tensor)}
            for state in states
        ],
    }
    for filename in source["files"]:
        original = Path(filename)
        destination = args.output / "source" / original.relative_to("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    shutil.copyfile(args.input, args.output / "input_evidence.pt")
    shutil.copyfile(candidate_native["artifact_path"], args.output / "candidate.so")
    shutil.copyfile(production_native[0]["library"]["path"], args.output / "baseline.so")
    write(args.output / "identity.json", identity)
    write(args.output / "native.json", native)
    with torch.inference_mode():
        if args.mode == "check":
            result = run_check(args, states, module, adapter)
        else:
            receipt = require_receipt(args.receipt, kind=KIND, identity=identity)
            result = (
                run_bench(args, states, module, adapter)
                if args.mode == "bench"
                else run_profile(args, states, module, adapter)
            )
            result["receipt_sha256"] = receipt["receipt_sha256"]
    require(sources(args.source_dir) == source, "Sources changed during run")
    require(
        collect_local_native_artifacts(required=False) == mapped_native,
        "Mapped native artifacts changed during run",
    )
    require(
        digest(candidate_native["artifact_path"]) == candidate_native["artifact_sha256"],
        "Candidate native changed during run",
    )
    require(digest(args.input) == identity["input_sha256"], "Input changed during run")
    if args.mode == "check":
        artifacts = {path.name: path for path in args.output.iterdir() if path.is_file()}
        write_receipt(
            args.output / "receipt.json",
            kind=KIND,
            identity=identity,
            checks=result,
            artifacts=artifacts,
        )
    write(
        args.output / "result.json",
        {"mode": args.mode, "run_id": args.output.name, "identity": identity, **result},
    )
    print(json.dumps({"passed": True, "mode": args.mode, "output": str(args.output)}))


if __name__ == "__main__":
    main()
