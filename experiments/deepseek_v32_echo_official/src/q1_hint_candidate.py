"""Private exact-mean candidate acceptance and paired complete hint API timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_echo_official.src import q1_hint_baseline as base
from experiments.deepseek_v32_echo_official.src import q1_hint_exact as candidate
from operators.deepseek_v32.indexer import decode_hint, prefetch_hint

KIND = "deepseek-q1-hint-exact-candidate-v1"
ARMS = ("baseline", "candidate")
EXTRA_SOURCES = (
    "experiments/deepseek_v32_echo_official/src/q1_hint_candidate.py",
    "experiments/deepseek_v32_echo_official/src/q1_hint_exact.py",
    "experiments/deepseek_v32_echo_official/src/q1_hint_exact.cu",
    "operators/deepseek_v32/indexer/_native_cache.py",
)


def make_cases(corpus):
    """Deterministic CPU fixtures, including ownership outside each score view."""
    cases = {}

    def add(name, owner, scores=None, *, native=True, kth=-3.25, old=7.125, grad=False, cpu=False):
        values = torch.full((1, 2048), kth, dtype=torch.float32)
        offset = torch.arange(16, dtype=torch.float32)
        offset[1] = old
        cases[name] = {
            "owner": owner,
            "scores": owner if scores is None else scores,
            "values": values,
            "initial": offset,
            "native": native,
            "grad": grad,
            "cpu": cpu,
        }

    for name, row in corpus.items():
        owner = row["scores"].clone()
        add(name, owner, owner[:, :65537], old=0.0)
        cases[name]["values"] = row["values"].clone()
    n = 65537

    def repeated_bits(values):
        pattern = torch.tensor(values, dtype=torch.int64).to(torch.int32).view(torch.float32)
        return pattern[torch.arange(n) % len(pattern)].unsqueeze(0)

    add("all_nonfinite", repeated_bits([0x7FC01234, 0x7F800000, 0xFF800000, 0x7F800001]))
    add(
        "mixed_nonfinite",
        repeated_bits([0x3F800000, 0xC0000000, 0x7FC01234, 0x7F800000, 0xFF800000, 0x80000000]),
    )
    add("signed_zero", repeated_bits([0, 0x80000000]), kth=-0.0, old=-0.0)
    add(
        "subnormal",
        repeated_bits([1, 2, 0x7FFFFF, 0x800001, 0x80000001, 0x807FFFFF]),
        kth=1e-40,
        old=-1e-40,
    )
    add("overflow", torch.full((1, n), torch.finfo(torch.float32).max), kth=3e38, old=3e38)
    add(
        "overflow_cancellation",
        repeated_bits([0x7F7FFFFF, 0x7F7FFFFF, 0xFF7FFFFF, 0xFF7FFFFF]),
        kth=-3e38,
        old=3e38,
    )
    add(
        "large_small",
        repeated_bits([0x60AD78EC, 0x1E3CE508, 0xE0AD78EC, 0x3F800000]),
        kth=-1e20,
        old=1e-20,
    )
    boundary = torch.zeros(1, n)
    positions = [
        0,
        1,
        2,
        3,
        4,
        31,
        32,
        63,
        64,
        127,
        128,
        255,
        256,
        511,
        512,
        1023,
        1024,
        2047,
        2048,
        2049,
        32767,
        32768,
        65532,
        65533,
        65534,
        65535,
        65536,
    ]
    values = torch.tensor([1e20, -1e20, 1.0, -1.0, 1e-30], dtype=torch.float32)
    boundary[0, positions] = values[torch.arange(len(positions)) % len(values)]
    add("vector_tree_tail", boundary)
    tail = torch.zeros(1, n)
    tail[0, -1] = -13.25
    add("tail_only", tail)
    real = corpus["layer_0"]["scores"][:, :n]
    owner = torch.full((1, n + 2), 123.0)
    owner[:, 1 : n + 1].copy_(real)
    add("unaligned", owner, owner[:, 1 : n + 1], native=False)
    owner = torch.full((1, 2 * n + 2), 123.0)
    owner[:, 1 : 2 * n + 1 : 2].copy_(real)
    add("strided", owner, owner[:, 1 : 2 * n + 1 : 2], native=False)
    add("short_shape", real[:, :-1].clone(), native=False)
    add("two_queries", torch.cat((real, real.neg())), native=False)
    add("grad_enabled", real.clone(), native=False, grad=True)
    add("cpu_dispatch", real.clone(), native=False, cpu=True)
    return cases


def case_identities(cases):
    return {
        name: {
            key: base.tensor_identity(value) if isinstance(value, torch.Tensor) else value
            for key, value in case.items()
        }
        for name, case in cases.items()
    }


def identity(corpus, cases):
    return {
        "kind": KIND,
        "baseline": base.source_identity(corpus),
        "sources": {name: base.digest(base.ROOT / name) for name in EXTRA_SOURCES},
        "candidate_build": candidate.build_info(),
        "check_cases": case_identities(cases),
        "contract": "Exact mean plus unchanged Q1 EMA; Q>1 keeps offset[1]. Independent installed CUDA Torch oracle; input and all16 offset bits preserved.",
        "bench": "Three real layers; zero offsets before each arm outside events;100 balanced AB/BA pairs per graph/eager stratum by default. No model or candidate promotion claim.",
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TVM_FFI_CACHE_DIR",
                "TVM_FFI_CUDA_ARCH_LIST",
                "CXX",
                "CC",
                "CUDA_HOME",
                "CUDA_PATH",
            )
        },
    }


def runtime(destination=None):
    """Check-only extra Triton variants are retained; timing must use exact members."""
    import triton
    import tvm_ffi

    native = candidate.runtime_info()
    base.require(native is not None, "Candidate native was not loaded")
    kernels = {}
    for jit in (*prefetch_hint._kernels(), decode_hint._kernel()):
        for cache in jit.device_caches.values():
            for compiled in cache[0].values():
                assembly = {
                    key: value if isinstance(value, bytes) else value.encode()
                    for key, value in compiled.asm.items()
                }
                kernels[compiled.hash] = {
                    "name": compiled.name,
                    "metadata": json.loads(json.dumps(compiled.metadata._asdict(), default=str)),
                    "asm_sha256": {
                        key: hashlib.sha256(value).hexdigest() for key, value in assembly.items()
                    },
                }
                if destination is not None:
                    for key, value in assembly.items():
                        path = destination / "triton" / compiled.hash / key
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(value)
    base.require(kernels, "Baseline hint specializations are missing")
    roots = (Path(torch.__file__).resolve().parent, Path(tvm_ffi.__file__).resolve().parent)
    libraries = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5]).resolve()
            if ".so" in path.name and (
                any(path.is_relative_to(p) for p in roots)
                or path.name.startswith(("libcuda.so", "libcudart.so"))
                or str(path) == native["artifact_path"]
            ):
                libraries.add(path)
    base.require(Path(native["artifact_path"]) in libraries, "Candidate DSO is not mapped")
    return {
        "native": native,
        "triton_version": triton.__version__,
        "kernels": kernels,
        "loaded_libraries": {str(path): base.digest(path) for path in sorted(libraries)},
    }


def require_runtime(actual, accepted):
    for key in ("native", "triton_version", "loaded_libraries"):
        base.require(actual[key] == accepted[key], f"Runtime differs: {key}")
    for key, entry in actual["kernels"].items():
        base.require(accepted["kernels"].get(key) == entry, "Unaccepted Triton specialization")


def device_case(cpu):
    device = "cpu" if cpu["cpu"] else "cuda"
    owner = cpu["owner"].to(device, copy=True)
    return {
        **cpu,
        "owner": owner,
        "scores": owner.as_strided(
            cpu["scores"].shape, cpu["scores"].stride(), cpu["scores"].storage_offset()
        ),
        "values": cpu["values"].to(device, copy=True),
        "initial": cpu["initial"].to(device, copy=True),
        "offset": cpu["initial"].to(device, copy=True),
    }


def call(arm, case):
    function = candidate.update if arm == "candidate" else prefetch_hint.update_prefetch_hint
    function(case["scores"], case["offset"])
    if case["scores"].shape[0] == 1:
        decode_hint.update_decode_hint(case["values"], case["offset"])


def oracle(case):
    expected = case["offset"].clone()
    tail = case["scores"][-4:]
    finite = torch.isfinite(tail)
    expected[0] = tail.masked_fill(~finite, 0).sum() / finite.sum().clamp_min(1)
    if case["scores"].shape[0] == 1:
        expected[1] = expected[1] * 0.5 + case["values"][0, -1] * 0.5
    return expected


@contextmanager
def audited_dispatch():
    original = candidate.module
    counter = [0]

    def observed():
        counter[0] += 1
        return original()

    observed.cache_info = original.cache_info
    candidate.module = observed
    try:
        yield counter
    finally:
        candidate.module = original


def run_check(cases, destination):
    evidence = []

    def compare(label, arm, case, action, counter, expected_dispatches):
        original_owner, original_values = case["owner"].clone(), case["values"].clone()
        expected = oracle(case)
        before = counter[0]
        consumed = action()
        if not case["cpu"]:
            torch.cuda.synchronize()
        actual = case["offset"].cpu()
        base.require(counter[0] - before == expected_dispatches, "Incorrect native dispatch")
        for actual_tensor, expected_tensor, name in (
            (actual, expected.cpu(), "offset"),
            (case["owner"].cpu(), original_owner.cpu(), "owner"),
            (case["values"].cpu(), original_values.cpu(), "values"),
        ):
            base.require(
                torch.equal(base.bits(actual_tensor), base.bits(expected_tensor)),
                f"Bitwise mismatch: {label}/{arm}/{name}",
            )
        if isinstance(consumed, torch.Tensor):
            base.require(
                torch.equal(base.bits(consumed), base.bits(expected)),
                "Stream consumer saw different hint bits",
            )
        evidence.append(
            {
                "case": label,
                "arm": arm,
                "native_host_dispatches": counter[0] - before,
                "actual": actual,
                "expected": expected.cpu(),
                "stream_consumer": consumed.cpu() if isinstance(consumed, torch.Tensor) else None,
                "owner_before": original_owner.cpu(),
                "owner_after": case["owner"].cpu(),
                "values_before": original_values.cpu(),
                "values_after": case["values"].cpu(),
            }
        )

    with audited_dispatch() as counter:
        for name, cpu in cases.items():
            for arm in ARMS:
                case = device_case(cpu)
                with torch.set_grad_enabled(cpu["grad"]):
                    compare(
                        name + "/eager",
                        arm,
                        case,
                        lambda arm=arm, case=case: call(arm, case),
                        counter,
                        int(arm == "candidate" and cpu["native"]),
                    )
                    if cpu["cpu"]:
                        continue
                    graph = torch.cuda.CUDAGraph()
                    before = counter[0]
                    with torch.cuda.graph(graph):
                        call(arm, case)
                    base.require(
                        counter[0] - before == int(arm == "candidate" and cpu["native"]),
                        "Incorrect graph capture dispatch",
                    )
                    case["offset"].copy_(case["initial"])
                    compare(name + "/graph", arm, case, graph.replay, counter, 0)
        for arm in ARMS:
            case = device_case(cases["layer_0"])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call(arm, case)
            for name in base.LAYERS:
                changed = device_case(cases[name])
                case["owner"].copy_(changed["owner"])
                case["values"].copy_(changed["values"])
                case["offset"].copy_(changed["initial"])
                compare("changed/" + name, arm, case, graph.replay, counter, 0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda._sleep(1_000_000)
                case["owner"].copy_(changed["owner"])
                case["values"].copy_(changed["values"])
                case["offset"].copy_(changed["initial"])

                def consume(arm=arm, case=case):
                    call(arm, case)
                    return case["offset"].clone()

                compare("nondefault_stream", arm, case, consume, counter, int(arm == "candidate"))
            torch.cuda.current_stream().wait_stream(stream)
        dispatches = counter[0]
    torch.save(evidence, destination / "evidence.pt")
    return {
        "passed": True,
        "comparisons": len(evidence),
        "case_names": list(cases),
        "native_host_dispatches": dispatches,
        "dispatch_counter_boundary": "Check-only host dispatch/capture calls; graph replays are verified by outputs, not counted as host calls",
        "bitwise_offset_owner_values": True,
        "changed_graph_inputs": 3,
        "nondefault_stream_per_arm": True,
        "full_model_acceptance": False,
    }


def run_bench(cases, pairs, warmups):
    rows = []
    for name in base.LAYERS:
        prepared = {}
        for arm in ARMS:
            case = device_case(cases[name])
            for _ in range(3):
                call(arm, case)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call(arm, case)
            prepared[arm] = (case, graph)
        for execution in ("graph", "eager"):
            for arm in ARMS:
                case, graph = prepared[arm]
                for _ in range(warmups):
                    case["offset"].zero_()
                    torch.cuda.synchronize()
                    graph.replay() if execution == "graph" else call(arm, case)
            torch.cuda.synchronize()
            events = {
                arm: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for arm in ARMS
            }
            for pair in range(pairs):
                order = ARMS if pair % 2 == 0 else ARMS[::-1]
                for arm in order:
                    case, graph = prepared[arm]
                    case["offset"].zero_()
                    torch.cuda.synchronize()
                    begin, end = events[arm]
                    begin.record()
                    start = time.perf_counter_ns()
                    graph.replay() if execution == "graph" else call(arm, case)
                    end.record()
                    end.synchronize()
                    wall = (time.perf_counter_ns() - start) / 1000
                    rows.append(
                        {
                            "layer": name,
                            "execution": execution,
                            "pair": pair,
                            "order": "AB" if pair % 2 == 0 else "BA",
                            "arm": arm,
                            "gpu_us": begin.elapsed_time(end) * 1000,
                            "call_end_event_sync_wall_us": wall,
                        }
                    )
    summaries = []
    for name in base.LAYERS:
        for execution in ("graph", "eager"):
            selected = [
                row for row in rows if row["layer"] == name and row["execution"] == execution
            ]
            by_pair = {(row["pair"], row["arm"]): row for row in selected}
            delta = [
                by_pair[(pair, "candidate")]["gpu_us"] - by_pair[(pair, "baseline")]["gpu_us"]
                for pair in range(pairs)
            ]
            summaries.append(
                {
                    "layer": name,
                    "execution": execution,
                    "median_gpu_us": {
                        arm: statistics.median(
                            row["gpu_us"] for row in selected if row["arm"] == arm
                        )
                        for arm in ARMS
                    },
                    "paired_delta_gpu_us": delta,
                    "median_paired_delta_gpu_us": statistics.median(delta),
                    "candidate_wins": sum(value < 0 for value in delta),
                    "order_median_delta_gpu_us": {
                        order: statistics.median(delta[i] for i in range(pairs) if i % 2 == parity)
                        for parity, order in enumerate(("AB", "BA"))
                    },
                }
            )
    return {
        "samples": rows,
        "summary": summaries,
        "boundary": "Events enclose one complete mean+EMA API call/replay. Zero reset and pre-event sync excluded. Wall spans call, end-event record and synchronization; not model latency.",
        "preparation": "Three eager calls and one graph capture per arm/layer, then requested warmups separately per execution",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "bench"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-receipt", type=Path, default=base.DEFAULT_CORPUS)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=20)
    args = parser.parse_args()
    base.require(
        args.pairs >= 2 and args.pairs % 2 == 0 and args.warmups > 0,
        "Positive warmups and an even pair count required",
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
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
        "SM90 GPU required",
    )
    base.require(
        str(torch.__version__) == "2.12.1+cu130"
        and torch.version.git_version == "7269437d655783a26cba32aa88195b741ff496aa",
        "Pinned installed Torch required",
    )
    torch.set_num_threads(8)
    corpus = base.load_corpus(args.corpus_receipt)
    cases = make_cases(corpus)
    before = identity(corpus, cases)
    destination = args.output_dir.resolve()
    base.require(
        args.mode != "check" or not destination.is_relative_to(base.EXPERIMENT / "output"),
        "Independent checks belong outside experiment output",
    )
    destination.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        candidate.module()
        for arm in ARMS:
            call(arm, device_case(cases["layer_0"]))
        torch.cuda.synchronize()
        receipt = None
        if args.mode == "check":
            payload = run_check(cases, destination)
        else:
            receipt = require_receipt(args.receipt, kind=KIND, identity=before)
            accepted = json.loads(Path(receipt["artifact_paths"]["result.json"]).read_text())
            require_runtime(runtime(), accepted["runtime"])
            payload = run_bench(cases, args.pairs, args.warmups)
        observed = runtime(destination)
        base.require(
            observed["native"]["build_identity"]["source_identity"] == before["candidate_build"],
            "Loaded candidate source identity differs",
        )
        if receipt is not None:
            require_runtime(observed, accepted["runtime"])
        base.require(
            identity(base.load_corpus(args.corpus_receipt), make_cases(corpus)) == before,
            "Execution sources or inputs changed",
        )
    for relative in (*base.SOURCE_PATHS, *EXTRA_SOURCES):
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.ROOT / relative, target)
        expected = before["sources"].get(relative, before["baseline"]["sources"].get(relative))
        base.require(base.digest(target) == expected, "Source archive differs")
    native_path = Path(observed["native"]["artifact_path"])
    shutil.copy2(native_path, destination / "candidate.so")
    base.require(
        base.digest(destination / "candidate.so") == observed["native"]["artifact_sha256"],
        "Native archive differs",
    )
    torch.save(cases, destination / "inputs.pt")
    base.write(
        destination / "result.json",
        {
            "completed": True,
            "mode": args.mode,
            "run_id": destination.name,
            "identity": before,
            "runtime": observed,
            "payload": payload,
            "warmups": args.warmups,
            "pairs": args.pairs,
            "receipt": None
            if receipt is None
            else {"path": str(args.receipt.resolve()), "sha256": base.digest(args.receipt)},
        },
    )
    if args.mode == "check":
        artifacts = {
            str(path.relative_to(destination)): path
            for path in destination.rglob("*")
            if path.is_file()
        }
        write_receipt(
            destination / "receipt.json",
            kind=KIND,
            identity=before,
            checks=payload,
            artifacts=artifacts,
        )
    print(destination / "result.json", flush=True)


if __name__ == "__main__":
    main()
