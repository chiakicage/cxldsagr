"""Independent check, clean paired timing and one-call profile of Q1 promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from evaluation.validation import identity_digest, require_receipt, write_receipt
from operators.deepseek_v32.indexer import _native_cache, official_prefetch

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT
    / "experiments/deepseek_v32_echo_official/output/data/deepseek_q1_promotion_candidate_20261008_01/source"
)
DEFAULT_INPUT = (
    ROOT
    / "experiments/deepseek_v32_echo_official/output/data/q1_inputs_20261008_01/kernel_inputs_layer_2.pt"
)
ADAPTER_TEST = ROOT / "operators/deepseek_v32/indexer/tests/test_official_prefetch.py"
MISSING = 2**31 - 1
FLAGS = ["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"]
KIND = "deepseek-official-q1-promotion-candidate-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_module(path, variant):
    identity = {"source_sha256": {str(path): digest(path)}}
    name = f"cxldsagr_q1_promotion_{variant}_{identity_digest(identity)[:16]}"
    previous = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    try:
        module = _native_cache.load(
            name=name,
            sources=[str(path)],
            extra_include_paths=[],
            extra_cuda_cflags=FLAGS,
            extra_ldflags=[],
            source_identity=identity,
        )
    finally:
        if previous is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous
    return module, _native_cache.native_info(name)


def allocate_offset(shape, dtype, offset):
    elements = 1
    for size in shape:
        elements *= size
    return torch.empty(elements + offset, dtype=dtype, device="cuda")[offset:].view(shape)


def make_case(*, slots, attempts, occupied, layout, offsets, payload, seed=7301):
    count = min(attempts, 64)
    host_capacity = max(slots, 384)
    generator = torch.Generator().manual_seed(seed)
    if payload is None:
        ids = torch.cat(
            (torch.zeros(1, dtype=torch.int64), torch.randperm(127, generator=generator)[:63] + 1)
        )
        stage_bits = torch.randint(-32768, 32768, (64, 576), dtype=torch.int16, generator=generator)
        stage_cpu = stage_bits.view(torch.bfloat16)
    else:
        ids = payload["indices"].long().flatten()
        ids = ids[(ids >= 0) & (ids < 65536)][:64]
        if len(ids) != 64 or ids.unique().numel() != 64:
            raise ValueError("Layer-2 payload does not provide 64 distinct history IDs")
        stage_cpu = payload["kv"][ids].contiguous()
    selected = torch.arange(1, 65, dtype=torch.int64)
    if layout == "random":
        selected = torch.randperm(slots, generator=generator)[:64] + 1
    stage_ids = torch.full((64,), -1, dtype=torch.int32)
    stage_ids[:count] = ids[:count].int()
    h2d = torch.full((host_capacity,), MISSING, dtype=torch.int32)
    d2h = torch.full((slots + 1,), MISSING, dtype=torch.int64)
    allocations = torch.full_like(d2h, MISSING)
    old_ids = torch.arange(host_capacity - occupied, host_capacity, dtype=torch.int64)
    occupied_slots = selected[:occupied]
    if occupied:
        h2d[old_ids] = occupied_slots.int()
        d2h[occupied_slots] = old_ids
    h2d[ids[:count]] = slots + 1 + torch.arange(count, dtype=torch.int32)
    initial_records = torch.zeros((slots + 1, 576), dtype=torch.bfloat16)
    records = allocate_offset(initial_records.shape, initial_records.dtype, offsets[1])
    records.copy_(initial_records)
    stage = allocate_offset((64, 576), torch.bfloat16, offsets[0])
    stage.copy_(stage_cpu)
    tensors = {
        "stage_ids": stage_ids.cuda(),
        "stage": stage,
        "counter": torch.tensor([attempts], dtype=torch.uint32, device="cuda"),
        "records": records,
        "h2d": h2d.cuda(),
        "d2h": d2h.cuda(),
        "sorted_slots": selected.int().cuda(),
        "allocations": allocations.cuda(),
        "stats": torch.full((3,), 99, dtype=torch.int64, device="cuda"),
    }
    expected_records = initial_records.clone()
    expected_records[selected[:count]] = stage_cpu[:count]
    expected_h2d, expected_d2h = h2d.clone(), d2h.clone()
    evicted = d2h[selected[:count]]
    evicted = evicted[evicted != MISSING]
    expected_h2d[evicted] = MISSING
    expected_h2d[ids[:count]] = selected[:count].int()
    expected_d2h[selected[:count]] = ids[:count]
    expected_log = allocations.clone()
    expected_log[selected[:count]] = ids[:count]
    expected = {
        "records": expected_records,
        "h2d": expected_h2d,
        "d2h": expected_d2h,
        "allocations": expected_log,
        "stats": torch.tensor([count, len(evicted), attempts - count], dtype=torch.int64),
    }
    reset = {key: tensors[key].clone() for key in ("h2d", "d2h", "allocations", "stats")}
    return {
        "tensors": tensors,
        "reset": reset,
        "expected": expected,
        "initial_records": initial_records,
    }


def restore(case, *, records=False):
    for name, value in case["reset"].items():
        case["tensors"][name].copy_(value)
    if records:
        case["tensors"]["records"].copy_(case["initial_records"])


def call(module, case):
    tensors = case["tensors"]
    official_prefetch._call(
        module,
        "official_prefetch_promote",
        tensors["records"].device,
        *(
            tensors[key]
            for key in (
                "stage_ids",
                "stage",
                "counter",
                "records",
                "h2d",
                "d2h",
                "sorted_slots",
                "allocations",
                "stats",
            )
        ),
    )


def exact_bytes(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


def check_case(modules, params):
    case = make_case(**params)
    outputs = {}
    for variant, module in modules.items():
        restore(case, records=True)
        call(module, case)
        torch.cuda.synchronize()
        outputs[variant] = {}
        for key, expected in case["expected"].items():
            actual = case["tensors"][key].cpu()
            if not exact_bytes(actual, expected):
                raise AssertionError(f"CPU reference mismatch for {variant}/{key}: {params}")
            outputs[variant][key] = actual
    for key in case["expected"]:
        if not exact_bytes(outputs["baseline"][key], outputs["candidate"][key]):
            raise AssertionError(f"Candidate differs from baseline: {key}")


def adapter_tests(module):
    import pytest

    class Plugin:
        def __init__(self):
            self.passed = 0
            self.skipped = 0

        @pytest.fixture(autouse=True)
        def candidate_module(self, monkeypatch):
            monkeypatch.setattr(official_prefetch, "_module", lambda: module)

        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed:
                self.passed += 1
            if report.skipped:
                self.skipped += 1

    plugin = Plugin()
    code = pytest.main(
        [str(ADAPTER_TEST), "-q", "-x", "--tb=short", "-p", "no:cacheprovider"], plugins=[plugin]
    )
    if code != 0 or plugin.passed != 22 or plugin.skipped:
        raise AssertionError(
            f"Adapter test result differs: {code}, {plugin.passed}, {plugin.skipped}"
        )
    return {"passed": plugin.passed, "skipped": plugin.skipped, "module_only_substitution": True}


def formal_params(payload, *, layout, occupied, offset=0):
    return {
        "slots": 65600,
        "attempts": 64,
        "occupied": occupied,
        "layout": layout,
        "offsets": (offset, offset),
        "payload": payload,
    }


def run_check(args, modules, identity, payload):
    rows = []
    for attempts in (0, 1, 17, 64, 91, 2**32 - 1):
        for occupied in (0, 32, 64):
            for offsets in ((0, 0), (1, 0), (0, 1), (3, 5)):
                params = {
                    "slots": 96,
                    "attempts": attempts,
                    "occupied": occupied,
                    "layout": "random",
                    "offsets": offsets,
                    "payload": None,
                }
                check_case(modules, params)
                rows.append({key: value for key, value in params.items() if key != "payload"})
    for layout in ("consecutive", "random"):
        for occupied in (0, 32, 64):
            for offset in (0, 1):
                params = formal_params(payload, layout=layout, occupied=occupied, offset=offset)
                check_case(modules, params)
                rows.append({key: value for key, value in params.items() if key != "payload"})
    suite = adapter_tests(modules["candidate"])
    invalid = []
    for case in ("duplicate_slot", "invalid_host", "owner_mismatch", "occupied_journal"):
        command = [
            sys.executable,
            "-B",
            "-m",
            "experiments.deepseek_v32_echo_official.src.q1_promotion",
            "--mode",
            "invalid",
            "--invalid-case",
            case,
            "--source-dir",
            str(args.source_dir),
            "--input",
            str(args.input),
            "--output-dir",
            str(args.output_dir / ("invalid_" + case)),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        (args.output_dir / ("invalid_" + case + ".stderr")).write_text(completed.stderr)
        if completed.returncode == 0 or "device-side assert triggered" not in completed.stderr:
            raise AssertionError(
                f"Invalid input did not preserve device failure: {case}\n{completed.stderr}"
            )
        invalid.append({"case": case, "returncode": completed.returncode, "device_assert": True})
    result = {
        "passed": True,
        "custom_cases": rows,
        "custom_case_count": len(rows),
        "adapter_tests": suite,
        "invalid_inputs": invalid,
        "boundary": "Exact promotion-only state comparisons and existing adapter tests; no complete-model correctness or performance claim.",
    }
    write_json(args.output_dir / "check.json", result)
    return result


def capture(module, case):
    restore(case)
    call(module, case)
    torch.cuda.synchronize()
    restore(case)
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(graph):
        start.record()
        call(module, case)
        end.record()
    torch.cuda.synchronize()
    return graph, start, end


def run_bench(args, modules, payload):
    rows = []
    for layout in ("consecutive", "random"):
        for occupied in (0, 32, 64):
            params = formal_params(payload, layout=layout, occupied=occupied)
            case = make_case(**params)
            graphs = {variant: capture(module, case) for variant, module in modules.items()}
            for _ in range(args.warmups):
                for graph, _, _ in graphs.values():
                    restore(case)
                    graph.replay()
            torch.cuda.synchronize()
            samples = {variant: [] for variant in modules}
            for pair in range(args.repeats):
                order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
                for variant in order:
                    restore(case)
                    graph, start, end = graphs[variant]
                    graph.replay()
                    end.synchronize()
                    samples[variant].append(start.elapsed_time(end) * 1000)
            rows.append(
                {
                    "layout": layout,
                    "occupied_selected_slots": occupied,
                    "N": 65537,
                    "P": 65600,
                    "stage_records": 64,
                    "record_width": 576,
                    "record_dtype": "bfloat16",
                    "offset_bytes": 0,
                    "samples_us": samples,
                    "median_us": {
                        variant: statistics.median(values) for variant, values in samples.items()
                    },
                    "candidate_paired_wins": sum(
                        candidate < baseline
                        for candidate, baseline in zip(
                            samples["candidate"], samples["baseline"], strict=True
                        )
                    ),
                }
            )
    return {
        "passed": True,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "cases": rows,
        "timing": "External CUDA event nodes captured immediately before and after the complete promote call in one graph. Metadata restore and host replay dispatch are outside the measured interval. Both candidate kernels are included. Alternating paired order, no cache flush.",
        "boundary": "Local staging promotion harness only, not full-model performance or official score/prefetch performance.",
    }


def run_profile(args, modules, payload):
    case = make_case(**formal_params(payload, layout=args.layout, occupied=args.occupied))
    module = modules[args.variant]
    for _ in range(args.warmups):
        restore(case)
        call(module, case)
    torch.cuda.synchronize()
    restore(case)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range(
        f"q1_promotion/{args.variant}/{args.layout}/occupied_{args.occupied}"
    ):
        call(module, case)
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    return {
        "passed": True,
        "variant": args.variant,
        "layout": args.layout,
        "occupied_selected_slots": args.occupied,
        "profiled_calls": 1,
        "kernels_in_call": 1 if args.variant == "baseline" else 2,
    }


def invalid_call(args, module):
    case = make_case(
        slots=96, attempts=64, occupied=64, layout="consecutive", offsets=(0, 0), payload=None
    )
    tensors = case["tensors"]
    if args.invalid_case == "duplicate_slot":
        tensors["sorted_slots"][1] = tensors["sorted_slots"][0]
    elif args.invalid_case == "invalid_host":
        tensors["stage_ids"][0] = -1
    elif args.invalid_case == "owner_mismatch":
        tensors["h2d"][320] = MISSING
    elif args.invalid_case == "occupied_journal":
        tensors["allocations"][1] = 3
    else:
        raise ValueError("Unknown invalid case")
    call(module, case)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile", "invalid"), required=True)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--variant", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--layout", choices=("consecutive", "random"), default="consecutive")
    parser.add_argument("--occupied", type=int, choices=(0, 32, 64), default=0)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--invalid-case")
    args = parser.parse_args()
    args.source_dir, args.input, args.output_dir = (
        args.source_dir.resolve(),
        args.input.resolve(),
        args.output_dir.resolve(),
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Promotion candidate requires a visible SM90 GPU")
    if args.warmups < 1 or args.repeats < 1:
        raise ValueError("Positive warmup/repeat counts are required")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    modules, native = {}, {}
    for variant in ("baseline", "candidate"):
        modules[variant], native[variant] = load_module(
            args.source_dir / (variant + ".cu"), variant
        )
    if args.mode == "invalid":
        invalid_call(args, modules["candidate"])
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_files = [
        Path(__file__),
        ADAPTER_TEST,
        Path(official_prefetch.__file__),
        Path(_native_cache.__file__),
        *(args.source_dir / (variant + ".cu") for variant in modules),
    ]
    sources = {str(path): digest(path) for path in source_files}
    for path in source_files:
        destination = args.output_dir / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    identity = {
        "schema": KIND,
        "source_sha256": sources,
        "input_sha256": digest(args.input),
        "native": {
            variant: {
                "path": record["artifact_path"],
                "sha256": record["artifact_sha256"],
                "build_identity_sha256": identity_digest(record["build_identity"]),
            }
            for variant, record in native.items()
        },
        "device_capability": list(torch.cuda.get_device_capability()),
        "device_name": torch.cuda.get_device_name(),
    }
    write_json(args.output_dir / "native.json", native)
    write_json(args.output_dir / "identity.json", identity)
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    if (
        payload["query_start"] != 65536
        or tuple(payload["kv"].shape) != (65537, 576)
        or payload["kv"].dtype != torch.bfloat16
    ):
        raise ValueError("Unexpected actual layer-2 payload")
    if args.mode == "check":
        result = run_check(args, modules, identity, payload)
    else:
        require_receipt(args.receipt, kind=KIND, identity=identity)
        result = (
            run_bench(args, modules, payload)
            if args.mode == "bench"
            else run_profile(args, modules, payload)
        )
    for path, expected in sources.items():
        if digest(path) != expected:
            raise RuntimeError(f"Source changed during execution: {path}")
    if digest(args.input) != identity["input_sha256"]:
        raise RuntimeError("Input changed during execution")
    for record in native.values():
        if digest(record["artifact_path"]) != record["artifact_sha256"]:
            raise RuntimeError("Native artifact changed during execution")
    if args.mode == "check":
        write_receipt(
            args.output_dir / "receipt.json",
            kind=KIND,
            identity=identity,
            checks=result,
            artifacts={
                "check.json": args.output_dir / "check.json",
                "native.json": args.output_dir / "native.json",
            },
        )
    result.update(
        mode=args.mode,
        run_id=args.output_dir.name,
        identity=identity,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch_version=torch.__version__,
    )
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"identity", "custom_cases", "cases"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
