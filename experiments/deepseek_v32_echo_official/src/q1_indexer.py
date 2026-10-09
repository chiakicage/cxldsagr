"""Independent check, full-API benchmark, and single-call profile for Q1 MQA.

The saved model inputs came from an extra eager diagnostic forward. Their
content hashes are bound here for the first time, not by the original model
numerical receipt. This operator check does not accept a full model change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
DEFAULT_INPUT = (
    ROOT
    / "experiments/deepseek_v32_echo_official/output/data"
    / "q1_inputs_20261008_01/kernel_inputs_layer_0.pt"
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def environment():
    for name, relative in {
        "DG_JIT_CACHE_DIR": "deep-gemm",
        "TRITON_CACHE_DIR": "triton",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer-workspace",
        "FLASHINFER_CUBIN_DIR": "flashinfer-cubin",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "CUDA_CACHE_PATH": "cuda",
        "TMPDIR": "tmp",
    }.items():
        path = EXPERIMENT / "output/runtime/q1-indexer" / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["DG_JIT_WITH_LINEINFO"] = "1"


def identity(args, torch, deep_gemm):
    package = Path(deep_gemm.__file__).resolve().parent
    sources = [
        Path(__file__),
        ROOT / "operators/deepseek_v32/indexer/echo.py",
        ROOT / "operators/deepseek_v32/indexer/selection.py",
    ]
    dependencies = sorted(
        path
        for path in package.rglob("*")
        if path.is_file() and path.suffix in {".py", ".so", ".h", ".hpp", ".cuh", ".cpp", ".cu"}
    )
    return {
        "inputs_sha256": {str(path.resolve()): digest(path) for path in args.inputs},
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "deep_gemm_installed_sha256": {str(path): digest(path) for path in dependencies},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "num_sms": deep_gemm.get_num_sms(),
        "gpu_uuid": subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.physical_device),
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "physical_device": args.physical_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "lineinfo": os.environ["DG_JIT_WITH_LINEINFO"],
        "boundary": "Borrowed causal bounds; full current-input packing, block table, metadata, "
        "official MQA and causal tail masking. No persistent packed cache. Exact top-k is checked "
        "separately and excluded from MQA timing.",
        "input_provenance": "Extra eager diagnostic forward after restored prefix, outside the "
        "original profile capture. Original publication binds generator/run identity, not these "
        "saved input bytes. This record newly binds their content hashes.",
    }


class Case:
    def __init__(self, torch, data, name):
        self.torch = torch
        self.name = name
        self.q = data["index_q"].cuda()
        self.k = data["index_keys"].cuda()
        self.weights = data["index_weights"].cuda()
        self.scales = data["index_scales"].cuda()
        self.start = int(data["query_start"])
        self.columns = len(self.k)
        self.width = (self.columns + 255) // 256 * 256
        self.starts = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.ends = torch.tensor([self.start + 1], dtype=torch.int32, device="cuda")
        self.saved_indices = data.get("indices")
        self.expect_ties = data.get("expect_ties", False)
        require(tuple(self.q.shape) == (1, 64, 128), "Input is not Q1/H64/D128")
        require(0 <= self.start < self.columns, "Invalid causal endpoint")

    def call(self, method):
        import deep_gemm

        from operators.deepseek_v32.indexer.echo import (
            _paged_q1_logits,
            _resident_tail_mask_kernel,
        )

        if method == "nonpaged":
            # This is deliberately the fixed old API, never echo.logits dispatch.
            scales = self.scales
            if self.columns % 4:
                scales = self.torch.nn.functional.pad(scales, (0, 4 - self.columns % 4))
            result = deep_gemm.fp8_fp4_mqa_logits(
                (self.q, None),
                (self.k, scales[: self.columns]),
                self.weights,
                self.starts,
                self.ends,
                max_seqlen_k=self.width,
            )
        else:
            result = _paged_q1_logits(
                self.q, self.k, self.weights, self.scales, self.ends, self.width
            )
        if self.start + 1 < self.width:
            tail = self.width - self.start
            _resident_tail_mask_kernel()[((tail + 255) // 256,)](
                result,
                self.ends,
                1,
                tail,
                self.start,
                result.stride(0),
                1,
                BLOCK=256,
                num_warps=4,
            )
        return result


def bits(tensor):
    import torch

    return tensor.contiguous().view(torch.int32)


def check_pair(case):
    import torch

    from operators.deepseek_v32.indexer.selection import exact_topk

    reference, candidate = case.call("nonpaged"), case.call("paged")
    torch.cuda.synchronize()
    torch.testing.assert_close(bits(candidate), bits(reference), rtol=0, atol=0)
    ref_values, ref_indices = exact_topk(reference, min(2048, case.columns))
    got_values, got_indices = exact_topk(candidate, min(2048, case.columns))
    torch.testing.assert_close(bits(got_values), bits(ref_values), rtol=0, atol=0)
    torch.testing.assert_close(got_indices, ref_indices, rtol=0, atol=0)
    if case.saved_indices is not None:
        torch.testing.assert_close(ref_indices.cpu(), case.saved_indices, rtol=0, atol=0)
    if case.expect_ties:
        expected = torch.arange(min(2048, case.columns), device="cuda", dtype=torch.int32)
        expected[expected > case.start] = -1
        torch.testing.assert_close(ref_indices[0], expected, rtol=0, atol=0)
    # Independent arithmetic oracle, retaining the pre-existing FP8 WGMMA tolerance.
    raw = torch.einsum("qhd,nd->qhn", case.q.float(), case.k.float()).relu()
    oracle = (raw * case.weights[:, :, None]).sum(1) * case.scales
    oracle[:, case.start + 1 :] = -torch.inf
    torch.testing.assert_close(reference[:, : case.columns], oracle, atol=2e-3, rtol=1e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = case.call("paged")
    for _ in range(8):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(bits(captured), bits(reference), rtol=0, atol=0)
    # Replay must consume new key/scale bytes and the live causal endpoint.
    original_k, original_s = case.k.clone(), case.scales.clone()
    case.k.zero_()
    case.scales.fill_(0.75)
    endpoint_changed = case.start + 1 < case.width
    if endpoint_changed:
        # The wrapper's captured mask branch is fixed by its Python shape and
        # start. A whole-row case has no mask node and must retain its endpoint.
        case.ends.fill_(case.start)
    graph.replay()
    changed = case.call("nonpaged")
    torch.cuda.synchronize()
    torch.testing.assert_close(bits(captured), bits(changed), rtol=0, atol=0)
    case.k.copy_(original_k)
    case.scales.copy_(original_s)
    case.ends.fill_(case.start + 1)
    del graph
    return {
        "case": case.name,
        "logical_k": case.columns,
        "output_stride": case.width,
        "valid_logits_bitwise": True,
        "padded_logits_bitwise": True,
        "exact_topk_values_and_ids_bitwise": True,
        "saved_selection_equal": case.saved_indices is not None,
        "small_index_tie_oracle_checked": case.expect_ties,
        "fp32_oracle_atol": 2e-3,
        "fp32_oracle_rtol": 1e-3,
        "graph_replays_bitwise": 8,
        "graph_current_input_checked": True,
        "graph_causal_endpoint_checked": endpoint_changed,
        "packed_storage_bytes": ((case.columns + 63) // 64) * 64 * 132,
    }


def synthetic_cases(torch):
    cases = []
    for columns, start, tied in [(4096, 4095, True), (4099, 1023, True), (65537, 65536, False)]:
        generator = torch.Generator().manual_seed(912 + columns)
        q = torch.randn(1, 64, 128, generator=generator).to(torch.float8_e4m3fn)
        k = torch.randn(columns, 128, generator=generator).to(torch.float8_e4m3fn)
        if tied:
            k[1:] = k[0]
        data = {
            "index_q": q,
            "index_keys": k,
            "index_weights": torch.rand(1, 64, generator=generator) / 64,
            "index_scales": torch.full((columns,), 0.75),
            "query_start": start,
            "expect_ties": tied,
        }
        cases.append(Case(torch, data, f"synthetic_n{columns}_end{start + 1}_tie{tied}"))
    return cases


def shortened_cases(torch, data, lengths):
    result = []
    for length in lengths:
        require(2 <= length <= len(data["index_keys"]), "Shortened length must be inside input")
        if length == len(data["index_keys"]):
            continue
        shortened = {
            **data,
            "index_keys": data["index_keys"][:length],
            "index_scales": data["index_scales"][:length],
            "query_start": length - 1,
            "indices": None,
        }
        result.append(Case(torch, shortened, f"layer{data['layer']}_shortened_n{length}"))
    return result


def receipt_matches(accepted, actual):
    accepted = dict(accepted)
    actual = dict(actual)
    accepted_inputs = accepted.pop("inputs_sha256")
    actual_inputs = actual.pop("inputs_sha256")
    require(accepted == actual, "Check source/environment identity changed")
    require(
        all(accepted_inputs.get(path) == value for path, value in actual_inputs.items()),
        "Inputs are not an exact-content subset of the independent check",
    )


def benchmark(case, args):
    import torch

    rows = []
    for method in ("nonpaged", "paged"):
        for _ in range(args.warmups):
            case.call(method)
        torch.cuda.synchronize()
        for execution in ("eager", "graph"):
            graph = None
            if execution == "graph":
                allocated_before = torch.cuda.memory_allocated()
                reserved_before = torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = case.call(method)
                torch.cuda.synchronize()
                memory = {
                    "capture_allocated_delta": torch.cuda.memory_allocated() - allocated_before,
                    "capture_reserved_delta": torch.cuda.memory_reserved() - reserved_before,
                    "capture_peak_allocated_delta": torch.cuda.max_memory_allocated()
                    - allocated_before,
                }
                call = graph.replay
            else:
                memory = None
                call = lambda method=method: case.call(method)
            samples, walls = [], []
            for _ in range(args.repeats):
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                begin.record()
                for _ in range(args.iterations):
                    call()
                end.record()
                end.synchronize()
                walls.append((time.perf_counter_ns() - start) / args.iterations / 1e3)
                samples.append(begin.elapsed_time(end) * 1000 / args.iterations)
            rows.append(
                {
                    "case": case.name,
                    "method": method,
                    "execution": execution,
                    "gpu_us_samples": samples,
                    "gpu_us_median": statistics.median(samples),
                    "wall_us_samples": walls,
                    "wall_us_median": statistics.median(walls),
                    "iterations_per_sample": args.iterations,
                    "warmups": args.warmups,
                    "logical_k": case.columns,
                    "output_stride": case.width,
                    "graph_memory": memory,
                    "paged_packed_storage_bytes": ((case.columns + 63) // 64) * 64 * 132
                    if method == "paged"
                    else 0,
                }
            )
            if graph is not None:
                del graph, output
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("check", "bench", "profile"))
    parser.add_argument("--method", choices=("nonpaged", "paged"), default="paged")
    parser.add_argument("--inputs", nargs="+", type=Path, default=[DEFAULT_INPUT])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--physical-device", required=True, type=int)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 65, 256, 1024, 4096, 65537])
    args = parser.parse_args()
    require(
        args.output_dir.resolve().is_relative_to(EXPERIMENT / "output"),
        "Output must stay in experiment",
    )
    require(min(args.warmups, args.repeats, args.iterations) > 0, "Counts must be positive")
    require(
        args.mode == "check" or args.receipt is not None,
        "Bench/profile require independent receipt",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    environment()
    import deep_gemm
    import torch

    require(torch.cuda.get_device_capability() == (9, 0), "SM90 required")
    require(torch.cuda.device_count() == 1, "Set CUDA_VISIBLE_DEVICES to one physical GPU")
    metadata = identity(args, torch, deep_gemm)
    if args.mode != "check":
        accepted = json.loads(args.receipt.read_text())
        require(accepted["accepted"] is True, "Check was not accepted")
        receipt_matches(accepted["identity"], metadata)
        require(set(args.lengths) <= set(accepted["checked_lengths"]), "Lengths were not checked")
    for source in metadata["source_sha256"]:
        destination = args.output_dir / "source" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / source).read_bytes())
    result = {
        "identity": metadata,
        "mode": args.mode,
        "argv": sys.argv,
        "run_id": args.output_dir.name,
    }
    with torch.inference_mode():
        data = [torch.load(path, map_location="cpu", weights_only=True) for path in args.inputs]
        cases = [Case(torch, item, path.name) for item, path in zip(data, args.inputs, strict=True)]
        shortened = shortened_cases(torch, data[0], args.lengths)
        if args.mode == "check":
            result["checks"] = [
                check_pair(case) for case in [*cases, *shortened, *synthetic_cases(torch)]
            ]
            result["checked_lengths"] = args.lengths
            result["accepted"] = True
        elif args.mode == "bench":
            result["receipt"] = {
                "path": str(args.receipt.resolve()),
                "sha256": digest(args.receipt),
            }
            result["samples"] = [
                row for case in [*cases, *shortened] for row in benchmark(case, args)
            ]
        else:
            require(len(cases) == 1, "Profile requires exactly one input file")
            case = cases[0]
            for _ in range(args.warmups):
                case.call(args.method)
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            result_tensor = case.call(args.method)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            result["profile_method"] = args.method
            result["profile_calls"] = 1
            result["result_shape"] = list(result_tensor.shape)
    torch.cuda.synchronize()
    result["allocated_bytes_after"] = torch.cuda.memory_allocated()
    result["reserved_bytes_after"] = torch.cuda.memory_reserved()
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "result.json"),
                "mode": args.mode,
                "accepted": result.get("accepted"),
            }
        )
    )


if __name__ == "__main__":
    main()
