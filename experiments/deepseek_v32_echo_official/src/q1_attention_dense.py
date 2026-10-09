"""Check and time page64 FlashMLA dense decode on exact sparse selections.

Diagnostic candidate only. The wrapper compacts valid IDs on GPU, preserving
order and duplicates, gathers selected BF16 records, builds fresh scheduling
metadata, and calls the existing BF16 dense decode backend. All work is timed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from functools import cache
from pathlib import Path

from experiments.deepseek_v32_echo_official.src.q1_indexer import DEFAULT_INPUT, digest, require

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]


@cache
def kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def compact(Ids, Compact, Count, Length, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
        slot = tl.arange(0, B)
        index = tl.load(Ids + slot, slot < S, -1).to(tl.int64)
        valid = (slot < S) & (index >= 0) & (index < N)
        rank = tl.cumsum(valid.to(tl.int32)) - 1
        tl.store(Compact + rank, index, valid)
        count = tl.sum(valid.to(tl.int32), 0)
        tl.store(Count, count)
        # An empty row attends to one zero record. It cannot read source KV.
        tl.store(Length, tl.maximum(count, 1))

    @triton.jit
    def gather(KV, Compact, Count, Packed, K0: tl.constexpr, K1: tl.constexpr, D: tl.constexpr):
        token = tl.program_id(0)
        dim = tl.arange(0, D)
        count = tl.load(Count)
        index = tl.load(Compact + token, token < count, 0).to(tl.int64)
        value = tl.load(KV + index * K0 + dim * K1, (token < count) & (dim < 576), 0)
        tl.store(Packed + token * 576 + dim, value, dim < 576)

    return compact, gather


def dense_candidate(q, kv, indices, scale):
    import flash_mla
    import torch
    import triton

    from operators.deepseek_v32.attention._validation import _validate

    _validate(q, kv, indices, scale, 512)
    require(tuple(q.shape[:1]) == (1,) and q.shape[2] == 576, "Only Q1/D576 is supported")
    require(q.dtype == torch.bfloat16 and q.is_cuda, "BF16 CUDA input required")
    require(q.shape[1] in (64, 128), "Expected H64 or H128")
    slots = indices.shape[1]
    if slots == 0:
        return torch.zeros((1, q.shape[1], 512), dtype=q.dtype, device=q.device)
    require(slots <= 16384, "Diagnostic compaction is bounded to 16384 slots")
    q, indices = q.contiguous(), indices.contiguous()
    if q.data_ptr() % 16:
        q = q.clone()
    pages = (slots + 63) // 64
    packed = torch.empty((pages, 64, 1, 576), dtype=q.dtype, device=q.device)
    compacted = torch.empty((slots,), dtype=torch.int64, device=q.device)
    count = torch.empty((1,), dtype=torch.int32, device=q.device)
    length = torch.empty_like(count)
    compact_kernel, gather_kernel = kernels()
    compact_kernel[(1,)](
        indices,
        compacted,
        count,
        length,
        len(kv),
        slots,
        triton.next_power_of_2(slots),
        num_warps=4,
    )
    gather_kernel[(pages * 64,)](
        kv,
        compacted,
        count,
        packed,
        kv.stride(0),
        kv.stride(1),
        1024,
        num_warps=4,
    )
    block_table = torch.arange(pages, dtype=torch.int32, device=q.device).unsqueeze(0)
    metadata, _ = flash_mla.get_mla_metadata()
    result, _ = flash_mla.flash_mla_with_kvcache(
        q.unsqueeze(0),
        packed,
        block_table,
        length,
        512,
        metadata,
        softmax_scale=scale,
        causal=False,
        is_fp8_kvcache=False,
    )
    return result.squeeze(0)


def signature(args, torch):
    native = importlib.import_module("flash_mla.cuda")
    interface = importlib.import_module("flash_mla.flash_mla_interface")
    paths = [
        Path(__file__),
        Path(native.__file__).resolve(),
        Path(interface.__file__).resolve(),
        ROOT / "experiments/deepseek_v32_echo_official/src/q1_indexer.py",
        ROOT / "operators/deepseek_v32/attention/device_only/mla.py",
        ROOT / "operators/deepseek_v32/attention/reference/torch.py",
        ROOT / "operators/deepseek_v32/attention/_validation.py",
    ]
    return {
        "input_sha256": digest(args.input),
        "input": str(args.input.resolve()),
        "source_sha256": {str(path): digest(path) for path in paths},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "physical_device": args.physical_device,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "input_boundary": "Saved extra eager diagnostic forward; newly bound input hash, "
        "not original model receipt or measured graph capture input binding.",
    }


def metrics(actual, reference):
    import torch

    delta = (actual.float() - reference.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "bitwise_equal": torch.equal(actual.view(torch.int16), reference.view(torch.int16)),
    }


def validate(q, kv, indices, scale):
    import torch

    from operators.deepseek_v32.attention.device_only.mla import sparse_mla
    from operators.deepseek_v32.attention.reference.torch import reference_sparse_mla

    selection_cases = {
        "real": indices,
        "duplicates_invalid_i64": torch.tensor(
            [[-1, 17, 17, 0, len(kv), 2**40, -(2**40), 3, 17]], dtype=torch.int64, device=q.device
        ),
        "empty_valid_set": torch.tensor([[-1, 2**40, len(kv)]], dtype=torch.int64, device=q.device),
        "empty_slots": torch.empty((1, 0), dtype=torch.int32, device=q.device),
    }
    results = []
    for name, selected in selection_cases.items():
        reference = reference_sparse_mla(q, kv, selected, scale)
        original = sparse_mla(q, kv, selected, scale)
        candidate = dense_candidate(q, kv, selected, scale)
        torch.cuda.synchronize()
        torch.testing.assert_close(original, reference, atol=4e-3, rtol=2e-2)
        torch.testing.assert_close(candidate, reference, atol=4e-3, rtol=2e-2)
        torch.testing.assert_close(candidate, original, atol=4e-3, rtol=2e-2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = dense_candidate(q, kv, selected, scale)
        for _ in range(4):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, candidate, atol=0, rtol=0)
        if selected.numel():
            original_ids = selected.clone()
            selected.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            require(output.eq(0).all().item(), "All-invalid graph replay must return zero")
            selected.copy_(original_ids)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, candidate, atol=0, rtol=0)
        results.append(
            {
                "case": name,
                "candidate_vs_oracle": metrics(candidate, reference),
                "candidate_vs_sparse": metrics(candidate, original),
                "fixed_graph_replays": 4,
                "valid_count_change_graph_checked": bool(selected.numel()),
            }
        )
    return results


def measure(q, kv, indices, scale, args):
    import torch

    from operators.deepseek_v32.attention.device_only.mla import sparse_mla

    rows = []
    for method, function in (("sparse_prefill", sparse_mla), ("dense_compact", dense_candidate)):
        for _ in range(args.warmups):
            function(q, kv, indices, scale)
        torch.cuda.synchronize()
        for execution in ("eager", "graph"):
            call = lambda function=function: function(q, kv, indices, scale)
            graph = None
            if execution == "graph":
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = call()
                memory = {
                    "peak_allocated_delta": torch.cuda.max_memory_allocated() - before,
                    "reserved_delta": torch.cuda.memory_reserved() - reserved,
                }
                call = graph.replay
            else:
                memory = None
            samples, wall = [], []
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
                samples.append(begin.elapsed_time(end) * 1000 / args.iterations)
                wall.append((time.perf_counter_ns() - start) / 1000 / args.iterations)
            rows.append(
                {
                    "method": method,
                    "execution": execution,
                    "gpu_us": samples,
                    "gpu_us_median": statistics.median(samples),
                    "wall_us": wall,
                    "wall_us_median": statistics.median(wall),
                    "memory": memory,
                }
            )
            if graph is not None:
                del graph, output
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("check", "bench"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    require(
        args.output_dir.resolve().is_relative_to(EXPERIMENT / "output"), "Use experiment output"
    )
    require(args.mode == "check" or args.receipt is not None, "Bench requires check receipt")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key, name in (
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("TMPDIR", "tmp"),
    ):
        path = EXPERIMENT / "output/runtime/q1-attention-dense" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import torch

    require(
        torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0),
        "One SM90 GPU required",
    )
    identity = signature(args, torch)
    if args.mode == "bench":
        receipt = json.loads(args.receipt.read_text())
        require(receipt["accepted"] is True and receipt["identity"] == identity, "Receipt mismatch")
    data = torch.load(args.input, map_location="cpu", weights_only=True)
    result = {
        "identity": identity,
        "run_id": args.output_dir.name,
        "argv": sys.argv,
        "mode": args.mode,
    }
    (args.output_dir / "q1_attention_dense.py").write_bytes(Path(__file__).read_bytes())
    with torch.inference_mode():
        q, kv, indices = data["q"].cuda(), data["kv"].cuda(), data["indices"].cuda()
        if args.mode == "check":
            result["checks"] = validate(q, kv, indices, data["attention_scale"])
            result["accepted"] = True
        else:
            result["receipt_sha256"] = digest(args.receipt)
            result["samples"] = measure(q, kv, indices, data["attention_scale"], args)
            result["boundary"] = (
                "Full wrapper: compaction preserving duplicates, BF16 gather/page padding, "
                "block table, fresh GPU scheduler metadata, dense decode and combine; "
                "no CPU count read."
            )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output_dir / "result.json"), "accepted": result.get("accepted")}
        )
    )


if __name__ == "__main__":
    main()
