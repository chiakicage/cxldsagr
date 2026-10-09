"""Independent Q1 sparse split-KV prototype, including all wrapper work.

Each split repeats the original query and consumes its original selected slots.
FlashMLA returns BF16 partial outputs and natural-log LSE (+inf for an empty
shard). The merge keeps duplicate IDs and excludes empty shards explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from functools import cache, partial
from pathlib import Path

from experiments.deepseek_v32_echo_official.src.q1_attention_dense import (
    EXPERIMENT,
    ROOT,
    dense_candidate,
    metrics,
    signature,
)
from experiments.deepseek_v32_echo_official.src.q1_indexer import DEFAULT_INPUT, digest, require


@cache
def kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def combine(V, Lse, Out, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
        head = tl.program_id(0)
        split = tl.arange(0, S)
        dim = tl.arange(0, D)
        lse = tl.load(Lse + split * H + head)
        # Official sparse prefill uses +inf as the empty-shard sentinel.
        present = lse != float("inf")
        lse = tl.where(present, lse, -float("inf"))
        peak = tl.max(lse, 0)
        peak = tl.where(peak == -float("inf"), 0.0, peak)
        weight = tl.exp(lse - peak)
        denominator = tl.sum(weight, 0)
        value = tl.load(V + (split[:, None] * H + head) * D + dim[None, :]).to(tl.float32)
        value = tl.where(present[:, None], value, 0.0)
        result = tl.sum(value * weight[:, None], 0) / tl.maximum(denominator, 1.0)
        tl.store(Out + head * D + dim, result)

    @triton.jit
    def natural_to_base2(Lse, Scores, N: tl.constexpr, B: tl.constexpr):
        index = tl.arange(0, B)
        lse = tl.load(Lse + index, index < N, float("inf"))
        score = tl.where(lse == float("inf"), -float("inf"), lse * 1.4426950408889634)
        tl.store(Scores + index, score, index < N)

    return combine, natural_to_base2


def split_candidate(q, kv, indices, scale, *, splits, merge="triton", return_partials=False):
    import flash_mla
    import torch
    import triton
    from torch.nn import functional as F

    from operators.deepseek_v32.attention._validation import _validate

    _validate(q, kv, indices, scale, 512)
    require(q.shape == (1, 128, 576), "Prototype requires Q1/H128/D576")
    require(q.dtype == torch.bfloat16 and q.is_cuda, "BF16 CUDA inputs required")
    require(splits in (4, 8, 16), "Use 4, 8 or 16 independent sparse queries")
    slots = indices.shape[1]
    if not slots or not len(kv):
        return torch.zeros((1, 128, 512), dtype=q.dtype, device=q.device)
    q, kv = q.contiguous(), kv.contiguous()
    if kv.data_ptr() % 16:
        kv = kv.clone()
    if indices.dtype == torch.int64:
        valid = (indices >= 0) & (indices < len(kv))
        indices = indices.masked_fill(~valid, -1).to(torch.int32)
    indices = indices.contiguous()
    shard = math.ceil(slots / (splits * 128)) * 128
    if shard * splits != slots:
        indices = F.pad(indices, (0, shard * splits - slots), value=-1)
    repeated_q = q.repeat(splits, 1, 1)
    output, _, lse = flash_mla.flash_mla_sparse_fwd(
        repeated_q, kv.unsqueeze(1), indices.view(splits, 1, shard), float(scale), 512
    )
    combine, convert = kernels()
    if merge == "triton":
        result = torch.empty((1, 128, 512), dtype=q.dtype, device=q.device)
        combine[(128,)](output, lse, result, 128, splits, 512, num_warps=4)
    elif merge == "flashinfer":
        import flashinfer

        scores = torch.empty_like(lse)
        convert[(1,)](lse, scores, lse.numel(), triton.next_power_of_2(lse.numel()))
        result, _ = flashinfer.merge_states(output.unsqueeze(0), scores.unsqueeze(0))
    else:
        raise ValueError(merge)
    return (result, lse, indices.view(splits, shard)) if return_partials else result


def validate(q, kv, indices, scale, args):
    import torch

    from operators.deepseek_v32.attention.device_only.mla import sparse_mla
    from operators.deepseek_v32.attention.reference.torch import reference_sparse_mla

    only_one_shard = torch.full_like(indices, -1)
    only_one_shard[:, :128] = indices[:, :128]
    cases = {
        "real": indices,
        "only_first_shard_valid": only_one_shard,
        "duplicates_invalid_i64": torch.tensor(
            [[-1, 17, 17, 0, len(kv), 2**40, -(2**40), 3, 17]],
            dtype=torch.int64,
            device=q.device,
        ),
        "all_invalid": torch.full_like(indices, -1),
        "empty_slots": torch.empty((1, 0), dtype=torch.int32, device=q.device),
    }
    checks = []
    for case, selected in cases.items():
        oracle = reference_sparse_mla(q, kv, selected, scale)
        original = sparse_mla(q, kv, selected, scale)
        torch.testing.assert_close(original, oracle, atol=4e-3, rtol=2e-2)
        for splits in args.splits:
            for merge in args.merges:
                call = partial(split_candidate, splits=splits, merge=merge)
                candidate = call(q, kv, selected, scale)
                torch.testing.assert_close(candidate, oracle, atol=4e-3, rtol=2e-2)
                torch.testing.assert_close(candidate, original, atol=4e-3, rtol=2e-2)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = call(q, kv, selected, scale)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(replayed, candidate, atol=0, rtol=0)
                if selected.numel():
                    saved = selected.clone()
                    selected.fill_(-1)
                    graph.replay()
                    torch.cuda.synchronize()
                    require(replayed.eq(0).all().item(), "Empty shards must merge to zero")
                    selected.copy_(saved)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(replayed, candidate, atol=0, rtol=0)
                checks.append(
                    {
                        "case": case,
                        "splits": splits,
                        "merge": merge,
                        "vs_oracle": metrics(candidate, oracle),
                        "vs_sparse": metrics(candidate, original),
                    }
                )
                del graph, replayed
    # Verify the LSE scale numerically, independently of the merge implementation.
    _, lse, shards = split_candidate(
        q, kv, indices, scale, splits=args.splits[0], return_partials=True
    )
    shard_ids = shards[0]
    shard_ids = shard_ids[(shard_ids >= 0) & (shard_ids < len(kv))].long()
    expected = torch.logsumexp(q[0].float() @ kv[shard_ids].float().T * scale, dim=-1)
    torch.testing.assert_close(lse[0], expected, atol=4e-3, rtol=2e-2)
    return {"cases": checks, "lse_units": "natural_log", "empty_shard_lse": "+inf"}


def measure(q, kv, indices, scale, args):
    import torch

    from operators.deepseek_v32.attention.device_only.mla import sparse_mla

    methods = {"sparse_prefill": sparse_mla, "dense_compact": dense_candidate}
    methods.update(
        (f"split{splits}_{merge}", partial(split_candidate, splits=splits, merge=merge))
        for splits in args.splits
        for merge in args.merges
    )
    rows = []
    # Same CUDA-event and synchronized wall-clock boundaries as q1_attention_dense.
    for method, function in methods.items():
        for _ in range(args.warmups):
            function(q, kv, indices, scale)
        torch.cuda.synchronize()
        for execution in ("eager", "graph"):
            call = partial(function, q, kv, indices, scale)
            graph = None
            if execution == "graph":
                before = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
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
            samples, walls = [], []
            for _ in range(args.repeats):
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                begin.record()
                for _ in range(args.iterations):
                    call()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000 / args.iterations)
                walls.append((time.perf_counter_ns() - started) / 1000 / args.iterations)
            rows.append(
                {
                    "method": method,
                    "execution": execution,
                    "gpu_us": samples,
                    "gpu_us_median": statistics.median(samples),
                    "wall_us": walls,
                    "wall_us_median": statistics.median(walls),
                    "memory": memory,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
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
    parser.add_argument("--splits", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--merges", nargs="+", choices=("triton", "flashinfer"), default=["triton"])
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
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
        ("TMPDIR", "tmp"),
    ):
        path = EXPERIMENT / "output/runtime/q1-attention-split" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    require(
        torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0),
        "One SM90 GPU required",
    )
    identity = signature(args, torch)
    identity.update(splits=args.splits, merges=args.merges)
    for path in (
        Path(__file__),
        ROOT / "operators/deepseek_v32/attention/_config.py",
        ROOT / "3rdparty/FlashMLA/csrc/kernels/sm90/prefill/sparse/phase1.cuh",
    ):
        identity["source_sha256"][str(path)] = digest(path)
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
    (args.output_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    with torch.inference_mode():
        q, kv, indices = data["q"].cuda(), data["kv"].cuda(), data["indices"].cuda()
        result["shapes"] = {
            "q": list(q.shape),
            "kv": list(kv.shape),
            "indices": list(indices.shape),
        }
        if args.mode == "check":
            result["checks"] = validate(q, kv, indices, data["attention_scale"], args)
            result["accepted"] = True
        else:
            result["receipt_sha256"] = digest(args.receipt)
            result["samples"] = measure(q, kv, indices, data["attention_scale"], args)
            result["boundary"] = (
                "Full wrapper: query repeat, index conversion/padding, sparse core, LSE conversion if needed, partial merge."
            )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output_dir / "result.json"), "accepted": result.get("accepted")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
