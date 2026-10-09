"""Independent correctness and clean timing for the official ECHO source bridge."""

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
INPUT_ROOT = EXPERIMENT / "output/data/q1_inputs_20261008_01"
MISSING = 2**31 - 1
CAP = 64


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    root = EXPERIMENT / "output/runtime/q1-official-prefetch"
    for name, suffix in {
        "DG_JIT_CACHE_DIR": "deep-gemm",
        "TRITON_CACHE_DIR": "triton",
        "TVM_FFI_CACHE_DIR": "tvm",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer-workspace",
        "FLASHINFER_CUBIN_DIR": "flashinfer-cubin",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "CUDA_CACHE_PATH": "cuda",
        "TMPDIR": "tmp",
    }.items():
        target = root / suffix
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    os.environ["DG_JIT_WITH_LINEINFO"] = "1"


def identity(args):
    import deep_gemm
    import flashinfer
    import flashinfer.triton
    import torch
    import triton

    from operators.deepseek_v32.indexer import official_decode

    info = official_decode.build_info()
    files = [
        Path(__file__),
        ROOT / "operators/deepseek_v32/indexer/echo.py",
        ROOT / "operators/deepseek_v32/indexer/page64.py",
        ROOT / "operators/deepseek_v32/indexer/selection.py",
    ]
    info.update(
        inputs_sha256={str(path.relative_to(ROOT)): sha256(path) for path in args.inputs},
        harness_sha256={str(path.relative_to(ROOT)): sha256(path) for path in files},
        reference_deep_gemm_sha256={
            str(path): sha256(path)
            for path in sorted(Path(deep_gemm.__file__).parent.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".so", ".cuh", ".hpp", ".h"}
        },
        flashinfer_sha256={
            str(path): sha256(path)
            for path in sorted(Path(flashinfer.__file__).resolve().parent.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".so", ".cu", ".cuh", ".hpp", ".h"}
        },
        triton=triton.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
        physical_device=args.physical_device,
        gpu_uuid=subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.physical_device),
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        input_provenance="Saved extra eager diagnostic forward; these input bytes are independently bound here.",
        boundaries={
            "prepared": "Official fused kernel plus official causal clean; page packing, block table, and metadata prepared outside call.",
            "full": "Current key/scale packing, block table creation, official metadata, official fused kernel, and official causal clean.",
            "both": "Complete raw logits and actual 64-slot staging prefetch. Caller state reset is outside timing. Exact top-k, staging promotion, and residual recall are excluded.",
        },
    )
    return info


def pack(k, scales):
    from operators.deepseek_v32.indexer.echo import _pack_q1_keys

    return _pack_q1_keys(k, scales)


class Case:
    def __init__(self, data, name):
        import torch

        from operators.deepseek_v32.indexer import official_decode

        self.name = name
        self.q = data["index_q"].cuda().unsqueeze(0)
        self.k = data["index_keys"].cuda()
        self.scales = data["index_scales"].cuda()
        self.weights = data["index_weights"].cuda()
        self.n = len(self.k)
        self.history = int(data["query_start"])
        require(
            self.n == self.history + 1,
            "This bridge requires the final column to be the pending query",
        )
        self.ends = torch.tensor([self.n], device="cuda", dtype=torch.int32)
        self.starts = torch.zeros(1, device="cuda", dtype=torch.int32)
        pages = (self.n + 63) // 64
        self.packed = pack(self.k, self.scales)
        self.blocks = torch.arange(pages, device="cuda", dtype=torch.int32)[None]
        self.schedule = official_decode.metadata(self.ends)
        self.host = torch.full((pages * 64, 576), -13, dtype=torch.bfloat16, pin_memory=True)
        self.host[: self.history].copy_(data["kv"][: self.history])
        self.pool = torch.full((len(self.host) + 1, 576), 1.5, device="cuda", dtype=torch.bfloat16)
        self.pool_before = self.pool.clone()
        self.table = torch.arange(self.n, device="cuda", dtype=torch.int32)[None]
        self.h2d = torch.full((len(self.host) + 1,), MISSING, device="cuda", dtype=torch.int32)
        self.h2d[-1] = 0
        self.initial_h2d = self.h2d.clone()
        self.id_storage = torch.full((CAP + 2,), -71, device="cuda", dtype=torch.int32)
        self.ids = self.id_storage[1:-1][None]
        self.kv_storage = torch.full((CAP + 2, 576), 2.5, device="cuda", dtype=torch.bfloat16)
        self.stage = self.kv_storage[1:-1][None]
        self.counter = torch.zeros(1, device="cuda", dtype=torch.uint32)
        self.threshold = torch.zeros(1, device="cuda", dtype=torch.float32)
        self.saved_indices = data.get("indices")
        self.reset()

    def reset(self):
        self.h2d.copy_(self.initial_h2d)
        self.ids.fill_(-1)
        self.stage.fill_(2.5)
        self.counter.zero_()

    def call(self, boundary):
        import torch

        from operators.deepseek_v32.indexer import official_decode

        if boundary == "full":
            packed = pack(self.k, self.scales)
            blocks = torch.arange(len(packed), device="cuda", dtype=torch.int32)[None]
            schedule = official_decode.metadata(self.ends)
        else:
            packed, blocks, schedule = self.packed, self.blocks, self.schedule
        return official_decode.logits(
            self.q,
            packed,
            self.weights,
            self.ends,
            blocks,
            schedule,
            self.n,
            self.table,
            self.pool,
            self.host,
            self.ids,
            self.stage,
            self.h2d,
            self.counter,
            self.threshold,
        )

    def reference(self):
        import deep_gemm
        import torch

        scales = torch.nn.functional.pad(self.scales, (0, (-self.n) % 4))
        result = deep_gemm.fp8_fp4_mqa_logits(
            (self.q[0], None),
            (self.k, scales[: self.n]),
            self.weights,
            self.starts,
            self.ends,
            max_seqlen_k=self.n,
        )
        mask = torch.arange(self.n, device="cuda")[None] >= self.ends[:, None]
        return result.masked_fill(mask, -torch.inf)

    def policy(self, name, reference):
        import torch

        self.initial_h2d.fill_(MISSING)
        self.initial_h2d[-1] = 0
        if name == "warm":
            self.initial_h2d[: self.history] = torch.arange(
                1, self.history + 1, device="cuda", dtype=torch.int32
            )
        elif name == "partial":
            self.initial_h2d[: self.history : 2] = torch.arange(
                1, self.history + 1, 2, device="cuda", dtype=torch.int32
            )
        if name == "empty":
            self.threshold.fill_(torch.inf)
        elif name == "small":
            self.threshold.copy_(torch.topk(reference[:, : self.history], 17).values[:, -1])
        elif name == "learned":
            self.threshold.copy_(torch.topk(reference, min(2048, self.n)).values[:, -1])
        else:
            self.threshold.zero_()
        self.reset()


def validate(case, output, reference, *, saved=False):
    import torch

    from operators.deepseek_v32.indexer.selection import exact_topk

    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.contiguous().view(torch.int32),
        reference.contiguous().view(torch.int32),
        atol=0,
        rtol=0,
    )
    backing = output.as_strided((1, output.stride(0)), output.stride())
    require(bool(torch.isneginf(backing[:, case.n :]).all()), "Unmasked physical score tail")
    values, indices = exact_topk(output, min(2048, case.n))
    ref_values, ref_indices = exact_topk(reference, min(2048, case.n))
    torch.testing.assert_close(
        values.view(torch.int32), ref_values.view(torch.int32), atol=0, rtol=0
    )
    torch.testing.assert_close(indices, ref_indices, atol=0, rtol=0)
    if saved and case.saved_indices is not None:
        torch.testing.assert_close(indices.cpu(), case.saved_indices, atol=0, rtol=0)
    live_history = int(case.ends.item()) - 1
    eligible = reference[0, :live_history] > case.threshold[0]
    mapped_hosts = case.table[0, :live_history].long()
    eligible &= case.initial_h2d[mapped_hosts] == MISSING
    eligible_ids = mapped_hosts[eligible].cpu()
    expected = min(CAP, len(eligible_ids))
    actual_ids = case.ids[0].cpu()
    ids = actual_ids[:expected].long()
    require(len(ids.unique()) == expected, "Duplicate official staging host IDs")
    require(
        bool(torch.isin(ids, eligible_ids).all()),
        "Staged host ID fails official strict threshold or residency",
    )
    require(
        bool((actual_ids[expected:] == -1).all()), "Unexpected writes beyond granted staging IDs"
    )
    require(
        expected <= int(case.counter.item()) <= len(eligible_ids),
        "Official attempt counter out of bounds",
    )
    expected_h2d = case.initial_h2d.clone()
    expected_h2d[ids.cuda()] = torch.arange(
        len(case.pool), len(case.pool) + expected, device="cuda", dtype=torch.int32
    )
    torch.testing.assert_close(case.h2d, expected_h2d, atol=0, rtol=0)
    torch.testing.assert_close(
        case.stage[0, :expected].cpu().view(torch.int16),
        case.host[ids].view(torch.int16),
        atol=0,
        rtol=0,
    )
    require(bool((case.stage[0, expected:] == 2.5).all()), "Unexpected unused staging KV writes")
    require(bool((case.id_storage[[0, -1]] == -71).all()), "Staging ID canary overwritten")
    require(bool((case.kv_storage[[0, -1]] == 2.5).all()), "Staging KV canary overwritten")
    torch.testing.assert_close(case.pool, case.pool_before, atol=0, rtol=0)
    require(int(case.h2d[-1]) == 0, "Host sentinel changed")
    return {
        "eligible_misses": len(eligible_ids),
        "staged": expected,
        "attempted": int(case.counter.item()),
        "score_bits": True,
        "topk_bits": True,
        "staging_and_mappings": True,
        "canaries": True,
    }


def check_case(case):
    import torch

    reference = case.reference()
    rows = []
    for boundary in ("prepared", "full"):
        for policy in ("zero", "small", "learned", "empty", "warm", "partial"):
            case.policy(policy, reference)
            output = case.call(boundary)
            rows.append(
                {
                    "case": case.name,
                    "boundary": boundary,
                    "policy": policy,
                    **validate(case, output, reference, saved=True),
                }
            )
    case.policy("empty", reference)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(1_000_000)
        case.threshold.zero_()
        case.reset()
        output = case.call("full")
        finished = side.record_event()
    finished.synchronize()
    rows.append(
        {
            "case": case.name,
            "boundary": "full_nondefault_stream",
            **validate(case, output, reference, saved=True),
        }
    )
    case.policy("zero", reference)
    # Warm all source code before capture; caller-owned state is outside graph.
    case.call("full")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = case.call("full")
    for replay in range(3):
        case.reset()
        graph.replay()
        rows.append(
            {
                "case": case.name,
                "boundary": "full_graph",
                "replay": replay,
                **validate(case, output, reference, saved=True),
            }
        )
    # Same addresses, different current tensor bytes, host records, and length.
    case.q.copy_(case.q.float().flip(-1).to(case.q.dtype))
    case.k.copy_(case.k.float().roll(1, 0).to(case.k.dtype))
    case.scales.mul_(1.03125)
    case.weights.mul_(0.875)
    case.host[: case.history].add_(0.5)
    case.ends.fill_(max(1, case.n - 5))
    reference = case.reference()
    case.threshold.copy_(torch.topk(reference, min(2048, case.n)).values[:, -1])
    for replay in range(3):
        case.reset()
        graph.replay()
        rows.append(
            {
                "case": case.name,
                "boundary": "full_graph_changed",
                "replay": replay,
                **validate(case, output, reference),
            }
        )
    del graph
    return rows


def benchmark(case, args):
    import torch

    reference = case.reference()
    rows = []
    for policy in ("zero", "warm"):
        case.policy(policy, reference)
        for boundary in ("prepared", "full"):
            for _ in range(args.warmups):
                case.reset()
                case.call(boundary)
            torch.cuda.synchronize()
            for execution in ("eager", "graph"):
                graph = None
                if execution == "graph":
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = case.call(boundary)
                    call = graph.replay
                else:
                    call = lambda boundary=boundary: case.call(boundary)
                events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                gpu, wall, attempts = [], [], []
                for _ in range(args.repeats):
                    case.reset()
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    events[0].record()
                    call()
                    events[1].record()
                    events[1].synchronize()
                    wall.append((time.perf_counter_ns() - start) / 1000)
                    gpu.append(events[0].elapsed_time(events[1]) * 1000)
                    attempts.append(int(case.counter.item()))
                row = {
                    "case": case.name,
                    "policy": policy,
                    "boundary": boundary,
                    "execution": execution,
                    "gpu_us_samples": gpu,
                    "gpu_us_median": statistics.median(gpu),
                    "wall_us_samples": wall,
                    "wall_us_median": statistics.median(wall),
                    "reservation_attempts_samples": attempts,
                    "prefetched_records_samples": [min(value, CAP) for value in attempts],
                    "h2d_bytes_samples": [min(value, CAP) * 1152 for value in attempts],
                    "calls_per_sample": 1,
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                    "packed_bytes": case.packed.numel(),
                    "staging_bytes": CAP * (576 * 2 + 4),
                    "official_metadata_bytes": case.schedule.numel() * 4,
                }
                rows.append(row)
                print(
                    json.dumps(
                        {key: value for key, value in row.items() if not key.endswith("samples")}
                    ),
                    flush=True,
                )
                if graph is not None:
                    del graph, output
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "bench"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[INPUT_ROOT / f"kernel_inputs_layer_{layer}.pt" for layer in range(3)],
    )
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id, "run-id must be one path component")
    require(args.physical_device != 7, "GPU7 is excluded from this task")
    require(
        args.mode == "check" or args.receipt is not None,
        "Clean timing requires a matching independent receipt",
    )
    require(min(args.warmups, args.repeats) > 0, "Warmup and repetition counts must be positive")
    environment()
    import torch

    require(torch.cuda.device_count() == 1, "Select one physical GPU with CUDA_VISIBLE_DEVICES")
    require(torch.cuda.get_device_capability() == (9, 0), "SM90 required")
    before = identity(args)
    if args.mode == "bench":
        receipt = json.loads(args.receipt.read_text())
        require(
            receipt["accepted"] is True and receipt["identity"] == before,
            "Acceptance identity mismatch",
        )
    output_dir = (
        Path("/tmp/cxldsagr-checks/q1-official-prefetch")
        if args.mode == "check"
        else EXPERIMENT / "output/data"
    ) / args.run_id
    require(not output_dir.exists(), "Run ID already exists")
    result = {
        "run_id": args.run_id,
        "mode": args.mode,
        "argv": sys.argv,
        "identity": before,
        "rows": [],
    }
    with torch.inference_mode():
        for path in args.inputs:
            case = Case(torch.load(path, map_location="cpu", weights_only=True), path.stem)
            result["rows"].extend(
                check_case(case) if args.mode == "check" else benchmark(case, args)
            )
            print(f"Completed {args.mode}: {case.name}", flush=True)
            del case
    torch.cuda.synchronize()
    require(before == identity(args), "Source or input identity changed during this run")
    from operators.deepseek_v32.indexer.official_decode import native_info

    native = native_info(before["num_sms"])
    result["native"] = native
    if args.mode == "check":
        result["accepted"] = True
    else:
        require(native == receipt["native"], "Native build differs from accepted check")
        result["receipt"] = {"path": str(args.receipt), "sha256": sha256(args.receipt)}
    output_dir.mkdir(parents=True)
    for field in ("source_sha256", "harness_sha256"):
        for source in before[field]:
            path = ROOT / source
            target = output_dir / "source" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(output_dir / "result.json", flush=True)


if __name__ == "__main__":
    main()
