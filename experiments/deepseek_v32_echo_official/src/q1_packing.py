"""Independent page64 adaptation check, clean timing, and one-call profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

from experiments.deepseek_v32_echo_official.src import q1_indexer as common


def reference_pack(k, scales):
    """Fixed pre-optimization tensor-copy implementation."""
    import torch

    columns = len(k)
    full_pages, remainder = divmod(columns, 64)
    pages = full_pages + bool(remainder)
    packed = torch.empty((pages, 64 * 132), dtype=torch.uint8, device=k.device)
    key_bytes = k.view(torch.uint8)
    scale_bytes = scales.view(torch.uint8)
    if full_pages:
        packed[:full_pages, : 64 * 128].copy_(
            key_bytes[: full_pages * 64].reshape(full_pages, 64 * 128)
        )
        packed[:full_pages, 64 * 128 :].copy_(
            scale_bytes[: full_pages * 64 * 4].reshape(full_pages, 64 * 4)
        )
    if remainder:
        packed[full_pages].zero_()
        packed[full_pages, : remainder * 128].copy_(key_bytes[full_pages * 64 :].reshape(-1))
        packed[full_pages, 64 * 128 : 64 * 128 + remainder * 4].copy_(
            scale_bytes[full_pages * 64 * 4 :]
        )
    return packed


def kernel_identity():
    from operators.deepseek_v32.indexer.page64 import _kernel

    result = []
    for cache in _kernel().device_caches.values():
        for compiled in cache[0].values():
            result.append(
                {
                    "hash": compiled.hash,
                    "metadata": json.loads(
                        json.dumps(dict(compiled.metadata._asdict()), default=str)
                    ),
                    "asm_sha256": {
                        name: hashlib.sha256(
                            value if isinstance(value, bytes) else value.encode()
                        ).hexdigest()
                        for name, value in compiled.asm.items()
                    },
                }
            )
    return result


def check_bytes():
    import torch

    from operators.deepseek_v32.indexer.page64 import pack_q1_keys

    results = []
    for columns in (1, 63, 64, 65, 127, 128, 129, 32768, 65536, 65537):
        keys = torch.arange(columns * 128, device="cuda", dtype=torch.int64).to(torch.uint8)
        keys = keys.view(columns, 128).view(torch.float8_e4m3fn)
        patterns = torch.tensor(
            [0, -2147483648, 0x3F800000, 0x7FC01234], dtype=torch.int32, device="cuda"
        )
        scales = patterns.repeat((columns + 3) // 4)[:columns].view(torch.float32)
        expected = reference_pack(keys, scales)
        actual = pack_q1_keys(keys, scales)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = pack_q1_keys(keys, scales)
        for value in (1, 127, 255):
            keys.view(torch.uint8).fill_(value)
            scales.view(torch.int32).fill_(0x7FA01234 + value)
            graph.replay()
            torch.testing.assert_close(captured, reference_pack(keys, scales), atol=0, rtol=0)
        results.append({"tokens": columns, "all_bytes_equal": True, "changed_graph_replays": 3})
    return results


def run_call(case, boundary, method):
    from operators.deepseek_v32.indexer.page64 import pack_q1_keys

    pack = reference_pack if method == "copies" else pack_q1_keys
    if boundary == "packing":
        return pack(case.k, case.scales)
    with patch("operators.deepseek_v32.indexer.echo._pack_q1_keys", pack):
        return case.call("paged")


def bench(case, args):
    import torch

    rows = []
    for boundary in ("packing", "full_mqa"):
        for method in ("copies", "fused"):
            call = lambda boundary=boundary, method=method: run_call(case, boundary, method)
            for _ in range(args.warmups):
                call()
            torch.cuda.synchronize()
            for execution in ("eager", "graph"):
                graph = None
                if execution == "graph":
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = call()
                    measure = graph.replay
                else:
                    measure = call
                gpu, wall = [], []
                for _ in range(args.repeats):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    torch.cuda.synchronize()
                    before = time.perf_counter_ns()
                    start.record()
                    for _ in range(args.iterations):
                        measure()
                    end.record()
                    end.synchronize()
                    gpu.append(start.elapsed_time(end) * 1000 / args.iterations)
                    wall.append((time.perf_counter_ns() - before) / args.iterations / 1000)
                rows.append(
                    {
                        "case": case.name,
                        "tokens": case.columns,
                        "boundary": boundary,
                        "method": method,
                        "execution": execution,
                        "gpu_us_samples": gpu,
                        "gpu_us_median": statistics.median(gpu),
                        "wall_us_samples": wall,
                        "iterations": args.iterations,
                        "warmups": args.warmups,
                    }
                )
                if graph is not None:
                    del graph, output
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), required=True)
    parser.add_argument("--method", choices=("copies", "fused"), default="fused")
    parser.add_argument("--inputs", nargs="+", type=Path, default=[common.DEFAULT_INPUT])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    common.require(min(args.warmups, args.repeats, args.iterations) > 0, "Counts must be positive")
    destination = args.output_dir.resolve()
    allowed = Path("/tmp/cxldsagr-checks") if args.mode == "check" else common.EXPERIMENT / "output"
    common.require(destination.is_relative_to(allowed), f"Output must be under {allowed}")
    common.require(args.mode == "check" or args.receipt is not None, "Matching receipt required")
    destination.mkdir(parents=True, exist_ok=False)
    common.environment()
    import deep_gemm
    import flashinfer
    import flashinfer.triton
    import torch
    import triton

    common.require(torch.cuda.get_device_capability() == (9, 0), "Requires SM90")
    common.require(torch.cuda.device_count() == 1, "Expose one GPU")
    identity = common.identity(args, torch, deep_gemm)
    sources = [
        Path(__file__),
        Path(common.__file__),
        common.ROOT / "operators/deepseek_v32/indexer/page64.py",
    ]
    identity["source_sha256"].update(
        {str(path.relative_to(common.ROOT)): common.digest(path) for path in sources}
    )
    identity["triton"] = triton.__version__
    identity["flashinfer"] = flashinfer.__version__
    identity["flashinfer_sha256"] = {
        str(path): common.digest(path)
        for path in sorted(Path(flashinfer.__file__).resolve().parent.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".so", ".cu", ".cuh", ".h", ".hpp"}
    }
    identity["precision"] = {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32}
    identity["boundary"] = (
        "Fresh page64 packing only or full paged MQA including current packing, page table, schedule, and causal masking; exact top-k excluded. No persistent cache."
    )
    for source in identity["source_sha256"]:
        target = destination / "source" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((common.ROOT / source).read_bytes())
    result = {"identity": identity, "run_id": destination.name, "mode": args.mode}
    if args.mode != "check":
        receipt = json.loads(args.receipt.read_text())
        common.require(receipt["accepted"] is True, "Acceptance required")
        common.receipt_matches(receipt["identity"], identity)
        result["receipt"] = {"path": str(args.receipt), "sha256": common.digest(args.receipt)}
    with torch.inference_mode():
        cases = [
            common.Case(torch, torch.load(path, weights_only=True, map_location="cpu"), path.name)
            for path in args.inputs
        ]
        if args.mode == "check":
            result["bytes"] = check_bytes()
            from operators.deepseek_v32.indexer.page64 import pack_q1_keys

            with patch("operators.deepseek_v32.indexer.echo._pack_q1_keys", pack_q1_keys):
                result["scores_and_selection"] = [common.check_pair(case) for case in cases]
            result["accepted"] = True
        elif args.mode == "bench":
            result["samples"] = [row for case in cases for row in bench(case, args)]
        else:
            common.require(len(cases) == 1, "Profile one input")
            for _ in range(args.warmups):
                run_call(cases[0], "packing", args.method)
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            output = run_call(cases[0], "packing", args.method)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            result["profile"] = {"method": args.method, "output_shape": list(output.shape)}
    result["compiled_kernels"] = kernel_identity()
    if args.mode != "check":
        accepted_kernels = {item["hash"]: item for item in receipt["compiled_kernels"]}
        common.require(
            all(accepted_kernels.get(item["hash"]) == item for item in result["compiled_kernels"]),
            "Compiled packing kernel identity differs from acceptance",
        )
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"output": str(destination / "result.json"), "accepted": result.get("accepted")})
    )


if __name__ == "__main__":
    main()
