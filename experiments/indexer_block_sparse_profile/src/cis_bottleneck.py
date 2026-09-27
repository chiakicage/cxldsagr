"""Measure the existing CIS callable with eager submission and CUDA Graph replay.

Synthetic BF16 inputs reproduce the model's row-strided V view. This isolates
submission overhead; it is not a model forward benchmark or a fused replacement.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from models.nosa.scoring import cis_scores


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--counters", action="store_true", help="One CUDA profiler API range")
    args = parser.parse_args(argv)
    if min(args.warmup, args.repeats, args.batch) <= 0:
        parser.error("warmup, repeats and batch must be positive")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This experiment requires SM90")
    torch.manual_seed(42)
    storage = torch.randn(1024, 4608, device=args.device, dtype=torch.bfloat16)
    values = storage[:, 4352:].view(1024, 2, 128)
    delta = torch.randn(2, 256, device=args.device, dtype=torch.bfloat16) / 16
    scale = torch.randn(2, device=args.device, dtype=torch.bfloat16)

    def call():
        return cis_scores(values, delta, scale)

    for _ in range(args.warmup):
        reference = call()
    torch.cuda.synchronize()
    if args.counters:
        torch.cuda.cudart().cudaProfilerStart()
        result = call()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        return

    # A batch in one graph amortizes the host graph-launch cost. All nodes still
    # call the original six-kernel CIS implementation, without fusion.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.batch):
            graph_result = call()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_result, reference, rtol=0, atol=0)
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    end.record()
    end.synchronize()
    samples = {"eager": [], "graph": []}
    for repeat in range(args.repeats):
        # Alternate order to reduce clock/thermal ordering bias.
        for mode in ("eager", "graph") if repeat % 2 == 0 else ("graph", "eager"):
            torch.cuda.synchronize()
            started = time.perf_counter()
            begin.record()
            if mode == "graph":
                graph.replay()
            else:
                for _ in range(args.batch):
                    result = call()
            end.record()
            end.synchronize()
            samples[mode].append(
                {
                    "wall_us_per_call": (time.perf_counter() - started) * 1e6 / args.batch,
                    "cuda_us_per_call": begin.elapsed_time(end) * 1000 / args.batch,
                }
            )
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    props = torch.cuda.get_device_properties(args.device)
    paths = [Path(__file__), Path("models/nosa/scoring.py")]
    report = {
        "run_id": args.run_id,
        "gpu": {
            "name": props.name,
            "uuid": str(props.uuid),
            "sm_count": props.multi_processor_count,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "input": {
            "shape": list(values.shape),
            "stride": list(values.stride()),
            "dtype": str(values.dtype),
            "seed": 42,
            "bias": False,
        },
        "scope": "Original CIS, synthetic same-input hot-cache replay; allocation included in eager, graph pool reused; no model speedup claim",
        "graph_matches_eager_exactly": True,
        "matrix_flops": 2 * 1024 * 256 * 2,
        "projection_minimum_bytes": (1024 * 256 + 256 * 2 + 1024 * 2) * 2,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "samples": samples,
        "medians": {
            mode: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
            for mode, rows in samples.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["medians"], indent=2))


if __name__ == "__main__":
    main()
