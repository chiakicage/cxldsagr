"""Check and profile the current FIFO preparation on saved H64K/A1 state."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

from evaluation.local_native import collect_local_native_artifacts
from operators.deepseek_v32.indexer import cache_ops, echo

ROOT = Path(__file__).resolve().parents[3]
MISSING = 2**31 - 1
DEFAULT_INPUT = Path(
    "/tmp/cxldsagr-checks/deepseek_v32_mfu/data/"
    "deepseek_h64k_a1_cub_20261008_01_h65536_a1_check/"
    "echo_default_graph_prefetch_evidence.pt"
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def exact(left, right, label):
    require(left.shape == right.shape and left.dtype == right.dtype, label + " geometry")
    require(
        torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)),
        label + " bits",
    )


def source_identity():
    build = echo.build_info()
    paths = {ROOT / name for name in (*build["source_sha256"], *build["shared_header_sha256"])}
    paths.update((Path(__file__), ROOT / "evaluation/local_native.py"))
    # CUB is a header library selected by the current CUDA compiler.
    cccl = Path("/usr/local/cuda/include/cccl")
    paths.update(path for path in cccl.rglob("*") if path.is_file())
    return {
        "echo_build": build,
        "files": {str(path.resolve()): digest(path) for path in sorted(paths)},
        "nvcc": subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True),
    }


def load_input(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        payload["single_session"] and payload["residency"] == "cold", "Expected cold sole session"
    )
    require(len(payload["layers"]) == 3, "Expected three layer states")
    layers = payload["layers"]
    first = layers[0]
    initial = first["stages"]["initial"]
    for layer in layers:
        require(
            (layer["H"], layer["A"], layer["slots"]) == (65536, 1, 65600), "Input shape changed"
        )
        exact(layer["stages"]["initial"]["priority"], initial["priority"], "Layer priorities")
        exact(layer["stages"]["initial"]["clock"], initial["clock"], "Layer clocks")
    require(initial["priority"].dtype == torch.int64, "Expected int64 priorities")
    require(int(initial["clock"][0]) == 256, "Expected current clock 256")
    return first, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "profile"), required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--warmups", type=int, default=20)
    args = parser.parse_args()
    require(args.warmups > 0, "Warmups must be positive")
    require(torch.cuda.get_device_capability() == (9, 0), "SM90 required")
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    for path in source["files"]:
        original = Path(path)
        target = args.output / "source" / original.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    layer, initial = load_input(args.input)
    slots = layer["slots"]
    priority = initial["priority"].cuda()
    buffers = {
        "free_slots": torch.empty(slots, dtype=torch.int32, device="cuda"),
        "allocation_log": torch.empty(slots + 1, dtype=torch.int64, device="cuda"),
        "counter": torch.empty(1, dtype=torch.uint32, device="cuda"),
        "prefetch_stats": torch.empty(3, dtype=torch.int64, device="cuda"),
        "keys": torch.empty(slots, dtype=torch.int64, device="cuda"),
    }

    def call():
        return cache_ops.prepare_prefetch(
            priority,
            **buffers,
            timestamp=int(initial["clock"][0]),
            host_capacity=layer["host_arena_tokens"],
            query_count=1,
        )

    def dirty():
        for value in buffers.values():
            value.fill_(123)

    expected_slots = (torch.argsort(initial["priority"][1:], stable=True) + 1).int()

    def check(token):
        exact(priority.cpu(), initial["priority"], "Unchanged priority")
        exact(buffers["free_slots"].cpu(), expected_slots, "Complete stable FIFO")
        exact(
            buffers["allocation_log"].cpu(),
            torch.full((slots + 1,), MISSING, dtype=torch.int64),
            "Full journal reset",
        )
        exact(buffers["counter"].cpu(), torch.zeros(1, dtype=torch.uint32), "Counter reset")
        exact(buffers["prefetch_stats"].cpu(), torch.zeros(3, dtype=torch.int64), "Stats reset")
        require(not token.used and token.rows == 1, "Fresh one-use metadata")
        exact(token.metadata[0].cpu(), torch.ones(1, dtype=torch.int32), "Query length")
        exact(token.metadata[1].cpu(), torch.zeros(1, dtype=torch.int32), "Request index")

    with torch.inference_mode():
        dirty()
        token = call()
        torch.cuda.synchronize()
        native = collect_local_native_artifacts(required=False)
        require(
            len(native) == 1 and native[0]["category"] == "echo_indexer", "Unexpected native set"
        )
        identity = {
            "schema": "q1-current-fifo-baseline-v1",
            "source": source,
            "input": {"path": str(args.input.resolve()), "sha256": digest(args.input)},
            "native": native,
            "gpu": str(torch.cuda.get_device_properties(0).uuid),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "shape": {"H": 65536, "P": slots, "Q": 1, "clock": 256},
        }
        for entry in native:
            shutil.copyfile(entry["library"]["path"], args.output / entry["name"])
        write(args.output / "identity.json", identity)
        if args.mode == "check":
            check(token)
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                graph_token = call()
            for _ in range(4):
                dirty()
                capture.replay()
                torch.cuda.synchronize()
                check(graph_token)
            torch.save(
                {
                    "priority": priority.cpu(),
                    "clock": initial["clock"],
                    **{name: value.cpu() for name, value in buffers.items()},
                },
                args.output / "state.pt",
            )
            result = {
                "eager_cases": 1,
                "graph_replays": 4,
                "all_three_saved_inputs_identical": True,
            }
        else:
            require(args.check is not None, "Independent check required")
            receipt = json.loads(args.check.read_text())
            require(receipt["mode"] == "check" and receipt["passed"], "Invalid check")
            require(receipt["identity"] == identity, "Check source/native/input/runtime differs")
            for _ in range(args.warmups):
                call()
            dirty()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range("q1_prepare_baseline/current_complete_call"):
                call()
                torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            result = {"warmups": args.warmups, "profiled_complete_calls": 1}
        require(source_identity() == source, "Source changed during run")
        require(
            collect_local_native_artifacts(required=False) == native, "Native changed during run"
        )
        require(digest(args.input) == identity["input"]["sha256"], "Input changed during run")
        write(
            args.output / "result.json",
            {
                "passed": True,
                "mode": args.mode,
                "identity": identity,
                "result": result,
                "boundary": "Current complete FIFO preparation, no candidate. NCU kernel replay excludes CPU dispatch, memsets and inter-kernel gaps; not clean API latency.",
            },
        )
    print(json.dumps({"passed": True, "mode": args.mode, "output": str(args.output)}))


if __name__ == "__main__":
    main()
