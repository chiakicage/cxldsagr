"""Validate and measure parallel official-staging publication checks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_echo_official.src import q1_promotion as base

KIND = "deepseek-official-q1-validation-candidate-v1"


def compare(case):
    for key, expected in case["expected"].items():
        if not base.exact_bytes(case["tensors"][key].cpu(), expected):
            raise AssertionError(f"Changed graph/stream input mismatch for {key}")


def graph_checks(module):
    replays, streams = 0, 0
    for attempts in (0, 1, 31, 32, 33, 63, 64, 91):
        for occupied in (0, 32, 64):
            params = {
                "slots": 96,
                "attempts": attempts,
                "occupied": occupied,
                "layout": "random",
                "offsets": (1, 3),
                "payload": None,
            }
            case = base.make_case(**params)
            graph, _, end = base.capture(module, case)
            for seed in (7301, 8173, 12319):
                changed = base.make_case(**params, seed=seed)
                for name, tensor in case["tensors"].items():
                    tensor.copy_(changed["tensors"][name])
                case["expected"] = changed["expected"]
                graph.replay()
                end.synchronize()
                compare(case)
                replays += 1
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                changed = base.make_case(**params, seed=9319)
                for name, tensor in case["tensors"].items():
                    tensor.copy_(changed["tensors"][name])
                case["expected"] = changed["expected"]
                base.call(module, case)
            stream.synchronize()
            compare(case)
            streams += 1
    return {"changed_graph_replays": replays, "nondefault_stream_calls": streams}


def invalid_call(args, module):
    case = base.make_case(
        slots=96, attempts=64, occupied=64, layout="consecutive", offsets=(0, 0), payload=None
    )
    destination, source = map(int, args.invalid_pair.split(","))
    case["tensors"]["sorted_slots"][destination] = case["tensors"]["sorted_slots"][source]
    base.call(module, case)
    torch.cuda.synchronize()


def additional_checks(args, module):
    result = graph_checks(module)
    invalid = []
    for pair in ("32,0", "63,0", "63,32", "63,62"):
        command = [
            sys.executable,
            "-B",
            "-m",
            "experiments.deepseek_v32_echo_official.src.q1_validation",
            "--mode",
            "invalid",
            "--invalid-pair",
            pair,
            "--source-dir",
            str(args.source_dir),
            "--input",
            str(args.input),
            "--output-dir",
            str(args.output_dir / ("invalid_" + pair.replace(",", "_"))),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        (args.output_dir / ("invalid_" + pair.replace(",", "_") + ".stderr")).write_text(
            completed.stderr
        )
        if completed.returncode == 0 or "device-side assert triggered" not in completed.stderr:
            raise AssertionError(f"Cross-warp duplicate did not assert: {pair}")
        invalid.append({"pair": pair, "returncode": completed.returncode, "device_assert": True})
    result["additional_invalid_inputs"] = invalid
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile", "invalid"), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=base.DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--variant", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--layout", choices=("consecutive", "random"), default="consecutive")
    parser.add_argument("--occupied", type=int, choices=(0, 32, 64), default=0)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--invalid-pair")
    args = parser.parse_args()
    args.source_dir, args.input, args.output_dir = (
        args.source_dir.resolve(),
        args.input.resolve(),
        args.output_dir.resolve(),
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Validation candidate requires a visible SM90 GPU")
    if min(args.warmups, args.repeats) < 1:
        raise ValueError("Positive warmup/repeat counts are required")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    modules, native = {}, {}
    for variant in ("baseline", "candidate"):
        modules[variant], native[variant] = base.load_module(
            args.source_dir / (variant + ".cu"), variant
        )
    if args.mode == "invalid":
        invalid_call(args, modules["candidate"])
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_files = [
        Path(__file__),
        Path(base.__file__),
        base.ADAPTER_TEST,
        Path(base.official_prefetch.__file__),
        Path(base._native_cache.__file__),
        *(args.source_dir / (variant + ".cu") for variant in modules),
    ]
    sources = {str(path): base.digest(path) for path in source_files}
    for path in source_files:
        destination = args.output_dir / "source" / path.relative_to(base.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    identity = {
        "schema": KIND,
        "source_sha256": sources,
        "input_sha256": base.digest(args.input),
        "native": {
            variant: {
                "path": record["artifact_path"],
                "sha256": record["artifact_sha256"],
                "build_identity_sha256": base.identity_digest(record["build_identity"]),
            }
            for variant, record in native.items()
        },
        "device_capability": list(torch.cuda.get_device_capability()),
        "device_name": torch.cuda.get_device_name(),
    }
    base.write_json(args.output_dir / "native.json", native)
    base.write_json(args.output_dir / "identity.json", identity)
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    if (
        payload["query_start"] != 65536
        or tuple(payload["kv"].shape) != (65537, 576)
        or payload["kv"].dtype != torch.bfloat16
    ):
        raise ValueError("Unexpected real layer-2 payload")
    if args.mode == "check":
        result = base.run_check(args, modules, identity, payload)
        result.update(additional_checks(args, modules["candidate"]))
        base.write_json(args.output_dir / "check.json", result)
    else:
        require_receipt(args.receipt, kind=KIND, identity=identity)
        if args.mode == "bench":
            result = base.run_bench(args, modules, payload)
        else:
            result = base.run_profile(args, modules, payload)
            result["kernels_in_call"] = 2
    for path, expected in sources.items():
        if base.digest(path) != expected:
            raise RuntimeError(f"Source changed during execution: {path}")
    if base.digest(args.input) != identity["input_sha256"]:
        raise RuntimeError("Input changed during execution")
    for record in native.values():
        if base.digest(record["artifact_path"]) != record["artifact_sha256"]:
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
    base.write_json(args.output_dir / "result.json", result)
    print(args.output_dir / "result.json")


if __name__ == "__main__":
    main()
