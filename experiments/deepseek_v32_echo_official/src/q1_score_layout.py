"""Independent exact-top-k check and timing of official Q1 physical score padding."""

import argparse
import hashlib
import json
import os
import shutil
import statistics
from pathlib import Path

import torch

from experiments.deepseek_v32_echo_official.src import q1_official_prefetch as raw
from experiments.deepseek_v32_mfu.src.backend_provenance import collect_flashinfer_runtime_artifacts
from operators.deepseek_v32.indexer import official_decode
from operators.deepseek_v32.indexer.selection import _nonfinite_index_mask_kernel, exact_topk


def identity(args):
    result = raw.identity(args)
    result["harness_sha256"][str(Path(__file__).relative_to(raw.ROOT))] = raw.sha256(__file__)
    provenance = Path("experiments/deepseek_v32_mfu/src/backend_provenance.py")
    result["harness_sha256"][str(provenance)] = raw.sha256(provenance)
    result["boundaries"] = {
        "measured": "Complete exact_topk(scores,2048), including sorting and nonfinite ID mask. "
        "Official logits are prepared outside both arms; candidate only exposes existing -inf "
        "physical padding without copying or launching a kernel."
    }
    return result


def padded(scores):
    return scores.as_strided((1, scores.stride(0)), scores.stride())


def exact(left, right):
    for a, b in zip(left, right, strict=True):
        raw.require(torch.equal(a.view(torch.int32), b.view(torch.int32)), "Top-k differs")


def graph(scores):
    for _ in range(3):
        exact_topk(scores, 2048)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        output = exact_topk(scores, 2048)
    return g, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "bench"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[raw.INPUT_ROOT / f"kernel_inputs_layer_{i}.pt" for i in range(3)],
    )
    args = parser.parse_args()
    raw.require(Path(args.run_id).name == args.run_id and args.pairs > 0, "Invalid run settings")
    raw.require(args.mode == "check" or args.receipt is not None, "Benchmark requires receipt")
    # Use the main model's existing FlashInfer workspace for the measured API.
    # Raw ECHO compilation still uses the isolated official-kernel workspace.
    flashinfer_environment = {
        name: os.environ.get(name) for name in ("FLASHINFER_WORKSPACE_BASE", "FLASHINFER_CUBIN_DIR")
    }
    raw.environment()
    for name, value in flashinfer_environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    torch.set_num_threads(8)
    before = identity(args)
    if args.mode == "bench":
        receipt = json.loads(args.receipt.read_text())
        raw.require(receipt["accepted"] and receipt["identity"] == before, "Receipt differs")
    root = (
        Path("/tmp/cxldsagr-checks/q1-score-layout")
        if args.mode == "check"
        else Path(__file__).resolve().parents[1] / "output/data"
    )
    destination = root / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    with torch.inference_mode():
        for path in args.inputs:
            case = raw.Case(torch.load(path, map_location="cpu", weights_only=True), path.stem)
            case.threshold.fill_(torch.inf)
            scores = case.call("prepared")
            aligned = padded(scores)
            raw.require(torch.isneginf(aligned[:, case.n :]).all().item(), "Uninitialized tail")
            graphs = {
                arm: graph(value) for arm, value in (("logical", scores), ("padded", aligned))
            }
            if args.mode == "check":
                variants = []
                for kind in ("actual", "ties", "changed", "causal"):
                    if kind == "ties":
                        scores.zero_()
                    elif kind == "changed":
                        scores.copy_(torch.arange(case.n, device="cuda").remainder(3072).float())
                    elif kind == "causal":
                        scores[:, 32000:].fill_(-torch.inf)
                    expected = exact_topk(scores, 2048)
                    for _ in range(4):
                        for capture, output in graphs.values():
                            capture.replay()
                            exact(output, expected)
                    variants.append(kind)
                results.append({"case": case.name, "bitwise_variants": variants, "replays": 4})
            else:
                for capture, _ in graphs.values():
                    for _ in range(10):
                        capture.replay()
                torch.cuda.synchronize()
                samples = []
                for pair in range(args.pairs):
                    for arm in ("logical", "padded") if pair % 2 == 0 else ("padded", "logical"):
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        begin.record()
                        graphs[arm][0].replay()
                        end.record()
                        end.synchronize()
                        samples.append(
                            {"pair": pair, "arm": arm, "gpu_us": begin.elapsed_time(end) * 1000}
                        )
                results.append(
                    {
                        "case": case.name,
                        "samples": samples,
                        "calls_per_sample": 1,
                        "median_gpu_us": {
                            arm: statistics.median(s["gpu_us"] for s in samples if s["arm"] == arm)
                            for arm in graphs
                        },
                    }
                )
            print(case.name, results[-1].get("median_gpu_us", "passed"), flush=True)
        torch.cuda.synchronize()
    masks = []
    for cache in _nonfinite_index_mask_kernel().device_caches.values():
        for compiled in cache[0].values():
            masks.append(
                {
                    "hash": compiled.hash,
                    "metadata": json.loads(json.dumps(compiled.metadata._asdict(), default=str)),
                    "asm_sha256": {
                        name: hashlib.sha256(
                            value if isinstance(value, bytes) else value.encode()
                        ).hexdigest()
                        for name, value in compiled.asm.items()
                    },
                }
            )
    native = {
        "official": official_decode.native_info(before["num_sms"]),
        "flashinfer": collect_flashinfer_runtime_artifacts()["native_jit"],
        "mask": sorted(masks, key=lambda item: item["hash"]),
    }
    raw.require(
        any(
            item["name"] == "topk" and item["loaded_in_this_process"]
            for item in native["flashinfer"]
        ),
        "No loaded top-k native artifact was observed",
    )
    raw.require(identity(args) == before, "Sources changed during execution")
    if args.mode == "bench":
        raw.require(native == receipt["native"], "Native differs from check")
    snapshot = destination / "source"
    for path in before["harness_sha256"]:
        target = snapshot / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw.ROOT / path, target)
    (destination / "result.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "mode": args.mode,
                "identity": before,
                "native": native,
                "results": results,
                "receipt": None
                if args.receipt is None
                else {
                    "path": str(args.receipt),
                    "sha256": raw.sha256(args.receipt),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(destination / "result.json")


if __name__ == "__main__":
    main()
