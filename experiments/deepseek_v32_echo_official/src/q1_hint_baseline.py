"""Check, time and profile the unchanged Q1 finite-mean plus decode-EMA APIs.

The pinned corpus contains extra eager checkpoint captures, not formal timed
forward inputs. Saved accepted top-k values avoid loading a scorer or selector.
This baseline-only receipt covers the three real inputs, not candidate promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
from pathlib import Path

import torch

from evaluation.validation import require_receipt, write_receipt
from operators.deepseek_v32.indexer import decode_hint, prefetch_hint

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
KIND = "deepseek-q1-hint-baseline-v1"
LAYERS = ("layer_0", "layer_1", "layer_2")
DEFAULT_CORPUS = Path("/tmp/cxldsagr-checks/q1-topk/q1_topk_check_20261008_02/receipt.json")
PINNED = {
    "receipt": "b12886bebc33311bd75512face7a06f61a21fd04f3b4b6ff3d141f1cd8185823",
    "scores": "1e2727034d9b777f4f4c4ca89ed32a70db431be2ed2deb8d503a23799c13f74a",
    "evidence": "4575c5be70c9c716745c324771d56ac8a4e0cdaa13bdedf2775646975ebeac6f",
}
SOURCE_PATHS = (
    "experiments/deepseek_v32_echo_official/src/q1_hint_baseline.py",
    "experiments/deepseek_v32_echo_official/scripts/profile_q1_hint.sh",
    "evaluation/validation.py",
    "operators/deepseek_v32/indexer/prefetch_hint.py",
    "operators/deepseek_v32/indexer/decode_hint.py",
    "models/deepseek_v32/attention.py",
)
HEADERS = ("ATen/native/cuda/Reduce.cuh", "ATen/native/cuda/MemoryAccess.cuh")
ENV_KEYS = (
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_CACHE_PATH",
    "TRITON_CACHE_DIR",
    "TRITON_PTXAS_PATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bits(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8)


def tensor_identity(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "storage_offset": tensor.storage_offset(),
        "sha256": hashlib.sha256(bits(tensor).numpy().tobytes()).hexdigest(),
    }


def load_corpus(path):
    """Verify the original receipt and use only its accepted logical top-k rows."""
    require(digest(path) == PINNED["receipt"], "Unexpected top-k corpus receipt")
    raw = json.loads(path.read_text())
    receipt = require_receipt(
        path, kind="deepseek-q1-topk-fused-sort-mask-v2", identity=raw["identity"]
    )
    for key in ("scores", "evidence"):
        require(digest(receipt["artifact_paths"][key]) == PINNED[key], f"Changed {key} fixture")
    scores = torch.load(receipt["artifact_paths"]["scores"], map_location="cpu", weights_only=True)
    evidence = torch.load(
        receipt["artifact_paths"]["evidence"], map_location="cpu", weights_only=True
    )
    require(set(scores) == set(LAYERS), "Expected exactly three real score layers")
    corpus = {}
    for name in LAYERS:
        source = scores[name]
        require(
            source.dtype == torch.float32
            and source.shape == (1, 65792)
            and source.stride() == (65792, 1),
            "Unexpected physical score layout",
        )
        require(
            torch.isfinite(source[:, :65537]).all().item()
            and torch.isneginf(source[:, 65537:]).all().item(),
            "Unexpected score padding",
        )
        rows = [
            r for r in evidence["comparisons"] if r["name"] == name + "/logical" and r["k"] == 2048
        ]
        require(len(rows) == 1, "Missing unique logical top-k evidence")
        row = rows[0]
        for arm in ("baseline", "candidate"):
            require(
                all(
                    torch.equal(bits(a), bits(b))
                    for a, b in zip(row[arm], row["oracle"], strict=True)
                ),
                "Saved top-k values or indices differ from their accepted oracle",
            )
        values, indices = row["oracle"]
        require(
            values.dtype == torch.float32 and values.shape == (1, 2048), "Unexpected top-k values"
        )
        require(
            torch.equal(bits(source.gather(1, indices.long())), bits(values)),
            "Top-k values differ from scores",
        )
        corpus[name] = {"scores": source, "values": values}
    return corpus


def source_identity(corpus):
    torch_root = Path(torch.__file__).resolve().parent
    props = torch.cuda.get_device_properties(0)
    return {
        "kind": KIND,
        "sources": {p: digest(ROOT / p) for p in SOURCE_PATHS},
        "corpus_files": PINNED,
        "inputs": {
            name: {key: tensor_identity(tensor) for key, tensor in row.items()}
            for name, row in corpus.items()
        },
        "logical_slice": [0, 65537],
        "offset": "FP32 contiguous [16], positive zero before every measured call",
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "torch_config": torch.__config__.show(),
        "cuda_build": torch.version.cuda,
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_cuda_binary": {
            "path": str(torch_root / "lib/libtorch_cuda.so"),
            "sha256": digest(torch_root / "lib/libtorch_cuda.so"),
        },
        "inspected_headers": {name: digest(torch_root / "include" / name) for name in HEADERS},
        "header_boundary": "Inspected installed headers, not the wheel compiler include closure or proof of line information",
        "gpu": {
            "name": props.name,
            "uuid": str(props.uuid),
            "capability": [props.major, props.minor],
            "sms": props.multi_processor_count,
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "environment": {name: os.environ.get(name) for name in ENV_KEYS},
        "contract": "Unchanged update_prefetch_hint(scores[:, :65537], offset), then update_decode_hint(saved_values, offset)",
        "boundary": "Isolated real-input component; saved scores are extra eager checkpoint captures, not formal timed-forward inputs. No scorer/top-k/model execution or candidate build.",
    }


def runtime_identity(destination=None):
    """Record only loaded Torch/CUDA libraries and the three participating Triton kernels."""
    import triton

    libraries = set()
    torch_root = Path(torch.__file__).resolve().parent
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5]).resolve()
        if ".so" in path.name and (
            path.is_relative_to(torch_root) or path.name.startswith(("libcuda.so", "libcudart.so"))
        ):
            libraries.add(path)
    require(
        any(p.name == "libtorch_cuda.so" for p in libraries), "Loaded Torch CUDA binary missing"
    )
    kernels = {}
    for jit in (*prefetch_hint._kernels(), decode_hint._kernel()):
        entries = [
            compiled for cache in jit.device_caches.values() for compiled in cache[0].values()
        ]
        require(len(entries) == 1, f"Expected one Q1 specialization for {jit.__name__}")
        compiled = entries[0]
        assembly = {
            key: value if isinstance(value, bytes) else value.encode()
            for key, value in compiled.asm.items()
        }
        kernels[compiled.name] = {
            "hash": compiled.hash,
            "metadata": json.loads(json.dumps(compiled.metadata._asdict(), default=str)),
            "asm_sha256": {
                key: hashlib.sha256(value).hexdigest() for key, value in assembly.items()
            },
        }
        if destination is not None:
            for key, value in assembly.items():
                path = destination / "triton" / compiled.hash / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
    require(len(kernels) == 3, "Expected mask/count, publication and EMA kernels")
    return {
        "triton": triton.__version__,
        "kernels": kernels,
        "loaded_libraries": {str(p): digest(p) for p in sorted(libraries)},
        "lineinfo": "Installed PyTorch wheel is measured unchanged; source correlation is unverified",
    }


def call(case):
    prefetch_hint.update_prefetch_hint(case["scores"][:, :65537], case["offset"])
    decode_hint.update_decode_hint(case["values"], case["offset"])


def prepare(cpu):
    case = {key: tensor.cuda() for key, tensor in cpu.items()}
    case["offset"] = torch.zeros(16, device="cuda")
    require(case["scores"].data_ptr() % 16 == 0, "Aligned scores required")
    for _ in range(3):
        call(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(case)
    case["graph"] = graph
    return case


def reset(case):
    case["offset"].zero_()
    torch.cuda.synchronize()


def check(cases, corpus, destination):
    evidence = []
    for name, case in cases.items():
        for execution in ("eager", "graph"):
            # Canary slots expose writes outside the two hint scalars.
            initial = torch.arange(16, dtype=torch.float32, device="cuda")
            initial[:2] = 0
            case["offset"].copy_(initial)
            expected = initial.clone()
            tail = case["scores"][:, :65537][-4:]
            finite = torch.isfinite(tail)
            expected[0] = tail.masked_fill(~finite, 0).sum() / finite.sum().clamp_min(1)
            expected[1] = expected[1] * 0.5 + case["values"][0, -1] * 0.5
            case["graph"].replay() if execution == "graph" else call(case)
            torch.cuda.synchronize()
            require(torch.equal(bits(case["offset"]), bits(expected)), "Hint output bits differ")
            for key in ("scores", "values"):
                require(
                    torch.equal(bits(case[key]), bits(corpus[name][key])), f"Input changed: {key}"
                )
            evidence.append(
                {
                    "layer": name,
                    "execution": execution,
                    "actual": case["offset"].cpu(),
                    "expected": expected.cpu(),
                }
            )
    torch.save(evidence, destination / "evidence.pt")
    return {
        "passed": True,
        "real_layers": 3,
        "comparisons": len(evidence),
        "all_offset_and_input_bits_equal": True,
        "candidate_promotion_coverage": False,
    }


def benchmark(cases, warmups, repeats):
    rows = []
    for name, case in cases.items():
        for execution in ("graph", "eager"):
            action = case["graph"].replay if execution == "graph" else lambda case=case: call(case)
            for _ in range(warmups):
                reset(case)
                action()
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for sample in range(repeats):
                reset(case)
                begin.record()
                action()
                end.record()
                end.synchronize()
                rows.append(
                    {
                        "layer": name,
                        "execution": execution,
                        "sample": sample,
                        "gpu_us": begin.elapsed_time(end) * 1000,
                    }
                )
    return {
        "samples": rows,
        "median_gpu_us": {
            name: {
                execution: statistics.median(
                    row["gpu_us"]
                    for row in rows
                    if row["layer"] == name and row["execution"] == execution
                )
                for execution in ("eager", "graph")
            }
            for name in cases
        },
        "boundary": "CUDA events around one complete mean+EMA call/replay; reset and synchronization before begin are excluded; not model latency",
    }


def profile(cases, layer, warmups):
    case = cases[layer]
    for _ in range(warmups):
        reset(case)
        call(case)
    reset(case)
    label = "q1_hint_baseline_" + layer
    torch.cuda.profiler.start()
    try:
        with torch.cuda.nvtx.range(label):
            call(case)
            torch.cuda.synchronize()
    except BaseException as error:
        try:
            torch.cuda.profiler.stop()
        except BaseException as cleanup:  # noqa: BLE001 - retain both original failures
            raise BaseExceptionGroup(
                "Hint profile and profiler stop failed", [error, cleanup]
            ) from None
        raise
    else:
        torch.cuda.profiler.stop()
    return {
        "layer": layer,
        "execution": "eager",
        "nvtx_range": label,
        "calls_in_range": 1,
        "expected_kernels_in_call": 4,
        "boundary": "Invasive NCU input; use the independent bench result for latency",
    }


def verify_profile_result(destination, receipt_path, layer):
    """NCU success alone does not establish that the child finished its audits."""
    result_path = destination / "result.json"
    result = json.loads(result_path.read_text())
    require(
        result["completed"] is True
        and result["mode"] == "profile"
        and result["run_id"] == destination.name,
        "Profile child did not complete",
    )
    expected = {
        "layer": layer,
        "execution": "eager",
        "nvtx_range": "q1_hint_baseline_" + layer,
        "calls_in_range": 1,
        "expected_kernels_in_call": 4,
    }
    require(
        all(result["payload"].get(key) == value for key, value in expected.items()),
        "Unexpected profile payload",
    )
    require_receipt(receipt_path, kind=KIND, identity=result["identity"])
    require(
        result["receipt"] == {"path": str(receipt_path.resolve()), "sha256": digest(receipt_path)},
        "Profile used a different independent receipt",
    )
    return {
        "completed": True,
        "result_sha256": digest(result_path),
        "receipt_sha256": digest(receipt_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "bench", "profile", "verify-profile"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-receipt", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--layer", choices=LAYERS, default="layer_0")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "verify-profile":
        require(args.receipt is not None, "The expected hint receipt is required")
        print(
            json.dumps(verify_profile_result(args.output_dir.resolve(), args.receipt, args.layer))
        )
        return
    require(args.warmups >= 1 and args.repeats >= 1, "Positive warmup/repeat counts required")
    for name, suffix in (("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda")):
        cache = Path("/tmp/cxldsagr-q1-hint-cache") / suffix
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(cache)
    require(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
        "A visible SM90 GPU is required",
    )
    require(
        torch.__version__ == "2.12.1+cu130"
        and torch.version.git_version == "7269437d655783a26cba32aa88195b741ff496aa",
        "Expected pinned Torch wheel",
    )
    torch.set_num_threads(8)
    corpus = load_corpus(args.corpus_receipt)
    before = source_identity(corpus)
    destination = args.output_dir.resolve()
    require(
        args.mode != "check" or not destination.is_relative_to(EXPERIMENT / "output"),
        "Independent checks must be outside experiment output",
    )
    destination.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        if args.mode == "profile":
            torch.cuda.profiler.stop()
        cases = {name: prepare(cpu) for name, cpu in corpus.items()}
        torch.cuda.synchronize()
        native = runtime_identity()
        identity = {"source": before, "runtime": native}
        receipt = (
            None
            if args.mode == "check"
            else require_receipt(args.receipt, kind=KIND, identity=identity)
        )
        if args.mode == "check":
            payload = check(cases, corpus, destination)
        elif args.mode == "bench":
            payload = benchmark(cases, args.warmups, args.repeats)
        else:
            payload = profile(cases, args.layer, args.warmups)
        require(
            source_identity(load_corpus(args.corpus_receipt)) == before,
            "Source, input or environment changed",
        )
        require(runtime_identity(destination) == native, "Participating runtime changed")
    for relative in SOURCE_PATHS:
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        require(digest(target) == before["sources"][relative], "Source archive differs")
    for name in HEADERS:
        target = destination / "headers" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(torch.__file__).resolve().parent / "include" / name, target)
        require(digest(target) == before["inspected_headers"][name], "Header archive differs")
    torch.save(corpus, destination / "inputs.pt")
    shutil.copy2(args.corpus_receipt, destination / "corpus_receipt.json")
    write(
        destination / "result.json",
        {
            "completed": True,
            "mode": args.mode,
            "run_id": destination.name,
            "identity": identity,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "preparation": {
                "eager_calls_per_layer": 3,
                "graph_captures_per_layer": 1,
                "separate_from_requested_warmups": True,
            },
            "payload": payload,
            "receipt": None
            if receipt is None
            else {"path": str(args.receipt.resolve()), "sha256": digest(args.receipt)},
        },
    )
    if args.mode == "check":
        artifacts = {
            str(p.relative_to(destination)): p for p in destination.rglob("*") if p.is_file()
        }
        write_receipt(
            destination / "receipt.json",
            kind=KIND,
            identity=identity,
            checks=payload,
            artifacts=artifacts,
        )
    print(destination / "result.json", flush=True)


if __name__ == "__main__":
    main()
