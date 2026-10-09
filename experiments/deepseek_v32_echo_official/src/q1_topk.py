"""Independent bitwise acceptance, paired API timing and profiling for Q1 top-k."""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np
import torch

from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_echo_official.src import q1_topk_cub
from experiments.deepseek_v32_echo_official.src.q1_topk_candidate import (
    _sort_mask_kernel,
    exact_topk_candidate,
)
from experiments.deepseek_v32_mfu.src.backend_provenance import (
    _installed_flashinfer,
    collect_flashinfer_runtime_artifacts,
)
from operators.deepseek_v32.indexer.selection import exact_topk

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
KIND = "deepseek-q1-topk-fused-sort-mask-v2"
CALLS = {"baseline": exact_topk, "candidate": exact_topk_candidate}
SOURCE_PATHS = (
    "experiments/deepseek_v32_echo_official/src/q1_topk.py",
    "experiments/deepseek_v32_echo_official/src/q1_topk_candidate.py",
    "experiments/deepseek_v32_echo_official/src/q1_topk_cub.py",
    "experiments/deepseek_v32_echo_official/src/q1_topk_sort.cu",
    "experiments/deepseek_v32_mfu/src/backend_provenance.py",
    "evaluation/validation.py",
    "evaluation/local_native.py",
    "operators/deepseek_v32/indexer/selection.py",
    "operators/deepseek_v32/indexer/_native_cache.py",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment():
    base = EXPERIMENT / "output/runtime/q1-topk"
    for name, suffix in (
        ("TRITON_CACHE_DIR", "triton"),
        ("DG_JIT_CACHE_DIR", "deep-gemm"),
        ("TVM_FFI_CACHE_DIR", "tvm"),
        ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("TMPDIR", "tmp"),
    ):
        path = base / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["DG_JIT_WITH_LINEINFO"] = "1"
    import flashinfer.triton  # noqa: F401 - initialize the normal compiler environment

    torch.set_num_threads(8)


def identity(args):
    import triton

    require(torch.cuda.get_device_capability() == (9, 0), "Hopper is required")
    return {
        "kind": KIND,
        "candidate": args.candidate,
        "candidate_build": q1_topk_cub.build_info() if args.candidate == "cub" else None,
        "source_sha256": {p: digest(ROOT / p) for p in SOURCE_PATHS},
        "flashinfer_distribution": _installed_flashinfer(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "physical_device": args.physical_device,
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
        "device_name": torch.cuda.get_device_name(),
        "num_sms": torch.cuda.get_device_properties(0).multi_processor_count,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "TRITON_CACHE_DIR",
                "FLASHINFER_WORKSPACE_BASE",
                "FLASHINFER_CUBIN_DIR",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "contract": "Exact FP32 radix value descending, logical ID ascending, SMALL membership, nonfinite IDs -1",
        "measured": "Complete exact_topk API, including selection, ordering and invalid-ID mask; score preparation excluded",
    }


def triton_kernels():
    for cache in _sort_mask_kernel().device_caches.values():
        yield from cache[0].values()


def native(candidate):
    entries = [
        row for row in collect_flashinfer_runtime_artifacts()["native_jit"] if row["name"] == "topk"
    ]
    require(
        len(entries) == 1 and entries[0]["loaded_in_this_process"], "Actual top-k native is missing"
    )
    if candidate == "cub":
        return {"topk": entries[0], "candidate": candidate, "sort_mask": q1_topk_cub.runtime_info()}
    kernels = {}
    for compiled in triton_kernels():
        kernels[compiled.hash] = {
            "name": compiled.name,
            "metadata": json.loads(json.dumps(compiled.metadata._asdict(), default=str)),
            "asm_sha256": {
                name: hashlib.sha256(
                    value if isinstance(value, bytes) else value.encode()
                ).hexdigest()
                for name, value in compiled.asm.items()
            },
        }
    return {"topk": entries[0], "candidate": candidate, "sort_mask": kernels}


def require_native(actual, expected):
    require(actual["candidate"] == expected["candidate"], "Candidate changed after acceptance")
    require(actual["topk"] == expected["topk"], "Loaded top-k native differs from acceptance")
    require(actual["sort_mask"], "Candidate compiled kernel is missing")
    if actual["candidate"] == "cub":
        require(actual["sort_mask"] == expected["sort_mask"], "CUB native differs from acceptance")
        return
    for key, value in actual["sort_mask"].items():
        require(
            expected["sort_mask"].get(key) == value,
            "Candidate specialization differs from acceptance",
        )


def archive_native(destination, observed):
    artifacts = {}

    def copy(source, relative, expected):
        target = destination / "native" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(digest(target) == expected, "Native artifact changed while archiving")
        artifacts["native/" + relative] = target

    topk = observed["topk"]["library"]
    copy(topk["path"], "topk.so", topk["sha256"])
    if observed["candidate"] == "cub":
        cub = observed["sort_mask"]
        copy(cub["artifact_path"], "cub_sort_mask.so", cub["artifact_sha256"])
    else:
        for compiled in triton_kernels():
            for name, value in compiled.asm.items():
                relative = f"{compiled.hash}/{name}"
                target = destination / "native" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value if isinstance(value, bytes) else value.encode())
                require(
                    digest(target) == observed["sort_mask"][compiled.hash]["asm_sha256"][name],
                    "Triton artifact changed while archiving",
                )
                artifacts["native/" + relative] = target
    return artifacts


def oracle(scores, k):
    source = scores.detach().cpu().contiguous()
    bits = source.numpy().view(np.uint32)
    ordered = bits ^ np.where(
        bits & np.uint32(0x80000000), np.uint32(0xFFFFFFFF), np.uint32(0x80000000)
    )
    ids = np.broadcast_to(np.arange(source.shape[1], dtype=np.uint32), bits.shape)
    selected = np.lexsort((ids, ~ordered), axis=-1)[:, : min(k, source.shape[1])].copy()
    values = source.gather(1, torch.from_numpy(selected).long())
    indices = torch.from_numpy(selected).int()
    indices[~torch.isfinite(values)] = -1
    return values, indices


def exact(actual, expected, label):
    for kind, left, right in zip(("values", "indices"), actual, expected, strict=True):
        left = left.detach().cpu().contiguous().view(torch.int32)
        right = right.detach().cpu().contiguous().view(torch.int32)
        require(torch.equal(left, right), f"Bitwise {kind} mismatch: {label}")


def capture(scores, k, arm):
    for _ in range(3):
        CALLS[arm](scores, k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = CALLS[arm](scores, k)
    return graph, output


def real_scores():
    from experiments.deepseek_v32_echo_official.src import q1_official_prefetch as raw
    from operators.deepseek_v32.indexer import official_decode

    result, sources = {}, {}
    for layer in range(3):
        path = raw.INPUT_ROOT / f"kernel_inputs_layer_{layer}.pt"
        data = torch.load(path, map_location="cpu", weights_only=True)
        case = raw.Case(data, f"layer_{layer}")
        case.threshold.fill_(torch.inf)
        scores = case.call("prepared")
        aligned = scores.as_strided((1, scores.stride(0)), scores.stride())
        require(
            torch.isneginf(aligned[:, case.n :]).all().item(), "Real score tail is not initialized"
        )
        result[f"layer_{layer}"] = aligned.cpu()
        sources[str(path.relative_to(ROOT))] = digest(path)
    preparation = {
        "inputs_sha256": sources,
        "raw_harness_sha256": digest(raw.__file__),
        "official_build": official_decode.build_info(),
        "official_native": official_decode.native_info(
            torch.cuda.get_device_properties(0).multi_processor_count
        ),
        "boundary": "Official no-miss raw score generation is outside the measured top-k API; saved Q1 inputs have their own source identities",
    }
    return result, preparation


def check(destination):
    corpus, preparation = real_scores()
    generator = torch.Generator().manual_seed(7421)
    size = 65792
    random = torch.randn(1, size, generator=generator)
    causal = random.clone()
    causal[:, 65537:] = -torch.inf
    short = random.clone()
    short[:, 3:] = -torch.inf
    zeros = torch.zeros(1, size)
    zeros.view(torch.int32)[:, ::2] = -2147483648
    ties = torch.full((1, size), 1.25)
    near = (torch.arange(size, dtype=torch.int32).remainder(19) + 0x3F800000).view(torch.float32)[
        None
    ]
    inputs = {
        **corpus,
        "random": random,
        "causal": causal,
        "short_finite": short,
        "signed_zero": zeros,
        "all_equal": ties,
        "all_minus_inf": torch.full((1, size), -torch.inf),
        "ulp_ties": near,
    }
    saved, records = [], []

    def compare(name, scores, k):
        expected = oracle(scores, k)
        baseline = exact_topk(scores, k)
        candidate = CALLS["candidate"](scores, k)
        exact(baseline, expected, name + "/baseline")
        exact(candidate, expected, name + "/candidate")
        saved.append(
            {
                "name": name,
                "k": k,
                "baseline": tuple(x.cpu() for x in baseline),
                "candidate": tuple(x.cpu() for x in candidate),
                "oracle": expected,
            }
        )
        records.append(
            {
                "name": name,
                "shape": list(scores.shape),
                "stride": list(scores.stride()),
                "k": k,
                "bitwise": True,
            }
        )

    boundaries = (1, 2, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048)
    for name, cpu in inputs.items():
        scores = cpu.cuda()
        for k in (
            boundaries
            if name in ("random", "signed_zero", "all_equal", "all_minus_inf", "ulp_ties")
            else (1, 129, 2048)
        ):
            compare(name, scores, k)
        if name.startswith("layer"):
            compare(name + "/logical", scores[:, :65537], 2048)
            storage = torch.empty((1, size * 2 + 2), device="cuda")
            strided = storage[:, 1:-1:2]
            strided.copy_(scores)
            compare(name + "/strided", strided, 2048)
        print(name, "bitwise passed", flush=True)
    for length in (1, 2, 63, 127, 128, 129, 2047, 2048, 2049):
        scores = random[:, :length].cuda()
        compare(f"short_N_{length}", scores, 2048)
    compare("two_rows", torch.cat((random, zeros)).cuda(), 2048)
    compare("empty_rows", torch.empty((0, size), device="cuda"), 2048)
    changed = []
    scores = corpus["layer_0"].cuda()
    graphs = {arm: capture(scores, 2048, arm) for arm in CALLS}
    for iteration, (name, cpu) in enumerate(inputs.items()):
        scores.copy_(cpu)
        expected = oracle(cpu, 2048)
        for arm, (graph, output) in graphs.items():
            graph.replay()
            exact(output, expected, f"changed_graph/{iteration}/{arm}")
            changed.append({"input": name, "arm": arm, "output": tuple(x.cpu() for x in output)})
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for name in ("random", "all_equal", "signed_zero"):
            compare("nondefault/" + name, inputs[name].cuda(), 2048)
    side.synchronize()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    torch.save(corpus, destination / "scores.pt")
    torch.save(
        {"inputs": inputs, "comparisons": saved, "changed_graph": changed},
        destination / "evidence.pt",
    )
    return {
        "passed": True,
        "comparisons": records,
        "changed_graph_replays": len(changed),
        "nondefault_stream_cases": 3,
        "preparation": preparation,
    }


def benchmark(corpus, pairs):
    result = []
    for name, cpu in corpus.items():
        scores = cpu.cuda()
        graphs = {arm: capture(scores, 2048, arm) for arm in CALLS}
        for graph, _ in graphs.values():
            for _ in range(10):
                graph.replay()
        torch.cuda.synchronize()
        rows = []
        for repetitions in (1, 20):
            for pair in range(pairs):
                order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
                for arm in order:
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    for _ in range(repetitions):
                        graphs[arm][0].replay()
                    end.record()
                    end.synchronize()
                    rows.append(
                        {
                            "pair": pair,
                            "order": "AB" if pair % 2 == 0 else "BA",
                            "arm": arm,
                            "replays": repetitions,
                            "gpu_us_per_api": begin.elapsed_time(end) * 1000 / repetitions,
                        }
                    )
        summary = {}
        for repetitions in (1, 20):
            selected = [r for r in rows if r["replays"] == repetitions]
            deltas = [
                next(
                    r["gpu_us_per_api"]
                    for r in selected
                    if r["pair"] == pair and r["arm"] == "candidate"
                )
                - next(
                    r["gpu_us_per_api"]
                    for r in selected
                    if r["pair"] == pair and r["arm"] == "baseline"
                )
                for pair in range(pairs)
            ]
            summary[str(repetitions)] = {
                "median_us": {
                    arm: statistics.median(r["gpu_us_per_api"] for r in selected if r["arm"] == arm)
                    for arm in CALLS
                },
                "paired_delta_us": deltas,
                "paired_median_delta_us": statistics.median(deltas),
                "wins": sum(d < 0 for d in deltas),
                "order_median_delta_us": {
                    order: statistics.median(
                        deltas[p] for p in range(pairs) if (p % 2 == 0) == (order == "AB")
                    )
                    for order in ("AB", "BA")
                },
            }
        result.append({"case": name, "samples": rows, "summary": summary})
        print(name, summary, flush=True)
    return result


def profile(corpus, arm, execution):
    prepared = []
    for name, cpu in corpus.items():
        scores = cpu.cuda()
        for method in CALLS if arm == "both" else (arm,):
            graph, output = capture(scores, 2048, method)
            prepared.append((name, scores, method, graph, output))
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for name, scores, method, graph, _ in prepared:
        torch.cuda.nvtx.range_push(f"q1_topk|case={name}|arm={method}|execution={execution}")
        if execution == "graph":
            graph.replay()
        else:
            CALLS[method](scores, 2048)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()
    return [
        {"case": name, "arm": method, "execution": execution} for name, _, method, _, _ in prepared
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "bench", "profile"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--candidate", choices=("triton", "cub"), default="cub")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--arm", choices=("baseline", "candidate", "both"), default="both")
    parser.add_argument("--execution", choices=("graph", "eager"), default="graph")
    parser.add_argument("--case", choices=("layer_0", "layer_1", "layer_2"))
    args = parser.parse_args()
    CALLS["candidate"] = (
        q1_topk_cub.exact_topk_cub if args.candidate == "cub" else exact_topk_candidate
    )
    require(Path(args.run_id).name == args.run_id and args.pairs >= 2, "Invalid run ID or pairs")
    environment()
    before = identity(args)
    base = (
        Path("/tmp/cxldsagr-checks/q1-topk") if args.mode == "check" else EXPERIMENT / "output/data"
    )
    destination = base / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        receipt = None
        if args.mode == "check":
            payload = check(destination)
        else:
            receipt = require_receipt(args.receipt, kind=KIND, identity=before)
            accepted = json.loads(Path(receipt["artifact_paths"]["result"]).read_text())
            corpus = torch.load(
                receipt["artifact_paths"]["scores"], map_location="cpu", weights_only=True
            )
            if args.case:
                corpus = {args.case: corpus[args.case]}
            for scores in corpus.values():
                for call in CALLS.values():
                    call(scores.cuda(), 2048)
            torch.cuda.synchronize()
            require_native(native(args.candidate), accepted["native"])
            payload = (
                benchmark(corpus, args.pairs)
                if args.mode == "bench"
                else profile(corpus, args.arm, args.execution)
            )
        observed = native(args.candidate)
        if receipt is not None:
            require_native(observed, accepted["native"])
        require(identity(args) == before, "Execution sources or environment changed")
    artifacts = {}
    for relative in SOURCE_PATHS:
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        artifacts["source/" + relative] = target
    result = {
        "accepted": True,
        "mode": args.mode,
        "run_id": args.run_id,
        "identity": before,
        "native": observed,
        "payload": payload,
        "receipt": None
        if receipt is None
        else {"path": str(args.receipt), "sha256": digest(args.receipt)},
    }
    write(destination / "result.json", result)
    if args.mode == "check":
        artifacts.update(archive_native(destination, observed))
        artifacts.update(
            {
                "result": destination / "result.json",
                "scores": destination / "scores.pt",
                "evidence": destination / "evidence.pt",
            }
        )
        write_receipt(
            destination / "receipt.json",
            kind=KIND,
            identity=before,
            checks={
                "passed": True,
                "comparisons": len(payload["comparisons"]),
                "changed_graph_replays": payload["changed_graph_replays"],
            },
            artifacts=artifacts,
        )
    print(destination / "result.json", flush=True)


if __name__ == "__main__":
    main()
