"""Measure useful MFU of the resident NOSA attention and pooled-score operators.

Inputs are seeded synthetic tensors with NOSA-8B dimensions and model Q strides.
Selection and compression call the model's existing implementations outside the
timers. Both backends consume identical tensors and attention selections.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.indexer_block_sparse_profile.src.mfu import work_counts
from experiments.nosa_gr_65536_1024.src.sources import source_hashes
from layers.attention import BlockSelection
from models.nosa.indexer import compressed_scores_reference, prepare_indexer_inputs
from operators.sm90._native import build_info
from operators.sm90.nosa_attention import (
    nosa_block_sparse_attention,
    reference_nosa_block_sparse_attention,
)
from operators.sm90.nosa_indexer import (
    _pool_qa,
    pooled_scores,
    select_contiguous_blocks,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def tensor_hash(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def measure(function, args):
    for _ in range(args.warmup):
        function()
    torch.cuda.synchronize()
    expected = function().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.graph_calls):
            output = function()
    # A replay also warms graph-specific storage and executable state.
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    result = {}
    for label, call, divisor in (
        ("graph", graph.replay, args.graph_calls),
        ("eager", function, 1),
    ):
        samples = []
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        end.record()
        end.synchronize()
        for _ in range(args.repeats):
            begin.record()
            call()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / divisor)
        result[label] = {"ms": distribution(samples), "samples_ms": samples}
    # Keep the final graph result alive until the timing has completed.
    del output
    return result


def numerical_check(
    q, keys, values, compressed, selection, bias, prefix, workspace, *, all_rows=False
):
    """Sample the actual full-batch outputs against independent FP32 QK/AV."""
    attention = nosa_block_sparse_attention(q, keys, values, selection, prefix, bias)
    score = pooled_scores(q.view(-1, 2, 16, 128), compressed, None, prefix, len(keys), workspace)
    assert torch.isfinite(attention).all()
    assert not torch.isnan(score).any()
    samples = sorted({0, 1, 15, 31, 63, 64, 127, 128, len(q) // 2, len(q) - 1} & set(range(len(q))))
    if all_rows:
        samples = list(range(len(q)))
    positions = torch.tensor([prefix + row for row in samples], device=q.device)
    previous = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        reference_score = compressed_scores_reference(
            q[samples].view(-1, 2, 16, 128), compressed, positions
        )
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous
    blocks = workspace.shape[-1]
    reference_pool = torch.empty((len(samples) * 2, blocks), device=q.device, dtype=q.dtype)
    _pool_qa[(len(samples) * 2, (blocks + 127) // 128)](
        reference_score, positions, reference_pool, len(compressed), 2, blocks, 0, False, 128
    )
    actual_pool = score.view(len(q), 2, blocks)[samples].reshape_as(reference_pool)
    torch.testing.assert_close(actual_pool, reference_pool, rtol=0.012, atol=0.001)
    finite = torch.isfinite(reference_pool)
    reference_finite = reference_pool[finite].float()
    score_difference = actual_pool[finite].float() - reference_finite
    score_error = score_difference.abs().max()
    attention_error = 0.0
    attention_difference_squared = 0.0
    attention_reference_squared = 0.0
    attention_strict_mismatches = 0
    for row in samples:
        chosen = BlockSelection(
            selection.block_ids[row : row + 1], 64, selection.valid_mask[row : row + 1]
        )
        expected = reference_nosa_block_sparse_attention(
            q[row : row + 1], keys, values, chosen, prefix + row, bias
        )
        actual = attention[row : row + 1]
        # Use the existing BF16+CIS operator contract. An absolute 0.001
        # derived from unit-variance synthetic V does not scale to checkpoint V.
        torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.016)
        difference = actual.float() - expected.float()
        attention_error = max(attention_error, difference.abs().max().item())
        attention_difference_squared += difference.square().sum().item()
        attention_reference_squared += expected.float().square().sum().item()
        attention_strict_mismatches += (
            (difference.abs() > 0.001 + 0.016 * expected.float().abs()).sum().item()
        )
    return {
        "sampled_rows": samples,
        "attention_max_abs": attention_error,
        "attention_relative_l2": (
            attention_difference_squared / max(attention_reference_squared, 1e-30)
        )
        ** 0.5,
        "attention_rtol": 0.016,
        "attention_atol": 0.016,
        "attention_elements_outside_atol_0_001_rtol_0_016": attention_strict_mismatches,
        "attention_checked_elements": len(samples) * 32 * 128,
        "pooled_scores_max_abs": score_error.item(),
        "pooled_scores_relative_l2": (
            score_difference.norm() / reference_finite.norm().clamp_min(1e-30)
        ).item(),
        "pooled_scores_max_relative": (
            score_difference.abs() / reference_finite.abs().clamp_min(1e-30)
        )
        .max()
        .item(),
        "reference": "FP32 QK, per-head softmax and AV; score GQA sum rounded before pooling",
        "scope": "FP32 reference acceptance of all rows"
        if all_rows
        else "FP32 reference acceptance of sampled full-batch rows",
    }


def make_case(prefix, args):
    generator = torch.Generator(device=args.device).manual_seed(args.seed + prefix)
    kwargs = {"device": args.device, "dtype": torch.bfloat16, "generator": generator}
    q = torch.randn(args.queries, 4608, **kwargs)[:, :4096].view(args.queries, 32, 128)
    keys = torch.randn(prefix + args.queries, 2, 128, **kwargs)
    values = torch.randn(prefix + args.queries, 2, 128, **kwargs)
    bias = torch.randn(prefix + args.queries, 2, **kwargs) * 0.1
    compressed, cis, stable_cis, _ = prepare_indexer_inputs(keys, bias, len(q))
    grouped = q.view(-1, 2, 16, 128)
    os.environ["CXLDSAGR_SM90_BACKEND"] = "native"
    ids, valid = select_contiguous_blocks(
        grouped, compressed, cis, prefix, len(keys), return_valid_mask=True, pooled_cis=stable_cis
    )
    return benchmark_case(
        {
            "q": q,
            "k": keys,
            "v": values,
            "cis": bias,
            "compressed_k": compressed,
            "ids": ids,
            "valid_mask": valid,
        },
        prefix,
        args,
        label=f"synthetic_{prefix}",
    )


def benchmark_case(tensor_inputs, prefix, args, *, label):
    q, keys, values, bias, compressed, ids, valid = (
        tensor_inputs[name] for name in ("q", "k", "v", "cis", "compressed_k", "ids", "valid_mask")
    )
    if q.shape != (args.queries, 32, 128) or q.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 NOSA-8B Q with --queries rows")
    if keys.shape != (prefix + len(q), 2, 128) or values.shape != keys.shape:
        raise ValueError("K/V must span the full visible context")
    if compressed.shape != (len(keys) // 16 - 1, 2, 128) or compressed.dtype != q.dtype:
        raise ValueError("Expected complete stride-16 compressed keys in Q's dtype")
    if ids.shape != (len(q), 2, 64) or valid.shape != ids.shape:
        raise ValueError("Expected per-query 64-block selection")
    grouped = q.view(-1, 2, 16, 128)
    selection = BlockSelection(ids, 64, valid)
    counts = work_counts(prefix, args.queries, args.queries)
    # Check the policy assumed by the FLOP formula against the actual selection.
    qblocks = torch.arange(prefix, len(keys), device=q.device)[:, None, None] // 64
    assert valid.all() and (ids >= 0).all() and (ids == qblocks).sum(-1).eq(1).all()
    assert ids.sort(-1).values.diff(dim=-1).gt(0).all()
    assert (ids <= qblocks).all()
    metadata = {
        "case": label,
        "prefix": prefix,
        "queries": len(q),
        "q_heads": 32,
        "kv_heads": 2,
        "head_dim": 128,
        "blocks_per_query": 64,
        "compressed_keys": len(compressed),
        "work_counts": counts,
        "tensors": {
            name: {
                "shape": list(t.shape),
                "stride": list(t.stride()),
                "dtype": str(t.dtype),
                "sha256": tensor_hash(t),
            }
            for name, t in tensor_inputs.items()
        },
    }
    workspace = torch.empty(
        (args.queries * 2, (len(keys) + 63) // 64), device=q.device, dtype=q.dtype
    )
    results = []
    for backend in ("native", "triton"):
        os.environ["CXLDSAGR_SM90_BACKEND"] = backend
        checked = numerical_check(
            q,
            keys,
            values,
            compressed,
            selection,
            bias,
            prefix,
            workspace,
            all_rows=args.reference_all,
        )
        for name, flops, call in (
            (
                "block_sparse_attention",
                4 * 32 * 128 * counts["sparse_token_pairs_per_q_head_layer"],
                lambda: nosa_block_sparse_attention(q, keys, values, selection, prefix, bias),
            ),
            (
                "pooled_scores",
                2 * 32 * 128 * counts["compressed_key_pairs_per_q_head_layer"],
                lambda: pooled_scores(grouped, compressed, None, prefix, len(keys), workspace),
            ),
        ):
            from operators.sm90 import _nosa_attention_cuda, _nosa_scores_cuda

            owner, entry = (
                (_nosa_attention_cuda, "launch_nosa_block_attention")
                if name == "block_sparse_attention"
                else (_nosa_scores_cuda, "scores_out")
            )
            # Observe dispatch instead of duplicating its shape/length rules.
            with patch.object(owner, entry, wraps=getattr(owner, entry)) as native_call:
                call()
            actual_backend = "cuda_tvm_ffi" if native_call.called else "triton"
            timings = measure(call, args)
            for method in timings.values():
                method["useful_tflops"] = flops / method["ms"]["median"] / 1e9
                method["mfu_pct"] = method["useful_tflops"] / args.peak_tflops * 100
            result = {
                "case": label,
                "prefix": prefix,
                "backend": backend,
                "operator": name,
                "actual_backend": actual_backend,
                "graph_matches_eager_exactly": True,
                "useful_flops": flops,
                "numerical_acceptance": checked,
                **timings,
            }
            print(
                json.dumps(
                    {
                        "case": label,
                        "backend": backend,
                        "actual_backend": actual_backend,
                        "operator": name,
                        "graph_ms": timings["graph"]["ms"]["median"],
                        "graph_mfu_pct": timings["graph"]["mfu_pct"],
                        "eager_ms": timings["eager"]["ms"]["median"],
                        "checked_rows": len(checked["sampled_rows"]),
                        "attention_relative_l2": checked["attention_relative_l2"],
                        "pooled_scores_relative_l2": checked["pooled_scores_relative_l2"],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
            results.append(result)
    return metadata, results


def copy_to_device(tensor, device):
    # .to() may compact sliced QKV views; preserve the measured model strides.
    output = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device=device)
    output.copy_(tensor)
    return output


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--input-dir", type=Path, help="Completed capture_inputs run; replaces synthetic cases"
    )
    parser.add_argument("--prefixes", nargs="+", type=int, default=[4096, 16384, 32768, 65536])
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--graph-calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-all",
        action="store_true",
        help="Check every query against FP32 reference outside timing",
    )
    parser.add_argument("--peak-tflops", type=float)
    args = parser.parse_args(argv)
    if any(n < 1 for n in (args.queries, args.warmup, args.repeats, args.graph_calls)):
        parser.error("queries/warmup/repeats/graph-calls must be positive")
    if any(n < 4096 or n % 64 for n in args.prefixes):
        parser.error("prefixes must be multiples of 64 and >=4096")
    if max(args.prefixes) + args.queries > 262144:
        parser.error("The NOSA indexer supports at most 262144 tokens")
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if (props.major, props.minor) != (9, 0):
        parser.error("This experiment requires SM90/Hopper")
    if args.peak_tflops is None:
        if "H200" not in props.name or "NVL" in props.name.upper():
            parser.error("Other hardware requires explicit --peak-tflops")
        args.peak_tflops = 989.0
    if not 0 < args.peak_tflops < float("inf"):
        parser.error("peak-tflops must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_hashes(
        *sorted((EXPERIMENT / "src").glob("*.py")),
        *sorted((EXPERIMENT / "scripts").glob("*.sh")),
        ROOT / "experiments/indexer_block_sparse_profile/src/mfu.py",
        ROOT / "experiments/indexer_block_sparse_profile/src/analyze.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )
    for name, digest in sources.items():
        content = (ROOT / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError("Source changed while capturing snapshot")
        destination = args.output_dir / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "gpu": {
            "name": props.name,
            "uuid": str(props.uuid),
            "sm_count": props.multi_processor_count,
            "capability": [props.major, props.minor],
            "total_memory": props.total_memory,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("triton", "flashinfer-python", "apache-tvm-ffi")
        },
        "native_build": build_info(),
        "nvidia_smi_before": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,power.limit,clocks.sm,clocks.mem",
                "--format=csv",
            ],
            text=True,
        ),
        "kernel_environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("CXLDSAGR_") or key == "CUDA_VISIBLE_DEVICES"
        },
        "source_sha256": sources,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "peak_tflops": args.peak_tflops,
        "peak_definition": "Nominal dense BF16 Tensor Core peak; not 2:4 sparse or measured sustainable peak",
        "measurement": {
            "graph": "CUDA-event time / graph_calls for repeated same-input calls in one graph; includes all operator kernels and graph gaps, excludes host dispatch",
            "eager": "CUDA-event interval for one public operator call, including host launch gaps",
            "excluded": "Input creation, selection, compression, validation checks, compilation and warmup",
            "allocation": "Attention output and native score scratch allocations occur in the operator call; graph storage reused",
            "flops": "Causal QK+AV for attention, one causal QK for score; exclude recomputation, padding, softmax, GQA sum, pooling, memory ops",
            "scope": "Resident operator workload on synthetic or captured model tensors, repeatedly reused inputs; not model forward or offload",
        },
    }
    cases, results = [], []
    if args.input_dir is None:
        for prefix in args.prefixes:
            case, measured = make_case(prefix, args)
            cases.append(case)
            results.extend(measured)
    else:
        input_metadata_bytes = (args.input_dir / "metadata.json").read_bytes()
        input_metadata = json.loads(input_metadata_bytes)
        if (
            input_metadata.get("kind") != "actual_sparse_model_operator_inputs"
            or input_metadata.get("query_start") != 65536
            or input_metadata.get("queries") != args.queries
            or not input_metadata.get("layers")
        ):
            raise ValueError("Expected a nonempty completed 64K-prefix model input capture")
        metadata["input_capture"] = {
            "path": str(args.input_dir.resolve()),
            "metadata_sha256": hashlib.sha256(input_metadata_bytes).hexdigest(),
            "metadata": input_metadata,
        }
        for entry in input_metadata["layers"]:
            path = args.input_dir / entry["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["file_sha256"]:
                raise ValueError(f"Capture file hash mismatch: {path}")
            tensors = torch.load(path, map_location="cpu", weights_only=True)
            tensors = {
                name: copy_to_device(tensor, args.device) for name, tensor in tensors.items()
            }
            case, measured = benchmark_case(
                tensors, input_metadata["query_start"], args, label=f"layer_{entry['layer']:02d}"
            )
            cases.append(case)
            results.extend(measured)
    # Refuse to publish measurements spanning a concurrent implementation edit.
    if sources != source_hashes(*(ROOT / name for name in sources)):
        raise RuntimeError("Source changed during measurement; rerun from a stable tree")
    metadata["cases"] = cases
    metadata["nvidia_smi_after"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,power.limit,clocks.sm,clocks.mem",
            "--format=csv",
        ],
        text=True,
    )
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "results.json", {"run_id": args.run_id, "results": results})
    with (args.output_dir / "summary.csv").open("w") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "case",
                "prefix",
                "backend",
                "operator",
                "useful_flops",
                "graph_ms",
                "graph_mfu_pct",
                "eager_ms",
                "eager_mfu_pct",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **{
                        name: result[name]
                        for name in ("case", "prefix", "backend", "operator", "useful_flops")
                    },
                    **{
                        f"{mode}_{metric}": result[mode]["ms"]["median"]
                        if metric == "ms"
                        else result[mode][metric]
                        for mode in ("graph", "eager")
                        for metric in ("ms", "mfu_pct")
                    },
                }
            )


if __name__ == "__main__":
    main()
