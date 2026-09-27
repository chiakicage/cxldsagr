"""Derive end-to-end useful matrix MFU from a completed NOSA sparse benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.indexer_block_sparse_profile.src.analyze import (
    PHASES,
    WORKLOAD,
    _duration,
    _integer,
)
from experiments.nosa_gr_65536_1024.src.mfu import matrix_flops as dense_matrix_flops

POLICY = {
    "attention_mode": "sparse",
    "backend": "triton",
    "dtype": "bfloat16",
    "block_size": 64,
    "block_budget": 64,
    "local_blocks": 17,
    "query_stage_blocks": 33,
}
DIMENSIONS = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
)
NATIVE_SOURCES = {
    "operators/sm90/_native.py",
    "operators/sm90/csrc/nosa_attention.cu",
    "operators/sm90/csrc/nosa_scores.cu",
}


def validate_kernel_backend(workload, metadata=None):
    """Validate actual dispatch separately from the historical sparse API name.

    Older captures did not declare kernel_backend and used Triton throughout.
    New captures identify both implementations and the native build inputs,
    including the Triton control measured from the same final source tree.
    This check reads captured JSON only; offline analysis never loads CUDA.
    """
    backend = workload.get("kernel_backend", "triton")
    if backend not in ("triton", "cuda_tvm_ffi"):
        raise ValueError(f"Unsupported kernel_backend: {backend}")
    if "kernel_backend" not in workload:
        if "selection_backend" in workload or (
            metadata is not None
            and ("native_build" in metadata or "kernel_backend" in metadata.get("args", {}))
        ):
            raise ValueError("Native-era capture metadata requires an explicit kernel_backend")
        return backend
    if workload.get("selection_backend") != "flashinfer":
        raise ValueError("Declared kernel_backend requires selection_backend='flashinfer'")
    if metadata is None:
        return backend
    selected = "native" if backend == "cuda_tvm_ffi" else "triton"
    if metadata.get("args", {}).get("kernel_backend") != selected:
        raise ValueError("Capture arguments disagree with kernel_backend")
    build = metadata.get("native_build")
    if not isinstance(build, dict) or build.get("selected_backend") != selected:
        raise ValueError("native_build selected_backend disagrees with kernel_backend")
    if not metadata.get("tvm_ffi") or build.get("tvm_ffi") != metadata["tvm_ffi"]:
        raise ValueError("native_build tvm_ffi version disagrees with runtime metadata")
    compiler = build.get("compiler", {})
    if not isinstance(compiler, dict) or any(
        not isinstance(compiler.get(key), str) or not compiler[key] for key in ("path", "version")
    ):
        raise ValueError("native_build requires compiler path and version")
    flags = build.get("cuda_flags")
    if not isinstance(flags, list) or not flags or any(not isinstance(flag, str) for flag in flags):
        raise ValueError("native_build requires CUDA compiler flags")
    cutlass = build.get("cutlass", {})
    if not isinstance(cutlass, dict) or any(
        not isinstance(cutlass.get(key), str) or not cutlass[key]
        for key in ("commit", "version_header_sha256")
    ):
        raise ValueError("native_build requires shared CUTLASS commit and header fingerprint")
    hashes = build.get("source_sha256")
    captured = metadata.get("source_sha256", {})
    if not isinstance(hashes, dict) or not NATIVE_SOURCES.issubset(hashes):
        raise ValueError("native_build requires complete native source fingerprints")
    if not isinstance(captured, dict) or any(not isinstance(name, str) for name in captured):
        raise ValueError("Captured source fingerprints must map source paths to hashes")
    compiled_sources = {name for name in captured if name.startswith("operators/sm90/csrc/")}
    if not compiled_sources.issubset(hashes):
        raise ValueError("Captured native sources are missing from native_build")
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or captured.get(name) != digest
        for name, digest in hashes.items()
    ):
        raise ValueError("native_build source fingerprints disagree with captured sources")
    return backend


def work_counts(prefix, query, chunk_size):
    """Count useful causal pairs per Q head/layer for the fixed NOSA policy.

    The current block is mandatory. All other selected blocks are complete,
    unique historical blocks, so token counts do not depend on the ranked IDs.
    Indexer QK is bypassed when the entire model chunk fits within 64 blocks.
    Count each complete causal 32-token, stride-16 compressed key once; the
    two-pass recomputation, pooling-halo overlap, future masks and tile padding
    are excluded.
    """
    _integer(prefix, "prefix")
    _integer(query, "query", minimum=1)
    _integer(chunk_size, "chunk_size", minimum=1)
    sparse_pairs = sum(
        64 * min(position // 64, 63) + position % 64 + 1
        for position in range(prefix, prefix + query)
    )
    compressed_pairs = scored_queries = 0
    for start in range(prefix, prefix + query, chunk_size):
        end = min(start + chunk_size, prefix + query)
        if (end + 63) // 64 <= 64:
            continue
        scored_queries += end - start
        compressed_pairs += sum(max(0, (position - 31) // 16 + 1) for position in range(start, end))
    return {
        "sparse_token_pairs_per_q_head_layer": sparse_pairs,
        "compressed_key_pairs_per_q_head_layer": compressed_pairs,
        "indexer_scored_queries": scored_queries,
        "indexer_bypassed_queries": query - scored_queries,
    }


def matrix_flops(config, prefix, query, chunk_size):
    """Count GEMMs, sparse QK/AV, CIS projection and one logical indexer QK."""
    for name in DIMENSIONS:
        _integer(config[name], name, minimum=1)
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    if heads % kv_heads:
        raise ValueError("Q heads must divide into KV groups")
    dim = _integer(config.get("head_dim", config["hidden_size"] // heads), "head_dim", minimum=1)
    layers = config["num_hidden_layers"]
    counts = work_counts(prefix, query, chunk_size)
    flops = dense_matrix_flops(config, prefix, query)
    del flops["attention_core"]
    flops["block_sparse_attention"] = (
        4 * layers * heads * dim * counts["sparse_token_pairs_per_q_head_layer"]
    )
    # delta maps flattened V [Q, Hkv * D] to [Q, Hkv].
    flops["cis_projection"] = 2 * layers * query * (kv_heads * dim) * kv_heads
    flops["indexer_qk"] = 2 * layers * heads * dim * counts["compressed_key_pairs_per_q_head_layer"]
    return flops


def build_report(metadata, summary, *, peak_tflops=None):
    """Use validated uninstrumented wall medians, never profile intervals."""
    if not metadata.get("run_id") or metadata["run_id"] != summary.get("run_id"):
        raise ValueError("metadata and summary run_id must agree")
    if metadata["args"].get("mode") != "benchmark":
        raise ValueError("MFU requires benchmark metadata, not instrumented profile metadata")
    if summary.get("schema_version") != 1:
        raise ValueError("Expected summary schema_version=1")
    workload = summary["workload"]
    kernel_backend = validate_kernel_backend(workload, metadata)
    for name, expected in {**WORKLOAD, **POLICY}.items():
        if workload.get(name) != expected:
            raise ValueError(f"MFU counting requires {name}={expected!r}")
    for name in WORKLOAD.keys() - {"total_tokens"}:
        if metadata["args"][name] != workload[name]:
            raise ValueError(f"metadata and summary must agree on {name}")
    config = metadata["model_config"]
    if summary["num_layers"] != config["num_hidden_layers"]:
        raise ValueError("metadata and summary num_layers must agree")
    if "no LM head" not in metadata["measurement_boundary"]["output"]:
        raise ValueError("MFU counting requires normalized hidden output with no LM head")
    repeats = _integer(workload["repeats"], "repeats", minimum=1)
    if metadata["args"]["repeats"] != repeats:
        raise ValueError("metadata and summary repeats must agree")
    peak_source = "Explicit --peak-tflops override"
    if peak_tflops is None:
        gpu_name = metadata["gpu"]["name"]
        if "H200" not in gpu_name or "NVL" in gpu_name.upper():
            raise ValueError("Pass --peak-tflops for hardware other than H200")
        peak_tflops = 989.0
        peak_source = (
            "Nominal H200 SXM dense BF16 peak; assumed, not measured sustainable throughput"
        )
    peak_tflops = _duration(peak_tflops, "peak_tflops")
    phases = {}
    for phase in PHASES:
        prefix, query = (
            (0, workload["total_tokens"])
            if phase == "full_prefill"
            else (workload["prefix_tokens"], workload["new_tokens"])
        )
        timing = summary["timings"][phase]
        if timing["tokens"] != query:
            raise ValueError(f"{phase} timing token count disagrees with workload")
        samples = _integer(timing["sample_count"], "sample_count", minimum=1)
        if samples != repeats or timing["wall_ms"]["count"] != repeats:
            raise ValueError(f"{phase} timing sample count must match repeats")
        wall_ms = _duration(timing["wall_ms"]["median"], f"{phase}.wall_ms")
        flops = matrix_flops(config, prefix, query, workload["chunk_size"])
        total = sum(flops.values())
        effective_tflops = total / wall_ms / 1e9
        phases[phase] = {
            "prefix_tokens": prefix,
            "query_tokens": query,
            "work_counts": work_counts(prefix, query, workload["chunk_size"]),
            "matrix_flops": flops,
            "total_matrix_flops": total,
            "sample_count": samples,
            "wall_ms": wall_ms,
            "effective_tflops": effective_tflops,
            "mfu_pct": effective_tflops / peak_tflops * 100,
        }
    return {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "gpu": metadata["gpu"],
        "peak_tflops": peak_tflops,
        "peak_source": peak_source,
        "peak_kind": "BF16 dense Tensor Core peak, not 2:4 structured sparsity peak",
        "workload": workload,
        "implementation": {
            "sparse_backend_api": workload["backend"],
            "kernel_backend": kernel_backend,
            "selection_backend": workload.get("selection_backend"),
            "tvm_ffi": metadata.get("tvm_ffi"),
            "native_build": metadata.get("native_build"),
        },
        "dimensions": {
            **{name: config[name] for name in DIMENSIONS},
            "head_dim": config.get(
                "head_dim", config["hidden_size"] // config["num_attention_heads"]
            ),
        },
        "definitions": {
            "mfu_pct": "100 * useful matrix FLOPs / (median unprofiled wall seconds * peak FLOP/s)",
            "matmul": "2 FLOPs per multiply-add; forward only, no LM head",
            "sparse_attention": "QK + AV over selected causal tokens; current block is mandatory",
            "indexer_qk": "One logical QK per complete causal compressed key; short chunks bypass QK",
            "excluded_flops": (
                "Two-pass QK recomputation, pooling-halo overlap, masked future work, tile padding, compression, "
                "softmax/softplus, pooling/top-k, GQA score reduction, CIS bias, norms, RoPE, "
                "activations and memory operations; their time stays in the denominator"
            ),
            "interpretation": "Useful matrix MFU, not hardware instruction utilization or SM occupancy",
        },
        "phases": phases,
    }


def analyze(data_dir, *, peak_tflops=None):
    """Append an offline derivation without changing the captured measurements."""
    data_dir = Path(data_dir)
    inputs = {name: (data_dir / name).read_bytes() for name in ("metadata.json", "summary.json")}
    report = build_report(
        json.loads(inputs["metadata.json"]),
        json.loads(inputs["summary.json"]),
        peak_tflops=peak_tflops,
    )
    report["derived_at_utc"] = datetime.now(UTC).isoformat()
    report["input_sha256"] = {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()}
    root = Path(__file__).resolve().parents[3]
    report["analysis_source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "experiments/indexer_block_sparse_profile/src/mfu.py",
            "experiments/indexer_block_sparse_profile/src/analyze.py",
            "experiments/nosa_gr_65536_1024/src/mfu.py",
        )
    }
    (data_dir / "mfu.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir", type=Path, help="Completed run with metadata.json and summary.json"
    )
    parser.add_argument("--peak-tflops", type=float, help="Dense BF16 peak; default 989 for H200")
    args = parser.parse_args(argv)
    report = analyze(args.data_dir, peak_tflops=args.peak_tflops)
    print(json.dumps(report["phases"], indent=2))


if __name__ == "__main__":
    main()
