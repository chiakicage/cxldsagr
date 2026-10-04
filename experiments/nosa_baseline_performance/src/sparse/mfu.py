"""Derive end-to-end useful matrix MFU from a completed NOSA sparse benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.nosa_baseline_performance.src.dense.mfu import matrix_flops as dense_matrix_flops
from experiments.nosa_baseline_performance.src.sparse.analyze import (
    PHASES,
    WORKLOAD,
    _duration,
    _integer,
)

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
FUSED_NATIVE_SOURCES = {
    "operators/sm90/csrc/nosa_indexer.cu",
    "operators/sm90/csrc/nosa_scores_fused.cuh",
    "operators/sm90/csrc/nosa_selection.cu",
    "operators/sm90/csrc/nosa_selection_prefix.cuh",
    "operators/sm90/csrc/nosa_prepare.cu",
    "operators/sm90/csrc/nosa_prepare_ranked.cu",
}
PRUNED_NATIVE_SOURCES = {
    "operators/sm90/csrc/nosa_scores_pruned.cuh",
    "operators/sm90/csrc/nosa_selection_cutoff.cuh",
}
NATIVE_INDEXER_REVISIONS = {f"cached_native_v{revision}": revision for revision in (2, 3, 4, 5)}

# The model-organized loader records the local include closure of its seven
# components. FA3 has a separate build, and captured offload/common sources are
# not implicitly compiled by this loader. Keep the historical flat-layout
# requirements above unchanged so archived metadata remains verifiable.
MODEL_NATIVE_SOURCE_INVENTORY_VERSION = 2
MODEL_NATIVE_GUARDED_SOURCES = {"operators/nosa/indexer/csrc/nosa_guarded_buffers.cuh"}
MODEL_NATIVE_SOURCES = {
    "operators/nosa/_native.py",
    "operators/nosa/attention/device_only/csrc/nosa_attention.cu",
    "operators/nosa/attention/device_only/csrc/nosa_attention_grouped.cuh",
    *MODEL_NATIVE_GUARDED_SOURCES,
    *(
        f"operators/nosa/indexer/csrc/{name}"
        for name in (
            "nosa_indexer.cu",
            "nosa_indexer_checked.cu",
            "nosa_prepare.cu",
            "nosa_prepare_ranked.cu",
            "nosa_scores.cu",
            "nosa_scores_fused.cuh",
            "nosa_scores_pruned.cuh",
            "nosa_selection.cu",
            "nosa_selection_cutoff.cuh",
            "nosa_selection_prefix.cuh",
        )
    ),
}
MODEL_NATIVE_SOURCE_INVENTORIES = {
    1: MODEL_NATIVE_SOURCES - MODEL_NATIVE_GUARDED_SOURCES,
    2: MODEL_NATIVE_SOURCES,
}
# Verified pre-guard local include closure at Git revision
# 4a02aef12dcf60d78625242856c2196e518b7315. An absent version cannot establish
# historical provenance by itself: require this exact recorded source identity.
MODEL_NATIVE_PRE_GUARD_SOURCE_SHA256 = {
    "operators/nosa/_native.py": "4ed3277e55f21abac43259da7f678af74825a3ef37bbd830558bbd127f3e9b92",
    "operators/nosa/attention/device_only/csrc/nosa_attention.cu": "01c0699b507baebe005b2df44e2f78ec149083a6239b96e7d24990edf2bf09f4",
    "operators/nosa/attention/device_only/csrc/nosa_attention_grouped.cuh": "30f35b34a80c2a26c249b92e055c3158a5a2e40e5c818ff9dadc3074dc525927",
    "operators/nosa/indexer/csrc/nosa_indexer.cu": "91e4639259cd02839fa4d2606b118ff40343ac75f0419274ec4f3a868e42ed34",
    "operators/nosa/indexer/csrc/nosa_indexer_checked.cu": "49a303a3f89886ebd373611fde3faa7b6ee0adae3c8f1ebc78855734b2b1b16d",
    "operators/nosa/indexer/csrc/nosa_prepare.cu": "340f7455f11f602ac4e0380ff7bd6963e2cdaa32275bbd470ae763ea0a0a1715",
    "operators/nosa/indexer/csrc/nosa_prepare_ranked.cu": "1c40cdd8edcf6da94cc0ec59e4477035c8db91441b1c3acb76bfcdd163388d09",
    "operators/nosa/indexer/csrc/nosa_scores.cu": "d5c5b543e08a6e164701b9e5070458079c8c843f35a99de96b4c66427dfb01af",
    "operators/nosa/indexer/csrc/nosa_scores_fused.cuh": "56e6cbe675525b419db3699ae57dcf984ab64bf9e12f1bbe3b49ffd21bcd197a",
    "operators/nosa/indexer/csrc/nosa_scores_pruned.cuh": "99e0e6ce9f5685e639016e24732cce249a3b2c043ff0629b021866ad0f051409",
    "operators/nosa/indexer/csrc/nosa_selection.cu": "f2ce6b685336002d5e137ed1af05d1b2bfdda37f89b57323d053a8daa953e9e9",
    "operators/nosa/indexer/csrc/nosa_selection_cutoff.cuh": "7eaabb06b7d39d5ea0779ea209b80a3e7d54e3b137748a35f4cb47836a7f00fe",
    "operators/nosa/indexer/csrc/nosa_selection_prefix.cuh": "e5b447d6427271df99751c93e6d8c14e3cedc7fee9d76c130d8967410a3b6b35",
}

FA3_SOURCE_PATHS = {
    "_nosa_attention_fa3.py": "operators/sm90/_nosa_attention_fa3.py",
    "nosa_attention_fa3.cu": "operators/sm90/csrc/nosa_attention_fa3.cu",
    "nosa_attention.cu": "operators/sm90/csrc/nosa_attention.cu",
    "nosa_attention_grouped.cuh": "operators/sm90/csrc/nosa_attention_grouped.cuh",
}
MODEL_FA3_SOURCE_PATHS = {
    "_fa3.py": "operators/nosa/attention/device_only/_fa3.py",
    **{
        name: f"operators/nosa/attention/device_only/csrc/{name}"
        for name in (
            "nosa_attention_fa3.cu",
            "nosa_attention.cu",
            "nosa_attention_grouped.cuh",
        )
    },
}
FA3_EXTRA_FLAGS = [
    "-use_fast_math",
    "-DFA3_GROUP=8",
    "-DFA3_KV=128",
    "-DFA3_TMA=1",
    "-DFA3_STAGES=2",
]
FA3_CTA_ORDER = {
    "cta_order": "descending_union_tiles",
    "cta_order_work_items": 256,
    "cta_order_ties": "ascending_logical_batch",
}


def _validate_fa3_build(metadata, attention_execution, source_paths=FA3_SOURCE_PATHS):
    """Check the separate FA3 compiler flags and installed-header identity."""
    build = metadata["native_build"]
    fa3 = build.get("attention_fa3")
    if not isinstance(fa3, dict) or not fa3:
        raise ValueError("FA3-era captures require native_build.attention_fa3")
    toolchain = {name: build[name] for name in ("tvm_ffi", "compiler", "cutlass", "cuda_flags")}
    if fa3.get("toolchain") != toolchain:
        raise ValueError("FA3 toolchain disagrees with common native build metadata")
    if fa3.get("cuda_flags") != build["cuda_flags"] + FA3_EXTRA_FLAGS:
        raise ValueError("FA3 requires its reviewed separate CUDA compiler flags")
    if fa3.get("torch") != metadata.get("torch") or not fa3.get("torch"):
        raise ValueError("FA3 torch version disagrees with runtime metadata")
    if fa3.get("flashinfer") != "0.6.18" or fa3["flashinfer"] != metadata.get("flashinfer"):
        raise ValueError("FA3 requires the reviewed installed FlashInfer 0.6.18 headers")
    include = fa3.get("flashinfer_include")
    digest = fa3.get("flashinfer_headers_sha256")
    if (
        not isinstance(include, str)
        or not include.startswith("/")
        or (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        )
    ):
        raise ValueError("FA3 requires its installed include path and complete header fingerprint")
    if any(
        fa3.get(name) != expected
        for name, expected in (
            ("group_queries", 8),
            ("kv_tile_tokens", 128),
            ("stages", 2),
            ("minimum_fa3_queries", 1),
            ("q_transfer", "direct_strided_tma"),
            ("output_store", "direct"),
            ("numerical_repair", "nonfinite_output_postcheck"),
            ("native_pv_accumulation", "bf16_power_of_two_scale_finite_output_guard"),
        )
    ):
        raise ValueError(
            "Unreviewed FA3 geometry, query cutoff, transfer or numerical configuration"
        )
    ordering_fields = set(FA3_CTA_ORDER).intersection(fa3)
    if attention_execution == "native_fa3_v2" and ordering_fields:
        raise ValueError("FA3 v2 cannot declare the v3 CTA ordering contract")
    # Triton controls retain the same native build provenance as their paired
    # native run, so accept either the legacy v2 or complete v3 build contract.
    if (attention_execution == "native_fa3_v3" or ordering_fields) and any(
        type(fa3.get(name)) is not type(expected) or fa3.get(name) != expected
        for name, expected in FA3_CTA_ORDER.items()
    ):
        raise ValueError("Missing or unreviewed FA3 CTA ordering contract")
    hashes = fa3.get("source_sha256")
    captured = metadata["source_sha256"]
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(source_paths)
        or any(
            not isinstance(hashes[name], str)
            or len(hashes[name]) != 64
            or any(character not in "0123456789abcdef" for character in hashes[name])
            or captured.get(path) != hashes[name]
            for name, path in source_paths.items()
        )
    ):
        raise ValueError("FA3 source fingerprints disagree with captured wrapper/kernel sources")


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
        if (
            "selection_backend" in workload
            or "indexer_preparation" in workload
            or (
                metadata is not None
                and ("native_build" in metadata or "kernel_backend" in metadata.get("args", {}))
            )
        ):
            raise ValueError("Native-era capture metadata requires an explicit kernel_backend")
        return backend
    attention = workload.get("attention_execution")
    if attention not in (None, "native_fa3_v2", "native_fa3_v3", "triton_v1"):
        raise ValueError("Unreviewed attention_execution")
    if attention is not None and attention not in (
        ("native_fa3_v2", "native_fa3_v3") if backend == "cuda_tvm_ffi" else ("triton_v1",)
    ):
        raise ValueError("attention_execution disagrees with kernel_backend")
    selection = workload.get("selection_backend")
    if selection not in ("flashinfer", "cuda_tvm_ffi"):
        raise ValueError("Declared kernel_backend requires a reviewed selection_backend")
    if selection == "cuda_tvm_ffi" and (
        backend != "cuda_tvm_ffi"
        or workload.get("indexer_execution") not in NATIVE_INDEXER_REVISIONS
    ):
        raise ValueError("Native selection requires a reviewed cached native dispatcher")
    preparation = workload.get("indexer_preparation")
    if preparation not in (
        None,
        "triton_v1",
        "native_guarded_v1",
        "native_guarded_ranked_v1",
        "native_guarded_ranked_checked_v1",
    ):
        raise ValueError("Unreviewed indexer_preparation")
    if preparation in (
        "native_guarded_v1",
        "native_guarded_ranked_v1",
        "native_guarded_ranked_checked_v1",
    ) and (
        backend != "cuda_tvm_ffi"
        or workload.get("indexer_execution") not in NATIVE_INDEXER_REVISIONS
    ):
        raise ValueError("Native preparation requires a reviewed cached native dispatcher")
    if preparation == "native_guarded_ranked_v1" and workload.get("indexer_execution") not in (
        "cached_native_v3",
        "cached_native_v4",
        "cached_native_v5",
    ):
        raise ValueError("Ranked preparation requires cached_native_v3")
    revision = NATIVE_INDEXER_REVISIONS.get(workload.get("indexer_execution"))
    if (
        revision is not None
        and revision >= 3
        and (
            backend != "cuda_tvm_ffi"
            or selection != "cuda_tvm_ffi"
            or workload.get("native_kernel_revision") != revision
        )
    ):
        raise ValueError(
            "Cached native dispatch requires its matching revision and native selection"
        )
    if preparation == "native_guarded_ranked_checked_v1" and revision not in (4, 5):
        raise ValueError("Checked ranked preparation requires cached_native_v4 or cached_native_v5")
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
    if not isinstance(hashes, dict) or any(not isinstance(name, str) for name in hashes):
        raise ValueError("native_build requires complete native source fingerprints")
    if not isinstance(captured, dict) or any(not isinstance(name, str) for name in captured):
        raise ValueError("Captured source fingerprints must map source paths to hashes")
    model_layout = "operators/nosa/_native.py" in hashes
    if model_layout:
        # Older captures may not declare an inventory version. If either hash
        # map contains the guarded-buffer include, require the newer inventory;
        # otherwise the historical identity check below still applies. New
        # captures declare v2 even when a dependency record is missing.
        guarded_sources = MODEL_NATIVE_GUARDED_SOURCES.intersection(hashes.keys() | captured.keys())
        inventory_version = build.get("source_inventory_version", 2 if guarded_sources else 1)
        if (
            type(inventory_version) is not int
            or inventory_version not in MODEL_NATIVE_SOURCE_INVENTORIES
            or (inventory_version == 1 and guarded_sources)
        ):
            raise ValueError("Unreviewed model native source inventory version")
        if not MODEL_NATIVE_SOURCE_INVENTORIES[inventory_version].issubset(hashes):
            raise ValueError("Model native build requires all seven components and local includes")
        if inventory_version == 1 and hashes != MODEL_NATIVE_PRE_GUARD_SOURCE_SHA256:
            raise ValueError("Model native v1 requires a reviewed pre-guard source identity")
        if any(name.startswith("operators/sm90/") for name in hashes):
            raise ValueError("Native build cannot mix historical and model source layouts")
    else:
        if not NATIVE_SOURCES.issubset(hashes):
            raise ValueError("native_build requires complete native source fingerprints")
        if revision is not None and revision >= 3 and not FUSED_NATIVE_SOURCES.issubset(hashes):
            raise ValueError(
                "Fused native dispatch requires all score/selection/preparation sources"
            )
        if revision in (4, 5) and "operators/sm90/csrc/nosa_indexer_checked.cu" not in hashes:
            raise ValueError("Checked native dispatch requires its complete source fingerprint")
        if workload.get("native_kernel_revision") == 5 and not PRUNED_NATIVE_SOURCES.issubset(
            hashes
        ):
            raise ValueError("Pruned native revision requires complete bound and cutoff sources")
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
    if (model_layout or "attention_fa3" in build) and attention is None:
        raise ValueError("FA3-era captures require an explicit attention_execution")
    if model_layout or attention is not None or "attention_fa3" in build:
        _validate_fa3_build(
            metadata,
            attention,
            MODEL_FA3_SOURCE_PATHS if model_layout else FA3_SOURCE_PATHS,
        )
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
            "attention_execution": workload.get("attention_execution"),
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
    root = Path(__file__).resolve().parents[4]
    report["analysis_source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "experiments/nosa_baseline_performance/src/sparse/mfu.py",
            "experiments/nosa_baseline_performance/src/sparse/analyze.py",
            "experiments/nosa_baseline_performance/src/dense/mfu.py",
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
