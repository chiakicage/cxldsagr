"""Check sparse useful-work counts and the uninstrumented MFU denominator."""

import json

import pytest

from experiments.indexer_block_sparse_profile.src.mfu import (
    NATIVE_SOURCES,
    POLICY,
    WORKLOAD,
    analyze,
    build_report,
    matrix_flops,
    validate_kernel_backend,
    work_counts,
)

CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 16384,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
}


@pytest.mark.parametrize("position", [0, 30, 31, 63, 64, 4095, 4096, 4159, 65536, 66559])
def test_sparse_pairs_match_explicit_causal_tokens(position):
    previous = list(range(position // 64))
    # Different ranked IDs have equal useful work, provided current is mandatory.
    for history in (previous[:63], previous[-63:]):
        selected = [*history, position // 64]
        visible = [
            token
            for block in selected
            for token in range(block * 64, (block + 1) * 64)
            if token <= position
        ]
        counts = work_counts(position, 1, 1024)
        assert counts["sparse_token_pairs_per_q_head_layer"] == len(visible)


@pytest.mark.parametrize(
    ("prefix", "query", "chunk"),
    [(0, 4096, 1024), (0, 4160, 1024), (4032, 128, 128), (4064, 96, 32), (65536, 1024, 1024)],
)
def test_indexer_pairs_enumerate_complete_causal_windows(prefix, query, chunk):
    expected_pairs = expected_queries = 0
    for start in range(prefix, prefix + query, chunk):
        end = min(start + chunk, prefix + query)
        # Match the chunk-level bypass, including chunks straddling the threshold.
        blocks = list(range(0, end, 64))
        if len(blocks) <= 64:
            continue
        windows = [range(offset, offset + 32) for offset in range(0, end - 31, 16)]
        for position in range(start, end):
            expected_queries += 1
            expected_pairs += sum(window[-1] <= position for window in windows)
    counts = work_counts(prefix, query, chunk)
    assert counts["compressed_key_pairs_per_q_head_layer"] == expected_pairs
    assert counts["indexer_scored_queries"] == expected_queries
    assert counts["indexer_bypassed_queries"] == query - expected_queries


def test_flops_conserved_at_prefix_boundary_and_no_dense_attention():
    full = matrix_flops(CONFIG, 0, 66560, 1024)
    prefix = matrix_flops(CONFIG, 0, 65536, 1024)
    extend = matrix_flops(CONFIG, 65536, 1024, 1024)
    assert all(full[name] == prefix[name] + extend[name] for name in full)
    assert sum(full.values()) == 1183130731937792
    assert sum(extend.values()) == 18820462804992
    assert extend["cis_projection"] == 33554432
    assert extend["block_sparse_attention"] == 2182111821824
    assert extend["indexer_qk"] == 1107715686400
    assert "attention_core" not in extend
    counts = work_counts(0, 66560, 1024)
    assert counts["indexer_scored_queries"] == 62464
    assert counts["indexer_bypassed_queries"] == 4096


def inputs():
    metadata = {
        "run_id": "test_run",
        "args": {**WORKLOAD, "mode": "benchmark", "repeats": 5},
        "model_config": dict(CONFIG),
        "gpu": {"name": "NVIDIA H200", "capability": [9, 0]},
        "measurement_boundary": {"output": "normalized hidden states; no LM head"},
        "instrumented_timings": {"full_prefill": 16053.869, "extend": 268.304},
    }
    summary = {
        "schema_version": 1,
        "run_id": "test_run",
        "num_layers": 32,
        "workload": {**WORKLOAD, **POLICY, "repeats": 5},
        "timings": {
            "full_prefill": {
                "tokens": 66560,
                "sample_count": 5,
                "wall_ms": {"median": 10580.463325604796, "count": 5},
            },
            "extend": {
                "tokens": 1024,
                "sample_count": 5,
                "wall_ms": {"median": 194.83687542378902, "count": 5},
            },
        },
    }
    return metadata, summary


def native_inputs(*, backend="native"):
    metadata, summary = inputs()
    summary["workload"].update(
        kernel_backend="cuda_tvm_ffi" if backend == "native" else "triton",
        selection_backend="flashinfer",
    )
    metadata["args"]["kernel_backend"] = backend
    hashes = {name: "a" * 64 for name in NATIVE_SOURCES}
    hashes["operators/sm90/csrc/detail/pipeline.cuh"] = "b" * 64
    metadata.update(
        source_sha256=dict(hashes),
        tvm_ffi="fixture-ffi-version",
        native_build={
            "selected_backend": backend,
            "tvm_ffi": "fixture-ffi-version",
            "compiler": {"path": "/cuda/bin/nvcc", "version": "fixture-nvcc-version"},
            "cuda_flags": ["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            "cutlass": {"commit": "c" * 40, "version_header_sha256": "d" * 64},
            "source_sha256": hashes,
        },
    )
    return metadata, summary


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_native_and_remeasured_triton_control_preserve_math_and_build_identity(backend):
    metadata, summary = native_inputs(backend=backend)
    report = build_report(metadata, summary)
    legacy = build_report(*inputs())
    assert report["phases"] == legacy["phases"]
    assert report["implementation"]["kernel_backend"] == summary["workload"]["kernel_backend"]
    assert report["implementation"]["native_build"] == metadata["native_build"]


@pytest.mark.parametrize(
    "problem",
    [
        "kernel_backend",
        "missing_kernel_backend",
        "selection_backend",
        "args",
        "missing_build",
        "selected_backend",
        "ffi_version",
        "compiler",
        "flags",
        "cutlass",
        "missing_source",
        "header_mismatch",
        "uncaptured_header",
        "invalid_digest",
        "invalid_source_map",
    ],
)
def test_native_backend_metadata_requires_complete_matching_build_inputs(problem):
    metadata, summary = native_inputs()
    workload = summary["workload"]
    build = metadata["native_build"]
    if problem == "kernel_backend":
        workload["kernel_backend"] = "unknown"
    elif problem == "missing_kernel_backend":
        del workload["kernel_backend"]
    elif problem == "selection_backend":
        workload["selection_backend"] = "unknown"
    elif problem == "args":
        metadata["args"]["kernel_backend"] = "triton"
    elif problem == "missing_build":
        del metadata["native_build"]
    elif problem == "selected_backend":
        build["selected_backend"] = "triton"
    elif problem == "ffi_version":
        build["tvm_ffi"] = "different"
    elif problem == "compiler":
        del build["compiler"]["version"]
    elif problem == "flags":
        build["cuda_flags"] = []
    elif problem == "cutlass":
        del build["cutlass"]["commit"]
    elif problem == "missing_source":
        del build["source_sha256"]["operators/sm90/csrc/nosa_scores.cu"]
    elif problem == "header_mismatch":
        metadata["source_sha256"]["operators/sm90/csrc/detail/pipeline.cuh"] = "c" * 64
    elif problem == "uncaptured_header":
        metadata["source_sha256"]["operators/sm90/csrc/missing.cuh"] = "d" * 64
    elif problem == "invalid_source_map":
        metadata["source_sha256"] = []
    else:
        name = "operators/sm90/csrc/detail/pipeline.cuh"
        metadata["source_sha256"][name] = build["source_sha256"][name] = "invalid"
    with pytest.raises(ValueError):
        validate_kernel_backend(workload, metadata)


@pytest.mark.parametrize("marker", ["selection_backend", "native_build", "args"])
def test_modern_capture_cannot_omit_dispatch_identity(marker):
    metadata, summary = inputs()
    workload = summary["workload"]
    if marker == "selection_backend":
        workload[marker] = "flashinfer"
    elif marker == "native_build":
        metadata[marker] = {}
    else:
        metadata["args"]["kernel_backend"] = "native"
    with pytest.raises(ValueError, match="explicit kernel_backend"):
        validate_kernel_backend(workload, metadata)


def test_report_uses_wall_medians_and_dense_bf16_peak():
    metadata, summary = inputs()
    report = build_report(metadata, summary)
    assert report["peak_tflops"] == 989.0
    assert report["phases"]["full_prefill"]["mfu_pct"] == pytest.approx(11.306592955854088)
    assert report["phases"]["extend"]["mfu_pct"] == pytest.approx(9.76703740454975)
    metadata["gpu"]["name"] = "Different GPU"
    with pytest.raises(ValueError, match="peak-tflops"):
        build_report(metadata, summary)
    other = build_report(metadata, summary, peak_tflops=494.5)
    assert other["phases"]["extend"]["mfu_pct"] == report["phases"]["extend"]["mfu_pct"] * 2


@pytest.mark.parametrize("peak", [0, -1, float("nan"), float("inf")])
def test_invalid_peak_rejected(peak):
    with pytest.raises(ValueError, match="peak_tflops"):
        build_report(*inputs(), peak_tflops=peak)


@pytest.mark.parametrize(
    "mismatch", ["run_id", "policy", "chunk", "layers", "tokens", "lm_head", "mode", "repeats"]
)
def test_reject_incompatible_measurement(mismatch):
    metadata, summary = inputs()
    if mismatch == "run_id":
        summary["run_id"] = "another_run"
    elif mismatch == "policy":
        summary["workload"]["block_budget"] = 32
    elif mismatch == "chunk":
        metadata["args"]["chunk_size"] = 2048
    elif mismatch == "layers":
        summary["num_layers"] = 1
    elif mismatch == "tokens":
        summary["timings"]["extend"]["tokens"] = 66560
    elif mismatch == "lm_head":
        metadata["measurement_boundary"]["output"] = "logits"
    elif mismatch == "mode":
        metadata["args"]["mode"] = "profile"
    else:
        summary["timings"]["extend"]["wall_ms"]["count"] = 4
    with pytest.raises(ValueError):
        build_report(metadata, summary)


def test_offline_analysis_preserves_inputs_and_records_provenance(tmp_path):
    metadata, summary = inputs()
    originals = {"metadata.json": json.dumps(metadata), "summary.json": json.dumps(summary)}
    for name, content in originals.items():
        (tmp_path / name).write_text(content)
    report = analyze(tmp_path)
    assert json.loads((tmp_path / "mfu.json").read_text()) == report
    assert set(report["input_sha256"]) == set(originals)
    assert len(report["analysis_source_sha256"]) == 3
    assert all(len(digest) == 64 for digest in report["input_sha256"].values())
    assert all((tmp_path / name).read_text() == content for name, content in originals.items())
