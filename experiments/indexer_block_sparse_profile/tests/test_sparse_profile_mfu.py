"""Check sparse useful-work counts and the uninstrumented MFU denominator."""

import json

import pytest

from experiments.indexer_block_sparse_profile.src.mfu import (
    POLICY,
    WORKLOAD,
    analyze,
    build_report,
    matrix_flops,
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
