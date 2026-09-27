"""Synthetic asynchronous launches exercise module attribution, without a GPU."""

import sqlite3
from copy import deepcopy

import pytest

from experiments.indexer_block_sparse_profile.src.mfu import POLICY
from experiments.indexer_block_sparse_profile.src.module_mfu import (
    FUSED_STAGES,
    PARENTS,
    SUPPORTED_GRAPHS,
    _flops_by_module,
    _validate_source_graph,
    attribute_kernels,
    build_module_report,
    validate_trace_gpu,
)

TIME_SCALE = 100


@pytest.mark.parametrize("graph", SUPPORTED_GRAPHS)
def test_reviewed_graph_versions_accept_extra_captured_sources(graph):
    _validate_source_graph(graph | {"capture.py": "additional_source_snapshot"})


@pytest.mark.parametrize("problem", ["missing", "unreviewed", "mixed_versions"])
def test_incomplete_or_unreviewed_graph_is_rejected(problem):
    hashes = dict(SUPPORTED_GRAPHS[1])
    if problem == "missing":
        hashes.pop("models/nosa/indexer.py")
    elif problem == "unreviewed":
        hashes["models/nosa/indexer.py"] = "unreviewed_implementation"
    else:
        hashes["operators/sm90/nosa_indexer.py"] = SUPPORTED_GRAPHS[0][
            "operators/sm90/nosa_indexer.py"
        ]
    with pytest.raises(ValueError, match="Unreviewed inference graph"):
        _validate_source_graph(hashes)


def trace(
    prefix=64,
    query_length=64,
    indexer_query_chunk_size=None,
    *,
    cached=False,
    empty_cache_update=False,
):
    total = prefix + query_length
    workload = {
        "prefix_tokens": prefix,
        "new_tokens": query_length,
        "total_tokens": total,
        "chunk_size": query_length,
    }
    if cached:
        workload.update(indexer_execution="cached_flashinfer_v1", indexer_query_chunk_size=None)
    elif indexer_query_chunk_size is not None:
        workload["indexer_query_chunk_size"] = indexer_query_chunk_size
    scopes, kernels = [], []
    tid = (1 << 24) + 3

    def scope(text, start, end):
        scopes.append(
            {
                "text": text,
                "start": int(start * TIME_SCALE),
                "end": int(end * TIME_SCALE),
                "globalTid": tid,
            }
        )

    def kernel(name, launch, duration=10, grid=(2, 64, 1)):
        kernels.append(
            {
                "name": name,
                "launch_start": int(launch * TIME_SCALE),
                "launch_end": int((launch + 1) * TIME_SCALE),
                # Execution is deliberately outside the launching module's CPU scope.
                "start": int((launch + 3000) * TIME_SCALE),
                "end": int((launch + 3000 + duration) * TIME_SCALE),
                "globalTid": tid,
                "globalPid": 1 << 24,
                "correlationId": len(kernels) + 1,
                "deviceId": 0,
                "contextId": 1,
                "streamId": 7,
                "gridX": grid[0],
                "gridY": grid[1],
                "gridZ": grid[2],
            }
        )

    phase_width = (total // query_length + 10) * 1000
    for phase, offset, positions in (
        ("full_prefill", 0, list(range(0, total, query_length))),
        ("extend", phase_width, [prefix]),
    ):
        scope(f"NOSA/profile/{phase}/0", offset, offset + phase_width - 1000)
        for call, position in enumerate(positions):
            base = offset + 100 + call * 1000

            def stage(name, start, end, *, phase=phase, position=position, base=base):
                scope(
                    f"NOSA/{phase}/layer_0/{name}/q{position}+{query_length}",
                    base + start,
                    base + end,
                )

            kernel("nvjet_sm90_tst_144x128_64x6_1x2_h_bz_TNN", base + 10, grid=(2, 66, 1))
            kernel("BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel", base + 50)
            stage("cis_projection", 100, 180)
            kernel("cutlass_bf16_gemm", base + 120, grid=(32, 1, 1))
            kernel("softplus_kernel", base + 140, duration=5)
            stage("indexer_total", 200, 400)
            scored = (position + query_length + 63) // 64 > 64
            if cached:
                stage("indexer_validate", 220, 240)
                kernel("_finite_partials", base + 225, duration=2)
                kernel("_finite_reduce", base + 232, duration=2)
                stage("indexer_cache_update", 245, 270)
                if not empty_cache_update:
                    kernel("_compress_records", base + 250, duration=6)
                    kernel("_pool_cis_prefix", base + 260, duration=4)
                if scored:
                    for name, kernel_name, start in (
                        ("pooled_scores", "_scores", 280),
                        ("topk_qa", "flashinfer_topk", 302),
                        ("prepare_cis", "_prepare_cis", 324),
                        ("topk_cis", "flashinfer_topk", 346),
                        ("finish_selection", "_finish_selection", 368),
                    ):
                        stage(name, start, start + 20)
                        kernel(kernel_name, base + start + 5, duration=5)
                else:
                    kernel("_select_all_blocks", base + 280, duration=5)
            else:
                kernel("indexer_validation", base + 205)
                stage("compression_k", 220, 260)
                kernel("mean_k", base + 230, duration=20)
                stage("compression_cis", 270, 310)
                kernel("mean_cis", base + 280, duration=15)
                kernel("select_short_blocks", base + 320, duration=5)
                if scored:
                    query_tile = indexer_query_chunk_size or 64
                    tiles = (query_length + query_tile - 1) // query_tile
                    width = 60 / tiles
                    for tile in range(tiles):
                        start = 325 + tile * width
                        stage("compressed_scores", start, start + width / 3)
                        kernel("_scores", base + start + width / 6, duration=width / 6)
                        stage("select_from_scores", start + width / 3, start + width)
                        kernel("_select", base + start + width / 2, duration=width / 6)
            stage("block_sparse_attention", 410, 480)
            kernel("_nosa_block_attention", base + 430, duration=40)
            kernel("nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT", base + 500)
            kernel("fused_add_rmsnorm", base + 520)
            kernel("nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT", base + 540, grid=(2, 66, 1))
            kernel("act_and_mul_kernel", base + 570)
            kernel("nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT", base + 600)
    return scopes, kernels, workload


def test_async_attribution_and_inclusive_indexer_children():
    scopes, kernels, workload = trace()
    report = attribute_kernels(scopes, kernels, 1, workload)
    assert len(report["runs"]) == 2
    full, extend = report["runs"]
    assert full["phase"] == "full_prefill"
    assert extend["phase"] == "extend"
    for run, calls in ((full, 2), (extend, 1)):
        modules = run["modules"]
        for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            assert modules[name]["kernel_count"] == calls
            assert modules[name]["kernel_ms"] == pytest.approx(calls * 10 * TIME_SCALE / 1e6)
        assert modules["cis_projection"]["kernel_count"] == calls * 2
        assert modules["cis_projection"]["kernel_ms"] == pytest.approx(
            calls * 15 * TIME_SCALE / 1e6
        )
        assert modules["cis_projection_gemm"]["kernel_ms"] == pytest.approx(
            calls * 10 * TIME_SCALE / 1e6
        )
        assert modules["cis_projection_gemm"]["parent_module"] == "cis_projection"
        assert modules["indexer_total"]["kernel_ms"] == pytest.approx(calls * 50 * TIME_SCALE / 1e6)
        assert modules["compression_k"]["kernel_ms"] == pytest.approx(calls * 20 * TIME_SCALE / 1e6)
        assert modules["compression_k"]["parent_module"] == "indexer_total"
        assert modules["block_sparse_attention"]["kernel_ms"] == pytest.approx(
            calls * 40 * TIME_SCALE / 1e6
        )


def test_input_order_does_not_affect_launch_order_attribution():
    scopes, kernels, workload = trace()
    expected = attribute_kernels(scopes, kernels, 1, workload)
    actual = attribute_kernels(scopes[::-1], kernels[::-1], 1, workload)
    assert actual == expected


def test_compressed_scoring_bypass_and_parent_child_accounting():
    scopes, kernels, workload = trace(prefix=4096)
    result = attribute_kernels(scopes, kernels, 1, workload)
    for run in result["runs"]:
        modules = run["modules"]
        assert modules["compressed_scores"]["kernel_count"] == 1
        assert modules["select_from_scores"]["kernel_count"] == 1
        assert modules["compressed_scores"]["parent_module"] == "indexer_total"
        calls = 65 if run["phase"] == "full_prefill" else 1
        assert modules["indexer_total"]["kernel_ms"] == pytest.approx(
            (calls * 50 + 20) * TIME_SCALE / 1e6
        )


@pytest.mark.parametrize("query_tile", [64, 512, 1024, 2048])
def test_recorded_indexer_query_tile_controls_long_context_child_counts(query_tile):
    scopes, kernels, workload = trace(
        prefix=4096, query_length=1024, indexer_query_chunk_size=query_tile
    )
    report = attribute_kernels(scopes, kernels, 1, workload)
    for run in report["runs"]:
        expected = (1024 + query_tile - 1) // query_tile
        assert run["modules"]["compressed_scores"]["kernel_count"] == expected
        assert run["modules"]["select_from_scores"]["kernel_count"] == expected


def test_legacy_workload_without_indexer_query_tile_keeps_64_query_chunks():
    scopes, kernels, workload = trace(prefix=4096, query_length=1024)
    report = attribute_kernels(scopes, kernels, 1, workload)
    assert all(run["modules"]["compressed_scores"]["kernel_count"] == 16 for run in report["runs"])
    workload["indexer_query_chunk_size"] = 1024
    with pytest.raises(ValueError, match="query tiling"):
        attribute_kernels(scopes, kernels, 1, workload)


@pytest.mark.parametrize("empty_cache_update", [False, True])
def test_cached_indexer_siblings_conserve_parent_time_and_short_context_bypass(empty_cache_update):
    scopes, kernels, workload = trace(
        prefix=4096, query_length=1024, cached=True, empty_cache_update=empty_cache_update
    )
    report = attribute_kernels(scopes, kernels, 1, workload)
    for run in report["runs"]:
        modules = run["modules"]
        calls = 5 if run["phase"] == "full_prefill" else 1
        assert modules["indexer_validate"]["kernel_count"] == calls * 2
        assert modules["indexer_cache_update"]["kernel_count"] == (
            0 if empty_cache_update else calls * 2
        )
        if empty_cache_update:
            assert modules["indexer_cache_update"]["kernel_ms"] == 0
        for stage in FUSED_STAGES[2:]:
            assert modules[stage]["kernel_count"] == 1
            assert modules[stage]["parent_module"] == "indexer_total"
        assert modules["compressed_scores"]["kernel_count"] == 0
        children = sum(modules[stage]["kernel_ns"] for stage in FUSED_STAGES)
        # Short contexts emit the all-block kernel directly under indexer_total.
        assert modules["indexer_total"]["kernel_ns"] == children + (calls - 1) * 5 * TIME_SCALE
        assert (
            sum(row["kernel_count"] for name, row in modules.items() if name not in PARENTS)
            == run["kernel_count"]
        )


@pytest.mark.parametrize(
    "problem", ["missing_stage", "wrong_order", "empty_validate", "wrong_score", "legacy_scope"]
)
def test_cached_indexer_rejects_incomplete_or_mixed_launch_attribution(problem):
    scopes, kernels, workload = trace(prefix=4096, query_length=1024, cached=True)
    if problem == "missing_stage":
        scopes = [scope for scope in scopes if "/topk_qa/" not in scope["text"]]
    elif problem == "wrong_order":
        for scope in scopes:
            if "/topk_qa/" in scope["text"]:
                scope["text"] = scope["text"].replace("/topk_qa/", "/prepare_cis/")
            elif "/prepare_cis/" in scope["text"]:
                scope["text"] = scope["text"].replace("/prepare_cis/", "/topk_qa/")
    elif problem == "empty_validate":
        kernels = [kernel for kernel in kernels if not kernel["name"].startswith("_finite")]
    elif problem == "wrong_score":
        for kernel in kernels:
            if kernel["name"] == "_scores":
                kernel["name"] = "unexpected_score_implementation"
    else:
        for scope in scopes:
            scope["text"] = scope["text"].replace("/indexer_cache_update/", "/compression_k/")
    with pytest.raises(ValueError):
        attribute_kernels(scopes, kernels, 1, workload)


@pytest.mark.parametrize("problem", ["unknown_execution", "chunked_cached", "unmarked_cached"])
def test_cached_indexer_dispatch_metadata_cannot_be_ambiguous(problem):
    scopes, kernels, workload = trace(prefix=4096, query_length=1024, cached=True)
    if problem == "unknown_execution":
        workload["indexer_execution"] = "unreviewed_backend"
    elif problem == "chunked_cached":
        workload["indexer_query_chunk_size"] = 64
    else:
        workload.pop("indexer_execution")
    with pytest.raises(ValueError):
        attribute_kernels(scopes, kernels, 1, workload)


@pytest.mark.parametrize(
    "source",
    ["cache/indexer_cache.py", "models/nosa/cache.py", "operators/sm90/nosa_compression.py"],
)
def test_cached_graph_requires_new_derived_cache_dependencies(source):
    hashes = dict(SUPPORTED_GRAPHS[-1])
    hashes.pop(source)
    with pytest.raises(ValueError, match="Unreviewed inference graph"):
        _validate_source_graph(hashes)


@pytest.mark.parametrize("cached", [False, True])
def test_module_report_assigns_qk_once_and_accepts_absent_other_generation_stages(cached):
    scopes, kernels, workload = trace(prefix=65536, query_length=1024, cached=cached)
    workload.update(POLICY, repeats=1, profile_repeats=1)
    config = {
        "num_hidden_layers": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
    }
    metadata = {
        "run_id": "synthetic",
        "model_config": config,
        "gpu": {"name": "synthetic"},
        "args": workload | {"mode": "benchmark"},
        "measurement_boundary": {"output": "normalized hidden states; no LM head"},
    }
    summary = {
        "schema_version": 1,
        "run_id": "synthetic",
        "num_layers": 1,
        "workload": workload,
        "timings": {},
        "profiles": {},
    }
    for phase, tokens in (("full_prefill", 66560), ("extend", 1024)):
        present = {
            scope["text"].split("/")[3]
            for scope in scopes
            if scope["text"].startswith(f"NOSA/{phase}/")
        }
        summary["timings"][phase] = {
            "tokens": tokens,
            "sample_count": 1,
            "wall_ms": {"count": 1, "median": 100.0},
        }
        summary["profiles"][phase] = {
            "sample_count": 1,
            "stage_totals": {stage: {"cuda_elapsed_ms": {"median": 1.0}} for stage in present},
        }
    attribution = attribute_kernels(scopes, kernels, 1, workload)
    report = build_module_report(
        metadata,
        summary,
        {"instrumented_timings": {phase: [{}] for phase in summary["profiles"]}},
        attribution,
        peak_tflops=1.0,
    )
    active, absent = (
        ("pooled_scores", "compressed_scores") if cached else ("compressed_scores", "pooled_scores")
    )
    for phase, rows in report["phases"].items():
        modules = rows["modules"]
        assert (
            modules[active]["matrix_flops_per_run"]
            == modules["indexer_total"]["matrix_flops_per_run"]
        )
        assert modules[active]["matrix_flops_per_run"] > 0
        assert modules[absent]["matrix_flops_per_run"] == 0
        assert modules[absent]["mfu_pct_median"] is None
        assert "event_interval_ms_median" not in modules[absent]
        prefix, query = (0, 66560) if phase == "full_prefill" else (65536, 1024)
        expected = _flops_by_module(config, prefix, query, 1024, cached=cached)
        run = next(run for run in report["runs"] if run["phase"] == phase)
        for name in (active, absent, "indexer_total"):
            assert (
                sum(call["modules"][name]["matrix_flops"] for call in run["layer_calls"])
                == (expected[name])
            )


@pytest.mark.parametrize("change", ["missing_gemm", "extra_gemm", "missing_silu", "missing_scope"])
def test_incomplete_or_ambiguous_call_rejected(change):
    scopes, kernels, workload = trace()
    if change == "missing_gemm":
        kernels = kernels[1:]
    elif change == "extra_gemm":
        extra = deepcopy(kernels[0])
        extra.update(
            correlationId=999,
            launch_start=130 * TIME_SCALE,
            start=3130 * TIME_SCALE,
            end=3140 * TIME_SCALE,
        )
        kernels.append(extra)
    elif change == "missing_silu":
        kernels = [kernel for kernel in kernels if "act_and_mul" not in kernel["name"]]
    else:
        scopes = [scope for scope in scopes if "block_sparse_attention" not in scope["text"]]
    with pytest.raises(ValueError):
        attribute_kernels(scopes, kernels, 1, workload)


def test_multiple_streams_rejected_for_serial_reconstruction():
    scopes, kernels, workload = trace()
    kernels[0]["streamId"] = 8
    with pytest.raises(ValueError):
        attribute_kernels(scopes, kernels, 1, workload)


def test_small_timestamp_overlap_preserves_duration_sum():
    scopes, kernels, workload = trace()
    expected = attribute_kernels(scopes, kernels, 1, workload)
    # Nsight can show sub-microsecond timestamp overlap on the same stream.
    kernels[1]["start"] = kernels[0]["end"] - 1
    kernels[1]["end"] = kernels[1]["start"] + 10 * TIME_SCALE
    actual = attribute_kernels(scopes, kernels, 1, workload)
    assert [run["modules"] for run in actual["runs"]] == [
        run["modules"] for run in expected["runs"]
    ]


@pytest.mark.parametrize("mismatch", [None, "uuid", "sm_count", "device"])
def test_sqlite_gpu_identity_matches_kernel_device_and_metadata(tmp_path, mismatch):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE TARGET_INFO_GPU "
            "(id INTEGER,uuid TEXT,name TEXT,smCount INTEGER,computeMajor INTEGER,computeMinor INTEGER)"
        )
        connection.executemany(
            "INSERT INTO TARGET_INFO_GPU VALUES (?,?,?,?,?,?)",
            [(0, "gpu-a", "NVIDIA M403", 132, 9, 0), (1, "gpu-b", "NVIDIA M403", 132, 9, 0)],
        )
    gpu = {"uuid": "GPU-A", "sm_count": 132, "capability": [9, 0]}
    if mismatch == "uuid":
        gpu["uuid"] = "GPU-B"
    elif mismatch == "sm_count":
        gpu["sm_count"] = 100
    device = 2 if mismatch == "device" else 0
    if mismatch:
        with pytest.raises(ValueError):
            validate_trace_gpu(path, device, gpu)
    else:
        assert validate_trace_gpu(path, device, gpu)["id"] == 0
