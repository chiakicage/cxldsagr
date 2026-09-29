"""Synthetic asynchronous launches exercise module attribution, without a GPU."""

import hashlib
import json
import sqlite3
from copy import deepcopy

import pytest

from experiments.indexer_block_sparse_profile.src.mfu import POLICY
from experiments.indexer_block_sparse_profile.src.module_mfu import (
    FUSED_SCORE_FAMILY,
    FUSED_STAGES,
    GROUPED_ATTENTION_SOURCE,
    PARENTS,
    QK_MODULES,
    SUPPORTED_GRAPHS,
    _flops_by_module,
    _fused_selection_flag,
    _validate_inputs,
    _validate_operator_kernels,
    _validate_source_graph,
    attribute_kernels,
    build_module_report,
    validate_trace_gpu,
)

TIME_SCALE = 100


@pytest.mark.parametrize("graph", SUPPORTED_GRAPHS)
def test_reviewed_graph_versions_accept_extra_captured_sources(graph):
    assert _validate_source_graph(graph | {"capture.py": "additional_source_snapshot"}) == (
        4 if GROUPED_ATTENTION_SOURCE in graph else 1
    )


def test_unreferenced_grouped_header_does_not_change_legacy_graph_dispatch():
    graph = SUPPORTED_GRAPHS[3] | {
        GROUPED_ATTENTION_SOURCE: SUPPORTED_GRAPHS[4][GROUPED_ATTENTION_SOURCE]
    }
    assert _validate_source_graph(graph) == 1


@pytest.mark.parametrize("problem", ["missing", "changed", "mixed_dispatcher"])
def test_grouped_graph_requires_matching_cuda_dispatcher_and_header(problem):
    hashes = dict(SUPPORTED_GRAPHS[4])
    if problem == "missing":
        hashes.pop(GROUPED_ATTENTION_SOURCE)
    elif problem == "changed":
        hashes[GROUPED_ATTENTION_SOURCE] = "unreviewed_grouped_implementation"
    else:
        name = "operators/sm90/_nosa_attention_cuda.py"
        hashes[name] = SUPPORTED_GRAPHS[3][name]
    with pytest.raises(ValueError, match="Unreviewed inference graph"):
        _validate_source_graph(hashes)


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
    native=False,
    native_attention_group_size=4,
    native_revision=1,
    native_prepare=False,
    native_ranked=False,
    num_kv_heads=2,
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
    if native:
        workload.update(kernel_backend="cuda_tvm_ffi", selection_backend="flashinfer")
        if native_revision >= 2:
            workload.update(
                native_kernel_revision=native_revision,
                selection_backend="cuda_tvm_ffi",
                indexer_execution=f"cached_native_v{native_revision}",
            )
    if native_prepare:
        workload["indexer_preparation"] = "native_guarded_v1"
    if native_ranked:
        workload["indexer_preparation"] = "native_guarded_ranked_v1"
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
                native_score = native and position + query_length >= (
                    32768 if query_length >= 1024 else 8192
                )
                shared = (
                    query_length >= 128
                    and (position + query_length + 63) // 64 <= 1056
                    and position // 64 >= 64
                    and (position + query_length - 1) // 64 - position // 64 <= 16
                )
                prepared_ranking = native_ranked and shared
                if native_prepare or native_ranked:
                    stage(
                        "native_prepare_ranked" if prepared_ranking else "native_prepare", 220, 270
                    )
                    kernel("nosa_prepare::finite_partials<bf16>", base + 225, duration=2)
                    kernel(
                        "nosa_prepare_ranked::guarded_compression_and_ranking<bf16>"
                        if prepared_ranking
                        else "nosa_prepare::guarded_compression<bf16,128>",
                        base + 250,
                        duration=6,
                    )
                else:
                    stage("indexer_validate", 220, 240)
                    kernel("_finite_partials", base + 225, duration=2)
                    kernel("_finite_reduce", base + 232, duration=2)
                    stage("indexer_cache_update", 245, 270)
                    if not empty_cache_update:
                        kernel("_compress_records", base + 250, duration=6)
                        kernel("_pool_cis_prefix", base + 260, duration=4)
                if scored and native and native_revision == 3:
                    joint = query_length >= 128 and position + query_length >= 32768
                    score_fused = (
                        native_score
                        and query_length >= 1024
                        and (query_length + 15) // 16 * num_kv_heads >= 128
                    )
                    selection_fused = joint and score_fused and query_length <= 1088 and shared
                    stage(
                        "native_indexer" if joint else "pooled_scores", 280, 388 if joint else 300
                    )
                    if not joint:
                        stage("native_selection", 302, 388)
                    if selection_fused and not prepared_ranking:
                        kernel("nosa_selection::prepare_prefix_ranking", base + 282, duration=2)
                    if score_fused:
                        kernel(
                            f"{FUSED_SCORE_FAMILY}<cutlass::bfloat16_t, cutlass::bfloat16_t, true, {'true' if selection_fused else 'false'}, Map<Shape<16, 64>, 128>>()",
                            base + 289,
                            duration=6,
                        )
                    elif native_score:
                        kernel("nosa_scores::normalizer_kernel<bf16>", base + 284, duration=2)
                        kernel("nosa_scores::scores_kernel<bf16>", base + 289, duration=5)
                    else:
                        kernel("_scores", base + 285, duration=5)
                    if not selection_fused:
                        if shared and not prepared_ranking:
                            kernel("nosa_selection::prepare_prefix_ranking", base + 305, duration=2)
                        kernel(
                            "nosa_selection::selection_prefix_kernel"
                            if shared
                            else "nosa_selection::selection_kernel",
                            base + 325,
                            duration=5,
                        )
                elif scored and native and native_revision == 2:
                    if native_score:
                        stage("native_indexer", 280, 388)
                        kernel(
                            "void nosa_scores::fused_scores::fused_scores_kernel<bf16>()",
                            base + 285,
                            duration=6,
                        )
                    else:
                        stage("pooled_scores", 280, 300)
                        kernel("_scores", base + 285, duration=5)
                        stage("native_selection", 302, 388)
                    kernel("nosa_selection::prepare_prefix_ranking", base + 305, duration=2)
                    kernel("nosa_selection::selection_prefix_kernel", base + 325, duration=5)
                elif scored:
                    for name, kernel_name, start in (
                        (
                            "pooled_scores",
                            "void nosa_scores::scores_kernel<cutlass::gemm::Shape<16>>()"
                            if native_score
                            else "_scores",
                            280,
                        ),
                        ("topk_qa", "flashinfer_topk", 302),
                        ("prepare_cis", "_prepare_cis", 324),
                        ("topk_cis", "flashinfer_topk", 346),
                        ("finish_selection", "_finish_selection", 368),
                    ):
                        stage(name, start, start + 20)
                        if native_score and name == "pooled_scores":
                            kernel(
                                "void nosa_scores::normalizer_kernel<cutlass::bfloat16_t>()",
                                base + start + 3,
                                duration=1,
                            )
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
                        rows = min(query_tile, query_length - tile * query_tile)
                        native_score = native and position + query_length >= (
                            32768 if rows >= 1024 else 8192
                        )
                        start = 325 + tile * width
                        stage("compressed_scores", start, start + width / 3)
                        if native_score:
                            kernel(
                                "void nosa_scores::normalizer_kernel<cutlass::bfloat16_t>()",
                                base + start + width / 12,
                                duration=width / 20,
                            )
                        kernel(
                            "void nosa_scores::scores_kernel<cutlass::gemm::Shape<16>>()"
                            if native_score
                            else "_scores",
                            base + start + width / 6,
                            duration=width / 6,
                        )
                        stage("select_from_scores", start + width / 3, start + width)
                        kernel("_select", base + start + width / 2, duration=width / 6)
            stage("block_sparse_attention", 410, 480)
            grouped = native and native_attention_group_size == 4 and query_length >= 4
            if grouped:
                if native_revision >= 2:
                    kernel("nosa_attention::nonfinite_blocks_kernel", base + 415, duration=3)
                kernel(
                    "void nosa_attention::grouped_attention_kernel<cutlass::gemm::Shape<64>>()",
                    base + 420,
                    duration=37,
                )
            kernel(
                "void nosa_attention::attention_kernel<cutlass::gemm::Shape<16>>()"
                if native
                else "_nosa_block_attention",
                base + (460 if grouped else 430),
                duration=8 if grouped else 40,
            )
            kernel("nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT", base + 500)
            kernel("fused_add_rmsnorm", base + 520)
            kernel("nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT", base + 540, grid=(2, 66, 1))
            kernel("act_and_mul_kernel", base + 570)
            kernel("nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT", base + 600)
    return scopes, kernels, workload


@pytest.mark.parametrize("native_prepare", [False, True])
def test_native_joint_submission_splits_score_and_selection_without_losing_auxiliaries(
    native_prepare,
):
    scopes, kernels, workload = trace(
        prefix=31744,
        query_length=1024,
        cached=True,
        native=True,
        native_revision=2,
        native_prepare=native_prepare,
    )
    result = attribute_kernels(scopes, kernels, 1, workload)
    extend = next(run for run in result["runs"] if run["phase"] == "extend")
    modules = extend["modules"]
    assert modules["pooled_scores"]["kernel_count"] == 1
    assert modules["pooled_scores"]["kernel_ns"] == 6 * TIME_SCALE
    assert modules["native_selection"]["kernel_count"] == 2
    assert modules["native_selection"]["kernel_ns"] == 7 * TIME_SCALE
    assert modules["indexer_total"]["kernel_ns"] == (21 if native_prepare else 27) * TIME_SCALE
    assert modules["native_prepare"]["kernel_count"] == (2 if native_prepare else 0)
    assert modules["block_sparse_attention"]["kernel_count"] == 3
    assert modules["block_sparse_attention"]["kernel_ns"] == 48 * TIME_SCALE
    assert "native_indexer" not in modules
    assert (
        sum(
            modules[name]["kernel_ns"]
            for name in (
                "indexer_validate",
                "indexer_cache_update",
                "native_prepare",
                "pooled_scores",
                "native_selection",
            )
        )
        == modules["indexer_total"]["kernel_ns"]
    )


@pytest.mark.parametrize(
    "missing",
    ["prepare_prefix_ranking", "nonfinite_blocks_kernel", "finite_partials", "guarded_compression"],
)
def test_native_revision_requires_every_auxiliary_kernel(missing):
    scopes, kernels, workload = trace(
        prefix=31744,
        query_length=1024,
        cached=True,
        native=True,
        native_revision=2,
        native_prepare=True,
    )
    kernels = [kernel for kernel in kernels if missing not in kernel["name"]]
    with pytest.raises(ValueError, match="kernel sequence"):
        attribute_kernels(scopes, kernels, 1, workload)


@pytest.mark.parametrize("ranked", [False, True])
def test_fused_selection_owns_combined_duration_and_mixed_prefill_dispatch(ranked):
    scopes, kernels, workload = trace(
        prefix=65536,
        query_length=1024,
        cached=True,
        native=True,
        native_revision=3,
        native_prepare=True,
        native_ranked=ranked,
    )
    report = attribute_kernels(scopes, kernels, 1, workload)
    full, extend = report["runs"]
    assert [call["indexer_qk_module"] for call in full["layer_calls"]] == (
        [None] * 4 + ["pooled_scores"] * 27 + ["score_selection"] * 34
    )
    assert full["modules"]["score_selection"]["kernel_count"] == 34
    assert full["modules"]["pooled_scores"]["kernel_count"] == 27
    modules = extend["modules"]
    assert modules["pooled_scores"]["kernel_count"] == 0
    assert modules["score_selection"]["kernel_ns"] == 6 * TIME_SCALE
    assert modules["score_selection"]["inclusive"]
    assert modules["native_selection"]["kernel_count"] == (0 if ranked else 1)
    assert modules["indexer_total"]["kernel_ns"] == (14 if ranked else 16) * TIME_SCALE
    assert modules["block_sparse_attention"]["kernel_count"] == 3
    for run in report["runs"]:
        for call in run["layer_calls"]:
            children = [
                name
                for name, parent in PARENTS.items()
                if parent == "indexer_total" and name in call["modules"]
            ]
            child_ns = sum(call["modules"][name]["kernel_ns"] for name in children)
            assert child_ns <= call["modules"]["indexer_total"]["kernel_ns"]


@pytest.mark.parametrize("rows,kv_heads", [(128, 2), (512, 2), (1024, 1)])
def test_native_joint_short_rows_or_small_grid_keeps_separate_score_selection(rows, kv_heads):
    scopes, kernels, workload = trace(
        prefix=65536,
        query_length=rows,
        cached=True,
        native=True,
        native_revision=3,
        native_ranked=True,
        num_kv_heads=kv_heads,
    )
    report = attribute_kernels(scopes, kernels, 1, workload, num_kv_heads=kv_heads)
    extend = report["runs"][-1]
    assert extend["layer_calls"][0]["native_joint_submission"]
    assert extend["layer_calls"][0]["prepared_ranking"]
    assert extend["modules"]["pooled_scores"]["kernel_count"] == 2
    assert extend["modules"]["native_selection"]["kernel_count"] == 1
    assert extend["modules"]["score_selection"]["kernel_count"] == 0


@pytest.mark.parametrize("flag", ["true", "false", "1", "0"])
def test_fused_template_parser_handles_nested_type_arguments(flag):
    kernel = {
        "name": f"void {FUSED_SCORE_FAMILY}<Type<A, B>, Type<C, D>, true, {flag}, Map<X, Y>>()"
    }
    assert _fused_selection_flag(kernel) is (flag in ("true", "1"))


@pytest.mark.parametrize(
    "problem", ["wrong_flag", "missing_finite", "missing_guard", "extra_ranking", "extra_selector"]
)
def test_fused_prepared_sequence_rejects_wrong_or_extra_work(problem):
    scopes, kernels, workload = trace(
        prefix=65536,
        query_length=1024,
        cached=True,
        native=True,
        native_revision=3,
        native_ranked=True,
    )
    fused = next(kernel for kernel in reversed(kernels) if FUSED_SCORE_FAMILY in kernel["name"])
    if problem == "wrong_flag":
        fused["name"] = fused["name"].replace("true, true", "true, false")
    elif problem in ("missing_finite", "missing_guard"):
        family = (
            "finite_partials" if problem == "missing_finite" else "guarded_compression_and_ranking"
        )
        kernels = [kernel for kernel in kernels if family not in kernel["name"]]
    else:
        extra = dict(fused)
        extra["name"] = "nosa_selection::" + (
            "prepare_prefix_ranking" if problem == "extra_ranking" else "selection_prefix_kernel"
        )
        extra["correlationId"] = len(kernels) + 1
        extra["launch_start"] += TIME_SCALE
        kernels.append(extra)
    with pytest.raises(ValueError, match="kernel sequence"):
        attribute_kernels(scopes, kernels, 1, workload)


def test_fused_selection_cannot_be_classified_as_score_only():
    kernel = {"name": f"{FUSED_SCORE_FAMILY}<bf16, bf16, true, true, Map<1,2>>()"}
    for revision in (2, 3):
        with pytest.raises(ValueError, match="kernel sequence"):
            _validate_operator_kernels(
                "pooled_scores",
                [kernel],
                "cuda_tvm_ffi",
                query_length=1024,
                total_length=66560,
                native_revision=revision,
            )


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


def test_native_operator_gemms_are_owned_by_scopes_not_model_projections():
    scopes, kernels, workload = trace(prefix=31744, query_length=1024, cached=True, native=True)
    result = attribute_kernels(scopes, kernels, 1, workload)
    for run in result["runs"]:
        calls = 32 if run["phase"] == "full_prefill" else 1
        modules = run["modules"]
        assert modules["pooled_scores"]["kernel_count"] == (29 if calls == 32 else 2)
        assert run["score_dispatch_scope_counts"] == (
            {"triton": 27, "cuda_tvm_ffi": 1} if calls == 32 else {"cuda_tvm_ffi": 1}
        )
        assert len(run["layer_calls"]) == calls
        assert modules["block_sparse_attention"]["kernel_count"] == 2 * calls
        assert modules["block_sparse_attention"]["kernel_ns"] == calls * (37 + 8) * TIME_SCALE
        assert modules["qkv_proj"]["kernel_count"] == calls
        assert modules["cis_projection_gemm"]["kernel_count"] == calls
        assert (
            sum(row["kernel_count"] for name, row in modules.items() if name not in PARENTS)
            == run["kernel_count"]
        )


@pytest.mark.parametrize("problem", ["missing_grouped", "missing_fallback", "reversed"])
def test_grouped_attention_requires_both_launches_in_order(problem):
    scopes, kernels, workload = trace(cached=True, native=True)
    if problem == "missing_grouped":
        kernels = [
            kernel for kernel in kernels if "::grouped_attention_kernel" not in kernel["name"]
        ]
    elif problem == "missing_fallback":
        kernels = [kernel for kernel in kernels if "::attention_kernel" not in kernel["name"]]
    else:
        for kernel in kernels:
            kernel["name"] = (
                kernel["name"]
                .replace("::grouped_attention_kernel", "::TEMP_KERNEL")
                .replace("::attention_kernel", "::grouped_attention_kernel")
                .replace("::TEMP_KERNEL", "::attention_kernel")
            )
    with pytest.raises(ValueError, match="kernel sequence"):
        attribute_kernels(scopes, kernels, 1, workload)


@pytest.mark.parametrize("rows", [1, 3, 4, 1024])
@pytest.mark.parametrize("group_size", [1, 4])
def test_native_attention_launch_sequence_uses_reviewed_revision_and_query_length(rows, group_size):
    fallback = {"name": "void nosa_attention::attention_kernel<cutlass::bfloat16_t, Map>()"}
    grouped = {"name": "void nosa_attention::grouped_attention_kernel<cutlass::bfloat16_t, Map>()"}
    selected = [grouped, fallback] if group_size == 4 and rows >= 4 else [fallback]
    assert (
        _validate_operator_kernels(
            "block_sparse_attention",
            selected,
            "cuda_tvm_ffi",
            query_length=rows,
            total_length=65536 + rows,
            native_attention_group_size=group_size,
        )
        is None
    )
    opposite = [fallback] if len(selected) == 2 else [grouped, fallback]
    with pytest.raises(ValueError, match="kernel sequence"):
        _validate_operator_kernels(
            "block_sparse_attention",
            opposite,
            "cuda_tvm_ffi",
            query_length=rows,
            total_length=65536 + rows,
            native_attention_group_size=group_size,
        )


def test_legacy_native_attention_attribution_retains_one_launch_per_layer_call():
    scopes, kernels, workload = trace(cached=True, native=True, native_attention_group_size=1)
    result = attribute_kernels(scopes, kernels, 1, workload, native_attention_group_size=1)
    assert result["validation"]["native_attention_group_size"] == 1
    for run in result["runs"]:
        attention = run["modules"]["block_sparse_attention"]
        assert attention["kernel_count"] == len(run["layer_calls"])
        assert attention["kernel_ns"] == len(run["layer_calls"]) * 40 * TIME_SCALE


@pytest.mark.parametrize("stage", ["pooled_scores", "block_sparse_attention"])
@pytest.mark.parametrize(
    "problem", ["missing_kernel", "missing_scope", "fallback", "extra_kernel", "wrong_family"]
)
def test_native_operator_attribution_rejects_missing_or_unreviewed_work(stage, problem):
    scopes, kernels, workload = trace(prefix=31744, query_length=1024, cached=True, native=True)
    family = "nosa_scores::" if stage == "pooled_scores" else "nosa_attention::"
    selected = [kernel for kernel in kernels if family in kernel["name"]]
    if problem == "missing_kernel":
        kernels = [kernel for kernel in kernels if family not in kernel["name"]]
    elif problem == "missing_scope":
        scopes = [scope for scope in scopes if f"/{stage}/" not in scope["text"]]
    elif problem == "fallback":
        for kernel in selected:
            kernel["name"] = "_scores" if stage == "pooled_scores" else "_nosa_block_attention"
    elif problem == "wrong_family":
        for kernel in selected:
            kernel["name"] = kernel["name"].replace("kernel<", "kernel_helper<")
    else:
        extra = deepcopy(selected[0])
        extra.update(name="unreviewed_helper", correlationId=100_000)
        kernels.append(extra)
    with pytest.raises(ValueError):
        attribute_kernels(scopes, kernels, 1, workload)


def test_native_scores_require_normalizer_before_score_output():
    scopes, kernels, workload = trace(prefix=31744, query_length=1024, cached=True, native=True)
    for kernel in kernels:
        kernel["name"] = (
            kernel["name"]
            .replace("normalizer_kernel", "TEMP_KERNEL")
            .replace("scores_kernel", "normalizer_kernel")
            .replace("TEMP_KERNEL", "scores_kernel")
        )
    with pytest.raises(ValueError, match="kernel sequence"):
        attribute_kernels(scopes, kernels, 1, workload)


@pytest.mark.parametrize("stage", ["pooled_scores", "compressed_scores"])
@pytest.mark.parametrize(
    ("rows", "total", "expected_backend"),
    [
        (1024, 32767, "triton"),
        (1024, 32768, "cuda_tvm_ffi"),
        (1023, 8191, "triton"),
        (1023, 8192, "cuda_tvm_ffi"),
        (64, 8191, "triton"),
        (64, 8192, "cuda_tvm_ffi"),
    ],
)
def test_native_score_dispatch_requires_exact_kernel_branch_at_key_thresholds(
    stage, rows, total, expected_backend
):
    branches = {
        "triton": [{"name": "_scores"}],
        "cuda_tvm_ffi": [
            {"name": "void nosa_scores::normalizer_kernel<cutlass::bfloat16_t, Map>()"},
            {"name": "void nosa_scores::scores_kernel<cutlass::bfloat16_t, Map>()"},
        ],
    }
    assert (
        _validate_operator_kernels(
            stage,
            branches[expected_backend],
            "cuda_tvm_ffi",
            query_length=rows,
            total_length=total,
        )
        == expected_backend
    )
    opposite = "triton" if expected_backend == "cuda_tvm_ffi" else "cuda_tvm_ffi"
    with pytest.raises(ValueError, match="kernel sequence"):
        _validate_operator_kernels(
            stage,
            branches[opposite],
            "cuda_tvm_ffi",
            query_length=rows,
            total_length=total,
        )


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


@pytest.mark.parametrize("problem", [None, "native_build", "header"])
def test_module_input_validation_checks_native_build_and_snapshot_contents(
    tmp_path, monkeypatch, problem
):
    from experiments.indexer_block_sparse_profile.src import module_mfu
    from experiments.indexer_block_sparse_profile.tests.test_sparse_profile_mfu import native_inputs

    metadata, summary = native_inputs()
    metadata["args"].update(profile_repeats=1, device="cuda:0")
    hashes = {}
    for name in metadata["source_sha256"]:
        path = tmp_path / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source snapshot for {name}\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["source_sha256"] = dict(hashes)
    metadata["native_build"]["source_sha256"] = dict(hashes)
    # The fixture graph is fully hashed too; this avoids blessing an evolving
    # production native graph before its final implementation is reviewed.
    monkeypatch.setattr(module_mfu, "SUPPORTED_GRAPHS", (hashes,))
    profile = deepcopy(metadata)
    profile["args"]["mode"] = "profile"
    profile["validation"] = {"finite": True}
    audit = [
        {
            "stage": "block_sparse_attention",
            "layer_idx": layer,
            "status": "ok",
            "phase": "extend",
            "query_start": 65536,
            "query_length": 1024,
            "details": {
                "q_shape": [1024, 32, 128],
                "k_shape": [66560, 2, 128],
                "v_shape": [66560, 2, 128],
                "selection_shape": [1024, 2, 64],
                "block_size": 64,
                "block_budget": 64,
                "valid_blocks_min": 64,
                "valid_blocks_max": 64,
            },
        }
        for layer in range(32)
    ]
    (tmp_path / "attention_audit.json").write_text(json.dumps(audit))
    if problem == "native_build":
        profile["native_build"]["compiler"]["version"] = "different compiler"
    elif problem == "header":
        (tmp_path / "sources/operators/sm90/csrc/detail/pipeline.cuh").write_text("changed\n")
    if problem is None:
        assert _validate_inputs(tmp_path, metadata, summary, profile) == hashes
    else:
        with pytest.raises(ValueError, match="native_build|snapshot hash"):
            _validate_inputs(tmp_path, metadata, summary, profile)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_module_report_counts_useful_matrix_work_once_and_accepts_absent_stages(cached, native):
    scopes, kernels, workload = trace(prefix=65536, query_length=1024, cached=cached, native=native)
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
    if native:
        from experiments.indexer_block_sparse_profile.tests.test_sparse_profile_mfu import (
            native_inputs,
        )

        native_metadata, _ = native_inputs()
        for key in ("source_sha256", "tvm_ffi", "native_build"):
            metadata[key] = native_metadata[key]
        metadata["args"]["kernel_backend"] = "native"
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
    assert report["implementation"]["kernel_backend"] == ("cuda_tvm_ffi" if native else "triton")
    assert report["implementation"]["native_build"] == metadata.get("native_build")
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
        attention = modules["block_sparse_attention"]
        calls = 65 if phase == "full_prefill" else 1
        assert len(run["layer_calls"]) == calls
        assert attention["kernel_count_per_run"] == [calls * (2 if native else 1)]
        kernel_ms = calls * (45 if native else 40) * TIME_SCALE / 1e6
        assert attention["kernel_ms_median"] == pytest.approx(kernel_ms)
        assert attention["matrix_flops_per_run"] == expected["block_sparse_attention"]
        assert attention["mfu_pct_median"] == pytest.approx(
            expected["block_sparse_attention"] / kernel_ms / 1e9 * 100
        )
        for name in (active, absent, "indexer_total", "block_sparse_attention"):
            assert (
                sum(call["modules"][name]["matrix_flops"] for call in run["layer_calls"])
                == (expected[name])
            )


def fused_module_report_fixture():
    from experiments.indexer_block_sparse_profile.src.mfu import (
        FUSED_NATIVE_SOURCES,
        PRUNED_NATIVE_SOURCES,
    )
    from experiments.indexer_block_sparse_profile.tests.test_sparse_profile_mfu import native_inputs

    scopes, kernels, workload = trace(
        prefix=65536,
        query_length=1024,
        cached=True,
        native=True,
        native_revision=3,
        native_ranked=True,
    )
    workload.update(POLICY, repeats=1, profile_repeats=1)
    metadata, _ = native_inputs()
    hashes = {name: "c" * 64 for name in FUSED_NATIVE_SOURCES | PRUNED_NATIVE_SOURCES}
    metadata["source_sha256"].update(hashes)
    metadata["native_build"]["source_sha256"].update(hashes)
    metadata.update(
        run_id="synthetic-fused",
        model_config={
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
        },
        args=workload | {"mode": "benchmark", "kernel_backend": "native"},
    )
    summary = {
        "schema_version": 1,
        "run_id": metadata["run_id"],
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
    profiles = {"instrumented_timings": {phase: [{}] for phase in summary["profiles"]}}
    return metadata, summary, profiles, attribution


def test_mixed_dispatch_flops_are_assigned_per_call_before_phase_aggregation():
    metadata, summary, profiles, attribution = fused_module_report_fixture()
    report = build_module_report(metadata, summary, profiles, attribution)
    config = metadata["model_config"]
    for run in report["runs"]:
        modules = run["modules"]
        expected = {name: 0 for name in QK_MODULES}
        for call in run["layer_calls"]:
            qk = _flops_by_module(config, call["query_start"], call["query_length"], 1024)[
                "indexer_total"
            ]
            if owner := call["indexer_qk_module"]:
                expected[owner] += qk
            else:
                assert qk == 0
            assert sum(call["modules"][name]["matrix_flops"] for name in QK_MODULES) == qk
        assert sum(expected.values()) == modules["indexer_total"]["matrix_flops"]
        for name in QK_MODULES:
            assert modules[name]["matrix_flops"] == expected[name]
        for name in ("native_prepare", "native_prepare_ranked", "native_selection"):
            assert modules[name]["matrix_flops"] == 0
            assert modules[name]["mfu_pct"] is None
        fused = report["phases"][run["phase"]]["modules"]["score_selection"]
        assert fused["matrix_flops_per_run"] == expected["score_selection"] > 0
        assert "event_interval_ms_median" not in fused
        assert fused["mfu_pct_median"] == pytest.approx(
            expected["score_selection"]
            / fused["kernel_ms_median"]
            / 1e9
            / report["peak_tflops"]
            * 100
        )
    full = report["phases"]["full_prefill"]["modules"]
    assert (
        0
        < full["pooled_scores"]["matrix_flops_per_run"]
        < full["score_selection"]["matrix_flops_per_run"]
    )
    extend = report["phases"]["extend"]["modules"]
    assert extend["pooled_scores"]["matrix_flops_per_run"] == 0


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_owner"])
def test_module_report_rejects_ambiguous_per_call_qk_ownership(problem):
    metadata, summary, profiles, attribution = fused_module_report_fixture()
    call = attribution["runs"][-1]["layer_calls"][0]
    if problem == "missing":
        call["modules"]["score_selection"]["kernel_count"] = 0
    elif problem == "duplicate":
        call["modules"]["pooled_scores"]["kernel_count"] = 1
    else:
        call["indexer_qk_module"] = "pooled_scores"
    with pytest.raises(ValueError, match="QK"):
        build_module_report(metadata, summary, profiles, attribution)


def test_module_report_rejects_inconsistent_qk_ownership_between_repeats():
    metadata, summary, profiles, attribution = fused_module_report_fixture()
    summary["workload"]["profile_repeats"] = 2
    for phase in summary["profiles"]:
        summary["profiles"][phase]["sample_count"] = 2
        profiles["instrumented_timings"][phase] *= 2
    repeat = deepcopy(attribution["runs"])
    for run in repeat:
        run["iteration"] = 1
    call = repeat[-1]["layer_calls"][0]
    call["modules"]["pooled_scores"] = call["modules"]["score_selection"]
    call["modules"]["score_selection"] = {"kernel_count": 0, "kernel_ms": 0}
    call["indexer_qk_module"] = "pooled_scores"
    repeat[-1]["modules"]["pooled_scores"] = repeat[-1]["modules"]["score_selection"]
    repeat[-1]["modules"]["score_selection"] = {"kernel_count": 0, "kernel_ms": 0}
    attribution["runs"].extend(repeat)
    with pytest.raises(ValueError, match="Repeated profile runs"):
        build_module_report(metadata, summary, profiles, attribution)


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


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "missing_mapping",
        "wrong_process",
        "duplicate_mapping",
        "missing_process",
        "duplicate_process",
        "missing_process_table",
        "missing_global_pid",
        "unknown_physical_gpu",
        "wrong_physical_gpu",
        "wrong_mapping_uuid",
        "wrong_mapping_sm_count",
        "wrong_memory",
        "duplicate_physical_gpu",
    ],
)
def test_sqlite_process_local_cuda_ordinal_maps_to_the_verified_physical_gpu(tmp_path, problem):
    path = tmp_path / "trace.sqlite"
    global_pid = 47 << 24
    gpu = {
        "uuid": "GPU-B",
        "sm_count": 132,
        "capability": [9, 0],
        "total_memory": 150121545728,
    }
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE TARGET_INFO_GPU (
                id INTEGER, uuid TEXT, name TEXT, smCount INTEGER, computeMajor INTEGER,
                computeMinor INTEGER, totalMemory INTEGER, busLocation TEXT
            );
            INSERT INTO TARGET_INFO_GPU VALUES
                (0,'gpu-a','same model',132,9,0,150121545728,'0000:63:00.0'),
                (1,'gpu-b','same model',132,9,0,150121545728,'0000:67:00.0');
            CREATE TABLE PROCESSES (globalPid INTEGER, pid INTEGER);
            CREATE TABLE TARGET_INFO_CUDA_DEVICE (
                gpuId INTEGER, cudaId INTEGER, pid INTEGER, uuid TEXT, numMultiprocessors INTEGER
            );
            INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES
                (1,0,47,NULL,132), (0,0,48,'GPU-A',132);
            """
        )
        connection.executemany(
            "INSERT INTO PROCESSES VALUES (?,?)", [(global_pid, 47), (48 << 24, 48)]
        )
        if problem == "missing_mapping":
            connection.execute("DELETE FROM TARGET_INFO_CUDA_DEVICE WHERE pid=47")
        elif problem == "wrong_process":
            global_pid = 48 << 24
        elif problem == "duplicate_mapping":
            connection.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (1,0,47,NULL,132)")
        elif problem == "missing_process":
            connection.execute("DELETE FROM PROCESSES WHERE pid=47")
        elif problem == "duplicate_process":
            connection.execute("INSERT INTO PROCESSES VALUES (?,47)", (global_pid,))
        elif problem == "missing_process_table":
            connection.execute("DROP TABLE PROCESSES")
        elif problem == "missing_global_pid":
            global_pid = None
        elif problem in ("unknown_physical_gpu", "wrong_physical_gpu"):
            physical = 9 if problem == "unknown_physical_gpu" else 0
            connection.execute(
                "UPDATE TARGET_INFO_CUDA_DEVICE SET gpuId=? WHERE pid=47", (physical,)
            )
        elif problem == "wrong_mapping_uuid":
            connection.execute("UPDATE TARGET_INFO_CUDA_DEVICE SET uuid='GPU-A' WHERE pid=47")
        elif problem == "wrong_mapping_sm_count":
            connection.execute(
                "UPDATE TARGET_INFO_CUDA_DEVICE SET numMultiprocessors=100 WHERE pid=47"
            )
        elif problem == "wrong_memory":
            gpu["total_memory"] = 1
        elif problem == "duplicate_physical_gpu":
            connection.execute(
                "INSERT INTO TARGET_INFO_GPU SELECT * FROM TARGET_INFO_GPU WHERE id=1"
            )
    if problem:
        with pytest.raises(ValueError):
            validate_trace_gpu(path, 0, gpu, global_pid=global_pid)
    else:
        device = validate_trace_gpu(path, 0, gpu, global_pid=global_pid)
        assert device["id"] == 1
        assert device["kernel_device_id"] == 0
        assert device["busLocation"] == "0000:67:00.0"
        assert device["cuda_device_mapping"] == {
            "gpuId": 1,
            "cudaId": 0,
            "pid": 47,
            "uuid": None,
            "numMultiprocessors": 132,
            "globalPid": global_pid,
        }
