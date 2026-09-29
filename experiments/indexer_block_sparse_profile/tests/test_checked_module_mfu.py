"""A checked host submission is a container, not a combined GPU kernel."""

from copy import deepcopy

import pytest

from experiments.indexer_block_sparse_profile.src.capture import operator_workload
from experiments.indexer_block_sparse_profile.src.module_mfu import (
    FUSED_SCORE_FAMILY,
    PARENTS,
    QK_MODULES,
    attribute_kernels,
    build_module_report,
)
from experiments.indexer_block_sparse_profile.tests.test_sparse_module_mfu import (
    TIME_SCALE,
    fused_module_report_fixture,
    trace,
)


def checked_trace(rows=1024, heads=2):
    scopes, kernels, workload = trace(
        prefix=65536,
        query_length=rows,
        cached=True,
        native=True,
        native_revision=3,
        native_ranked=True,
        num_kv_heads=heads,
    )
    workload.update(operator_workload("native"))
    workload.pop("attention_execution")
    removed = []
    for scope in scopes:
        if "/native_indexer/" not in scope["text"]:
            continue
        prepare = next(
            other
            for other in scopes
            if other["text"] == scope["text"].replace("/native_indexer/", "/native_prepare_ranked/")
        )
        scope["start"] = prepare["start"]
        scope["text"] = scope["text"].replace("/native_indexer/", "/native_checked_indexer/")
        removed.append(prepare)
    return [scope for scope in scopes if scope not in removed], kernels, workload


@pytest.mark.parametrize("rows,heads", [(128, 2), (512, 2), (1024, 2), (1024, 1)])
def test_checked_container_partitions_finite_preparation_and_actual_qk(rows, heads):
    scopes, kernels, workload = checked_trace(rows, heads)
    report = attribute_kernels(scopes, kernels, 1, workload, num_kv_heads=heads)
    extend = report["runs"][-1]
    modules = extend["modules"]
    assert modules["native_finite_check"]["kernel_ns"] == 2 * TIME_SCALE
    assert modules["native_ranked_compression"]["kernel_ns"] == 6 * TIME_SCALE
    fused = rows == 1024 and heads == 2
    assert modules["score_selection"]["kernel_count"] == int(fused)
    assert modules["pooled_scores"]["kernel_count"] == (0 if fused else 2)
    assert modules["native_selection"]["kernel_count"] == (0 if fused else 1)
    assert modules["native_prepare_ranked"]["kernel_count"] == 0
    assert "native_checked_indexer" not in modules
    assert (
        sum(
            row["kernel_ns"]
            for name, row in modules.items()
            if PARENTS.get(name) == "indexer_total"
        )
        == modules["indexer_total"]["kernel_ns"]
    )
    assert modules["block_sparse_attention"]["kernel_ns"] == 48 * TIME_SCALE
    if fused:
        full = report["runs"][0]
        assert [call["indexer_qk_module"] for call in full["layer_calls"]] == [None] * 4 + [
            "pooled_scores"
        ] * 27 + ["score_selection"] * 34
        assert full["modules"]["native_finite_check"]["kernel_count"] == 34
        assert full["modules"]["native_prepare_ranked"]["kernel_count"] == 27 * 2


@pytest.mark.parametrize("problem", ["finite", "guard", "order", "extra_rank", "wrong_fusion"])
def test_checked_container_rejects_missing_or_misclassified_kernel_work(problem):
    scopes, kernels, workload = checked_trace()
    scope = next(scope for scope in reversed(scopes) if "/native_checked_indexer/" in scope["text"])
    selected = [
        kernel for kernel in kernels if scope["start"] <= kernel["launch_start"] < scope["end"]
    ]
    if problem in ("finite", "guard"):
        kernels.remove(selected[0 if problem == "finite" else 1])
    elif problem == "order":
        selected[0]["name"], selected[1]["name"] = selected[1]["name"], selected[0]["name"]
    elif problem == "wrong_fusion":
        assert FUSED_SCORE_FAMILY in selected[2]["name"]
        selected[2]["name"] = selected[2]["name"].replace("true, true", "true, false")
    else:
        extra = deepcopy(selected[2])
        extra["name"] = "nosa_selection::prepare_prefix_ranking"
        extra["launch_start"] -= TIME_SCALE
        extra["correlationId"] = len(kernels) + 1
        kernels.append(extra)
    with pytest.raises(ValueError, match="kernel sequence"):
        attribute_kernels(scopes, kernels, 1, workload)


def test_checked_mfu_preserves_complete_work_and_never_gives_preparation_qk_flops():
    metadata, summary, profile, _ = fused_module_report_fixture()
    scopes, kernels, workload = checked_trace()
    summary["workload"].update(workload)
    metadata["args"].update(workload | {"kernel_backend": "native"})
    source = "operators/sm90/csrc/nosa_indexer_checked.cu"
    metadata["source_sha256"][source] = "e" * 64
    metadata["native_build"]["source_sha256"][source] = "e" * 64
    attribution = attribute_kernels(scopes, kernels, 1, summary["workload"])
    for phase in summary["profiles"]:
        names = {
            scope["text"].split("/")[3]
            for scope in scopes
            if scope["text"].startswith(f"NOSA/{phase}/")
        }
        summary["profiles"][phase]["stage_totals"] = {
            name: {"cuda_elapsed_ms": {"median": 1.0}} for name in names
        }
    report = build_module_report(metadata, summary, profile, attribution)
    for phase in report["phases"].values():
        modules = phase["modules"]
        assert (
            sum(modules[name]["matrix_flops_per_run"] for name in QK_MODULES)
            == modules["indexer_total"]["matrix_flops_per_run"]
        )
        for name in ("native_finite_check", "native_ranked_compression"):
            assert modules[name]["matrix_flops_per_run"] == 0
            assert modules[name]["mfu_pct_median"] is None
            assert "event_interval_ms_median" not in modules[name]
        assert "event_interval_ms_median" not in modules["score_selection"]
