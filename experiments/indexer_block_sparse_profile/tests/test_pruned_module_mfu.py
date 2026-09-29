"""Pruning keeps the complete fused duration and unchanged useful QK ownership."""

from copy import deepcopy

import pytest

from experiments.indexer_block_sparse_profile.src.module_mfu import (
    FUSED_SCORE_FAMILY,
    PRUNED_SCORE_FAMILY,
    _validate_operator_kernels,
    attribute_kernels,
)
from experiments.indexer_block_sparse_profile.tests.test_checked_module_mfu import checked_trace

PRUNED = f"{PRUNED_SCORE_FAMILY}<cutlass::bfloat16_t, cutlass::bfloat16_t, true, true, Map>"


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("resource_fallback", [False, True])
@pytest.mark.parametrize("bool_cast", [False, True])
def test_pruned_dispatch_requires_complete_rank_and_fused_sequence(
    prepared, resource_fallback, bool_cast
):
    names = [] if prepared else ["nosa_selection::prepare_prefix_ranking"]
    names.append(
        PRUNED.replace(PRUNED_SCORE_FAMILY, FUSED_SCORE_FAMILY) if resource_fallback else PRUNED
    )
    if bool_cast:
        names[-1] = names[-1].replace("true", "(bool)1")
    assert (
        _validate_operator_kernels(
            "native_indexer",
            [{"name": name} for name in names],
            "cuda_tvm_ffi",
            query_length=1024,
            total_length=66560,
            native_revision=5,
            prepared_ranking=prepared,
            num_kv_heads=2,
        )
        == "cuda_tvm_ffi"
    )


@pytest.mark.parametrize(
    "problem", ["revision", "rows", "length", "heads", "score_only", "selection_flag", "ranking"]
)
def test_pruned_kernel_cannot_be_attributed_to_an_unreviewed_path(problem):
    options = {"query_length": 1024, "total_length": 66560, "native_revision": 5, "num_kv_heads": 2}
    stage, name, prepared = "native_indexer", PRUNED, True
    if problem == "revision":
        options["native_revision"] = 4
    elif problem == "rows":
        options["query_length"] = 1025
    elif problem == "length":
        options["total_length"] = 65536
    elif problem == "heads":
        options["num_kv_heads"] = 3
    elif problem == "score_only":
        stage = "pooled_scores"
    elif problem == "selection_flag":
        name = name.replace("true, true", "true, false")
    else:
        prepared = False
    with pytest.raises(ValueError, match="kernel sequence"):
        _validate_operator_kernels(
            stage,
            [{"name": name}],
            "cuda_tvm_ffi",
            prepared_ranking=prepared,
            **options,
        )


def test_pruned_checked_scope_preserves_all_kernel_time_in_score_selection_and_indexer():
    scopes, kernels, workload = checked_trace()
    baseline = attribute_kernels(scopes, kernels, 1, workload)
    changed = deepcopy(kernels)
    scope = next(scope for scope in reversed(scopes) if "/native_checked_indexer/" in scope["text"])
    replaced = 0
    for kernel in changed:
        if (
            scope["start"] <= kernel["launch_start"] < scope["end"]
            and FUSED_SCORE_FAMILY in kernel["name"]
        ):
            kernel["name"] = kernel["name"].replace(FUSED_SCORE_FAMILY, PRUNED_SCORE_FAMILY)
            replaced += 1
    assert replaced == 1
    actual = attribute_kernels(scopes, changed, 1, workload)
    assert actual["runs"] == baseline["runs"]
