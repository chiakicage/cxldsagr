"""FA3 v3 ordering is declared and charged only for the reviewed geometry."""

from copy import deepcopy

import pytest

from experiments.indexer_block_sparse_profile.src.mfu import validate_kernel_backend
from experiments.indexer_block_sparse_profile.src.module_mfu import (
    _validate_operator_kernels,
    attribute_kernels,
    build_module_report,
)
from experiments.indexer_block_sparse_profile.tests.test_fa3_module_mfu import (
    FA3_NAMES,
    add_fa3_build,
    fa3_trace,
)
from experiments.indexer_block_sparse_profile.tests.test_sparse_module_mfu import (
    TIME_SCALE,
    fused_module_report_fixture,
)
from experiments.indexer_block_sparse_profile.tests.test_sparse_profile_mfu import native_inputs

SORT = "nosa_fa3::sort_work_by_union_size"
SORTED_NAMES = (FA3_NAMES[0], SORT, *FA3_NAMES[1:])


def validate(names, *, rows=1024, heads=2, version=3):
    return _validate_operator_kernels(
        "block_sparse_attention",
        [{"name": name} for name in names],
        "cuda_tvm_ffi",
        query_length=rows,
        total_length=65536 + rows,
        num_kv_heads=heads,
        native_revision=5,
        attention_execution=f"native_fa3_v{version}",
    )


@pytest.mark.parametrize(
    "rows,heads,sorted_work",
    [
        (1, 2, False),
        (8, 2, False),
        (1016, 2, False),
        (1017, 2, True),
        (1021, 2, True),
        (1024, 2, True),
        (1025, 2, False),
        (1024, 1, False),
        (2041, 1, True),
        (512, 4, True),
        (256, 8, True),
    ],
)
def test_v3_requires_exact_sort_presence_from_query_and_kv_head_geometry(rows, heads, sorted_work):
    expected, rejected = (SORTED_NAMES, FA3_NAMES) if sorted_work else (FA3_NAMES, SORTED_NAMES)
    validate(expected, rows=rows, heads=heads)
    with pytest.raises(ValueError, match="kernel sequence"):
        validate(rejected, rows=rows, heads=heads)


@pytest.mark.parametrize("rows", [1, 1017, 1024, 1025])
def test_v3_keeps_previously_reviewed_native_layout_fallback_alternatives(rows):
    names = (
        ("nosa_attention::attention_kernel",)
        if rows < 4
        else (
            "nosa_attention::nonfinite_blocks_kernel",
            "nosa_attention::grouped_attention_kernel",
            "nosa_attention::attention_kernel",
        )
    )
    validate(names, rows=rows)
    with pytest.raises(ValueError, match="kernel sequence"):
        validate((SORT, *names), rows=rows)


def test_v2_does_not_accept_the_new_sorter_and_empty_v3_launches_nothing():
    validate(FA3_NAMES, version=2)
    with pytest.raises(ValueError, match="kernel sequence"):
        validate(SORTED_NAMES, version=2)
    validate((), rows=0)
    for names in (FA3_NAMES, SORTED_NAMES, (SORT,)):
        with pytest.raises(ValueError, match="kernel sequence"):
            validate(names, rows=0)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_prepare",
        "missing_main",
        "missing_repair",
        "duplicate_sort",
        "order",
        "foreign_sort",
    ],
)
def test_v3_rejects_partial_reordered_or_misidentified_four_kernel_scopes(problem):
    names = list(SORTED_NAMES)
    if problem.startswith("missing_"):
        del names[{"missing_prepare": 0, "missing_main": 2, "missing_repair": 3}[problem]]
    elif problem == "duplicate_sort":
        names.insert(2, SORT)
    elif problem == "order":
        names[0], names[1] = names[1], names[0]
    else:
        names[1] = SORT + "_unreviewed"
    with pytest.raises(ValueError, match="kernel sequence"):
        validate(names)


@pytest.mark.parametrize("heads", [0, -1, 2.0, True])
def test_v3_rejects_invalid_sort_geometry_metadata(heads):
    with pytest.raises(ValueError, match="num_kv_heads"):
        validate(SORTED_NAMES, heads=heads)


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_v3_runtime_contract_is_recorded_for_native_and_matching_triton_control(backend):
    metadata, summary = native_inputs(backend=backend)
    add_fa3_build(metadata, version=3)
    summary["workload"]["attention_execution"] = (
        "native_fa3_v3" if backend == "native" else "triton_v1"
    )
    assert validate_kernel_backend(summary["workload"], metadata) == (
        "cuda_tvm_ffi" if backend == "native" else "triton"
    )


@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize(
    "problem",
    [
        "ordering",
        "items",
        "float_items",
        "ties",
        "missing_ordering",
        "missing_items",
        "missing_ties",
    ],
)
def test_v3_rejects_missing_or_unreviewed_runtime_ordering_fields(backend, problem):
    metadata, summary = native_inputs(backend=backend)
    add_fa3_build(metadata, version=3)
    summary["workload"]["attention_execution"] = (
        "native_fa3_v3" if backend == "native" else "triton_v1"
    )
    fa3 = metadata["native_build"]["attention_fa3"]
    if problem.startswith("missing_"):
        del fa3[
            {
                "missing_ordering": "cta_order",
                "missing_items": "cta_order_work_items",
                "missing_ties": "cta_order_ties",
            }[problem]
        ]
    else:
        key, value = {
            "ordering": ("cta_order", "ascending_union_tiles"),
            "items": ("cta_order_work_items", 128),
            "float_items": ("cta_order_work_items", 256.0),
            "ties": ("cta_order_ties", "descending_logical_batch"),
        }[problem]
        fa3[key] = value
    with pytest.raises(ValueError, match="ordering contract"):
        validate_kernel_backend(summary["workload"], metadata)


def test_v3_cannot_use_a_v2_build_or_disguise_the_new_build_as_v2():
    metadata, summary = native_inputs()
    add_fa3_build(metadata)
    summary["workload"]["attention_execution"] = "native_fa3_v3"
    with pytest.raises(ValueError, match="ordering contract"):
        validate_kernel_backend(summary["workload"], metadata)
    add_fa3_build(metadata, version=3)
    summary["workload"]["attention_execution"] = "native_fa3_v2"
    with pytest.raises(ValueError, match="v2 cannot"):
        validate_kernel_backend(summary["workload"], metadata)


def test_v3_attributes_sorter_to_complete_attention_without_extra_matrix_work():
    metadata, summary, profile, _ = fused_module_report_fixture()
    scopes, kernels, workload = fa3_trace(version=3)
    summary["workload"].update(workload)
    metadata["args"].update(workload | {"kernel_backend": "native"})
    source = "operators/sm90/csrc/nosa_indexer_checked.cu"
    metadata["source_sha256"][source] = "e" * 64
    metadata["native_build"]["source_sha256"][source] = "e" * 64
    add_fa3_build(metadata, version=3)
    attribution = attribute_kernels(scopes, kernels, 1, summary["workload"])
    for run in attribution["runs"]:
        count = len(run["layer_calls"])
        assert run["attention_dispatch_scope_counts"] == {"native_fa3_v3": count}
        attention = run["modules"]["block_sparse_attention"]
        assert attention["kernel_count"] == 4 * count
        assert attention["kernel_ns"] == 47 * TIME_SCALE * count
        assert all(call["attention_execution"] == "native_fa3_v3" for call in run["layer_calls"])
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
    module = report["phases"]["extend"]["modules"]["block_sparse_attention"]
    assert module["kernel_ms_median"] == 47 * TIME_SCALE / 1e6
    assert module["kernel_count_per_run"] == [4]
    baseline_metadata, baseline_summary = deepcopy(metadata), deepcopy(summary)
    baseline_summary["workload"]["attention_execution"] = "native_fa3_v2"
    add_fa3_build(baseline_metadata)
    base_scopes, base_kernels, base_workload = fa3_trace()
    baseline_attribution = attribute_kernels(base_scopes, base_kernels, 1, base_workload)
    baseline_report = build_module_report(
        baseline_metadata, baseline_summary, profile, baseline_attribution
    )
    baseline_module = baseline_report["phases"]["extend"]["modules"]["block_sparse_attention"]
    assert module["matrix_flops_per_run"] == baseline_module["matrix_flops_per_run"]
    assert module["mfu_pct_median"] == pytest.approx(
        module["matrix_flops_per_run"]
        / module["kernel_ms_median"]
        / 1e9
        / report["peak_tflops"]
        * 100
    )
