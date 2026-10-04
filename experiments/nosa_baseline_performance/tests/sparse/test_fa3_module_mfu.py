"""FA3 v2 attribution includes preparation, direct-output attention and repair."""

from copy import deepcopy

import pytest

from experiments.nosa_baseline_performance.src.sparse.mfu import (
    FA3_EXTRA_FLAGS,
    FA3_SOURCE_PATHS,
    validate_kernel_backend,
)
from experiments.nosa_baseline_performance.src.sparse.module_mfu import (
    FA3_ATTENTION_SEQUENCE,
    _validate_operator_kernels,
    attribute_kernels,
    build_module_report,
)
from experiments.nosa_baseline_performance.tests.sparse.test_checked_module_mfu import checked_trace
from experiments.nosa_baseline_performance.tests.sparse.test_sparse_module_mfu import (
    TIME_SCALE,
    fused_module_report_fixture,
)
from experiments.nosa_baseline_performance.tests.sparse.test_sparse_profile_mfu import native_inputs

FA3_MAIN = (
    "void flashinfer::PrefillWithKVCacheKernel<nosa_fa3::ML, "
    "nosa_fa3::EP, nosa_fa3::KT, 0, 0, "
    "nosa_fa3::Scheduler, 0>(T1::Params, T2::Params, T6::Params)"
)
FA3_NAMES = (FA3_ATTENTION_SEQUENCE[0], FA3_MAIN, FA3_ATTENTION_SEQUENCE[2])


def add_fa3_build(metadata, *, version=2):
    build = metadata["native_build"]
    sources = {}
    for name, path in FA3_SOURCE_PATHS.items():
        digest = metadata["source_sha256"].setdefault(path, "f" * 64)
        if path.startswith("operators/sm90/csrc/"):
            build["source_sha256"][path] = digest
        sources[name] = digest
    metadata.update(torch="fixture-torch-version", flashinfer="0.6.18")
    build["attention_fa3"] = {
        "toolchain": deepcopy(
            {name: build[name] for name in ("tvm_ffi", "compiler", "cutlass", "cuda_flags")}
        ),
        "torch": metadata["torch"],
        "flashinfer": metadata["flashinfer"],
        "flashinfer_include": "/installed/flashinfer/data/include",
        "flashinfer_headers_sha256": "b" * 64,
        "cuda_flags": build["cuda_flags"] + FA3_EXTRA_FLAGS,
        "source_sha256": sources,
        "group_queries": 8,
        "kv_tile_tokens": 128,
        "stages": 2,
        "minimum_fa3_queries": 1,
        "q_transfer": "direct_strided_tma",
        "output_store": "direct",
        "numerical_repair": "nonfinite_output_postcheck",
        "native_pv_accumulation": "bf16_power_of_two_scale_finite_output_guard",
    }
    if version == 3:
        build["attention_fa3"].update(
            cta_order="descending_union_tiles",
            cta_order_work_items=256,
            cta_order_ties="ascending_logical_batch",
        )


def fa3_trace(*, version=2):
    scopes, kernels, workload = checked_trace()
    workload["attention_execution"] = f"native_fa3_v{version}"
    names, durations = FA3_NAMES, (2, 38, 4)
    if version == 3:
        names = (FA3_NAMES[0], "nosa_fa3::sort_work_by_union_size", *FA3_NAMES[1:])
        durations = (2, 3, 38, 4)
    for scope in scopes:
        if "/block_sparse_attention/" not in scope["text"]:
            continue
        selected = [k for k in kernels if scope["start"] <= k["launch_start"] < scope["end"]]
        for kernel in selected:
            kernels.remove(kernel)
        launch = scope["start"] + TIME_SCALE
        gpu_start = selected[0]["start"]
        for index, (name, duration) in enumerate(zip(names, durations, strict=True)):
            kernel = deepcopy(selected[0])
            kernel.update(
                name=name,
                start=gpu_start,
                end=gpu_start + duration * TIME_SCALE,
                launch_start=launch + index * TIME_SCALE,
                correlationId=max(k["correlationId"] for k in kernels) + 1,
            )
            kernels.append(kernel)
            gpu_start = kernel["end"]
    return scopes, kernels, workload


def test_fa3_complete_attention_time_and_per_call_dispatch_are_preserved():
    scopes, kernels, workload = fa3_trace()
    report = attribute_kernels(scopes, kernels, 1, workload)
    for run in report["runs"]:
        count = len(run["layer_calls"])
        assert run["attention_dispatch_scope_counts"] == {"native_fa3_v2": count}
        assert run["modules"]["block_sparse_attention"]["kernel_count"] == 3 * count
        assert run["modules"]["block_sparse_attention"]["kernel_ns"] == 44 * TIME_SCALE * count
        assert all(call["attention_execution"] == "native_fa3_v2" for call in run["layer_calls"])


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 128, 1024])
@pytest.mark.parametrize("fallback", [False, True])
def test_fa3_sequence_and_native_fp16_or_independent_stride_fallback_remain_valid(rows, fallback):
    names = (
        ["nosa_attention::attention_kernel"]
        if fallback and rows < 4
        else [
            "nosa_attention::nonfinite_blocks_kernel",
            "nosa_attention::grouped_attention_kernel",
            "nosa_attention::attention_kernel",
        ]
        if fallback
        else FA3_NAMES
    )
    _validate_operator_kernels(
        "block_sparse_attention",
        [{"name": name} for name in names],
        "cuda_tvm_ffi",
        query_length=rows,
        total_length=65536 + rows,
        native_revision=4,
        attention_execution="native_fa3_v2",
    )


@pytest.mark.parametrize("false_spelling", ["(bool)0", "false"])
def test_fa3_actual_profiler_demangled_bool_template_is_recognized(false_spelling):
    names = [name.replace(", 0", f", {false_spelling}") for name in FA3_NAMES]
    _validate_operator_kernels(
        "block_sparse_attention",
        [{"name": name} for name in names],
        "cuda_tvm_ffi",
        query_length=1024,
        total_length=66560,
        native_revision=4,
        attention_execution="native_fa3_v2",
    )


def test_fa3_empty_query_dispatch_requires_no_kernels():
    options = {
        "query_length": 0,
        "total_length": 65536,
        "native_revision": 4,
        "attention_execution": "native_fa3_v2",
    }
    _validate_operator_kernels(
        "block_sparse_attention",
        [],
        "cuda_tvm_ffi",
        **options,
    )
    with pytest.raises(ValueError, match="kernel sequence"):
        _validate_operator_kernels(
            "block_sparse_attention",
            [{"name": name} for name in FA3_NAMES],
            "cuda_tvm_ffi",
            **options,
        )


@pytest.mark.parametrize(
    "problem",
    [
        "prepare",
        "main",
        "repair",
        "order",
        "foreign_flashinfer",
        "old_epilogue",
        "old_five_kernel_sequence",
        "old_declaration",
        "undeclared",
    ],
)
def test_fa3_rejects_partial_or_misidentified_attention_scopes(problem):
    names = list(FA3_NAMES)
    declaration = "native_fa3_v2"
    if problem in ("prepare", "main", "repair"):
        del names[{"prepare": 0, "main": 1, "repair": 2}[problem]]
    elif problem == "order":
        names[0], names[2] = names[2], names[0]
    elif problem == "foreign_flashinfer":
        names[1] = FA3_MAIN.replace("nosa_fa3::ML", "different_attention::ML")
    elif problem == "old_epilogue":
        names[1] = FA3_MAIN.replace("nosa_fa3::EP", "flashinfer::CollectiveEpilogue<nosa_fa3::KT>")
    elif problem == "old_five_kernel_sequence":
        names.insert(0, "nosa_attention::nonfinite_blocks_kernel")
        names.insert(3, "nosa_fa3::unpack")
    elif problem == "old_declaration":
        declaration = "native_fa3_v1"
    else:
        declaration = None
    with pytest.raises(ValueError, match="kernel sequence"):
        _validate_operator_kernels(
            "block_sparse_attention",
            [{"name": name} for name in names],
            "cuda_tvm_ffi",
            query_length=1024,
            total_length=66560,
            native_revision=4,
            attention_execution=declaration,
        )


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_fa3_build_identity_is_recorded_for_native_and_triton_control(backend):
    metadata, summary = native_inputs(backend=backend)
    add_fa3_build(metadata)
    summary["workload"]["attention_execution"] = (
        "native_fa3_v2" if backend == "native" else "triton_v1"
    )
    assert validate_kernel_backend(summary["workload"], metadata) == (
        "cuda_tvm_ffi" if backend == "native" else "triton"
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing_build",
        "headers",
        "include",
        "flags",
        "toolchain",
        "torch",
        "flashinfer",
        "tile",
        "stages",
        "cutoff",
        "missing_cutoff",
        "q_transfer",
        "output_store",
        "numerical_repair",
        "native_pv_accumulation",
        "source",
        "missing_source",
        "declaration",
        "undeclared",
    ],
)
def test_fa3_build_rejects_incomplete_or_inconsistent_provenance(problem):
    metadata, summary = native_inputs()
    add_fa3_build(metadata)
    summary["workload"]["attention_execution"] = "native_fa3_v2"
    fa3 = metadata["native_build"]["attention_fa3"]
    if problem == "missing_build":
        del metadata["native_build"]["attention_fa3"]
    elif problem == "headers":
        fa3["flashinfer_headers_sha256"] = "not-a-sha256"
    elif problem == "include":
        fa3["flashinfer_include"] = ""
    elif problem == "flags":
        fa3["cuda_flags"].remove("-use_fast_math")
    elif problem == "toolchain":
        fa3["toolchain"]["compiler"]["version"] = "different compiler"
    elif problem in ("torch", "flashinfer"):
        fa3[problem] = "different version"
    elif problem == "tile":
        fa3["kv_tile_tokens"] = 64
    elif problem == "stages":
        fa3["stages"] = 3
    elif problem == "cutoff":
        fa3["minimum_fa3_queries"] = 4
    elif problem == "missing_cutoff":
        del fa3["minimum_fa3_queries"]
    elif problem in ("q_transfer", "output_store", "numerical_repair", "native_pv_accumulation"):
        del fa3[problem]
    elif problem == "source":
        fa3["source_sha256"]["_nosa_attention_fa3.py"] = "0" * 64
    elif problem == "missing_source":
        del fa3["source_sha256"]["nosa_attention_fa3.cu"]
    elif problem == "undeclared":
        del summary["workload"]["attention_execution"]
    else:
        summary["workload"]["attention_execution"] = "triton_v1"
    with pytest.raises(ValueError):
        validate_kernel_backend(summary["workload"], metadata)


def test_fa3_mfu_uses_full_three_kernel_duration_and_unchanged_useful_flops():
    metadata, summary, profile, _ = fused_module_report_fixture()
    scopes, kernels, workload = fa3_trace()
    summary["workload"].update(workload)
    metadata["args"].update(workload | {"kernel_backend": "native"})
    source = "operators/sm90/csrc/nosa_indexer_checked.cu"
    metadata["source_sha256"][source] = "e" * 64
    metadata["native_build"]["source_sha256"][source] = "e" * 64
    add_fa3_build(metadata)
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
    module = report["phases"]["extend"]["modules"]["block_sparse_attention"]
    assert module["kernel_ms_median"] == 44 * TIME_SCALE / 1e6
    assert module["kernel_count_per_run"] == [3]
    assert module["mfu_pct_median"] == pytest.approx(
        module["matrix_flops_per_run"]
        / module["kernel_ms_median"]
        / 1e9
        / report["peak_tflops"]
        * 100
    )
