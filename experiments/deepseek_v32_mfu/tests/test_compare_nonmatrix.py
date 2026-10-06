"""Exclusive scope grouping and runtime attribution checks, without CUDA."""

import hashlib
import json

import pytest

from experiments.deepseek_v32_mfu.src.compare_nonmatrix import (
    TYPED_NORM_FACTORY,
    TYPED_NORM_SIGNATURE,
    aggregate,
    classify,
    verify_runtime,
)


def row(stage, count, ns, *, matrix=False):
    return {
        "mode": "offload",
        "phase": "extend_annotated",
        "layer": "all_layers",
        "stage": stage,
        "kernel_count": count,
        "kernel_ns": ns,
        "useful_flops": 10 if matrix else None,
        "metadata_call_count_matches": True,
        # Deliberately enormous inclusive durations must never enter GPU totals.
        "scope_host_union_ms": 10**9,
        "memcpy_ns": 10**9,
        "memset_ns": 10**9,
    }


def test_old_and_new_nested_norm_scopes_are_counted_only_exclusively():
    rows = [
        row("rms_norm", 19, 100),
        row("residual_rms_norm", 28, 200),
        row("input_residual_norm", 0, 0),
        row("post_attention_residual_norm", 0, 0),
        row("cache_write", 3, 400),
        row("q_a_proj", 9, 500, matrix=True),
        row("embedding", 1, 20),
    ]
    result = aggregate(rows)
    key = ("offload", "extend_annotated")
    assert result[(*key, "norm")]["kernel_count"] == 47
    assert result[(*key, "norm")]["kernel_ns"] == 300
    assert result[(*key, "nonmatrix_excluding_cache")]["kernel_ns"] == 320
    assert result[(*key, "all_nonmatrix_scopes")]["kernel_ns"] == 720
    assert result[(*key, "all_kernel_scopes")]["kernel_ns"] == 1220
    assert result[(*key, "all_kernel_scopes")]["kernel_count"] == 60


def test_rotary_and_activation_mapping_includes_materialization_helpers():
    rows = [
        row("apply_rope_pair", 9, 120),
        row("prepare_rotary_cache", 12, 18),
        row("dense_mlp", 0, 0),
        row("silu_mul", 6, 200),
    ]
    result = aggregate(rows)
    key = ("offload", "extend_annotated")
    assert result[(*key, "rotary")]["kernel_count"] == 21
    assert result[(*key, "rotary")]["kernel_ns"] == 138
    assert result[(*key, "mlp_activation")]["stages"] == {"dense_mlp", "silu_mul"}


def test_overlapping_duplicate_or_unknown_cache_rows_fail():
    with pytest.raises(ValueError, match="Overlapping"):
        classify("same", False, {"a": ("same",), "b": ("same",)})
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([row("rms_norm", 1, 1), row("rms_norm", 2, 2)])
    with pytest.raises(ValueError, match="Unmapped cache"):
        aggregate([row("offload_new_stage", 1, 1)])
    with pytest.raises(ValueError, match="Matrix scope"):
        aggregate([row("quantize_index", 1, 1, matrix=True)])
    with pytest.raises(ValueError, match="Invalid exclusive"):
        aggregate([row("quantize_index", 1, -1)])


def test_actual_runtime_kernel_must_match_registered_fp32_norm_and_scope():
    result = {
        "flashinfer_runtime_artifacts": {
            "native_jit": [],
            "cute_jit": [
                {
                    "compile_key": ["float32", 7168],
                    "kernel_names": ["flashinfernormkernelsFusedAddRMSNorm"],
                }
            ],
        }
    }
    kernel = {
        "kernel_name": "flashinfernormkernelsFusedAddRMSNorm",
        "stage": "residual_rms_norm",
        "unattributed_reason": "",
    }
    assert verify_runtime(result, [kernel])["runtime_scope_mapping_verified"]
    with pytest.raises(ValueError, match="wrong non-matrix scope"):
        verify_runtime(result, [{**kernel, "stage": "rms_norm"}])
    with pytest.raises(ValueError, match="lacks a recorded"):
        verify_runtime(result, [{**kernel, "kernel_name": "flashinfernormkernelsUnknown"}])
    result["flashinfer_runtime_artifacts"]["cute_jit"][0]["compile_key"][0] = "bfloat16"
    with pytest.raises(ValueError, match="FP32"):
        verify_runtime(result, [kernel])


def typed_runtime(kind="plain", width=7168):
    # Match the saved production CuTe metadata schema, including the mangled
    # local namespace; fixture hashes represent recorded bytes, not live files.
    sources = {
        "local/" + name: {"path": "/snapshot/norm/" + name, "sha256": "a" * 64}
        for name in (
            "api.py",
            "_compile.py",
            "_fingerprint.py",
            "_layout.py",
            "_plain.py",
            "_fused.py",
        )
    }
    versions = {"torch": "fixture", "nvidia-cutlass-dsl": "fixture"}
    fingerprint = hashlib.sha256(
        json.dumps(
            {"versions": versions, "sources": {name: "a" * 64 for name in sources}},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    name = (
        "kernel_cutlass_kernel_operatorsdeepseek_v32norm_"
        + kind
        + "Local"
        + kind.title()
        + "RMSNorm_object_at__fixture"
    )
    result = {
        "backend_provenance": {
            "typed_norm": {
                "fingerprint": fingerprint,
                "versions": versions,
                "sources": sources,
                "loaded_process_identity": True,
            }
        },
        "flashinfer_runtime_artifacts": {
            "native_jit": [],
            "cute_jit": [
                {
                    "factory": TYPED_NORM_FACTORY,
                    "compile_key": [
                        kind,
                        width,
                        0,
                        True,
                        fingerprint,
                        TYPED_NORM_SIGNATURE,
                        "--enable-tvm-ffi",
                    ],
                    "kernel_names": [name],
                    "mlir_bytecode": {"sha256": "b" * 64, "bytes": 1234},
                }
            ],
        },
    }
    kernel = {"kernel_name": name, "stage": "residual_rms_norm", "unattributed_reason": ""}
    return result, result["flashinfer_runtime_artifacts"]["cute_jit"][0], kernel


@pytest.mark.parametrize(
    "kind,width", [("plain", 512), ("plain", 1536), ("plain", 7168), ("fused", 7168)]
)
def test_typed_runtime_source_contract_and_factory_kind_determine_scope(kind, width):
    result, _, kernel = typed_runtime(kind, width)
    verified = verify_runtime(result, [kernel])
    assert verified["runtime_scope_mapping_verified"]
    assert (
        verified["typed_norm_source_fingerprint"]
        == result["backend_provenance"]["typed_norm"]["fingerprint"]
    )
    assert verified["observed_cute_kernel_names"] == [kernel["kernel_name"]]
    if kind == "plain":
        assert verify_runtime(result, [{**kernel, "stage": "rms_norm"}])[
            "runtime_scope_mapping_verified"
        ]
    else:
        with pytest.raises(ValueError, match="wrong non-matrix scope"):
            verify_runtime(result, [{**kernel, "stage": "rms_norm"}])
    with pytest.raises(ValueError, match="wrong non-matrix scope"):
        verify_runtime(result, [{**kernel, "stage": "silu_mul"}])


@pytest.mark.parametrize(
    "index,value",
    [
        (0, "unknown"),
        (1, 64),
        (1, True),
        (2, -1),
        (2, True),
        (2, "0"),
        (3, 1),
        (4, "f" * 64),
        (5, "bf16_bf16_bf16"),
        (6, "--other-options"),
    ],
)
def test_typed_runtime_rejects_changed_compile_contract(index, value):
    result, module, kernel = typed_runtime()
    module["compile_key"][index] = value
    with pytest.raises(ValueError, match="Typed norm compile"):
        verify_runtime(result, [kernel])


@pytest.mark.parametrize("length", [0, 6, 8])
def test_typed_runtime_rejects_changed_key_schema(length):
    result, module, kernel = typed_runtime()
    module["compile_key"] = (module["compile_key"] + [None])[:length]
    with pytest.raises(ValueError, match="seven fields"):
        verify_runtime(result, [kernel])


def test_typed_fused_runtime_rejects_unvalidated_width():
    result, _, kernel = typed_runtime("fused", 1536)
    with pytest.raises(ValueError, match="validated BF16/FP32"):
        verify_runtime(result, [kernel])


@pytest.mark.parametrize(
    "missing",
    ["backend_provenance", "typed_norm", "versions", "sources", "loaded_process_identity"],
)
def test_typed_runtime_requires_source_identity(missing):
    result, _, kernel = typed_runtime()
    if missing == "backend_provenance":
        del result[missing]
    elif missing == "typed_norm":
        del result["backend_provenance"][missing]
    else:
        del result["backend_provenance"]["typed_norm"][missing]
    with pytest.raises(ValueError, match="recorded source and toolchain"):
        verify_runtime(result, [kernel])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_local",
        "changed_source",
        "bad_hash",
        "bad_path",
        "changed_version",
        "bad_fingerprint",
    ],
)
def test_typed_runtime_checks_complete_source_fingerprint(mutation):
    result, _, kernel = typed_runtime()
    identity = result["backend_provenance"]["typed_norm"]
    source = identity["sources"]["local/_compile.py"]
    if mutation == "missing_local":
        del identity["sources"]["local/_compile.py"]
    elif mutation == "changed_source":
        source["sha256"] = "c" * 64
    elif mutation == "bad_hash":
        source["sha256"] = "malformed"
    elif mutation == "bad_path":
        source["path"] = None
    elif mutation == "changed_version":
        identity["versions"]["torch"] = "changed"
    else:
        identity["fingerprint"] = "d" * 64
    with pytest.raises(ValueError, match="Typed norm source"):
        verify_runtime(result, [kernel])


@pytest.mark.parametrize("field", ["kernel_names", "mlir_bytecode"])
def test_typed_runtime_requires_compiled_artifact_metadata(field):
    result, module, kernel = typed_runtime()
    del module[field]
    with pytest.raises(ValueError, match="Typed norm"):
        verify_runtime(result, [kernel])


def test_typed_kernel_cannot_bypass_factory_or_registration_checks():
    result, module, kernel = typed_runtime()
    module["factory"] += "_unknown"
    with pytest.raises(ValueError, match="typed compiler factory"):
        verify_runtime(result, [kernel])
    module["factory"] = "flashinfer.norm.kernels.rmsnorm._get_compiled_rmsnorm_kernel"
    module["compile_key"] = ["float32", 7168]
    with pytest.raises(ValueError, match="typed compiler factory"):
        verify_runtime(result, [kernel])
    result["flashinfer_runtime_artifacts"]["cute_jit"] = []
    with pytest.raises(ValueError, match="lacks a recorded"):
        verify_runtime(result, [kernel])


def test_typed_kernel_name_must_match_registered_kind_and_exact_name():
    result, module, kernel = typed_runtime()
    module["compile_key"][0] = "fused"
    with pytest.raises(ValueError, match="compiled factory kind"):
        verify_runtime(result, [kernel])
    module["compile_key"][0] = "plain"
    with pytest.raises(ValueError, match="lacks a recorded"):
        verify_runtime(result, [{**kernel, "kernel_name": kernel["kernel_name"] + "_unknown"}])
