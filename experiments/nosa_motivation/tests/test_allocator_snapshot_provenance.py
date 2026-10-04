"""Saved adapter evidence must match its exact mapped build or explicit fallback."""

from copy import deepcopy

import pytest

from experiments.nosa_motivation.src.provenance import audit_native_identity, manifest_digest


def metadata():
    files = {"header.h": {"sha256": "a" * 64, "size": 1}}
    dependency = {"files": files, "sha256": manifest_digest(files)}
    identity = {"source_and_dependency_sha256": {"adapter.cpp": "c" * 64}}
    build = {
        "dependencies": {
            "cutlass": {"include": dependency, "util_include": dependency},
            **{
                name: dependency
                for name in (
                    "flashinfer_include",
                    "tvm_ffi_include",
                    "dlpack_include",
                    "cuda_include",
                )
            },
        },
        "allocator_snapshot_build_info": {
            "available": True,
            "identity": identity,
            "fingerprint": manifest_digest(identity),
        },
    }
    build["sha256"] = manifest_digest(build)
    artifacts = {
        "/fixture/cxldsagr_allocator.so": {"sha256": "d" * 64},
        "/fixture/cxldsagr_token.so": {"sha256": "e" * 64},
    }
    case = {
        "native_artifacts_before": artifacts,
        "native_artifacts_after": deepcopy(artifacts),
        "token_validation": {
            "backend": "cpython_native",
            "loaded_binary_path": "/fixture/cxldsagr_token.so",
            "loaded_binary_sha256": "e" * 64,
        },
        "allocator_snapshot": {
            "backend": "private_cpp",
            "fallback_reason": None,
            "fingerprint": manifest_digest(identity),
            "loaded_binary_path": "/fixture/cxldsagr_allocator.so",
            "loaded_binary_sha256": "d" * 64,
        },
    }
    return {
        "native_provenance": {
            "build_before": build,
            "build_after": deepcopy(build),
            "artifacts_final": artifacts,
        },
        "cases": [case],
    }


def test_exact_native_adapter_identity_is_required():
    value = metadata()
    assert audit_native_identity(value) == value["native_provenance"]["build_before"]["sha256"]
    value["cases"][0]["allocator_snapshot"]["loaded_binary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="adapter differs"):
        audit_native_identity(value)


def test_fallback_requires_explicit_reason_and_preserves_old_metadata_audit():
    value = metadata()
    value["cases"][0]["allocator_snapshot"] = {
        "backend": "torch_official",
        "fallback_reason": "compiler failed",
    }
    audit_native_identity(value)
    value["cases"][0]["allocator_snapshot"]["fallback_reason"] = None
    with pytest.raises(ValueError, match="fallback evidence"):
        audit_native_identity(value)
    # Previous accepted runs did not include an adapter; their original native
    # source/library evidence remains independently auditable.
    for name in ("build_before", "build_after"):
        build = value["native_provenance"][name]
        del build["allocator_snapshot_build_info"]
        del build["sha256"]
        build["sha256"] = manifest_digest(build)
    del value["cases"][0]["allocator_snapshot"]
    audit_native_identity(value)


def test_runtime_fingerprint_cannot_vouch_for_a_different_planned_identity():
    value = metadata()
    for name in ("build_before", "build_after"):
        build = value["native_provenance"][name]
        build["allocator_snapshot_build_info"]["identity"]["source_and_dependency_sha256"][
            "adapter.cpp"
        ] = "f" * 64
        del build["sha256"]
        build["sha256"] = manifest_digest(build)
    with pytest.raises(ValueError, match="adapter differs"):
        audit_native_identity(value)
