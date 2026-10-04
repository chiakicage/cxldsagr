"""Precision provenance and mapped binary identity work without CUDA execution."""

import hashlib
import os
from types import SimpleNamespace

import pytest

from experiments.nosa_baseline_performance.src.runtime import (
    native_artifacts,
    runtime_settings,
    verify_native_artifacts,
    verify_runtime_settings,
)


def mapped_library(tmp_path):
    library = tmp_path / "libcublas.so"
    library.write_bytes(b"loaded binary")
    stat = library.stat()
    maps = tmp_path / "maps"
    maps.write_text(
        f"1000-2000 r-xp 0000 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} {stat.st_ino} {library}\n"
    )
    return library, maps


def test_mapped_library_identity_uses_executable_bytes(tmp_path):
    library, maps = mapped_library(tmp_path)
    records = native_artifacts(maps)
    assert records[str(library)]["sha256"] == hashlib.sha256(b"loaded binary").hexdigest()
    verify_native_artifacts(records, maps)
    library.write_bytes(b"edited binary")
    with pytest.raises(ValueError, match="file identity changed"):
        verify_native_artifacts(records, maps)


def test_mapped_library_replacement_is_rejected_even_with_equal_bytes(tmp_path):
    library, maps = mapped_library(tmp_path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(library.read_bytes())
    replacement.replace(library)
    with pytest.raises(ValueError, match="was replaced"):
        native_artifacts(maps)


def test_runtime_settings_capture_precision_and_dispatch(monkeypatch):
    matmul = SimpleNamespace(
        allow_tf32=False,
        allow_bf16_reduced_precision_reduction=True,
        allow_fp16_reduced_precision_reduction=True,
        allow_fp16_accumulation=False,
    )
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=matmul, preferred_blas_library=lambda: "Cublas")
        ),
        get_float32_matmul_precision=lambda: "highest",
        are_deterministic_algorithms_enabled=lambda: False,
    )
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    settings = runtime_settings(torch)
    assert settings["environment"]["CXLDSAGR_SM90_BACKEND"] == "native"
    assert settings["matmul_allow_bf16_reduced_precision_reduction"] is True
    matmul.allow_tf32 = True
    with pytest.raises(ValueError, match="settings changed"):
        verify_runtime_settings(settings, torch)
