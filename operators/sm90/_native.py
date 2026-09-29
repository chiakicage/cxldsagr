"""Build the project's Hopper kernels with tvm-ffi and shared CUTLASS headers.

Compilation is lazy and cached outside the source tree. CPU reference imports
do not load tvm-ffi, CUDA, or a compiled extension.
"""

import hashlib
import os
import shutil
import subprocess
from functools import cache, lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = Path(__file__).resolve().parent / "csrc"
_COMPONENTS = {
    "nosa_scores",
    "nosa_attention",
    "nosa_selection",
    "nosa_indexer",
    "nosa_prepare",
    "nosa_prepare_ranked",
    "nosa_indexer_checked",
}
_SOURCE_SUFFIXES = {".cu", ".cuh", ".c", ".cpp", ".h", ".hpp"}
_CUDA_FLAGS = (
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-lineinfo",
)


def backend_name():
    """Select native kernels or the retained Triton measurement control."""
    backend = os.environ.get("CXLDSAGR_SM90_BACKEND", "native")
    if backend not in ("native", "triton"):
        raise ValueError("CXLDSAGR_SM90_BACKEND must be native or triton")
    return backend


def native_enabled():
    return backend_name() == "native"


@lru_cache(maxsize=1)
def _toolchain():
    import tvm_ffi

    compiler = shutil.which("nvcc")
    if compiler is None:
        raise RuntimeError("The native SM90 backend requires nvcc on PATH")
    version = subprocess.check_output([compiler, "--version"], text=True).strip()
    cutlass = _ROOT / "3rdparty/cutlass"
    commit = subprocess.check_output(
        ["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True
    ).strip()
    header = cutlass / "include/cutlass/version.h"
    return {
        "tvm_ffi": tvm_ffi.__version__,
        "compiler": {"path": str(Path(compiler).resolve()), "version": version},
        "cutlass": {
            "commit": commit,
            "version_header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        },
        "cuda_flags": list(_CUDA_FLAGS),
    }


def build_info():
    """Serializable build provenance without compiling or initializing CUDA."""
    paths = [Path(__file__), *_source_files()]
    info = {
        **_toolchain(),
        "selected_backend": backend_name(),
        "source_sha256": {
            str(path.relative_to(_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    from operators.sm90._nosa_attention_fa3 import build_info as attention_build_info

    info["attention_fa3"] = attention_build_info()
    return info


def _source_files():
    return sorted(
        path for path in _SOURCE.rglob("*") if path.is_file() and path.suffix in _SOURCE_SUFFIXES
    )


@cache
def load_module(component):
    """Return a source/version-keyed module compiled for SM90a."""
    if component not in _COMPONENTS:
        raise ValueError(f"Unknown SM90 kernel component: {component}")
    import tvm_ffi.cpp

    source = _SOURCE / f"{component}.cu"
    include = _ROOT / "3rdparty/cutlass/include"
    if not (include / "cute/tensor.hpp").is_file():
        raise RuntimeError("Prepare shared CUTLASS with python scripts/prepare_3rdparty.py --init")
    digest = hashlib.sha256()
    for path in sorted({Path(__file__), *_source_files()}):
        digest.update(str(path.relative_to(_ROOT)).encode())
        digest.update(path.read_bytes())
    # CUTLASS's version header invalidates the cache across header releases;
    # source/include dependencies are also tracked by the native build system.
    digest.update(repr(_toolchain()).encode())
    return tvm_ffi.cpp.load(
        name=f"cxldsagr_sm90_{component}_{digest.hexdigest()[:16]}",
        sources=[str(source)],
        extra_include_paths=[str(include), str(_SOURCE)],
        extra_cuda_cflags=list(_CUDA_FLAGS),
    )
