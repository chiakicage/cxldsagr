"""Isolated bindings to the unmodified official ECHO checkout.

The first build downloads ECHO's pinned fmt headers into the user cache. The
project's installed mainline DeepGEMM remains importable in the same process.
Only binding and build code lives here; kernels come directly from upstream.
"""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from functools import cache
from pathlib import Path

UPSTREAM_REVISION = "bc1b75c1000010d0ac6f032ebaac283255c050b1"
CUTLASS_REVISION = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
FMT_REVISION = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
FMT_ARCHIVE_SHA256 = "c314292789d28c3c3b420e75a7b2d1706f685f7fb63289128d46aeaea2c6be71"
_ROOT = Path(__file__).resolve().parents[3]
_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cxldsagr/echo-official"
_CXX_FLAGS = ["-O3", "-std=c++17", "-fvisibility=hidden", "-Wno-deprecated-declarations"]
_CUDA_FLAGS = ["-O3", "--extended-lambda", "-gencode=arch=compute_90a,code=sm_90a"]
_ATTENTION_BINDING = """#include <pybind11/pybind11.h>
#include <torch/python.h>
#include "apis/attention.hpp"
#include "apis/runtime.hpp"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    deep_gemm::attention::register_apis(m);
    deep_gemm::runtime::register_apis(m);
    m.def("_set_jit_cache", [](const std::string& path) {
        deep_gemm::compiler->cache_dir_path = path;
    });
}
"""
_TOPK_BINDING = """#include <torch/extension.h>
void fast_argmin_bounded_interface(at::Tensor, at::Tensor, at::Tensor);
void fast_topk_interface(at::Tensor, at::Tensor, at::Tensor);
void fast_topk_transform_interface(at::Tensor, at::Tensor, at::Tensor,
                                  at::Tensor, at::Tensor, std::optional<at::Tensor>);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_argmin_bounded", &fast_argmin_bounded_interface);
    m.def("fast_topk", &fast_topk_interface);
    m.def("fast_topk_transform", &fast_topk_transform_interface);
}
"""
_LOADED = {}


def upstream_root() -> Path:
    """Return the checkout required by this adapter, without importing CUDA."""
    return _ROOT / "3rdparty/ECHO"


def _git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@cache
def _check_sources():
    for root, revision in (
        (upstream_root(), UPSTREAM_REVISION),
        (_ROOT / "3rdparty/cutlass", CUTLASS_REVISION),
    ):
        if _git(root, "rev-parse", "HEAD") != revision:
            raise RuntimeError(f"Official ECHO requires {root} at {revision}")
        result = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=False
        )
        if result.returncode:
            raise RuntimeError(f"Official ECHO requires unmodified tracked files in {root}")


@cache
def _fmt_include():
    dependency_root = _CACHE / "dependencies"
    expected = dependency_root / f"fmt-{FMT_REVISION}"
    if (expected / "include/fmt/base.h").is_file():
        return expected / "include"
    dependency_root.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/fmtlib/fmt/tar.gz/{FMT_REVISION}"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != FMT_ARCHIVE_SHA256:
        raise RuntimeError("The pinned ECHO fmt archive failed SHA256 verification")
    with tempfile.TemporaryDirectory(dir=dependency_root) as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(temporary, filter="data")
        extracted = Path(temporary) / expected.name
        try:
            extracted.rename(expected)
        except FileExistsError:
            pass
    return expected / "include"


def _hash_files(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


@cache
def _identity():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    _check_sources()
    if CUDA_HOME is None:
        raise RuntimeError("The official ECHO build requires a CUDA toolkit")
    fmt_include = _fmt_include()
    source = upstream_root() / "DeepGEMM"
    files = {Path(__file__), upstream_root() / "sglang/sgl-kernel/csrc/elementwise/topk.cu"}
    cache_source = upstream_root() / "sglang/python/sglang/srt/mem_cache"
    files.update(
        cache_source / name
        for name in ("recall_ops.py", "allocator.py", "memory_pool.py", "memory_pool_host.py")
    )
    files.update(
        upstream_root() / name
        for name in (
            "sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py",
            "sglang/python/sglang/srt/layers/attention/nsa/utils.py",
            "sglang/python/sglang/srt/layers/attention/nsa_backend.py",
            "sglang/sgl-kernel/python/sgl_kernel/top_k.py",
        )
    )
    for directory in (
        source / "csrc",
        source / "deep_gemm/include",
        _ROOT / "3rdparty/cutlass/include",
        fmt_include,
    ):
        files.update(path for path in directory.rglob("*") if path.is_file())
    identity = {
        "upstream_url": "https://github.com/sjtu-zhao-lab/ECHO",
        "upstream_revision": UPSTREAM_REVISION,
        "cutlass_revision": CUTLASS_REVISION,
        "fmt_revision": FMT_REVISION,
        "fmt_archive_sha256": FMT_ARCHIVE_SHA256,
        "source_files": _hash_files(files),
        "torch_version": torch.__version__,
        "torch_cxx11_abi": torch.compiled_with_cxx11_abi(),
        "python_version": sys.version,
        "cuda_home": CUDA_HOME,
        "nvcc_version": subprocess.check_output(
            [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
        ),
        "cxx_flags": _CXX_FLAGS,
        "cuda_flags": _CUDA_FLAGS,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return identity, digest


def _load(name, sources, directory, *, cuda=False):
    from torch.utils.cpp_extension import CUDA_HOME, load

    # A direct .venv/bin/python invocation need not have its Ninja on PATH.
    prior_path = os.environ.get("PATH", "")
    prior_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{prior_path}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    source = upstream_root() / "DeepGEMM"
    try:
        return load(
            name=name,
            sources=[str(path) for path in sources],
            extra_cflags=_CXX_FLAGS,
            extra_cuda_cflags=_CUDA_FLAGS if cuda else None,
            extra_include_paths=[
                str(source / "csrc"),
                str(source / "deep_gemm/include"),
                str(_ROOT / "3rdparty/cutlass/include"),
                str(_fmt_include()),
                str(Path(CUDA_HOME) / "include"),
                str(Path(CUDA_HOME) / "include/cccl"),
            ],
            extra_ldflags=[
                f"-L{CUDA_HOME}/lib64",
                f"-L{CUDA_HOME}/lib64/stubs",
                "-lcuda",
                "-lcudart",
                "-lnvrtc",
                "-lcublasLt",
            ],
            build_directory=str(directory),
            verbose=os.environ.get("ECHO_OFFICIAL_BUILD_VERBOSE") == "1",
        )
    finally:
        os.environ["PATH"] = prior_path
        if prior_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prior_arch


def _write_binding(path, contents):
    if not path.exists() or path.read_text() != contents:
        path.write_text(contents)


@cache
def module():
    """Load ECHO's original attention and runtime API under a unique name."""
    identity, digest = _identity()
    directory = _CACHE / digest / "attention"
    directory.mkdir(parents=True, exist_ok=True)
    binding = directory / "binding.cpp"
    _write_binding(binding, _ATTENTION_BINDING)
    native = _load(f"cxldsagr_official_echo_attention_{digest}", [binding], directory)
    runtime = directory / "runtime"
    includes = runtime / "include"
    includes.mkdir(parents=True, exist_ok=True)
    for name, target in (
        ("deep_gemm", upstream_root() / "DeepGEMM/deep_gemm/include/deep_gemm"),
        ("cute", _ROOT / "3rdparty/cutlass/include/cute"),
        ("cutlass", _ROOT / "3rdparty/cutlass/include/cutlass"),
    ):
        link = includes / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    native.init(str(runtime), identity["cuda_home"])
    # Upstream's compiler object is isolated by the module's hidden symbols.
    # Initialize its cache once; do not modify mainline DG_JIT_CACHE_DIR.
    native._set_jit_cache(str(directory / "jit"))
    _LOADED["attention"] = native
    return native


@cache
def topk_module():
    """Load the exact upstream bounded argmin and exact top-2048 kernels."""
    _, digest = _identity()
    directory = _CACHE / digest / "topk"
    directory.mkdir(parents=True, exist_ok=True)
    binding = directory / "binding.cpp"
    _write_binding(binding, _TOPK_BINDING)
    native = _load(
        f"cxldsagr_official_echo_topk_{digest}",
        [binding, upstream_root() / "sglang/sgl-kernel/csrc/elementwise/topk.cu"],
        directory,
        cuda=True,
    )
    _LOADED["topk"] = native
    return native


def provenance():
    """Hash source inputs and currently built native modules/JIT kernels.

    Call after warmup to include the shape-specialized upstream CUBINs. Source
    hashes are verified outside measured invocations; native hashes are refreshed.
    """
    identity, digest = _identity()
    current_sources = _hash_files(Path(path) for path in identity["source_files"])
    if current_sources != identity["source_files"]:
        changed = [
            path
            for path, digest in current_sources.items()
            if digest != identity["source_files"][path]
        ]
        raise RuntimeError(f"Official ECHO source changed after build: {changed}")
    directory = _CACHE / digest
    files = {Path(native.__file__) for native in _LOADED.values()}
    for suffix in ("*.cubin", "*.cu", "binding.cpp"):
        files.update(path for path in directory.rglob(suffix) if path.is_file())
    return {
        **identity,
        "build_identity": digest,
        "native_files": _hash_files(files),
        "runtime_workspace": {
            "bytes": 32 * 1024 * 1024,
            "allocation": "torch::empty uint8 CUDA tensor in original DeviceRuntime",
            "included_in_pytorch_allocated": True,
            "source": str(upstream_root() / "DeepGEMM/csrc/jit/device_runtime.hpp"),
        },
    }
