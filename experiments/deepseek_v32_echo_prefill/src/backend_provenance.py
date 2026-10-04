"""Source and installed-library identity for the official DeepSeek backends.

All inspection occurs outside measured ranges. Merely importing this module or
listing source files does not import Torch, DeepGEMM or FlashMLA.
"""

import gc
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PINS = {
    "DeepGEMM": "057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7",
    "DeepJIT": "2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43",
    "FlashMLA": "ba89a3466e9470ad08ab39738d4e7bb66989e1e7",
    "cutlass": "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
}
SOURCE_ROOTS = {
    "DeepGEMM": ("csrc", "deep_gemm", "setup.py", "scripts", ".gitmodules"),
    "DeepJIT": ("include",),
    "FlashMLA": ("csrc", "flash_mla", "setup.py", ".gitmodules"),
    "cutlass": ("include", "tools/util/include"),
}


def _git(directory, *arguments):
    return subprocess.check_output(["git", "-C", str(directory), *arguments]).decode().strip()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_files():
    """Return tracked official sources and local headers used in these builds.

    Include all FlashMLA translation units: its official setup compiles the full
    extension with SM90a targets, even though only sparse prefill is called.
    Uninitialized nested gitlinks are represented by the shared top-level trees.
    """
    files = {
        Path(__file__).resolve(),
        ROOT / "models/deepseek_v32/nonmatrix.py",
        ROOT / "operators/flashinfer.py",
        ROOT / ".gitmodules",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "scripts/prepare_3rdparty.py",
    }
    files.update((ROOT / "operators/deepseek_v32/norm").glob("*.py"))
    files.update((ROOT / "operators/deepseek_v32/linear").glob("*.py"))
    for name, prefixes in SOURCE_ROOTS.items():
        directory = ROOT / "3rdparty" / name
        listed = subprocess.check_output(
            ["git", "-C", str(directory), "ls-files", "-z", "--", *prefixes]
        )
        files.update(
            path
            for entry in listed.split(b"\0")
            if entry and (path := directory / entry.decode()).is_file()
        )
    return files


def _installed(name, native_name):
    distribution = importlib.metadata.distribution(name)
    native = importlib.import_module(native_name)
    package = importlib.import_module(native_name.split(".")[0])
    paths = {"native": Path(native.__file__).resolve(), "python": Path(package.__file__).resolve()}
    result = {
        "distribution_version": distribution.version,
        "direct_url": json.loads(distribution.read_text("direct_url.json") or "null"),
        "files": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
    }
    package_directory = paths["python"].parent
    result["python_sha256"] = {
        str(path.relative_to(package_directory)): digest(path)
        for path in sorted(package_directory.rglob("*.py"))
    }
    if name == "deep-gemm":
        # DeepGEMM compiles the wheel's copied headers at runtime. Identifying
        # only the API extension would miss the actual numerical kernel source.
        header_roots = {
            "deep_gemm": ROOT / "3rdparty/DeepGEMM/deep_gemm/include/deep_gemm",
            "cute": ROOT / "3rdparty/cutlass/include/cute",
            "cutlass": ROOT / "3rdparty/cutlass/include/cutlass",
        }
        installed_headers = {}
        for prefix, expected_root in header_roots.items():
            actual_root = package_directory / "include" / prefix
            expected = _tree_hashes(expected_root)
            actual = _tree_hashes(actual_root)
            if not expected or actual != expected:
                raise RuntimeError(
                    f"Installed DeepGEMM {prefix} headers differ; rebuild with uv sync"
                )
            installed_headers.update({f"{prefix}/{key}": value for key, value in actual.items()})
        result["jit_headers"] = {
            "path": str(package_directory / "include"),
            "sha256": installed_headers,
            "matches_pinned_sources": True,
        }
    return result


def _tree_hashes(directory):
    return {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def _file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def _installed_flashinfer():
    """Hash the installed distribution, without visiting its personal JIT cache."""
    distribution = importlib.metadata.distribution("flashinfer-python")
    if distribution.version != "0.6.18":
        raise RuntimeError("FlashInfer version changed; revalidate the pinned 0.6.18 APIs")
    package = Path(distribution.locate_file("flashinfer/__init__.py")).resolve().parent
    python_files = sorted(
        path for path in package.rglob("*.py") if "data" not in path.relative_to(package).parts
    )
    translation_units = (
        "norm.cu",
        "flashinfer_norm_binding.cu",
        "rope.cu",
        "flashinfer_rope_binding.cu",
        "topk.cu",
        "flashinfer_topk_binding.cu",
        "flashinfer_fast_topk_clusters_binding.cu",
        "tvm_ffi_utils.h",
    )
    sources = [package / "data/csrc" / name for name in translation_units]
    # Include shipped header dependencies, including CUB used by exact top-k.
    # These are distribution files, not arbitrary files in a user's cache.
    include_roots = (
        "data/include",
        "data/cccl/cub",
        "data/cccl/libcudacxx/include",
        "data/cccl/thrust",
        "data/cutlass/include",
    )
    include_hashes = {}
    for relative in include_roots:
        directory = package / relative
        hashes = _tree_hashes(directory)
        if not hashes:
            raise RuntimeError(f"Missing installed FlashInfer include tree: {directory}")
        include_hashes.update({f"{relative}/{name}": value for name, value in hashes.items()})
    return {
        "distribution_version": distribution.version,
        "direct_url": json.loads(distribution.read_text("direct_url.json") or "null"),
        "package_path": str(package),
        "python_sha256": {str(path.relative_to(package)): digest(path) for path in python_files},
        "source_sha256": {str(path.relative_to(package)): digest(path) for path in sources},
        "include_sha256": include_hashes,
        "installed_native_files": [_file_identity(path) for path in sorted(package.glob("*.so"))],
        "compiler_distributions": {
            name: importlib.metadata.version(name)
            for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi")
        },
        "boundary": "Installed source identity; live native/CuTe JIT artifacts are collected "
        "separately after execution. No binaries are copied.",
    }


def _typed_norm_identity():
    """Identify the local norm compiler and reject edits after its first use."""
    from operators.deepseek_v32.norm._fingerprint import build_info

    identity = build_info()
    for source in identity["sources"].values():
        if digest(source["path"]) != source["sha256"]:
            raise RuntimeError(f"Typed norm source changed during execution: {source['path']}")
    return identity


def _linear_quantization_identity():
    """Recheck the declared source bytes and any already-compiled kernel identity."""
    from operators.deepseek_v32.linear.quantization import build_info, runtime_info

    identity = build_info()
    sources = identity["source_and_dependency_sha256"]
    if not sources:
        raise RuntimeError("Linear quantization has no declared source identity")
    for path, expected in sources.items():
        if digest(path) != expected:
            raise RuntimeError(f"Linear quantization source changed during execution: {path}")
    runtime = runtime_info()
    if runtime is not None and runtime["build_info"] != identity:
        raise RuntimeError("Compiled linear quantization differs from current build identity")
    return identity


def _mapped_libraries():
    """Read loaded file mappings without searching any cache directory."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return set()
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[-1].startswith("/"):
            paths.add(fields[-1])
    return paths


def _compile_key(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_compile_key(item) for item in value]
    # functools' keyword separator is an object sentinel, not an address.
    return {"type": type(value).__name__}


def _cute_cached_artifacts(module):
    """Inspect only this module's live functools caches, never compile a kernel."""
    entries = []
    for name, function in sorted(vars(module).items()):
        if not name.startswith("_get_compiled_") or not hasattr(function, "cache_info"):
            continue
        for referent in gc.get_referents(function):
            if not isinstance(referent, dict):
                continue
            for key, compiled in referent.items():
                if not isinstance(key, tuple) or not hasattr(compiled, "function_name"):
                    continue
                entry = {
                    "factory": f"{module.__name__}.{name}",
                    "compile_key": _compile_key(key),
                    "function_name": compiled.function_name,
                    "kernel_names": sorted(getattr(compiled, "kernel_info", {})),
                    "mlir_bytecode": None,
                    "files": [],
                }
                ir_module = getattr(compiled, "ir_module", None)
                if ir_module is not None:
                    buffer = io.BytesIO()
                    ir_module.operation.write_bytecode(buffer)
                    bytecode = buffer.getvalue()
                    entry["mlir_bytecode"] = {
                        "sha256": hashlib.sha256(bytecode).hexdigest(),
                        "bytes": len(bytecode),
                    }
                artifacts = getattr(compiled, "artifacts", None)
                for field in ("PTX", "CUBIN", "SASS", "MLIR", "device_object_path"):
                    path = getattr(artifacts, field, None)
                    if isinstance(path, (str, Path)) and Path(path).is_file():
                        entry["files"].append({"kind": field, **_file_identity(path)})
                if entry["mlir_bytecode"] is None and not entry["files"]:
                    entry["unavailable_reason"] = "This live executor exposes no compiled IR/file"
                entries.append(entry)
    return sorted(entries, key=lambda row: (row["factory"], json.dumps(row["compile_key"])))


def collect_flashinfer_runtime_artifacts():
    """Identify this process's FlashInfer artifacts after warmup/execution.

    This is deliberately separate from the stable before/after library
    identity: normal JIT compilation grows these live registries. Inspection
    does not invoke module factories, compile kernels, or scan personal caches.
    """
    native = []
    module = sys.modules.get("flashinfer.jit.core")
    mapped = _mapped_libraries()
    if module is not None:
        specs = module.jit_spec_registry.get_all_specs()
        for name in sorted({"norm", "rope", "topk", "silu_and_mul"} & specs.keys()):
            spec = specs[name]
            library = Path(spec.get_library_path()).resolve()
            if str(library) + " (deleted)" in mapped:
                raise RuntimeError(
                    f"Loaded FlashInfer library was removed during execution: {library}"
                )
            loaded = str(library) in mapped
            row = {
                "name": name,
                "loaded_in_this_process": loaded,
                "library": _file_identity(library) if loaded else None,
                "sources": [_file_identity(path) for path in getattr(spec, "sources", [])],
                "build_metadata": None,
                "cuda_flags": getattr(spec, "extra_cuda_cflags", None),
                "cxx_flags": getattr(spec, "extra_cflags", None),
            }
            ninja = getattr(spec, "ninja_path", None)
            if ninja is not None and Path(ninja).is_file():
                row["build_metadata"] = _file_identity(ninja)
            native.append(row)
    cute = []
    for name in (
        "flashinfer.norm.kernels.rmsnorm",
        "flashinfer.norm.kernels.fused_add_rmsnorm",
        "operators.deepseek_v32.norm._compile",
    ):
        module = sys.modules.get(name)
        if module is not None:
            cute.extend(_cute_cached_artifacts(module))
    quantizer = sys.modules.get("operators.deepseek_v32.linear.quantization")
    linear_quantization = None if quantizer is None else quantizer.runtime_info()
    return {
        "schema_version": 1,
        "native_jit": native,
        "cute_jit": cute,
        "linear_quantization_triton": linear_quantization,
        "boundary": "Registered modules plus process mappings identify loaded native libraries; "
        "live vendor and local norm CuTe cache entries identify compiled specializations. "
        "The linear quantizer reports its observed Triton specializations separately. "
        "MLIR bytecode is hashed in "
        "memory, including embedded compiled objects when exposed. No personal cache scan or "
        "binary copy; this is source identity, not proof of which GPU launches were timed.",
    }


def collect_backend_provenance():
    """Reject mismatched pins or installed versions and identify loaded binaries."""
    revisions = {}
    for name, expected in PINS.items():
        directory = ROOT / "3rdparty" / name
        actual = _git(directory, "rev-parse", "HEAD")
        if actual != expected:
            raise RuntimeError(f"{name} revision changed; revalidate official backend: {actual}")
        dirty = _git(directory, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise RuntimeError(f"Tracked {name} sources are modified: {dirty}")
        revisions[name] = actual
    installed = {
        "deep_gemm": _installed("deep-gemm", "deep_gemm._C"),
        "flash_mla": _installed("flash-mla", "flash_mla.cuda"),
        "flashinfer": _installed_flashinfer(),
    }
    for name, expected in (("deep_gemm", "2.8.1+057ca59"), ("flash_mla", "1.0.0+ba89a34")):
        if installed[name]["distribution_version"] != expected:
            raise RuntimeError(f"Installed {name} does not match source pin; run uv sync")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {
        "git_revisions": revisions,
        "installed": installed,
        "build_variables": config["tool"]["uv"].get("extra-build-variables", {}),
        "jit_environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("DG_", "DJ_", "FLASHINFER_", "CUTE_DSL_"))
        },
        "linear_activation_quantization": _linear_quantization_identity(),
        "typed_norm": _typed_norm_identity(),
        "flash_mla_upstream_cutlass_pin": "147f5673d0c1c3dcf66f78d677fd647e4a020219",
        "flash_mla_shared_cutlass_pin": revisions["cutlass"],
        "boundary": "Loaded official libraries and tracked build sources; hashes are identity, "
        "not a numerical or performance acceptance result.",
    }
