"""Source/toolchain identity for the local norm CuTe compilation."""

import hashlib
import importlib
import importlib.metadata
import json
import sys
from functools import lru_cache
from pathlib import Path

import torch


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_path(name):
    module = importlib.import_module(name)
    path = getattr(module, "__file__", None)
    if not path:
        raise RuntimeError(f"Cannot identify loaded norm dependency {name}")
    return Path(path).resolve()


@lru_cache(maxsize=1)
def build_info():
    """Freeze the loaded source identity; importing the public API does not call this."""
    directory = Path(__file__).resolve().parent
    paths = {
        f"local/{name}": directory / name
        for name in (
            "api.py",
            "_compile.py",
            "_fingerprint.py",
            "_layout.py",
            "_plain.py",
            "_fused.py",
        )
    }
    vendor_modules = (
        "flashinfer.norm.kernels.rmsnorm",
        "flashinfer.norm.kernels.fused_add_rmsnorm",
        "flashinfer.norm.utils",
        "flashinfer.cute_dsl.utils",
        "flashinfer.utils",
    )
    for name in vendor_modules:
        paths[name] = _module_path(name)
    # Resolve the actually imported CTK flavor, not an arbitrary installed .so.
    paths["cutlass.loaded_native"] = _module_path("cutlass._mlir._mlir_libs._cutlass_ir")
    for package in ("cutlass", "tvm_ffi"):
        root = _module_path(package).parent
        paths[package] = root / "__init__.py"
        for extension in ("*.py", "*.h", "*.hpp", "*.cuh"):
            for path in sorted(root.rglob(extension)):
                paths[f"{package}/{path.relative_to(root)}"] = path
        if package == "tvm_ffi":
            for path in sorted(root.rglob("*.so")):
                paths[f"{package}/{path.relative_to(root)}"] = path
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("cutlass.") and path and Path(path).suffix == ".so":
            paths[name] = Path(path).resolve()
    versions = {
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_runtime": torch.version.cuda,
        **{
            name: importlib.metadata.version(name)
            for name in ("flashinfer-python", "nvidia-cutlass-dsl", "apache-tvm-ffi")
        },
    }
    sources = {
        label: {"path": str(path), "sha256": _digest(path)} for label, path in sorted(paths.items())
    }
    identity = {
        "versions": versions,
        "sources": {label: record["sha256"] for label, record in sources.items()},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {
        "fingerprint": fingerprint,
        "versions": versions,
        "sources": sources,
        "scope": "Local norm, imported vendor helpers and resolved CuTe/TVM FFI tooling; no other model sources",
        "loaded_process_identity": True,
        "cache": "Process memoizer includes this fingerprint; direct cute.compile bypasses its IR cache",
    }
