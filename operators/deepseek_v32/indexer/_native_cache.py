"""Immutable, source-bound TVM-FFI artifacts for independent ECHO adapters."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

_LOADED = {}
_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _command_identity(command):
    argv = shlex.split(command)
    executable = shutil.which(argv[0])
    if executable is None:
        raise FileNotFoundError(f"Compiler not found: {argv[0]}")
    resolved = Path(executable).resolve()
    return {
        "command": argv,
        "executable": str(resolved),
        "sha256": _sha256(resolved),
        "version": subprocess.check_output([*argv, "--version"], text=True).strip(),
    }


def _environment_identity():
    import tvm_ffi
    from tvm_ffi.cpp.extension import _find_cuda_home

    ffi_root = Path(tvm_ffi.__file__).resolve().parent
    cuda_root = Path(_find_cuda_home()).resolve()
    return {
        "cxx": _command_identity(os.environ.get("CXX", "c++")),
        "cc": _command_identity(os.environ.get("CC", "cc")),
        "nvcc": _command_identity(str(cuda_root / "bin/nvcc")),
        "cuda_compiler_sha256": {
            str(path): _sha256(path)
            for path in (
                cuda_root / "bin/ptxas",
                cuda_root / "bin/nvlink",
                cuda_root / "nvvm/bin/cicc",
            )
        },
        "tvm_ffi_version": tvm_ffi.__version__,
        "tvm_ffi_root": str(ffi_root),
        "tvm_ffi_sha256": {
            str(path.relative_to(ffi_root)): _sha256(path)
            for path in sorted(ffi_root.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".so", ".h", ".hpp"}
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CC",
                "CXX",
                "CUDA_HOME",
                "CUDA_PATH",
                "TVM_FFI_GPU_BACKEND",
                "TVM_FFI_CUDA_ARCH_LIST",
            )
        },
    }


def _validate(entry, name, key, identity):
    record = json.loads((entry / "record.json").read_text())
    if (
        record.get("schema") != 1
        or record.get("cache_key") != key
        or record.get("build_identity") != identity
        or record.get("artifact_name") != f"{name}.so"
    ):
        raise RuntimeError(f"Invalid immutable native cache record: {entry}")
    artifact = entry / record["artifact_name"]
    if _sha256(artifact) != record["artifact_sha256"]:
        raise RuntimeError(f"Immutable native artifact hash mismatch: {artifact}")
    return {**record, "artifact_path": str(artifact.resolve())}


def _verify_sources(identity):
    declared = identity.get("source_sha256")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("Native source identity requires its complete source_sha256 closure")
    for name, expected in declared.items():
        path = Path(name)
        if not path.is_absolute():
            path = _ROOT / path
        if _sha256(path) != expected:
            raise RuntimeError(f"Native source changed during build: {path}")


def load(*, name, sources, extra_include_paths, extra_cuda_cflags, extra_ldflags, source_identity):
    """Build once, then load precisely those bytes for the same build identity.

    Existing corrupt records/artifacts fail. This is a normal exact cache hit,
    never a retry, recovery, or backend fallback. Callers provide the complete
    project header closure in source_identity; compiler and TVM ABI are added
    here. The per-key file lock serializes publication across processes.
    """
    import tvm_ffi
    import tvm_ffi.cpp

    if Path(name).name != name:
        raise ValueError("Native module name must be one path component")
    _verify_sources(source_identity)
    identity = json.loads(
        json.dumps(
            {
                "source_identity": source_identity,
                "sources_sha256": {str(Path(path).resolve()): _sha256(path) for path in sources},
                "loader_sha256": _sha256(__file__),
                "include_paths": [str(Path(path).resolve()) for path in extra_include_paths],
                "cuda_flags": list(extra_cuda_cflags),
                "link_flags": list(extra_ldflags),
                "toolchain": _environment_identity(),
            },
            sort_keys=True,
        )
    )
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    key = f"{name}_{digest}"
    cache_root = Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")).expanduser()
    root = cache_root / "immutable"
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    entry = root / key
    with (locks / f"{key}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not entry.exists():
            staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
            try:
                artifact = staging / f"{name}.so"
                tvm_ffi.cpp.build(
                    name=name,
                    sources=sources,
                    extra_include_paths=extra_include_paths,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_ldflags=extra_ldflags,
                    build_directory=str(staging / "build"),
                    output=str(artifact),
                )
                _verify_sources(source_identity)
                if _environment_identity() != identity["toolchain"]:
                    raise RuntimeError("Native compiler or TVM-FFI identity changed during build")
                if any(
                    _sha256(path) != expected
                    for path, expected in identity["sources_sha256"].items()
                ):
                    raise RuntimeError("Native translation unit changed during build")
                record = {
                    "schema": 1,
                    "cache_key": key,
                    "build_identity": identity,
                    "artifact_name": artifact.name,
                    "artifact_sha256": _sha256(artifact),
                }
                with (staging / "record.json").open("x") as stream:
                    json.dump(record, stream, sort_keys=True, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                with artifact.open("rb") as stream:
                    os.fsync(stream.fileno())
                artifact.chmod(0o444)
                (staging / "record.json").chmod(0o444)
                staging.rename(entry)
            except BaseException as error:
                try:
                    shutil.rmtree(staging)
                except BaseException as cleanup_error:  # noqa: BLE001 - preserve both failures
                    raise BaseExceptionGroup(
                        "Native build and staging cleanup failed", [error, cleanup_error]
                    ) from None
                raise
        record = _validate(entry, name, key, identity)
        module = tvm_ffi.load_module(record["artifact_path"])
        _LOADED[name] = record
        return module


def native_info(name):
    """Return the actual loaded immutable artifact and its complete identity."""
    record = _LOADED[name]
    if _sha256(record["artifact_path"]) != record["artifact_sha256"]:
        raise RuntimeError("Loaded native artifact changed on disk")
    return copy.deepcopy(record)
