"""Prepared host dispatch over original native recall and record-transfer functions.

Importing this module does not compile, load CUDA modules or initialize a device.
Pool construction explicitly prepares native bindings before creating storage or
sessions. Requests only consume those bindings and never rebuild after failure.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
from functools import cache
from pathlib import Path

_SOURCE = Path(__file__).with_name("csrc") / "echo_recall_dispatch.cpp"
_CFLAGS = ["-O3", "-std=c++17"]
_BINDINGS = {}
_RUNTIME_INFO = None


def _digest(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        result = hashlib.file_digest(stream, "sha256").hexdigest()
        finished = os.fstat(stream.fileno())
    after = path.stat()
    if any(
        len({getattr(state, key) for state in (before, opened, finished, after)}) != 1
        for key in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    ):
        raise RuntimeError(f"recall dispatch build input changed while hashing: {path}")
    return result


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def _cuda_home():
    location = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if location is None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise FileNotFoundError("recall dispatch cannot locate CUDA headers")
        location = Path(nvcc).resolve().parent.parent
    return Path(location).resolve()


def build_info():
    """Fresh actual preprocessing closure and toolchain identity; no CUDA calls."""
    import tvm_ffi.cpp.extension
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path, find_libtvm_ffi

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler:
        raise ValueError("recall dispatch compiler command is empty")
    resolved = shutil.which(compiler[0])
    if resolved is None:
        raise FileNotFoundError(f"recall dispatch compiler not found: {compiler[0]}")
    compiler[0] = str(Path(resolved).resolve())
    cuda = _cuda_home()
    cuda_include = str((cuda / "include").resolve())
    includes = [find_include_path(), find_dlpack_include_path(), cuda_include]
    # Match TVM FFI's host-C++ rule. Its generator source is also fingerprinted.
    flags = ["-std=c++17", "-fPIC", "-O2", *_CFLAGS]
    preprocessing = [*compiler, *flags, *(f"-I{p}" for p in includes), str(_SOURCE)]
    dependency_text = _run([*preprocessing, "-M", "-MT", "dependencies"])
    _, separator, dependencies = dependency_text.replace("\\\n", " ").partition(":")
    if not separator:
        raise RuntimeError("compiler did not report recall dispatch dependencies")
    linked = [Path(find_libtvm_ffi()).resolve(), (cuda / "lib64/libcudart.so").resolve(strict=True)]
    paths = {
        _SOURCE.resolve(),
        Path(__file__).resolve(),
        Path(compiler[0]),
        Path(tvm_ffi.cpp.extension.__file__).resolve(),
        *linked,
        *(Path(p).resolve() for p in shlex.split(dependencies.replace("$$", "$"))),
    }
    tools = {}
    for name in ("cc1plus", "as", "ld", "collect2"):
        reported = _run([*compiler, f"-print-prog-name={name}"]).strip()
        tool = shutil.which(reported)
        if tool is not None:
            tools[name] = str(Path(tool).resolve())
            paths.add(Path(tool).resolve())
    ldflags = [f"-L{cuda / 'lib64'}", "-lcudart"]
    identity = {
        "schema": "deepseek-host-recall-dispatch-build-v1",
        "tvm_ffi_version": importlib.metadata.version("apache-tvm-ffi"),
        "compiler_command": compiler,
        "compiler_version": _run([*compiler, "--version"]).strip(),
        "compiler_tools": tools,
        "compiler_environment": {
            name: os.environ.get(name)
            for name in (
                "PATH",
                "CXX",
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "LIBRARY_PATH",
                "COMPILER_PATH",
                "GCC_EXEC_PREFIX",
                "CUDA_HOME",
                "CUDA_PATH",
            )
        },
        "preprocessing_command": preprocessing,
        "extra_cflags": _CFLAGS,
        "extra_include_paths": [cuda_include],
        "extra_ldflags": ldflags,
        "linked_libraries": [str(p) for p in linked],
        "source_and_dependency_sha256": {str(p): _digest(p) for p in sorted(paths)},
        "boundary": "Host C++ dispatch only; original ECHO/common modules own all CUDA kernels.",
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"available": True, "fingerprint": fingerprint, "identity": identity}


@cache
def _module():
    global _RUNTIME_INFO
    import tvm_ffi.cpp

    info = build_info()
    details = info["identity"]
    module = tvm_ffi.cpp.load(
        name=f"cxldsagr_echo_recall_dispatch_{info['fingerprint'][:16]}",
        sources=[str(_SOURCE)],
        extra_cflags=details["extra_cflags"],
        extra_include_paths=details["extra_include_paths"],
        extra_ldflags=details["extra_ldflags"],
    )
    if build_info() != info:
        raise RuntimeError("recall dispatch dependencies changed during native preparation")
    _RUNTIME_INFO = info
    return module


def runtime_info():
    """Identity retained after successful module build/load, or None before use."""
    return copy.deepcopy(_RUNTIME_INFO)


def initialize(device):
    """Build and bind once per CUDA device before pool storage/session creation."""
    import torch

    from operators.common import kv_transfer
    from operators.deepseek_v32.indexer.echo import _module as echo_module

    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("recall dispatch initialization requires an explicit CUDA device")
    if device in _BINDINGS:
        return
    with torch.cuda.device(device):
        module = _module()
        echo, common = echo_module(), kv_transfer._module()
        prepared = {free: module.make_bridge(echo, common, free) for free in (False, True)}
    # Both native allocator paths must bind successfully before publishing state.
    _BINDINGS[device] = prepared


def prepared_call(device, free_only):
    """Return a prepared native Function; requests cannot trigger initialization."""
    try:
        prepared = _BINDINGS[device]
    except KeyError:
        raise RuntimeError("native recall dispatch was not initialized before pool use") from None
    return prepared[free_only]
