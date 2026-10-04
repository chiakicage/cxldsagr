"""Private fresh allocator snapshots; import and CPU budgeting never compile.

The optional CPython adapter omits block dictionaries and deduplicates only the
segment ownership/expandability predicates used by NOSA's existing guard. Every
call still takes a fresh all-pool C++ allocator snapshot. Setup failure chooses
the official PyTorch API once per process and retains an explicit reason. A
failure while executing either snapshot is propagated, never hidden by retry.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from pathlib import Path

_NAME = "_cxldsagr_nosa_allocator_snapshot"
_FLAGS = ("-O3", "-std=c++17", "-fPIC", "-shared")
_LOCK = threading.Lock()
_provider = None
_runtime = None


def _digest(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        result = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise RuntimeError(f"allocator snapshot build input changed while hashing: {path}")
    return result


def _fingerprint(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _unsupported_reason():
    import torch

    from models.nosa.allocation_budget import SUPPORTED_TORCH_REVISION

    if sys.implementation.name != "cpython" or sysconfig.get_config_var("Py_GIL_DISABLED"):
        return "private allocator adapter requires CPython with the GIL"
    if not sys.platform.startswith("linux"):
        return "private allocator adapter loader requires Linux"
    if torch.version.git_version != SUPPORTED_TORCH_REVISION:
        return "private allocator ABI has not been verified for this PyTorch revision"
    if torch.version.cuda is None:
        return "PyTorch was built without CUDA allocator support"
    return None


def _run(command, environment):
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=120, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"allocator adapter compiler exited {result.returncode}: {result.stderr.strip()[:2000]}"
        )
    return result.stdout


def _build_identity():
    """Discover exact preprocessing/link inputs without compiling or using CUDA."""
    import torch

    reason = _unsupported_reason()
    if reason is not None:
        raise RuntimeError(reason)
    environment = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C"}
    compiler = shlex.split(os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++")
    if not compiler:
        raise ValueError("allocator adapter compiler command is empty")
    resolved = shutil.which(compiler[0], path=environment["PATH"])
    if resolved is None:
        raise FileNotFoundError(f"allocator adapter compiler not found: {compiler[0]}")
    compiler[0] = str(Path(resolved).resolve())
    if not re.fullmatch(
        r"(?:.*-)?(?:g\+\+|clang(?:\+\+)?|c\+\+)(?:-\d+(?:\.\d+)*)?", Path(compiler[0]).name
    ) or any(argument != "-pthread" for argument in compiler[1:]):
        raise ValueError(
            "allocator adapter requires a direct GNU/Clang driver without custom flags"
        )
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda is None:
        nvcc = shutil.which("nvcc", path=environment["PATH"])
        if nvcc is None:
            raise FileNotFoundError("allocator adapter cannot locate CUDA headers")
        cuda = str(Path(nvcc).resolve().parent.parent)
    torch_root = Path(torch.__file__).resolve().parent
    source = Path(__file__).resolve().parent / "csrc/allocator_snapshot.cpp"
    includes = sorted(
        {
            str(torch_root / "include"),
            str(Path(cuda).resolve() / "include"),
            sysconfig.get_path("include"),
            sysconfig.get_path("platinclude"),
        }
        - {None}
    )
    libraries = [torch_root / "lib/libc10_cuda.so", torch_root / "lib/libc10.so"]
    flags = [*_FLAGS, f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    preprocessing = [*compiler, *flags, *(f"-I{path}" for path in includes), str(source)]
    dependency_text = _run([*preprocessing, "-M", "-MT", "dependencies"], environment)
    _, separator, dependencies = dependency_text.replace("\\\n", " ").partition(":")
    if not separator:
        raise RuntimeError("compiler did not report allocator adapter dependencies")
    paths = {
        source,
        Path(__file__).resolve(),
        Path(sys.executable).resolve(),
        Path(compiler[0]),
        *(p.resolve() for p in libraries),
        *(Path(p).resolve() for p in shlex.split(dependencies.replace("$$", "$"))),
    }
    tools = {}
    for name in ("cc1plus", "as", "ld", "collect2"):
        reported = _run([*compiler, f"-print-prog-name={name}"], environment).strip()
        path = shutil.which(reported, path=environment["PATH"])
        if path is not None:
            tools[name] = str(Path(path).resolve())
            paths.add(Path(path).resolve())
    runtime_libraries = {}
    for name in ("libstdc++.so", "libgcc_s.so.1", "crtbeginS.o", "crtendS.o"):
        reported = _run([*compiler, f"-print-file-name={name}"], environment).strip()
        if reported != name and Path(reported).is_file():
            runtime_libraries[name] = str(Path(reported).resolve())
            paths.add(Path(reported).resolve())
    command = [*preprocessing, *(str(p) for p in libraries), f"-Wl,-rpath,{torch_root / 'lib'}"]
    identity = {
        "schema": "nosa-private-allocator-build-v1",
        "python": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "torch_version": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "compiler_command": compiler,
        "compiler_version": _run([*compiler, "--version"], environment).strip(),
        "compiler_tools": tools,
        "compiler_runtime_libraries": runtime_libraries,
        "compile_command_without_output": command,
        "compiler_environment": environment,
        "source_and_dependency_sha256": {str(p): _digest(p) for p in sorted(paths)},
        "boundary": "CPU-only adapter over the pinned PyTorch C++ CUDA allocator ABI; exact preprocessing closure, compiler tools, Python/torch ABI and linked libc10 libraries are identified. No GPU kernel compilation.",
    }
    return identity, command, environment


def build_info():
    """Fresh build provenance for experiment snapshots; does not compile/load."""
    try:
        identity, _, _ = _build_identity()
        return {"available": True, "fingerprint": _fingerprint(identity), "identity": identity}
    except (ImportError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        return {
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
            "source_sha256": {
                str(path): _digest(path)
                for path in (
                    Path(__file__).resolve(),
                    Path(__file__).resolve().parent / "csrc/allocator_snapshot.cpp",
                )
            },
        }


def _verified_manifest(directory, identity):
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        binary = directory / (_NAME + identity["extension_suffix"])
        if manifest["build_identity"] == identity and manifest["binary_sha256"] == _digest(binary):
            return manifest
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _verify_build_identity(expected):
    # Re-run preprocessing as well as hashing: an include added between the
    # initial dependency scan and hashing would not appear in the old closure.
    current, _, _ = _build_identity()
    if current != expected:
        raise RuntimeError("allocator adapter dependency changed during build/load preparation")


def _load_native():
    import fcntl

    identity, command, environment = _build_identity()
    fingerprint = _fingerprint(identity)
    cache = (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "cxldsagr/allocator-snapshot"
    )
    cache.mkdir(parents=True, exist_ok=True)
    directory = cache / fingerprint
    with (cache / f"{fingerprint}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _verified_manifest(directory, identity)
        if manifest is None:
            if directory.exists():
                # Never unlink or replace an artifact that another process may
                # already have mapped. Official inspection remains available.
                raise RuntimeError(f"allocator adapter cache integrity failed: {directory}")
            with tempfile.TemporaryDirectory(prefix=f".{fingerprint}-", dir=cache) as temporary:
                staging = Path(temporary)
                binary = staging / (_NAME + identity["extension_suffix"])
                _run([*command, "-o", str(binary)], environment)
                _verify_build_identity(identity)
                manifest = {"build_identity": identity, "binary_sha256": _digest(binary)}
                (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
                for path in (binary, staging / "manifest.json"):
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                staging.rename(directory)
        else:
            # Input discovery preceded flock; another process may have held it
            # while headers, compiler resolution or link inputs changed.
            _verify_build_identity(identity)
        binary = (directory / (_NAME + identity["extension_suffix"])).resolve()
        spec = importlib.util.spec_from_file_location(_NAME, binary)
        if spec is None or spec.loader is None:
            raise ImportError("cannot create allocator snapshot extension spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded_path = Path(module.__file__).resolve()
        loaded_digest = _digest(loaded_path)
        if loaded_path != binary or loaded_digest != manifest["binary_sha256"]:
            raise RuntimeError("loaded allocator adapter differs from verified artifact")
    return module.snapshot, {
        "backend": "private_cpp",
        "fallback_reason": None,
        "fingerprint": fingerprint,
        "loaded_binary_path": str(loaded_path),
        "loaded_binary_sha256": loaded_digest,
    }


def _official_snapshot():
    import torch

    return torch._C._cuda_memorySnapshot((0, 0, False))


def _prepare():
    global _provider, _runtime
    with _LOCK:
        if _provider is None:
            try:
                _provider, _runtime = _load_native()
            except (
                ImportError,
                OSError,
                RuntimeError,
                ValueError,
                subprocess.SubprocessError,
            ) as error:
                _provider = _official_snapshot
                _runtime = {
                    "backend": "torch_official",
                    "fallback_reason": f"{type(error).__name__}: {error}",
                }
        return _provider


def snapshot():
    """Return a fresh snapshot and small adapter evidence, without global patching."""
    provider = _prepare()
    return provider(), dict(_runtime)


def runtime_info():
    """Small retained adapter identity, without initialization or filesystem I/O."""
    return copy.deepcopy(_runtime)
