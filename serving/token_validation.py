"""Prepare one CPU token predicate before serving; importing needs no compiler."""

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
from itertools import repeat
from operator import is_
from pathlib import Path

_NAME = "_cxldsagr_token_validation"
_FLAGS = ("-O3", "-Wall", "-Wextra", "-Werror", "-std=c++17", "-fPIC", "-shared")
_LOCK = threading.Lock()
_predicate = None
_runtime = None


def reference(snapshot: list[int]) -> bool:
    """Original Python scan, also available on systems without a native compiler."""
    if type(snapshot) is not list:
        raise TypeError("expected an exact list snapshot")
    return bool(snapshot) and all(map(is_, map(type, snapshot), repeat(int))) and min(snapshot) >= 0


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fingerprint(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _unsupported_reason() -> str | None:
    if sys.implementation.name != "cpython":
        return "native token validation requires CPython"
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        return "native token validation requires a CPython build with the GIL"
    if not sys.platform.startswith("linux"):
        return "native token validation loader supports Linux"
    return None


def _run(command: list[str], environment: dict[str, str]) -> str:
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=60, check=False
    )
    if result.returncode:
        raise RuntimeError(f"compiler exited {result.returncode}: {result.stderr.strip()[:2000]}")
    return result.stdout


def _build_identity() -> tuple[dict, list[str], dict[str, str]]:
    # Use only this declared environment for dependency discovery and compilation.
    environment = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C"}
    compiler = shlex.split(os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++")
    if not compiler:
        raise ValueError("token validation compiler command is empty")
    resolved = shutil.which(compiler[0], path=environment["PATH"])
    if resolved is None:
        raise FileNotFoundError(f"token validation compiler not found: {compiler[0]}")
    compiler[0] = str(Path(resolved).resolve())
    # Launcher payloads and custom tool/plugin flags need identities of their own.
    # This small loader supports direct GNU/Clang drivers, including sysconfig's
    # usual -pthread, rather than silently fingerprinting only a wrapper binary.
    if not re.fullmatch(
        r"(?:.*-)?(?:g\+\+|clang(?:\+\+)?|c\+\+)(?:-\d+(?:\.\d+)*)?",
        Path(compiler[0]).name,
    ) or any(argument != "-pthread" for argument in compiler[1:]):
        raise ValueError(
            "token validation requires a direct GNU/Clang C++ driver without custom flags"
        )
    source = Path(__file__).resolve().parent / "csrc" / "token_validation.cpp"
    includes = sorted({sysconfig.get_path("include"), sysconfig.get_path("platinclude")} - {None})
    command = [*compiler, *_FLAGS, *(f"-I{path}" for path in includes), str(source)]
    dependencies = _run([*command, "-M", "-MT", "dependencies"], environment)
    _, separator, dependencies = dependencies.replace("\\\n", " ").partition(":")
    if not separator:
        raise RuntimeError("compiler did not report token validation header dependencies")
    paths = {
        source,
        Path(__file__).resolve(),
        Path(sys.executable).resolve(),
        Path(compiler[0]),
        *(Path(path).resolve() for path in shlex.split(dependencies.replace("$$", "$"))),
    }
    # GCC uses separate frontend/assembler/linker tools; Clang may integrate them.
    tools = {}
    for name in ("cc1plus", "as", "ld", "collect2"):
        reported = _run([*compiler, f"-print-prog-name={name}"], environment).strip()
        resolved = shutil.which(reported, path=environment["PATH"])
        if resolved is not None:
            path = Path(resolved).resolve()
            tools[name] = str(path)
            paths.add(path)
    identity = {
        "schema_version": 1,
        "python": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "compiler_command": compiler,
        "compiler_version": _run([*compiler, "--version"], environment).strip(),
        "compiler_tools": tools,
        "compile_command_without_output": command,
        "compiler_environment": environment,
        "source_and_dependency_sha256": {str(path): _digest(path) for path in sorted(paths)},
    }
    return identity, command, environment


def _verified_manifest(directory: Path, identity: dict) -> dict | None:
    try:
        directory.lstat()
    except FileNotFoundError:
        return None
    manifest = json.loads((directory / "manifest.json").read_text())
    binary = directory / (_NAME + identity["extension_suffix"])
    if manifest["build_identity"] != identity:
        raise RuntimeError("cached token validation build identity differs from expected identity")
    if manifest["binary_sha256"] != _digest(binary):
        raise RuntimeError("cached token validation binary differs from its verified digest")
    return manifest


def _load_native() -> tuple[object, dict]:
    import fcntl

    identity, command, environment = _build_identity()
    key = _fingerprint(identity)
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    cache = cache / "cxldsagr" / "token-validation"
    cache.mkdir(parents=True, exist_ok=True)
    directory = cache / key
    # Serialize first publication and verified reuse without repairing bad artifacts.
    with (cache / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _verified_manifest(directory, identity)
        if manifest is None:
            staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=cache))
            try:
                binary = staging / (_NAME + identity["extension_suffix"])
                _run([*command, "-o", str(binary)], environment)
                if any(
                    _digest(Path(path)) != digest
                    for path, digest in identity["source_and_dependency_sha256"].items()
                ):
                    raise RuntimeError("token validation dependency changed during compilation")
                manifest = {"build_identity": identity, "binary_sha256": _digest(binary)}
                (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
                staging.rename(directory)
            except BaseException as error:
                error.add_note(f"token validation build artifacts retained at {staging}")
                raise
        binary = (directory / (_NAME + identity["extension_suffix"])).resolve()
        spec = importlib.util.spec_from_file_location(_NAME, binary)
        if spec is None or spec.loader is None:
            raise ImportError("cannot create token validation extension spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded_path = Path(module.__file__).resolve()
        loaded_digest = _digest(loaded_path)
        if loaded_path != binary or loaded_digest != manifest["binary_sha256"]:
            raise RuntimeError("loaded token validation binary differs from verified artifact")
    return module.nonnegative_builtin_ints, {
        "backend": "cpython_native",
        "fallback_reason": None,
        "fingerprint": key,
        "build_identity": identity,
        "loaded_binary_path": str(loaded_path),
        "loaded_binary_sha256": loaded_digest,
    }


def prepare():
    """Prepare native once; setup errors propagate before runner ownership."""
    global _predicate, _runtime
    with _LOCK:
        if _predicate is None:
            reason = _unsupported_reason()
            if reason is not None:
                raise NotImplementedError(reason)
            _predicate, _runtime = _load_native()
        return _predicate


def runtime_info() -> dict | None:
    """Return retained setup identity, without loading, compiling or filesystem I/O."""
    return copy.deepcopy(_runtime)
