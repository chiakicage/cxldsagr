"""Authenticated CPython pool-referrer adapter.

Import and CPU budgeting do not prepare the adapter. Setup and runtime errors
propagate, including unsupported ABI and audit/callback failures. Failed setup
does not publish a provider selection or switch to the Python implementation.
The exact private ABI remains restricted to the checked executable and headers.
"""

from __future__ import annotations

import ctypes
import errno
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
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, ModuleType

_NAME = "_cxldsagr_nosa_pool_referrers"
_ABI_MANIFEST_SHA256 = "e0537a6cc6c16867894a36bff1d84bc54b53cce64de99c5a0886b34b3bdc8079"
_FLAGS = ("-O3", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror")
_setup_claim = {}
_selection = None


class _SetupUnavailable(RuntimeError):
    """An identified setup limitation, propagated without provider fallback."""


def identity_resolver(target):
    return target


def _stat_identity(stat):
    return tuple(
        getattr(stat, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def _digest(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        value = hashlib.file_digest(stream, "sha256").hexdigest()
        finished = os.fstat(stream.fileno())
    after = path.stat()
    if len({_stat_identity(s) for s in (before, opened, finished, after)}) != 1:
        raise _SetupUnavailable(f"pool-referrer input changed while hashing: {path}")
    return value


def _fingerprint(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _sources():
    root = Path(__file__).resolve().parent
    return (
        Path(__file__).resolve(),
        root / "csrc/pool_referrers.c",
        root / "csrc/pool_referrers_abi.json",
    )


def _read_abi():
    path = _sources()[2]
    before = path.stat()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read()
        finished = os.fstat(stream.fileno())
    after = path.stat()
    if len({_stat_identity(s) for s in (before, opened, finished, after)}) != 1:
        raise _SetupUnavailable("pool-referrer ABI whitelist changed while reading")
    if hashlib.sha256(raw).hexdigest() != _ABI_MANIFEST_SHA256:
        raise _SetupUnavailable("pool-referrer ABI whitelist bytes changed")
    return json.loads(raw)


def _symbol_hosts(names, executable):
    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("filename", ctypes.c_char_p),
            ("base", ctypes.c_void_p),
            ("symbol", ctypes.c_char_p),
            ("address", ctypes.c_void_p),
        ]

    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    hosts = {}
    for name in names:
        address = (
            ctypes.addressof(ctypes.c_char.in_dll(ctypes.pythonapi, name))
            if name == "_PyRuntime"
            else ctypes.cast(getattr(ctypes.pythonapi, name), ctypes.c_void_p).value
        )
        info = DlInfo()
        if (
            not dladdr(address, ctypes.byref(info))
            or not info.filename
            or not info.base
            or Path(info.filename.decode()).resolve() != executable
        ):
            raise _SetupUnavailable(f"Python symbol host differs from static executable: {name}")
        hosts[name] = {"host": str(executable), "offset": address - info.base}
    return hosts


def _authenticate_runtime(abi):
    if (
        sys.implementation.name != "cpython"
        or not sys.platform.startswith("linux")
        or sysconfig.get_config_var("Py_GIL_DISABLED")
        or sys.version != abi["python_version"]
    ):
        raise _SetupUnavailable("pool-referrer adapter requires the pinned Linux CPython with GIL")
    executable = Path(sys.executable).resolve()
    actual = Path("/proc/self/exe")
    if actual.resolve() != executable:
        raise _SetupUnavailable("running Python ELF differs from sys.executable")
    running, named = actual.stat(), executable.stat()
    if (running.st_dev, running.st_ino) != (named.st_dev, named.st_ino):
        raise _SetupUnavailable("running Python ELF was replaced")
    executable_hash = _digest(actual)
    if executable_hash != abi["executable_sha256"] or _digest(executable) != executable_hash:
        raise _SetupUnavailable("running Python executable differs from ABI whitelist")
    mappings = Path("/proc/self/maps").read_text().splitlines()
    if abi["runtime_linkage"] != "static_executable" or any(
        len(fields := line.split(maxsplit=5)) == 6 and "/libpython" in fields[5]
        for line in mappings
    ):
        raise _SetupUnavailable("Python linkage differs from static-executable whitelist")
    return {
        "python": sys.version,
        "executable": str(executable),
        "executable_sha256": executable_hash,
        "runtime_linkage": "static_executable",
        "symbol_hosts": _symbol_hosts(abi["symbol_hosts"], executable),
        "binding_precheck_symbol_hosts": _symbol_hosts(
            ("PyCFunction_GetFunction", "PyCFunction_GetFlags", "PyCFunction_GetSelf"), executable
        ),
    }


def _check_builtin_bindings():
    """Inspect public module method tables only after exact runtime authentication."""

    class MethodDef(ctypes.Structure):
        _fields_ = [
            ("name", ctypes.c_char_p),
            ("method", ctypes.c_void_p),
            ("flags", ctypes.c_int),
            ("doc", ctypes.c_char_p),
        ]

    class ModuleDefPrefix(ctypes.Structure):
        # Public PyModuleDef_Base followed by PyModuleDef through m_methods.
        _fields_ = [
            ("refcount", ctypes.c_ssize_t),
            ("type", ctypes.c_void_p),
            ("init", ctypes.c_void_p),
            ("index", ctypes.c_ssize_t),
            ("copy", ctypes.c_void_p),
            ("name", ctypes.c_char_p),
            ("doc", ctypes.c_char_p),
            ("size", ctypes.c_ssize_t),
            ("methods", ctypes.POINTER(MethodDef)),
        ]

    get_function = ctypes.pythonapi.PyCFunction_GetFunction
    get_function.argtypes = [ctypes.py_object]
    get_function.restype = ctypes.c_void_p
    get_flags = ctypes.pythonapi.PyCFunction_GetFlags
    get_flags.argtypes = [ctypes.py_object]
    get_flags.restype = ctypes.c_int
    get_self = ctypes.pythonapi.PyCFunction_GetSelf
    get_self.argtypes = [ctypes.py_object]
    get_self.restype = ctypes.c_void_p
    for name, bindings in (
        ("gc", ("get_objects", "get_referrers", "get_stats", "get_freeze_count")),
        ("_signal", ("getsignal", "default_int_handler", "valid_signals")),
    ):
        module = sys.modules.get(name)
        if type(module) is not ModuleType:
            raise _SetupUnavailable(f"{name} must be an exact builtin module")
        namespace = ModuleType.__getattribute__(module, "__dict__")
        initializer = getattr(ctypes.pythonapi, f"PyInit_{name}")
        initializer.argtypes = []
        initializer.restype = ctypes.POINTER(ModuleDefPrefix)
        methods = initializer().contents.methods
        expected = {}
        for index in range(512):
            method = methods[index]
            if method.name is None:
                break
            expected[method.name.decode()] = (method.method, method.flags)
        else:
            raise _SetupUnavailable(f"{name} builtin method table is not terminated")
        for binding in bindings:
            value = namespace.get(binding)
            if (
                type(value) is not BuiltinFunctionType
                or binding not in expected
                or (get_function(value), get_flags(value)) != expected[binding]
                or get_self(value) != id(module)
            ):
                raise _SetupUnavailable(
                    f"{name}.{binding} differs from the pinned builtin definition"
                )


def _run(command, environment):
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=120, check=False
    )
    if result.returncode:
        raise _SetupUnavailable(
            f"pool-referrer compiler exited {result.returncode}: {result.stderr.strip()[:2000]}"
        )
    return result.stdout


def _build_identity():
    """Authenticate and rediscover actual preprocessing inputs; never compile/load."""
    abi = _read_abi()
    runtime = _authenticate_runtime(abi)
    environment = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C"}
    compiler = shlex.split(os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc")
    if not compiler:
        raise _SetupUnavailable("pool-referrer compiler command is empty")
    resolved = shutil.which(compiler[0], path=environment["PATH"])
    if resolved is None:
        raise _SetupUnavailable(f"pool-referrer compiler is unavailable: {compiler[0]}")
    compiler[0] = str(Path(resolved).resolve())
    if not re.fullmatch(
        r"(?:.*-)?(?:gcc|clang|cc)(?:-\d+(?:\.\d+)*)?", Path(compiler[0]).name
    ) or any(argument != "-pthread" for argument in compiler[1:]):
        raise _SetupUnavailable(
            "pool-referrer build requires a direct C compiler without custom flags"
        )
    source = _sources()[1]
    includes = sorted(
        {
            str(Path(path).resolve())
            for path in (sysconfig.get_path("include"), sysconfig.get_path("platinclude"))
            if path
        }
    )
    command = [*compiler, *_FLAGS, *(f"-I{path}" for path in includes), str(source)]
    dependency_text = _run([*command, "-M", "-MT", "dependencies"], environment)
    _, separator, dependencies = dependency_text.replace("\\\n", " ").partition(":")
    if not separator:
        raise _SetupUnavailable("compiler did not report pool-referrer dependencies")
    paths = {Path(p).resolve() for p in shlex.split(dependencies.replace("$$", "$"))}
    if source not in paths:
        raise _SetupUnavailable("compiler dependency closure omitted pool-referrer source")
    headers = paths - {source}
    if {str(p) for p in headers} != set(abi["headers"]) or len(headers) != 335:
        raise _SetupUnavailable("rediscovered pool-referrer header closure differs from whitelist")
    actual_headers = {str(p): _digest(p) for p in sorted(headers)}
    if actual_headers != abi["headers"]:
        raise _SetupUnavailable("pool-referrer ABI header bytes differ from whitelist")
    paths.update((*_sources(), Path(runtime["executable"]), Path(compiler[0])))
    tools = {}
    for name in ("cc1", "as", "ld", "collect2"):
        reported = _run([*compiler, f"-print-prog-name={name}"], environment).strip()
        path = shutil.which(reported, path=environment["PATH"])
        if path is not None:
            tools[name] = str(Path(path).resolve())
            paths.add(Path(path).resolve())
    libraries = {}
    for name in (
        "libgcc.a",
        "libgcc_s.so",
        "libc.so",
        "crtbeginS.o",
        "crtendS.o",
        "crti.o",
        "crtn.o",
    ):
        reported = _run([*compiler, f"-print-file-name={name}"], environment).strip()
        if reported != name and Path(reported).is_file():
            libraries[name] = str(Path(reported).resolve())
            paths.add(Path(reported).resolve())
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith("libc.so"):
            path = Path(fields[5]).resolve()
            libraries["mapped_libc"] = str(path)
            paths.add(path)
    identity = {
        "schema": "nosa-pool-referrers-build-v1",
        "module_name": _NAME,
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "soabi": sysconfig.get_config_var("SOABI"),
        "abi_manifest_sha256": _ABI_MANIFEST_SHA256,
        "authenticated_runtime": runtime,
        "header_count": len(headers),
        "headers_sha256": actual_headers,
        "compiler_command": compiler,
        "compiler_version": _run([*compiler, "--version"], environment).strip(),
        "compiler_tools": tools,
        "compiler_runtime_libraries": libraries,
        "compile_command_without_output": command,
        "compiler_environment": environment,
        "source_and_dependency_sha256": {str(p): _digest(p) for p in sorted(paths)},
        "boundary": "Exact static CPython executable, critical symbol hosts, pinned and freshly rediscovered 335-header private ABI, loader/C/whitelist and C compiler/link inputs. CPU-only build; no CUDA or torch initialization.",
    }
    if not identity["extension_suffix"]:
        raise _SetupUnavailable("Python extension suffix is unavailable")
    return identity, command, environment


def build_info():
    identity, _, _ = _build_identity()
    return {"available": True, "fingerprint": _fingerprint(identity), "identity": identity}


def _verified_manifest(directory, identity):
    manifest_path = directory / "manifest.json"
    binary = directory / (_NAME + identity["extension_suffix"])
    if not manifest_path.is_file() or not binary.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if (
        type(manifest) is dict
        and manifest.get("build_identity") == identity
        and manifest.get("binary_sha256") == _digest(binary)
    ):
        return manifest
    return None


def _verify_build_identity(expected):
    current, _, _ = _build_identity()
    if current != expected:
        raise _SetupUnavailable("pool-referrer dependency changed during build/load preparation")


def _loaded_binary_digest(binary, maps_path=Path("/proc/self/maps")):
    before = binary.stat()
    matched = False
    for line in maps_path.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        name = fields[5]
        deleted = name.endswith(" (deleted)")
        path = Path(name.removesuffix(" (deleted)"))
        if path.resolve() != binary:
            continue
        if deleted:
            raise _SetupUnavailable("mapped pool-referrer binary was deleted")
        device = tuple(int(value, 16) for value in fields[3].split(":"))
        if (
            device != (os.major(before.st_dev), os.minor(before.st_dev))
            or int(fields[4]) != before.st_ino
        ):
            raise _SetupUnavailable("mapped pool-referrer binary differs from file inode/device")
        matched = True
    if not matched:
        raise _SetupUnavailable("authenticated pool-referrer binary is not mapped")
    digest = _digest(binary)
    if _stat_identity(before) != _stat_identity(binary.stat()):
        raise _SetupUnavailable("mapped pool-referrer file changed during verification")
    return digest


def _load_native():
    import fcntl

    identity, command, environment = _build_identity()
    _check_builtin_bindings()
    # Check the syscall result directly: Python audit callbacks can themselves
    # raise BlockingIOError, which must not be confused with lock contention.
    flock = ctypes.CDLL(None, use_errno=True).flock
    flock.argtypes = [ctypes.c_int, ctypes.c_int]
    flock.restype = ctypes.c_int
    fingerprint = _fingerprint(identity)
    cache = (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "cxldsagr/pool-referrers"
    )
    cache.mkdir(parents=True, exist_ok=True)
    directory = cache / fingerprint
    with (cache / f"{fingerprint}.lock").open("a") as lock:
        setup_failure = None
        try:
            if flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB):
                error = ctypes.get_errno()
                if error == errno.EWOULDBLOCK:
                    raise _SetupUnavailable(
                        "pool-referrer cache preparation is already in progress"
                    )
                raise OSError(error, os.strerror(error))
            manifest = _verified_manifest(directory, identity)
            if manifest is None:
                if directory.exists():
                    raise _SetupUnavailable(f"pool-referrer cache integrity failed: {directory}")
                with tempfile.TemporaryDirectory(prefix=f".{fingerprint}-", dir=cache) as temporary:
                    staging = Path(temporary)
                    binary = staging / (_NAME + identity["extension_suffix"])
                    _run([*command, "-o", str(binary)], environment)
                    _verify_build_identity(identity)
                    # Some filesystems normalize a new file's ctime on first read.
                    # Materialize it before the independent, strict stat-bound hash.
                    binary.read_bytes()
                    manifest = {"build_identity": identity, "binary_sha256": _digest(binary)}
                    (staging / "manifest.json").write_text(
                        json.dumps(manifest, sort_keys=True) + "\n"
                    )
                    for path in (binary, staging / "manifest.json"):
                        with path.open("rb") as stream:
                            os.fsync(stream.fileno())
                    staging.rename(directory)
            else:
                _verify_build_identity(identity)
            binary = (directory / (_NAME + identity["extension_suffix"])).resolve()
            _check_builtin_bindings()
            spec = importlib.util.spec_from_file_location(_NAME, binary)
            if spec is None or spec.loader is None:
                raise _SetupUnavailable("cannot create pool-referrer extension spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded_path = Path(module.__file__).resolve()
            loaded_digest = _loaded_binary_digest(loaded_path)
            if loaded_path != binary or loaded_digest != manifest["binary_sha256"]:
                raise _SetupUnavailable("loaded pool-referrer adapter differs from verified binary")
        except BaseException as error:
            setup_failure = error
            raise
        finally:
            try:
                if flock(lock.fileno(), fcntl.LOCK_UN):
                    error = ctypes.get_errno()
                    raise OSError(error, os.strerror(error))
            except BaseException as cleanup_error:
                if setup_failure is not None:
                    raise BaseExceptionGroup(
                        "Pool-referrer setup and lock release failed",
                        [setup_failure, cleanup_error],
                    ) from None
                raise
    return module, {
        "backend": "cpython_native",
        "fallback_reason": None,
        "fingerprint": fingerprint,
        "abi_manifest_sha256": _ABI_MANIFEST_SHA256,
        "loaded_binary_path": str(loaded_path),
        "loaded_binary_sha256": loaded_digest,
    }


def prepare(helper, pool_types):
    """Publish one successful setup; reject concurrent or reentrant preparation."""
    global _selection
    if _selection is not None:
        return _selection[0]
    token = object()
    selected = None
    try:
        # One builtin dict operation claims setup under the supported CPython GIL.
        # No Python lock is held across lines where tracing/callbacks can reenter.
        owner = _setup_claim.setdefault("owner", token)
        if owner is not token:
            raise RuntimeError("pool-referrer preparation is already in progress")
        # A trace callback before the claim may already have completed setup.
        if _selection is not None:
            return _selection[0]
        if type(helper) is not FunctionType or type(pool_types) is not FunctionType:
            raise _SetupUnavailable("pool-referrer helper and inventory must be Python functions")
        module, runtime = _load_native()
        module.configure_helper(helper, pool_types)
        selected = (module.resolve_referrers, runtime, module)
    finally:
        # Publish one completed tuple; failures leave the provider unselected.
        if _setup_claim.get("owner") is token:
            if selected is not None:
                _selection = selected
            _setup_claim.pop("owner", None)
    return selected[0]


def runtime_info():
    """Retained provider and native counters only; never prepare or inspect files."""
    selected = _selection
    if selected is None:
        return None
    _, runtime, module = selected
    return {**runtime, "counters": module.counters()}
