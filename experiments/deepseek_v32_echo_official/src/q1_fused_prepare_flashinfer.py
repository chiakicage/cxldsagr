"""Experiment-private, immutable loading of three unchanged FlashInfer kernels.

Install ``pinned()`` before any operator factory runs. The registered JitSpec
continues to expose the actual ELF and original Ninja recipe to the normal
FlashInfer provenance collector. Cache hits verify bytes and never invoke a
builder. Missing, incomplete, changed or unloadable artifacts are errors.
"""

from __future__ import annotations

import copy
import dataclasses
import fcntl
import hashlib
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = (
    ROOT / "experiments/deepseek_v32_echo_official/output/runtime/q1-fused-prepare-model/flashinfer"
)
MODULES = frozenset({"rope", "silu_and_mul", "topk"})
SCHEMA = "private-fused-prepare-flashinfer-immutable-v1"
_LOADED = {}
_ACTIVE = False
_ENVIRONMENT = (
    "CC",
    "CXX",
    "CUDA_HOME",
    "CUDA_PATH",
    "PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "FLASHINFER_NVCC",
    "FLASHINFER_CXX_LAUNCHER",
    "FLASHINFER_NVCC_LAUNCHER",
    "FLASHINFER_EXTRA_CFLAGS",
    "FLASHINFER_EXTRA_CUDAFLAGS",
    "FLASHINFER_EXTRA_LDFLAGS",
    "FLASHINFER_CUDA_ARCH_LIST",
    "FLASHINFER_JIT_DEBUG",
    "FLASHINFER_JIT_VERBOSE",
    "FLASHINFER_JIT_LINEINFO",
    "FLASHINFER_NVCC_THREADS",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _environment():
    return {name: os.environ.get(name) for name in _ENVIRONMENT}


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _verify_file(record):
    if _file(record["path"]) != record:
        raise RuntimeError(f"Immutable FlashInfer input/artifact changed: {record['path']}")


def _verify_files(value):
    if isinstance(value, dict):
        if set(value) == {"path", "bytes", "sha256"}:
            _verify_file(value)
        else:
            for child in value.values():
                _verify_files(child)
    elif isinstance(value, list):
        for child in value:
            _verify_files(child)


def _tree_files(roots):
    """Follow include-directory links, reject broken links, and avoid cycles."""
    pending = list(roots)
    directories, files = set(), set()
    while pending:
        path = Path(pending.pop()).resolve(strict=True)
        if path.is_dir():
            if path not in directories:
                directories.add(path)
                pending.extend(path.iterdir())
        elif path.is_file():
            files.add(path)
        else:
            raise RuntimeError(f"Unsupported FlashInfer include entry: {path}")
    return [_file(path) for path in sorted(files)]


def _command(command):
    parts = shlex.split(command)
    if not parts:
        raise RuntimeError("Empty FlashInfer compiler command")
    executable = shutil.which(parts[0])
    if executable is None:
        raise FileNotFoundError(parts[0])
    version = subprocess.run([*parts, "--version"], check=True, capture_output=True, text=True)
    return {
        "command": parts,
        "executable": _file(executable),
        "version": version.stdout + version.stderr,
    }


def _installed():
    # raw.environment() in the launcher must run before these lazy imports.
    return SimpleNamespace(
        core=importlib.import_module("flashinfer.jit.core"),
        env=importlib.import_module("flashinfer.jit.env"),
        cpp=importlib.import_module("flashinfer.jit.cpp_ext"),
        activation=importlib.import_module("flashinfer.jit.activation"),
        ffi=importlib.import_module("tvm_ffi"),
        flashinfer=importlib.import_module("flashinfer"),
        torch=importlib.import_module("torch"),
    )


def _request(spec):
    result = dataclasses.asdict(spec)
    result["sources"] = [str(Path(path).resolve()) for path in spec.sources]
    result["extra_include_dirs"] = (
        [str(Path(path).resolve()) for path in spec.extra_include_dirs]
        if spec.extra_include_dirs is not None
        else None
    )
    return result


def _validate_request(spec, installed):
    if type(spec) is not installed.core.JitSpecNvcc:
        raise RuntimeError("Unexpected FlashInfer spec implementation")
    sources = {
        "rope": ["rope.cu", "flashinfer_rope_binding.cu"],
        "topk": [
            "topk.cu",
            "flashinfer_topk_binding.cu",
            "flashinfer_fast_topk_clusters_binding.cu",
        ],
    }
    if spec.name == "silu_and_mul":
        expected_sources = [installed.env.FLASHINFER_GEN_SRC_DIR / "silu_and_mul.cu"]
        expected_text = installed.activation.get_act_and_mul_cu_str(
            "silu", installed.activation.act_func_def_str["silu"]
        )
        if expected_sources[0].read_text() != expected_text:
            raise RuntimeError("Unexpected FlashInfer silu generated source")
    else:
        expected_sources = [installed.env.FLASHINFER_CSRC_DIR / name for name in sources[spec.name]]
    expected = installed.core.gen_jit_spec(
        spec.name,
        expected_sources,
        extra_cuda_cflags=["-lineinfo"] if spec.name == "topk" else None,
    )
    if _request(spec) != _request(expected):
        raise RuntimeError(f"Unexpected FlashInfer build request: {spec.name}")
    if spec.is_aot:
        raise RuntimeError("Private FlashInfer harness requires the declared JIT build path")


def _ninja(spec, installed, jit_root):
    with patch.object(installed.env, "FLASHINFER_JIT_DIR", jit_root):
        return installed.core.generate_ninja_build_for_op(
            name=spec.name,
            sources=spec.sources,
            extra_cflags=spec.extra_cflags,
            extra_cuda_cflags=spec.extra_cuda_cflags,
            extra_ldflags=spec.extra_ldflags,
            extra_include_dirs=spec.extra_include_dirs,
            needs_device_linking=spec.needs_device_linking,
        )


def _input_identity(spec, installed):
    cuda = Path(installed.cpp.get_cuda_path()).resolve(strict=True)
    roots = [
        *(spec.extra_include_dirs or []),
        *installed.cpp.get_cccl_includes(),
        *[
            Path(str(path).replace("$cuda_home", str(cuda)))
            for path in installed.cpp.get_system_includes(str(cuda))
        ],
    ]
    commands = {
        "cxx": os.environ.get("CXX", "c++"),
        "nvcc": os.environ.get("FLASHINFER_NVCC", str(cuda / "bin/nvcc")),
        "ninja": "ninja",
    }
    if os.environ.get("CC"):
        commands["cc"] = os.environ["CC"]
    for variable in ("FLASHINFER_CXX_LAUNCHER", "FLASHINFER_NVCC_LAUNCHER"):
        if os.environ.get(variable):
            commands[variable] = os.environ[variable]
    compiler_files = []
    # These are the native compilation components used by the declared nvcc.
    for relative in ("bin/ptxas", "bin/nvlink", "nvvm/bin/cicc", "nvvm/libdevice/libdevice.10.bc"):
        compiler_files.append(_file(cuda / relative))
    cxx = shlex.split(commands["cxx"])
    for program in ("cc1plus", "ld"):
        resolved = subprocess.run(
            [*cxx, f"-print-prog-name={program}"], check=True, capture_output=True, text=True
        ).stdout.strip()
        compiler_files.append(_file(shutil.which(resolved) or resolved))
    python_sources = []
    for package in (installed.flashinfer, installed.ffi):
        root = Path(package.__file__).resolve().parent
        python_sources.extend(
            _file(path) for path in sorted(root.rglob("*")) if path.suffix in {".py", ".so"}
        )
    template = _ninja(spec, installed, RUNTIME / "ninja-template")
    return {
        "schema": SCHEMA,
        "loader": _file(__file__),
        "request": _request(spec),
        "sources": [_file(path) for path in spec.sources],
        "include_roots": [str(Path(path).resolve(strict=True)) for path in roots],
        "include_files": _tree_files(roots),
        "ninja_template": template,
        "ninja_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "ninja_diagnostics": ["-d", "keepdepfile"],
        "environment": _environment(),
        "toolchain": {
            "commands": {name: _command(command) for name, command in commands.items()},
            "compiler_components": compiler_files,
            "python": {"executable": _file(sys.executable), "version": sys.version},
            "binding_files": python_sources,
            "torch_version": installed.torch.__version__,
            "torch_cuda": installed.torch.version.cuda,
            "torch_cxx11_abi": installed.torch._C._GLIBCXX_USE_CXX11_ABI,
        },
    }


def _run_ninja(entry):
    # Preserve the compiler's own dependency files. Ninja's database can report
    # STALE after a successful build when filesystem timestamp roundoff changes
    # an object's mtime; exact compiler depfile bytes avoid that proxy entirely.
    command = [
        "ninja",
        "-v",
        "-d",
        "keepdepfile",
        "-C",
        str(entry),
        "-f",
        str(entry / "build.ninja"),
    ]
    jobs = os.environ.get("MAX_JOBS")
    if jobs is not None and jobs.isdigit():
        command.extend(["-j", jobs])
    subprocess.run(command, cwd=entry, check=True)


def _dependencies(entry):
    paths = set()
    depfiles = sorted(entry.glob("*.o.d"))
    for depfile in depfiles:
        target, separator, body = depfile.read_text().replace("\\\n", " ").partition(":")
        expected_target = depfile.with_suffix("")
        targets = shlex.split(target)
        if not separator or targets != [str(expected_target)]:
            raise RuntimeError(f"Unexpected FlashInfer compiler dependency target: {depfile}")
        dependencies = shlex.split(body)
        if not dependencies:
            raise RuntimeError(f"Empty FlashInfer compiler dependencies: {depfile}")
        for name in dependencies:
            path = Path(name)
            paths.add(path if path.is_absolute() else entry / path)
    if not paths or not depfiles:
        raise RuntimeError("FlashInfer build did not preserve its compiler dependencies")
    return {
        "files": [_file(path) for path in sorted(paths)],
        "metadata": [_file(path) for path in depfiles],
    }


def _write_json(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_entry(entry, name, key, identity):
    text = (entry / "record.json").read_text()
    record = json.loads(text)
    if text != json.dumps(record, sort_keys=True, indent=2) + "\n":
        raise RuntimeError(f"Immutable FlashInfer manifest bytes changed: {entry}")
    if (
        record.get("schema") != SCHEMA
        or record.get("name") != name
        or record.get("cache_key") != key
        or record.get("build_identity") != identity
        or record.get("library", {}).get("path") != str(entry / f"{name}.so")
        or record.get("build_metadata", {}).get("path") != str(entry / "build.ninja")
        or not record.get("dependencies")
        or not record.get("dependency_metadata")
    ):
        raise RuntimeError(f"Invalid immutable FlashInfer record: {entry}")
    _verify_files(record)
    return {**record, "manifest": _file(entry / "record.json")}


def _mapped_libraries():
    result = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith("/"):
            result[columns[5]] = (columns[3], int(columns[4]))
    return result


def _require_mapped(record):
    library = record["library"]
    path = Path(library["path"])
    observed = _mapped_libraries().get(str(path))
    stat = path.stat()
    expected = (f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}", stat.st_ino)
    if observed != expected:
        raise RuntimeError(f"Immutable FlashInfer ELF is not the actually mapped file: {path}")


def _bind(spec, entry, installed):
    class BoundSpec(installed.core.JitSpecNvcc):
        @property
        def build_dir(self):
            return self._immutable_entry

        @property
        def jit_library_path(self):
            return self.build_dir / f"{self.name}.so"

        @property
        def ninja_path(self):
            return self.build_dir / "build.ninja"

        def get_library_path(self):
            return self.jit_library_path

        def build(self, *args, **kwargs):
            raise RuntimeError("An immutable FlashInfer spec cannot be rebuilt")

    spec.__class__ = BoundSpec
    spec._immutable_entry = entry


def _load(spec, installed):
    _validate_request(spec, installed)
    identity = _input_identity(spec, installed)
    key = hashlib.sha256(_json(identity).encode()).hexdigest()
    entry = RUNTIME / "entries" / key / spec.name
    locks = RUNTIME / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{spec.name}-{key}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not entry.exists():
            if os.environ.get("FLASHINFER_DISABLE_JIT"):
                raise installed.core.MissingJITCacheError(
                    "Private FlashInfer artifact is absent", spec=spec
                )
            # Build at its final, content-addressed path so the archived Ninja
            # recipe identifies the actual outputs. The record is committed last.
            # An interrupted/failed build leaves an incomplete entry, which fails
            # on subsequent use instead of rebuilding or replacing any bytes.
            entry.mkdir(parents=True)
            (entry / "build.ninja").write_text(_ninja(spec, installed, entry.parent))
            _run_ninja(entry)
            _verify_files(identity)
            if (
                _environment() != identity["environment"]
                or _request(spec) != identity["request"]
                or _ninja(spec, installed, RUNTIME / "ninja-template") != identity["ninja_template"]
            ):
                raise RuntimeError(
                    "FlashInfer request or compiler environment changed during build"
                )
            dependencies = _dependencies(entry)
            expected_depfiles = {
                str(entry / f"{path.parent.name}_{path.stem}.cuda.o.d") for path in spec.sources
            }
            if expected_depfiles != {item["path"] for item in dependencies["metadata"]}:
                raise RuntimeError("FlashInfer compiler dependency files differ from the request")
            if not {str(Path(path).resolve()) for path in spec.sources} <= {
                item["path"] for item in dependencies["files"]
            }:
                raise RuntimeError("FlashInfer dependency manifest omitted a translation unit")
            record = {
                "schema": SCHEMA,
                "name": spec.name,
                "cache_key": key,
                "build_identity": identity,
                "library": _file(entry / f"{spec.name}.so"),
                "build_metadata": _file(entry / "build.ninja"),
                "dependencies": dependencies["files"],
                "dependency_metadata": dependencies["metadata"],
            }
            _write_json(entry / "record.json", record)
            for name in (
                f"{spec.name}.so",
                "build.ninja",
                "record.json",
                *(Path(item["path"]).name for item in dependencies["metadata"]),
            ):
                (entry / name).chmod(0o444)
        record = _validate_entry(entry, spec.name, key, identity)
        if spec.name in _LOADED and _LOADED[spec.name] != record:
            raise RuntimeError(f"FlashInfer module identity changed in this process: {spec.name}")
        module = installed.ffi.load_module(record["library"]["path"])
        _require_mapped(record)
        _LOADED[spec.name] = record
        _bind(spec, entry, installed)
        registered = installed.core.jit_spec_registry.get_all_specs().get(spec.name)
        if registered is not spec:
            # Factories normally create one spec. Do not silently leave an older
            # registry object pointing at an unrelated artifact.
            raise RuntimeError(f"FlashInfer registry does not own the loaded spec: {spec.name}")
        return module


def native_info():
    """Return exact source/spec/toolchain/ELF identities after actual loading."""
    for record in _LOADED.values():
        _verify_files(record)
        _require_mapped(record)
    return copy.deepcopy(_LOADED)


def archive(destination):
    """Archive all actual ELFs and their original recipes before completion."""
    records = native_info()
    if set(records) != MODULES:
        raise RuntimeError("Cannot archive an incomplete immutable FlashInfer inventory")
    target = Path(destination) / "flashinfer_native"
    target.mkdir(parents=True, exist_ok=False)
    archived = {}
    for name, record in sorted(records.items()):
        directory = target / name
        directory.mkdir()
        archived[name] = {}
        for field in ("library", "build_metadata", "manifest"):
            original = record[field]
            path = directory / Path(original["path"]).name
            with Path(original["path"]).open("rb") as source, path.open("xb") as sink:
                shutil.copyfileobj(source, sink)
            saved = _file(path)
            if saved["sha256"] != original["sha256"] or saved["bytes"] != original["bytes"]:
                raise RuntimeError(f"FlashInfer artifact changed during archival: {path}")
            archived[name][field] = {"original": original, "archived": saved}
        archived[name]["dependency_metadata"] = []
        for original in record["dependency_metadata"]:
            path = directory / Path(original["path"]).name
            with Path(original["path"]).open("rb") as source, path.open("xb") as sink:
                shutil.copyfileobj(source, sink)
            saved = _file(path)
            if saved["sha256"] != original["sha256"] or saved["bytes"] != original["bytes"]:
                raise RuntimeError(f"FlashInfer dependency file changed during archival: {path}")
            archived[name]["dependency_metadata"].append({"original": original, "archived": saved})
    if native_info() != records:
        raise RuntimeError("FlashInfer identity changed during archival")
    _write_json(target / "manifest.json", {"schema": SCHEMA, "modules": archived})
    return archived


@contextmanager
def pinned():
    """Intercept exact supported specs before warming either full-model arm."""
    global _ACTIVE
    if _ACTIVE or _LOADED:
        raise RuntimeError("Private FlashInfer pinning requires a fresh process")
    installed = _installed()
    if MODULES & installed.core.jit_spec_registry.get_all_specs().keys():
        raise RuntimeError(
            "Install private FlashInfer pinning before any supported spec is created"
        )
    if any(
        Path(path.removesuffix(" (deleted)")).name in {f"{name}.so" for name in MODULES}
        for path in _mapped_libraries()
    ):
        raise RuntimeError("A supported FlashInfer ELF was already loaded before pinning")
    original = installed.core.JitSpecNvcc.build_and_load

    def load(spec):
        if spec.name not in MODULES:
            return original(spec)
        return _load(spec, installed)

    _ACTIVE = True
    try:
        with (
            patch.object(installed.env, "FLASHINFER_GEN_SRC_DIR", RUNTIME / "generated"),
            patch.object(installed.core.JitSpecNvcc, "build_and_load", load),
        ):
            yield
    finally:
        _ACTIVE = False
