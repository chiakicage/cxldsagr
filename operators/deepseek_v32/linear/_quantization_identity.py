"""Scoped source identity and observed Triton artifacts, without importing Triton."""

import copy
import dataclasses
import hashlib
import importlib.metadata
import os
from pathlib import Path

_digests = {}
_ASM_KINDS = frozenset({"ptx", "cubin", "ttir", "ttgir", "llir"})
# Non-TRITON descriptors used by installed triton.knobs build/compilation/nvidia knobs.
_COMPILER_ENVIRONMENT_KEYS = frozenset(
    {
        "CC",
        "DISABLE_PTXAS_OPT",
        "LLVM_EXTRACT_DI_LOCAL_VARIABLES",
        "NVPTX_ENABLE_DUMP",
        "PTXAS_OPTIONS",
        "USE_IR_LOC",
    }
)


def _digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    previous = _digests.get(path)
    if previous is None or previous[0] != stamp:
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if stamp != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"Quantizer identity file changed while reading: {path}")
        _digests[path] = stamp, value
    return _digests[path][1]


def _plain(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return _plain(value._asdict())
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    raise TypeError(f"Unsupported live Triton metadata type: {type(value).__name__}")


class Identity:
    def __init__(self, wrapper_path, kernel_path, policy):
        self.sources = tuple(
            sorted(
                {
                    Path(wrapper_path).resolve(),
                    Path(kernel_path).resolve(),
                    Path(__file__).resolve(),
                }
            )
        )
        self.policy = copy.deepcopy(policy)
        self.imported_sources = {str(path): _digest(path) for path in self.sources}
        self.captured = None
        self.observed = {}

    def build_info(self):
        """Describe current scoped bytes; no package import, compilation or CUDA query."""
        distribution = importlib.metadata.distribution("triton")
        root = Path(distribution.locate_file("triton")).resolve()
        paths = set(self.sources)
        for relative in distribution.files or ():
            path = Path(distribution.locate_file(relative)).resolve()
            if (
                path.is_relative_to(root)
                and path.is_file()
                and (path.suffix in {".py", ".so", ".bc"} or path.name.startswith("ptxas"))
            ):
                paths.add(path)
        environment = {
            key: value
            for key, value in sorted(os.environ.items())
            if (key.startswith("TRITON_") or key in _COMPILER_ENVIRONMENT_KEYS)
            and key not in {"TRITON_CACHE_DIR", "TRITON_DUMP_DIR"}
        }
        for key in ("TRITON_PTXAS_PATH", "TRITON_LIBDEVICE_PATH"):
            if environment.get(key):
                path = Path(environment[key]).resolve()
                if not path.is_file():
                    raise RuntimeError(f"Configured Triton compiler dependency is missing: {path}")
                paths.add(path)
        return {
            "schema_version": 1,
            "backend": "handwritten_triton",
            "kernel_policy": copy.deepcopy(self.policy),
            "torch_version": importlib.metadata.version("torch"),
            "triton_version": distribution.version,
            "triton_root": str(root),
            "compiler_environment": environment,
            "source_and_dependency_sha256": {str(path): _digest(path) for path in sorted(paths)},
        }

    def capture(self):
        """Freeze identity before lazy kernel import; subsequent calls are constant-time."""
        if self.captured is None:
            current = self.build_info()
            if any(
                current["source_and_dependency_sha256"][path] != digest
                for path, digest in self.imported_sources.items()
            ):
                raise RuntimeError("Quantizer source changed after wrapper import")
            self.captured = current
        return self.captured

    def observe(self, compiled):
        """Retain actual returned objects; never inspect a driver, load or run a kernel."""
        values = vars(compiled)
        key = values["hash"]
        if self.observed.get(key) is compiled:
            return
        if self.captured is None:
            raise RuntimeError("Quantizer identity must be captured before compilation")
        if self.build_info() != self.captured:
            raise RuntimeError("Quantizer identity changed before observing a new specialization")
        metadata = _plain(values["metadata"])
        target = metadata.get("target")
        if target != {"backend": "cuda", "arch": 90, "warp_size": 32}:
            raise RuntimeError(f"Observed quantizer target differs from SM90: {target}")
        asm = values["asm"]
        if not {"ptx", "cubin"} <= asm.keys():
            raise RuntimeError("Observed quantizer is missing PTX or CUBIN evidence")
        self.observed[key] = compiled

    def runtime_info(self):
        """Read retained live metadata/ASM; keep the original compilation identity."""
        if not self.observed:
            return None
        specializations = []
        for key, compiled in sorted(self.observed.items()):
            values = vars(compiled)
            kernel = values.get("kernel")
            if kernel is not None and not isinstance(kernel, bytes):
                raise TypeError("Unsupported retained Triton kernel binary type")
            artifacts = {}
            for kind in sorted(_ASM_KINDS & values["asm"].keys()):
                value = values["asm"][kind]
                if not isinstance(value, (str, bytes)):
                    raise TypeError(f"Unsupported Triton {kind} artifact type")
                artifacts[kind] = hashlib.sha256(
                    value.encode() if isinstance(value, str) else value
                ).hexdigest()
            specializations.append(
                {
                    "hash": values["hash"],
                    "observed_cache_key": key,
                    "name": values["name"],
                    "metadata": _plain(values["metadata"]),
                    "artifact_sha256": artifacts,
                    "retained_kernel_sha256": (
                        hashlib.sha256(kernel).hexdigest() if kernel is not None else None
                    ),
                    "module_handle_set": values.get("module") is not None,
                    "function_handle_set": values.get("function") is not None,
                    "metadata_group": _plain(values.get("metadata_group", {})),
                }
            )
        return {"build_info": copy.deepcopy(self.captured), "specializations": specializations}
