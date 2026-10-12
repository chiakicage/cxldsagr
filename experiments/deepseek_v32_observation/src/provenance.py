"""Bind observation to actual source, checkpoint inventory, and loaded backends."""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import torch

from evaluation.validation import identity_digest

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_inventory():
    files = set()
    for directory in (
        "experiments/deepseek_v32_observation/src",
        "models/deepseek_v32",
        "operators/deepseek_v32",
    ):
        files.update(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}
            and "tests" not in path.parts
        )
    files.update(
        ROOT / name
        for name in (
            "models/attention_contracts.py",
            "operators/flashinfer.py",
            "evaluation/validation.py",
            "experiments/deepseek_v32_mfu/src/operator_flops.py",
            "experiments/deepseek_v32_mfu/src/backend_provenance.py",
            "pyproject.toml",
            "uv.lock",
            "experiments/deepseek_v32_mfu/src/profile_hardware.py",
        )
    )
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(files)}


@cache
def _cached_installed_backends():
    from experiments.deepseek_v32_mfu.src.backend_provenance import (
        _installed,
        _installed_flashinfer,
    )

    return {
        "deep_gemm": _installed("deep-gemm", "deep_gemm._C"),
        "flash_mla": _installed("flash-mla", "flash_mla.cuda"),
        "flashinfer": _installed_flashinfer(),
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
    }


def installed_backends(*, refresh=False):
    """Refresh on boundary checks so changed installed files cannot hide in cache."""
    if refresh:
        _cached_installed_backends.cache_clear()
    return _cached_installed_backends()


def _plain_metadata(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "_asdict"):
        return _plain_metadata(value._asdict())
    if isinstance(value, dict):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_metadata(item) for item in value]
    # Device descriptors and compile options occasionally expose an enum or
    # dataclass. Its type/string are reproducibility metadata, not addresses.
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _plain_metadata(asdict(value))
    return {"type": type(value).__name__, "value": str(value)}


def _live_triton_artifacts(factory):
    """Inspect existing cached JIT functions without invoking their factories.

    functools has no public cache-values API. Reading its direct GC referents
    finds only already retained JIT functions; their device caches contain the
    compiled launch artifacts used in this process. No personal cache path or
    on-disk candidate is visited and no kernel is compiled or launched.
    """
    compiled_objects = {}
    for referent in gc.get_referents(factory):
        if not isinstance(referent, dict):
            continue
        for candidate in referent.values():
            device_caches = getattr(candidate, "device_caches", None)
            if not isinstance(device_caches, dict):
                continue
            for device_cache in device_caches.values():
                if not isinstance(device_cache, (tuple, list)) or not device_cache:
                    continue
                kernels = device_cache[0]
                if not isinstance(kernels, dict):
                    continue
                for compiled in kernels.values():
                    if hasattr(compiled, "asm") and hasattr(compiled, "hash"):
                        compiled_objects[compiled.hash] = compiled
    result = []
    for key, compiled in sorted(compiled_objects.items()):
        artifacts = {}
        for name, value in compiled.asm.items():
            if not isinstance(value, (bytes, str)):
                raise TypeError(f"Unsupported live Triton {name} artifact type")
            artifacts[name] = hashlib.sha256(
                value if isinstance(value, bytes) else value.encode()
            ).hexdigest()
        if not {"ptx", "cubin"} <= artifacts.keys():
            raise RuntimeError("Live resident mask kernel lacks PTX/CUBIN identity")
        result.append(
            {
                "hash": key,
                "name": compiled.name,
                "metadata": _plain_metadata(compiled.metadata),
                "asm_sha256": artifacts,
            }
        )
    return result


def collect_resident_mask_artifacts():
    """Return only mask modules already imported and compiled in this process."""
    specifications = {
        "resident_tail_mask": ("operators.deepseek_v32.indexer.echo", "_resident_tail_mask_kernel"),
        "selection_nonfinite_mask": (
            "operators.deepseek_v32.indexer.selection",
            "_nonfinite_index_mask_kernel",
        ),
    }
    result = {}
    for name, (module_name, factory_name) in specifications.items():
        module = sys.modules.get(module_name)
        factory = None if module is None else vars(module).get(factory_name)
        result[name] = [] if factory is None else _live_triton_artifacts(factory)
    return result


def runtime_artifact_identity(runtime):
    """Normalize stable native/JIT identity, excluding memory and process details.

    Every loaded native library or live compiled artifact's content digest is
    included. Allocator sizes, PID, addresses, command, clocks and filesystem
    timestamps are intentionally outside this cross-process comparison. Vendor
    DeepGEMM JIT CUBINs are not exposed here; installed official headers and its
    native API remain covered by backend_sha256 in execution identity.
    """
    native = runtime.get("native_runtime", {})

    def hashes(value):
        result = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    isinstance(item, str)
                    and len(item) == 64
                    and all(character in "0123456789abcdef" for character in item)
                    and ("sha256" in key or key in {"ptx", "cubin", "ttir", "ttgir", "llir"})
                ):
                    result.add(item)
                elif isinstance(item, (dict, list, tuple)):
                    result.update(hashes(item))
        elif isinstance(value, (list, tuple)):
            for item in value:
                result.update(hashes(item))
        return result

    artifacts = {}
    for name, value in sorted(native.items()):
        if name in {"schema_version", "boundary"}:
            continue
        artifacts[name] = sorted(hashes(value))
    # Match whole specializations as well as bytes. Binary hashes alone do
    # not identify a launch's architecture, warp count and compile options.
    specialization_fields = {
        "hash",
        "name",
        "factory",
        "compile_key",
        "metadata",
        "function_name",
        "kernel_names",
        "asm_sha256",
        "artifact_sha256",
        "artifact_sha256s",
        "module_handle_set",
        "function_handle_set",
    }

    def specialization_digests(value):
        result = set()
        if isinstance(value, dict):
            if "asm_sha256" in value or "artifact_sha256" in value or "compile_key" in value:
                stable = {key: value[key] for key in sorted(specialization_fields & value.keys())}
                result.add(identity_digest(stable))
            for item in value.values():
                if isinstance(item, (dict, list, tuple)):
                    result.update(specialization_digests(item))
        elif isinstance(value, (tuple, list)):
            for item in value:
                result.update(specialization_digests(item))
        return result

    specializations = {
        name: sorted(specialization_digests(value))
        for name, value in sorted(native.items())
        if name not in {"schema_version", "boundary"}
    }
    loaded = sorted(
        row["name"]
        for row in native.get("native_jit", [])
        if row.get("loaded_in_this_process") is True
    )
    return {
        "backend_sha256": runtime["backend_sha256"],
        "loaded_flashinfer_modules": loaded,
        "artifact_sha256": artifacts,
        "specialization_sha256": specializations,
    }


def require_runtime_match(checked, measured):
    """Require measured compiled artifacts to be covered by independent check.

    Reference and diagnostic checks may load extra accepted specializations.
    The clean benchmark's observed set must be a subset, with identical
    installed backend and module identities; it cannot add an unchecked
    binary or specialize accepted bytes with different launch metadata.
    """
    expected = runtime_artifact_identity(checked)
    actual = runtime_artifact_identity(measured)
    mismatched = []
    if actual["backend_sha256"] != expected["backend_sha256"]:
        mismatched.append("installed_backends")
    if not set(actual["loaded_flashinfer_modules"]) <= set(expected["loaded_flashinfer_modules"]):
        mismatched.append("loaded_flashinfer_modules")
    for field in ("artifact_sha256", "specialization_sha256"):
        for name, values in actual[field].items():
            if not set(values) <= set(expected[field].get(name, [])):
                mismatched.append(f"{field}/{name}")
    if mismatched:
        raise RuntimeError(f"Numerical acceptance native/JIT artifacts differ: {mismatched}")
    return {"passed": True, "stable_artifact_sha256": identity_digest(actual), "identity": actual}


def checkpoint_identity(path):
    path = Path(path).resolve(strict=True)
    return {
        "path": str(path),
        "config_sha256": digest(path / "config.json"),
        "shards": {
            file.name: {"bytes": file.stat().st_size, "mtime_ns": file.stat().st_mtime_ns}
            for file in sorted(path.glob("*.safetensors"))
        },
        "boundary": "Checkpoint config content hash and shard size/mtime inventory; not whole-weight content hashes",
    }


def execution_identity(*, group, key, seed, checkpoint, operators, workload_identity):
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "group": group,
        "key": key,
        "seed": seed,
        "input": workload_identity,
        "operators": operators,
        "source_sha256": source_inventory(),
        "backend_sha256": identity_digest(installed_backends(refresh=True)),
        "checkpoint": checkpoint_identity(checkpoint),
        "cuda": {
            "capability": [properties.major, properties.minor],
            "sms": properties.multi_processor_count,
        },
        "precision_policy": {"tf32": False, "grad": False},
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("DG_", "DJ_", "FLASHINFER_", "CUTE_DSL_"))
        },
    }


def runtime_evidence():
    from experiments.deepseek_v32_mfu.src.backend_provenance import (
        collect_flashinfer_runtime_artifacts,
    )

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    native = collect_flashinfer_runtime_artifacts(require_local_native=False)
    native["resident_masks_triton"] = collect_resident_mask_artifacts()
    smi = subprocess.run(
        [
            "nvidia-smi",
            f"--id=GPU-{properties.uuid}",
            "--query-gpu=index,name,uuid,pci.device_id,compute_cap,memory.total,memory.used,clocks.sm,clocks.mem,power.limit,temperature.gpu,utilization.gpu,driver_version",
            "--format=csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return {
        "python": sys.version,
        "device_name": properties.name,
        "uuid": str(properties.uuid),
        "total_memory": properties.total_memory,
        "sms": properties.multi_processor_count,
        "affinity": sorted(os.sched_getaffinity(0)),
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "device_free_total": list(torch.cuda.mem_get_info()),
        "backend_sha256": identity_digest(installed_backends()),
        "native_runtime": native,
        "nvidia_smi": {"command": smi.args, "stdout": smi.stdout, "stderr": smi.stderr},
        "command": sys.argv,
    }


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
