"""Source snapshots and machine observations with explicit measurement boundaries."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from evaluation import pool_scan_provenance as pool_scan
from experiments.nosa_motivation.src.config import ROOT
from experiments.nosa_motivation.src.cpu_environment import cpu_environment


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def publish_directories(sources, targets):
    """Copy complete staged categories, rolling back only paths created here."""
    created = []
    try:
        for name, target in targets.items():
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir(exist_ok=False)
            created.append(target)
            shutil.copytree(sources[name], target, dirs_exist_ok=True)
    except BaseException as error:
        cleanup_errors = []
        for target in reversed(created):
            try:
                shutil.rmtree(target)
            except BaseException as cleanup_error:  # noqa: BLE001 -- retain every cleanup failure.
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "publication and rollback failed", [error, *cleanup_errors]
            ) from None
        raise


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def file_identity(path):
    """Hash a file while rejecting replacements or writes during observation."""
    path = Path(path)
    before = path.stat()
    sha256 = digest(path)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise ValueError(f"file changed during identity observation: {path}")
    return {"size": before.st_size, "sha256": sha256}


def header_tree_identity(directory):
    directory = Path(directory).resolve()
    files = {
        str(path.relative_to(directory)): file_identity(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError(f"missing native dependency headers: {directory}")
    return {"path": str(directory), "files": files, "sha256": manifest_digest(files)}


def native_build_identity():
    """Capture the actual build APIs plus fresh dependency bytes, without compiling.

    Build APIs may cache their answers. Fresh inventories are deliberately not
    cached: source snapshots and these inventories detect edits after a module
    was loaded, including dirty CUTLASS headers missed by submodule HEAD alone.
    """
    from tvm_ffi import libinfo

    from cache.allocator.snapshot import build_info as allocator_snapshot_build_info
    from operators.nosa import _native
    from operators.nosa.attention.offload import _fused

    native, fused = _native.build_info(), _fused.build_info()
    if native["selected_backend"] != "native":
        raise ValueError("NOSA motivation requires the native Hopper backend")
    cutlass = ROOT / "3rdparty/cutlass"
    build = {
        "native_build_info": native,
        "fused_build_info": fused,
        "allocator_snapshot_build_info": allocator_snapshot_build_info(),
        "pool_referrers_build_info": pool_scan.build_info(),
        "compiler": file_identity(native["compiler"]["path"]),
        "dependencies": {
            "cutlass": {
                "revision": git("-C", str(cutlass), "rev-parse", "HEAD"),
                "dirty_status": git(
                    "-C", str(cutlass), "status", "--short", "--untracked-files=all"
                ),
                "include": header_tree_identity(cutlass / "include"),
                "util_include": header_tree_identity(cutlass / "tools/util/include"),
            },
            "flashinfer_include": header_tree_identity(fused["flashinfer_include"]),
            "tvm_ffi_include": header_tree_identity(libinfo.find_include_path()),
            "dlpack_include": header_tree_identity(libinfo.find_dlpack_include_path()),
            "cuda_include": header_tree_identity(
                Path(native["compiler"]["path"]).parent.parent / "include"
            ),
        },
        "boundary": (
            "Actual NOSA build_info APIs and fresh CUTLASS/FlashInfer/TVM-FFI/DLPack/CUDA header trees; "
            "dirty dependencies are identified by content, not assumed equivalent to HEAD. "
            "Mapped project, FlashInfer, PyTorch and CUDA matrix/runtime .so artifacts are "
            "captured separately after warmup. Driver-generated device code is outside this inventory."
        ),
    }
    return {**build, "sha256": manifest_digest(build)}


def verify_native_build_identity(expected):
    current = native_build_identity()
    if current != expected:
        raise ValueError("native build or dependency headers changed during execution/replay")
    return current


def loaded_native_artifacts(maps_path=Path("/proc/self/maps")):
    """Identify mapped compute/runtime libraries, without guessing cache filenames."""
    artifacts = {}
    for line in Path(maps_path).read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        name = fields[5]
        filename = Path(name).name.lower()
        relevant = (
            "cxldsagr" in filename
            or "/flashinfer/" in name
            or "/torch/" in name
            or filename.startswith(("libcublas", "libcudart", "libc10", "libtvm_ffi"))
        )
        if not relevant or ".so" not in filename:
            continue
        if name.endswith(" (deleted)"):
            raise ValueError(f"loaded native library was deleted: {name}")
        path = Path(name)
        stat = path.stat()
        mapped_device = tuple(int(value, 16) for value in fields[3].split(":"))
        if stat.st_ino != int(fields[4]) or mapped_device != (
            os.major(stat.st_dev),
            os.minor(stat.st_dev),
        ):
            raise ValueError(f"loaded native library was replaced: {path}")
        resolved = str(path.resolve())
        if resolved not in artifacts:
            artifacts[resolved] = file_identity(path)
    if not any("cxldsagr" in Path(path).name for path in artifacts):
        raise ValueError("no mapped project native libraries after warmup")
    return dict(sorted(artifacts.items()))


def verify_native_artifacts(expected, *, allow_additions=False):
    current = loaded_native_artifacts()
    if any(current.get(path) != item for path, item in expected.items()) or (
        not allow_additions and current != expected
    ):
        raise ValueError("loaded native library identity changed during execution/replay")
    return current


def audit_native_identity(metadata, *, require_pool_referrers=False, expected_abi_sha256=None):
    """Validate saved observations without requiring old cache paths to still exist."""
    native = metadata["native_provenance"]
    before, after = native["build_before"], native["build_after"]
    if before != after or before["sha256"] != manifest_digest(
        {name: item for name, item in before.items() if name != "sha256"}
    ):
        raise ValueError("saved native build/dependency identity differs")
    for dependency in (
        before["dependencies"]["cutlass"]["include"],
        before["dependencies"]["cutlass"]["util_include"],
        before["dependencies"]["flashinfer_include"],
        before["dependencies"]["tvm_ffi_include"],
        before["dependencies"]["dlpack_include"],
        before["dependencies"]["cuda_include"],
    ):
        if not dependency["files"] or dependency["sha256"] != manifest_digest(dependency["files"]):
            raise ValueError("saved native dependency inventory differs")
    planned_pool = before.get("pool_referrers_build_info")
    if planned_pool is None and (
        require_pool_referrers or any("pool_referrers" in case for case in metadata["cases"])
    ):
        raise ValueError("current NOSA source requires pool-referrer provenance")
    if planned_pool is not None:
        pool_scan.audit_build(planned_pool, after["pool_referrers_build_info"])
    final = native["artifacts_final"]
    if not final:
        raise ValueError("missing loaded native artifacts")
    for case in metadata["cases"]:
        artifacts = case["native_artifacts_before"]
        if (
            not artifacts
            or artifacts != case["native_artifacts_after"]
            or any(final.get(path) != item for path, item in artifacts.items())
        ):
            raise ValueError("saved loaded native artifacts differ across observations")
        if "allocator_snapshot_build_info" in before:
            adapter = case["allocator_snapshot"]
            if adapter["backend"] == "private_cpp":
                planned = before["allocator_snapshot_build_info"]
                if (
                    not planned["available"]
                    or planned["fingerprint"] != manifest_digest(planned["identity"])
                    or adapter["fingerprint"] != planned["fingerprint"]
                    or adapter["fallback_reason"] is not None
                    or artifacts.get(adapter["loaded_binary_path"], {}).get("sha256")
                    != adapter["loaded_binary_sha256"]
                ):
                    raise ValueError("allocator snapshot adapter differs from mapped build")
            elif adapter["backend"] != "torch_official" or not adapter["fallback_reason"]:
                raise ValueError("missing allocator snapshot fallback evidence")
        if planned_pool is not None:
            pool_scan.audit_case(
                case["pool_referrers"],
                planned_pool,
                expected_abi_sha256=expected_abi_sha256,
                mapped_artifacts=artifacts,
            )
        validation = case["token_validation"]
        if (
            validation["backend"] != "cpython_native"
            or artifacts.get(validation["loaded_binary_path"], {}).get("sha256")
            != validation["loaded_binary_sha256"]
        ):
            raise ValueError("token validation binary differs from mapped artifact")
    return before["sha256"]


def source_paths():
    paths = {ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "models/attention_contracts.py"}
    suffixes = {".py", ".cu", ".cuh", ".c", ".cpp", ".h", ".hpp", ".sh"}
    for name in (
        "models/nosa",
        "cache",
        "executor",
        "serving",
        "GR",
        "evaluation",
        "operators/nosa",
        "operators/common",
        "experiments/nosa_motivation/src",
        "experiments/nosa_motivation/scripts",
    ):
        directory = ROOT / name
        paths.update(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in suffixes
            and not {"tests", "output", "generated", "__pycache__"}.intersection(
                path.relative_to(directory).parts
            )
        )
    paths.update((ROOT / "operators").glob("*.py"))
    paths.add(ROOT / pool_scan.POOL_SCAN_ABI)
    paths.add(ROOT / "experiments/nosa_offload_overlap/src/analyze.py")
    return sorted(paths)


def snapshot_sources(output):
    manifest = {}
    for path in source_paths():
        relative = path.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[str(relative)] = digest(path)
    write_json(output / "source_manifest.json", manifest)
    return manifest_digest(manifest)


def verify_source_snapshot(output, *, check_current=False):
    manifest = json.loads((output / "source_manifest.json").read_text())
    for name, expected in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source manifest must use repository-relative paths")
        if digest(output / "source" / relative) != expected:
            raise ValueError(f"saved source identity changed: {name}")
        if check_current and digest(ROOT / relative) != expected:
            raise ValueError(f"runtime source changed during measurement: {name}")
    if check_current and {str(path.relative_to(ROOT)) for path in source_paths()} != set(manifest):
        raise ValueError("runtime source inventory changed during measurement")
    return manifest_digest(manifest)


def git(*arguments):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def runtime_environment(torch, device):
    properties = torch.cuda.get_device_properties(device)
    packages = {}
    for name in ("torch", "triton", "flashinfer-python", "nvidia-cublas", "cuda-bindings"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "hostname": platform.node(),
        "cpu_environment": cpu_environment(torch),
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "name": properties.name,
        "uuid": str(properties.uuid),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory": properties.total_memory,
        "sm_count": properties.multi_processor_count,
        "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "submodule_status": git("submodule", "status"),
    }


def checkpoint_inventory(path):
    path = Path(path).resolve()
    files = list(path.glob("*.safetensors"))
    if not files:
        raise ValueError("checkpoint directory contains no safetensors shards")
    return {
        "path": str(path),
        "shards": {
            item.name: {"size": item.stat().st_size, "mtime_ns": item.stat().st_mtime_ns}
            for item in sorted(files)
        },
        "metadata_sha256": {
            item.name: digest(item)
            for item in sorted(path.glob("*.json"))
            if item.name in ("config.json", "tokenizer.json", "model.safetensors.index.json")
        },
        "identity_boundary": "metadata hashes and shard stat inventory; weights are not hashed",
    }


def memory_sample(torch, device, stage):
    free, total = torch.cuda.mem_get_info(device)
    return {
        "stage": stage,
        "torch_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_reserved_bytes": torch.cuda.memory_reserved(device),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_free_bytes": free,
        "device_total_bytes": total,
        "device_used_bytes": total - free,
    }
