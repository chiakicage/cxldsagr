"""Opt-in, process-once DeepGEMM header overlay for the pinned ECHO runtime.

No checkout or installed package is modified. An overlay keeps every unchanged
include entry symlinked to the installed package and materializes one patched
header. Compiler.library_version hashes the changed .cuh; a stable overlay path
also stabilizes the compiler's -I flag and therefore its JIT cache key.

This scope is deliberately NOT reversible. DeepGEMM's lazy Compiler permanently
captures its include flags on first JIT, so restoring only its static root/hash
would leave an inconsistent compiler. Exit keeps the selected configuration and
closes the process gate. Start another process to use the native kernel again.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

PREFETCH_PHASE_PATCH_ID = "echo_sm90_prefetch_phase_snapshot_v1"


@dataclass(frozen=True)
class HeaderPatch:
    identifier: str
    relative_path: str
    source_sha256: str
    patched_sha256: str
    before: bytes
    after: bytes


PREFETCH_PHASE_PATCH = HeaderPatch(
    identifier=PREFETCH_PHASE_PATCH_ID,
    relative_path="include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh",
    source_sha256="38db9dfc0086f7d7b5d058df6e87465e3bdd51c68bc0b5d6bbb15db1723b7691",
    patched_sha256="a7f2db638388150dae94664af5fca7b0277ca3c46759749db28a583da12269f4",
    before=(
        b"            num_total_kv_blocks += num_kv_blocks;\n\n"
        b"            if (!s_prefetch_enabled) {"
    ),
    after=(
        b"            num_total_kv_blocks += num_kv_blocks;\n\n"
        b"            // Snapshot the phase flag before another warp can update it.\n"
        b"            const bool prefetch_enabled = s_prefetch_enabled;\n"
        b"            topk_bar.sync();\n"
        b"            if (!prefetch_enabled) {"
    ),
)
_PATCHES = {PREFETCH_PHASE_PATCH_ID: PREFETCH_PHASE_PATCH}
_PROCESS_STATE = "unused"


@dataclass(frozen=True)
class HeaderOverlay:
    root: Path
    metadata: dict


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def apply_header_patch(original: bytes, patch: HeaderPatch) -> bytes:
    if _sha256(original) != patch.source_sha256:
        raise ValueError("DeepGEMM source header SHA256 does not match the approved patch")
    if original.count(patch.before) != 1:
        raise ValueError("DeepGEMM patch anchor must occur exactly once")
    result = original.replace(patch.before, patch.after, 1)
    if _sha256(result) != patch.patched_sha256:
        raise ValueError("DeepGEMM patched header SHA256 does not match the approved patch")
    return result


def _patch_relative_path(patch: HeaderPatch) -> Path:
    relative = PurePosixPath(patch.relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in patch.relative_path
        or not relative.parts
        or relative.parts[0] != "include"
        or relative.suffix != ".cuh"
    ):
        raise ValueError("patch must name a .cuh below the library include directory")
    return Path(*relative.parts)


def _include_manifest(include: Path) -> list[tuple[str, str]]:
    entries = []
    for path in sorted(include.rglob("*")):
        if path.is_symlink():
            raise ValueError("installed include tree must not contain untracked symlink sources")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported include entry: {path}")
        entries.append((path.relative_to(include).as_posix(), _sha256(path.read_bytes())))
    if not entries:
        raise ValueError("installed DeepGEMM include tree is empty")
    return entries


def _populate_overlay(source: Path, destination: Path, relative: Path, patched: bytes):
    destination.mkdir()
    for child in source.iterdir():
        target = destination / child.name
        if child.name != relative.parts[0]:
            target.symlink_to(child, target_is_directory=child.is_dir())
        elif len(relative.parts) == 1:
            with target.open("xb") as handle:
                handle.write(patched)
        else:
            if not child.is_dir():
                raise ValueError("patch ancestor is not a directory")
            _populate_overlay(child, target, Path(*relative.parts[1:]), patched)


def _check_overlay_links(source: Path, destination: Path, relative: Path, patched: bytes):
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("overlay patch ancestors must be real directories")
    children = {child.name: child for child in source.iterdir()}
    if set(children) != {child.name for child in destination.iterdir()}:
        raise ValueError("overlay include entries differ from the installed tree")
    for name, child in children.items():
        target = destination / name
        if name != relative.parts[0]:
            if not target.is_symlink() or target.resolve(strict=True) != child:
                raise ValueError("overlay contains an unexpected or redirected include entry")
        elif len(relative.parts) == 1:
            if target.is_symlink() or target.read_bytes() != patched:
                raise ValueError("overlay patched header was modified")
        else:
            _check_overlay_links(child, target, Path(*relative.parts[1:]), patched)


def build_header_overlay(
    echo_path: Path,
    library_root: Path,
    cache_root: Path,
    *,
    patch: HeaderPatch = PREFETCH_PHASE_PATCH,
) -> HeaderOverlay:
    """Pure filesystem preparation; neither imports DeepGEMM nor initializes CUDA."""
    echo_path = Path(echo_path).resolve(strict=True)
    library_root = Path(library_root).resolve(strict=True)
    cache_root = Path(cache_root).expanduser().resolve()
    if cache_root.is_relative_to(echo_path) or cache_root.is_relative_to(library_root):
        raise ValueError("overlay cache must be outside the ECHO checkout and installed package")
    if not re.fullmatch(r"[A-Za-z0-9/_.+\-]+", str(cache_root)):
        raise ValueError("overlay path must be safe for DeepGEMM's unquoted compiler command")
    relative = _patch_relative_path(patch)
    original_path = library_root / relative
    source_path = echo_path / "DeepGEMM/deep_gemm" / relative
    original = original_path.read_bytes()
    if source_path.read_bytes() != original:
        raise ValueError("installed DeepGEMM header differs from the fixed ECHO checkout")
    patched = apply_header_patch(original, patch)
    include = library_root / "include"
    manifest = _include_manifest(include)
    include_sha = _sha256(json.dumps(manifest, separators=(",", ":")).encode())
    metadata = {
        "schema": 1,
        "patch_id": patch.identifier,
        "relative_header": patch.relative_path,
        "original_library_root": str(library_root),
        "checkout_header": str(source_path),
        "original_header_sha256": patch.source_sha256,
        "patched_header_sha256": patch.patched_sha256,
        "installed_include_manifest_sha256": include_sha,
        "unchanged_includes": "symlinked to installed package; package must stay immutable",
    }
    identity = _sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    destination = cache_root / identity
    metadata["overlay_identity"] = identity
    metadata["overlay_root"] = str(destination)
    encoded = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
    cache_root.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=cache_root))
        try:
            _populate_overlay(include, temporary / "include", Path(*relative.parts[1:]), patched)
            with (temporary / "manifest.json").open("xb") as handle:
                handle.write(encoded)
            try:
                temporary.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    manifest_path = destination / "manifest.json"
    if (
        destination.is_symlink()
        or manifest_path.is_symlink()
        or manifest_path.read_bytes() != encoded
    ):
        raise ValueError("existing overlay manifest does not match the selected source and patch")
    _check_overlay_links(include, destination / "include", Path(*relative.parts[1:]), patched)
    return HeaderOverlay(destination, metadata)


def assert_echo_kernel_process_usable():
    """Optional runner entry guard; closed scopes require a fresh process."""
    if _PROCESS_STATE == "closed":
        raise RuntimeError("DeepGEMM configuration is process-once; start a fresh process")


@contextmanager
def scoped_echo_kernel(
    echo_path: str | Path,
    *,
    patch_id: str | None = None,
    cache_root: str | Path | None = None,
):
    """Select native (default) or an approved overlay BEFORE adapter/SGLang imports.

    Cover the entire ModelRunner lifetime. The caller must drain GPU work before
    leaving; this initializer deliberately does not synchronize a potentially hung
    device. Exit permanently closes the gate and, for an overlay, leaves cpp.init
    guarded and its selected static configuration unchanged. It cannot reset the
    C++ Compiler singleton or revoke already-held external kernel function objects.
    No further runner or DeepGEMM operation is supported after this scope exits.
    """
    global _PROCESS_STATE
    if _PROCESS_STATE != "unused":
        raise RuntimeError("DeepGEMM configuration is process-once; start a fresh process")
    if any(
        name in ("deep_gemm", "deep_gemm_cpp") or name.startswith(("deep_gemm.", "sglang.srt"))
        for name in sys.modules
    ):
        raise RuntimeError(
            "select the kernel before importing DeepGEMM or SGLang in a fresh process"
        )
    if patch_id is not None and patch_id not in _PATCHES:
        raise ValueError(f"unknown ECHO kernel patch: {patch_id}")
    _PROCESS_STATE = "active"
    info = {
        "enabled": patch_id is not None,
        "patch_id": patch_id,
        "process_once": True,
        "reversible": False,
        "scope_closed": False,
    }
    try:
        if patch_id is None:
            yield info
            return
        from models.deepseek_v32.echo_adapter import verify_echo_checkout

        checkout, revision, checkout_diff = verify_echo_checkout(echo_path)
        spec = importlib.util.find_spec("deep_gemm")
        if spec is None or spec.origin is None:
            raise RuntimeError("the isolated ECHO DeepGEMM package is not installed")
        library_root = Path(spec.origin).resolve(strict=True).parent
        version = importlib.metadata.version("deep-gemm")
        if version != "2.1.1+bc1b75c":
            raise RuntimeError(f"expected pinned ECHO deep-gemm 2.1.1+bc1b75c, found {version}")
        selected_cache = (
            Path(cache_root)
            if cache_root is not None
            else Path.home() / ".cache/cxldsagr/echo-deepgemm"
        )
        overlay = build_header_overlay(
            checkout,
            library_root,
            selected_cache,
            patch=_PATCHES[patch_id],
        )
        # The extension needs libc10 loaded, but importing torch does not invoke its kernels.
        importlib.import_module("torch")
        package = importlib.import_module("deep_gemm")
        cpp = importlib.import_module("deep_gemm_cpp")
        if Path(package.__file__).resolve().parent != library_root:
            raise RuntimeError("DeepGEMM package changed between inspection and import")
        binary = Path(cpp.__file__).resolve(strict=True)
        if (
            binary.parent not in (library_root, library_root.parent)
            or not binary.name.startswith("deep_gemm_cpp.")
            or binary.suffix != ".so"
        ):
            raise RuntimeError("DeepGEMM C++ extension is outside the inspected installation")
        if not callable(getattr(cpp, "fp8_mqa_logits_fuse_prefetch", None)):
            raise TypeError("DeepGEMM extension lacks a callable ECHO fused prefetch API")
        cuda_home = Path(package._find_cuda_home()).resolve(strict=True)
        if not re.fullmatch(r"[A-Za-z0-9/_.+\-]+", str(cuda_home)):
            raise ValueError("CUDA path is unsafe for DeepGEMM's unquoted compiler command")
        original_init = cpp.init

        def guarded_init(library_root_path, cuda_home_path_by_python):
            if _PROCESS_STATE != "active":
                raise RuntimeError("DeepGEMM scope is closed; start a fresh process")
            requested_root = Path(library_root_path).resolve()
            requested_cuda = Path(cuda_home_path_by_python).resolve()
            if requested_root not in (library_root, overlay.root) or requested_cuda != cuda_home:
                raise RuntimeError("cannot reconfigure the selected DeepGEMM compiler")
            return original_init(str(overlay.root), str(cuda_home))

        cpp.init = guarded_init
        cpp.init(str(library_root), str(cuda_home))
        info.update(overlay.metadata)
        info.update(
            echo_revision=revision,
            echo_checkout_diff_sha256=checkout_diff,
            deep_gemm_version=version,
            cpp_extension_path=str(binary),
            cpp_extension_sha256=_sha256(binary.read_bytes()),
            cuda_home=str(cuda_home),
            initializer_sha256=_sha256(Path(__file__).read_bytes()),
            jit_cache_dir=os.environ.get("DG_JIT_CACHE_DIR"),
            jit_key="DeepGEMM .cuh content digest + compiler signature/flags + generated code",
            closed_behavior="keep selected compiler configuration; refuse further init or scopes",
        )
        yield info
    finally:
        info["scope_closed"] = True
        _PROCESS_STATE = "closed"
