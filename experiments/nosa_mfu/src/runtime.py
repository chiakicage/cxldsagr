"""Precision controls and the compute libraries actually mapped after warmup."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def runtime_settings(torch):
    matmul = torch.backends.cuda.matmul
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": matmul.allow_tf32,
        "matmul_allow_bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
        "matmul_allow_fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
        "matmul_allow_fp16_accumulation": getattr(matmul, "allow_fp16_accumulation", None),
        "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(("FLASHINFER_", "CXLDSAGR_SM90_", "TRITON_"))
            or name
            in ("NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "TORCH_BLAS_PREFER_CUBLASLT")
        },
    }


def verify_runtime_settings(expected, torch):
    if runtime_settings(torch) != expected:
        raise ValueError("Precision or backend dispatch settings changed during execution")


def _mapped_libraries(maps_path):
    result = {}
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
            raise ValueError(f"Mapped compute library was deleted: {name}")
        path = Path(name)
        stat = path.stat()
        if stat.st_ino != int(fields[4]) or tuple(int(v, 16) for v in fields[3].split(":")) != (
            os.major(stat.st_dev),
            os.minor(stat.st_dev),
        ):
            raise ValueError(f"Mapped compute library was replaced: {name}")
        result[str(path.resolve())] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }
    if not result:
        raise ValueError("No mapped compute libraries after model warmup")
    return dict(sorted(result.items()))


def native_artifacts(maps_path=Path("/proc/self/maps")):
    """Hash the loaded Torch, BLAS, FlashInfer and project .so bytes.

    Capture follows every workload's warmup and precedes timed samples. This
    identifies the executable libraries rather than assuming equal package
    version strings imply equal binaries. Driver-generated device code is not
    part of this host shared-library inventory.
    """
    before = _mapped_libraries(maps_path)
    artifacts = {}
    for name, observation in before.items():
        with Path(name).open("rb") as stream:
            artifacts[name] = {
                **observation,
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
            }
    if _mapped_libraries(maps_path) != before:
        raise ValueError("Mapped compute libraries changed while reading their bytes")
    return artifacts


def verify_native_artifacts(expected, maps_path=Path("/proc/self/maps")):
    # Same-size writes can retain identical timestamps on coarse filesystems.
    # Re-read bytes once after the run rather than treating stat equality as proof.
    if native_artifacts(maps_path) != expected:
        raise ValueError("Mapped compute library set or file identity changed during execution")
