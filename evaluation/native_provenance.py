"""Identify mapped native libraries and detect changes during measurements."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
