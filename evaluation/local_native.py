"""Identify actual mapped local native DSOs without loading CUDA or searching caches.

Device/inode and file timestamps are checked locally but are deliberately absent
from the returned cross-process identity. An unmapped cache file is never input.
"""

import hashlib
import os
import re
import stat
from pathlib import Path

_LOCAL_DSO = re.compile(r"(?:lib)?cxldsagr_[^/]+\.so(?:\.[0-9]+)*$")


def _hash_stream(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def _stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _category(name):
    stem = re.sub(r"\.so(?:\.[0-9]+)*$", "", name.removeprefix("lib"))
    if stem == "cxldsagr_echo_indexer" or stem.startswith("cxldsagr_echo_indexer_"):
        return "echo_indexer"
    if stem == "cxldsagr_kv_transfer" or stem.startswith("cxldsagr_kv_transfer_"):
        return "record_transfer"
    return "other_local_native"


def collect_local_native_artifacts(required=True):
    """Return every mapped cxldsagr DSO with stable path/hash/size identity.

    Required mode additionally demands exactly one ECHO and one common transfer
    DSO. Future cxldsagr bridge libraries are included automatically. Missing,
    deleted, ambiguous, inode-replaced or concurrently changed files fail loudly.
    Call before measurement and compare with a second call after measurement.
    """
    if type(required) is not bool:
        raise TypeError("required must be bool")
    mapped = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        raw_path = fields[5]
        deleted = raw_path.endswith(" (deleted)")
        path_text = raw_path.removesuffix(" (deleted)")
        path = Path(path_text)
        if not _LOCAL_DSO.fullmatch(path.name):
            continue
        if deleted:
            raise RuntimeError(f"Mapped local native library was deleted: {path}")
        if not path.is_absolute():
            raise RuntimeError(f"Mapped local native library has no absolute path: {path}")
        major, minor = (int(value, 16) for value in fields[3].split(":"))
        file_identity = (os.makedev(major, minor), int(fields[4]))
        if file_identity[1] <= 0:
            raise RuntimeError(f"Mapped local native library has no file inode: {path}")
        mapped.setdefault(path, set()).add(file_identity)

    result = []
    for path, identities in sorted(mapped.items()):
        if len(identities) != 1:
            raise RuntimeError(
                f"Mapped local native library has inconsistent segment identities: {path}"
            )
        (mapped_identity,) = identities
        # Open once, validate that descriptor against the mappings, hash that
        # descriptor, and recheck both descriptor and current pathname. This
        # detects replacement between reading maps/open/hash/path revalidation.
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino) != mapped_identity
            ):
                raise RuntimeError(f"Mapped local native library inode/device was replaced: {path}")
            sha256 = _hash_stream(stream)
            after = os.fstat(stream.fileno())
            current = path.stat()
            if _stat_identity(before) != _stat_identity(after) or _stat_identity(
                after
            ) != _stat_identity(current):
                raise RuntimeError(f"Mapped local native library changed while hashing: {path}")
        result.append(
            {
                "name": path.name,
                "category": _category(path.name),
                "library": {"path": str(path), "sha256": sha256, "bytes": before.st_size},
            }
        )

    for category in ("echo_indexer", "record_transfer"):
        count = sum(item["category"] == category for item in result)
        if count > 1 or required and count != 1:
            raise RuntimeError(f"Expected one mapped {category} library, found {count}")
    return result
