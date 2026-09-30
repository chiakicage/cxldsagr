"""Scoped, source-pinned 64-bit address correction for ECHO extend recall.

The native kernel forms host_idx * 576 in signed int32. A host slot above
floor(INT32_MAX / 576) therefore addresses the wrong location despite valid
token indices. Clone its Triton JIT object with a single promotion before the
multiply; leave the Python wrapper, scheduling, copies, and checkout unchanged.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

RECALL_ADDRESS_PATCH_ID = "echo_extend_recall_int64_address_v1"
SOURCE_SHA256 = "8d978bd94de96582fc697d8264a0565ba8dd0e1056a1ba5b014425623f086ea5"
PATCHED_SHA256 = "bd5ae4998fb5733b50c00d1f6ea1658e3cb94ebb64fd0d5eeabcffd4137adb63"
BEFORE = "host_idx = tid * BLOCK + i"
AFTER = "host_idx = (tid * BLOCK + i).to(tl.int64)"


def patched_recall_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("ECHO recall source SHA does not match the approved address fix")
    if source.count(BEFORE) != 1:
        raise ValueError("ECHO recall address anchor must occur exactly once")
    modified = source.replace(BEFORE, AFTER, 1)
    if hashlib.sha256(modified.encode()).hexdigest() != PATCHED_SHA256:
        raise ValueError("ECHO recall patched SHA mismatch")
    return modified


def _new_jit(original):
    import triton

    return triton.JITFunction(
        original.fn,
        version=original.version,
        do_not_specialize=original.do_not_specialize,
        do_not_specialize_on_alignment=original.do_not_specialize_on_alignment,
        debug=original.debug,
        noinline=original.noinline,
        repr=original._repr,
        launch_metadata=original.launch_metadata,
    )


@contextmanager
def scoped_extend_recall_address_fix(recall_module):
    name = "_recall_update_extend_kernel"
    original = getattr(recall_module, name)
    modified = patched_recall_source(original.src)
    replacement = _new_jit(original)
    # A fresh JIT object has no compiled device cache or callers with cached hashes.
    replacement._unsafe_update_src(modified)
    info = {
        "patch_id": RECALL_ADDRESS_PATCH_ID,
        "source_sha256": SOURCE_SHA256,
        "patched_sha256": PATCHED_SHA256,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "only extend recall host index promoted before MLA element-offset multiplication",
        "source_files_modified": False,
    }
    setattr(recall_module, name, replacement)
    try:
        yield info
    finally:
        setattr(recall_module, name, original)
