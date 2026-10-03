"""Scoped promotion before ECHO's large index-buffer byte-offset arithmetic."""

import hashlib
from contextlib import contextmanager
from pathlib import Path

INDEX_ADDRESS_PATCH_ID = "echo_index_read_int64_address_v1"
INDEX_SOURCE_SHA256 = "8cf6118057d3747fe0f37394b3dab79dc8f6634dfa8a406f7e53236048eb1c6c"


@contextmanager
def scoped_index_address_fix(accessor, torch):
    source_sha = hashlib.sha256(Path(accessor.__file__).read_bytes()).hexdigest()
    if source_sha != INDEX_SOURCE_SHA256:
        raise ValueError("ECHO index accessor source differs from the approved version")
    originals = {cls: vars(cls)["execute"] for cls in (accessor.GetK, accessor.GetS)}

    def wrap(original):
        def execute(cls, pool, buf, seq_len, page_indices):
            # Upstream multiplies int32 page IDs by an 8448-byte stride. The
            # returned indices must be int64 BEFORE multiplication, not after.
            if buf.numel() > 2**31 - 1:
                page_indices = page_indices.to(torch.int64)
            return original.__get__(None, cls)(pool, buf, seq_len, page_indices)

        return classmethod(execute)

    try:
        for cls, original in originals.items():
            cls.execute = wrap(original)
        yield {
            "patch_id": INDEX_ADDRESS_PATCH_ID,
            "source_sha256": source_sha,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "GetK/GetS page IDs promoted before byte-offset arithmetic for buffers above INT32_MAX bytes; small buffers unchanged",
            "source_files_modified": False,
        }
    finally:
        for cls, original in originals.items():
            cls.execute = original
