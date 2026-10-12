"""Mapped native artifacts retain their identity checks after helper migration."""

import os

import pytest

from evaluation import native_provenance as provenance


def test_loaded_native_inventory_checks_mapped_inode_and_digest(tmp_path):
    binary = tmp_path / "cxldsagr_test.so"
    binary.write_bytes(b"actual mapped file")
    stat = binary.stat()
    maps = tmp_path / "maps"
    prefix = (
        f"1000-2000 r-xp 0000 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} {stat.st_ino} "
    )
    maps.write_text(prefix + str(binary) + "\n" + prefix + str(binary) + "\n")
    assert provenance.loaded_native_artifacts(maps) == {
        str(binary): provenance.file_identity(binary)
    }
    replacement = tmp_path / "new.so"
    replacement.write_bytes(binary.read_bytes())
    replacement.replace(binary)
    with pytest.raises(ValueError, match="replaced"):
        provenance.loaded_native_artifacts(maps)


def test_deleted_mapped_library_is_not_silently_ignored(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text("1000-2000 r-xp 0000 00:00 1 /tmp/cxldsagr_test.so (deleted)\n")
    with pytest.raises(ValueError, match="deleted"):
        provenance.loaded_native_artifacts(maps)


def test_native_drift_and_unwarmed_artifacts_are_rejected(monkeypatch):
    before = {"/fixture/cxldsagr.so": {"size": 1, "sha256": "old"}}
    after = {**before, "/fixture/cxldsagr_new.so": {"size": 1, "sha256": "new"}}
    monkeypatch.setattr(provenance, "loaded_native_artifacts", lambda: after)
    with pytest.raises(ValueError, match="changed"):
        provenance.verify_native_artifacts(before)
    assert provenance.verify_native_artifacts(before, allow_additions=True) == after
