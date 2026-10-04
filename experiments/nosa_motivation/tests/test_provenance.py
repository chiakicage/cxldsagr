"""Native provenance observes mapped artifacts and detects dependency drift."""

import os
import shutil

import pytest

from experiments.nosa_motivation.src import provenance


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


def test_header_inventory_detects_dirty_content_and_new_headers(tmp_path):
    (tmp_path / "version.h").write_text("unchanged version")
    other = tmp_path / "detail.h"
    other.write_text("old")
    old = provenance.header_tree_identity(tmp_path)
    other.write_text("new")
    changed = provenance.header_tree_identity(tmp_path)
    assert old["sha256"] != changed["sha256"]
    assert old["files"]["version.h"] == changed["files"]["version.h"]
    (tmp_path / "new.h").write_text("new dependency")
    assert changed["sha256"] != provenance.header_tree_identity(tmp_path)["sha256"]


def test_native_drift_and_unwarmed_artifacts_are_rejected(monkeypatch):
    before = {"/fixture/cxldsagr.so": {"size": 1, "sha256": "old"}}
    after = {**before, "/fixture/cxldsagr_new.so": {"size": 1, "sha256": "new"}}
    monkeypatch.setattr(provenance, "loaded_native_artifacts", lambda: after)
    with pytest.raises(ValueError, match="changed"):
        provenance.verify_native_artifacts(before)
    assert provenance.verify_native_artifacts(before, allow_additions=True) == after
    monkeypatch.setattr(provenance, "native_build_identity", lambda: {"headers": "new"})
    with pytest.raises(ValueError, match="headers changed"):
        provenance.verify_native_build_identity({"headers": "old"})


def test_cross_category_publication_failure_rolls_back_owned_paths(tmp_path, monkeypatch):
    sources, targets = {}, {}
    for name in ("data", "profile"):
        sources[name] = tmp_path / "staged" / name
        sources[name].mkdir(parents=True)
        (sources[name] / "artifact").write_text(name)
        targets[name] = tmp_path / "published" / name
    original = shutil.copytree

    def copy_then_fail(source, target, **kwargs):
        original(source, target, **kwargs)
        if source == sources["profile"]:
            raise OSError("simulated partial copy failure")

    monkeypatch.setattr(provenance.shutil, "copytree", copy_then_fail)
    with pytest.raises(OSError, match="partial"):
        provenance.publish_directories(sources, targets)
    assert all(not path.exists() for path in targets.values())
    assert all((path / "artifact").exists() for path in sources.values())


def test_publication_preserves_existing_target(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (target / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        provenance.publish_directories({"data": source}, {"data": target})
    assert (target / "keep").read_text() == "existing"
