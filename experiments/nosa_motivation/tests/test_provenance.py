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


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_cross_category_publication_failure_rolls_back_owned_paths(
    tmp_path, monkeypatch, failure_type
):
    sources, targets = {}, {}
    for name in ("data", "profile"):
        sources[name] = tmp_path / "staged" / name
        sources[name].mkdir(parents=True)
        (sources[name] / "artifact").write_text(name)
        targets[name] = tmp_path / "published" / name
    original = shutil.copytree
    failure = failure_type("simulated partial copy failure")

    def copy_then_fail(source, target, **kwargs):
        original(source, target, **kwargs)
        if source == sources["profile"]:
            raise failure

    monkeypatch.setattr(provenance.shutil, "copytree", copy_then_fail)
    with pytest.raises(failure_type) as caught:
        provenance.publish_directories(sources, targets)
    assert caught.value is failure
    assert all(not path.exists() for path in targets.values())
    assert all((path / "artifact").exists() for path in sources.values())


def test_publication_preserves_existing_target(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    owned = tmp_path / "owned"
    source.mkdir()
    target.mkdir()
    (target / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        provenance.publish_directories(
            {"data": source, "profile": source}, {"data": owned, "profile": target}
        )
    assert not owned.exists()
    assert (target / "keep").read_text() == "existing"
    assert source.is_dir()


@pytest.mark.parametrize(
    ("publication_error_type", "cleanup_error_type"),
    [(OSError, OSError), (KeyboardInterrupt, OSError), (OSError, SystemExit)],
)
def test_publication_retains_all_errors_and_continues_rollback(
    tmp_path, monkeypatch, publication_error_type, cleanup_error_type
):
    sources, targets = {}, {}
    for name in ("data", "profile", "report"):
        sources[name] = tmp_path / "staged" / name
        sources[name].mkdir(parents=True)
        (sources[name] / "artifact").write_text(name)
        targets[name] = tmp_path / "published" / name
    original_copy = shutil.copytree
    original_remove = shutil.rmtree
    publication_error = publication_error_type("partial report copy")
    cleanup_errors = {
        targets["report"]: cleanup_error_type("report cleanup failed"),
        targets["profile"]: cleanup_error_type("profile cleanup failed"),
    }
    attempted = []

    def copy_then_fail(source, target, **kwargs):
        original_copy(source, target, **kwargs)
        if source == sources["report"]:
            raise publication_error

    def remove_or_fail(target, *args, **kwargs):
        attempted.append(target)
        if target in cleanup_errors:
            raise cleanup_errors[target]
        return original_remove(target, *args, **kwargs)

    monkeypatch.setattr(provenance.shutil, "copytree", copy_then_fail)
    monkeypatch.setattr(provenance.shutil, "rmtree", remove_or_fail)
    with pytest.raises(BaseExceptionGroup) as caught:
        provenance.publish_directories(sources, targets)

    expected_errors = (publication_error, *cleanup_errors.values())
    assert len(caught.value.exceptions) == len(expected_errors)
    assert all(
        actual is expected
        for actual, expected in zip(caught.value.exceptions, expected_errors, strict=True)
    )
    assert attempted == [targets["report"], targets["profile"], targets["data"]]
    assert not targets["data"].exists()
    assert all((target / "artifact").is_file() for target in cleanup_errors)
    assert all((path / "artifact").read_text() == name for name, path in sources.items())
