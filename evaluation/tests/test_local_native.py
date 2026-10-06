"""CPU-only mapped-DSO identity and pathname race checks."""

import hashlib
import os
from pathlib import Path

import pytest

from evaluation import local_native as collector


def make_file(tmp_path, name, data=b"native bytes"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def map_line(path, *, inode=None, device=None, suffix=""):
    value = path.stat()
    device = value.st_dev if device is None else device
    dev = f"{os.major(device):02x}:{os.minor(device):02x}"
    inode = value.st_ino if inode is None else inode
    return f"1000-2000 r-xp 00000000 {dev} {inode} {path}{suffix}\n"


def set_maps(monkeypatch, text):
    original = Path.read_text

    def read(path, *args, **kwargs):
        return text if path == Path("/proc/self/maps") else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)


def expected_files(tmp_path):
    return [
        make_file(tmp_path, "cxldsagr_echo_indexer_abc.so"),
        make_file(tmp_path, "cxldsagr_kv_transfer.so"),
    ]


def test_actual_maps_only_duplicate_segments_and_future_bridge(tmp_path, monkeypatch):
    files = expected_files(tmp_path)
    files.append(make_file(tmp_path, "libcxldsagr_future_bridge_123.so"))
    make_file(tmp_path, "cxldsagr_unmapped_cache_candidate.so")
    set_maps(monkeypatch, "".join(map_line(path) * 3 for path in files))
    result = collector.collect_local_native_artifacts()
    assert len(result) == 3
    assert {x["category"] for x in result} == {
        "echo_indexer",
        "record_transfer",
        "other_local_native",
    }
    assert [x["library"]["path"] for x in result] == sorted(str(path) for path in files)
    for item in result:
        assert set(item) == {"name", "category", "library"}
        assert set(item["library"]) == {"path", "sha256", "bytes"}
        assert item["library"]["sha256"] == hashlib.sha256(b"native bytes").hexdigest()


@pytest.mark.parametrize("missing", [0, 1])
def test_missing_required_category(tmp_path, monkeypatch, missing):
    files = expected_files(tmp_path)
    set_maps(monkeypatch, map_line(files[missing]))
    with pytest.raises(RuntimeError, match="Expected one mapped"):
        collector.collect_local_native_artifacts()
    assert len(collector.collect_local_native_artifacts(required=False)) == 1


def test_optional_unloaded_process_returns_empty(monkeypatch):
    set_maps(monkeypatch, "1000-2000 rw-p 00000000 00:00 0 [heap]\n")
    assert collector.collect_local_native_artifacts(required=False) == []


def test_versioned_library_names_retain_required_roles(tmp_path, monkeypatch):
    files = [
        make_file(tmp_path, "libcxldsagr_echo_indexer.so.1"),
        make_file(tmp_path, "cxldsagr_kv_transfer.so.2.0"),
    ]
    set_maps(monkeypatch, "".join(map_line(path) for path in files))
    result = collector.collect_local_native_artifacts()
    assert {row["category"] for row in result} == {"echo_indexer", "record_transfer"}


@pytest.mark.parametrize(
    "fault", ["deleted", "missing", "replaced", "device", "inode", "inconsistent"]
)
def test_fail_for_invalid_mapped_backing(tmp_path, monkeypatch, fault):
    echo, transfer = expected_files(tmp_path)
    echo_line = map_line(echo)
    if fault == "deleted":
        echo_line = map_line(echo, suffix=" (deleted)")
    elif fault == "missing":
        echo.unlink()
    elif fault == "replaced":
        make_file(tmp_path, "replacement", b"new").replace(echo)
    elif fault == "device":
        echo_line = map_line(echo, device=os.makedev(255, 255))
    elif fault == "inode":
        echo_line = map_line(echo, inode=echo.stat().st_ino + 1)
    else:
        echo_line += map_line(echo, inode=echo.stat().st_ino + 1)
    set_maps(monkeypatch, echo_line + map_line(transfer))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        collector.collect_local_native_artifacts()


def test_multiple_echo_libraries_are_ambiguous(tmp_path, monkeypatch):
    files = expected_files(tmp_path) + [make_file(tmp_path, "cxldsagr_echo_indexer_other.so")]
    set_maps(monkeypatch, "".join(map_line(path) for path in files))
    with pytest.raises(RuntimeError, match="Expected one mapped echo_indexer"):
        collector.collect_local_native_artifacts()


@pytest.mark.parametrize("change", ["in_place", "replace"])
def test_hashing_race_is_rejected(tmp_path, monkeypatch, change):
    files = expected_files(tmp_path)
    set_maps(monkeypatch, "".join(map_line(path) for path in files))
    original = collector._hash_stream

    def changing(stream):
        path = Path(stream.name)
        if change == "in_place":
            path.write_bytes(b"new longer bytes")
        else:
            make_file(tmp_path, "replacement", b"new file").replace(path)
        return original(stream)

    monkeypatch.setattr(collector, "_hash_stream", changing)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        collector.collect_local_native_artifacts()
