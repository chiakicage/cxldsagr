"""Only the exact nvidia-smi deadline line may differ across matrix snapshots."""

from pathlib import Path

import pytest

from experiments.deepseek_v32_mfu.src import shape_matrix_sources as sources


def source_bytes(timeout=20):
    return (
        '"""Fixture source."""\n'
        "def gather_hardware(device):\n"
        '    command = ["nvidia-smi", str(device)]\n'
        "    completed = subprocess.run(command, check=True, capture_output=True, "
        f"text=True, timeout={timeout})\n"
        "    response = urllib.request.urlopen(url, timeout=20)\n"
        "    return completed\n"
    ).encode()


def snapshot(tmp_path, name, data, *, source_name=sources.HARDWARE_SOURCE, current=None):
    archive = tmp_path / name
    source = archive / source_name
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    (archive / "operator.py").write_text("unchanged operator\n")
    manifest = {
        source_name: sources.digest(source),
        "operator.py": sources.digest(archive / "operator.py"),
    }
    return sources.source_snapshot(manifest, archive, current_hardware=current)


def test_exact_deadline_change_retains_both_raw_hashes_and_current_binding(tmp_path):
    current = tmp_path / "current.py"
    current.write_bytes(source_bytes(120))
    old = snapshot(tmp_path, "old", source_bytes(20), current=current)
    new = snapshot(tmp_path, "new", source_bytes(120), current=current)
    assert old["source_identity_sha256"] != new["source_identity_sha256"]
    assert old["comparison_source_sha256"] == new["comparison_source_sha256"]
    result = sources.compare_source_snapshots([old, new])
    assert result["matrix_hardware_probe_deadline_differs"] is True
    assert len(result["original_source_identity_sha256"]) == 2
    assert {item["nvidia_smi_timeout_seconds"] for item in result["hardware_probe_variants"]} == {
        20,
        120,
    }
    assert any(str(current) in item["paths"] for item in result["hardware_probe_variants"])


@pytest.mark.parametrize(
    "changed",
    [
        source_bytes(120).replace(b"Fixture source", b"Fixture sources"),
        source_bytes(120).replace(b"urlopen(url, timeout=20)", b"urlopen(url, timeout=120)"),
        source_bytes(1200),
        source_bytes(120) + sources._PROBE_LINES[120],
        source_bytes(120).replace(b"    completed =", b"        completed ="),
        source_bytes(120).replace(b"    completed =", b"#    completed ="),
    ],
)
def test_rejects_any_other_hardware_source_delta(tmp_path, changed):
    current = tmp_path / "current.py"
    current.write_bytes(source_bytes(20))
    with pytest.raises(ValueError, match="Hardware probe"):
        snapshot(tmp_path, "changed", changed, current=current)


def test_cross_shape_comparison_rejects_unrelated_change_without_current_reference(tmp_path):
    old = snapshot(tmp_path, "old", source_bytes(20))
    new = snapshot(tmp_path, "new", source_bytes(120).replace(b"return completed", b"return None"))
    with pytest.raises(ValueError, match="Execution sources differ"):
        sources.compare_source_snapshots([old, new])


def test_canonicalization_never_rewrites_an_earlier_comment_lookalike(tmp_path):
    commented = {timeout: b"#" + line for timeout, line in sources._PROBE_LINES.items()}
    first = snapshot(tmp_path, "first", commented[120] + commented[20] + source_bytes(120))
    second = snapshot(tmp_path, "second", commented[20] + commented[120] + source_bytes(120))
    with pytest.raises(ValueError, match="Execution sources differ"):
        sources.compare_source_snapshots([first, second])


def test_wrong_path_and_manifest_key_change_are_not_compatible(tmp_path):
    old = snapshot(tmp_path, "old", source_bytes(20), source_name="other/profile_hardware.py")
    new = snapshot(tmp_path, "new", source_bytes(120), source_name="other/profile_hardware.py")
    with pytest.raises(ValueError, match="Execution sources differ"):
        sources.compare_source_snapshots([old, new])
    renamed = snapshot(tmp_path, "renamed", source_bytes(20), source_name="renamed.py")
    with pytest.raises(ValueError, match="Execution sources differ"):
        sources.compare_source_snapshots([old, renamed])


def test_rejects_stale_archive_hash_and_changed_current_bytes(tmp_path):
    item = snapshot(tmp_path, "archive", source_bytes(20))
    manifest = dict(item["comparison_source_sha256"])
    archive = tmp_path / "archive"
    (archive / sources.HARDWARE_SOURCE).write_bytes(source_bytes(120))
    with pytest.raises(ValueError, match="Archived source changed"):
        sources.source_snapshot(manifest, archive)
    current = tmp_path / "current.py"
    current.write_bytes(source_bytes(120) + b"# extra change\n")
    with pytest.raises(ValueError, match="differs beyond"):
        snapshot(tmp_path, "changed_current", source_bytes(20), current=current)


def test_every_archived_and_current_path_is_registered_with_evidence(tmp_path):
    current = tmp_path / "current.py"
    current.write_bytes(source_bytes(120))
    snapshot(tmp_path, "archive", source_bytes(20))
    archive = tmp_path / "archive"
    manifest = {
        name: sources.digest(archive / name) for name in (sources.HARDWARE_SOURCE, "operator.py")
    }
    seen = set()

    def tracked(path):
        seen.add(Path(path).resolve())
        return sources.digest(path)

    sources.source_snapshot(manifest, archive, current_hardware=current, digest_file=tracked)
    assert seen == {current, archive / sources.HARDWARE_SOURCE, archive / "operator.py"}


def test_variant_bytes_must_still_match_original_manifest_after_initial_digest(tmp_path):
    snapshot(tmp_path, "archive", source_bytes(20))
    archive = tmp_path / "archive"
    source = archive / sources.HARDWARE_SOURCE
    manifest = {sources.HARDWARE_SOURCE: sources.digest(source)}
    calls = 0

    def mutated_digest(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            source.write_bytes(source_bytes(120))
        return sources.digest(path)

    with pytest.raises(ValueError, match="Archived source changed"):
        sources.source_snapshot(manifest, archive, digest_file=mutated_digest)
