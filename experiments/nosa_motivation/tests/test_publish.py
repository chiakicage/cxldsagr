"""Exercise the publisher's commit boundary with a stub successful builder."""

import json
from pathlib import Path

import pytest

from experiments.nosa_motivation.src import publish as publisher


def test_committed_publication_survives_staging_removal_failure(tmp_path, monkeypatch):
    destination = tmp_path / "published"
    acceptance = {"status": "complete"}
    staging_roots = []
    cleanup_error = OSError("staging removal failed after commit")
    original_rmdir = Path.rmdir

    def build_publication(bench_dir, profile_dir, traces, target, attention_dir=None):
        target.mkdir()
        (target / "publication.json").write_text(json.dumps(acceptance))
        (target / "results.md").write_text("Complete fixture report.\n")
        staging_roots.append(target.parent)
        return acceptance

    def remove_or_fail(path):
        if path in staging_roots:
            raise cleanup_error
        return original_rmdir(path)

    monkeypatch.setattr(publisher, "_build_publication", build_publication)
    monkeypatch.setattr(Path, "rmdir", remove_or_fail)

    with pytest.raises(OSError) as caught:
        publisher.publish("bench", "profile", "traces", destination)

    assert caught.value is cleanup_error
    assert json.loads((destination / "publication.json").read_text()) == acceptance
    assert (destination / "results.md").read_text() == "Complete fixture report.\n"
    assert len(staging_roots) == 1
    assert staging_roots[0].is_dir()
    assert list(staging_roots[0].iterdir()) == []
