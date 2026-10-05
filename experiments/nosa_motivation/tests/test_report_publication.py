"""Publication cleanup contracts using a stub audit and report body."""

import json
import os
from pathlib import Path

import pytest

from experiments.nosa_motivation.src import report


@pytest.fixture
def publication_case(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    destination = tmp_path / "published"
    audit = {"fixture": "accepted"}
    metadata = {"schema": report.BENCH_SCHEMA}

    def allocate_staging(*, prefix):
        staging.mkdir()
        return str(staging)

    def write_fixture(directory, target, metadata, rows, audit):
        (target / "report.json").write_text(json.dumps({"status": "complete"}))
        (target / "results.md").write_text("Complete fixture report.\n")

    monkeypatch.setattr(report, "audit_run", lambda *args, **kwargs: (metadata, [], audit))
    monkeypatch.setattr(report.tempfile, "mkdtemp", allocate_staging)
    monkeypatch.setattr(report, "_write_report", write_fixture)
    return staging, destination, audit


def fail_staging_removal(monkeypatch, staging, failure):
    # Inject below shutil.rmtree so the old ignore_errors=True really suppresses
    # OSError. Replacing rmtree with an unconditional raise would miss that bug.
    original = os.rmdir

    def remove_or_fail(path, *args, **kwargs):
        if Path(path) == staging:
            raise failure
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "rmdir", remove_or_fail)


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_report_failure_propagates_original_after_successful_cleanup(
    publication_case, monkeypatch, failure_type
):
    staging, destination, _ = publication_case
    failure = failure_type("report generation failed")

    def fail_write(*args):
        (staging / "partial").write_text("unfinished")
        raise failure

    monkeypatch.setattr(report, "_write_report", fail_write)
    with pytest.raises(failure_type) as caught:
        report.write_report("input", destination)
    assert caught.value is failure
    assert not staging.exists()
    assert not destination.exists()


@pytest.mark.parametrize(
    ("phase", "failure_type", "cleanup_type"),
    [
        ("write", OSError, OSError),
        ("write", KeyboardInterrupt, OSError),
        ("publish", OSError, KeyboardInterrupt),
    ],
)
def test_report_and_cleanup_failures_keep_both_exception_objects(
    publication_case, monkeypatch, phase, failure_type, cleanup_type
):
    staging, destination, _ = publication_case
    failure = failure_type("report failed")
    cleanup_failure = cleanup_type("staging cleanup failed")
    fail_staging_removal(monkeypatch, staging, cleanup_failure)

    if phase == "write":

        def fail_write(*args):
            raise failure

        monkeypatch.setattr(report, "_write_report", fail_write)
    else:
        original_copy = report.shutil.copytree

        def copy_then_fail(source, target, **kwargs):
            original_copy(source, target, **kwargs)
            raise failure

        monkeypatch.setattr(report.shutil, "copytree", copy_then_fail)

    with pytest.raises(BaseExceptionGroup) as caught:
        report.write_report("input", destination)
    assert len(caught.value.exceptions) == 2
    assert caught.value.exceptions[0] is failure
    assert caught.value.exceptions[1] is cleanup_failure
    assert staging.is_dir()
    assert not destination.exists()


@pytest.mark.parametrize("cleanup_type", [OSError, KeyboardInterrupt])
def test_cleanup_only_failure_preserves_complete_committed_report(
    publication_case, monkeypatch, cleanup_type
):
    staging, destination, _ = publication_case
    cleanup_failure = cleanup_type("cleanup after commit failed")
    fail_staging_removal(monkeypatch, staging, cleanup_failure)

    with pytest.raises(cleanup_type) as caught:
        report.write_report("input", destination)
    assert caught.value is cleanup_failure
    assert json.loads((destination / "report.json").read_text()) == {"status": "complete"}
    assert (destination / "results.md").read_text() == "Complete fixture report.\n"
    assert staging.is_dir()


def test_existing_destination_is_preserved_before_staging_allocation(publication_case):
    staging, destination, _ = publication_case
    destination.mkdir()
    (destination / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        report.write_report("input", destination)
    assert (destination / "keep").read_text() == "existing"
    assert not staging.exists()
