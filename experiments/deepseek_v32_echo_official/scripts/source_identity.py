"""Separate check compatibility from complete Engine execution provenance.

Every run keeps all launcher/observer/profile source hashes. Check reuse compares
the exact model worker, its execution helpers, and all recorded runtime/input
fields. Historical sources are verified from an immutable per-run byte archive.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
EXPERIMENT_PREFIX = "experiments/deepseek_v32_echo_official/"
REPRODUCTION_PREFIX = "3rdparty/ECHO/reproduction/cxldsagr/"

CORE_SOURCES = frozenset(
    {
        EXPERIMENT_PREFIX + "src/engine_run.py",
        REPRODUCTION_PREFIX + "src/preflight.py",
        REPRODUCTION_PREFIX + "src/capacity.py",
        REPRODUCTION_PREFIX + "src/workload_client.py",
    }
)
EXCLUDED_SOURCE_ROLES = {
    EXPERIMENT_PREFIX + "scripts/run_engine.py": "launcher and observer orchestration",
    EXPERIMENT_PREFIX + "scripts/run_engine.sh": "environment and launcher entrypoint",
    EXPERIMENT_PREFIX + "scripts/profile_lifecycle.py": "profile process observation and cleanup",
    EXPERIMENT_PREFIX
    + "scripts/source_identity.py": "source archival and check compatibility audit",
    EXPERIMENT_PREFIX + "src/engine_profile_hooks.py": "profile-only observational hooks",
    REPRODUCTION_PREFIX
    + "scripts/run.py": "reused launcher and observer helpers; effective environment remains compared",
    REPRODUCTION_PREFIX
    + "scripts/lifecycle.py": "reused process observation and lifecycle helpers",
    REPRODUCTION_PREFIX
    + "env/activate.sh": "environment activation; effective environment remains compared",
}

COMPARISON_RULE = {
    "id": "echo-engine-exact-worker-runtime-v1",
    "scope": "Compatibility with a performance-only structure/finite-output check; never numerical-equivalence acceptance.",
    "identity_fields": "All execution_identity fields are compared exactly; only explicitly listed entries of reproduction_source_sha256 are omitted from this key.",
    "source_policy": "Exact worker and execution-helper bytes are required. Unlisted source paths remain in the key. Omitted source bytes remain archived and bound to each full execution identity.",
    "required_core_sources": sorted(CORE_SOURCES),
    "excluded_source_roles": EXCLUDED_SOURCE_ROLES,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _source_path(name):
    _require(isinstance(name, str), "Source path must be a string")
    path = PurePosixPath(name)
    _require(
        not path.is_absolute()
        and path.parts
        and ".." not in path.parts
        and "\\" not in name
        and path.as_posix() == name,
        f"Source path must be canonical and repository-relative: {name}",
    )
    return Path(*path.parts)


def _sources(record):
    sources = record["reproduction_source_sha256"]
    _require(
        isinstance(sources, dict)
        and sources
        and sources == record["execution_identity"]["reproduction_source_sha256"],
        "Recorded source maps are empty or inconsistent",
    )
    for name, digest in sources.items():
        _source_path(name)
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"Invalid source digest: {name}",
        )
    return sources


def source_identity():
    """Hash the full current worker, helper, launcher, observer and profile set."""
    names = sorted(CORE_SOURCES | EXCLUDED_SOURCE_ROLES.keys())
    result = {}
    for name in names:
        path = (REPO_ROOT / _source_path(name)).resolve(strict=True)
        _require(path.is_relative_to(REPO_ROOT.resolve()), f"Source escapes repository: {name}")
        result[name] = _digest(path.read_bytes())
    return result


def acceptance_identity(execution_identity):
    """Produce an explicit, conservative compatibility key without mutating input."""
    core = _json_copy(execution_identity)
    sources = core["reproduction_source_sha256"]
    _require(CORE_SOURCES <= sources.keys(), "Required worker/execution-helper sources are absent")
    for name in sources:
        _source_path(name)
    core["reproduction_source_sha256"] = {
        name: digest for name, digest in sources.items() if name not in EXCLUDED_SOURCE_ROLES
    }
    return {
        "schema": "echo-engine-acceptance-identity-v1",
        "comparison_rule": _json_copy(COMPARISON_RULE),
        "execution_identity": core,
    }


def source_differences(check_identity, current_identity):
    """Describe all source changes retained outside or inside the compatibility key."""
    previous = check_identity["reproduction_source_sha256"]
    current = current_identity["reproduction_source_sha256"]
    return {
        name: {"check": previous.get(name), "current": current.get(name)}
        for name in sorted(previous.keys() | current.keys())
        if previous.get(name) != current.get(name)
    }


def _read_run(run_dir, record):
    run_dir = Path(run_dir).resolve(strict=True)
    _require(
        run_dir.is_relative_to((ROOT / "output").resolve()),
        "Source archives must remain inside this experiment's output directory",
    )
    run_file = run_dir / "run.json"
    data = run_file.read_bytes()
    _require(json.loads(data) == record, "Supplied record differs from immutable run.json")
    _require(
        record.get("schema") == "echo-sglang-engine-run-v1"
        and record.get("run_id") == run_dir.name
        and record.get("status") == "completed"
        and record.get("exitcode") == 0,
        "Only a completed, successful Engine run may have a verified source archive",
    )
    return run_dir, run_file, data


def verify_source_archive(run_dir, record):
    """Verify historical bytes and the original run file; never consult live sources."""
    run_dir, run_file, run_data = _read_run(run_dir, record)
    manifest_path = run_dir / "source_snapshot.json"
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    expected = _sources(record)
    _require(
        manifest.get("schema") == "echo-engine-execution-source-archive-v1"
        and manifest.get("run_id") == record["run_id"]
        and manifest.get("archived_after_run") is True
        and manifest.get("run_file_sha256_at_archive") == _digest(run_data)
        and set(manifest.get("sources", {})) == set(expected),
        "Source archive is not bound to this exact run and full source set",
    )
    verified = {}
    for name, digest in expected.items():
        row = manifest["sources"][name]
        relative = (Path("source_snapshot") / _source_path(name)).as_posix()
        _require(
            row.get("sha256") == digest and row.get("archive_path") == relative,
            f"Archive source identity/path differs: {name}",
        )
        path = (run_dir / relative).resolve(strict=True)
        _require(
            path.is_relative_to((run_dir / "source_snapshot").resolve())
            and path.is_relative_to(run_dir),
            f"Archive source escapes run directory: {name}",
        )
        _require(_digest(path.read_bytes()) == digest, f"Archived source bytes changed: {name}")
        verified[name] = {**row, "absolute_path": str(path)}
    _require(run_file.read_bytes() == run_data, "Run record changed while checking source archive")
    _require(
        manifest_path.read_bytes() == manifest_data,
        "Source manifest changed during verification",
    )
    return {
        "schema": "echo-engine-source-archive-verification-v1",
        "run_id": record["run_id"],
        "manifest": {"path": str(manifest_path), "sha256": _digest(manifest_data)},
        "run_file": {"path": str(run_file), "sha256": _digest(run_data)},
        "sources": verified,
        "verified": True,
    }


def archive_sources(run_dir, record):
    """Create the full archive after completion, without rewriting the run record."""
    run_dir, run_file, run_data = _read_run(run_dir, record)
    manifest_path = run_dir / "source_snapshot.json"
    archive_dir = run_dir / "source_snapshot"
    if manifest_path.exists() or archive_dir.exists():
        raise FileExistsError("Source snapshot already exists")
    expected = _sources(record)
    contents = {}
    for name, digest in expected.items():
        path = (REPO_ROOT / _source_path(name)).resolve(strict=True)
        _require(path.is_relative_to(REPO_ROOT.resolve()), f"Source escapes repository: {name}")
        data = path.read_bytes()
        _require(_digest(data) == digest, f"Live source no longer matches completed run: {name}")
        contents[name] = data
    _require(run_file.read_bytes() == run_data, "Run record changed before source archival")
    staging = Path(tempfile.mkdtemp(prefix=".source_snapshot_", dir=run_dir))
    installed = False
    manifest_created = False
    primary = None
    try:
        for name, data in contents.items():
            path = staging / _source_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as target:
                target.write(data)
        manifest = {
            "schema": "echo-engine-execution-source-archive-v1",
            "run_id": record["run_id"],
            "archived_after_run": True,
            "run_file_sha256_at_archive": _digest(run_data),
            "scope": "Byte-identical archive created after all live sources matched the completed run. The original run record is unchanged.",
            "sources": {
                name: {
                    "sha256": expected[name],
                    "archive_path": (Path("source_snapshot") / _source_path(name)).as_posix(),
                }
                for name in sorted(contents)
            },
        }
        staging.rename(archive_dir)
        installed = True
        with manifest_path.open("x") as target:
            manifest_created = True
            json.dump(manifest, target, sort_keys=True, indent=2)
            target.write("\n")
        return verify_source_archive(run_dir, record)
    except BaseException as error:
        primary = error
        raise
    finally:
        if primary is not None:
            try:
                if installed:
                    if manifest_created and manifest_path.exists():
                        manifest_path.unlink()
                    shutil.rmtree(archive_dir)
                elif staging.exists():
                    shutil.rmtree(staging)
            except BaseException as cleanup:  # noqa: BLE001
                raise BaseExceptionGroup("Source archival and cleanup failed", [primary, cleanup])
