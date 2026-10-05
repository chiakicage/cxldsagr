"""Provider receipts bind actual mapped bytes and preserve archived metadata."""

import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import pool_scan_provenance as pool


def native_fixture():
    identity = {
        "schema": "nosa-pool-referrers-build-v1",
        "module_name": "_cxldsagr_nosa_pool_referrers",
        "extension_suffix": ".cpython-312-x86_64-linux-gnu.so",
        "abi_manifest_sha256": "a" * 64,
    }
    fingerprint = pool._digest(identity)
    build = {"available": True, "identity": identity, "fingerprint": fingerprint}
    path = f"/fixture/cxldsagr/pool-referrers/{fingerprint}/{identity['module_name']}{identity['extension_suffix']}"
    runtime = {
        "backend": "cpython_native",
        "fallback_reason": None,
        "fingerprint": fingerprint,
        "abi_manifest_sha256": "a" * 64,
        "loaded_binary_path": path,
        "loaded_binary_sha256": "b" * 64,
        "counters": dict.fromkeys(pool.COUNTERS, 0),
    }
    runtime["counters"]["filtered_calls"] = 2
    before = {"runtime": runtime, "artifacts": {path: {"size": 17, "sha256": "b" * 64}}}
    after = deepcopy(before)
    after["runtime"]["counters"]["filtered_calls"] = 5
    return build, {
        "before": before,
        "after": after,
        "counter_delta": {**dict.fromkeys(pool.COUNTERS, 0), "filtered_calls": 3},
        "scope": pool.CASE_SCOPE,
    }


def test_saved_native_provider_checks_no_current_files():
    build, record = native_fixture()
    pool.audit_build(build, deepcopy(build))
    pool.audit_case(record, build, expected_abi_sha256="a" * 64)


@pytest.mark.parametrize(
    "mutation",
    [
        "fingerprint",
        "binary_hash",
        "binary_name",
        "binary_parent",
        "abi",
        "counter_negative",
        "counter_bool",
        "counter_delta",
        "provider_changed",
        "scope",
    ],
)
def test_saved_native_receipts_reject_mismatches(mutation):
    build, record = native_fixture()
    runtime = record["after"]["runtime"]
    if mutation == "fingerprint":
        runtime["fingerprint"] = "c" * 64
    elif mutation == "binary_hash":
        runtime["loaded_binary_sha256"] = "c" * 64
    elif mutation in ("binary_name", "binary_parent"):
        old = runtime["loaded_binary_path"]
        new = (
            str(Path(old).with_name("unrelated.so"))
            if mutation == "binary_name"
            else old.replace(build["fingerprint"], "c" * 64)
        )
        runtime["loaded_binary_path"] = new
        record["after"]["artifacts"] = {new: record["after"]["artifacts"][old]}
    elif mutation == "abi":
        runtime["abi_manifest_sha256"] = "c" * 64
    elif mutation == "counter_negative":
        runtime["counters"]["filtered_calls"] = 1
    elif mutation == "counter_bool":
        runtime["counters"]["audit_errors"] = False
    elif mutation == "counter_delta":
        record["counter_delta"]["filtered_calls"] = 4
    elif mutation == "provider_changed":
        runtime["fallback_reason"] = "unexpected"
    else:
        record["scope"] = "individual request scan time"
    with pytest.raises(ValueError):
        pool.audit_case(record, build, expected_abi_sha256="a" * 64)


def test_build_fingerprint_does_not_accept_rehashed_outer_metadata():
    build, _ = native_fixture()
    build["identity"]["abi_manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        pool.audit_build(build, deepcopy(build))


def test_native_whitelist_must_match_both_saved_source_and_planned_build():
    build, record = native_fixture()
    with pytest.raises(ValueError, match="whitelist"):
        pool.audit_case(record, build, expected_abi_sha256="c" * 64)
    # The source manifest cannot override a conflicting planned ABI identity.
    build["identity"]["abi_manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="whitelist"):
        pool.audit_case(record, build, expected_abi_sha256="a" * 64)


def test_fallback_is_explicit_and_does_not_claim_native_artifacts():
    build = {
        "available": False,
        "reason": "unsupported runtime",
        "source_sha256": {"loader": "d" * 64},
    }
    runtime = {
        "backend": "python_original",
        "fallback_reason": "unsupported runtime",
        "fingerprint": None,
        "abi_manifest_sha256": "a" * 64,
        "loaded_binary_path": None,
        "loaded_binary_sha256": None,
        "counters": dict.fromkeys(pool.COUNTERS, 0),
    }
    record = {
        "before": {"runtime": None, "artifacts": {}},
        "after": {"runtime": runtime, "artifacts": {}},
        "counter_delta": dict.fromkeys(pool.COUNTERS, 0),
        "scope": pool.CASE_SCOPE,
    }
    pool.audit_build(build, deepcopy(build))
    pool.audit_case(record, build, expected_abi_sha256="a" * 64)
    runtime["fallback_reason"] = None
    with pytest.raises(ValueError, match="fallback"):
        pool.audit_case(record, build)


def test_discovery_and_runtime_snapshots_never_prepare_or_import_extension(monkeypatch, tmp_path):
    calls = []
    loader = SimpleNamespace(
        build_info=lambda: calls.append("discover") or {"available": False},
        runtime_info=lambda: None,
        prepare=lambda *args: pytest.fail("provenance prepared native provider"),
    )
    monkeypatch.setitem(sys.modules, "cache.allocator.pool_referrers", loader)
    assert pool.build_info() == {"available": False}
    assert pool.snapshot(maps_path=tmp_path / "absent") == {"runtime": None, "artifacts": {}}
    assert calls == ["discover"]
    with pytest.raises(ValueError, match="not prepared"):
        pool.finish_case({"runtime": None, "artifacts": {}})


def test_selected_mapped_binary_inode_and_bytes_are_checked(tmp_path):
    binary = tmp_path / "provider.so"
    binary.write_bytes(b"retained native bytes")
    runtime = {
        "backend": "cpython_native",
        "loaded_binary_path": str(binary),
        "loaded_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
    stat = binary.stat()
    maps = tmp_path / "maps"
    maps.write_text(
        f"1000-2000 r-xp 0000 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} {stat.st_ino} {binary}\n"
    )
    assert (
        pool._mapped_artifact(runtime, maps)[str(binary)]["sha256"]
        == runtime["loaded_binary_sha256"]
    )
    other = tmp_path / "replacement"
    other.write_bytes(binary.read_bytes())
    other.replace(binary)
    with pytest.raises(ValueError, match="replaced"):
        pool._mapped_artifact(runtime, maps)


def test_old_metadata_compatibility_does_not_weaken_current_source_gate():
    assert pool.audit_gr({"cases": []}) is None
    with pytest.raises(ValueError, match="requires"):
        pool.audit_gr({"cases": []}, required=True)
    assert not pool.requires_provenance({"cache/allocator/budget.py": "old"})
    assert pool.requires_provenance(dict.fromkeys(pool.POOL_SCAN_SOURCES, "new"))
    with pytest.raises(ValueError, match="coverage"):
        pool.requires_provenance({pool.POOL_SCAN_ABI: "new"})


def test_pre_migration_provider_sources_keep_their_saved_abi_binding():
    historical = {
        "models/nosa/_pool_referrers.py": "loader",
        "models/nosa/csrc/pool_referrers.c": "source",
        "models/nosa/csrc/pool_referrers_abi.json": "saved-abi",
    }
    assert pool.requires_provenance(historical)
    assert pool.source_abi_sha256(historical) == "saved-abi"
    del historical["models/nosa/csrc/pool_referrers.c"]
    with pytest.raises(ValueError, match="coverage"):
        pool.source_abi_sha256(historical)


def test_gr_profile_requires_each_actual_case_and_matching_build():
    build, record = native_fixture()
    metadata = {
        "pool_referrers_provenance": {"build_before": build, "build_after": deepcopy(build)},
        "samples": [{"mode": "overlap", "case": "request_000008"}],
        "pool_referrers_cases": [
            {"mode": "overlap", "case": "request_000008", "pool_referrers": record}
        ],
    }
    assert pool.audit_gr(metadata, required=True, profile=True) == build["fingerprint"]
    metadata["pool_referrers_cases"] *= 2
    with pytest.raises(ValueError, match="coverage"):
        pool.audit_gr(metadata, required=True, profile=True)


def test_gr_profile_reopens_formal_provider_and_binary_identity():
    build, record = native_fixture()
    provenance = {"build_before": build, "build_after": deepcopy(build)}
    formal = {
        "pool_referrers_provenance": provenance,
        "cases": [{"model": "nosa", "pool_referrers": record}],
    }
    profile = {
        "pool_referrers_provenance": deepcopy(provenance),
        "pool_referrers_cases": [{"pool_referrers": deepcopy(record)}],
    }
    pool.audit_profile_reference(profile, formal)
    profile["pool_referrers_cases"][0]["pool_referrers"]["after"]["runtime"][
        "loaded_binary_sha256"
    ] = "c" * 64
    with pytest.raises(ValueError, match="provider/binary"):
        pool.audit_profile_reference(profile, formal)
    with pytest.raises(ValueError, match="build differs"):
        pool.audit_profile_reference({}, formal)
    pool.audit_profile_reference({}, None)


def test_retained_runtime_marker_cannot_hide_removed_build_record():
    _, record = native_fixture()
    with pytest.raises(ValueError, match="requires"):
        pool.audit_gr({"cases": [{"model": "nosa", "pool_referrers": record}]})


def test_all_gr_source_inventories_include_c_and_only_whitelisted_json(tmp_path, monkeypatch):
    from evaluation import provenance

    repo = tmp_path / "repo"
    for name in (
        *pool.POOL_SCAN_SOURCES,
        "evaluation/pool_scan_provenance.py",
        "cache/allocator/csrc/ignored.json",
    ):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    output = tmp_path / "snapshot"
    output.mkdir()
    monkeypatch.setattr(provenance, "ROOT", repo)
    provenance.source_snapshot(output)
    manifest = json.loads((output / "source_manifest.json").read_text())
    assert set(pool.POOL_SCAN_SOURCES) <= set(manifest)
    assert "cache/allocator/csrc/ignored.json" not in manifest
    provenance.verify_source_snapshot(output)


def test_fixed_audit_binds_new_provider_and_keeps_older_shape():
    from experiments.nosa_motivation.src.provenance import audit_native_identity, manifest_digest
    from experiments.nosa_motivation.tests.test_allocator_snapshot_provenance import metadata

    value = metadata()
    audit_native_identity(value)
    with pytest.raises(ValueError, match="requires"):
        audit_native_identity(value, require_pool_referrers=True)
    build, record = native_fixture()
    for side in ("build_before", "build_after"):
        outer = value["native_provenance"][side]
        outer["pool_referrers_build_info"] = deepcopy(build)
        del outer["sha256"]
        outer["sha256"] = manifest_digest(outer)
    case = value["cases"][0]
    case["pool_referrers"] = record
    for target in (
        case["native_artifacts_before"],
        case["native_artifacts_after"],
        value["native_provenance"]["artifacts_final"],
    ):
        target.update(record["after"]["artifacts"])
    audit_native_identity(value, require_pool_referrers=True, expected_abi_sha256="a" * 64)
    record["after"]["runtime"]["loaded_binary_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="mapped build"):
        audit_native_identity(value, require_pool_referrers=True)


def test_fixed_source_inventory_whitelists_exact_abi_file(tmp_path, monkeypatch):
    from experiments.nosa_motivation.src import provenance

    for name in (
        *pool.POOL_SCAN_SOURCES,
        "evaluation/pool_scan_provenance.py",
        "cache/allocator/csrc/ignored.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    paths = {str(path.relative_to(tmp_path)) for path in provenance.source_paths()}
    assert set(pool.POOL_SCAN_SOURCES) <= paths
    assert "cache/allocator/csrc/ignored.json" not in paths
    assert "evaluation/pool_scan_provenance.py" in paths
