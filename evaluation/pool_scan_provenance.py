"""CPU-only NOSA pool-provider receipts shared by fixed and GR experiments.

Build discovery is explicit and never prepares the provider. Runtime snapshots
only inspect retained loader state and the selected mapped file. Saved audits
read no current binary and require no model, native-extension or CUDA import.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

POOL_SCAN_ABI = "models/nosa/csrc/pool_referrers_abi.json"
POOL_SCAN_SOURCES = (
    "models/nosa/_pool_referrers.py",
    "models/nosa/csrc/pool_referrers.c",
    POOL_SCAN_ABI,
)
COUNTERS = ("filtered_calls", "unfiltered_calls", "fallback_calls", "audit_errors")
CASE_SCOPE = (
    "Whole recorded case, including its setup/cleanup where bracketed; route counts "
    "do not locate or time scans within individual requests."
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def requires_provenance(manifest):
    present = set(POOL_SCAN_SOURCES).intersection(manifest)
    if present and present != set(POOL_SCAN_SOURCES):
        raise ValueError("incomplete pool-referrer source coverage")
    return bool(present)


def build_info():
    from models.nosa._pool_referrers import build_info as discover

    return discover()


def verify_build_info(expected):
    current = build_info()
    if current != expected:
        raise ValueError("pool-referrer build identity changed during execution")
    return current


def _mapped_artifact(runtime, maps_path):
    if runtime is None or runtime["backend"] != "cpython_native":
        return {}
    path = Path(runtime["loaded_binary_path"])
    if not path.is_absolute() or str(path.resolve()) != str(path):
        raise ValueError("pool-referrer binary path is not canonical")
    before = path.stat()
    found = False
    for line in Path(maps_path).read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        name = fields[5]
        if name.removesuffix(" (deleted)") != str(path):
            continue
        if name.endswith(" (deleted)"):
            raise ValueError("mapped pool-referrer binary was deleted")
        device = tuple(int(part, 16) for part in fields[3].split(":"))
        if int(fields[4]) != before.st_ino or device != (
            os.major(before.st_dev),
            os.minor(before.st_dev),
        ):
            raise ValueError("mapped pool-referrer binary was replaced")
        found = True
    if not found:
        raise ValueError("selected pool-referrer binary is not mapped")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (
        any(
            getattr(before, key) != getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        )
        or digest != runtime["loaded_binary_sha256"]
    ):
        raise ValueError("pool-referrer binary changed during observation")
    return {str(path): {"size": before.st_size, "sha256": digest}}


def snapshot(*, maps_path=Path("/proc/self/maps")):
    from models.nosa._pool_referrers import runtime_info

    runtime = runtime_info()
    return {"runtime": runtime, "artifacts": _mapped_artifact(runtime, maps_path)}


def finish_case(before):
    after = snapshot()
    start = before["runtime"]
    end = after["runtime"]
    if end is None:
        raise ValueError("pool-referrer provider was not prepared in allocator-using case")
    return {
        "before": before,
        "after": after,
        "counter_delta": {
            name: end["counters"][name] - (0 if start is None else start["counters"][name])
            for name in COUNTERS
        },
        "scope": CASE_SCOPE,
    }


def audit_build(before, after):
    if before != after:
        raise ValueError("saved pool-referrer build identity changed")
    if before.get("available") is True:
        identity = before["identity"]
        if identity.get("schema") != "nosa-pool-referrers-build-v1" or (
            before["fingerprint"] != _digest(identity)
        ):
            raise ValueError("saved pool-referrer build fingerprint differs")
    elif (
        before.get("available") is not False
        or not before.get("reason")
        or not before.get("source_sha256")
    ):
        raise ValueError("missing pool-referrer build fallback evidence")


def _audit_runtime(snapshot, planned, *, allow_unprepared, expected_abi_sha256=None):
    runtime, artifacts = snapshot["runtime"], snapshot["artifacts"]
    if runtime is None:
        if not allow_unprepared or artifacts:
            raise ValueError("missing prepared pool-referrer provider")
        return
    if set(runtime["counters"]) != set(COUNTERS) or any(
        type(value) is not int or value < 0 for value in runtime["counters"].values()
    ):
        raise ValueError("invalid pool-referrer counters")
    if runtime["backend"] == "cpython_native":
        identity = planned.get("identity", {})
        path = Path(runtime["loaded_binary_path"])
        if (
            planned.get("available") is not True
            or runtime["fingerprint"] != planned["fingerprint"]
            or runtime["fallback_reason"] is not None
            or not runtime["abi_manifest_sha256"]
            or artifacts.get(runtime["loaded_binary_path"], {}).get("sha256")
            != runtime["loaded_binary_sha256"]
            or not runtime["loaded_binary_sha256"]
            or len(artifacts) != 1
            or not path.is_absolute()
            or path.parent.name != planned["fingerprint"]
            or path.name != identity.get("module_name", "") + identity.get("extension_suffix", "")
        ):
            raise ValueError("pool-referrer provider differs from planned/mapped build")
        planned_abi = identity.get("abi_manifest_sha256")
        if runtime["abi_manifest_sha256"] != planned_abi or (
            expected_abi_sha256 is not None and planned_abi != expected_abi_sha256
        ):
            raise ValueError("pool-referrer whitelist differs from planned source")
    elif runtime["backend"] == "python_original":
        if (
            not runtime["fallback_reason"]
            or runtime["loaded_binary_path"] is not None
            or runtime["loaded_binary_sha256"] is not None
            or artifacts
        ):
            raise ValueError("missing pool-referrer runtime fallback evidence")
        if runtime["fingerprint"] is not None and runtime["fingerprint"] != planned.get(
            "fingerprint"
        ):
            raise ValueError("pool-referrer fallback fingerprint differs")
        if expected_abi_sha256 is not None and runtime["abi_manifest_sha256"] not in (
            None,
            expected_abi_sha256,
        ):
            raise ValueError("pool-referrer fallback whitelist differs")
    else:
        raise ValueError("unknown pool-referrer provider")


def audit_case(record, planned, *, expected_abi_sha256=None, mapped_artifacts=None):
    for name in ("before", "after"):
        _audit_runtime(
            record[name],
            planned,
            allow_unprepared=name == "before",
            expected_abi_sha256=expected_abi_sha256,
        )
    before, after = record["before"]["runtime"], record["after"]["runtime"]
    if before is not None and {k: v for k, v in before.items() if k != "counters"} != {
        k: v for k, v in after.items() if k != "counters"
    }:
        raise ValueError("pool-referrer provider changed within case")
    expected = {
        name: after["counters"][name] - (0 if before is None else before["counters"][name])
        for name in COUNTERS
    }
    if any(value < 0 for value in expected.values()) or record["counter_delta"] != expected:
        raise ValueError("pool-referrer case counter delta differs")
    if record.get("scope") != CASE_SCOPE:
        raise ValueError("pool-referrer case counter scope differs")
    if mapped_artifacts is not None and any(
        mapped_artifacts.get(path) != value
        for side in ("before", "after")
        for path, value in record[side]["artifacts"].items()
    ):
        raise ValueError("pool-referrer binary differs from case mapped artifacts")


def audit_gr(metadata, *, required=False, expected_abi_sha256=None, profile=False):
    provenance = metadata.get("pool_referrers_provenance")
    if provenance is None:
        if (
            required
            or "pool_referrers_cases" in metadata
            or any("pool_referrers" in case for case in metadata.get("cases", []))
        ):
            raise ValueError("current NOSA source requires pool-referrer provenance")
        return None
    before = provenance["build_before"]
    audit_build(before, provenance["build_after"])
    if profile:
        records = metadata["pool_referrers_cases"]
        expected_keys = {(r["mode"], r["case"]) for r in metadata["samples"]}
        keys = [(r["mode"], r["case"]) for r in records]
        if len(keys) != len(expected_keys) or set(keys) != expected_keys:
            raise ValueError("pool-referrer profile case coverage differs")
        cases = records
    else:
        cases = [case for case in metadata["cases"] if case["model"] == "nosa"]
        if not cases:
            raise ValueError("pool-referrer receipt has no NOSA allocator cases")
    for case in cases:
        audit_case(case["pool_referrers"], before, expected_abi_sha256=expected_abi_sha256)
    return before.get("fingerprint")


def audit_profile_reference(metadata, formal):
    """Bind diagnostic provider selection to its saved formal build identity."""
    current = metadata.get("pool_referrers_provenance")
    reference = None if formal is None else formal.get("pool_referrers_provenance")
    if current is None and reference is None:
        return
    if current is None or reference is None or current["build_before"] != reference["build_after"]:
        raise ValueError("pool-referrer profile build differs from saved formal measurement")
    expected = {
        json.dumps(
            {
                key: value
                for key, value in case["pool_referrers"]["after"]["runtime"].items()
                if key != "counters"
            },
            sort_keys=True,
        )
        for case in formal["cases"]
        if case["model"] == "nosa"
    }
    for case in metadata["pool_referrers_cases"]:
        runtime = case["pool_referrers"]["after"]["runtime"]
        if (
            json.dumps(
                {key: value for key, value in runtime.items() if key != "counters"}, sort_keys=True
            )
            not in expected
        ):
            raise ValueError(
                "pool-referrer profile provider/binary differs from saved formal measurement"
            )
