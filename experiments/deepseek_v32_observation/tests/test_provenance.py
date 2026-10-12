"""Live artifact inspection never compiles and ignores volatile runtime details."""

from functools import cache
from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_observation.src import provenance


def test_live_triton_inspection_reads_factory_cache_without_invoking_it():
    invoked = []

    @cache
    def factory():
        invoked.append(True)
        compiled = SimpleNamespace(
            hash="compiled-key",
            name="resident_mask",
            metadata={"num_warps": 4},
            asm={"ptx": "PTX", "cubin": b"CUBIN"},
        )
        return SimpleNamespace(device_caches={0: ({"key": compiled}, {}, {})})

    assert provenance._live_triton_artifacts(factory) == []
    assert invoked == []
    factory()
    actual = provenance._live_triton_artifacts(factory)
    assert len(actual) == 1 and invoked == [True]
    assert actual[0]["hash"] == "compiled-key"
    assert set(actual[0]["asm_sha256"]) == {"ptx", "cubin"}


def fixture_runtime():
    return {
        "backend_sha256": "backend",
        "allocated": 4,
        "pid": 23,
        "native_runtime": {
            "schema_version": 2,
            "native_jit": [
                {
                    "name": "topk",
                    "loaded_in_this_process": True,
                    "library": {"path": "/tmp/first.so", "sha256": "a" * 64},
                }
            ],
            "resident_masks_triton": {
                "resident_tail_mask": [
                    {
                        "hash": "compile-key",
                        "module": 123,
                        "asm_sha256": {"ptx": "b" * 64, "cubin": "c" * 64},
                    }
                ]
            },
        },
    }


def test_runtime_match_ignores_allocator_pid_addresses_and_paths():
    checked = fixture_runtime()
    measured = fixture_runtime()
    measured["allocated"], measured["pid"] = 999, 421
    measured["native_runtime"]["native_jit"][0]["library"]["path"] = "/different/second.so"
    measured["native_runtime"]["resident_masks_triton"]["resident_tail_mask"][0]["module"] = 456
    assert provenance.require_runtime_match(checked, measured)["passed"] is True


def test_runtime_match_detects_native_library_or_mask_binary_change():
    checked = fixture_runtime()
    measured = fixture_runtime()
    measured["native_runtime"]["resident_masks_triton"]["resident_tail_mask"][0]["asm_sha256"][
        "cubin"
    ] = "d" * 64
    with pytest.raises(RuntimeError, match="resident_masks_triton"):
        provenance.require_runtime_match(checked, measured)


def test_runtime_match_accepts_checked_superset_but_rejects_unchecked_launch_metadata():
    checked = fixture_runtime()
    measured = fixture_runtime()
    checked["native_runtime"]["resident_masks_triton"]["resident_tail_mask"].append(
        {
            "hash": "oracle-only-key",
            "metadata": {"num_warps": 8},
            "asm_sha256": {"ptx": "e" * 64, "cubin": "f" * 64},
        }
    )
    assert provenance.require_runtime_match(checked, measured)["passed"] is True
    measured["native_runtime"]["resident_masks_triton"]["resident_tail_mask"][0]["metadata"] = {
        "num_warps": 8
    }
    with pytest.raises(RuntimeError, match="specialization_sha256"):
        provenance.require_runtime_match(checked, measured)


def test_backend_refresh_rebuilds_cached_file_inventory(monkeypatch):
    values = []

    @cache
    def collect():
        values.append(True)
        return {"version": len(values)}

    monkeypatch.setattr(provenance, "_cached_installed_backends", collect)
    assert provenance.installed_backends() == {"version": 1}
    assert provenance.installed_backends() == {"version": 1}
    assert provenance.installed_backends(refresh=True) == {"version": 2}
