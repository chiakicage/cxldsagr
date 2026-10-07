"""Matrix integrity tests use synthetic CPU records, never performance evidence."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.deepseek_v32_motivation.src import report_matrix as matrix


@pytest.fixture
def saved_matrix(tmp_path, monkeypatch):
    records, directories = {}, []
    cpu = {
        "affinity": [48, 49],
        "allowed": {"Cpus_allowed_list": "48-49", "Mems_allowed_list": "0-1"},
        "torch_num_threads": 2,
        "torch_num_interop_threads": 1,
        "numa_policy": {"available": True, "mode": "bind", "nodes": [1]},
        "environment": {"OMP_NUM_THREADS": "2"},
    }
    hardware = {
        "hostname": "fixture",
        "python": "fixture",
        "torch": "fixture",
        "cuda": "fixture",
        "device": "cuda:0",
        "name": "fixture",
        "uuid": "fixture-gpu",
        "total_memory": 1000,
        "sm_count": 132,
        "cpu_environment": cpu,
    }
    for history in matrix.HISTORIES:
        for candidates in matrix.CANDIDATES:
            directory = tmp_path / f"h{history}_a{candidates}"
            check = tmp_path / f"check_h{history}_a{candidates}"
            directory.mkdir()
            check.mkdir()
            directories.append(directory)
            config = {
                **matrix.FIXED_CONFIG,
                "history_tokens": history,
                "candidate_tokens": candidates,
                "padded_history_tokens": history,
                "schemes": list(matrix.report.SCHEMES),
                "warmup_policy": "capacity-dependent-fixture",
                "indexer_dispatch_policy": "fixture",
            }
            rows, ledger, cases = [], [], []
            for scale, scheme in enumerate(matrix.report.SCHEMES, 1):
                bank = {
                    "enabled": True,
                    "allocated": True,
                    "policy_revision": matrix.GRAPH_POLICY,
                    "eager_fallbacks": 0,
                    "graphs": [
                        {"layer": layer, "queries": query}
                        for layer in range(10)
                        for query in sorted({1024, candidates})
                    ],
                    "static_storage_bytes": 8,
                    "static_allocated_bytes": 10,
                    "private_reserved_bytes": 20,
                }
                cases.append(
                    {
                        "scheme": scheme,
                        "backend": {"compute_graphs": bank},
                        "torch_peak_allocated_bytes": 100,
                        "torch_peak_reserved_bytes": 200,
                    }
                )
                for index in range(32):
                    memories = {}
                    for side in ("before", "after"):
                        sample = {
                            "stage": f"{scheme}/{side}_request_{index}",
                            "torch_allocated_bytes": 90,
                            "torch_reserved_bytes": 190,
                            "torch_peak_allocated_bytes": 100,
                            "torch_peak_reserved_bytes": 200,
                            "cuda_free_bytes": 700,
                            "cuda_total_bytes": 1000,
                        }
                        memories[f"memory_{side}"] = sample
                        ledger.append(sample)
                    hit = index >= 16 and (scheme != "hbm" or history == 4096)
                    rows.append(
                        {
                            "scheme": scheme,
                            "request_id": index,
                            "input_sha256": f"input-{index}",
                            "is_revisit": index >= 16,
                            "prefix_cache_hit": hit,
                            "prefix_hit_tier": "hbm" if hit else "miss",
                            "evicted_users": [],
                            "cached_users": min(index + 1, 16, 65536 // history)
                            if scheme == "hbm"
                            else min(index + 1, 16),
                            "latency_ms": scale * (history / 4096 + candidates / 128 + index + 1),
                            "prefix_ms": 1.0,
                            "extend_ms": 2.0,
                            "cache_hbm_bytes": 50,
                            "cache_dram_bytes": 60,
                            "reserved_hbm_bytes": 70,
                            "reserved_dram_bytes": 80,
                            "cache_diagnostics": {
                                "host_to_device_bytes": 0 if scheme == "hbm" else 1000,
                                "device_to_host_bytes": 0,
                                "selection_records": 1 if index % 2 == 0 else 3,
                                "resident_selection_records": 1,
                            },
                            **memories,
                        }
                    )
            receipt = check / "receipt.json"
            receipt.write_text(json.dumps({"artifacts": {"metadata": {"path": "metadata.json"}}}))
            metadata = {
                "schema": matrix.BENCH_SCHEMA,
                "status": "accepted",
                "run_id": directory.name,
                "config": config,
                "cases": cases,
                "hardware": deepcopy(hardware),
                "cpu_environment_final": deepcopy(cpu),
                "cpu_environment_audit": {"status": "recorded_and_equal"},
                "source_sha256": "fixture-source",
                "checkpoint": {"fixture": True},
                "precision_settings": {"tf32": False},
                "backend_provenance": {"fixture": True},
                "execution_environment": {},
                "model_dimensions": {"hidden": 7168, "vocabulary": 129280},
                "model_boundary": "fixture",
                "measurement_boundary": "fixture",
                "diagnostics_boundary": "candidate only",
                "memory_boundary": "request samples",
                "validation_identity": {"base": {"sources": {"fixture.py": "fixture-sha"}}},
                "correctness_receipt": {
                    "path": str(receipt),
                    "sha256": matrix.report.digest(receipt),
                },
                "workload_sha256": f"fixture-input-h{history}-a{candidates}",
            }
            check_metadata = deepcopy(metadata)
            check_metadata.update(schema=matrix.CHECK_SCHEMA, run_id=check.name)
            for path, meta in ((directory, metadata), (check, check_metadata)):
                (path / "metadata.json").write_text(json.dumps(meta))
                (path / "memory.json").write_text(json.dumps(ledger))
            (directory / "workload").mkdir()
            (directory / "workload/workload.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "model": "deepseek_v32",
                            "num_users": 16,
                            "requests": 32,
                            "history_tokens": history,
                            "candidate_tokens": candidates,
                            "context_limit": history + candidates,
                            "seed": 42,
                            "sampling": "sequential",
                        },
                        "heat_sha256": None,
                        "tokenizer_sha256": "fixture-tokenizer",
                    }
                )
            )
            (directory / "source_manifest.json").write_text("{}")
            (directory / "measurements.jsonl").write_text(
                "fixture delegated to mocked point auditor\n"
            )
            records[directory] = metadata, rows, {"status": "passed"}
            records[check] = check_metadata, deepcopy(rows), {"status": "passed"}
    monkeypatch.setattr(matrix.report, "audit_run", lambda path: records[Path(path).resolve()])
    return directories, records


def test_complete_matrix_preserves_statistics_and_capacity_regimes(saved_matrix):
    directories, _ = saved_matrix
    result = matrix.collect_matrix(reversed(directories))
    assert len(result["points"]) == 12 and len(result["summary"]) == 96
    rows = {
        (r["history_tokens"], r["candidate_tokens"], r["scheme"], r["visit_kind"]): r
        for r in result["summary"]
    }
    row = rows[4096, 128, "serial_sparse", "revisit"]
    assert row["latency_mean_ms"] == 79.5
    assert row["latency_p95_ms"] == 99.75
    assert row["trace_latency_sum_ms"] == 1776
    assert row["candidate_consumer_union_resident_ratio"] == 0.5
    assert row["candidate_host_to_device_bytes"] == 16000
    assert row["torch_peak_allocated_bytes"] == 100
    assert row["torch_peak_reserved_bytes"] == 200
    assert row["maximum_after_request_device_used_bytes"] == 300
    assert [rows[h, 1024, "hbm", "revisit"]["prefix_hits"] for h in matrix.HISTORIES] == [16, 0, 0]


@pytest.mark.parametrize("defect", ["missing", "duplicate_directory", "duplicate_point"])
def test_rejects_incomplete_or_duplicated_matrix(saved_matrix, defect):
    directories, records = saved_matrix
    if defect == "missing":
        directories = directories[:-1]
    elif defect == "duplicate_directory":
        directories[-1] = directories[0]
    else:
        metadata = records[directories[1]][0]
        metadata["config"]["candidate_tokens"] = 128
        (directories[1] / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="12 unique|duplicate H/A"):
        matrix.collect_matrix(directories)


@pytest.mark.parametrize(
    "field,value", [("sparse_pool_tokens", 131072), ("seed", 43), ("chunk_size", 512)]
)
def test_rejects_mixed_fixed_configuration(saved_matrix, field, value):
    directories, records = saved_matrix
    metadata = records[directories[-1]][0]
    metadata["config"][field] = value
    (directories[-1] / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=f"mixed matrix config: {field}"):
        matrix.collect_matrix(directories)


@pytest.mark.parametrize("defect", ["gpu", "source", "precision", "input_rules", "affinity"])
def test_rejects_execution_or_input_policy_mixing(saved_matrix, defect):
    directories, records = saved_matrix
    metadata = records[directories[-1]][0]
    if defect == "gpu":
        metadata["hardware"]["uuid"] = "other-device"
    elif defect == "source":
        metadata["source_sha256"] = "other-source"
    elif defect == "precision":
        metadata["precision_settings"]["tf32"] = True
    elif defect == "affinity":
        metadata["hardware"]["cpu_environment"]["affinity"] = [50, 51]
        metadata["cpu_environment_final"]["affinity"] = [50, 51]
    else:
        path = directories[-1] / "workload/workload.json"
        workload = json.loads(path.read_text())
        workload["tokenizer_sha256"] = "different-tokenizer"
        path.write_text(json.dumps(workload))
    with pytest.raises(
        ValueError, match="mixed implementation, hardware, precision or input rules"
    ):
        matrix.collect_matrix(directories)


def test_ignores_process_identity_but_preserves_original_hardware(saved_matrix):
    directories, records = saved_matrix
    for index, directory in enumerate(directories):
        records[directory][0]["hardware"]["pid"] = index
        records[directory][0]["hardware"]["cpu_environment"]["pid"] = index
        records[directory][0]["cpu_environment_final"]["pid"] = index
    result = matrix.collect_matrix(directories)
    assert [point["hardware"]["pid"] for point in result["points"]] == list(range(12))
    assert "pid" not in result["common_identity"]["hardware"]
    assert "pid" not in result["common_identity"]["hardware"]["cpu_environment"]


@pytest.mark.parametrize("defect", ["cache_count", "memory_ledger", "request_count", "graph_shape"])
def test_rejects_inconsistent_point_evidence(saved_matrix, defect):
    directories, records = saved_matrix
    directory = directories[3]  # A=1024 shares the history graph shape.
    metadata, rows, _ = records[directory]
    if defect == "cache_count":
        rows[32]["cache_diagnostics"]["host_to_device_bytes"] += 1
    elif defect == "memory_ledger":
        path = directory / "memory.json"
        ledger = json.loads(path.read_text())
        ledger[1]["torch_allocated_bytes"] -= 1
        path.write_text(json.dumps(ledger))
    elif defect == "request_count":
        rows.pop()
    else:
        bank = metadata["cases"][0]["backend"]["compute_graphs"]
        bank["graphs"].append({"layer": 0, "queries": 128})
    with pytest.raises(
        ValueError,
        match="cache_diagnostics differs|memory differs|incomplete request|graph layer/shape",
    ):
        matrix.collect_matrix(directories)


def test_failed_render_does_not_publish_partial_matrix(saved_matrix, tmp_path, monkeypatch):
    directories, _ = saved_matrix
    monkeypatch.setattr(
        matrix.report, "write_report", lambda source, output: output.mkdir(parents=True)
    )

    def fail_render(*args):
        raise RuntimeError("render failed")

    monkeypatch.setattr(matrix, "plot_matrix", fail_render)
    output = tmp_path / "published"
    with pytest.raises(RuntimeError, match="render failed"):
        matrix.write_matrix(directories, output)
    assert not output.exists()


def test_writes_point_reports_and_one_audited_matrix(saved_matrix, tmp_path, monkeypatch):
    directories, _ = saved_matrix
    written = []

    def point_report(source, output):
        written.append(Path(source).name)
        output.mkdir(parents=True)

    monkeypatch.setattr(matrix.report, "write_report", point_report)
    monkeypatch.setattr(matrix, "plot_matrix", lambda summary, output: None)
    output = tmp_path / "published"
    result = matrix.write_matrix(directories, output)
    assert len(written) == 12
    assert len((output / "summary.csv").read_text().splitlines()) == 97
    assert json.loads((output / "summary.json").read_text())["summary"] == result["summary"]
    assert all((output / point["report"]).is_dir() for point in result["points"])
    with pytest.raises(FileExistsError):
        matrix.write_matrix(directories, output)


def counter_records():
    return {
        "layers": [
            {
                "prefetched_records": 8192,
                "prefetch_capacity_failures": 0,
                "device_slots": 65536,
                "candidate_slots": 256,
                "pool_scope": "shared_per_layer",
                "host_to_device_bytes": 1234,
                "selection_records": 9201,
                "resident_selection_records": 8448,
            }
            for _ in range(10)
        ],
        "layer_diagnostics": [None] * 10,
        "host_to_device_bytes": 12340,
    }


def fixed_cap_fixture():
    return {"derived_prefetch_cap": 8192, "candidate_slots": 256}


def test_preserves_all_original_saturated_counter_differences():
    check = counter_records()
    bench = deepcopy(check)
    bench["layers"][5]["prefetch_capacity_failures"] = 4
    check["layers"][8]["prefetch_capacity_failures"] = 3
    originals = deepcopy((bench, check))
    differences = matrix.compare_cache_diagnostics(bench, check, "echo", 16, fixed_cap_fixture)
    assert differences == [
        {
            "scheme": "echo",
            "request_id": 16,
            "layer": layer,
            "field": f"cache_diagnostics.layers[{layer}].prefetch_capacity_failures",
            "bench": actual,
            "check": reference,
            "derived_prefetch_cap": 8192,
            "prefetched_records": 8192,
        }
        for layer, actual, reference in ((5, 4, 0), (8, 0, 3))
    ]
    assert (bench, check) == originals


@pytest.mark.parametrize("value", [-1, True, False, 0.0, 2.0, "0", None])
@pytest.mark.parametrize("both", [False, True])
def test_rejects_invalid_counter_types_even_when_equal(value, both):
    check = counter_records()
    bench = deepcopy(check)
    bench["layers"][0]["prefetch_capacity_failures"] = value
    if both:
        check["layers"][0]["prefetch_capacity_failures"] = value
    with pytest.raises(ValueError, match="nonnegative integer"):
        matrix.compare_cache_diagnostics(bench, check, "echo", 16, fixed_cap_fixture)


@pytest.mark.parametrize("scheme", ["hbm", "dense", "serial_sparse"])
def test_rejects_counter_differences_outside_echo(scheme):
    check = counter_records()
    bench = deepcopy(check)
    bench["layers"][0]["prefetch_capacity_failures"] = 1
    with pytest.raises(ValueError, match="cache_diagnostics differs"):
        matrix.compare_cache_diagnostics(bench, check, scheme, 16, fixed_cap_fixture)


@pytest.mark.parametrize(
    "defect",
    [
        "unsaturated",
        "float_success",
        "wrong_slots",
        "wrong_candidates",
        "wrong_scope",
        "bytes",
        "selection",
        "resident",
        "nested_counter",
        "extra_key",
        "missing_key",
        "extra_layer",
        "integer_type",
        "bool_type",
    ],
)
def test_rejects_every_other_diagnostic_difference(defect):
    check = counter_records()
    bench = deepcopy(check)
    bench["layers"][5]["prefetch_capacity_failures"] = 4
    layer = bench["layers"][5]
    if defect == "unsaturated":
        layer["prefetched_records"] = 8191
    elif defect == "float_success":
        layer["prefetched_records"] = 8192.0
    elif defect == "wrong_slots":
        layer["device_slots"] = 65535
    elif defect == "wrong_candidates":
        layer["candidate_slots"] = 128
    elif defect == "wrong_scope":
        layer["pool_scope"] = "other"
    elif defect == "bytes":
        layer["host_to_device_bytes"] += 1
    elif defect == "selection":
        layer["selection_records"] += 1
    elif defect == "resident":
        layer["resident_selection_records"] += 1
    elif defect == "nested_counter":
        bench["layer_diagnostics"][0] = {"prefetch_capacity_failures": 1}
        check["layer_diagnostics"][0] = {"prefetch_capacity_failures": 0}
    elif defect == "extra_key":
        layer["unreviewed_counter"] = 1
    elif defect == "missing_key":
        del layer["selection_records"]
    elif defect == "extra_layer":
        bench["layers"].append(deepcopy(layer))
    elif defect == "integer_type":
        bench["host_to_device_bytes"] = float(check["host_to_device_bytes"])
    else:
        bench["other"] = True
        check["other"] = 1
    with pytest.raises(ValueError, match="cache_diagnostics differs"):
        matrix.compare_cache_diagnostics(bench, check, "echo", 16, fixed_cap_fixture)


def test_counter_lookalike_outside_layer_metrics_is_exact():
    bench, check = {"prefetch_capacity_failures": 1}, {"prefetch_capacity_failures": 0}
    with pytest.raises(ValueError, match="cache_diagnostics differs"):
        matrix.compare_cache_diagnostics(bench, check, "echo", 16, fixed_cap_fixture)


@pytest.fixture
def proof_records(tmp_path, monkeypatch):
    source_name = "reviewed.py"
    source = tmp_path / source_name
    source.write_text("reviewed fixture source\n")
    sources = {source_name: matrix.report.digest(source)}
    monkeypatch.setattr(matrix, "COUNTER_SOURCE_SHA256", sources)
    native = {"/fixture/cxldsagr_echo_indexer_fixture.so": {"size": 1, "sha256": "a" * 64}}
    backend = {
        "scheme": "echo",
        "physical_layers": 10,
        "sparse_pool_tokens": 65536,
        "pool_scope": "backend_per_layer",
        "cache_policy_revision": "echo-global-pages-fifo-v1",
        "echo_flags": deepcopy(matrix.ECHO_FLAGS),
        "resource_plan": {
            "resource_mode": "fixed_pools",
            "pool_scope": "backend_per_layer",
            "cache_policy_revision": "echo-global-pages-fifo-v1",
            "shared_token_pool": True,
            "sparse_pool_tokens": 65536,
            "candidate_persistence": "gpu_transient",
            "candidate_slots": 256,
        },
    }
    metadata = {
        "config": {"sparse_pool_tokens": 65536, "layers": 10, "candidate_tokens": 256},
        "validation_identity": {
            "base": {"sources": deepcopy(sources)},
            "methods": {
                "echo": {
                    "backend_type": "models.deepseek_v32.execution.adapter.DeepSeekServingBackend",
                    "backend": deepcopy(backend),
                    "native_artifacts": deepcopy(native),
                }
            },
        },
        "cases": [
            {
                "scheme": "echo",
                "backend": deepcopy(backend),
                "resource_plan": deepcopy(backend["resource_plan"]),
                "native_artifacts_before": deepcopy(native),
                "native_artifacts_after": deepcopy(native),
            }
        ],
    }
    directories = [tmp_path / name for name in ("bench", "check")]
    for directory in directories:
        (directory / "source").mkdir(parents=True)
        (directory / "source" / source_name).write_bytes(source.read_bytes())
        (directory / "source_manifest.json").write_text(json.dumps(sources))
    return directories[0], metadata, directories[1], deepcopy(metadata)


def test_counter_proof_binds_source_and_saved_native_identity(proof_records):
    proof = matrix.echo_counter_proof(*proof_records)
    assert proof["derived_prefetch_cap"] == 8192
    assert proof["status"] == "source_reviewed"
    assert proof["source_sha256"] == matrix.COUNTER_SOURCE_SHA256
    assert len(proof["echo_indexer_native_artifacts"]) == 1
    assert "do not prove identical logical prefetch sets" in proof["boundary"]


@pytest.mark.parametrize(
    "defect",
    [
        "flags",
        "flag_type",
        "missing_flags",
        "persistence",
        "pool",
        "source_hash",
        "source_bytes",
        "missing_source",
        "execution_hash",
        "native_missing",
        "native_changed",
        "native_after",
        "policy",
        "method_changed",
    ],
)
def test_counter_proof_rejects_unproven_identity(proof_records, defect):
    directory, metadata, check_dir, check = proof_records
    method = check["validation_identity"]["methods"]["echo"]
    if defect == "flags":
        method["backend"]["echo_flags"]["early_evict"] = True
    elif defect == "flag_type":
        method["backend"]["echo_flags"]["fused_logits_recall_extend"] = 1
    elif defect == "missing_flags":
        del method["backend"]["echo_flags"]
    elif defect == "persistence":
        method["backend"]["resource_plan"]["candidate_persistence"] = "persistent"
    elif defect == "pool":
        check["config"]["sparse_pool_tokens"] = 16384
    elif defect == "source_hash":
        (check_dir / "source_manifest.json").write_text(json.dumps({"reviewed.py": "b" * 64}))
    elif defect == "source_bytes":
        (check_dir / "source/reviewed.py").write_text("changed source\n")
    elif defect == "missing_source":
        (check_dir / "source/reviewed.py").unlink()
    elif defect == "execution_hash":
        check["validation_identity"]["base"]["sources"]["reviewed.py"] = "b" * 64
    elif defect == "native_missing":
        method["native_artifacts"] = {}
    elif defect == "native_changed":
        next(iter(method["native_artifacts"].values()))["sha256"] = "b" * 64
    elif defect == "native_after":
        check["cases"][0]["native_artifacts_after"] = {}
    elif defect == "policy":
        method["backend"]["cache_policy_revision"] = "unreviewed-policy"
    else:
        method["new_identity_field"] = "changed"
    with pytest.raises((ValueError, FileNotFoundError)):
        matrix.echo_counter_proof(directory, metadata, check_dir, check)


def test_matrix_saves_complete_counter_differences_and_lazy_proof(saved_matrix, monkeypatch):
    directories, records = saved_matrix
    directory = directories[5]
    metadata, rows, _ = records[directory]
    check_dir = Path(metadata["correctness_receipt"]["path"]).parent
    check_rows = records[check_dir][1]
    index = next(
        index
        for index, row in enumerate(rows)
        if row["scheme"] == "echo" and row["request_id"] == 16
    )
    rows[index]["cache_diagnostics"] = counter_records()
    check_rows[index]["cache_diagnostics"] = counter_records()
    rows[index]["cache_diagnostics"]["layers"][5]["prefetch_capacity_failures"] = 4
    calls = []

    def prove(*args):
        calls.append(args)
        return fixed_cap_fixture()

    monkeypatch.setattr(matrix, "echo_counter_proof", prove)
    result = matrix.collect_matrix(directories)
    changed = result["points"][5]["cache_diagnostics_comparison"]
    assert changed["policy"] == matrix.COUNTER_COMPARISON_POLICY
    assert changed["proof"] == fixed_cap_fixture()
    assert len(changed["differences"]) == 1
    assert changed["differences"][0]["bench"] == 4
    assert changed["differences"][0]["check"] == 0
    assert len(calls) == 1
    assert all(
        point["cache_diagnostics_comparison"]["proof"] is None
        for index, point in enumerate(result["points"])
        if index != 5
    )
