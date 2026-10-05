"""CPU integrity/geometry tests; these do not validate Hopper API execution."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.nosa_motivation.src.attention_reference import (
    capture_attention_inputs,
    materialize_selected,
    restore_layout,
)
from experiments.nosa_motivation.src.attention_reference_audit import (
    audit_attention_timings,
    audit_capture,
    audit_history_reuse,
    descriptor,
    selection_geometry,
    summarize_attention_rows,
)
from experiments.nosa_motivation.src.profile_attention_reference import combined_comparison
from experiments.nosa_motivation.src.provenance import digest
from models.attention_contracts import BlockSelection
from models.nosa.attention import NosaFixedAttention


@pytest.mark.parametrize("helper_version", ["original", "changed", None])
@pytest.mark.parametrize(
    "helper",
    [
        "evaluation/pool_scan_provenance.py",
        "experiments/gr_serving/src/pool_scan_provenance.py",  # Published historical snapshot.
    ],
)
def test_attention_reference_binds_shared_pool_provenance_orchestration(
    tmp_path, monkeypatch, helper_version, helper
):
    from experiments.nosa_motivation.src import profile_attention_reference as api

    reference_dir, data, profile = (tmp_path / name for name in ("formal", "api", "profile"))
    for directory in (reference_dir, data, profile):
        directory.mkdir()
    original = {"models/nosa/model.py": "model", helper: "original"}
    current = dict(original)
    if helper_version is None:
        current.pop(helper)
    else:
        current[helper] = helper_version
    (reference_dir / "source_manifest.json").write_text(json.dumps(original))
    (data / "source_manifest.json").write_text(json.dumps(current))
    (profile / "metadata.json").write_text(json.dumps({"run_id": "profile"}))
    hardware = {
        name: "same"
        for name in (
            "uuid",
            "packages",
            "torch_cuda",
            "compute_capability",
            "total_memory",
            "sm_count",
        )
    }
    reference = {"run_id": "formal", "source_sha256": "source", "hardware": hardware}
    for name in ("config", "model_config", "checkpoint", "precision_settings", "workload_sha256"):
        reference[name] = {}
    (reference_dir / "metadata.json").write_text(json.dumps(reference))
    metadata = {
        **reference,
        "schema": "nosa-independent-attention-v1",
        "status": "diagnostic_valid",
        "reference_run_id": "formal",
        "reference_metadata_sha256": "digest",
        "reference_source_sha256": "source",
        "reference_audit": {},
        "matrix_profile_run_id": "profile",
        "matrix_profile_metadata_sha256": "digest",
        "matrix_profile_audit": {},
    }
    (data / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(api, "audit_run", lambda path: (reference, [], {}))
    monkeypatch.setattr(api, "audit_profile", lambda *args: {})
    monkeypatch.setattr(api, "digest", lambda path: "digest")
    monkeypatch.setattr(api, "validate_cpu_environment_record", lambda value: None)
    monkeypatch.setattr(api, "require_matching_cpu_environment", lambda *args: None)
    monkeypatch.setattr(api, "verify_source_snapshot", lambda path: "source")

    def reached_native_audit(*args, **kwargs):
        raise RuntimeError("source guard passed")

    monkeypatch.setattr(api, "audit_capture_native", reached_native_audit)
    if helper_version == "original":
        with pytest.raises(RuntimeError, match="source guard passed"):
            api.audit_reference(data, reference_dir, profile, tmp_path / "traces")
    else:
        with pytest.raises(ValueError, match="runtime/orchestration source differs"):
            api.audit_reference(data, reference_dir, profile, tmp_path / "traces")


def short_selection():
    ids = torch.full((2, 2, 64), -1, dtype=torch.int64)
    ids[:, :, 0] = 0
    ids[1, :, 5] = 1  # validity need not occupy a compact initial prefix
    return ids, ids >= 0


def test_materialized_selection_keeps_gqa_sharing_causality_and_padding_explicit():
    ids, valid = short_selection()
    q = torch.ones(2, 4, 2)
    keys = torch.arange(65 * 2 * 2).reshape(65, 2, 2).float()
    values, cis = keys + 10, torch.zeros(65, 2)
    query, key, value, _, allowed = materialize_selected(q, keys, values, ids, valid, cis, 63)
    assert query.shape == (4, 2, 2)
    assert key.shape == (4, 2, 128)
    assert value.shape == (4, 128, 2)
    assert allowed.sum(-1).tolist() == [64, 64, 65, 65]
    assert torch.equal(value[2, 64], values[64, 0])
    geometry = selection_geometry(ids, valid, 63, 2, 2, 4, 2, 65)
    assert geometry["useful_attention_flops"] == 4 * 2 * (64 + 65) * 4
    assert geometry["dense_executed_attention_flops"] == 4 * 2 * 2 * 4 * 128
    assert geometry["unique_selected_kv_tokens"] == 65 * 2
    assert geometry["materialized_kv_tokens"] == 2 * 2 * 128


def test_restore_original_gapped_q_layout():
    original = torch.arange(3 * 8 * 2).reshape(3, 8, 2)[:, :4]
    result = restore_layout(original.contiguous(), descriptor(original), "cpu")
    assert result.stride() == original.stride()
    assert torch.equal(result, original)


def create_capture(
    tmp_path, monkeypatch, *, mutate=False, mode="full_request", history_reference=None
):
    config = {"history_tokens": 64, "candidate_tokens": 1, "chunk_size": 64}
    model = {
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_attention_heads": 32,
        "head_dim": 128,
    }
    backend = SimpleNamespace(scheme="hbm", config=model, resources=object())
    records = [
        {
            "keys": torch.ones(65, 2, 128, dtype=torch.bfloat16) * (layer + 1),
            "values": torch.ones(65, 2, 128, dtype=torch.bfloat16),
            "cis_scores": torch.zeros(65, 2, dtype=torch.bfloat16),
        }
        for layer in range(2)
    ]
    cache = SimpleNamespace(_execution_resources=backend.resources)
    visible = [64]
    cache.layer_view = lambda layer: {
        name: tensor[: visible[0]] for name, tensor in records[layer].items()
    }
    monkeypatch.setattr(
        NosaFixedAttention, "__call__", lambda self, q, selection, cache, context: q * 2
    )
    with capture_attention_inputs(
        backend,
        tmp_path,
        config,
        0,
        {"reference_run_id": "accepted"},
        mode=mode,
        history_reference=history_reference,
    ) as capture:
        for start, queries in ((0, 64), (64, 1)):
            visible[0] = start + queries
            for layer in range(2):
                if mutate and start == 64:
                    records[layer]["keys"][0].add_(1)
                q = torch.ones(queries, 32, 128, dtype=torch.bfloat16)
                ids = torch.full((queries, 2, 64), -1, dtype=torch.int64)
                ids[:, :, 0] = 0
                if start:
                    ids[:, :, 1] = 1
                NosaFixedAttention()(
                    q,
                    BlockSelection(ids, 64, ids >= 0),
                    cache,
                    SimpleNamespace(layer_idx=layer, query_start=start),
                )
        capture.accept_output(torch.zeros(1, 8), torch.zeros(1, 8))
    return json.loads((tmp_path / "manifest.json").read_text())


def test_capture_covers_every_layer_and_proves_final_kv_prefix_reuse(tmp_path, monkeypatch):
    manifest = create_capture(tmp_path / "capture", monkeypatch)
    audit = audit_capture(tmp_path / "capture", manifest)
    assert audit["calls"] == 4
    assert audit["immutable_prefixes_verified"]
    assert len(list((tmp_path / "capture/layers").glob("*.pt"))) == 2
    candidate = create_capture(
        tmp_path / "candidate",
        monkeypatch,
        mode="candidate_only",
        history_reference=tmp_path / "capture",
    )
    assert len(candidate["calls"]) == 2
    assert len(candidate["reused_history_calls"]) == 2


def test_capture_rejects_history_mutation_before_reusing_final_snapshot(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="immutable slice"):
        create_capture(tmp_path / "capture", monkeypatch, mutate=True)


def test_capture_audit_rejects_missing_invocation_and_corrupted_payload(tmp_path, monkeypatch):
    manifest = create_capture(tmp_path / "capture", monkeypatch)
    missing = deepcopy(manifest)
    missing["calls"].pop()
    with pytest.raises(ValueError, match="every layer"):
        audit_capture(tmp_path / "capture", missing)
    payload = tmp_path / "capture" / manifest["calls"][0]["file"]
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="operand file changed"):
        audit_capture(tmp_path / "capture", manifest)


def timing_evidence(manifest):
    sample = {
        "samples_cuda_ms": [1.0] * 7,
        "samples_wall_ms": [2.0] * 7,
        "median_cuda_ms": 1.0,
        "median_wall_ms": 2.0,
    }
    rows = [
        {
            "layer": row["layer"],
            "query_start": row["query_start"],
            "queries": row["queries"],
            "geometry": row["geometry"],
            "fa3_output_exact": True,
            "fa3_output_sha256": row["output_sha256"],
            "raw_product_validation": {
                "passed": True,
                "batch_rows_checked": sorted({0, row["queries"], row["queries"] * 2 - 1}),
                "qk_atol": 2e-4,
                "qk_rtol": 2e-4,
                "pv_atol": 0.008,
                "pv_rtol": 0.008,
                "qk_max_abs": 0.0,
                "pv_max_abs": 0.0,
                "qk_max_tolerance_ratio": 0.0,
                "pv_max_tolerance_ratio": 0.0,
            },
            "raw_layouts": {
                name: descriptor(torch.empty(shape, dtype=dtype))
                for name, shape, dtype in (
                    ("q", (row["queries"] * 2, 16, 128), torch.bfloat16),
                    (
                        "k",
                        (
                            row["queries"] * 2,
                            128,
                            row["geometry"]["selected_slots_per_query_head_group"],
                        ),
                        torch.bfloat16,
                    ),
                    (
                        "v",
                        (
                            row["queries"] * 2,
                            row["geometry"]["selected_slots_per_query_head_group"],
                            128,
                        ),
                        torch.bfloat16,
                    ),
                    (
                        "qk",
                        (
                            row["queries"] * 2,
                            16,
                            row["geometry"]["selected_slots_per_query_head_group"],
                        ),
                        torch.float32,
                    ),
                    (
                        "p",
                        (
                            row["queries"] * 2,
                            16,
                            row["geometry"]["selected_slots_per_query_head_group"],
                        ),
                        torch.bfloat16,
                    ),
                    ("pv", (row["queries"] * 2, 16, 128), torch.bfloat16),
                )
            },
            **{name: deepcopy(sample) for name in ("qk_bmm", "pv_bmm", "fa3_eager")},
        }
        for row in manifest["calls"]
    ]
    result = {
        "schema": "nosa-attention-reference-v1",
        "capture_sha256": manifest["manifest_sha256"],
        "warmup": 3,
        "repeats": 7,
        "rows": rows,
    }
    result["summary"] = summarize_attention_rows(manifest, rows)
    return result


def test_raw_and_fa3_timings_keep_distinct_denominators_and_reject_sample_tampering(
    tmp_path, monkeypatch
):
    manifest = create_capture(tmp_path / "capture", monkeypatch)
    manifest["manifest_sha256"] = digest(tmp_path / "capture/manifest.json")
    timings = timing_evidence(manifest)
    summary = audit_attention_timings(manifest, timings)["captured_workload"]
    assert summary["raw_qk_pv_cuda_ms"] == 8
    assert summary["independent_fa3_cuda_ms"] == 4
    assert summary["independent_fa3_wall_ms"] == 8
    timings["rows"][0]["qk_bmm"]["samples_cuda_ms"][0] = float("nan")
    with pytest.raises(ValueError, match="samples"):
        audit_attention_timings(manifest, timings)


def test_complete_coverage_does_not_turn_slow_materialization_into_automatic_acceptance():
    row = {
        "request_id": 0,
        "latency_ms": 10.0,
        "effective_work": {
            "request_flops": {"linear": 800, "attention": 200},
            "peak_bf16_tflops": 100,
        },
    }
    matrix = {"useful_gemm_bmm_flops": 800, "equivalent_gemm_bmm_ms": 5.0}
    attention = {
        "useful_attention_flops": 200,
        "raw_qk_pv_cuda_ms": 10.0,
        "independent_fa3_cuda_ms": 1.0,
    }
    result = combined_comparison(row, matrix, attention)
    assert result["raw_useful_flop_coverage"] == 1.0
    assert result["raw_materialized_reference"]["wall_to_reference_mfu_ratio"] == 1.5
    assert result["independent_fa3_composition"]["wall_to_reference_mfu_ratio"] == 0.6
    assert result["lowest_tested_reference_sum_ms"] == 6.0
    assert result["efficiency_gate"].startswith("requires interpretation")


def test_capture_rejects_self_declared_wrong_tensor_dtype_and_illegal_stride(tmp_path, monkeypatch):
    directory = tmp_path / "capture"
    manifest = create_capture(directory, monkeypatch)
    bad_dtype = deepcopy(manifest)
    bad_dtype["calls"][0]["layouts"]["q"]["dtype"] = "torch.float32"
    with pytest.raises(ValueError, match="tensor contract"):
        audit_capture(directory, bad_dtype)
    bad_stride = deepcopy(manifest)
    bad_stride["calls"][0]["layouts"]["q"]["stride"][0] = 1
    with pytest.raises(ValueError, match="overlap|aligned"):
        audit_capture(directory, bad_stride)


def test_reopen_history_reuse_rejects_changed_call_proof_and_prefix(tmp_path, monkeypatch):
    first_dir, revisit_dir = tmp_path / "first", tmp_path / "revisit"
    first = create_capture(first_dir, monkeypatch)
    revisit = create_capture(
        revisit_dir, monkeypatch, mode="candidate_only", history_reference=first_dir
    )
    assert audit_history_reuse(first_dir, first, revisit_dir, revisit)["history_calls_reused"] == 2
    altered = deepcopy(revisit)
    altered["reused_history_calls"][0]["operand_hashes"]["q"] = "0" * 64
    with pytest.raises(ValueError, match="operand proof"):
        audit_history_reuse(first_dir, first, revisit_dir, altered)
    entry = revisit["layer_records"]["0"]
    path = revisit_dir / entry["file"]
    payload = torch.load(path, weights_only=True)
    payload["cis"][0, 0] += 1
    torch.save(payload, path)
    entry["sha256"] = digest(path)
    with pytest.raises(ValueError, match="final KV/CIS"):
        audit_history_reuse(first_dir, first, revisit_dir, revisit)


@pytest.mark.parametrize("cached", [False, True])
def test_reference_reopen_selects_hit_or_miss_work_and_rejects_changed_arithmetic(
    tmp_path, monkeypatch, cached
):
    from experiments.nosa_motivation.src import profile_attention_reference as module
    from experiments.nosa_motivation.src.attention_reference_audit import tensor_digest
    from experiments.nosa_motivation.src.provenance import write_json

    data, reference_dir, profile_dir = (tmp_path / name for name in ("data", "accepted", "profile"))
    for path in (data, reference_dir / "numerical/hbm", reference_dir / "workload", profile_dir):
        path.mkdir(parents=True)
    first_dir, revisit_dir = data / "capture_0", data / "capture_2"
    first = create_capture(first_dir, monkeypatch)
    revisit = create_capture(
        revisit_dir,
        monkeypatch,
        mode="candidate_only",
        history_reference=None if cached else first_dir,
    )
    config = {**first["config"], "num_users": 2}
    hardware = dict.fromkeys(
        ("uuid", "packages", "torch_cuda", "compute_capability", "total_memory", "sm_count"),
        "fixed",
    )
    reference = {
        "run_id": "accepted",
        "config": config,
        "model_config": first["model_config"],
        "checkpoint": {"identity": "checkpoint"},
        "precision_settings": {"tf32": False},
        "hardware": hardware,
        "source_sha256": "reference-source",
        "workload_sha256": "workload",
    }
    write_json(reference_dir / "metadata.json", reference)
    write_json(reference_dir / "source_manifest.json", {})
    write_json(data / "source_manifest.json", {})
    output = torch.zeros(1, 8)
    output_hash = tensor_digest(output)
    workload = [{"request_id": index, "input_sha256": f"input-{index}"} for index in range(3)]
    (reference_dir / "workload/requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in workload)
    )
    for index in range(3):
        torch.save({"hidden": output}, reference_dir / "numerical/hbm" / f"{index:06d}.pt")
    capture_entries, summaries = {}, {}
    for index, directory, manifest in ((0, first_dir, first), (2, revisit_dir, revisit)):
        manifest.update(config=config, request_id=index)
        manifest["identity"] = {
            "reference_run_id": "accepted",
            "reference_source_sha256": "reference-source",
            "runtime_source_sha256": "source",
            "native_build_sha256": "build",
            "device_uuid": "fixed",
            "workload_sha256": "workload",
            "input_sha256": f"input-{index}",
        }
        if index and not cached:
            manifest["history_reference_sha256"] = digest(first_dir / "manifest.json")
        write_json(directory / "manifest.json", manifest)
        manifest["manifest_sha256"] = digest(directory / "manifest.json")
        benchmark = timing_evidence(manifest)
        benchmark_path = data / f"reference_{index}.json"
        write_json(benchmark_path, benchmark)
        summaries[index] = benchmark["summary"]
        capture_entries[str(index)] = {
            "directory": directory.name,
            "manifest_sha256": manifest["manifest_sha256"],
            "benchmark_file": benchmark_path.name,
            "benchmark_sha256": digest(benchmark_path),
        }
    matrix = {
        "full_request": {"useful_gemm_bmm_flops": 800, "equivalent_gemm_bmm_ms": 5.0},
        "candidate_only": {"useful_gemm_bmm_flops": 100, "equivalent_gemm_bmm_ms": 1.0},
    }
    rows, comparisons = [], []
    for index in range(3):
        hit = index == 2 and cached
        attention = (
            summaries[0]["captured_workload"]
            if index != 2
            else (
                summaries[2]["captured_workload"]
                if hit
                else module.combine_reused_history(summaries[0], summaries[2])
            )
        )
        selected = matrix["candidate_only" if hit else "full_request"]
        row = {
            "request_id": index,
            "method": "hbm",
            "latency_ms": 100.0,
            "prefix_cache_hit": hit,
            "effective_work": {
                "request_flops": {
                    "matrix": selected["useful_gemm_bmm_flops"],
                    "attention": attention["useful_attention_flops"],
                },
                "peak_bf16_tflops": 100,
            },
        }
        rows.append(row)
        if index in (0, 2):
            comparisons.append(module.combined_comparison(row, selected, attention))
    raw_path = data / "matrix_reference.json"
    write_json(raw_path, {"raw": "fixture"})
    profile = {"run_id": "profile", "matrix_reference": {"sha256": digest(raw_path)}}
    write_json(profile_dir / "metadata.json", profile)
    metadata = {
        **reference,
        "run_id": "attention",
        "schema": "nosa-independent-attention-v1",
        "status": "validating",
        "source_sha256": "source",
        "reference_run_id": "accepted",
        "reference_source_sha256": "reference-source",
        "reference_metadata_sha256": digest(reference_dir / "metadata.json"),
        "reference_audit": {"passed": True},
        "matrix_profile_run_id": "profile",
        "matrix_profile_metadata_sha256": digest(profile_dir / "metadata.json"),
        "matrix_profile_audit": {"passed": True},
        "benchmark_native_provenance": {},
        "native_provenance": {},
        "benchmark_native_audit": {"passed": True},
        "matrix_reference_sha256": digest(raw_path),
        "capture_warmup": [],
        "captures": capture_entries,
        "comparisons": comparisons,
        "replay_requests": [
            {
                "request_id": row["request_id"],
                "metrics": {"prefix_cache_hit": row["prefix_cache_hit"]},
                "output_sha256": output_hash,
            }
            for row in rows
        ],
    }
    monkeypatch.setattr(module, "audit_run", lambda _: (reference, rows, {"passed": True}))
    monkeypatch.setattr(module, "audit_profile", lambda *args: {"passed": True})
    monkeypatch.setattr(module, "verify_source_snapshot", lambda _: "source")
    monkeypatch.setattr(module, "audit_capture_native", lambda *args, **kwargs: "build")
    monkeypatch.setattr(module, "audit_matrix_native_provenance", lambda *args: {"passed": True})
    monkeypatch.setattr(module, "audit_matrix_reference", lambda *args: matrix)
    monkeypatch.setattr(module, "validate_warmup", lambda *args: None)
    monkeypatch.setattr(module, "check_request", lambda *args: None)
    write_json(data / "metadata.json", metadata)
    result = module.audit_reference(data, reference_dir, profile_dir, profile_dir)
    assert result["passed"]
    assert (result["repeated_history"] is None) == cached
    metadata["comparisons"][1]["raw_materialized_reference"]["summed_api_ms"] += 1
    write_json(data / "metadata.json", metadata)
    with pytest.raises(ValueError, match="arithmetic"):
        module.audit_reference(data, reference_dir, profile_dir, profile_dir)
