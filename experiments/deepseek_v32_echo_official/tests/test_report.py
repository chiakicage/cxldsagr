import hashlib
import json
from copy import deepcopy

import pytest
import torch

from experiments.deepseek_v32_echo_official.src.measure import (
    LABELS,
    NUMERICAL_POLICY,
    NUMERICAL_POLICY_BASIS,
    PRECISION_POLICY,
    SCHEMA,
    SCHEMES,
    VERIFICATION_STAGES,
    configuration,
    digest,
    numerical_comparison,
    parser,
    snapshot_official,
    write_json,
)
from experiments.deepseek_v32_echo_official.src.report import audit_run, write_report
from experiments.gr_serving.src.workload import token_sha256


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def saved_run(tmp_path):
    config = configuration(
        parser().parse_args(
            [
                "--run-id",
                "fixture",
                "--num-users",
                "2",
                "--history-tokens",
                "2048",
                "--candidate-tokens",
                "2",
                "--chunk-size",
                "32",
                "--sparse-pool-tokens",
                "2048",
                "--host-arena-tokens",
                "4096",
            ]
        )
    )
    # Keep a regression fixture for saved reports predating graph configuration.
    config.pop("enable_compute_graphs", None)
    source = tmp_path / "source/example.py"
    source.parent.mkdir()
    source.write_text("source fixture\n")
    source_manifest = {"example.py": digest(source)}
    write_json(tmp_path / "source_manifest.json", source_manifest)
    native = tmp_path / "extension.so"
    native.write_bytes(b"binary identity fixture")
    provenance = {
        "source_files": {str(source): digest(source)},
        "native_files": {str(native): digest(native)},
    }
    official_id = snapshot_official(provenance, tmp_path)
    requests = []
    for index in range(4):
        ids = [index % 2 + 1] * 2048 + [index + 10, index + 20]
        requests.append(
            {
                "request_id": index,
                "user_id": index % 2,
                "visit_index": index // 2,
                "is_revisit": index >= 2,
                "input_ids": ids,
                "input_sha256": token_sha256(ids),
            }
        )
    identity = {
        "config": config,
        "heat_sha256": None,
        "tokenizer_sha256": "fixture",
        "requests": [
            {key: value for key, value in row.items() if key != "input_ids"} for row in requests
        ],
    }
    workload_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    (tmp_path / "workload").mkdir()
    write_rows(tmp_path / "workload/requests.jsonl", requests)
    write_json(tmp_path / "workload/workload.json", {**identity, "workload_sha256": workload_id})
    memory, rows, cases, warmup = [], [], [], {}
    for scheme in SCHEMES:
        hbm = scheme == "hbm"
        plan = {
            "resource_mode": "fixed_pools",
            "sparse_pool_tokens": 2048,
            "host_arena_tokens": 4096,
            "max_history_tokens": 2048,
            "max_candidate_tokens": 2,
            "max_session_capacity": 2050,
            "workspace_query_tokens": 32,
            "candidate_persistence": "gpu_transient",
        }
        cases.append(
            {
                "scheme": scheme,
                "requests": 4,
                "warmup_requests": 3,
                "warmup_request_ids": [0, 1, 2],
                "started_empty": True,
                "resource_plan": plan,
                "admission_hbm_token_capacity": 2048 if hbm else 0,
                "admission_page_capacity": 0 if hbm else 64,
                "torch_peak_allocated_bytes": 100,
                "torch_peak_reserved_bytes": 200,
                "backend": {
                    "physical_layers": 10,
                    "source_layers": [i % 3 for i in range(10)],
                    "input_semantics": "copy_source_layer_hidden_and_residual_for_each_physical_copy",
                    "linear_backend": "fp8",
                    "chunk_size": 32,
                    "total_parameters": 7827793408,
                    "output": "all_candidate_normalized_hidden_and_last_token_lm_head",
                },
            }
        )
        warmup[scheme] = []
        for index, stage in zip(
            (0, 1, 2),
            ("first_user_first_visit", "second_user_first_visit", "first_user_revisit"),
            strict=True,
        ):
            miss = index == 2 and not hbm
            warmup[scheme].append(
                {
                    **{key: value for key, value in requests[index].items() if key != "input_ids"},
                    "stage": stage,
                    "prefix_cache_hit": miss,
                    "candidate_persistence": "gpu_transient",
                    "retained_length": 2048,
                    "host_to_device_bytes": 1152 if miss else 0,
                    "device_to_host_bytes": 0,
                    "prefetched_records": 0,
                    "recalled_records": int(miss),
                }
            )
        (tmp_path / "numerical" / scheme).mkdir(parents=True)
        for request in requests:
            index = request["request_id"]
            relative = f"numerical/{scheme}/{index:06d}.pt"
            torch.save(
                {
                    "request_id": index,
                    "input_sha256": request["input_sha256"],
                    "hidden": torch.full((2, 3), index, dtype=torch.bfloat16),
                    "logits": torch.full((1, 5), index, dtype=torch.float32),
                },
                tmp_path / relative,
            )
            row = {key: value for key, value in request.items() if key != "input_ids"}
            row.update(
                {
                    "scheme": scheme,
                    "run_id": "fixture",
                    "round_index": index // 2,
                    "prefix_cache_hit": index >= 2 and not hbm,
                    "prefix_hit_tier": "dram" if index >= 2 and not hbm else "miss",
                    "stable_prefix_tokens": 2048,
                    "candidate_suffix_tokens": 2,
                    "resource_mode": "fixed_pools",
                    "hbm_budget_bytes": None,
                    "dram_budget_bytes": None,
                    "cached_users": 1 if hbm else min(index + 1, 2),
                    "evicted_users": [],
                    "latency_ms": 10,
                    "admission_ms": 1,
                    "prefix_ms": 2,
                    "extend_ms": 5,
                    "cleanup_ms": 2,
                    "cache_hbm_bytes": 100,
                    "cache_dram_bytes": 0 if hbm else 200,
                    "shared_cache_hbm_bytes": 10,
                    "shared_cache_dram_bytes": 0 if hbm else 20,
                    "cache_diagnostics": {
                        "candidate_persistence": "gpu_transient",
                        "retained_length": 2048,
                        "host_to_device_bytes": 0 if hbm else 256,
                        "device_to_host_bytes": 0,
                        "layers": [
                            {
                                "device_slots": 2048,
                                "host_token_capacity": 4096,
                                "session_host_tokens": 2048,
                                "device_to_host_bytes": 0,
                            }
                            for _ in range(10)
                        ],
                    },
                    "diagnostics_scope": "candidate_only",
                    "workload_sha256": workload_id,
                    "output_file": relative,
                    "output_sha256": digest(tmp_path / relative),
                }
            )
            for boundary in ("before", "after"):
                sample = {
                    "stage": f"{scheme}/{boundary}_request_{index}",
                    "torch_allocated_bytes": 100,
                    "torch_reserved_bytes": 200,
                    "torch_peak_allocated_bytes": 100,
                    "torch_peak_reserved_bytes": 200,
                    "cuda_free_bytes": 800,
                    "cuda_total_bytes": 1000,
                }
                memory.append(sample)
                row[f"memory_{boundary}"] = sample
            rows.append(row)
    validation = []
    for row in rows:
        payload = torch.load(tmp_path / row["output_file"], weights_only=True)
        reference = torch.load(
            tmp_path / f"numerical/hbm/{row['request_id']:06d}.pt", weights_only=True
        )
        row["numerical"] = {
            name: numerical_comparison(payload[name], reference[name], name)
            for name in ("hidden", "logits")
        }
        if row["scheme"] == "hbm":
            relative = f"numerical/hbm_repeat/{row['request_id']:06d}.pt"
            (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, tmp_path / relative)
            validation.append(
                {
                    **{key: value for key, value in row.items() if not key.endswith("_ms")},
                    "validation_label": "hbm_repeat",
                    "output_file": relative,
                    "output_sha256": digest(tmp_path / relative),
                }
            )
    write_rows(tmp_path / "measurements.jsonl", rows)
    write_rows(tmp_path / "validation.jsonl", validation)
    write_json(tmp_path / "memory.json", memory)
    precision = {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "bf16_reduced_precision_reduction": True,
        "fp16_reduced_precision_reduction": True,
        "cudnn_allow_tf32": True,
    }
    write_json(
        tmp_path / "metadata.json",
        {
            "schema": SCHEMA,
            "run_id": "fixture",
            "status": "accepted",
            "config": config,
            "scheme_labels": LABELS,
            "cases": cases,
            "warmup_traces": warmup,
            "numerical_policy": NUMERICAL_POLICY,
            "numerical_policy_basis": NUMERICAL_POLICY_BASIS,
            "precision_policy": PRECISION_POLICY,
            "precision_settings": precision,
            "precision_settings_final": precision,
            "validation_repeats": [
                {
                    "label": "hbm_repeat",
                    "scheme": "hbm",
                    "requests": 4,
                    "started_empty": True,
                    "timing_published": False,
                    "resource_plan": cases[0]["resource_plan"],
                    "admission_hbm_token_capacity": 2048,
                }
            ],
            "workload_sha256": workload_id,
            "official_provenance": provenance,
            "official_artifact_manifest_sha256": official_id,
            "identity_verifications": [
                {"stage": stage, "status": "passed"} for stage in VERIFICATION_STAGES
            ],
            "source_sha256": hashlib.sha256(
                json.dumps(source_manifest, sort_keys=True).encode()
            ).hexdigest(),
            "model_dimensions": {"hidden": 3, "vocabulary": 5},
            "hardware": {"name": "CPU fixture", "torch": "fixture", "cuda": "fixture"},
        },
    )
    return tmp_path


@pytest.fixture
def graph_run(saved_run):
    metadata = json.loads((saved_run / "metadata.json").read_text())
    metadata["config"]["enable_compute_graphs"] = True
    plan = {
        "compute_graphs_enabled": True,
        "compute_graph_policy_revision": "deepseek-compute-islands-v2",
        "compute_graph_query_sizes": [2, 32],
        "compute_graph_count": 40,
        "compute_graph_static_storage_bytes": 64,
        "compute_graph_static_allocation_limit_bytes": 100,
        "compute_graph_private_limit_bytes": 128,
        "compute_graph_reserved_limit_bytes": 228,
        "compute_graph_private_limit_kind": "chosen_upper_limit_not_observed_fixed_overhead",
    }
    bank = {
        "enabled": True,
        "allocated": True,
        "policy_revision": "deepseek-compute-islands-v2",
        "static_storage_bytes": 64,
        "static_allocated_bytes": 80,
        "private_reserved_bytes": 100,
        "chosen_private_limit_bytes": 128,
        "memory_at_allocation": {
            "pytorch_allocated": 100,
            "pytorch_reserved": 200,
            "device_used": 300,
            "device_total": 1000,
        },
        "captured_precision_policy": {
            **metadata["precision_settings"],
            "autocast_enabled": False,
        },
        "eager_fallbacks": 0,
        "setup_seconds": 1.0,
        "graphs": [
            {
                "layer": layer,
                "queries": queries,
                "residual_present": layer % 3 != 0,
                "projection_replays": 0,
                "finish_replays": 0,
            }
            for layer in range(10)
            for queries in (2, 32)
        ],
    }
    for filename in ("measurements.jsonl", "validation.jsonl"):
        path = saved_run / filename
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        states = {}
        for row in rows:
            before = deepcopy(states.get(row["scheme"], bank))
            after = deepcopy(before)
            for entry in after["graphs"]:
                delta = 1 if entry["queries"] == 2 else 0 if row["prefix_cache_hit"] else 64
                entry["projection_replays"] += delta
                entry["finish_replays"] += delta
            row["compute_graphs"] = {"before": before, "after": after}
            states[row["scheme"]] = after
        cases = (
            metadata["cases"]
            if filename == "measurements.jsonl"
            else metadata["validation_repeats"]
        )
        for case in cases:
            case["resource_plan"].update(plan)
            case["shared_reservation"] = {"hbm": 300, "dram": 0}
            case.setdefault("backend", {})["compute_graphs"] = states[case["scheme"]]
        write_rows(path, rows)
    write_json(saved_run / "metadata.json", metadata)
    return saved_run


def test_legacy_report_keeps_absent_graph_configuration(saved_run):
    metadata, _, audit = audit_run(saved_run)
    assert "enable_compute_graphs" not in metadata["config"]
    assert "compute_graphs" not in audit


def test_graph_report_audits_measured_and_independent_repeat(graph_run):
    _, _, audit = audit_run(graph_run)
    assert audit["compute_graphs"] == {
        "enabled": True,
        "measured_replays": {"hbm": 5200, "echo": 2640},
        "resident_repeat_replays": 5200,
        "warmup_replay_evidence": "not_saved",
    }
    assert audit["all_candidate_hidden_and_logits_numerical_pass"]


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize(
    "defect",
    ["missing", "policy", "allocation", "private", "precision", "memory", "branch", "final"],
)
def test_graph_report_rejects_bank_corruption(graph_run, repeat, defect):
    path = graph_run / "metadata.json"
    metadata = json.loads(path.read_text())
    case = metadata["validation_repeats"][0] if repeat else metadata["cases"][1]
    bank = case["backend"]["compute_graphs"]
    if defect == "missing":
        del case["backend"]["compute_graphs"]
    elif defect == "policy":
        bank["policy_revision"] = "unvalidated"
    elif defect == "allocation":
        bank["static_allocated_bytes"] = 101
    elif defect == "private":
        bank["private_reserved_bytes"] = 129
    elif defect == "precision":
        bank["captured_precision_policy"]["cuda_matmul_allow_tf32"] = True
    elif defect == "memory":
        bank["memory_at_allocation"]["device_used"] = 199
    elif defect == "branch":
        bank["graphs"][0]["residual_present"] = True
    else:
        bank["graphs"][0]["projection_replays"] += 1
    write_json(path, metadata)
    with pytest.raises(ValueError, match="compute graph"):
        audit_run(graph_run)


@pytest.mark.parametrize(
    "defect", ["query", "count", "storage", "bound_sum", "bound_kind", "shared", "mode"]
)
def test_graph_report_rejects_inconsistent_plan(graph_run, defect):
    path = graph_run / "metadata.json"
    metadata = json.loads(path.read_text())
    case = metadata["cases"][1]
    plan = case["resource_plan"]
    if defect == "query":
        plan["compute_graph_query_sizes"] = [2]
    elif defect == "count":
        plan["compute_graph_count"] = 20
    elif defect == "storage":
        plan["compute_graph_static_storage_bytes"] = True
    elif defect == "bound_sum":
        plan["compute_graph_reserved_limit_bytes"] += 1
    elif defect == "bound_kind":
        plan["compute_graph_private_limit_kind"] = "observed_fixed_overhead"
    elif defect == "shared":
        case["shared_reservation"]["hbm"] = 227
    else:
        metadata["config"].pop("enable_compute_graphs")
    write_json(path, metadata)
    with pytest.raises(ValueError, match="compute graph"):
        audit_run(graph_run)


@pytest.mark.parametrize("filename", ["measurements.jsonl", "validation.jsonl"])
@pytest.mark.parametrize(
    "defect", ["missing", "fallback", "shape", "delta", "initial", "continuity", "drift"]
)
def test_graph_report_rechecks_each_request_pair(graph_run, filename, defect):
    path = graph_run / filename
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    pair = rows[0 if defect == "initial" else 1]["compute_graphs"]
    if defect == "missing":
        del rows[1]["compute_graphs"]
    elif defect == "fallback":
        pair["after"]["eager_fallbacks"] = 1
    elif defect == "shape":
        pair["before"]["graphs"][0]["queries"] = 3
    elif defect == "delta":
        pair["after"]["graphs"][0]["finish_replays"] -= 1
    elif defect in ("initial", "continuity"):
        for state in pair.values():
            state["graphs"][0]["projection_replays"] += 1
    else:
        pair["after"]["setup_seconds"] += 1
    write_rows(path, rows)
    with pytest.raises(ValueError, match="compute graph"):
        audit_run(graph_run)


def test_complete_report_rechecks_two_schemes_and_full_outputs(saved_run):
    _, _, audit = audit_run(saved_run)
    assert audit["checked_requests"] == 8
    assert audit["compared_offload_requests"] == 4
    assert audit["compared_resident_repeat_requests"] == 4
    assert audit["reference_policy"] == "fresh_hbm_same_run"
    assert audit["official_artifacts"] == {"source": 1, "native": 1}
    summary = write_report(saved_run, saved_run / "report")
    assert len(summary) == 4
    text = (saved_run / "report/results.md").read_text()
    assert "逐位相等" in text and "只覆盖 candidate forward" in text
    assert "allocated" in text and "reserved" in text


@pytest.mark.parametrize(
    "defect", ["missing", "order", "timing", "host_write", "capacity", "memory"]
)
def test_report_rejects_invalid_trace_even_with_exact_flags(saved_run, defect):
    path = saved_run / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if defect == "missing":
        rows.pop()
    elif defect == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif defect == "timing":
        rows[4]["extend_ms"] += 1
    elif defect == "host_write":
        rows[4]["cache_diagnostics"]["device_to_host_bytes"] = 1152
    elif defect == "capacity":
        rows[4]["cache_diagnostics"]["layers"][0]["device_slots"] = 4096
    else:
        rows[4]["memory_after"]["torch_reserved_bytes"] += 1
    write_rows(path, rows)
    with pytest.raises(ValueError):
        audit_run(saved_run)


@pytest.mark.parametrize("name", ["hidden", "logits"])
def test_report_checks_tensor_values_instead_of_recorded_status(saved_run, name):
    path = saved_run / "numerical/echo/000001.pt"
    payload = torch.load(path, weights_only=True)
    payload[name][0, 0] += 1
    torch.save(payload, path)
    rows_path = saved_run / "measurements.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[5]["output_sha256"] = digest(path)
    write_rows(rows_path, rows)
    with pytest.raises(ValueError, match="numerical gate"):
        audit_run(saved_run)


@pytest.mark.parametrize("defect", ["native", "identity_check", "plan", "warmup"])
def test_report_rejects_corrupt_provenance_or_capacity(saved_run, defect):
    path = saved_run / "metadata.json"
    metadata = json.loads(path.read_text())
    if defect == "native":
        manifest = json.loads((saved_run / "official_artifact_manifest.json").read_text())
        binary = next(item for item in manifest if item["kind"] == "native")
        (saved_run / binary["snapshot"]).write_bytes(b"different binary")
    elif defect == "identity_check":
        metadata["identity_verifications"].pop()
    elif defect == "plan":
        metadata["cases"][1]["resource_plan"]["host_arena_tokens"] *= 2
    else:
        metadata["warmup_traces"]["echo"][2]["recalled_records"] = 0
    write_json(path, metadata)
    with pytest.raises(ValueError):
        audit_run(saved_run)


@pytest.mark.parametrize("defect", ["missing", "wrong_request", "latency", "tensor"])
def test_resident_repeat_is_independent_complete_and_audited(saved_run, defect):
    path = saved_run / "validation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if defect == "missing":
        rows.pop()
    elif defect == "wrong_request":
        rows[1]["request_id"] = 0
    elif defect == "latency":
        rows[0]["latency_ms"] = 10
    else:
        tensor_path = saved_run / rows[1]["output_file"]
        payload = torch.load(tensor_path, weights_only=True)
        payload["hidden"][0, 0] += 1
        torch.save(payload, tensor_path)
        rows[1]["output_sha256"] = digest(tensor_path)
    write_rows(path, rows)
    with pytest.raises(ValueError):
        audit_run(saved_run)


def test_report_accepts_numerical_pass_without_claiming_bitwise_equality(saved_run):
    path = saved_run / "numerical/echo/000001.pt"
    payload = torch.load(path, weights_only=True)
    reference = torch.load(saved_run / "numerical/hbm/000001.pt", weights_only=True)
    payload["hidden"][0, 0] += 1 / 128
    torch.save(payload, path)
    rows_path = saved_run / "measurements.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[5]["output_sha256"] = digest(path)
    rows[5]["numerical"] = {
        name: numerical_comparison(payload[name], reference[name], name)
        for name in ("hidden", "logits")
    }
    write_rows(rows_path, rows)
    _, _, audit = audit_run(saved_run)
    assert audit["all_candidate_hidden_and_logits_numerical_pass"]
    assert not audit["all_offload_candidate_hidden_and_logits_bitwise_equal"]
    assert audit["numerical_summary"]["echo"]["hidden"]["bitwise_equal_requests"] == 3


@pytest.mark.parametrize("defect", ["missing", "tf32", "drift", "dispatch"])
def test_report_requires_explicit_stable_precision_and_official_dispatch(saved_run, defect):
    path = saved_run / "metadata.json"
    metadata = json.loads(path.read_text())
    if defect == "missing":
        del metadata["precision_settings"]
    elif defect == "tf32":
        metadata["precision_settings"]["cuda_matmul_allow_tf32"] = True
    elif defect == "drift":
        metadata["precision_settings_final"]["cudnn_allow_tf32"] = False
    else:
        metadata["config"]["indexer_dispatch_policy"] = "local_bypass"
    write_json(path, metadata)
    with pytest.raises(ValueError):
        audit_run(saved_run)
