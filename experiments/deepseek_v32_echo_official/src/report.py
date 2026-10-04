"""Recheck every saved output, capacity and runtime identity before publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

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
    normalize_compute_graphs,
    numerical_comparison,
    provenance_files,
    write_json,
)
from experiments.deepseek_v32_motivation.src.measure import (
    check_compute_graph_replays,
    check_request,
    validate_warmup_trace,
)
from experiments.deepseek_v32_motivation.src.report import (
    flattened,
    percentile,
    read_jsonl,
    require,
    write_csv,
)


def contained(directory, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe artifact path")
    result = directory / path
    require(result.resolve().is_relative_to(directory.resolve()), "artifact escapes run directory")
    return result


def audit_official(directory, metadata):
    path = directory / "official_artifact_manifest.json"
    require(
        digest(path) == metadata["official_artifact_manifest_sha256"],
        "official artifact manifest changed",
    )
    manifest = json.loads(path.read_text())
    expected = set(provenance_files(metadata["official_provenance"]))
    actual = {(item["kind"], item["original"], item["sha256"]) for item in manifest}
    require(
        actual == expected and len(manifest) == len(expected), "official artifact inventory differs"
    )
    snapshots = [item["snapshot"] for item in manifest]
    require(len(set(snapshots)) == len(snapshots), "duplicate official artifact snapshot")
    for item in manifest:
        require(
            digest(contained(directory, item["snapshot"])) == item["sha256"],
            f"official artifact changed: {item['original']}",
        )
    require(
        metadata.get("identity_verifications")
        == [{"stage": stage, "status": "passed"} for stage in VERIFICATION_STAGES],
        "source/native identity checks are incomplete",
    )
    return {kind: sum(item["kind"] == kind for item in manifest) for kind in ("source", "native")}


def audit_case(case, config):
    scheme = case["scheme"]
    require(case["requests"] == config["requests_per_scheme"], "incomplete measured trace")
    require(
        case["warmup_requests"] == 3
        and case["warmup_request_ids"] == [0, 1, config["num_users"]]
        and case["started_empty"] is True,
        "invalid cache initialization or warmup",
    )
    plan = case["resource_plan"]
    for name, expected in {
        "resource_mode": "fixed_pools",
        "sparse_pool_tokens": config["sparse_pool_tokens"],
        "max_history_tokens": config["history_tokens"],
        "max_candidate_tokens": config["candidate_tokens"],
        "max_session_capacity": config["history_tokens"] + config["candidate_tokens"],
        "workspace_query_tokens": config["workspace_query_tokens"],
        "candidate_persistence": "gpu_transient",
    }.items():
        require(plan.get(name) == expected, f"capacity plan differs: {scheme}/{name}")
    if scheme == "hbm":
        require(
            case["admission_hbm_token_capacity"] == config["sparse_pool_tokens"],
            "HBM admission does not enforce P",
        )
    else:
        require(plan.get("host_arena_tokens") == config["host_arena_tokens"], "NH differs")
        require(
            case["admission_page_capacity"] == config["host_arena_tokens"] // 64,
            "host page admission does not enforce NH",
        )
    model = case["backend"]
    for name, expected in {
        "physical_layers": config["layers"],
        "source_layers": [layer % 3 for layer in range(config["layers"])],
        "input_semantics": "copy_source_layer_hidden_and_residual_for_each_physical_copy",
        "linear_backend": "fp8",
        "chunk_size": config["chunk_size"],
        "total_parameters": 7827793408,
        "output": "all_candidate_normalized_hidden_and_last_token_lm_head",
    }.items():
        require(model.get(name) == expected, f"model differs: {scheme}/{name}")


def audit_graph_bank(plan, shared, bank, config, precision):
    """Reconcile saved graph allocation evidence with its declared capacity plan."""
    enabled = normalize_compute_graphs(config)
    require(
        plan.get("compute_graphs_enabled", False) is enabled,
        "compute graph plan contradicts the configured mode",
    )
    if not enabled:
        require(
            bank is None
            or (
                isinstance(bank, dict)
                and bank.get("enabled", False) is False
                and bank.get("allocated", False) is False
            ),
            "compute graph bank contradicts eager mode",
        )
        return

    from models.deepseek_v32.compute_graphs import GRAPH_POLICY_REVISION

    require(isinstance(bank, dict), "missing compute graph bank")
    require(
        bank.get("enabled") is True and bank.get("allocated") is True,
        "compute graph bank was not allocated",
    )
    require(
        bank.get("policy_revision")
        == plan.get("compute_graph_policy_revision")
        == GRAPH_POLICY_REVISION,
        "compute graph policy differs",
    )
    history, chunk = config["history_tokens"], config["chunk_size"]
    sizes = {min(history, chunk), config["candidate_tokens"]}
    if history % chunk:
        sizes.add(history % chunk)
    planned_sizes = plan.get("compute_graph_query_sizes")
    require(
        isinstance(planned_sizes, list)
        and all(type(size) is int for size in planned_sizes)
        and planned_sizes == sorted(sizes),
        "compute graph query plan differs",
    )
    require(
        type(plan.get("compute_graph_count")) is int
        and plan["compute_graph_count"] == 2 * config["layers"] * len(sizes),
        "compute graph count differs",
    )
    for mapping, names in (
        (
            plan,
            (
                "compute_graph_static_storage_bytes",
                "compute_graph_static_allocation_limit_bytes",
                "compute_graph_private_limit_bytes",
                "compute_graph_reserved_limit_bytes",
            ),
        ),
        (
            bank,
            (
                "static_storage_bytes",
                "static_allocated_bytes",
                "private_reserved_bytes",
                "chosen_private_limit_bytes",
            ),
        ),
    ):
        require(
            all(type(mapping.get(name)) is int and mapping[name] > 0 for name in names),
            "invalid compute graph allocation bytes",
        )
    require(
        plan["compute_graph_static_storage_bytes"]
        == bank["static_storage_bytes"]
        <= bank["static_allocated_bytes"]
        <= plan["compute_graph_static_allocation_limit_bytes"],
        "compute graph static allocation exceeds its plan",
    )
    require(
        bank["private_reserved_bytes"]
        <= bank["chosen_private_limit_bytes"]
        == plan["compute_graph_private_limit_bytes"],
        "compute graph private reservation exceeds its chosen limit",
    )
    require(
        plan["compute_graph_reserved_limit_bytes"]
        == plan["compute_graph_static_allocation_limit_bytes"]
        + plan["compute_graph_private_limit_bytes"]
        and plan.get("compute_graph_private_limit_kind")
        == "chosen_upper_limit_not_observed_fixed_overhead",
        "compute graph reservation bound differs",
    )
    require(
        isinstance(shared, dict)
        and type(shared.get("hbm")) is int
        and shared["hbm"] >= plan["compute_graph_reserved_limit_bytes"],
        "compute graph reservation is missing from shared HBM planning",
    )
    memory = bank.get("memory_at_allocation", {})
    require(isinstance(memory, dict), "missing compute graph allocation memory observation")
    require(
        all(
            type(memory.get(name)) is int and memory[name] >= 0
            for name in (
                "pytorch_allocated",
                "pytorch_reserved",
                "device_used",
                "device_total",
            )
        )
        and bank["static_allocated_bytes"]
        <= memory["pytorch_allocated"]
        <= memory["pytorch_reserved"]
        <= memory["device_used"]
        <= memory["device_total"]
        and bank["static_allocated_bytes"] + bank["private_reserved_bytes"]
        <= memory["pytorch_reserved"],
        "invalid compute graph allocation memory observation",
    )
    captured = bank.get("captured_precision_policy", {})
    require(isinstance(captured, dict), "missing compute graph captured precision")
    require(
        captured.get("autocast_enabled") is False
        and all(
            type(captured.get(name)) is type(precision[name]) and captured[name] == precision[name]
            for name in (
                "float32_matmul_precision",
                "cuda_matmul_allow_tf32",
                "bf16_reduced_precision_reduction",
                "fp16_reduced_precision_reduction",
            )
        ),
        "compute graph captured precision differs",
    )
    require(
        type(bank.get("eager_fallbacks")) is int and bank["eager_fallbacks"] == 0,
        "compute graph formal trace used eager fallback",
    )
    entries = bank.get("graphs", [])
    require(
        isinstance(entries, list) and all(isinstance(entry, dict) for entry in entries),
        "invalid compute graph entries",
    )
    expected = {(layer, count) for layer in range(config["layers"]) for count in sizes}
    keys = [(entry.get("layer"), entry.get("queries")) for entry in entries]
    require(
        all(type(layer) is int and type(count) is int for layer, count in keys)
        and len(keys) == len(expected)
        and set(keys) == expected,
        "compute graph bank does not cover every layer and query size",
    )
    for entry in entries:
        require(
            entry.get("residual_present") is (entry["layer"] % 3 != 0)
            and all(
                type(entry.get(name)) is int and entry[name] >= 0
                for name in ("projection_replays", "finish_replays")
            ),
            "invalid compute graph branch or replay counter",
        )


def audit_graph_trace(rows, case, config, precision):
    """Audit complete replay coverage, an unchanged bank and independent empty starts."""
    plan, shared = case["resource_plan"], case.get("shared_reservation")
    final = case.get("backend", {}).get("compute_graphs")
    audit_graph_bank(plan, shared, final, config, precision)
    enabled = normalize_compute_graphs(config)
    previous, replays = None, 0
    for row in rows:
        pair = row.get("compute_graphs")
        if not enabled and pair is None:
            continue
        require(
            isinstance(pair, dict) and set(pair) == {"before", "after"},
            "missing compute graph request evidence",
        )
        before, after = pair["before"], pair["after"]
        for bank in (before, after):
            audit_graph_bank(plan, shared, bank, config, precision)
        if not enabled:
            continue
        check_compute_graph_replays(before, after, config, row)
        if previous is None:
            require(
                all(
                    entry[name] == 0
                    for entry in before["graphs"]
                    for name in ("projection_replays", "finish_replays")
                ),
                "compute graph trace did not start with a fresh bank",
            )
        else:
            require(before == previous, "compute graph bank changed between requests")
        require(
            {key: value for key, value in before.items() if key != "graphs"}
            == {key: value for key, value in after.items() if key != "graphs"},
            "compute graph allocation or policy changed during a request",
        )
        previous = after
        replays += sum(
            entry[name]
            for entry in after["graphs"]
            for name in ("projection_replays", "finish_replays")
        )
        replays -= sum(
            entry[name]
            for entry in before["graphs"]
            for name in ("projection_replays", "finish_replays")
        )
    if enabled:
        require(
            previous is not None and previous == final,
            "compute graph final bank differs from the recorded trace",
        )
    return replays


def audit_comparisons(payload, reference, recorded):
    results = {}
    for name in ("hidden", "logits"):
        result = numerical_comparison(payload[name], reference[name], name)
        require(result["numerical_pass"], f"{name} failed the fixed numerical gate")
        require(set(recorded[name]) == set(result), "incomplete recorded numerical metrics")
        for key, value in result.items():
            actual = recorded[name][key]
            if isinstance(value, float):
                require(
                    isinstance(actual, (int, float))
                    and math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-15),
                    f"recorded numerical metric differs: {name}/{key}",
                )
            else:
                require(actual == value, f"recorded numerical metric differs: {name}/{key}")
        results[name] = result
    return results


def summarize_numerical(groups):
    return {
        label: {
            "requests": len(rows),
            **{
                name: {
                    "max_abs": max(row[name]["max_abs"] for row in rows),
                    "max_relative_l2": max(row[name]["relative_l2"] for row in rows),
                    "minimum_elementwise_pass_fraction": min(
                        row[name]["elementwise_pass_fraction"] for row in rows
                    ),
                    "maximum_elementwise_outliers": max(
                        row[name]["elementwise_outliers"] for row in rows
                    ),
                    "bitwise_equal_requests": sum(row[name]["bitwise_equal"] for row in rows),
                    "allclose_requests": sum(row[name]["allclose"] for row in rows),
                    "numerical_pass_requests": sum(row[name]["numerical_pass"] for row in rows),
                }
                for name in ("hidden", "logits")
            },
        }
        for label, rows in groups.items()
    }


def audit_run(directory):
    import torch

    from GR.workload import token_sha256

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata.get("schema") == SCHEMA, "unknown measurement schema")
    require(metadata.get("status") == "accepted", "run is incomplete or failed")
    precision = metadata.get("precision_settings", {})
    require(metadata.get("precision_policy") == PRECISION_POLICY, "precision policy changed")
    require(
        set(precision)
        == {
            "float32_matmul_precision",
            "cuda_matmul_allow_tf32",
            "bf16_reduced_precision_reduction",
            "fp16_reduced_precision_reduction",
            "cudnn_allow_tf32",
        }
        and precision.get("float32_matmul_precision") == "highest"
        and precision.get("cuda_matmul_allow_tf32") is False
        and all(
            type(precision[name]) is bool
            for name in precision
            if name != "float32_matmul_precision"
        )
        and metadata.get("precision_settings_final") == precision,
        "precision settings are missing, invalid or changed during measurement",
    )
    require(metadata.get("numerical_policy") == NUMERICAL_POLICY, "fixed numerical policy changed")
    require(
        metadata.get("numerical_policy_basis") == NUMERICAL_POLICY_BASIS,
        "numerical policy basis changed",
    )
    config = metadata["config"]
    require(
        config == configuration(SimpleNamespace(run_id=metadata["run_id"], **config)),
        "experiment configuration differs from the declared policy",
    )
    require(metadata.get("scheme_labels") == LABELS, "implementation labels changed")
    cases = metadata["cases"]
    require([case["scheme"] for case in cases] == list(SCHEMES), "incomplete scheme matrix")
    for case in cases:
        audit_case(case, config)
    repeats = metadata.get("validation_repeats", [])
    require(len(repeats) == 1, "missing independent resident repeat")
    repeat = repeats[0]
    require(
        repeat.get("label") == "hbm_repeat"
        and repeat.get("scheme") == "hbm"
        and repeat.get("started_empty") is True
        and repeat.get("timing_published") is False
        and repeat.get("requests") == config["requests_per_scheme"]
        and repeat.get("resource_plan") == cases[0]["resource_plan"]
        and repeat.get("admission_hbm_token_capacity") == config["sparse_pool_tokens"],
        "invalid independent resident repeat",
    )
    source_manifest = json.loads((directory / "source_manifest.json").read_text())
    require(bool(source_manifest), "missing source snapshot")
    source_id = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest()
    require(source_id == metadata["source_sha256"], "source identity mismatch")
    for name, expected in source_manifest.items():
        require(
            digest(contained(directory / "source", name)) == expected, f"source changed: {name}"
        )
    official_counts = audit_official(directory, metadata)
    requests = read_jsonl(directory / "workload/requests.jsonl")
    workload = json.loads((directory / "workload/workload.json").read_text())
    count = config["requests_per_scheme"]
    require(len(requests) == count, "incomplete workload")
    require(
        workload["workload_sha256"] == metadata["workload_sha256"], "workload identity mismatch"
    )
    identity = {
        key: workload[key] for key in ("config", "heat_sha256", "tokenizer_sha256", "requests")
    }
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    require(identity_digest == metadata["workload_sha256"], "workload manifest changed")
    require(len(workload["requests"]) == count, "incomplete workload identity inventory")
    first_round, histories, suffixes = [], {}, {}
    for index, request in enumerate(requests):
        require(
            all(request.get(key) == value for key, value in workload["requests"][index].items()),
            "request differs from workload identity inventory",
        )
        require(request["request_id"] == index, "request order changed")
        ids = request["input_ids"]
        require(
            len(ids) == config["history_tokens"] + config["candidate_tokens"],
            "request token count changed",
        )
        require(token_sha256(ids) == request["input_sha256"], "input tokens changed")
        user = request["user_id"]
        if index < config["num_users"]:
            first_round.append(user)
        require(user == first_round[index % config["num_users"]], "trace is not sequential rounds")
        history, candidate = ids[: config["history_tokens"]], ids[config["history_tokens"] :]
        require(histories.setdefault(user, history) == history, "retained user history changed")
        require(user not in suffixes or suffixes[user] != candidate, "candidate did not change")
        suffixes[user] = candidate
    require(len(set(first_round)) == config["num_users"], "first round repeats a user")
    traces = metadata.get("warmup_traces", {})
    require(set(traces) == set(SCHEMES), "missing warmup trace")
    for scheme in SCHEMES:
        validate_warmup_trace(scheme, traces[scheme], config, requests)
    memory = json.loads((directory / "memory.json").read_text())
    samples = {item["stage"]: item for item in memory}
    require(len(samples) == len(memory), "duplicate memory sample stage")
    rows = read_jsonl(directory / "measurements.jsonl")
    require(len(rows) == count * len(SCHEMES), "incomplete measurement matrix")
    require(
        [(row["scheme"], row["request_id"]) for row in rows]
        == [(scheme, index) for scheme in SCHEMES for index in range(count)],
        "request measurement order changed",
    )
    by_key = {(row["scheme"], row["request_id"]): row for row in rows}
    graph_replays = {
        case["scheme"]: audit_graph_trace(
            [row for row in rows if row["scheme"] == case["scheme"]],
            case,
            config,
            precision,
        )
        for case in cases
    }
    tensor_hashes = {}
    numerical_groups = {"hbm_repeat": [], "echo": []}
    for index, request in enumerate(requests):
        reference = None
        for scheme in SCHEMES:
            row = by_key[(scheme, index)]
            require(row["run_id"] == metadata["run_id"], "run ID mismatch")
            require(row["workload_sha256"] == metadata["workload_sha256"], "row workload mismatch")
            require(row["input_sha256"] == request["input_sha256"], "row input mismatch")
            require(row["diagnostics_scope"] == "candidate_only", "ambiguous transfer scope")
            check_request(request, row, config)
            diagnostics = row["cache_diagnostics"]
            require(
                diagnostics.get("candidate_persistence") == "gpu_transient"
                and diagnostics.get("retained_length") == config["history_tokens"]
                and diagnostics["device_to_host_bytes"] == 0,
                "candidate lifecycle changed",
            )
            require(
                len(diagnostics.get("layers", [])) == config["layers"], "missing layer capacity"
            )
            for layer in diagnostics["layers"]:
                require(layer.get("device_to_host_bytes") == 0, "layer candidate wrote host KV")
                if scheme == "echo":
                    require(
                        layer.get("device_slots") == config["sparse_pool_tokens"]
                        and layer.get("host_token_capacity") == config["host_arena_tokens"]
                        and layer.get("session_host_tokens") == config["padded_history_tokens"],
                        "layer P/NH/history capacity differs",
                    )
            for name in ("latency_ms", "admission_ms", "prefix_ms", "extend_ms", "cleanup_ms"):
                require(math.isfinite(row[name]) and row[name] >= 0, f"invalid {name}")
            require(row["latency_ms"] > 0, "request latency must be positive")
            stages = sum(
                row[name] for name in ("admission_ms", "prefix_ms", "extend_ms", "cleanup_ms")
            )
            require(
                math.isclose(stages, row["latency_ms"], abs_tol=1e-5), "timing stages do not sum"
            )
            for boundary in ("before", "after"):
                sample = row[f"memory_{boundary}"]
                require(
                    sample == samples.get(f"{scheme}/{boundary}_request_{index}"),
                    "row memory sample differs from raw observation",
                )
                require(
                    0 <= sample["torch_allocated_bytes"] <= sample["torch_reserved_bytes"]
                    and sample["torch_peak_allocated_bytes"] <= sample["torch_peak_reserved_bytes"]
                    and 0 <= sample["cuda_free_bytes"] <= sample["cuda_total_bytes"],
                    "invalid memory observation",
                )
            relative = row["output_file"]
            require(relative == f"numerical/{scheme}/{index:06d}.pt", "unexpected output path")
            path = contained(directory, relative)
            require(digest(path) == row["output_sha256"], "output bytes changed")
            tensor_hashes[relative] = row["output_sha256"]
            payload = torch.load(path, weights_only=True, map_location="cpu")
            require(payload["request_id"] == index, "output request identity changed")
            require(
                payload["input_sha256"] == request["input_sha256"], "output input identity changed"
            )
            dims = metadata["model_dimensions"]
            require(
                tuple(payload["hidden"].shape) == (config["candidate_tokens"], dims["hidden"]),
                "output does not cover every candidate hidden state",
            )
            require(tuple(payload["logits"].shape) == (1, dims["vocabulary"]), "incomplete logits")
            if reference is None:
                reference = payload
            comparison = audit_comparisons(payload, reference, row["numerical"])
            if scheme == "echo":
                numerical_groups[scheme].append(comparison)
    validation = read_jsonl(directory / "validation.jsonl")
    require(len(validation) == count, "incomplete resident repeat outputs")
    repeat_replays = audit_graph_trace(validation, repeat, config, precision)
    for index, row in enumerate(validation):
        request = requests[index]
        require(
            row.get("validation_label") == "hbm_repeat"
            and row["request_id"] == index
            and row["run_id"] == metadata["run_id"]
            and row["workload_sha256"] == metadata["workload_sha256"]
            and row["input_sha256"] == request["input_sha256"],
            "resident repeat request identity differs",
        )
        require(not any(key.endswith("_ms") for key in row), "resident validation includes latency")
        require(row["scheme"] == "hbm", "resident validation used another backend")
        check_request(request, row, config)
        diagnostics = row["cache_diagnostics"]
        require(
            diagnostics.get("candidate_persistence") == "gpu_transient"
            and diagnostics.get("retained_length") == config["history_tokens"],
            "resident repeat candidate lifecycle differs",
        )
        relative = row["output_file"]
        require(relative == f"numerical/hbm_repeat/{index:06d}.pt", "unexpected repeat output path")
        path = contained(directory, relative)
        require(digest(path) == row["output_sha256"], "resident repeat output bytes changed")
        tensor_hashes[relative] = row["output_sha256"]
        payload = torch.load(path, weights_only=True, map_location="cpu")
        require(
            payload["request_id"] == index and payload["input_sha256"] == request["input_sha256"],
            "resident repeat output identity differs",
        )
        reference = torch.load(
            directory / f"numerical/hbm/{index:06d}.pt", weights_only=True, map_location="cpu"
        )
        numerical_groups["hbm_repeat"].append(
            audit_comparisons(payload, reference, row["numerical"])
        )
    for case in cases:
        final = by_key[(case["scheme"], count - 1)]["memory_after"]
        for name in ("torch_peak_allocated_bytes", "torch_peak_reserved_bytes"):
            require(case[name] == final[name], "case memory peak differs from final sample")
    return (
        metadata,
        rows,
        {
            "status": "passed",
            "checked_requests": len(rows),
            "compared_offload_requests": count,
            "compared_resident_repeat_requests": count,
            "all_candidate_hidden_and_logits_numerical_pass": True,
            "all_offload_candidate_hidden_and_logits_bitwise_equal": all(
                item[name]["bitwise_equal"]
                for item in numerical_groups["echo"]
                for name in ("hidden", "logits")
            ),
            "numerical_summary": summarize_numerical(numerical_groups),
            "reference_policy": "fresh_hbm_same_run",
            "source_sha256": source_id,
            "official_artifacts": official_counts,
            "checked_warmup_requests": 3 * len(SCHEMES),
            "host_recall_warmed_schemes": ["echo"],
            "numerical_sha256": tensor_hashes,
            **(
                {
                    "compute_graphs": {
                        "enabled": True,
                        "measured_replays": graph_replays,
                        "resident_repeat_replays": repeat_replays,
                        "warmup_replay_evidence": "not_saved",
                    },
                }
                if normalize_compute_graphs(config)
                else {}
            ),
        },
    )


def summarize(rows):
    summary = []
    for scheme in SCHEMES:
        for revisit in (False, True):
            group = [
                row for row in rows if row["scheme"] == scheme and row["is_revisit"] == revisit
            ]
            require(bool(group), f"missing first/revisit group for {scheme}")
            latency = [row["latency_ms"] for row in group]
            summary.append(
                {
                    "scheme": scheme,
                    "implementation": LABELS[scheme],
                    "visit_kind": "revisit" if revisit else "first",
                    "requests": len(group),
                    "prefix_hits": sum(row["prefix_cache_hit"] for row in group),
                    "latency_mean_ms": statistics.mean(latency),
                    "latency_median_ms": statistics.median(latency),
                    "latency_p95_ms": percentile(latency, 0.95),
                    "latency_sum_ms": sum(latency),
                    "prefix_mean_ms": statistics.mean(row["prefix_ms"] for row in group),
                    "extend_mean_ms": statistics.mean(row["extend_ms"] for row in group),
                    "candidate_host_to_device_bytes": sum(
                        row["cache_diagnostics"]["host_to_device_bytes"] for row in group
                    ),
                    "candidate_device_to_host_bytes": sum(
                        row["cache_diagnostics"]["device_to_host_bytes"] for row in group
                    ),
                    "max_cache_hbm_bytes": max(row["cache_hbm_bytes"] for row in group),
                    "max_cache_dram_bytes": max(row["cache_dram_bytes"] for row in group),
                }
            )
    baseline = {
        row["visit_kind"]: row["latency_mean_ms"] for row in summary if row["scheme"] == "hbm"
    }
    for row in summary:
        row["mean_latency_speedup_vs_hbm"] = baseline[row["visit_kind"]] / row["latency_mean_ms"]
    return summary


def markdown(metadata, summary, audit):
    config, hardware = metadata["config"], metadata["hardware"]
    lines = [
        f"# DeepSeek V3.2 官方 ECHO 适配实验（{metadata['run_id']}）",
        "",
        (
            f"本轮使用 {hardware['name']}，PyTorch {hardware['torch']} / CUDA {hardware['cuda']}。"
            f"{config['num_users']} 个用户顺序访问 {config['rounds']} 轮，H={config['history_tokens']:,}、"
            f"A={config['candidate_tokens']:,}、chunk={config['chunk_size']:,}、"
            f"P={config['sparse_pool_tokens']:,}、NH={config['host_arena_tokens']:,}。"
        ),
        "",
        (
            "官方 ECHO 的 allocator、融合 indexer/prefetch 和精确 recall 接入本地 GR 生命周期；"
            "HBM 对照使用官方 resident logits，两方案采用官方原始 top-k 语义。模型权重、"
            "投影和 FlashMLA 沿用 motivation，indexer 不加 Hadamard。HBM 对照也已替换 "
            "motivation 的 mainline DeepGEMM logits 和 FlashInfer top-k。结果覆盖这条适配路径，"
            "不能表述为未经修改的上游 serving。十个独立 dense block 复用真实 checkpoint 前三层"
            "及对应 source hidden/residual 输入，共 7,827,793,408 个参数；这是工作负载替身。"
        ),
        "",
        (
            "两方案各预热三条请求，包含两位用户首访及第一位用户复访；ECHO 最后一次预热验证"
            "实际 host recall。释放全部预热 cache 后，从空缓存执行正式轨迹。每条请求只测一次，"
            "端到端同步墙钟包含输入搬入、准入/淘汰、miss 时 history 构建、全部 candidate hidden、"
            "末 token LM head 与清理。forward 内的 GPU 计数累积和归约也计入延迟；"
            "加载、编译、预热、计数的 host 读取、输出保存和比较不计时。"
        ),
        "",
        "| 方案 | 访问 | 请求数 | history hit | 均值 ms | 中位数 ms | p95 ms | 相对 HBM 加速 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        visit = "复访" if row["visit_kind"] == "revisit" else "首次"
        lines.append(
            f"| {LABELS[row['scheme']]} | {visit} | {row['requests']} | {row['prefix_hits']}/{row['requests']} "
            f"| {row['latency_mean_ms']:.3f} | {row['latency_median_ms']:.3f} "
            f"| {row['latency_p95_ms']:.3f} | {row['mean_latency_speedup_vs_hbm']:.3f}× |"
        )
    totals = {
        scheme: sum(row["latency_sum_ms"] for row in summary if row["scheme"] == scheme)
        for scheme in SCHEMES
    }
    lines.extend(
        [
            "",
            (
                f"完整轨迹的请求时间之和：HBM-only {totals['hbm'] / 1000:.3f} s，"
                f"官方 ECHO 适配路径 {totals['echo'] / 1000:.3f} s，"
                f"加速 {totals['hbm'] / totals['echo']:.3f}×。被淘汰后的请求仍计为复访；"
                "history hit 只表示用户历史被保留，不代表所选 KV 已在 HBM。"
            ),
            "",
            "| 方案 | 访问 | candidate H2D GiB | candidate D2H GiB |",
            "|---|---|---:|---:|",
        ]
    )
    for row in summary:
        visit = "复访" if row["visit_kind"] == "revisit" else "首次"
        lines.append(
            f"| {LABELS[row['scheme']]} | {visit} | {row['candidate_host_to_device_bytes'] / 2**30:.6f} "
            f"| {row['candidate_device_to_host_bytes'] / 2**30:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "搬运计数只覆盖 candidate forward，不包含 history prefill。GPU 计数累积和归约"
                "包含在 forward 计时中，host 读取在计时外。候选整批在 GPU 执行后丢弃，"
                "NH 只保存历史。P/NH 是 token 容量，不是总 HBM/DRAM 字节数；不扣除经验性 headroom。"
            ),
            "",
            "| 方案 | CUDA allocated 峰值 GiB | reserved 峰值 GiB | cache HBM 最大 GiB | cache DRAM 最大 GiB |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for case in metadata["cases"]:
        group = [row for row in summary if row["scheme"] == case["scheme"]]
        lines.append(
            f"| {LABELS[case['scheme']]} | {case['torch_peak_allocated_bytes'] / 2**30:.6f} "
            f"| {case['torch_peak_reserved_bytes'] / 2**30:.6f} "
            f"| {max(row['max_cache_hbm_bytes'] for row in group) / 2**30:.6f} "
            f"| {max(row['max_cache_dram_bytes'] for row in group) / 2**30:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "allocator 峰值在释放预热 cache 后重置，包含已加载权重与完整正式轨迹。allocated 是活跃"
                "分配，reserved 还含 allocator 缓存，二者不能相加。cache 数字来自请求边界；"
                "设备 free memory 也只在边界采样，不等于连续进程峰值。"
            ),
            "",
            (
                "官方 top-k 的输出顺序会随执行变化，HBM 重复运行也可能产生不同的输出。"
                "本轮在 HBM 正式轨迹后，从空缓存重新执行完整 resident 轨迹，保存每条请求的"
                "全部 hidden/logits，与正式 HBM 输出比较。该重复只用于数值验证，不发布延迟。"
            ),
            "",
            "| 比较 | 输出 | 最大绝对误差 | 最大 relative L2 | 最低元素通过比例 | 逐位相等请求 | allclose 请求 | 数值通过请求 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, values in audit["numerical_summary"].items():
        label_text = "HBM 重复 / HBM" if label == "hbm_repeat" else "官方 ECHO / HBM"
        for name in ("hidden", "logits"):
            value = values[name]
            lines.append(
                f"| {label_text} | {name} | {value['max_abs']:.8g} | {value['max_relative_l2']:.8g} "
                f"| {value['minimum_elementwise_pass_fraction']:.8%} "
                f"| {value['bitwise_equal_requests']}/{values['requests']} "
                f"| {value['allclose_requests']}/{values['requests']} "
                f"| {value['numerical_pass_requests']}/{values['requests']} |"
            )
    lines.extend(
        [
            "",
            (
                "误差在 CPU 上用 FP64 重新计算。每条请求的 hidden / logits 须分别满足 "
                "relative L2≤0.005 / 0.01，且至少 99.9% 元素满足 "
                "|actual−reference|≤1/32+(1/64)×|reference|。全元素 allclose、最大绝对误差与"
                "逐位相等仅作为观测项，不单独决定验收。这些规则依据独立 64K resident 校准，"
                "在正式 64K offload 输出测量前固定；完整依据及 hash 保存在 summary.json 的 "
                "numerical_policy / numerical_policy_basis。top-k 的顺序和并列值可能影响结果，"
                "不能把全部差异都归因为已证实的舍入误差。"
                "报告也检查所有保存输出、每层 P/NH、候选生命周期，以及源码和 native 二进制快照。"
                "这些结果只适用于本轮合成 GR 输入和工作负载替身。"
            ),
            "",
            (
                "逐请求数据见 [per_request.csv](per_request.csv)，分组数据见 [summary.csv](summary.csv)，"
                "完整配置与验收见 [summary.json](summary.json)，来源见 "
                "[report_provenance.json](report_provenance.json)。"
                f"完整产物在 `output/data/{metadata['run_id']}/`。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(directory, output):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    metadata, rows, audit = audit_run(directory)
    summary = summarize(rows)
    output.mkdir(parents=True)
    write_csv(
        output / "per_request.csv",
        [{**flattened(row), "implementation": LABELS[row["scheme"]]} for row in rows],
    )
    write_csv(output / "summary.csv", summary)
    write_json(output / "summary.json", {"metadata": metadata, "summary": summary, "audit": audit})
    (output / "results.md").write_text(markdown(metadata, summary, audit))
    write_json(output / "numerical_summary.json", audit["numerical_summary"])
    write_json(
        output / "report_provenance.json",
        {
            "run_id": metadata["run_id"],
            "audit": audit,
            "input_sha256": {
                name: digest(directory / name)
                for name in (
                    "metadata.json",
                    "measurements.jsonl",
                    "validation.jsonl",
                    "source_manifest.json",
                    "memory.json",
                    "official_artifact_manifest.json",
                    "workload/requests.jsonl",
                    "workload/workload.json",
                )
            },
            "report_generator_sha256": digest(__file__),
        },
    )
    return summary


def main(argv=None):
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    args = command.parse_args(argv)
    write_report(args.run_dir, args.output_dir)
    print(f"verified report: {args.output_dir}")


if __name__ == "__main__":
    main()
