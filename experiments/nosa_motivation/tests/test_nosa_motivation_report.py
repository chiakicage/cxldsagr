"""Exercise the independent report audit with tiny CPU artifacts."""

import hashlib
import json
import shutil
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.nosa_motivation.src.config import BACKEND_SCHEMES, METHODS, SCHEMA, configuration
from experiments.nosa_motivation.src.measure import compare_hidden, effective_work
from experiments.nosa_motivation.src.provenance import digest, manifest_digest, write_json
from experiments.nosa_motivation.src.report import audit_run, summarize, write_report
from GR.workload import token_sha256


@pytest.fixture(params=[False, True], ids=["eager", "graphs"])
def saved_run(tmp_path, request):
    graphs_enabled = request.param
    config = configuration(
        SimpleNamespace(
            run_id="cpu_fixture",
            num_users=2,
            rounds=2,
            history_tokens=64,
            candidate_tokens=2,
            chunk_size=64,
            sparse_pool_tokens=64,
            host_arena_tokens=128,
            seed=42,
            compute_graphs=graphs_enabled,
        )
    )
    workload_config = {
        "model": "nosa",
        "num_users": 2,
        "requests": 4,
        "history_tokens": 64,
        "candidate_tokens": 2,
        "sampling": "sequential",
        "seed": 42,
        "context_limit": 66,
    }
    requests = []
    for index in range(4):
        history = [index % 2] * 64
        candidate = [4 + index, 8 + index]
        requests.append(
            {
                "request_id": index,
                "user_id": index % 2,
                "input_ids": history + candidate,
                "input_sha256": token_sha256(history + candidate),
                "prefix_sha256": token_sha256(history),
                "candidate_sha256": token_sha256(candidate),
            }
        )
    manifest = {
        "config": workload_config,
        "heat_sha256": None,
        "tokenizer_sha256": "fixture",
        "requests": [
            {key: value for key, value in row.items() if key != "input_ids"} for row in requests
        ],
    }
    workload_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    manifest["workload_sha256"] = workload_sha256
    (tmp_path / "workload").mkdir()
    write_json(tmp_path / "workload/workload.json", manifest)
    (tmp_path / "workload/requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in requests)
    )
    source = tmp_path / "source/models/nosa/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("# CPU fixture, never evidence of a GPU run\n")
    source_manifest = {"models/nosa/model.py": digest(source)}
    write_json(tmp_path / "source_manifest.json", source_manifest)
    metadata = {
        "schema": SCHEMA,
        "run_id": "cpu_fixture",
        "status": "accepted",
        "config": config,
        "model_config": {
            "num_hidden_layers": 32,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
        },
        "source_sha256": manifest_digest(source_manifest),
        "workload_sha256": workload_sha256,
        "cases": [],
        "warmup_traces": {},
        "model_boundary": "CPU fixture only",
        "measurement_boundary": "Synthetic fixture times",
        "hardware": {
            "total_memory": 4096,
            "uuid": "fixture",
            "packages": {},
            "torch_cuda": "fixture",
            "compute_capability": [9, 0],
            "sm_count": 1,
        },
        "checkpoint": {"identity_boundary": "fixture only"},
        "precision_settings": {},
    }
    header_files = {"fixture.h": {"size": 1, "sha256": "a" * 64}}
    tree = {
        "path": "/fixture/include",
        "files": header_files,
        "sha256": manifest_digest(header_files),
    }
    build = {
        "dependencies": {
            "cutlass": {"include": tree, "util_include": tree},
            **{
                name: tree
                for name in (
                    "flashinfer_include",
                    "tvm_ffi_include",
                    "dlpack_include",
                    "cuda_include",
                )
            },
        },
    }
    build["sha256"] = manifest_digest(build)
    artifacts = {"/fixture/cxldsagr.so": {"size": 1, "sha256": "b" * 64}}
    metadata["native_provenance"] = {
        "build_before": build,
        "build_after": build,
        "artifacts_final": artifacts,
    }

    def memory(stage):
        return {
            "stage": stage,
            "torch_allocated_bytes": 128,
            "torch_reserved_bytes": 256,
            "torch_peak_allocated_bytes": 128,
            "torch_peak_reserved_bytes": 256,
            "device_total_bytes": 4096,
            "device_used_bytes": 512,
            "device_free_bytes": 3584,
        }

    metadata["model_loaded_memory"] = memory("after_model_loading")
    samples = [metadata["model_loaded_memory"]]
    rows = []
    for method in METHODS:
        graph = {
            "enabled": True,
            "policy_revision": "nosa_compute_graphs_deferred_validation_v2",
            "query_sizes": [2, 64],
            "graphs": 128,
            "static_storage_bytes": 16,
            "static_allocated_bytes": 32,
            "private_reserved_bytes": 64,
            "chosen_private_limit_bytes": 128,
            "static_allocation_limit_bytes": 64,
            "setup_seconds": 0.25,
            "eager_fallbacks": 0,
            "project_replays": 0,
            "finish_replays": 0,
            "finite_validation": "per-layer device flags; one host decision before commit",
        }
        graph_charge = 96 if graphs_enabled else 0
        metadata["cases"].append(
            {
                "method": method,
                "requests": 4,
                "started_empty": True,
                "warmup_requests": 3,
                "admission_hbm_tokens": 64 if method == "hbm" else 0,
                "admission_host_pages": 0 if method == "hbm" else 2,
                "resource_plan": {
                    "sparse_pool_tokens": 64,
                    "host_arena_tokens": 128,
                    "candidate_persistence": "gpu_transient",
                    "logical_shared_hbm_bytes": 16,
                },
                "shared_reservation": {"hbm": 32 + graph_charge, "dram": 0},
                "torch_peak_allocated_bytes": 128,
                "torch_peak_reserved_bytes": 256,
                "native_artifacts_before": artifacts,
                "native_artifacts_after": artifacts,
                "token_validation": {
                    "backend": "cpython_native",
                    "loaded_binary_path": "/fixture/cxldsagr.so",
                    "loaded_binary_sha256": "b" * 64,
                },
            }
        )
        case = metadata["cases"][-1]
        case["backend"] = {
            "resource_plan": deepcopy(case["resource_plan"]),
            "shared_cache_bytes": {"hbm": 16 + graph_charge, "dram": 0},
            "compute_graphs": graph if graphs_enabled else {"enabled": False},
        }
        if graphs_enabled:
            # Use the actual planner wrapper: the backend retains its base
            # metadata while the runner receives graph metadata and capacity.
            from cache.prefix_pool import CacheFootprint
            from executor.serving_backend import SharedCachePlan
            from models.nosa.serving import NosaServingBackend

            base = SharedCachePlan(shared=CacheFootprint(hbm=32), metadata=case["resource_plan"])
            planning_backend = SimpleNamespace(
                device=torch.device("cpu"),
                resources=SimpleNamespace(plan_resources=lambda *_, value=base: value),
                compute_graphs=SimpleNamespace(
                    shared_bytes=lambda value=graph_charge: {"hbm": value, "dram": 0},
                    describe=lambda value=graph: deepcopy(value),
                ),
            )
            wrapped = NosaServingBackend.plan_resources(planning_backend, None, {})
            case["resource_plan"] = dict(wrapped.metadata)
            assert wrapped.shared.hbm == case["shared_reservation"]["hbm"]
        method_rows = []
        for workload_request in requests:
            index = workload_request["request_id"]
            h2d = 64 if index >= 2 and method != "hbm" else 0
            d2h = 0 if method == "hbm" or index >= 2 else 128
            row = {
                **{key: value for key, value in workload_request.items() if key != "input_ids"},
                "method": method,
                "scheme": BACKEND_SCHEMES[method],
                "run_id": "cpu_fixture",
                "workload_sha256": workload_sha256,
                "visit_index": index // 2,
                "is_revisit": index >= 2,
                "prefix_cache_hit": index >= 2 and method != "hbm",
                "stable_prefix_tokens": 64,
                "candidate_suffix_tokens": 2,
                "resource_mode": "fixed_pools",
                "hbm_budget_bytes": None,
                "dram_budget_bytes": None,
                "cached_users": 1 if method == "hbm" else min(index + 1, 2),
                "cache_diagnostics": {
                    "candidate_persistence": "gpu_transient",
                    "retained_length": 64,
                    "transfer_metrics_status": "complete",
                    "host_to_device_bytes": h2d,
                    "prefix_host_to_device_bytes": 0,
                    "candidate_host_to_device_bytes": h2d,
                    "candidate_main_kv_host_to_device_bytes": h2d,
                    "device_to_host_bytes": d2h,
                    "prefix_device_to_host_bytes": d2h,
                    "candidate_device_to_host_bytes": 0,
                },
                "latency_ms": 1.0,
                "admission_ms": 0.1,
                "prefix_ms": 0.2,
                "extend_ms": 0.3,
                "cleanup_ms": 0.4,
                "evicted_users": [],
                "cache_hbm_bytes": 64,
                "cache_dram_bytes": 0 if method == "hbm" else 128,
                "shared_cache_hbm_bytes": 16 + graph_charge,
                "shared_cache_dram_bytes": 0,
                "shared_reserved_hbm_bytes": 32 + graph_charge,
                "shared_reserved_dram_bytes": 0,
                "memory_after": memory(f"{method}/after_request_{index}"),
            }
            if graphs_enabled:
                before = deepcopy(graph)
                calls = 32 * (1 if row["prefix_cache_hit"] else 2)
                graph["project_replays"] += calls
                graph["finish_replays"] += calls
                row["compute_graphs"] = {"before": before, "after": deepcopy(graph)}
            samples.append(row["memory_after"])
            hidden = torch.full((2, 4), index, dtype=torch.bfloat16)
            row["numerical"] = compare_hidden(hidden, hidden, config)
            row["effective_work"] = effective_work(metadata["model_config"], config, row)
            relative = f"numerical/{method}/{index:06d}.pt"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "hidden": hidden,
                    "request_id": index,
                    "input_sha256": workload_request["input_sha256"],
                },
                path,
            )
            row.update(output_file=relative, output_sha256=digest(path))
            rows.append(row)
            method_rows.append(row)
        metadata["warmup_traces"][method] = [method_rows[index] for index in (0, 1, 2)]
    write_json(tmp_path / "metadata.json", metadata)
    write_json(tmp_path / "memory.json", samples)
    (tmp_path / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def test_audit_reopens_all_outputs_and_publishes_cpu_fixture(saved_run):
    metadata, rows, audit = audit_run(saved_run)
    assert audit["checked_requests"] == 16
    assert audit["offload_requests_compared_to_independent_hbm"] == 12
    assert audit["exact_requests"] == 16
    summary = summarize(rows)
    assert len(summary) == 8
    assert summary[1]["visit_kind"] == "revisit" and summary[1]["prefix_hits"] == 0
    assert summary[3]["prefix_hits"] == 2
    destination = saved_run / "report"
    write_report(saved_run, destination)
    assert (destination / "audit.json").exists()
    assert metadata["source_sha256"] in (destination / "results.md").read_text()
    with pytest.raises(FileExistsError):
        write_report(saved_run, destination)


def test_check_schema_validates_full_outputs_without_publishing_latency(saved_run):
    from experiments.nosa_motivation.src.config import CHECK_SCHEMA
    from experiments.nosa_motivation.src.validation import without_performance

    metadata = json.loads((saved_run / "metadata.json").read_text())
    metadata.update(schema=CHECK_SCHEMA, mode="check")
    write_json(saved_run / "metadata.json", metadata)
    rows = [
        without_performance(json.loads(line))
        for line in (saved_run / "measurements.jsonl").read_text().splitlines()
    ]
    (saved_run / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    _, reopened, audit = audit_run(saved_run)
    assert audit["all_candidate_hidden_checked"] and audit["checked_requests"] == 16
    assert all("latency_ms" not in row and "effective_work" not in row for row in reopened)
    with pytest.raises(ValueError, match="check runs do not publish"):
        write_report(saved_run, saved_run / "check_report")


def test_clean_bench_report_requires_receipt_and_never_loads_tensors(
    saved_run, tmp_path_factory, monkeypatch
):
    from evaluation.validation import write_receipt
    from experiments.nosa_motivation.src.config import BENCH_SCHEMA
    from experiments.nosa_motivation.src.measure import RECEIPT_KIND
    from experiments.nosa_motivation.src.validation import base_identity

    check_dir = tmp_path_factory.mktemp("independent_numerical_check")
    shutil.copytree(saved_run, check_dir, dirs_exist_ok=True)
    metadata = json.loads((saved_run / "metadata.json").read_text())
    metadata.update(schema=BENCH_SCHEMA, mode="bench", execution_environment={})
    identity = {
        "base": base_identity(metadata, saved_run),
        "methods": {name: {"cpu_fixture_only": True} for name in METHODS},
    }
    receipt = check_dir / "receipt.json"
    write_receipt(
        receipt,
        kind=RECEIPT_KIND,
        identity=identity,
        checks={"passed": True},
        artifacts={"checked_rows": check_dir / "measurements.jsonl"},
    )
    metadata["validation_identity"] = identity
    metadata["correctness_receipt"] = {
        "path": str(receipt),
        "sha256": digest(receipt),
        "kind": RECEIPT_KIND,
        "identity": identity,
    }
    write_json(saved_run / "metadata.json", metadata)
    rows = [
        {
            key: value
            for key, value in json.loads(line).items()
            if key not in {"numerical", "output_file", "output_sha256"}
        }
        for line in (saved_run / "measurements.jsonl").read_text().splitlines()
    ]
    (saved_run / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    shutil.rmtree(saved_run / "numerical")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("bench loaded tensors"))
    _, _, audit = audit_run(saved_run)
    assert not audit["all_candidate_hidden_checked"]
    assert audit["independent_correctness_receipt"]["checks"]["passed"]
    write_report(saved_run, saved_run / "bench_report")
    (check_dir / "measurements.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="evidence changed"):
        audit_run(saved_run)


@pytest.mark.parametrize(
    "corruption",
    ("output", "source", "incomplete", "phase", "config", "memory", "binary", "headers", "shared"),
)
def test_report_rejects_corrupted_evidence(saved_run, corruption):
    if corruption == "output":
        (saved_run / "numerical/async_sparse/000003.pt").write_bytes(b"changed")
    elif corruption == "source":
        (saved_run / "source/models/nosa/model.py").write_text("changed")
    elif corruption == "config":
        metadata = json.loads((saved_run / "metadata.json").read_text())
        metadata["config"]["numerical_atol"] = 1.0
        write_json(saved_run / "metadata.json", metadata)
    elif corruption == "memory":
        memory = json.loads((saved_run / "memory.json").read_text())
        memory[-1]["torch_reserved_bytes"] = 2048
        write_json(saved_run / "memory.json", memory)
    elif corruption in ("binary", "headers"):
        metadata = json.loads((saved_run / "metadata.json").read_text())
        if corruption == "binary":
            metadata["cases"][-1]["native_artifacts_after"]["/fixture/cxldsagr.so"]["sha256"] = (
                "changed"
            )
        else:
            metadata["native_provenance"]["build_after"]["dependencies"]["cutlass"]["include"][
                "files"
            ]["fixture.h"]["sha256"] = "changed"
        write_json(saved_run / "metadata.json", metadata)
    else:
        path = saved_run / "measurements.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if corruption == "incomplete":
            rows.pop()
        elif corruption == "shared":
            rows[-1]["shared_cache_hbm_bytes"] = 12
        else:
            rows[-1]["cache_diagnostics"]["candidate_device_to_host_bytes"] = 1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((ValueError, AssertionError)):
        audit_run(saved_run)


def test_report_failure_leaves_no_partial_publication(saved_run, monkeypatch):
    from experiments.nosa_motivation.src import report

    def failed_report(directory, destination, *args):
        (destination / "partial.csv").write_text("partial")
        raise RuntimeError("simulated rendering failure")

    monkeypatch.setattr(report, "_write_report", failed_report)
    destination = saved_run / "report"
    with pytest.raises(RuntimeError, match="rendering"):
        write_report(saved_run, destination)
    assert not destination.exists()


@pytest.mark.parametrize("saved_run", [True], indirect=True)
@pytest.mark.parametrize(
    "corruption",
    [
        "base_plan",
        "graph_description",
        "static_limit",
        "private_limit",
        "shared_charge",
        "reservation",
        "row_storage",
    ],
)
def test_graph_resource_accounting_rejects_changed_components(saved_run, corruption):
    metadata = json.loads((saved_run / "metadata.json").read_text())
    case = metadata["cases"][-1]
    if corruption == "base_plan":
        case["backend"]["resource_plan"]["logical_shared_hbm_bytes"] += 1
    elif corruption == "graph_description":
        case["resource_plan"]["compute_graphs"]["setup_seconds"] += 1
    elif corruption in ("static_limit", "private_limit"):
        name = (
            "static_allocation_limit_bytes"
            if corruption == "static_limit"
            else "chosen_private_limit_bytes"
        )
        case["backend"]["compute_graphs"][name] = 0
        case["resource_plan"]["compute_graphs"][name] = 0
    elif corruption == "shared_charge":
        case["backend"]["shared_cache_bytes"]["hbm"] = 16
    elif corruption == "reservation":
        case["shared_reservation"]["hbm"] = 111
    else:
        path = saved_run / "measurements.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for name in ("before", "after"):
            rows[-1]["compute_graphs"][name]["private_reserved_bytes"] += 1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(saved_run / "metadata.json", metadata)
    with pytest.raises(ValueError):
        audit_run(saved_run)


@pytest.fixture
def saved_profile(saved_run):
    from experiments.nosa_motivation.src.flops import matrix_flops
    from experiments.nosa_motivation.src.profile_hbm import analyze_trace as analyze_api_trace
    from experiments.nosa_motivation.src.timeline import analyze_trace

    reference = json.loads((saved_run / "metadata.json").read_text())
    rows = [
        json.loads(line) for line in (saved_run / "measurements.jsonl").read_text().splitlines()
    ]
    data, profiles = saved_run / "profile_data", saved_run / "profile_traces"
    data.mkdir()
    profiles.mkdir()
    shutil.copytree(saved_run / "source", data / "source")
    shutil.copyfile(saved_run / "source_manifest.json", data / "source_manifest.json")
    metadata = {
        key: reference[key]
        for key in (
            "config",
            "source_sha256",
            "workload_sha256",
            "native_provenance",
            "hardware",
            "checkpoint",
            "precision_settings",
        )
    }
    metadata.update(
        schema="nosa-motivation-profile-v2",
        status="validating",
        reference_run_id=reference["run_id"],
        methods=["hbm"],
        repeats=1,
    )
    metadata["cases"] = [{**reference["cases"][0], "sample": 0, "mode": "timeline"}]
    raw_rows = []
    shapes = {
        "qkv_proj": [12, 4],
        "o_proj": [4, 4],
        "gate_up_proj": [16, 4],
        "down_proj": [4, 8],
        "cis_projection": [1, 4],
    }
    for queries in (64, 2):
        work = matrix_flops(reference["model_config"], 0, queries, queries)
        for operation, shape in shapes.items():
            raw_rows.append(
                {
                    "operation": operation,
                    "queries": queries,
                    "layers": 32,
                    "weight_shape": shape,
                    "useful_flops": work[operation],
                    "dense_executed_flops": work[operation],
                    "samples_ms": [0.1],
                    "median_ms": 0.1,
                    "request_repetitions": 1,
                }
            )
    raw_useful = sum(row["useful_flops"] for row in raw_rows)
    raw_time = sum(row["median_ms"] for row in raw_rows)
    raw_path = data / "raw_matrix_api.json"
    write_json(
        raw_path,
        {
            "rows": raw_rows,
            "reference_dense_bf16_tflops": reference["config"]["peak_bf16_tflops"],
            "linear_repeats": 1,
            "bmm_repeats": 1,
            "equivalent_full_request_gemm_bmm_ms": raw_time,
            "useful_gemm_bmm_flops": raw_useful,
            "useful_mfu_pct": 100
            * raw_useful
            / (raw_time * reference["config"]["peak_bf16_tflops"] * 1e9),
        },
    )
    native = reference["native_provenance"]
    metadata["matrix_reference"] = {
        "file": raw_path.name,
        "sha256": digest(raw_path),
        "native_provenance": {
            "build_before": native["build_after"],
            "build_after": native["build_after"],
            "artifacts_before": native["artifacts_final"],
            "artifacts_after": native["artifacts_final"],
        },
    }
    records = []
    for request_id in (0, 2):
        row = next(
            row for row in rows if row["method"] == "hbm" and row["request_id"] == request_id
        )
        trace = {
            "traceEvents": [
                {
                    "cat": "user_annotation",
                    "ph": "X",
                    "name": "root",
                    "ts": 0,
                    "dur": 10000,
                    "tid": 1,
                    "args": {},
                },
            ]
        }
        scope_names = (
            "indexer",
            "attention",
            "linear",
            "linear",
            "linear",
            "linear",
            "cis_projection",
            "norm_helper",
            "norm_helper",
            "mlp_helper",
            "project_helper",
        )
        if reference["config"]["enable_compute_graphs"]:
            scope_names = ("indexer", "attention", "graph_project", "graph_finish")
        for index, name in enumerate(scope_names * 64, start=1):
            start = 10 * index
            trace["traceEvents"].extend(
                [
                    {
                        "cat": "user_annotation",
                        "ph": "X",
                        "name": f"matrix_api/{name}",
                        "ts": start,
                        "dur": 8,
                        "tid": 1,
                        "args": {},
                    },
                    {
                        "cat": "cuda_runtime",
                        "ph": "X",
                        "name": "cudaLaunchKernel",
                        "ts": start + 1,
                        "dur": 1,
                        "tid": 1,
                        "args": {"correlation": index},
                    },
                    {
                        "cat": "kernel",
                        "ph": "X",
                        "name": "fixture",
                        "ts": start + 3,
                        "dur": 2,
                        "tid": 2,
                        "args": {"correlation": index},
                    },
                ]
            )
        trace_path = profiles / f"{request_id}.json"
        write_json(trace_path, trace)
        hidden = torch.load(saved_run / row["output_file"], weights_only=True)["hidden"]
        output_path = data / f"{request_id}.pt"
        torch.save(hidden, output_path)
        record = {
            "method": "hbm",
            "request_id": request_id,
            "sample": 0,
            "mode": "timeline",
            "instrumented_runner_metrics": row,
            "compute_graphs": row.get("compute_graphs", {"before": None, "after": None}),
            "output_file": output_path.name,
            "output_sha256": digest(output_path),
            "numerical": row["numerical"],
            "root_name": "root",
            "trace_file": trace_path.name,
            "trace_sha256": digest(trace_path),
            "analysis": analyze_trace(trace, "root"),
            "api_analysis": analyze_api_trace(trace_path),
            "record_file": f"record{request_id}.json",
        }
        write_json(data / record["record_file"], record)
        records.append(record)
    metadata["records"] = records
    write_json(data / "metadata.json", metadata)
    return data, profiles, saved_run


def test_profile_validity_is_independent_of_performance_acceptance(saved_profile):
    from experiments.nosa_motivation.src.profile_audit import audit_profile

    data, profiles, reference = saved_profile
    gates = audit_profile(data, profiles, reference)
    assert gates["diagnostic_validity"] == "passed"
    assert gates["async_end_to_end_speedup"]["status"] == "failed"
    assert gates["async_every_sample_page_and_stripe_90pct"]["status"] == "pending"
    assert gates["formal_performance_acceptance"] == "pending"
    assert len(gates["matrix_api_evidence"]["request_comparisons"]) == 2
    assert gates["raw_matrix_api_mfu_proximity"] == "pending"


@pytest.mark.parametrize("saved_run", [True], indirect=True)
def test_profile_rejects_omitted_graph_shared_charge(saved_profile):
    from experiments.nosa_motivation.src.profile_audit import audit_profile

    data, profiles, reference = saved_profile
    metadata = json.loads((data / "metadata.json").read_text())
    metadata["cases"][0]["backend"]["shared_cache_bytes"]["hbm"] = 16
    write_json(data / "metadata.json", metadata)
    with pytest.raises(ValueError, match="base cache plus compute graphs"):
        audit_profile(data, profiles, reference)


def test_profile_recomputes_timeline_instead_of_trusting_duplicate_metrics(saved_profile):
    from experiments.nosa_motivation.src.profile_audit import audit_profile

    data, profiles, reference = saved_profile
    metadata = json.loads((data / "metadata.json").read_text())
    record = metadata["records"][0]
    record["analysis"]["gpu_activity_union_us"] = 99
    write_json(data / record["record_file"], record)
    write_json(data / "metadata.json", metadata)
    with pytest.raises(ValueError, match="timeline analysis"):
        audit_profile(data, profiles, reference)


def test_empty_profile_cannot_pass_diagnostic_validation(saved_profile):
    from experiments.nosa_motivation.src.profile_audit import audit_profile

    data, profiles, reference = saved_profile
    metadata = json.loads((data / "metadata.json").read_text())
    metadata.update(methods=[], cases=[], records=[])
    write_json(data / "metadata.json", metadata)
    with pytest.raises(ValueError, match="sample coverage"):
        audit_profile(data, profiles, reference)


@pytest.mark.parametrize("corruption", ["api_analysis", "raw_reference", "raw_binary"])
def test_profile_rechecks_independent_matrix_evidence(saved_profile, corruption):
    from experiments.nosa_motivation.src.profile_audit import audit_profile

    data, profiles, reference = saved_profile
    metadata = json.loads((data / "metadata.json").read_text())
    if corruption == "api_analysis":
        record = metadata["records"][0]
        record["api_analysis"]["api_execution_span_union_ms"] += 1
        write_json(data / record["record_file"], record)
    elif corruption == "raw_reference":
        (data / metadata["matrix_reference"]["file"]).write_text("{}")
    else:
        metadata["matrix_reference"]["native_provenance"]["artifacts_after"][
            "/fixture/cxldsagr.so"
        ]["sha256"] = "changed"
    write_json(data / "metadata.json", metadata)
    with pytest.raises(ValueError):
        audit_profile(data, profiles, reference)


def test_native_libraries_may_grow_between_method_warmups(saved_run):
    metadata = json.loads((saved_run / "metadata.json").read_text())
    next_library = {"size": 2, "sha256": "c" * 64}
    path = "/fixture/cxldsagr_offload_after_warmup.so"
    metadata["native_provenance"]["artifacts_final"][path] = next_library
    for case in metadata["cases"][1:]:
        case["native_artifacts_before"][path] = next_library
        case["native_artifacts_after"][path] = next_library
    write_json(saved_run / "metadata.json", metadata)
    assert audit_run(saved_run)[2]["passed"]
