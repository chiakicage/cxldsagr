"""Independent raw QK/PV and public FA3 reference for an accepted NOSA run.

Use SSD-backed TMPDIR for operand capture. This standalone replay does not
modify the accepted measurement or its existing timeline profile.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from cache.allocator.snapshot import runtime_info as allocator_snapshot_runtime_info
from evaluation import pool_scan_provenance as pool_scan
from experiments.nosa_motivation.src.attention_reference import (
    benchmark_attention_reference,
    capture_attention_inputs,
)
from experiments.nosa_motivation.src.attention_reference_audit import (
    audit_attention_timings,
    audit_capture,
    audit_history_reuse,
    evidence_path,
    require,
    tensor_digest,
)
from experiments.nosa_motivation.src.config import EXPERIMENT, resource_limits
from experiments.nosa_motivation.src.cpu_environment import (
    finish_cpu_environment,
    require_matching_cpu_environment,
    validate_cpu_environment_record,
)
from experiments.nosa_motivation.src.flops import model_dimensions
from experiments.nosa_motivation.src.matrix_comparison import (
    audit_matrix_native_provenance,
    audit_matrix_reference,
)
from experiments.nosa_motivation.src.measure import (
    backend_factory,
    check_graph_replays,
    check_request,
    precision_settings,
    validate_warmup,
    warmup,
)
from experiments.nosa_motivation.src.profile import compare_replay, runtime_sources
from experiments.nosa_motivation.src.profile_audit import audit_profile
from experiments.nosa_motivation.src.provenance import (
    audit_native_identity,
    checkpoint_inventory,
    digest,
    loaded_native_artifacts,
    native_build_identity,
    publish_directories,
    runtime_environment,
    snapshot_sources,
    verify_native_artifacts,
    verify_native_build_identity,
    verify_source_snapshot,
    write_json,
)
from experiments.nosa_motivation.src.report import audit_run, read_jsonl


def combined_comparison(row, matrix, attention, *, model_config=None, config=None):
    if "effective_work" in row:
        work = row["effective_work"]["request_flops"]
        peak = row["effective_work"]["peak_bf16_tflops"]
    else:
        from experiments.nosa_motivation.src.matrix_comparison import _work

        work = _work(model_config, config, cached=row["prefix_cache_hit"])
        peak = config["peak_bf16_tflops"]
    total = sum(work.values())
    require(
        matrix["useful_gemm_bmm_flops"] + attention["useful_attention_flops"] == total,
        "independent references do not cover complete request work",
    )
    wall = row.get("latency_ms")
    raw = matrix["equivalent_gemm_bmm_ms"] + attention["raw_qk_pv_cuda_ms"]
    fa3 = matrix["equivalent_gemm_bmm_ms"] + attention["independent_fa3_cuda_ms"]
    return {
        "request_id": row["request_id"],
        "accepted_runner_wall_ms": wall,
        "full_useful_matrix_flops": total,
        "raw_useful_flop_coverage": 1.0,
        "accepted_runner_mfu_pct": None if wall is None else 100 * total / (wall * peak * 1e9),
        "raw_materialized_reference": {
            "summed_api_ms": raw,
            "full_work_mfu_pct": 100 * total / (raw * peak * 1e9),
            "wall_to_reference_mfu_ratio": None if wall is None else raw / wall,
        },
        "independent_fa3_composition": {
            "summed_api_ms": fa3,
            "full_work_mfu_pct": 100 * total / (fa3 * peak * 1e9),
            "wall_to_reference_mfu_ratio": None if wall is None else fa3 / wall,
        },
        "lowest_tested_reference_sum_ms": min(raw, fa3),
        "efficiency_gate": "requires interpretation; beating the slower materialized reference alone is insufficient",
        "boundary": "All useful matrix FLOPs are covered, but both references sum independently measured API medians and are not observed requests. Raw QK/PV duplicates selected KV per query and materializes scores. FA3 composition preserves the optimized sparse access path and includes attention helpers. The matrix-profile reference was measured in its own identified run. Both outcomes remain visible; no reference is selected because it creates an easier pass.",
    }


def audit_capture_native(
    metadata, reference, *, require_pool_referrers=False, expected_abi_sha256=None
):
    native = metadata["native_provenance"]
    require(
        native["artifacts_before"] == native["artifacts_final"]
        and all(
            reference["native_provenance"]["artifacts_final"].get(path) == value
            for path, value in native["artifacts_before"].items()
        ),
        "capture native libraries differ from accepted replay or changed during capture",
    )
    native_id = audit_native_identity(
        {
            "native_provenance": native,
            "cases": [
                {
                    "native_artifacts_before": native["artifacts_before"],
                    "native_artifacts_after": native["artifacts_final"],
                    "token_validation": metadata["capture_token_validation"],
                    **(
                        {"pool_referrers": metadata["capture_pool_referrers"]}
                        if "pool_referrers_build_info" in native["build_before"]
                        else {}
                    ),
                    **(
                        {"allocator_snapshot": metadata["capture_allocator_snapshot"]}
                        if "allocator_snapshot_build_info" in native["build_before"]
                        else {}
                    ),
                }
            ],
        },
        require_pool_referrers=require_pool_referrers,
        expected_abi_sha256=expected_abi_sha256,
    )
    require(
        native["build_before"] == reference["native_provenance"]["build_after"],
        "capture native build differs from accepted measurement",
    )
    return native_id


def audit_reference(data, reference_dir, matrix_profile_run, matrix_profile_dir):
    from experiments.nosa_motivation.src.config import BENCH_SCHEMA
    from experiments.nosa_motivation.src.measure import RECEIPT_KIND
    from experiments.nosa_motivation.src.validation import reference_directory

    reference_dir = reference_directory(reference_dir, bench_schema=BENCH_SCHEMA, kind=RECEIPT_KIND)
    """Reopen complete reference evidence without initializing or using a GPU."""
    import torch

    data, reference_dir = Path(data), Path(reference_dir)
    matrix_profile_run = Path(matrix_profile_run)
    metadata = json.loads((data / "metadata.json").read_text())
    require(
        metadata["schema"] == "nosa-independent-attention-v1"
        and metadata["status"] in ("validating", "diagnostic_valid"),
        "attention reference is incomplete",
    )
    reference, rows, accepted = audit_run(reference_dir)
    profile_audit = audit_profile(matrix_profile_run, matrix_profile_dir, reference_dir)
    profile = json.loads((matrix_profile_run / "metadata.json").read_text())
    require(
        metadata["reference_run_id"] == reference["run_id"]
        and metadata["reference_metadata_sha256"] == digest(reference_dir / "metadata.json")
        and metadata["reference_source_sha256"] == reference["source_sha256"]
        and metadata["reference_audit"] == accepted
        and metadata["matrix_profile_run_id"] == profile["run_id"]
        and metadata["matrix_profile_metadata_sha256"]
        == digest(matrix_profile_run / "metadata.json")
        and metadata["matrix_profile_audit"] == profile_audit,
        "accepted measurement/profile identity differs",
    )
    for name in ("config", "model_config", "checkpoint", "precision_settings", "workload_sha256"):
        require(metadata[name] == reference[name], f"attention reference {name} differs")
    validate_cpu_environment_record(metadata)
    require_matching_cpu_environment(metadata["hardware"], reference["hardware"])
    for name in (
        "uuid",
        "packages",
        "torch_cuda",
        "compute_capability",
        "total_memory",
        "sm_count",
    ):
        require(
            metadata["hardware"][name] == reference["hardware"][name],
            f"attention hardware {name} differs",
        )
    require(
        verify_source_snapshot(data) == metadata["source_sha256"]
        and runtime_sources(json.loads((data / "source_manifest.json").read_text()))
        == runtime_sources(json.loads((reference_dir / "source_manifest.json").read_text())),
        "attention runtime/orchestration source differs",
    )
    saved_sources = json.loads((data / "source_manifest.json").read_text())
    native_id = audit_capture_native(
        metadata,
        reference,
        require_pool_referrers=pool_scan.requires_provenance(saved_sources),
        expected_abi_sha256=pool_scan.source_abi_sha256(saved_sources),
    )
    native_audit = audit_matrix_native_provenance(
        metadata["benchmark_native_provenance"], metadata["native_provenance"]
    )
    require(native_audit == metadata["benchmark_native_audit"], "benchmark native audit differs")
    raw_path = data / "matrix_reference.json"
    require(
        digest(raw_path)
        == metadata["matrix_reference_sha256"]
        == profile["matrix_reference"]["sha256"],
        "copied raw matrix reference differs",
    )
    matrix = audit_matrix_reference(
        json.loads(raw_path.read_text()), reference["model_config"], reference["config"]
    )
    config = reference["config"]
    workload = read_jsonl(reference_dir / "workload/requests.jsonl")
    validate_warmup("hbm", metadata["capture_warmup"], config, workload)
    keyed = {(row["method"], row["request_id"]): row for row in rows}
    expected_ids = (0, config["num_users"])
    require(
        set(metadata["captures"]) == {str(i) for i in expected_ids},
        "incomplete attention request coverage",
    )
    require(
        [row["request_id"] for row in metadata["replay_requests"]]
        == list(range(config["num_users"] + 1)),
        "capture replay request coverage differs",
    )
    for record in metadata["replay_requests"]:
        index = record["request_id"]
        check_request(workload[index], {**record["metrics"], "method": "hbm"}, config)
        require(
            record["metrics"]["prefix_cache_hit"] == keyed[("hbm", index)]["prefix_cache_hit"],
            "capture replay history residency differs",
        )
        expected = torch.load(
            reference_dir / "numerical/hbm" / f"{index:06d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        require(
            record["output_sha256"] == tensor_digest(expected["hidden"]),
            "capture replay output identity differs",
        )
    captures, summaries, comparisons, reuse = {}, {}, [], None
    for index in expected_ids:
        entry = metadata["captures"][str(index)]
        directory = evidence_path(data, entry["directory"])
        manifest_path = directory / "manifest.json"
        require(digest(manifest_path) == entry["manifest_sha256"], "capture manifest changed")
        manifest = json.loads(manifest_path.read_text())
        manifest["manifest_sha256"] = entry["manifest_sha256"]
        require(
            manifest["request_id"] == index
            and manifest["config"] == config
            and manifest["model_config"] == model_dimensions(reference["model_config"])
            and manifest["mode"] == ("full_request" if index == 0 else "candidate_only"),
            "capture request contract differs",
        )
        expected_identity = {
            "reference_run_id": reference["run_id"],
            "reference_source_sha256": reference["source_sha256"],
            "runtime_source_sha256": metadata["source_sha256"],
            "native_build_sha256": native_id,
            "device_uuid": reference["hardware"]["uuid"],
            "workload_sha256": reference["workload_sha256"],
            "input_sha256": workload[index]["input_sha256"],
        }
        require(manifest["identity"] == expected_identity, "capture operand identity differs")
        require(
            manifest["accepted_output_sha256"]
            == metadata["replay_requests"][index]["output_sha256"],
            "capture request output differs from accepted replay",
        )
        require(audit_capture(directory, manifest) == manifest["audit"], "capture audit differs")
        benchmark_path = evidence_path(data, entry["benchmark_file"])
        require(
            digest(benchmark_path) == entry["benchmark_sha256"], "attention timing file changed"
        )
        summaries[index] = audit_attention_timings(manifest, json.loads(benchmark_path.read_text()))
        captures[index] = (directory, manifest)
        cached = keyed[("hbm", index)]["prefix_cache_hit"]
        if index and not cached:
            reuse = audit_history_reuse(*captures[0], directory, manifest)
            attention = combine_reused_history(summaries[0], summaries[index])
        else:
            require(
                manifest["history_reference_sha256"] is None,
                "cached request must not reuse prefix timings",
            )
            attention = summaries[index]["captured_workload"]
        comparisons.append(
            combined_comparison(
                keyed[("hbm", index)],
                matrix["candidate_only" if cached else "full_request"],
                attention,
                model_config=reference["model_config"],
                config=reference["config"],
            )
        )
    require(
        comparisons == metadata["comparisons"], "combined attention/reference arithmetic differs"
    )
    return {
        "passed": True,
        "captured_requests": list(expected_ids),
        "attention_calls": sum(len(manifest["calls"]) for _, manifest in captures.values()),
        "native_build_sha256": native_id,
        "repeated_history": reuse,
        "full_useful_flop_coverage": True,
        "boundary": "Saved operand bytes, identities, numerical observations and timing arithmetic reopened on CPU. GPU operations are not rerun. Both API compositions remain distinct from observed request latency; no efficiency acceptance is inferred.",
    }


def combine_reused_history(first_summary, revisit_summary):
    return {
        name: first_summary["captured_workload"][name]
        - first_summary["candidate_only"][name]
        + revisit_summary["candidate_only"][name]
        for name in first_summary["captured_workload"]
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--audit-data", type=Path)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--matrix-profile-run", type=Path, required=True)
    parser.add_argument("--matrix-profile-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT / "output")
    args = parser.parse_args(argv)
    from experiments.nosa_motivation.src.config import BENCH_SCHEMA
    from experiments.nosa_motivation.src.measure import RECEIPT_KIND
    from experiments.nosa_motivation.src.validation import reference_directory

    args.reference_run = reference_directory(
        args.reference_run, bench_schema=BENCH_SCHEMA, kind=RECEIPT_KIND
    )
    if args.audit_data is not None:
        print(
            json.dumps(
                audit_reference(
                    args.audit_data,
                    args.reference_run,
                    args.matrix_profile_run,
                    args.matrix_profile_dir,
                ),
                indent=2,
            )
        )
        return
    require(
        args.run_id is not None and re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) is not None,
        "unsafe attention reference run ID",
    )
    destination = args.output_root / "data" / args.run_id
    require(not destination.exists(), "attention reference run already exists")
    reference, rows, accepted = audit_run(args.reference_run)
    profile_audit = audit_profile(
        args.matrix_profile_run, args.matrix_profile_dir, args.reference_run
    )
    profile = json.loads((args.matrix_profile_run / "metadata.json").read_text())
    raw = json.loads((args.matrix_profile_run / profile["matrix_reference"]["file"]).read_text())
    matrix = audit_matrix_reference(raw, reference["model_config"], reference["config"])
    staging = Path(tempfile.mkdtemp(prefix=f"nosa-attention-reference-{args.run_id}-"))
    data = staging / "data"
    data.mkdir()
    metadata = {
        "schema": "nosa-independent-attention-v1",
        "run_id": args.run_id,
        "status": "running",
    }
    try:
        import torch

        from models.nosa.model import NosaForCausalLM
        from serving.persistent import PersistentGRRunner

        config = reference["config"]
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
        source = snapshot_sources(data)
        saved = json.loads((data / "source_manifest.json").read_text())
        original = json.loads((args.reference_run / "source_manifest.json").read_text())
        require(
            runtime_sources(saved) == runtime_sources(original),
            "attention reference runtime/orchestration differs",
        )
        checkpoint = Path(reference["checkpoint"]["path"])
        require(
            checkpoint_inventory(checkpoint) == reference["checkpoint"],
            "attention checkpoint differs",
        )
        hardware = runtime_environment(torch, device)
        require_matching_cpu_environment(hardware, reference["hardware"])
        require(
            hardware["uuid"] == reference["hardware"]["uuid"],
            "attention reference must use accepted GPU UUID",
        )
        require(
            precision_settings(torch) == reference["precision_settings"],
            "attention precision differs",
        )
        build = native_build_identity()
        require(
            build == reference["native_provenance"]["build_after"], "attention native build differs"
        )
        metadata.update(
            reference_run_id=reference["run_id"],
            reference_metadata_sha256=digest(args.reference_run / "metadata.json"),
            reference_audit=accepted,
            source_sha256=source,
            reference_source_sha256=reference["source_sha256"],
            hardware=hardware,
            checkpoint=reference["checkpoint"],
            precision_settings=precision_settings(torch),
            workload_sha256=reference["workload_sha256"],
            model_config=reference["model_config"],
            config=config,
            matrix_profile_run_id=profile["run_id"],
            matrix_profile_metadata_sha256=digest(args.matrix_profile_run / "metadata.json"),
            matrix_profile_audit=profile_audit,
            matrix_reference_sha256=profile["matrix_reference"]["sha256"],
            native_provenance={"build_before": build},
        )
        shutil.copyfile(
            args.matrix_profile_run / profile["matrix_reference"]["file"],
            data / "matrix_reference.json",
        )
        require(
            digest(data / "matrix_reference.json") == metadata["matrix_reference_sha256"],
            "matrix reference changed after audit",
        )
        workload = read_jsonl(args.reference_run / "workload/requests.jsonl")
        keyed = {(row["method"], row["request_id"]): row for row in rows}
        model = NosaForCausalLM.from_pretrained(
            checkpoint,
            device=device,
            dtype=torch.bfloat16,
            attention_mode="sparse",
            sparse_backend="auto",
        )
        model.config = replace(
            model.config,
            max_position_embeddings=config["history_tokens"] + config["candidate_tokens"],
        )
        metadata["capture_warmup"] = warmup(
            backend_factory(model, "hbm", config),
            "hbm",
            SimpleNamespace(requests=workload),
            config,
            partial(PersistentGRRunner, native_token_validation=True),
        )
        before_capture = loaded_native_artifacts()
        pool_scan_before = pool_scan.snapshot()
        require(
            all(
                reference["native_provenance"]["artifacts_final"].get(path) == value
                for path, value in before_capture.items()
            ),
            "warmed capture native artifacts differ from accepted measurement",
        )
        metadata["native_provenance"]["artifacts_before"] = before_capture
        metadata["replay_requests"] = []
        backend = backend_factory(model, "hbm", config)
        selected = {0: "full_request", config["num_users"]: "candidate_only"}
        captures = {}
        try:
            with PersistentGRRunner(
                backend, resource_limits=resource_limits(config), native_token_validation=True
            ) as runner:
                metadata["capture_token_validation"] = runner.token_validation_identity
                metadata["capture_allocator_snapshot"] = allocator_snapshot_runtime_info()
                for request in workload[: config["num_users"] + 1]:
                    index = request["request_id"]
                    before = backend.describe().get("compute_graphs")
                    if index in selected:
                        directory = data / f"capture_{index:06d}"
                        identity = {
                            "reference_run_id": reference["run_id"],
                            "reference_source_sha256": reference["source_sha256"],
                            "runtime_source_sha256": source,
                            "native_build_sha256": build["sha256"],
                            "device_uuid": hardware["uuid"],
                            "workload_sha256": reference["workload_sha256"],
                            "input_sha256": request["input_sha256"],
                        }
                        expected = torch.load(
                            args.reference_run / "numerical/hbm" / f"{index:06d}.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                        with capture_attention_inputs(
                            backend,
                            directory,
                            config,
                            index,
                            identity,
                            mode=selected[index],
                            history_reference=(
                                captures.get(0)
                                if index and not keyed[("hbm", index)]["prefix_cache_hit"]
                                else None
                            ),
                        ) as capture:
                            result = runner.execute(request)
                            capture.accept_output(result.hidden, expected["hidden"])
                        captures[index] = directory
                    else:
                        result = runner.execute(request)
                    compare_replay(result, request, "hbm", config, args.reference_run)
                    check_request(request, {**result.metrics, "method": "hbm"}, config)
                    check_graph_replays(
                        before, backend.describe().get("compute_graphs"), config, result.metrics
                    )
                    require(
                        result.metrics["prefix_cache_hit"]
                        == keyed[("hbm", index)]["prefix_cache_hit"],
                        "capture replay history residency differs from accepted measurement",
                    )
                    metadata["replay_requests"].append(
                        {
                            "request_id": index,
                            "metrics": dict(result.metrics),
                            "output_sha256": tensor_digest(result.hidden.detach().cpu()),
                        }
                    )
                    del result
        finally:
            backend.close()
        metadata["capture_pool_referrers"] = pool_scan.finish_case(pool_scan_before)
        capture_artifacts = verify_native_artifacts(before_capture)
        metadata["native_provenance"].update(
            build_after=verify_native_build_identity(build), artifacts_final=capture_artifacts
        )
        audit_capture_native(metadata, reference)
        comparisons, benchmarks = [], {}
        for index, directory in captures.items():
            output = data / f"attention_reference_{index:06d}.json"
            result = benchmark_attention_reference(directory, output, device=device)
            manifest = json.loads((directory / "manifest.json").read_text())
            manifest["manifest_sha256"] = digest(directory / "manifest.json")
            audit_capture(directory, manifest)
            summaries = audit_attention_timings(manifest, result)
            cached = keyed[("hbm", index)]["prefix_cache_hit"]
            if index and not cached:
                first = json.loads((captures[0] / "manifest.json").read_text())
                audit_history_reuse(captures[0], first, directory, manifest)
                attention = combine_reused_history(benchmarks[0]["summary"], summaries)
            else:
                attention = summaries["captured_workload"]
            comparisons.append(
                combined_comparison(
                    keyed[("hbm", index)],
                    matrix["candidate_only" if cached else "full_request"],
                    attention,
                    model_config=reference["model_config"],
                    config=config,
                )
            )
            benchmarks[index] = result
        raw_native = {
            "build_before": build,
            "build_after": verify_native_build_identity(build),
            "artifacts_before": capture_artifacts,
            "artifacts_after": verify_native_artifacts(capture_artifacts, allow_additions=True),
        }
        metadata["benchmark_native_provenance"] = raw_native
        metadata["benchmark_native_audit"] = audit_matrix_native_provenance(
            raw_native, metadata["native_provenance"]
        )
        verify_source_snapshot(data, check_current=True)
        require(
            checkpoint_inventory(checkpoint) == reference["checkpoint"]
            and precision_settings(torch) == reference["precision_settings"],
            "attention environment changed during replay",
        )
        finish_cpu_environment(metadata, torch)
        metadata.update(
            status="validating",
            comparisons=comparisons,
            captures={
                str(i): {
                    "directory": path.name,
                    "manifest_sha256": digest(path / "manifest.json"),
                    "benchmark_file": f"attention_reference_{i:06d}.json",
                    "benchmark_sha256": digest(data / f"attention_reference_{i:06d}.json"),
                }
                for i, path in captures.items()
            },
            efficiency_gate="pending interpretation; full raw FLOP coverage is not by itself efficiency acceptance",
        )
        write_json(data / "metadata.json", metadata)
        metadata["acceptance"] = audit_reference(
            data, args.reference_run, args.matrix_profile_run, args.matrix_profile_dir
        )
        metadata["status"] = "diagnostic_valid"
        write_json(data / "metadata.json", metadata)
        publish_directories({"data": data}, {"data": destination})
        shutil.rmtree(staging)
        print(f"Independent attention reference: {destination}", flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        write_json(data / "metadata.json", metadata)
        print(
            f"Attention-reference diagnostics retained outside experiments: {staging}", flush=True
        )
        raise


if __name__ == "__main__":
    main()
