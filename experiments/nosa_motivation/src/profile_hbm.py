"""Bounded full-checkpoint HBM diagnostic: wall latency, CPU profile and GPU gaps.

This one-history diagnostic is separate from the four-method request trace.
All source snapshots, timings and analysis go to output/data/<run_id>; the
Chrome trace goes to output/profile/<run_id>. Nothing here publishes a report.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import shutil
import statistics
import tempfile
import time
from bisect import bisect_right
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from experiments.nosa_motivation.src.config import EXPERIMENT
from experiments.nosa_motivation.src.flops import matrix_flops, mfu_percent, observe_selection
from experiments.nosa_motivation.src.matrix_baseline import invocation_geometry
from experiments.nosa_motivation.src.provenance import (
    checkpoint_inventory,
    loaded_native_artifacts,
    native_build_identity,
    runtime_environment,
    snapshot_sources,
    verify_native_artifacts,
    verify_native_build_identity,
    verify_source_snapshot,
    write_json,
)


def union_us(intervals):
    total, right = 0.0, float("-inf")
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative activity duration")
        total += max(0.0, end - max(start, right))
        right = max(right, end)
    return total


@contextmanager
def _output_directories(output, profile):
    created = []
    try:
        for directory in (output, profile):
            directory.mkdir(parents=True, exist_ok=False)
            created.append(directory)
        yield
    except BaseException:
        # Failed measurements cannot remain among experiment run artifacts.
        # Keep their diagnosis outside the repository; never remove a directory
        # that existed before this invocation.
        for index, directory in enumerate(created):
            if directory.resolve().is_relative_to(EXPERIMENT.parent.resolve()):
                destination = Path(tempfile.mkdtemp(prefix="nosa_hbm_failed_")) / str(index)
                shutil.move(str(directory), destination)
        raise


def analyze_trace(path):
    """Count actual activities, excluding overlapping GPU annotation parents."""
    events = json.loads(Path(path).read_text())["traceEvents"]
    activities = [
        event for event in events if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    if not activities:
        raise ValueError("trace has no GPU activities")
    launches = {
        event["args"]["correlation"]: event
        for event in events
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}
        and "correlation" in event.get("args", {})
    }
    scopes = [
        event
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith("matrix_api/")
    ]
    scopes_by_thread = defaultdict(list)
    for scope in scopes:
        scopes_by_thread[scope["tid"]].append(scope)
    scope_indexes = {}
    for tid, thread_scopes in scopes_by_thread.items():
        thread_scopes.sort(key=lambda scope: scope["ts"])
        maximum_end, prefix_ends = float("-inf"), []
        for scope in thread_scopes:
            maximum_end = max(maximum_end, scope["ts"] + scope["dur"])
            prefix_ends.append(maximum_end)
        scope_indexes[tid] = ([scope["ts"] for scope in thread_scopes], prefix_ends, thread_scopes)

    def owners_for(launch):
        index = scope_indexes.get(launch["tid"])
        if index is None:
            return []
        starts, ends, entries = index
        position = bisect_right(starts, launch["ts"]) - 1
        owners = []
        while position >= 0 and ends[position] > launch["ts"]:
            scope = entries[position]
            if scope["ts"] + scope["dur"] > launch["ts"]:
                owners.append(scope)
            position -= 1
        return sorted(owners, key=lambda scope: scope["dur"])

    owner_by_correlation = {
        correlation: owners_for(launch) for correlation, launch in launches.items()
    }
    by_kernel = defaultdict(lambda: {"count": 0, "gpu_ms": 0.0})
    by_api = defaultdict(list)
    by_scope = defaultdict(list)
    active = union_us((event["ts"], event["ts"] + event["dur"]) for event in activities)
    hull = max(event["ts"] + event["dur"] for event in activities) - min(
        event["ts"] for event in activities
    )
    right, late, missing = min(event["ts"] for event in activities), 0.0, 0
    for event in sorted(activities, key=lambda value: value["ts"]):
        entry = by_kernel[event["name"]]
        entry["count"] += 1
        entry["gpu_ms"] += event["dur"] / 1000
        launch = launches.get(event.get("args", {}).get("correlation"))
        if launch is None:
            missing += 1
        else:
            late += max(0.0, min(event["ts"], launch["ts"]) - right)
            owners = owner_by_correlation[event["args"]["correlation"]]
            interval = (event["ts"], event["ts"] + event["dur"])
            if owners:
                by_api[owners[0]["name"]].append(interval)
                for owner in owners:
                    by_scope[id(owner)].append(interval)
        right = max(right, event["ts"] + event["dur"])
    spans_by_name = defaultdict(list)
    for scope in scopes:
        values = by_scope.get(id(scope))
        if values:
            spans_by_name[scope["name"]].append(
                (min(begin for begin, _ in values), max(end for _, end in values))
            )
    return {
        "gpu_active_union_ms": active / 1000,
        "gpu_hull_ms": hull / 1000,
        "gpu_gap_ms": (hull - active) / 1000,
        "before_next_submission_ms": late / 1000,
        "uncorrelated_activities": missing,
        "kernel_count": sum(event["cat"] == "kernel" for event in activities),
        "copy_memset_count": sum(event["cat"] != "kernel" for event in activities),
        "api_gpu_ms": {name: union_us(values) / 1000 for name, values in by_api.items()},
        "api_scope_counts": {
            name: sum(scope["name"] == name for scope in scopes)
            for name in sorted({scope["name"] for scope in scopes})
        },
        "api_gpu_union_ms": union_us(interval for values in by_api.values() for interval in values)
        / 1000,
        "api_execution_span_union_ms": union_us(
            interval for values in spans_by_name.values() for interval in values
        )
        / 1000,
        "api_execution_span_ms": {
            name: union_us(values) / 1000 for name, values in spans_by_name.items()
        },
        "api_scopes_with_activity": {name: len(values) for name, values in spans_by_name.items()},
        "gpu_by_name": [
            {"name": name, **value}
            for name, value in sorted(
                by_kernel.items(), key=lambda item: item[1]["gpu_ms"], reverse=True
            )
        ],
        "boundary": "Separate instrumented capture; API helpers included; no unprofiled subtraction. Activity unions exclude idle gaps. Each API execution span runs from its first correlated device activity to its last completion and includes gaps inside that invocation; span unions avoid double counting nested or overlapping APIs. These GPU spans exclude host work before the first activity and are not API CPU/wall latency.",
    }


@contextmanager
def api_scopes(backend):
    import torch

    from models.nosa import scoring
    from models.nosa.attention import NosaFixedAttention, NosaSparseAttention
    from models.nosa.execution.fixed import NosaFixedServingBackend
    from models.nosa.indexer import NosaIndexer

    def wrapper(original, name):
        def call(*args, **kwargs):
            with torch.profiler.record_function(f"matrix_api/{name}"):
                return original(*args, **kwargs)

        return call

    with ExitStack() as stack:
        for owner, attribute, name in (
            (torch.nn.Linear, "forward", "linear"),
            (scoring, "cis_scores", "cis_projection"),
            (NosaIndexer, "__call__", "indexer"),
            (
                NosaFixedAttention
                if isinstance(backend, NosaFixedServingBackend)
                else NosaSparseAttention,
                "__call__",
                "attention",
            ),
        ):
            stack.enter_context(
                patch.object(owner, attribute, wrapper(getattr(owner, attribute), name))
            )
        if backend.compute_graphs is not None:
            for pair in backend.compute_graphs.pairs.values():
                for kind in ("project", "finish"):
                    graph = getattr(pair, f"{kind}_graph")
                    stack.enter_context(
                        patch.object(graph, "replay", wrapper(graph.replay, f"graph_{kind}"))
                    )
        else:
            for layer in backend.model.model.layers:
                for module, attribute, name in (
                    (layer.input_layernorm, "forward", "norm_helper"),
                    (layer.post_attention_layernorm, "forward", "norm_helper"),
                    (layer.mlp, "forward", "mlp_helper"),
                    (layer.self_attn, "project", "project_helper"),
                ):
                    stack.enter_context(
                        patch.object(module, attribute, wrapper(getattr(module, attribute), name))
                    )
        yield


def expected_api_scopes(backend, invocations=1):
    layers = backend.config.num_hidden_layers * invocations
    result = {"matrix_api/indexer": layers, "matrix_api/attention": layers}
    if backend.compute_graphs is not None:
        result.update({"matrix_api/graph_project": layers, "matrix_api/graph_finish": layers})
    else:
        result.update(
            {
                "matrix_api/linear": 4 * layers,
                "matrix_api/cis_projection": layers,
                "matrix_api/norm_helper": 2 * layers,
                "matrix_api/mlp_helper": layers,
                "matrix_api/project_helper": layers,
            }
        )
    return result


def profile_full_request(backend, history, candidate, output, profile, *, repeats=3):
    """Profile complete backend work; allocation/upload are outside its timer."""
    import torch

    capacity = (
        len(history) if hasattr(backend, "extend_candidate") else len(history) + len(candidate)
    )
    extend = getattr(backend, "extend_candidate", backend.extend)

    @contextmanager
    def fresh_session():
        session = backend.create_session(capacity)
        try:
            yield session
        finally:
            backend.release_session(session)

    def invoke(session):
        backend.prefill(session, history)
        return extend(session, candidate)

    samples = []
    for _ in range(repeats):
        with fresh_session() as session:
            backend.synchronize()
            start = time.perf_counter()
            expected = invoke(session)
            backend.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    cpu = cProfile.Profile()
    with fresh_session() as session:
        cpu.enable()
        invoke(session)
        backend.synchronize()
        cpu.disable()
    cpu.dump_stats(str(output / "request_cpu.pstats"))
    with (output / "request_cpu.txt").open("w") as stream:
        pstats.Stats(cpu, stream=stream).sort_stats("cumtime").print_stats(70)
    with fresh_session() as session:
        with (
            api_scopes(backend),
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ) as profiler,
            torch.profiler.record_function("nosa_full_backend_request"),
        ):
            actual = invoke(session)
            backend.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    trace = profile / "request.json"
    profiler.export_chrome_trace(str(trace))
    timeline = analyze_trace(trace)
    write_json(output / "request_timeline.json", timeline)
    work = matrix_flops(backend.config, 0, len(history), backend.chunk_size)
    candidate_work = matrix_flops(backend.config, len(history), len(candidate), len(candidate))
    work = {name: count + candidate_work[name] for name, count in work.items()}
    samples_median = statistics.median(samples)
    calls = len(invocation_geometry(len(history), len(candidate), backend.chunk_size))
    api_complete = (
        timeline["api_scope_counts"] == expected_api_scopes(backend, calls)
        and timeline["api_scopes_with_activity"] == expected_api_scopes(backend, calls)
        and timeline["uncorrelated_activities"] == 0
    )
    result = {
        "samples_ms": samples,
        "wall_median_ms": samples_median,
        "matrix_flops": work,
        "wall_mfu_pct": mfu_percent(work, samples_median),
        "api_mfu_pct": mfu_percent(work, timeline["api_execution_span_union_ms"])
        if api_complete
        else None,
        "api_activity_only_mfu_pct": mfu_percent(work, timeline["api_gpu_union_ms"])
        if api_complete
        else None,
        "complete_api_attribution": api_complete,
        "exact_candidate_replay": True,
        "boundary": "Complete backend prefill + candidate + synchronization on a fresh empty "
        "session. Session allocation and input upload precede timing; not runner admission/eviction.",
    }
    write_json(output / "request_summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/ssd-wlcb/chenkaiqi/NOSA-8B"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--history-tokens", type=int, default=65536)
    parser.add_argument("--candidate-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("fixed", "generic"), default="fixed")
    parser.add_argument("--compute-graphs", action="store_true")
    parser.add_argument("--full-request", action="store_true")
    parser.add_argument("--raw-matrix-reference", action="store_true")
    parser.add_argument("--workload", choices=("random", "gr"), default="random")
    parser.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="Keep only a diagnostic when source changes during the run",
    )
    args = parser.parse_args(argv)
    if min(args.history_tokens, args.candidate_tokens, args.chunk_size, args.repeats) <= 0:
        parser.error("lengths, chunk size and repeats must be positive")
    if args.history_tokens + args.candidate_tokens > 262144:
        parser.error("history plus candidate exceeds context limit")
    import torch

    from cache.prefix_pool import CacheFootprint
    from models.nosa.execution.adapter import NosaServingBackend
    from models.nosa.execution.fixed import NosaFixedServingBackend
    from models.nosa.indexer import NosaIndexer

    output = args.output_dir or EXPERIMENT / "output/data" / args.run_id
    profile = args.profile_dir or EXPERIMENT / "output/profile" / args.run_id
    if args.allow_source_drift and any(
        path.resolve().is_relative_to(EXPERIMENT.parent.resolve()) for path in (output, profile)
    ):
        parser.error("--allow-source-drift requires both output directories outside experiments")
    with _output_directories(output, profile):
        source_hash = snapshot_sources(output)
        native_identity = native_build_identity()
        write_json(output / "native_build_before.json", native_identity)
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        device = torch.device("cuda:0")
        options = {
            "scheme": "hbm",
            "device": device,
            "chunk_size": args.chunk_size,
            "max_seq_len": args.history_tokens + args.candidate_tokens,
        }
        cls = NosaServingBackend
        budgets = CacheFootprint(1 << 37, 1 << 39)
        if args.backend == "fixed":
            cls, budgets = NosaFixedServingBackend, None
            options.update(sparse_pool_tokens=args.history_tokens, host_arena_tokens=16777216)
        elif args.compute_graphs:
            parser.error("compute graphs require --backend fixed")
        backend = cls.from_pretrained(args.model_path, **options)
        if args.compute_graphs:
            backend.enable_compute_graphs(
                [
                    queries
                    for _, queries in invocation_geometry(
                        args.history_tokens, args.candidate_tokens, args.chunk_size
                    )
                ]
            )
        backend.allocate_shared(
            backend.plan_resources(
                budgets,
                {
                    "max_session_capacity": args.history_tokens + args.candidate_tokens,
                    "max_candidate_tokens": args.candidate_tokens,
                    "max_history_tokens": args.history_tokens,
                },
            )
        )
        if args.workload == "gr":
            from GR.workload import WorkloadConfig, build_workload

            workload = build_workload(
                WorkloadConfig(
                    model="nosa",
                    num_users=16,
                    requests=32,
                    history_tokens=args.history_tokens,
                    candidate_tokens=args.candidate_tokens,
                    seed=args.seed,
                    sampling="sequential",
                    context_limit=args.history_tokens + args.candidate_tokens,
                ),
                tokenizer=args.model_path,
            )
            ids = torch.tensor(workload.requests[0]["input_ids"], device=device, dtype=torch.long)
            write_json(output / "gr_request.json", workload.requests[0])
            write_json(output / "gr_workload_manifest.json", workload.manifest)
        else:
            ids = torch.randint(
                0,
                backend.config.vocab_size,
                (args.history_tokens + args.candidate_tokens,),
                device=device,
            )
        history, candidate = ids[: args.history_tokens], ids[args.history_tokens :]
        capacity = len(history) if args.backend == "fixed" else len(ids)
        session = backend.create_session(capacity)
        extend = backend.extend_candidate if args.backend == "fixed" else backend.extend

        def restore():
            if args.backend == "generic":
                backend.truncate(session, len(history))
            else:
                backend.synchronize()
                if session.length != len(history):
                    raise AssertionError("candidate changed retained history")

        try:
            backend.prefill(session, history)
            extend(session, candidate)
            restore()
            native_artifacts = loaded_native_artifacts()
            write_json(output / "native_artifacts_after_warmup.json", native_artifacts)
            samples = []
            for _ in range(args.repeats):
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                backend.synchronize()
                start = time.perf_counter()
                begin.record()
                expected = extend(session, candidate)
                end.record()
                submitted = time.perf_counter()
                backend.synchronize()
                samples.append(
                    {
                        "wall_ms": (time.perf_counter() - start) * 1000,
                        "host_submit_ms": (submitted - start) * 1000,
                        "cuda_span_ms": begin.elapsed_time(end),
                    }
                )
                restore()
            write_json(output / "candidate_samples.json", samples)
            cpu = cProfile.Profile()
            cpu.enable()
            extend(session, candidate)
            backend.synchronize()
            cpu.disable()
            cpu.dump_stats(str(output / "candidate_cpu.pstats"))
            with (output / "candidate_cpu.txt").open("w") as stream:
                pstats.Stats(cpu, stream=stream).sort_stats("cumtime").print_stats(60)
            restore()
            with (
                api_scopes(backend),
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                ) as profiler,
                torch.profiler.record_function("nosa_hbm_candidate"),
            ):
                actual = extend(session, candidate)
                backend.synchronize()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            trace = profile / "candidate.json"
            profiler.export_chrome_trace(str(trace))
            timeline = analyze_trace(trace)
            write_json(output / "candidate_timeline.json", timeline)
            restore()
            selections = []
            original = NosaIndexer.__call__

            def observed(indexer, q, cache, context):
                selection = original(indexer, q, cache, context)
                selections.append(
                    {
                        "layer": context.layer_idx,
                        **observe_selection(
                            selection, query_start=context.query_start, query_heads=q.shape[1]
                        ),
                    }
                )
                return selection

            with patch.object(NosaIndexer, "__call__", observed):
                actual = extend(session, candidate)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            if len(selections) != backend.config.num_hidden_layers:
                raise AssertionError("did not observe every model layer")
            write_json(output / "candidate_selections.json", selections)
            restore()
            backend.release_session(session)
            session = backend.create_session(capacity)
            backend.synchronize()
            start = time.perf_counter()
            backend.prefill(session, history)
            backend.synchronize()
            prefix_ms = (time.perf_counter() - start) * 1000
            request_summary = None
            if args.full_request:
                backend.release_session(session)
                session = None
                request_summary = profile_full_request(backend, history, candidate, output, profile)
            work = matrix_flops(backend.config, len(history), len(candidate), len(candidate))
            wall = statistics.median(row["wall_ms"] for row in samples)
            api_complete = (
                timeline["api_scope_counts"] == expected_api_scopes(backend)
                and timeline["api_scopes_with_activity"] == expected_api_scopes(backend)
                and timeline["uncorrelated_activities"] == 0
            )
            summary = {
                "run_id": args.run_id,
                "source_manifest_sha256": source_hash,
                "runtime": runtime_environment(torch, device),
                "checkpoint": checkpoint_inventory(args.model_path),
                "backend": backend.describe(),
                "history_tokens": len(history),
                "candidate_tokens": len(candidate),
                "seed": args.seed,
                "repeats": args.repeats,
                "candidate_wall_median_ms": wall,
                "prefix_wall_single_ms": prefix_ms,
                "candidate_matrix_flops": work,
                "dense_bf16_reference_tflops": 989.5,
                "reference_peak_boundary": "H200 SXM dense BF16 specification; not achieved-clock normalization",
                "candidate_mfu_pct": mfu_percent(work, wall),
                "candidate_api_mfu_pct": (
                    mfu_percent(work, timeline["api_execution_span_union_ms"])
                    if api_complete
                    else None
                ),
                "candidate_api_activity_only_mfu_pct": (
                    mfu_percent(work, timeline["api_gpu_union_ms"]) if api_complete else None
                ),
                "complete_api_attribution": api_complete,
                "api_boundary": "Complete compute APIs including graph norm/RoPE/activation helpers, indexer and attention helpers; input staging copies are outside API scopes.",
                "numerical_replays_exact": True,
                "selection_layers_verified": len(selections),
                "workload": args.workload,
                "full_backend_request": request_summary,
                "boundary": "Full 32-layer HBM backend; no LM head; random or first generated GR request as declared, not the complete multi-user trace.",
            }
        finally:
            if session is not None:
                backend.release_session(session)
            backend.close()
        if args.raw_matrix_reference:
            from experiments.nosa_motivation.src.matrix_baseline import benchmark_matrix_apis

            reference = benchmark_matrix_apis(
                backend.model, len(history), len(candidate), args.chunk_size
            )
            write_json(output / "raw_matrix_api.json", reference)
            summary["raw_matrix_api_reference"] = {
                name: value for name, value in reference.items() if name != "rows"
            }
        verify_source_snapshot(output)
        try:
            verify_source_snapshot(output, check_current=True)
            verify_native_build_identity(native_identity)
            final_artifacts = verify_native_artifacts(native_artifacts)
            write_json(output / "native_artifacts_final.json", final_artifacts)
            summary["runtime_source_unchanged"] = True
        except ValueError as error:
            if not args.allow_source_drift:
                raise
            summary["runtime_source_unchanged"] = False
            summary["source_drift_diagnostic"] = str(error)
        write_json(output / "summary.json", summary)
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key.endswith(("_ms", "_pct"))},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
