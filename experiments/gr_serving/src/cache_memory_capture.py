"""Separate real-checkpoint 10-block GR cache-memory validation, never latency.

Uses the complete 16-user sequential two-pass 64K+128 trace. Profiles requests
0, 15, 16 and 31; byte/page LRU still determines actual prefix hits. The fixed
NH and P do not imply that every chunk choice can retain all sixteen sessions.
"""

from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint
from experiments.gr_serving.src.cache_memory_audit import (
    CacheMemoryAudit,
    host_allocator_counters,
    storage_inventory,
)
from experiments.gr_serving.src.measure import (
    backend_provenance,
    numerical_comparison,
    source_snapshot,
    verify_source_snapshot,
)
from experiments.gr_serving.src.workload import WorkloadConfig, build_workload
from models.deepseek_v32.echo_model import Config
from models.deepseek_v32.serving_backend import SCHEMES, DeepSeekServingBackend
from serving.persistent import PersistentGRRunner

CHUNKS = (256, 512, 1024, 2048)
PROFILE_REQUESTS = (0, 15, 16, 31)
HISTORY, CANDIDATES, USERS, CAPACITY = 65536, 128, 16, 65664
HOST_TOKENS, POOL_TOKENS = USERS * CAPACITY, 32768
HBM_BUDGET, DRAM_BUDGET = 4 * 2**30, 64 * 2**30


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def selected_schemes(schemes):
    schemes = tuple(schemes)
    if "hbm" not in schemes or len(set(schemes)) != len(schemes) or set(schemes) - set(SCHEMES):
        raise ValueError("schemes must be a distinct supported subset containing hbm reference")
    return tuple(scheme for scheme in SCHEMES if scheme in schemes)


def validate_completed_matrix(cases, expected, *, requests):
    observed = [{"chunk": case["chunk"], "scheme": case["scheme"]} for case in cases]
    if observed != expected or any(case["requests"] != requests for case in cases):
        raise RuntimeError("completed cases do not match the declared full case/request matrix")


def screen_cases(cfg, chunks=CHUNKS, *, workspace_query_tokens=None, schemes=SCHEMES):
    """Run the real admission planner without weights, CUDA or pool allocation."""
    rows = []
    if workspace_query_tokens is not None and workspace_query_tokens < max(chunks):
        raise ValueError("workspace bound must cover every requested chunk")
    for chunk in chunks:
        for scheme in selected_schemes(schemes):
            probe = DeepSeekServingBackend.__new__(DeepSeekServingBackend)
            probe.cfg, probe.max_seq_len = cfg, cfg.max_seq_len
            probe.scheme, probe.device, probe.num_layers = scheme, torch.device("cuda:0"), 10
            probe.host_arena_tokens, probe.slots = HOST_TOKENS, POOL_TOKENS
            probe.workspace_query_tokens = workspace_query_tokens or chunk
            probe.chunk_size = chunk
            probe.extend_chunk_size = None
            try:
                plan = probe.plan_resources(
                    CacheFootprint(HBM_BUDGET, DRAM_BUDGET),
                    {"max_session_capacity": CAPACITY, "max_candidate_tokens": CANDIDATES},
                )
                session = CacheFootprint.from_mapping(probe.estimate_session_bytes(CAPACITY))
                counts = [USERS, (HBM_BUDGET - plan.shared.hbm) // session.hbm]
                if session.dram:
                    counts.append((DRAM_BUDGET - plan.shared.dram) // session.dram)
                if plan.host_pages:
                    counts.append(plan.host_pages // probe.estimate_session_host_pages(CAPACITY))
                max_users = min(counts)
                if max_users < 1:
                    raise CacheBudgetExceeded("shared workspace plus one session exceeds budget")
                row = {
                    "chunk": chunk,
                    "scheme": scheme,
                    "feasible": True,
                    "shared": asdict(plan.shared),
                    "session": asdict(session),
                    "max_cached_users": max_users,
                    "resource_plan": dict(plan.metadata),
                    "host_page_capacity": plan.host_pages,
                }
            except CacheBudgetExceeded as error:
                row = {"chunk": chunk, "scheme": scheme, "feasible": False, "reason": str(error)}
            rows.append(row)
    return rows


def cache_tensors(backend):
    """Independent explicit inventory matching the backend's cache responsibility.

    Never recurse through a model object: doing so would include weights and
    ordinary activations. Repeated views of pools/staging are deduplicated later.
    """
    tensors = []
    staging = getattr(backend, "_dense_staging", None)
    if staging is not None:
        # Enumerate the owned fields independently of shared_bytes and staging's
        # production storage helper, including the two slots' single allocation.
        tensors.extend(staging._buffers.values())
    for pending in getattr(backend, "_dense_sources", ()):
        tensors.append(pending.source)
    pool = backend._shared_pool
    if pool is not None:
        if pool._writes:
            raise RuntimeError(
                "memory phase must start after the previous cache commit drains sources"
            )
        tensors.extend(
            getattr(pool, name)
            for name in (
                "free_slots",
                "allocation_log",
                "counter",
                "prefetch_stats",
                "miss_scratch",
                "_free_pages",
            )
        )
        for layer in pool.layers:
            tensors.extend(
                getattr(layer, name)
                for name in (
                    "host",
                    "records",
                    "host_to_device",
                    "device_to_host",
                    "priority",
                    "free",
                    "clock_tensor",
                )
            )
    for session in backend._sessions:
        tensors.extend(session.stages)
        tensors.extend(value for value in session.prefix_offsets if value is not None)
        for runner in session.runners:
            tensors.extend((runner.index_keys, runner.index_scales, runner.offset))
            if session.sparse_session is None:
                tensors.extend(
                    getattr(runner.cache, name)
                    for name in ("records", "host", "host_to_device", "device_to_host", "age")
                )
        if session.sparse_session is not None:
            tensors.extend(
                getattr(session.sparse_session, name)
                for name in ("_pages", "page_table", "_prefetch_totals")
            )
    return [tensor for tensor in tensors if tensor is not None]


def inventory_and_limits(backend, runner, *, observe_cpu_accounting=False):
    inventory = storage_inventory(cache_tensors(backend))
    actual = runner.pool.audit()
    device = str(backend.device)
    fixed = inventory["devices"]
    charged = {
        "hbm": fixed.get(device, {}).get("storage_bytes", 0),
        "dram": fixed.get("cpu", {}).get("allocator_bytes", 0),
    }
    inventory["backend_ledger_bytes"] = asdict(actual)
    inventory["independent_charged_bytes"] = charged
    inventory["backend_ledger_matches"] = charged == asdict(actual)
    if charged != asdict(actual) and (not observe_cpu_accounting or charged["hbm"] != actual.hbm):
        raise RuntimeError(
            f"independent cache inventory differs from backend ledger: {charged} != {actual}"
        )
    reserved = runner.pool.reserved
    limits = {
        device: reserved.hbm - fixed.get(device, {}).get("allocator_bytes", 0),
        "cpu": reserved.dram - fixed.get("cpu", {}).get("allocator_bytes", 0),
    }
    inventory["fixed_reservation_deficit_bytes"] = {
        key: max(0, -value) for key, value in limits.items()
    }
    if any(size < 0 for size in limits.values()) and (
        not observe_cpu_accounting or limits[device] < 0
    ):
        raise RuntimeError("rounded fixed allocations exceed cache reservation before execution")
    limits["cpu"] = max(0, limits["cpu"])
    return inventory, limits


def _restore_attribute(obj, name, had_local, local):
    if had_local:
        setattr(obj, name, local)
    else:
        delattr(obj, name)


class ServingMemorySampler:
    """External instrumentation; every wrapped backend method is restored."""

    def __init__(
        self,
        backend,
        runner,
        *,
        chunk,
        data_dir,
        profile_dir,
        history_max_entries,
        cpu_observation_limit_bytes=None,
    ):
        self.backend, self.runner, self.chunk = backend, runner, chunk
        self.data_dir, self.profile_dir = Path(data_dir), Path(profile_dir)
        self.history_max_entries = history_max_entries
        self.cpu_observation_limit_bytes = cpu_observation_limit_bytes
        self.request_id = None
        self.audit = None
        self.rows = []

    def capture(self, phase, original, *args, **kwargs):
        if self.request_id not in PROFILE_REQUESTS:
            return original(*args, **kwargs)
        if self.audit is not None:
            raise RuntimeError("nested memory capture would invalidate allocation histories")
        label = f"request_{self.request_id:02d}_{phase}"
        inventory, limits = inventory_and_limits(
            self.backend,
            self.runner,
            observe_cpu_accounting=self.cpu_observation_limit_bytes is not None,
        )
        reserved = asdict(self.runner.pool.reserved)
        torch.cuda.reset_peak_memory_stats(self.backend.device)
        audit = CacheMemoryAudit(
            self.profile_dir / f"{label}.json",
            devices=[self.backend.device],
            history_max_entries=self.history_max_entries,
            record_shapes=False,
            with_stack=False,
        )
        try:
            self.audit = audit
            with (
                audit,
                audit.track_projected_kv(self.backend.attentions),
                audit.track_scalar_transfers(),
                audit.scopes("cache_maintenance"),
            ):
                host_before = host_allocator_counters()
                result = original(*args, **kwargs)
                host_after = host_allocator_counters()
        finally:
            self.audit = None
        # Dense truncate only updates lengths and copies the saved hint in place.
        # A scoped phase with no allocations is valid; execution phases still
        # require allocation evidence and every phase reconciles pinned handouts.
        report = audit.result(
            limits=limits, allow_empty_memory=phase in ("truncate", "session_metrics")
        )
        pinned_handout_bytes = (
            host_after["active_bytes.allocated"] - host_before["active_bytes.allocated"]
        )
        explained_handouts = (
            report["pinned_scalar_handout_bytes"] + report["pinned_internal_handout_bytes"]
        )
        if pinned_handout_bytes != explained_handouts:
            report["evidence_errors"].append(
                f"execution used {pinned_handout_bytes} cumulative pinned handout bytes, "
                f"but scalar/nonzero scopes explain {explained_handouts}"
            )
        report["passed_observed_temporary_bound"] &= (
            inventory["backend_ledger_matches"]
            and not any(inventory["fixed_reservation_deficit_bytes"].values())
            and not report["evidence_errors"]
        )
        after, _ = inventory_and_limits(
            self.backend,
            self.runner,
            observe_cpu_accounting=self.cpu_observation_limit_bytes is not None,
        )
        before_keys = {
            (row["device"], row["address"], row["storage_bytes"]) for row in inventory["storages"]
        }
        after_keys = {
            (row["device"], row["address"], row["storage_bytes"]) for row in after["storages"]
        }
        if before_keys - after_keys:
            raise RuntimeError("a profiled execution phase replaced fixed baseline storage")
        expected_sources = (
            10 * (HISTORY // self.chunk if phase == "prefill" else 1)
            if phase in ("prefill", "extend")
            else 0
        )
        if report["projected_source_markers"] != expected_sources:
            raise RuntimeError(
                "projected KV source markers do not cover every executed layer batch"
            )
        if phase in ("prefill", "extend") and report["ordinary_result_markers"] != 1:
            raise RuntimeError("returned hidden concatenation lacks its ordinary-output marker")
        total_peaks = {
            device: inventory["devices"].get(device, {}).get("allocator_bytes", 0)
            + (row["observed_allocator_active_peak_bytes"] or 0)
            for device, row in report["devices"].items()
        }
        report.update(
            request_id=self.request_id,
            phase=phase,
            chunk=self.chunk,
            scheme=self.backend.scheme,
            baseline_fixed_cache=inventory,
            reserved_cache_bytes=reserved,
            total_cache_allocator_peak_bytes=total_peaks,
            cache_hard_budgets={"hbm": HBM_BUDGET, "dram": DRAM_BUDGET},
            process_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.backend.device),
            process_cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.backend.device),
            process_pinned_host_allocator={
                "before": host_before,
                "after": host_after,
                "execution_cumulative_handout_bytes": pinned_handout_bytes,
                "owned_fixed_block_bytes": sum(
                    row["allocator_bytes"] for row in inventory["storages"] if row["pinned"]
                ),
                "scope": "process-wide allocator retained blocks; includes cached unowned blocks",
                "peak_semantics": "allocated peak sums per-bin peaks, not exact concurrency",
                "active_stats_reliable": False,
                "active_stats_limitation": "installed torch 2.12.1 no-stream frees omit active decrement",
            },
            trace_file=audit.trace_path.name,
            cuda_history_file=audit.history_path.name,
            scope="prefill/extend/truncate/session_metrics; admission and session release use existing reservation audit",
            limit_semantics="total reserved cache minus rounded fixed baseline; includes reserved but not yet materialized hint state",
            profiler_options={"profile_memory": True, "record_shapes": False, "with_stack": False},
            observation_only=self.cpu_observation_limit_bytes is not None,
            cpu_observation_limit_bytes=self.cpu_observation_limit_bytes,
        )
        report_path = self.data_dir / f"{label}.json"
        write_json(report_path, report)
        self.rows.append(
            {"request_id": self.request_id, "phase": phase, "report": report_path.name}
        )
        cpu_observation_can_continue = (
            self.cpu_observation_limit_bytes is not None
            and not report["evidence_errors"]
            and report["devices"]["cpu"]["observed_allocator_active_peak_bytes"]
            <= self.cpu_observation_limit_bytes
            and all(
                row["fits_temporary_limit"]
                for device, row in report["devices"].items()
                if device != "cpu"
            )
            and total_peaks["cpu"] <= DRAM_BUDGET
        )
        if not report["passed_observed_temporary_bound"] and not cpu_observation_can_continue:
            raise RuntimeError(f"cache allocator audit failed: {report_path}")
        print(
            f"observed {self.chunk}/{self.backend.scheme}/{label}: "
            f"reserved_bound_passed={report['passed_observed_temporary_bound']} "
            f"cache_peak={total_peaks}",
            flush=True,
        )
        return result

    @contextmanager
    def installed(self):
        originals = []

        def replace(name, wrapped):
            originals.append((name, name in vars(self.backend), vars(self.backend).get(name)))
            setattr(self.backend, name, wrapped)

        original_forward = self.backend._forward

        def forward(*args, **kwargs):
            if self.audit is not None:
                kwargs["scope"] = self.audit.scopes
            result = original_forward(*args, **kwargs)
            if self.audit is not None:
                self.audit.mark_ordinary_result(result)
            return result

        try:
            replace("_forward", forward)
            for phase in ("prefill", "extend", "truncate", "session_metrics"):
                original = getattr(self.backend, phase)

                def wrapped(*args, _phase=phase, _original=original, **kwargs):
                    return self.capture(_phase, _original, *args, **kwargs)

                replace(phase, wrapped)
            yield self
        finally:
            for name, had_local, local in reversed(originals):
                _restore_attribute(self.backend, name, had_local, local)


def new_runner(backend):
    return PersistentGRRunner(
        backend,
        hbm_budget_bytes=HBM_BUDGET,
        dram_budget_bytes=DRAM_BUDGET,
        resource_limits={"max_session_capacity": CAPACITY, "max_candidate_tokens": CANDIDATES},
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, default=Path("/preset-models"))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--chunks", type=int, nargs="+", choices=CHUNKS, default=list(CHUNKS))
    result.add_argument(
        "--schemes",
        nargs="+",
        choices=SCHEMES,
        default=list(SCHEMES),
        help="scheme subset; hbm is required as the independent same-C numerical reference",
    )
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--workspace-query-tokens",
        type=int,
        help="reserve one explicit workspace Q for every C; default uses each C",
    )
    result.add_argument("--history-max-entries", type=int, default=4_000_000)
    result.add_argument(
        "--cpu-observation-limit-bytes",
        type=int,
        help="diagnostic continuation for CPU reservation failures; never accepted or published",
    )
    result.add_argument(
        "--plan-only", action="store_true", help="pure CPU feasibility, no model weights or GPU"
    )
    result.add_argument("--run-id", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--profile-dir", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id) or len(set(args.chunks)) != len(
        args.chunks
    ):
        raise ValueError("run ID must be path-safe and chunk choices distinct")
    if args.cpu_observation_limit_bytes is not None and args.cpu_observation_limit_bytes < 1:
        raise ValueError("CPU observation ceiling must be positive")
    schemes = selected_schemes(args.schemes)
    expected_matrix = [
        {"chunk": chunk, "scheme": scheme} for chunk in args.chunks for scheme in schemes
    ]
    cfg = Config.from_checkpoint(args.model)
    plans = screen_cases(
        cfg, args.chunks, workspace_query_tokens=args.workspace_query_tokens, schemes=schemes
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "configuration.json",
        {
            "parameters": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items()
            },
            "schemes": list(schemes),
            "expected_case_matrix": expected_matrix,
        },
    )
    write_json(args.output_dir / "feasibility.json", plans)
    if args.plan_only:
        print(json.dumps(plans, indent=2))
        return
    if args.profile_dir is None:
        raise ValueError("--profile-dir is required for GPU capture")
    args.profile_dir.mkdir(parents=True, exist_ok=False)
    if any(not row["feasible"] for row in plans):
        raise CacheBudgetExceeded("one or more required memory-audit cases are infeasible")
    if torch.cuda.get_device_capability(args.device) != (9, 0):
        raise RuntimeError("memory capture requires Hopper SM90")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    source_digest = source_snapshot(args.output_dir, include_official=True)
    provenance = backend_provenance()
    workload = build_workload(
        WorkloadConfig(
            model="deepseek_v32",
            num_users=USERS,
            requests=USERS * 2,
            history_tokens=HISTORY,
            candidate_tokens=CANDIDATES,
            seed=args.seed,
            context_limit=cfg.max_seq_len,
            sampling="sequential",
        ),
        tokenizer=args.model,
    )
    workload.write(args.output_dir / "workload")
    manifest = {
        "schema": "echo-serving-cache-memory-validation-v1",
        "accepted": False,
        "run_id": args.run_id,
        "checkpoint": str(args.model),
        "source_sha256": source_digest,
        "backend_provenance": provenance,
        "workload_sha256": workload.manifest["workload_sha256"],
        "profiled_requests": list(PROFILE_REQUESTS),
        "chunks": args.chunks,
        "schemes": list(schemes),
        "expected_case_matrix": expected_matrix,
        "workspace_query_tokens": args.workspace_query_tokens,
        "workspace_protocol": "fixed" if args.workspace_query_tokens else "per_chunk",
        "scope": "10 independently copied checkpoint blocks with source-input replay; complete 16-user two-pass 64K+128 trace",
        "latency_measurement": False,
        "observation_only": args.cpu_observation_limit_bytes is not None,
        "cpu_observation_limit_bytes": args.cpu_observation_limit_bytes,
        "cache_geometry": {"host_tokens": HOST_TOKENS, "pool_tokens": POOL_TOKENS},
        "budgets": {"hbm": HBM_BUDGET, "dram": DRAM_BUDGET},
        "hardware": {
            "device": args.device,
            "name": torch.cuda.get_device_name(args.device),
            "total_memory_bytes": torch.cuda.get_device_properties(args.device).total_memory,
        },
        "cases": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    backend = DeepSeekServingBackend(
        args.model,
        device=args.device,
        num_layers=10,
        chunk_size=min(args.chunks),
        sparse_pool_tokens=POOL_TOKENS,
        host_arena_tokens=HOST_TOKENS,
        workspace_query_tokens=args.workspace_query_tokens or min(args.chunks),
    )
    try:
        for chunk in args.chunks:
            # A same-scheme subset can keep its prior plan across loop
            # iterations; end that ownership before changing the query bound.
            backend.close()
            backend.chunk_size = chunk
            backend.workspace_query_tokens = args.workspace_query_tokens or chunk
            for scheme in schemes:
                backend.configure_scheme(scheme)
                # Compile/initialize on the complete shape outside the profiler,
                # then use independent empty cache state for the actual trace.
                with new_runner(backend) as warmup:
                    warm = warmup.execute(workload.requests[0])
                    del warm
                # Runner close releases only warmup sessions. Reuse this exact
                # shared plan for the independently empty formal session pool.
                case = f"C{chunk}/{scheme}"
                data_dir, profile_dir = args.output_dir / case, args.profile_dir / case
                data_dir.mkdir(parents=True)
                profile_dir.mkdir(parents=True)
                request_rows = []
                with new_runner(backend) as runner:
                    sampler = ServingMemorySampler(
                        backend,
                        runner,
                        chunk=chunk,
                        data_dir=data_dir,
                        profile_dir=profile_dir,
                        history_max_entries=args.history_max_entries,
                        cpu_observation_limit_bytes=args.cpu_observation_limit_bytes,
                    )
                    with sampler.installed():
                        for request in workload.requests:
                            sampler.request_id = request["request_id"]
                            output = runner.execute(request)
                            values = {
                                "hidden": output.hidden.cpu(),
                                "logits": backend.last_logits.cpu(),
                            }
                            reference = (
                                args.output_dir
                                / f"C{chunk}"
                                / "reference"
                                / f"request_{request['request_id']:02d}.pt"
                            )
                            if scheme == "hbm":
                                reference.parent.mkdir(parents=True, exist_ok=True)
                                torch.save(values, reference)
                                expected = values
                            else:
                                expected = torch.load(
                                    reference, map_location="cpu", weights_only=True
                                )
                            comparisons = {
                                name: numerical_comparison(value, expected[name], atol=0, rtol=0)
                                for name, value in values.items()
                            }
                            # These runs are intrusive validation. Discard timing
                            # fields so none can be mistaken for latency evidence.
                            metrics = {
                                key: value
                                for key, value in output.metrics.items()
                                if not key.endswith("_ms")
                            }
                            request_rows.append({**metrics, "numerics": comparisons})
                            write_json(data_dir / "requests.json", request_rows)
                            del output, values, expected
                    case_row = {
                        "chunk": chunk,
                        "scheme": scheme,
                        "profiled_phases": sampler.rows,
                        "requests": len(request_rows),
                        "accepted": args.cpu_observation_limit_bytes is None,
                        "max_cached_users": max(row["cached_users"] for row in request_rows),
                        "second_pass_hits": sum(
                            row["prefix_cache_hit"] for row in request_rows[USERS:]
                        ),
                        "resource_plan": asdict(runner.resource_plan),
                    }
                manifest["cases"].append(case_row)
                verify_source_snapshot(args.output_dir)
                write_json(args.output_dir / "manifest.json", manifest)
                print(
                    f"completed {'observation' if args.cpu_observation_limit_bytes is not None else 'validation'}: {case}",
                    flush=True,
                )
    finally:
        backend.close()
    verify_source_snapshot(args.output_dir)
    validate_completed_matrix(manifest["cases"], expected_matrix, requests=len(workload.requests))
    manifest["accepted"] = args.cpu_observation_limit_bytes is None
    write_json(args.output_dir / "manifest.json", manifest)
    if args.cpu_observation_limit_bytes is not None:
        raise SystemExit(3)  # Keep diagnostics in staging; never publish as accepted.


if __name__ == "__main__":
    main()
