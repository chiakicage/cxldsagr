"""Independent checks and clean paired complete-indexer timing for private preparation fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.local_native import collect_local_native_artifacts
from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_echo_official.src import q1_fused_prepare as candidate
from experiments.deepseek_v32_echo_official.src import q1_official_prefetch as raw
from experiments.deepseek_v32_echo_official.src import q1_packing, q1_promotion
from operators.deepseek_v32.indexer import (
    cache_ops,
    echo,
    official_decode,
    official_prefetch,
    q1_topk_cub,
)
from operators.deepseek_v32.indexer.page64 import pack_q1_keys
from operators.deepseek_v32.indexer.selection import exact_topk
from operators.deepseek_v32.indexer.tests.test_official_prefetch import _lease

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
KIND = "deepseek-private-q1-fused-prepare-v1"
VARIANTS = ("baseline", "candidate")
POLICIES = ("zero", "small", "partial", "warm", "empty")
N_VALUES = (1, 63, 64, 65, 127, 128, 129, 32768, 65536, 65537)
MISSING = 2**31 - 1


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def exact(actual, expected, message):
    require(q1_promotion.exact_bytes(actual.cpu(), expected.cpu()), message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def release(lease):
    """Only used after the caller has confirmed successful stream completion."""
    lease.pop(official_prefetch.CLEANUP_KEY, None)
    lease.pop(official_prefetch.STATE_KEY, None)
    lease.pop("_prepared", None)


def prepare_call(variant, keys, scales, lease, context):
    if variant == "candidate":
        return candidate.prepare(keys, scales, lease, context)
    packed = pack_q1_keys(keys, scales)
    blocks = torch.arange(len(packed), dtype=torch.int32, device=keys.device).view(1, -1)
    state = official_prefetch._prepare(lease, context, len(keys))
    return packed, blocks, state


def check_prepared(keys, scales, lease, context, actual):
    packed, blocks, state = actual
    torch.cuda.synchronize()
    exact(packed, q1_packing.reference_pack(keys, scales), "All packed key/scale/padding bytes")
    exact(blocks.cpu(), torch.arange(len(packed), dtype=torch.int32)[None], "Block-table identity")
    history = lease["history_length"]
    logical = torch.arange(history, dtype=torch.int64)
    table = torch.zeros(len(keys), dtype=torch.int32)
    if history:
        table[:history] = (lease["page_table"].cpu()[logical // 64] * 64 + logical % 64).int()
    exact(state.page_table.cpu()[0], table, "Current pages and pending/padded host token IDs")
    exact(state.host_ids.cpu(), torch.full((1, 64), -1, dtype=torch.int32), "Stage reset")
    exact(state.counter.cpu(), torch.zeros(1, dtype=torch.uint32), "Counter reset")
    exact(
        lease["allocation_log"].cpu(),
        torch.full_like(lease["allocation_log"].cpu(), MISSING),
        "Full journal reset",
    )
    exact(lease["prefetch_stats"].cpu(), torch.zeros(3, dtype=torch.int64), "Stats reset")
    exact(context.cpu(), torch.tensor([history + 1], dtype=torch.int32), "Context unchanged")
    return {
        "packed_sha256": tensor_digest(packed),
        "token_table_sha256": tensor_digest(table),
        "packed_bytes": packed.numel(),
    }


def byte_checks():
    rows = []
    for n in N_VALUES:
        capacity = max(384, (n + 63) // 64 * 64)
        lease = _lease(slots=96, history=n - 1, host_capacity=capacity)
        lease["page_table"] = torch.arange(capacity // 64, dtype=torch.int32, device="cuda").flip(0)
        context = torch.tensor([n], dtype=torch.int32, device="cuda")
        for key_offset in (0, 4):
            keys = (
                torch.arange(n * 128 + key_offset, device="cuda", dtype=torch.int64)
                .to(torch.uint8)[key_offset:]
                .view(n, 128)
                .view(torch.float8_e4m3fn)
            )
            patterns = torch.tensor(
                [0, -2147483648, 1, -2147483647, 0x7FC01234, 0x7F800000, -8388608],
                dtype=torch.int32,
                device="cuda",
            )
            scales = patterns.repeat((n + 6) // 7)[:n].view(torch.float32)
            initial = {
                key: lease[key].clone()
                for key in ("device", "host_to_device", "device_to_host", "priority", "offset")
            }
            for variant in VARIANTS:
                release(lease)
                lease["allocation_log"].fill_(57)
                lease["prefetch_stats"].fill_(57)
                lease["counter"].fill_(57)
                result = prepare_call(variant, keys, scales, lease, context)
                data = check_prepared(keys, scales, lease, context, result)
                for key, expected in initial.items():
                    exact(lease[key], expected, "Preparation changed " + key)
                rows.append({"n": n, "key_offset": key_offset, "variant": variant, **data})
                release(lease)
            # Captured preparation must read changed bytes and page contents.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = prepare_call("candidate", keys, scales, lease, context)
            for iteration in range(3):
                keys.view(torch.uint8).fill_(1 + 97 * iteration)
                scales.view(torch.int32).fill_(0x7FA00123 + iteration)
                lease["page_table"].copy_(lease["page_table"].roll(1))
                lease["counter"].fill_(71)
                lease["allocation_log"].fill_(71)
                lease["prefetch_stats"].fill_(71)
                graph.replay()
                data = check_prepared(keys, scales, lease, context, captured)
                rows.append({"n": n, "key_offset": key_offset, "graph_replay": iteration, **data})
            del graph, captured
            release(lease)
        del lease
    return rows


def lifecycle_checks():
    lease = _lease(history=128)
    keys = torch.zeros((129, 128), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.ones(129, dtype=torch.float32, device="cuda")
    context = torch.tensor([129], dtype=torch.int32, device="cuda")
    scratch = torch.empty(96, dtype=torch.int64, device="cuda")
    token = cache_ops.prepare_prefetch_free(
        lease["priority"],
        lease["free_bitmap"],
        lease["device_to_host"],
        lease["free_slots"],
        lease["allocation_log"],
        lease["counter"],
        lease["prefetch_stats"],
        scratch,
        timestamp=8,
    )
    lease.update(_prepared=token, max_prefetch=64, prepared_limit=64)
    result = candidate.prepare(keys, scales, lease, context)
    check_prepared(keys, scales, lease, context, result)
    require(token.used, "Bounded token consumed exactly once")
    try:
        candidate.prepare(keys, scales, lease, context)
    except ValueError:
        pass
    else:
        raise AssertionError("Already-owned lease was accepted")
    state = result[2]
    state.host_ids[0, :2] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    state.counter.fill_(2)
    lease["host_to_device"][:2] = torch.tensor(
        [len(lease["device"]), len(lease["device"]) + 1], device="cuda", dtype=torch.int32
    )
    state.clear()
    state.clear()
    torch.cuda.synchronize()
    require(bool(lease["host_to_device"].eq(MISSING).all()), "Host-zero idempotent cleanup")
    release(lease)
    error = RuntimeError("injected private preparation launch failure")
    original_call = official_prefetch._call

    def fail(module, name, device, *args):
        if name == "q1_fused_prepare":
            require(callable(lease.get(official_prefetch.CLEANUP_KEY)), "Cleanup precedes launch")
            raise error
        return original_call(module, name, device, *args)

    with patch.object(official_prefetch, "_call", fail):
        try:
            candidate.prepare(keys, scales, lease, context)
        except RuntimeError as observed:
            require(observed is error, "Original exception object preserved")
        else:
            raise AssertionError("Injected error disappeared")
    require(official_prefetch.STATE_KEY in lease, "Owner retains staging after failure")
    # No device publication occurred in the injected call. Do not consume its
    # uninitialized stage IDs/counter; discard only after confirmed completion.
    torch.cuda.synchronize()
    release(lease)
    return {
        "bounded_consumption": True,
        "duplicate_owner_rejected": True,
        "cleanup_and_error": True,
    }


class Case(raw.Case):
    """Reuse real-input construction, adding a legal local FIFO publication lease."""

    def __init__(self, payload, name):
        super().__init__(payload, name)
        self.lease = {
            "host": self.host,
            "device": self.pool,
            "host_to_device": self.h2d[:-1],
            "device_to_host": torch.empty(len(self.pool), dtype=torch.int64, device="cuda"),
            "free_slots": torch.empty(len(self.pool) - 1, dtype=torch.int32, device="cuda"),
            "allocation_log": torch.empty(len(self.pool), dtype=torch.int64, device="cuda"),
            "counter": self.counter,
            "prefetch_stats": torch.empty(3, dtype=torch.int64, device="cuda"),
            "page_table": torch.arange(len(self.host) // 64, dtype=torch.int32, device="cuda"),
            "offset": torch.zeros(16, dtype=torch.float32, device="cuda"),
            "history_length": self.history,
            "max_prefetch": 64,
            "prepared_limit": 64,
            "priority": torch.empty(len(self.pool), dtype=torch.int64, device="cuda"),
            "free_bitmap": torch.empty(len(self.pool), dtype=torch.bool, device="cuda"),
        }
        self.scratch = torch.empty(len(self.pool) - 1, dtype=torch.int64, device="cuda")
        self.reference_scores = self.reference()
        self.policy_name = None

    def set_policy(self, name):
        super().policy(name, self.reference_scores)
        self.lease["offset"][1:2].copy_(self.threshold)
        mapping = self.initial_h2d[:-1].cpu()
        self.initial_reverse = torch.full((len(self.pool),), MISSING, dtype=torch.int64)
        occupied_hosts = torch.nonzero(mapping != MISSING).flatten()
        self.initial_reverse[mapping[occupied_hosts].long()] = occupied_hosts
        self.initial_records = self.pool_before.cpu().clone()
        self.initial_records[mapping[occupied_hosts].long()] = self.host[occupied_hosts]
        self.initial_priority = torch.full((len(self.pool),), -1, dtype=torch.int64)
        self.initial_priority[0] = MISSING
        self.initial_priority[mapping[occupied_hosts].long()] = 7
        self.initial_bitmap = self.initial_priority.eq(-1)
        self.initial_records_device = self.initial_records.cuda()
        self.initial_reverse_device = self.initial_reverse.cuda()
        self.initial_priority_device = self.initial_priority.cuda()
        self.initial_bitmap_device = self.initial_bitmap.cuda()
        self.policy_name = name
        self.restore()

    def restore(self):
        # Completion is confirmed before replacing a staging owner. The entire
        # reset, including valid bounded FIFO preparation, is outside timing.
        torch.cuda.synchronize()
        release(self.lease)
        self.h2d.copy_(self.initial_h2d)
        self.pool.copy_(self.initial_records_device)
        for name, value in (
            ("device_to_host", self.initial_reverse_device),
            ("priority", self.initial_priority_device),
            ("free_bitmap", self.initial_bitmap_device),
        ):
            self.lease[name].copy_(value)
        self.prepare_fifo()
        torch.cuda.synchronize()

    def prepare_fifo(self):
        self.lease["_prepared"] = cache_ops.prepare_prefetch_free(
            self.lease["priority"],
            self.lease["free_bitmap"],
            self.lease["device_to_host"],
            self.lease["free_slots"],
            self.lease["allocation_log"],
            self.counter,
            self.lease["prefetch_stats"],
            self.scratch,
            timestamp=8,
        )

    def invoke(self, variant):
        function = echo.logits if variant == "baseline" else candidate.logits
        scores = function(
            self.q[0],
            self.k,
            self.weights,
            self.scales,
            self.history,
            prefetch=self.lease,
            _pad_to_stride=True,
        )
        state = self.lease[official_prefetch.STATE_KEY]
        self.lease[official_prefetch.CLEANUP_KEY]()
        return scores, state

    def audit(self, result):
        scores, state = result
        torch.cuda.synchronize()
        exact(scores[:, : self.n], self.reference_scores, "Current real score bits")
        require(bool(torch.isneginf(scores[:, self.n :]).all()), "Physical causal tail")
        values, ids = exact_topk(scores, 2048)
        expected_values, expected_ids = exact_topk(self.reference_scores, 2048)
        exact(values, expected_values, "Exact top-k values")
        exact(ids, expected_ids, "Exact top-k IDs")
        score_cpu = self.reference_scores[0, : self.history].cpu()
        eligible = (score_cpu > self.threshold.item()) & (
            self.initial_h2d[: self.history].cpu() == MISSING
        )
        eligible_ids = torch.nonzero(eligible).flatten()
        expected_count = min(64, len(eligible_ids))
        attempts = int(state.counter.item())
        count = min(64, attempts)
        require(count == expected_count, "Actual successful predictive count")
        require(count <= attempts <= len(eligible_ids), "Actual attempt count")
        staged = state.host_ids[0, :count].cpu().long()
        require(staged.unique().numel() == count, "Unique staged host IDs")
        require(bool(torch.isin(staged, eligible_ids).all()), "Strict FP32 eligible history misses")
        require(bool(state.host_ids[0, count:].eq(-1).all()), "No unused stage ID writes")
        slots = self.lease["free_slots"][:count].cpu().long()
        expected_h2d = self.initial_h2d.cpu().clone()
        expected_h2d[staged] = slots.int()
        expected_reverse = self.initial_reverse.clone()
        expected_reverse[slots] = staged
        expected_records = self.initial_records.clone()
        expected_records[slots] = self.host[staged]
        expected_journal = torch.full_like(expected_reverse, MISSING)
        expected_journal[slots] = staged
        exact(self.h2d, expected_h2d, "Complete host map including sentinel and cleared tags")
        exact(self.lease["device_to_host"], expected_reverse, "Complete reverse map")
        exact(self.pool, expected_records, "Complete promoted pool bytes")
        exact(state.records[0, :count], self.host[staged], "Actual staging payload bits")
        exact(self.lease["allocation_log"], expected_journal, "Complete allocation journal")
        exact(self.lease["priority"], self.initial_priority, "Priority awaits normal finalization")
        exact(self.lease["free_bitmap"], self.initial_bitmap, "Free bitmap awaits finalization")
        exact(
            self.lease["prefetch_stats"], torch.tensor([count, 0, attempts - count]), "Statistics"
        )
        return {
            "case": self.name,
            "policy": self.policy_name,
            "attempts": attempts,
            "staged": count,
            "h2d_bytes": count * 1152,
            "scores_sha256": tensor_digest(scores),
            "topk_ids_sha256": tensor_digest(ids),
            "staged_host_ids": staged.tolist(),
            "slots": slots.tolist(),
            "staged_records_sha256": tensor_digest(state.records[0, :count]),
            "full_state_passed": True,
        }


def complete_checks(args):
    rows = []
    for path in args.inputs:
        case = Case(torch.load(path, map_location="cpu", weights_only=True), path.stem)
        for policy in POLICIES:
            case.set_policy(policy)
            for variant in VARIANTS:
                case.restore()
                result = case.invoke(variant)
                rows.append({"variant": variant, "execution": "eager", **case.audit(result)})
                # Ordered, nondefault-stream consumption retains all stage owners.
                case.restore()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    case.lease["_prepared"] = cache_ops.prepare_prefetch_free(
                        case.lease["priority"],
                        case.lease["free_bitmap"],
                        case.lease["device_to_host"],
                        case.lease["free_slots"],
                        case.lease["allocation_log"],
                        case.counter,
                        case.lease["prefetch_stats"],
                        case.scratch,
                        timestamp=8,
                    )
                    result = case.invoke(variant)
                side.synchronize()
                rows.append({"variant": variant, "execution": "side_stream", **case.audit(result)})
                case.restore()
                graph = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream()
                with torch.cuda.stream(capture_stream):
                    case.prepare_fifo()
                capture_stream.synchronize()
                with torch.cuda.graph(graph, stream=capture_stream):
                    result = case.invoke(variant)
                for replay in range(3):
                    case.restore()
                    graph.replay()
                    rows.append(
                        {
                            "variant": variant,
                            "execution": "graph",
                            "replay": replay,
                            **case.audit(result),
                        }
                    )
                # Keep H/N and addresses fixed while changing all data consumed
                # by the candidate's preparation and the unchanged official core.
                saved = {
                    name: getattr(case, name).clone() for name in ("q", "k", "scales", "weights")
                }
                case.q.copy_(case.q.float().flip(-1).to(case.q.dtype))
                case.k.copy_(case.k.float().roll(1, 0).to(case.k.dtype))
                case.scales.mul_(1.03125)
                case.weights.mul_(0.875)
                case.reference_scores = case.reference()
                for replay in range(2):
                    case.restore()
                    graph.replay()
                    rows.append(
                        {
                            "variant": variant,
                            "execution": "changed_graph",
                            "replay": replay,
                            **case.audit(result),
                        }
                    )
                for name, value in saved.items():
                    getattr(case, name).copy_(value)
                case.reference_scores = case.reference()
                del graph, result, capture_stream
        torch.cuda.synchronize()
        del case
    return rows


def make_graph(case, variant):
    case.restore()
    case.invoke(variant)
    torch.cuda.synchronize()
    case.restore()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        case.prepare_fifo()
    capture_stream.synchronize()
    with torch.cuda.graph(graph, stream=capture_stream):
        start.record()
        result = case.invoke(variant)
        end.record()
    return graph, start, end, result, capture_stream


def benchmark(args):
    rows, samples = [], []
    for path in args.inputs:
        case = Case(torch.load(path, map_location="cpu", weights_only=True), path.stem)
        for policy in POLICIES:
            case.set_policy(policy)
            graphs = {variant: make_graph(case, variant) for variant in VARIANTS}
            for _ in range(args.warmups):
                for variant in VARIANTS:
                    case.restore()
                    graphs[variant][0].replay()
                    graphs[variant][2].synchronize()
            values = {variant: [] for variant in VARIANTS}
            for pair in range(args.pairs):
                order = VARIANTS if pair % 2 == 0 else VARIANTS[::-1]
                for variant in order:
                    case.restore()
                    graph, start, end, result, _ = graphs[variant]
                    graph.replay()
                    end.synchronize()
                    elapsed = start.elapsed_time(end) * 1000
                    attempts = int(result[1].counter.item())
                    values[variant].append(elapsed)
                    samples.append(
                        {
                            "case": case.name,
                            "policy": policy,
                            "pair": pair,
                            "order": list(order),
                            "variant": variant,
                            "gpu_us": elapsed,
                            "attempts": attempts,
                            "staged": min(64, attempts),
                            "h2d_bytes": min(64, attempts) * 1152,
                        }
                    )
            deltas = [b - a for a, b in zip(values["baseline"], values["candidate"], strict=True)]
            rows.append(
                {
                    "case": case.name,
                    "policy": policy,
                    "warmups": args.warmups,
                    "pairs": args.pairs,
                    "baseline_us": statistics.median(values["baseline"]),
                    "candidate_us": statistics.median(values["candidate"]),
                    "paired_delta_us": statistics.median(deltas),
                    "candidate_wins": sum(value < 0 for value in deltas),
                }
            )
            torch.cuda.synchronize()
            del graphs
        del case
    return {"rows": rows, "samples": samples}


def invalid_call(args):
    lease = _lease(history=64)
    keys = torch.zeros((65, 128), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.ones(65, dtype=torch.float32, device="cuda")
    context = torch.tensor([65], dtype=torch.int32, device="cuda")
    if args.invalid_case == "context":
        context.fill_(64)
    elif args.invalid_case == "negative_page":
        lease["page_table"][0] = -1
    elif args.invalid_case == "outside_page":
        lease["page_table"][0] = len(lease["host"]) // 64
    else:
        raise ValueError("Unknown malformed state")
    prepare_call(args.variant, keys, scales, lease, context)
    torch.cuda.synchronize()
    raise AssertionError("Malformed device input unexpectedly completed")


def invalid_checks(args):
    rows = []
    for variant in VARIANTS:
        for case in ("context", "negative_page", "outside_page"):
            command = [
                sys.executable,
                "-B",
                "-m",
                __spec__.name,
                "--mode",
                "invalid",
                "--physical-device",
                str(args.physical_device),
                "--variant",
                variant,
                "--invalid-case",
                case,
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            require(result.returncode != 0, "Malformed child must fail")
            require(
                "device-side assert" in result.stderr
                or ("Assertion " in result.stderr and " failed." in result.stderr),
                "Child failure must report a device assertion",
            )
            rows.append(
                {
                    "variant": variant,
                    "case": case,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
    return rows


def source_identity(args):
    declared = {}
    for info in (
        echo.build_info(),
        official_decode.build_info(),
        official_prefetch.build_info(),
        candidate.build_info(),
        q1_topk_cub.build_info(),
    ):
        for key in ("source_sha256", "shared_header_sha256"):
            for name, value in info.get(key, {}).items():
                path = Path(name) if Path(name).is_absolute() else ROOT / name
                declared[str(path.resolve())] = value
    files = [
        Path(__file__),
        Path(raw.__file__),
        Path(q1_packing.__file__),
        Path(q1_promotion.__file__),
        ROOT / "evaluation/validation.py",
        ROOT / "evaluation/local_native.py",
        ROOT / "operators/deepseek_v32/indexer/tests/test_official_prefetch.py",
        ROOT / "operators/deepseek_v32/indexer/selection.py",
        ROOT / "operators/deepseek_v32/indexer/q1_topk_cub.py",
    ]
    declared.update({str(path.resolve()): digest(path) for path in files})
    # Bind the executed reference and selection packages as well as local code.
    import deep_gemm
    import flashinfer

    runtime = {}
    for package in (deep_gemm, flashinfer):
        directory = Path(package.__file__).resolve().parent
        runtime.update(
            {
                str(path): digest(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file() and path.suffix in {".py", ".so", ".cu", ".cuh", ".h", ".hpp"}
            }
        )
    return {
        "sources": declared,
        "runtime_files": runtime,
        "inputs": {str(path.resolve()): digest(path) for path in args.inputs},
    }


def execution_identity(args, sources):
    import deep_gemm
    import flashinfer
    import triton

    return {
        "kind": KIND,
        **sources,
        "candidate_native": candidate.native_info(),
        "official_native": official_decode.native_info(
            torch.cuda.get_device_properties(0).multi_processor_count
        ),
        "mapped_local_native": collect_local_native_artifacts(required=False),
        "packing_kernels": q1_packing.kernel_identity(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "deep_gemm": str(Path(deep_gemm.__file__).resolve()),
        "flashinfer": flashinfer.__version__,
        "gpu_uuid": subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.physical_device),
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "physical_device": args.physical_device,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "boundary": "Complete echo.logits plus original staging cleanup; includes current packing, bounds, block table, official schedule, stage reset, unchanged fused scores, causal clean and promotion. Excludes caller FIFO preparation/reset, top-k and exact recall.",
        "residency": "Each sample restores independently specified zero/small/partial/warm/empty state before timing.",
        "precision": {"tf32": torch.backends.cuda.matmul.allow_tf32},
    }


def warm_identity(args):
    # Compile only the same complete-API variants/shapes in every mode before
    # collecting identity. Extra check-only packing shapes are reported separately.
    candidate.module()
    for path in args.inputs:
        case = Case(torch.load(path, map_location="cpu", weights_only=True), path.stem)
        case.set_policy("zero")
        for variant in VARIANTS:
            case.restore()
            scores, _ = case.invoke(variant)
            exact_topk(scores, 2048)
            torch.cuda.synchronize()
        del case


def profile(args):
    case = Case(
        torch.load(args.inputs[0], map_location="cpu", weights_only=True), args.inputs[0].stem
    )
    case.set_policy(args.policy)
    for _ in range(args.warmups):
        case.restore()
        case.invoke(args.variant)
        torch.cuda.synchronize()
    case.restore()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("q1_fused_prepare_complete_" + args.variant)
    result = case.invoke(args.variant)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    return {
        "variant": args.variant,
        "policy": args.policy,
        "attempts": int(result[1].counter.item()),
        "input": str(args.inputs[0]),
    }


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("check", "bench", "profile", "invalid"), required=True)
    value.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[raw.INPUT_ROOT / f"kernel_inputs_layer_{layer}.pt" for layer in range(3)],
    )
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--receipt", type=Path)
    value.add_argument("--physical-device", type=int, required=True)
    value.add_argument("--warmups", type=int, default=20)
    value.add_argument("--pairs", type=int, default=100)
    value.add_argument("--variant", choices=VARIANTS, default="candidate")
    value.add_argument("--policy", choices=POLICIES, default="zero")
    value.add_argument("--invalid-case", choices=("context", "negative_page", "outside_page"))
    return value


def archive_runtime(identity, destination):
    """Retain the actual mapped local ELF bytes and compiled packing assemblies."""
    from operators.deepseek_v32.indexer.page64 import _kernel

    artifacts = {}
    native = destination / "native"
    native.mkdir()
    for number, item in enumerate(identity["mapped_local_native"]):
        record = item["library"]
        source = Path(record["path"])
        require(digest(source) == record["sha256"], "Mapped native artifact changed")
        path = native / f"{number:02d}_{source.name}"
        path.write_bytes(source.read_bytes())
        artifacts[str(path.relative_to(destination))] = path
    assembly = destination / "packing_assembly"
    assembly.mkdir()
    for cache in _kernel().device_caches.values():
        for compiled in cache[0].values():
            for name, value in compiled.asm.items():
                path = assembly / f"{compiled.hash}.{name}"
                path.write_bytes(value if isinstance(value, bytes) else value.encode())
                artifacts[str(path.relative_to(destination))] = path
    return artifacts


def main():
    args = parser().parse_args()
    require(args.warmups > 0 and args.pairs > 0, "Positive measurement counts required")
    raw.environment()
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose exactly one CUDA GPU"
    )
    require(torch.cuda.get_device_capability() == (9, 0), "Requires SM90")
    if args.mode == "invalid":
        invalid_call(args)
        return
    require(args.output_dir is not None, "--output-dir is required")
    destination = args.output_dir.resolve()
    allowed = Path("/tmp/cxldsagr-checks") if args.mode == "check" else EXPERIMENT / "output"
    require(destination.is_relative_to(allowed), "Output directory must be under " + str(allowed))
    destination.mkdir(parents=True, exist_ok=False)
    sources = source_identity(args)
    archive = destination / "source"
    archive.mkdir()
    for number, (name, expected) in enumerate(sorted(sources["sources"].items())):
        source = Path(name)
        require(digest(source) == expected, "Declared source changed")
        target = archive / f"{number:04d}_{source.name}"
        target.write_bytes(source.read_bytes())
    warm_identity(args)
    before = execution_identity(args, sources)
    write(destination / "identity.json", before)
    if args.mode == "check":
        checks = {
            "byte_checks": byte_checks(),
            "lifecycle": lifecycle_checks(),
            "complete_checks": complete_checks(args),
            "invalid": invalid_checks(args),
        }
        after = execution_identity(args, sources)
        # Byte-only checks compile additional N values in the baseline packer.
        # Verify the accepted real-input compile identities are all unchanged.
        require(
            all(item in after["packing_kernels"] for item in before["packing_kernels"]),
            "Real-input packing identity changed",
        )
        require(
            {k: v for k, v in after.items() if k != "packing_kernels"}
            == {k: v for k, v in before.items() if k != "packing_kernels"},
            "Runtime identity changed",
        )
        require(source_identity(args) == sources, "Sources/inputs changed during check")
        write(destination / "check.json", checks)
        write(destination / "check_runtime.json", after)
        artifacts = {
            "check": destination / "check.json",
            "identity": destination / "identity.json",
            "check_runtime": destination / "check_runtime.json",
        }
        artifacts.update(archive_runtime(after, destination))
        artifacts.update({path.name: path for path in archive.iterdir()})
        write_receipt(
            destination / "receipt.json",
            kind=KIND,
            identity=before,
            checks={
                "passed": True,
                "complete_cases": len(checks["complete_checks"]),
                "byte_cases": len(checks["byte_checks"]),
            },
            artifacts=artifacts,
        )
        print(json.dumps({"passed": True, "receipt": str(destination / "receipt.json")}))
    else:
        receipt = require_receipt(args.receipt, kind=KIND, identity=before)
        result = benchmark(args) if args.mode == "bench" else profile(args)
        after = execution_identity(args, sources)
        require(after == before and source_identity(args) == sources, "Execution identity changed")
        result.update(identity=before, receipt_sha256=receipt["receipt_sha256"], mode=args.mode)
        artifacts = archive_runtime(after, destination)
        result["runtime_archives"] = {name: digest(path) for name, path in artifacts.items()}
        write(destination / "summary.json", result)
        print(json.dumps(result.get("rows", {"profile_complete": True})))


if __name__ == "__main__":
    main()
