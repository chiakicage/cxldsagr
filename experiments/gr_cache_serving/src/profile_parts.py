"""NVTX-only runtime annotations for paired Dense/DSA critical-path diagnosis.

No per-operator synchronization or model/kernel edits. These short instrumented
replays are diagnostic evidence, not replacements for the 512-request results.
"""

import argparse
import hashlib
import json
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from experiments.gr_cache_serving.src.replay import ROOT, load_trace, source_fingerprints


class PartRanges:
    def __init__(self, runner):
        import torch

        self.torch = torch
        self.runner = runner
        self.stack = ExitStack()
        self.local = threading.local()
        self.enabled = False
        self.ordinal = -1
        self.phase = "manager"
        self.chunk = -1
        self.pending = []
        self.dense_bytes = {}

    @contextmanager
    def span(self, part, layer=None):
        if not self.enabled:
            yield
            return
        layer = getattr(self.local, "layer", -1) if layer is None else layer
        label = f"gr/r={self.ordinal}/p={self.phase}/c={self.chunk}/l={layer}/s={part}"
        self.torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            self.torch.cuda.nvtx.range_pop()

    def override(self, obj, name, factory):
        present = name in vars(obj)
        saved = vars(obj).get(name)
        original = getattr(obj, name)
        setattr(obj, name, factory(original))

        def restore():
            if present:
                setattr(obj, name, saved)
            else:
                delattr(obj, name)

        self.stack.callback(restore)

    def annotate(self, obj, name, part, layer=None, layer_arg=False):
        def factory(original):
            def call(*args, **kwargs):
                old = getattr(self.local, "layer", -1)
                if layer is not None or layer_arg:
                    self.local.layer = args[0] if layer_arg else layer
                try:
                    with self.span(part):
                        return original(*args, **kwargs)
                finally:
                    self.local.layer = old

            return call

        self.override(obj, name, factory)

    def __enter__(self):
        try:
            return self.install()
        except BaseException:
            self.stack.close()
            raise

    def install(self):
        for method, phase in (("prefill", "prefill"), ("extend", "candidate")):

            def factory(original, phase=phase):
                def call(*args, **kwargs):
                    old = self.phase, self.chunk
                    self.phase, self.chunk = phase, -1
                    try:
                        with self.span("phase"):
                            return original(*args, **kwargs)
                    finally:
                        self.phase, self.chunk = old

                return call

            self.override(self.runner, method, factory)

        def forward_factory(original):
            def call(*args, **kwargs):
                self.chunk += 1
                with self.span("forward"):
                    result = original(*args, **kwargs)
                controller = self.runner.dense_controller
                if controller is not None:
                    key = self.phase
                    self.dense_bytes[key] = (
                        self.dense_bytes.get(key, 0)
                        + (controller.last_batch_stats["h2d_payload_bytes"])
                    )
                return result

            return call

        self.override(self.runner, "_forward", forward_factory)
        backend = self.runner.runner.attn_backend
        self.annotate(backend, "forward_extend", "cache_layout")
        self.annotate(backend, "_forward_flashmla_prefill", "main_attention")
        for i, layer in enumerate(self.runner.runner.model.model.layers):
            self.annotate(layer, "forward", "layer", i)
            self.annotate(layer.self_attn, "forward", "attention_other")
            self.annotate(layer.self_attn, "forward_absorb_prepare", "qkv_rope")
            self.annotate(layer.self_attn, "forward_absorb_core", "attention_output")
            self.annotate(layer.self_attn.indexer, "forward", "indexer")
            self.annotate(layer.mlp, "forward", "ffn_dense" if i < 3 else "ffn_moe")
            for method in ("prepare_attn", "prepare_mlp", "postprocess_layer"):
                self.annotate(layer.layer_communicator, method, "norm_comm")
        pool = self.runner.runner.token_to_kv_pool
        self.annotate(pool, "set_mla_kv_buffer", "kv_write")

        def recall_factory(original):
            def call(*args, **kwargs):
                with self.span("sparse_recall"):
                    result = original(*args, **kwargs)
                with self.span("measurement"):
                    self.pending.append(
                        (self.phase, args[2], pool.recall_counter.clone().reshape(()))
                    )
                return result

            return call

        if self.runner.dense_controller is None:
            self.override(pool, "recall_miss_tokens_extend_cuda_graph", recall_factory)
        else:
            dense = self.runner.dense_controller
            self.annotate(dense, "_enqueue_prefix", "dense_producer", layer_arg=True)
            self.annotate(dense, "_wait_prefix", "dense_consumer_wait", layer_arg=True)
            self.annotate(dense, "_prepare_batch", "dense_prepare")

            def gather_factory(original):
                def call(*args, **kwargs):
                    if args and args[0].device.type == "cpu":
                        with self.span("cpu_gather"):
                            return original(*args, **kwargs)
                    return original(*args, **kwargs)

                return call

            self.override(self.torch, "index_select", gather_factory)
        return self

    def finish_request(self):
        counts = {}
        if self.pending:
            values = self.torch.stack([x[2] for x in self.pending]).cpu().tolist()
            for (phase, layer, _), value in zip(self.pending, values, strict=True):
                key = f"{phase}/l={layer}"
                counts[key] = counts.get(key, 0) + int(value) * 1152
        result = {"sparse_recall_bytes": counts, "dense_h2d_bytes": self.dense_bytes}
        self.pending, self.dense_bytes = [], {}
        return result

    def __exit__(self, *exc):
        self.stack.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sparse_sync", "dense_prefetch"), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=Path, default=Path("/mnt/nfs/share/models/DeepSeek-V3.2")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-start", type=int, default=8)
    parser.add_argument("--capture-end", type=int, default=14)
    args = parser.parse_args(argv)
    if args.output_dir.exists() or not 0 <= args.capture_start < args.capture_end:
        parser.error("new output directory and valid capture interval required")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError(f"GPU already occupied: {active}")
    source = source_fingerprints()
    source[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    driver = "experiments/gr_cache_serving/scripts/run_parts_profile.sh"
    source[driver] = hashlib.sha256((ROOT / driver).read_bytes()).hexdigest()
    trace, metadata = load_trace(args.trace)
    if (metadata["stable_prefix_tokens"], metadata["candidate_suffix_tokens"]) != (65536, 1024):
        raise ValueError("requires fixed 64K history / 1K candidate workload")
    if args.capture_end > len(trace):
        raise ValueError("capture interval exceeds trace")
    from experiments.gr_cache_serving.src.host_memory import (
        HostCacheGuard,
        inspect_host_availability,
        inspect_pinned_capacity,
        plan_host_cache,
    )
    from experiments.gr_cache_serving.src.preflight import audit_checkpoint
    from serving.echo_budget import GIB, plan_echo_budget

    model_path = args.model_path
    audit = audit_checkpoint(model_path, 5)
    budget = plan_echo_budget(
        mode=args.mode,
        num_layers=5,
        users=128,
        prefix_tokens=65536,
        suffix_tokens=1024,
        checkpoint_stored_bytes=audit["selected_stored_bytes"],
        total_hbm_bytes=72 * GIB,
        workspace_reserve_bytes=8 * GIB,
        non_torch_reserve_bytes=2 * GIB,
        device_cache_tokens=66624,
    )
    host_plan = plan_host_cache(
        users=128, layers=5, prefix=65536, suffix=1024, budget_bytes=66 * GIB
    )
    availability = inspect_host_availability(host_plan)
    inspect_pinned_capacity(budget.logical_pool_tokens * 5 * 1152)
    import torch

    from experiments.gr_cache_serving.src.memory_guard import MemoryGuard
    from serving.echo_cache_manager import EchoCacheManager
    from serving.echo_runner import open_echo_runner

    rows = []
    guard = MemoryGuard(torch, total_bytes=72 * GIB, non_torch_reserve_bytes=2 * GIB)
    host_guard = HostCacheGuard(torch, host_plan)
    with (
        guard,
        open_echo_runner(
            model_path=model_path,
            echo_path=ROOT / "3rdparty/ECHO",
            num_layers=5,
            mode=args.mode,
            max_total_tokens=budget.logical_pool_tokens,
            device_cache_tokens=66624,
            prefill_chunk=1024,
            mem_fraction_static=budget.torch_limit_bytes
            / torch.cuda.get_device_properties(0).total_memory,
            dense_context_tokens=budget.dense_context_tokens,
            dense_prefetch_schedule="attention_window" if args.mode == "dense_prefetch" else None,
            dense_prefetch_transport="cpu_staging" if args.mode == "dense_prefetch" else None,
        ) as runner,
    ):
        warm = EchoCacheManager(runner, host_capacity_tokens=budget.logical_pool_tokens)
        try:
            for _ in range(2):
                result = warm.execute(trace[0])
                if not torch.isfinite(result.hidden_states).all().item():
                    raise RuntimeError("nonfinite warmup")
                del result
        finally:
            warm.close()
        host_guard.sample(runner)
        with PartRanges(runner) as ranges:
            for repetition in ("control_before", "profile", "control_after"):
                manager = EchoCacheManager(runner, host_capacity_tokens=budget.logical_pool_tokens)
                seen = set()
                capturing = False
                try:
                    for ordinal, request in enumerate(trace[: args.capture_end]):
                        if repetition == "profile" and ordinal == args.capture_start:
                            torch.cuda.synchronize()
                            torch.cuda.profiler.start()
                            ranges.enabled = capturing = True
                        ranges.ordinal = ordinal
                        start = time.perf_counter_ns()
                        with ranges.span("request"):
                            result = manager.execute(request)
                        service_ms = (time.perf_counter_ns() - start) / 1e6
                        payload = ranges.finish_request()
                        if not torch.isfinite(result.hidden_states).all().item():
                            raise RuntimeError("nonfinite output")
                        row = {
                            "repetition": repetition,
                            "ordinal": ordinal,
                            "user_id": request["user_id"],
                            "first_visit": request["user_id"] not in seen,
                            "prefix_reused": result.prefix_reused,
                            "service_ms": service_ms,
                            **payload,
                        }
                        rows.append(row)
                        seen.add(request["user_id"])
                        del result
                        guard.check()
                        host_guard.sample(runner, manager)
                        print(
                            json.dumps(
                                {k: row[k] for k in ("repetition", "ordinal", "service_ms")}
                            ),
                            flush=True,
                        )
                finally:
                    if capturing:
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()
                    ranges.enabled = False
                    manager.close()
        provenance = runner.provenance
        if (
            runner._prefixes
            or runner.runner.token_to_kv_pool_allocator.available_size()
            != budget.logical_pool_tokens
        ):
            raise RuntimeError("profile replay did not release all logical slots")
    for path, digest in source.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source changed during measurement: {path}")
    args.output_dir.mkdir(parents=True)
    summary = {
        "mode": args.mode,
        "capture_start": args.capture_start,
        "capture_end": args.capture_end,
        "trace_metadata": metadata,
        "source_sha256": source,
        "runner_provenance": provenance,
        "budget_plan": budget.metadata(),
        "host_plan": host_plan,
        "host_availability": availability,
        "memory_guard": guard.metadata,
        "host_memory": host_guard.metadata(),
        "rows": rows,
        "scope": "NVTX/CUPTI diagnostic; instrumented before/after controls; no per-op synchronization; not a 512-request serving result",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in source:
        target = args.output_dir / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())


if __name__ == "__main__":
    main()
