"""Diagnostic Nsight captures of a cold request and the first 16-user revisit.

The accepted measurement run supplies exact inputs and independent HBM outputs.
Instrumentation adds NVTX and CPU timestamps, not CUDA events or tensor reductions.
Its request latency is diagnostic and must not replace the formal measurement.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.deepseek_v32_motivation.src import measure

ROOT = measure.ROOT
EXPERIMENT = measure.EXPERIMENT
REFERENCE = EXPERIMENT / "output/data/motivation_c10_20261004_u16_r2_01"
SCHEMA = "deepseek-v32-motivation-profile-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def capture_range(cudart):
    """Close a diagnostic capture while preserving a simultaneous body failure."""
    cudart.cudaProfilerStart()
    body_error = None
    try:
        yield
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            cudart.cudaProfilerStop()
        except BaseException as completion_error:
            if body_error is not None:
                raise BaseExceptionGroup(
                    "profile execution and capture completion both failed",
                    [body_error, completion_error],
                ) from None
            raise


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id", required=True)
    command.add_argument("--reference-run", type=Path, default=REFERENCE)
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--scheme", choices=measure.SCHEMES)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--nsys", action="store_true", help="use CUDA profiler capture boundaries")
    return command


def tensor_metadata(value):
    """Read allocation metadata only; never inspect CUDA tensor values."""
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "elements": value.numel(),
            "bytes": value.numel() * value.element_size(),
        }
    if value is None or type(value) in (bool, int, float, str):
        return value
    return {"type": type(value).__name__}


class Scopes:
    """Nested NVTX scopes plus inclusive host duration and shape annotations."""

    def __init__(self, scheme, phase, capture_index, *, nvtx=True, graph_operators=None):
        self.scheme, self.phase = scheme, phase
        self.capture_index = capture_index
        self.segment, self.chunk, self.layer = "request", "shared", "shared"
        self.calls, self.active = [], []
        self.nvtx = nvtx
        self.graph_operators = graph_operators
        self._chunk_scope = None
        self._chunk_index = -1
        self._last_layer = -1
        self._forward_start = self._forward_tokens = self._forward_chunk = 0

    @contextmanager
    def context(self, **values):
        previous = {key: getattr(self, key) for key in values}
        for key, value in values.items():
            setattr(self, key, value)
        try:
            yield
        finally:
            for key, value in previous.items():
                setattr(self, key, value)

    @contextmanager
    def __call__(self, stage, **details):
        import torch

        stage = {
            "indexer": "indexer_aux",
            "indexer_prefetch": "indexer_prefetch_aux",
            "sparse_mla": "sparse_mla_aux",
        }.get(stage, stage)
        values = (self.scheme, self.phase, self.segment, self.chunk, self.layer, stage)
        if any("/" in str(value) for value in values):
            raise ValueError("NVTX scope components cannot contain slashes")
        label = (
            f"motivation/{self.scheme}/{self.phase}/{self.segment}/"
            f"chunk_{self.chunk}/layer_{self.layer}/{stage}"
        )
        row = {
            "call_id": len(self.calls),
            "parent_call_id": self.active[-1] if self.active else None,
            "capture_index": self.capture_index,
            "scheme": self.scheme,
            "mode": self.scheme,
            "phase": self.phase,
            "segment": self.segment,
            "chunk": str(self.chunk),
            "layer": str(self.layer),
            "stage": stage,
            "nvtx": label,
            "useful_flops": None,
            "precision": None,
            **details,
        }
        self.calls.append(row)
        self.active.append(row["call_id"])
        if self.nvtx:
            torch.cuda.nvtx.range_push(label)
        row["cpu_start_ns"] = time.perf_counter_ns()
        try:
            yield row
            if self.graph_operators is not None:
                self.calls.extend(self.graph_operators.expand_replay(row, len(self.calls)))
        except BaseException:
            row["raised"] = True
            raise
        finally:
            row["cpu_inclusive_ns"] = time.perf_counter_ns() - row["cpu_start_ns"]
            if self.nvtx:
                torch.cuda.nvtx.range_pop()
            if self.active.pop() != row["call_id"]:
                raise RuntimeError("crossing instrumentation scopes")

    def close_chunk(self):
        if self._chunk_scope is not None:
            self._chunk_scope.__exit__(None, None, None)
            self._chunk_scope = None
        self.chunk, self.layer = "shared", "shared"

    @contextmanager
    def model_scope(self, stage):
        if stage == "embedding":
            self.close_chunk()
            self._chunk_index += 1
            self.chunk, self._last_layer = self._chunk_index, -1
            start = self._chunk_index * self._forward_chunk
            self._chunk_scope = self(
                "chunk",
                query_start=self._forward_start + start,
                query_tokens=min(self._forward_chunk, self._forward_tokens - start),
            )
            self._chunk_scope.__enter__()
        match = re.fullmatch(r"layer_(\d+)_source_(\d+)", stage)
        if match:
            self._last_layer = int(match[1])
            with self.context(layer=self._last_layer), self("layer", source_layer=int(match[2])):
                yield
        elif stage == "replay_input_copy":
            with self.context(layer=self._last_layer + 1), self(stage):
                yield
        else:
            with self(stage):
                yield


class InstrumentServing:
    """Wrap actual production calls only for the selected captured request."""

    def __init__(self, backend, scopes, runner):
        self.backend, self.scopes = backend, scopes
        self.runner = runner
        self.stack = ExitStack()
        self.targets, self.segment_counters = [], {}

    def wrap(self, target, name, stage, *, layer=None, segment=None, context=False):
        original = getattr(target, name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            values = {}
            if layer is not None:
                value = layer(args, kwargs)
                if value is not None:
                    values["layer"] = value
            if segment is not None:
                values.update(segment=segment, chunk="shared")
            details = {
                "arguments": [
                    tensor_metadata(value)
                    for value in (args[1:] if isinstance(target, type) else args)
                ],
                "keyword_arguments": {key: tensor_metadata(value) for key, value in kwargs.items()},
            }
            with self.scopes.context(**values), self.scopes(stage, **details):
                return original(*args, **kwargs)

        @contextmanager
        @wraps(original)
        def managed(*args, **kwargs):
            values = {}
            if layer is not None:
                value = layer(args, kwargs)
                if value is not None:
                    values["layer"] = value
            with (
                self.scopes.context(**values),
                self.scopes(stage),
                original(*args, **kwargs) as value,
            ):
                yield value

        self.stack.enter_context(patch.object(target, name, managed if context else wrapped))
        owner = getattr(target, "__name__", type(target).__name__)
        self.targets.append(f"{owner}.{name}")

    def __enter__(self):
        import torch

        from cache.prefix_pool import PrefixSessionPool
        from cache.sparse_token_cache import SparseTokenCache
        from cache.sparse_token_pool import SharedSparseTokenPool
        from experiments.deepseek_v32_echo_prefill.src.operator_instrumentation import (
            InstrumentOperators,
        )
        from models.deepseek_v32.cache.prefetch import PoolHistoryPrefetch
        from operators.common import kv_transfer

        backend, scopes = self.backend, self.scopes
        original_forward = backend._forward
        original_prefill = backend.prefill
        original_candidate = backend.extend_candidate
        original_metrics = backend.session_metrics

        def forward(session, ids, **kwargs):
            if kwargs.get("scope") is not None:
                raise ValueError("profiling requires ownership of the model scope callback")
            scopes._chunk_index = -1
            scopes._forward_start, scopes._forward_tokens = session.length, len(ids)
            scopes._forward_chunk = kwargs.get("chunk_size") or len(ids)
            with scopes("forward"):
                try:
                    return original_forward(session, ids, **{**kwargs, "scope": scopes.model_scope})
                finally:
                    scopes.close_chunk()

        def prefill(session, ids, **kwargs):
            with (
                scopes.context(segment="history", chunk="shared", layer="shared"),
                scopes("history_prefill"),
            ):
                result = original_prefill(session, ids, **kwargs)
            # The production forward already synchronized. This extra small-counter
            # read is visibly separated and never used as formal request timing.
            with (
                scopes.context(segment="diagnostics", chunk="shared", layer="shared"),
                scopes("history_counters"),
            ):
                self.segment_counters["history"] = original_metrics(session)
            return result

        def candidate(session, ids, **kwargs):
            with (
                scopes.context(segment="candidate", chunk="shared", layer="shared"),
                scopes("candidate_extend"),
            ):
                return original_candidate(session, ids, **kwargs)

        def metrics(session):
            # Intercept the runner's existing candidate diagnostics; no duplicate read.
            with (
                scopes.context(segment="diagnostics", chunk="shared", layer="shared"),
                scopes("candidate_counters"),
            ):
                result = original_metrics(session)
                self.segment_counters["candidate"] = result
                return result

        for name, replacement in (
            ("_forward", forward),
            ("prefill", prefill),
            ("extend_candidate", candidate),
            ("session_metrics", metrics),
        ):
            self.stack.enter_context(patch.object(backend, name, replacement))
            self.targets.append(f"DeepSeekServingBackend.{name}")
        self.wrap(self.runner, "_validate", "request_validation", segment="admission")
        for name in ("acquire", "mark_ready", "discard"):
            self.wrap(PrefixSessionPool, name, f"prefix_pool_{name}", segment="admission")
        self.wrap(PrefixSessionPool, "audit", "prefix_pool_audit", segment="cleanup")
        for name in ("create_session", "release_session"):
            self.wrap(backend, name, f"backend_{name}", segment="admission")
        self.wrap(backend, "truncate", "backend_truncate", segment="cleanup")
        self.wrap(backend, "synchronize", "backend_synchronize")

        def cache_layer(args, kwargs):
            return getattr(args[0], "layer_id", None)

        for name in (
            "append",
            "ensure",
            "_available_slots",
            "_evict",
            "logical_to_global",
            "prepare_prefetch",
            "finalize_prefetch",
            "begin_step",
            "begin_transient",
            "commit",
            "discard_transient",
            "truncate",
            "rollback",
            "metrics",
        ):
            self.wrap(SparseTokenCache, name, f"cache_{name.lstrip('_')}", layer=cache_layer)
        self.wrap(SparseTokenCache, "operation", "cache_operation", layer=cache_layer, context=True)
        pool_layer_positions = {
            "write_host": (2, "layer"),
            "wait_host": (2, "layer"),
            "protect": (1, "layer_id"),
            "stamp": (1, "layer_id"),
            "release_ids": (1, "layer_id"),
            "finalize_prefetch": (1, "layer_id"),
            "operation": (2, "layer"),
        }
        for name in (
            "allocate_session",
            "release_session",
            "write_host",
            "wait_host",
            "drain",
            "_reap_writes",
            "_wait_previous",
            "protect",
            "stamp",
            "release_ids",
            "finalize_prefetch",
            "operation",
        ):
            position, key = pool_layer_positions.get(name, (None, None))

            def pool_layer(args, kwargs, position=position, key=key):
                return (
                    args[position]
                    if position is not None and len(args) > position
                    else kwargs.get(key)
                )

            self.wrap(
                SharedSparseTokenPool,
                name,
                f"pool_{name.lstrip('_')}",
                layer=pool_layer,
                context=name == "operation",
            )
        for name in ("prefetch", "wait"):

            def target_layer(args, kwargs, name=name):
                target = (
                    args[1]
                    if len(args) > 1
                    else kwargs["cache" if name == "prefetch" else "ticket"]
                )
                return (target if name == "prefetch" else target._cache).layer_id

            self.wrap(PoolHistoryPrefetch, name, f"pool_history_{name}", layer=target_layer)
        self.wrap(PoolHistoryPrefetch, "drain", "pool_history_drain")
        original_gather = kv_transfer.gather_host_records

        @wraps(original_gather)
        def gather(host, device, host_ids, device_ids):
            record_bytes = host.shape[1] * host.element_size()
            with scopes(
                "host_gather",
                records=host_ids.numel(),
                record_bytes=record_bytes,
                requested_bytes=host_ids.numel() * record_bytes,
            ):
                return original_gather(host, device, host_ids, device_ids)

        self.stack.enter_context(patch.object(kv_transfer, "gather_host_records", gather))
        self.targets.append("operators.common.kv_transfer.gather_host_records")
        for name in ("argsort", "unique"):
            self.wrap(torch, name, f"cache_torch_{name}")
        shim = SimpleNamespace(
            blocks=[
                SimpleNamespace(attention=SimpleNamespace(attention=attn), mlp=block.mlp)
                for attn, block in zip(backend.attentions, backend.blocks, strict=True)
            ],
            head_weight=backend.head_weight,
        )
        operators = self.stack.enter_context(InstrumentOperators(shim, scopes))
        graphs = getattr(backend, "_compute_graphs", None)
        if graphs is not None:
            from experiments.deepseek_v32_motivation.src.graph_instrumentation import (
                instrument_graph_queries,
            )

            self.stack.enter_context(instrument_graph_queries(graphs, operators))
            self.targets.append("DeepSeekComputeGraphs.forward_block: replay Q origin metadata")
        self.targets.append(
            "InstrumentOperators: actual checkpoint linears, MLA, indexer and auxiliaries"
        )
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def snapshot_sources(output):
    measure.snapshot_sources(output)
    manifest = json.loads((output / "source_manifest.json").read_text())
    for name in (
        "operator_instrumentation.py",
        "operator_flops.py",
        "analyze_nsys.py",
    ):
        path = ROOT / "experiments/deepseek_v32_echo_prefill/src" / name
        relative = path.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[str(relative)] = sha(path)
    measure.write_json(output / "source_manifest.json", manifest)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def load_reference(path):
    from GR.workload import Workload, token_sha256

    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("status") != "accepted" or metadata.get("schema") not in (
        measure.SCHEMA,
        measure.CHECK_SCHEMA,
    ):
        raise ValueError("reference must be an accepted independent check or legacy measurement")
    if metadata["schema"] == measure.CHECK_SCHEMA:
        from experiments.deepseek_v32_motivation.src.report import audit_run

        audit_run(path)
    config = metadata["config"]
    if (
        config.get("warmup_policy") != measure.WARMUP_POLICY
        or config.get("byte_subbudgets") is not None
    ):
        raise ValueError("reference must use host-recall warmup and fixed pools")
    manifest = json.loads((path / "workload/workload.json").read_text())
    requests = tuple(
        json.loads(line) for line in (path / "workload/requests.jsonl").read_text().splitlines()
    )
    identity = {
        key: manifest[key] for key in ("config", "heat_sha256", "tokenizer_sha256", "requests")
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if digest != manifest["workload_sha256"] or digest != metadata["workload_sha256"]:
        raise ValueError("reference workload identity mismatch")
    if len(requests) != config["requests_per_scheme"]:
        raise ValueError("reference request count mismatch")
    for index, request in enumerate(requests):
        if request["request_id"] != index or request["input_sha256"] != token_sha256(
            request["input_ids"]
        ):
            raise ValueError("reference request identity mismatch")
        if any(request[key] != value for key, value in manifest["requests"][index].items()):
            raise ValueError("saved request differs from the signed workload manifest")
    reference_sources = json.loads((path / "source_manifest.json").read_text())
    source_digest = hashlib.sha256(
        json.dumps(reference_sources, sort_keys=True).encode()
    ).hexdigest()
    if source_digest != metadata["source_sha256"]:
        raise ValueError("reference source manifest identity mismatch")
    # Profiling and report helpers are new, but the executed model/cache path is frozen.
    changed = [
        name
        for name, digest in reference_sources.items()
        if Path(name).parts[0]
        in {"models", "cache", "operators", "executor", "serving", "3rdparty"}
        and (not (ROOT / name).is_file() or sha(ROOT / name) != digest)
    ]
    if changed:
        raise ValueError(f"production differs from the accepted reference: {changed}")
    hbm = {
        row["request_id"]: row
        for line in (path / "measurements.jsonl").read_text().splitlines()
        if (row := json.loads(line))["scheme"] == "hbm"
    }
    return metadata, Workload(requests, manifest), hbm


def precision_settings():
    import torch

    settings = {"float32_matmul_precision": torch.get_float32_matmul_precision()}
    for label, owner, name in (
        ("cuda_matmul_allow_tf32", torch.backends.cuda.matmul, "allow_tf32"),
        ("cuda_matmul_fp32_precision", torch.backends.cuda.matmul, "fp32_precision"),
        ("backends_fp32_precision", torch.backends, "fp32_precision"),
        ("cudnn_allow_tf32", torch.backends.cudnn, "allow_tf32"),
        ("cudnn_fp32_precision", torch.backends.cudnn, "fp32_precision"),
        (
            "bf16_reduced_precision_reduction",
            torch.backends.cuda.matmul,
            "allow_bf16_reduced_precision_reduction",
        ),
        (
            "fp16_reduced_precision_reduction",
            torch.backends.cuda.matmul,
            "allow_fp16_reduced_precision_reduction",
        ),
    ):
        try:
            settings[label] = getattr(owner, name)
        except (AttributeError, RuntimeError) as error:
            settings[label] = {"unavailable": str(error)}
    return settings


def cpu_summary(calls):
    groups = defaultdict(lambda: {"calls": 0, "cpu_inclusive_ns": 0, "useful_flops": 0})
    for call in calls:
        key = (call["scheme"], call["phase"], call["segment"], call["stage"])
        row = groups[key]
        row["calls"] += 1
        row["cpu_inclusive_ns"] += call["cpu_inclusive_ns"]
        row["useful_flops"] += call.get("useful_flops") or 0
    return [
        dict(zip(("scheme", "phase", "segment", "stage"), key, strict=True), **value)
        for key, value in sorted(groups.items())
    ]


def reference_provenance(requested, numerical):
    """Retain the formal run while identifying its independent numerical source."""
    result = {}
    configurations = []
    for prefix, path in (("reference", requested), ("numerical_reference", numerical)):
        path = Path(path).resolve(strict=True)
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata.get("status") != "accepted":
            raise ValueError("profile reference is not accepted")
        configurations.append(metadata["config"])
        result.update(
            {
                prefix + "_run": str(path),
                prefix + "_run_id": metadata["run_id"],
                prefix + "_source_sha256": metadata["source_sha256"],
                prefix + "_metadata_sha256": sha(path / "metadata.json"),
            }
        )
    if configurations[0] != configurations[1]:
        raise ValueError("formal and numerical profile reference configurations differ")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run ID must contain letters, digits, underscores or hyphens")
    from experiments.nosa_motivation.src.validation import reference_directory

    reference = reference_directory(
        args.reference_run, bench_schema=measure.BENCH_SCHEMA, kind=measure.RECEIPT_KIND
    ).resolve(strict=True)
    reference_metadata, workload, hbm_rows = load_reference(reference)
    config = reference_metadata["config"]
    graph_enabled = config.get("enable_compute_graphs", False)
    if graph_enabled and not args.nsys:
        raise ValueError(
            "compute graph attribution requires Nsight with node tracing and setup captures"
        )
    schemes = [args.scheme] if args.scheme else list(measure.SCHEMES)
    target = args.output_dir or EXPERIMENT / "output/data" / args.run_id
    if target.exists() or args.run_id == reference_metadata["run_id"]:
        raise FileExistsError(target)
    output = Path(tempfile.mkdtemp(prefix=f"deepseek-motivation-profile-{args.run_id}-"))
    print(f"temporary output: {output}", flush=True)
    metadata = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "status": "running",
        "config": config,
        "schemes": schemes,
        "nsys": args.nsys,
        **reference_provenance(args.reference_run, reference),
        "workload_sha256": workload.manifest["workload_sha256"],
        "warmup_traces": {},
        "captures": [],
        "graph_setup_captures": [],
        "cases": [],
        "started_unix": time.time(),
        "execution_boundary": "Three warmups, release all resources, then requests 0..num_users; capture only 0 and num_users. All first-round users execute before revisit.",
        "instrumentation_boundary": "NVTX plus inclusive CPU perf_counter timestamps and tensor metadata; no added CUDA events or GPU reductions for annotations. Production synchronization remains unchanged. Instrumented latency is diagnostic, not formal performance.",
        "counter_boundary": "One added ordinary session_metrics call after completed history prefill, labeled diagnostics/history_counters. Existing runner candidate metrics intercepted once. Heavy collect_cache_diagnostics is disabled.",
        "chunk_boundary": "Each chunk opens immediately before embedding, closes before the next embedding; the last includes forward final concatenation, synchronization and commit. Layer scopes cover the actual independent replay block.",
        "flop_boundary": "Actual operator call shapes and checkpoint weight precision; final norm runs every chunk, while LM head runs once at the last history chunk and once for candidate.",
        "cpu_duration_boundary": "Inclusive nested host durations cannot be added; Nsight CUDA activity correlation supplies GPU work and synchronization attribution.",
        "graph_attribution_boundary": "Graph setup is traced separately from requests. Capture-time CUDA node sets identify matrix API work; Nsight clone lineage resolves those nodes in observed replays. Synthetic replay API rows claim no CPU duration or matrix NVTX range.",
        "missing_optional_instrumentation": {
            "SparseTokenCache._invalidate": "not an API; actual invalidation is _evict and pool release_ids"
        },
    }
    backend = None
    failure = None
    all_calls, evidence = [], []
    try:
        import torch

        from evaluation.provenance import (
            _git,
            backend_provenance,
            numerical_comparison,
            verify_source_snapshot,
        )
        from models.deepseek_v32.execution.adapter import DeepSeekServingBackend
        from serving.persistent import PersistentGRRunner

        device = torch.device(args.device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("profiling requires SM90/Hopper")
        torch.cuda.set_device(device)
        measure.configure_precision(torch)
        os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
        metadata.update(
            source_sha256=snapshot_sources(output),
            git_revision=_git("rev-parse", "HEAD"),
            git_status=_git("status", "--short"),
            backend_provenance=backend_provenance(),
            precision_settings=precision_settings(),
        )
        if metadata["source_sha256"] == metadata["reference_source_sha256"]:
            raise AssertionError("new profile instrumentation requires its own source identity")
        props = torch.cuda.get_device_properties(device)
        metadata["hardware"] = {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "name": props.name,
            "uuid": str(props.uuid),
            "total_memory": props.total_memory,
            "sm_count": props.multi_processor_count,
        }
        model_path = Path(reference_metadata["checkpoint"]["path"])
        checkpoint = {
            "path": str(model_path.resolve()),
            "files": {
                path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                for path in sorted(model_path.glob("*.safetensors"))
            },
            "identity_boundary": "checkpoint path and shard stat inventory, not weights hashes",
        }
        if checkpoint != reference_metadata["checkpoint"]:
            raise ValueError("checkpoint stat identity differs from the formal reference")
        metadata["checkpoint"] = checkpoint
        workload.write(output / "workload")
        measure.write_json(output / "metadata.json", metadata)
        backend = DeepSeekServingBackend(
            model_path,
            scheme=schemes[0],
            device=device,
            num_layers=config["layers"],
            chunk_size=config["chunk_size"],
            sparse_pool_tokens=config["sparse_pool_tokens"],
            host_arena_tokens=config["host_arena_tokens"],
            workspace_query_tokens=config["workspace_query_tokens"],
            enable_compute_graphs=graph_enabled,
        )
        backend.synchronize()

        def validate(result, request, phase):
            index = request["request_id"]
            payload = measure.tensor_payload(backend, result, request, config)
            expected_path = reference / hbm_rows[index]["output_file"]
            if sha(expected_path) != hbm_rows[index]["output_sha256"]:
                raise ValueError("reference HBM output digest mismatch")
            expected = torch.load(expected_path, weights_only=True, map_location="cpu")
            if expected["input_sha256"] != payload["input_sha256"]:
                raise ValueError("reference HBM token identity mismatch")
            comparison = {
                key: numerical_comparison(payload[key], expected[key], atol=0, rtol=0)
                for key in ("hidden", "logits")
            }
            destination = output / "numerical" / backend.scheme / phase / f"{index:06d}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, destination)
            row = {
                "scheme": backend.scheme,
                "phase": phase,
                "request_id": index,
                "input_sha256": request["input_sha256"],
                "numerical": comparison,
                "reference_output_sha256": hbm_rows[index]["output_sha256"],
                "output_file": str(destination.relative_to(output)),
                "output_sha256": sha(destination),
                "metrics": result.metrics,
            }
            evidence.append(row)
            with (output / "correctness.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

        class WarmupRunner(PersistentGRRunner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, native_token_validation=True, **kwargs)

            def execute(self, request):
                result = super().execute(request)
                validate(result, request, "warmup")
                return result

        for scheme in schemes:
            backend.configure_scheme(scheme)
            metadata.update(stage="warmup", active_scheme=scheme)
            measure.write_json(output / "metadata.json", metadata)
            print(f"warmup {scheme}: user0, user1, user0 host recall", flush=True)
            metadata["warmup_traces"][scheme] = measure.warmup(
                backend, workload, config, WarmupRunner
            )
            gc.collect()
            torch.cuda.empty_cache()
            graph_operators = None
            if graph_enabled:
                from experiments.deepseek_v32_motivation.src.graph_instrumentation import (
                    CaptureGraphOperators,
                )

                setup_index = len(metadata["captures"]) + len(metadata["graph_setup_captures"]) + 1
                runner = None
                try:
                    with capture_range(torch.cuda.cudart()):
                        with CaptureGraphOperators(backend) as graph_operators:
                            runner = PersistentGRRunner(
                                backend,
                                resource_limits=measure.resource_limits(config),
                                native_token_validation=True,
                            )
                        graph_ledger = graph_operators.finalize()
                    metadata["graph_setup_captures"].append(
                        {
                            "scheme": scheme,
                            "trace_index": setup_index,
                            "sqlite": f"capture_{setup_index}.sqlite",
                            **graph_ledger,
                        }
                    )
                    measure.write_json(
                        output / "graph_capture_ledger.json", metadata["graph_setup_captures"]
                    )
                except BaseException as setup_error:
                    if runner is not None:
                        try:
                            runner.close()
                        except BaseException as cleanup_error:  # noqa: BLE001 -- preserve setup
                            raise BaseExceptionGroup(
                                "graph profile setup and runner cleanup both failed",
                                [setup_error, cleanup_error],
                            ) from None
                    raise
            else:
                runner = PersistentGRRunner(
                    backend,
                    resource_limits=measure.resource_limits(config),
                    native_token_validation=True,
                )
            with runner:
                if len(runner.pool):
                    raise AssertionError("capture trajectory must start with an empty session pool")
                case = {
                    "scheme": scheme,
                    "started_empty": True,
                    "request_ids": list(range(config["num_users"] + 1)),
                    "resource_plan": dict(runner.resource_plan.metadata),
                    "token_validation": getattr(runner, "token_validation_identity", None),
                    "backend": backend.describe(),
                }
                for index in range(config["num_users"] + 1):
                    request = workload.requests[index]
                    phase = (
                        "cold"
                        if index == 0
                        else "revisit"
                        if index == config["num_users"]
                        else "prepare"
                    )
                    metadata.update(stage=phase, active_request=index)
                    measure.write_json(output / "metadata.json", metadata)
                    graphs_before = backend.describe()["compute_graphs"] if graph_enabled else None
                    if phase == "prepare":
                        result = runner.execute(request)
                    else:
                        capture_index = (
                            len(metadata["captures"]) + len(metadata["graph_setup_captures"]) + 1
                        )
                        scopes = Scopes(
                            scheme, phase, capture_index, graph_operators=graph_operators
                        )
                        instrument = InstrumentServing(backend, scopes, runner)
                        backend.synchronize()
                        with (
                            instrument,
                            capture_range(torch.cuda.cudart()) if args.nsys else nullcontext(),
                            scopes("request", request_id=index, user_id=request["user_id"]),
                        ):
                            result = runner.execute(request)
                        if scopes.active:
                            raise AssertionError("unclosed request scopes")
                        all_calls.extend(scopes.calls)
                        metadata["captures"].append(
                            {
                                "capture_index": capture_index,
                                "scheme": scheme,
                                "phase": phase,
                                "request_id": index,
                                "sqlite": f"capture_{capture_index}.sqlite",
                                "call_count": len(scopes.calls),
                                "instrumentation_targets": instrument.targets,
                                "segment_counters": instrument.segment_counters,
                                "diagnostic_runner_latency_ms": result.metrics["latency_ms"],
                                "runner_metrics": {
                                    "prefix_cache_hit": result.metrics["prefix_cache_hit"],
                                },
                            }
                        )
                        measure.write_json(output / "operator_calls.json", all_calls)
                        measure.write_json(
                            output / "cpu_stage_summary.json", cpu_summary(all_calls)
                        )
                    if graph_enabled:
                        measure.check_compute_graph_replays(
                            graphs_before,
                            backend.describe()["compute_graphs"],
                            config,
                            result.metrics,
                        )
                    measure.check_request(request, result.metrics, config)
                    validate(result, request, phase)
                    print(
                        f"{scheme} {phase} request={index} hit={result.metrics['prefix_cache_hit']} "
                        f"exact_outputs=True",
                        flush=True,
                    )
                    del result
                metadata["cases"].append(case)
            backend.close()
            measure.write_json(output / "metadata.json", metadata)
        expected_outputs = len(schemes) * (3 + config["num_users"] + 1)
        if len(evidence) != expected_outputs or len(metadata["captures"]) != 2 * len(schemes):
            raise AssertionError("incomplete profile execution")
        verify_source_snapshot(output)
        if backend_provenance() != metadata["backend_provenance"]:
            raise RuntimeError("installed backend identity changed during profiling")
        if precision_settings() != metadata["precision_settings"]:
            raise RuntimeError("numerical precision policy changed during profiling")
        from experiments.deepseek_v32_echo_prefill.src.backend_provenance import (
            collect_flashinfer_runtime_artifacts,
        )

        metadata.update(
            status="accepted",
            stage="complete",
            completed_unix=time.time(),
            exact_output_count=len(evidence),
            source_verified_after_execution=True,
            flashinfer_runtime_artifacts=collect_flashinfer_runtime_artifacts(),
        )
        measure.write_json(output / "metadata.json", metadata)
        measure.write_json(
            output / "result.json",
            {
                "accepted": True,
                "run_id": args.run_id,
                "captures": len(metadata["captures"]),
                "setup_captures": len(metadata["graph_setup_captures"]),
                "nsys_captures": len(metadata["captures"]) + len(metadata["graph_setup_captures"]),
                "exact_output_count": len(evidence),
                "source_sha256": metadata["source_sha256"],
                "reference_source_sha256": metadata["reference_source_sha256"],
            },
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(target))
        print(f"accepted profile: {target}", flush=True)
    except BaseException as error:
        failure = error
        try:
            metadata.update(status="failed", error=repr(error))
            measure.write_json(output / "metadata.json", metadata)
            print(f"failed profile retained outside experiments: {output}", flush=True)
        except BaseException as reporting_error:  # noqa: BLE001 -- retain original and report errors
            failure = BaseExceptionGroup(
                "DeepSeek profile and failure reporting both failed", [error, reporting_error]
            )
            raise failure from None
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as cleanup_error:
                if failure is not None:
                    raise BaseExceptionGroup(
                        "DeepSeek profile and resource cleanup both failed",
                        [failure, cleanup_error],
                    ) from None
                raise


if __name__ == "__main__":
    main()
