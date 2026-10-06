"""Observational NVTX wrappers for the unchanged official Engine scheduler.

Importing this module does not import torch or SGLang. Install the picklable
``scheduler_with_profile`` as the Engine module's scheduler spawn target.
Only profile runs use these hooks; independent benchmark runs do not.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _cpu_ints(value, name):
    """Read existing CPU metadata, rejecting accidental device scalar reads."""
    if value is None:
        return None
    device = getattr(value, "device", None)
    if device is not None and device.type != "cpu":
        raise ValueError(f"{name} must already be on CPU")
    values = value.tolist() if hasattr(value, "tolist") else list(value)
    result = []
    for item in values:
        device = getattr(item, "device", None)
        if device is not None and device.type != "cpu":
            raise ValueError(f"{name} contains a device scalar")
        result.append(int(item))
    return result


def describe_forward(batch, history_tokens=65536, candidate_tokens=128):
    """Describe geometry without tensor copies, new kernels or synchronization."""
    lengths = _cpu_ints(getattr(batch, "seq_lens_cpu", None), "seq_lens_cpu")
    queries = _cpu_ints(getattr(batch, "extend_seq_lens_cpu", None), "extend_seq_lens_cpu")
    prefixes = _cpu_ints(getattr(batch, "extend_prefix_lens_cpu", None), "extend_prefix_lens_cpu")
    request_indices = _cpu_ints(
        getattr(batch, "req_pool_indices_cpu", None), "req_pool_indices_cpu"
    )
    mode = getattr(batch, "forward_mode", None)
    mode_name = getattr(mode, "name", str(mode))
    metadata = {
        "batch_size": int(batch.batch_size),
        "mode": mode_name,
        "seq_lens": lengths,
        "query_lens": queries,
        "prefix_lens": prefixes,
        "request_pool_indices": request_indices,
        "q": int(batch.input_ids.shape[0]),
        "start": -1,
        "end": -1,
        "phase": "setup",
    }
    if lengths is None or queries is None or prefixes is None:
        return metadata
    if not (len(lengths) == len(queries) == len(prefixes) == batch.batch_size):
        raise ValueError("Inconsistent existing forward CPU metadata")
    if batch.batch_size != 1:
        metadata["phase"] = "other"
        return metadata
    q, start, end = queries[0], prefixes[0], lengths[0]
    if q + start != end:
        raise ValueError("Query and prefix lengths do not reach the sequence end")
    metadata.update(q=q, start=start, end=end, phase="other")
    if 0 <= start < end <= history_tokens:
        metadata["phase"] = "prefill"
    elif start == history_tokens and q == candidate_tokens:
        metadata["phase"] = "extend"
    return metadata


@contextmanager
def nvtx_range(nvtx, label):
    """Preserve a computation error if closing the NVTX range also fails."""
    nvtx.range_push(label)
    primary = None
    try:
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            nvtx.range_pop()
        except BaseException as error:
            if primary is not None:
                raise BaseExceptionGroup("Forward and NVTX cleanup failed", [primary, error])
            raise


class _Profiler:
    def __init__(self, torch, output, case):
        self.torch = torch
        self.output = output
        self.case = case
        self.local = threading.local()
        self.records = []
        self.patches = []
        self.last_batch = None
        self.last_metadata = None
        self.snapshots = []
        self.history_tokens = 65536
        self.candidate_tokens = 128
        self.recording_enabled = False
        self.formal_begin = None

    @property
    def current(self):
        return getattr(self.local, "current", None)

    def label(self, kind, **fields):
        values = {"kind": kind}
        frame = self.current
        if frame is not None:
            values.update(id=frame["id"], layer=frame.get("layer", -1))
        values.update(fields)
        return "echo_engine|" + "|".join(f"{key}={value}" for key, value in values.items())

    @contextmanager
    def stage(self, stage, **fields):
        if self.current is None:
            yield
            return
        counts = self.current["stage_calls"]
        counts[stage] = counts.get(stage, 0) + 1
        with nvtx_range(self.torch.cuda.nvtx, self.label("stage", stage=stage, **fields)):
            yield

    def patch(self, owner, name, make_wrapper):
        original = getattr(owner, name)
        wrapped = functools.wraps(original)(make_wrapper(original))
        setattr(owner, name, wrapped)
        binding = "python_source"
        try:
            source = inspect.getsourcefile(original)
        except TypeError:
            module = sys.modules.get(getattr(original, "__module__", ""))
            source = getattr(module, "__file__", None) or getattr(owner, "__file__", None)
            binding = "native_callable_module"
        self.patches.append(
            {
                "target": f"{owner.__name__}.{name}",
                "source": source,
                "source_sha256": _sha(source) if source else None,
                "source_binding": binding,
                "body_changed": False,
            }
        )

    def stage_wrapper(self, name):
        def decorate(original):
            def call(*args, **kwargs):
                stage = name() if callable(name) else name
                with self.stage(stage):
                    return original(*args, **kwargs)

            return call

        return decorate

    def forward_wrapper(self, original):
        def call(runner, forward_batch, *args, **kwargs):
            if not self.recording_enabled:
                return original(runner, forward_batch, *args, **kwargs)
            metadata = describe_forward(forward_batch, self.history_tokens, self.candidate_tokens)
            metadata.update(id=len(self.records), stage_calls={}, layers=[], completed=False)
            self.records.append(metadata)
            previous = self.current
            self.local.current = metadata
            try:
                with nvtx_range(
                    self.torch.cuda.nvtx,
                    self.label(
                        "forward",
                        phase=metadata["phase"],
                        q=metadata["q"],
                        start=metadata["start"],
                        end=metadata["end"],
                        case=self.case,
                    ),
                ):
                    result = original(runner, forward_batch, *args, **kwargs)
                metadata["completed"] = True
                self.last_batch = forward_batch
                self.last_metadata = metadata
                return result
            finally:
                self.local.current = previous

        return call

    def layer_wrapper(self, original):
        def call(layer, *args, **kwargs):
            if self.current is None:
                return original(layer, *args, **kwargs)
            frame = self.current
            previous = frame.get("layer", -1)
            layer_id = int(layer.layer_id)
            frame["layer"] = layer_id
            frame["layers"].append(layer_id)
            try:
                with nvtx_range(self.torch.cuda.nvtx, self.label("layer", layer=layer_id)):
                    return original(layer, *args, **kwargs)
            finally:
                frame["layer"] = previous

        return call

    def ragged_wrapper(self, original):
        def call(*args, **kwargs):
            previous = getattr(self.local, "in_ragged", False)
            self.local.in_ragged = True
            try:
                with self.stage("indexer_control"):
                    return original(*args, **kwargs)
            finally:
                self.local.in_ragged = previous

        return call

    def write_wrapper(self, original):
        def call(*args, **kwargs):
            destination = args[0] if args else kwargs["kv_buffer"]
            direction = "D2H" if destination.device.type == "cpu" else "device"
            stage = "io_d2h" if direction == "D2H" else "cache_write"
            with self.stage(stage, direction=direction):
                return original(*args, **kwargs)

        return call

    def persist(self):
        _write_json(self.output / "forwards.json", self.records)
        _write_json(
            self.output / "hook_manifest.json",
            {
                "schema": "echo-engine-nvtx-hooks-v1",
                "pid": os.getpid(),
                "case": self.case,
                "hook_source": {"path": str(Path(__file__).resolve()), "sha256": _sha(__file__)},
                "patched_callables": self.patches,
                "forward_count": len(self.records),
                "snapshots": self.snapshots,
                "numerical_acceptance": False,
                "recording_enabled": self.recording_enabled,
                "formal_begin": self.formal_begin,
                "boundary": "Intrusive profile only. Wrappers call each original exactly once; no body rewriting, tensor replay, synchronization or device readback inside forward wrappers. Independent Engine timings run without hooks.",
                "io_boundary": "GPU memcpy or kernel activities supply duration. io_d2h identifies a CPU destination passed to the unchanged mapped-host write kernel. io_h2d and indexer_prefetch are potential mapped-host read paths with dynamic traffic; API scope duration does not measure transfer time or bytes.",
                "control_boundary": "Inline ECHO prepare and hint are retained in indexer_control. Only callable allocator post_alloc and ragged update_priority carry echo_finalize; no inline body is rewritten.",
            },
        )

    def begin(self, parameters):
        """Enable observation after warmup; do not touch model/cache/CUDA state."""
        if (
            parameters
            or self.recording_enabled
            or self.records
            or self.snapshots
            or self.current is not None
        ):
            raise ValueError("Formal profiling must begin once, before any recorded forward")
        self.formal_begin = {
            "schema": "echo-engine-formal-profile-begin-v1",
            "record_count_before": 0,
            "first_forward_id": 0,
            "cache_mutated": False,
            "scope": "Recording enabled after synthetic warmup and before CUDAProfilerStart. Only profiler bookkeeping changes; no model/cache mutation, tensor readback or CUDA synchronization.",
        }
        self.recording_enabled = True
        _write_json(self.output / "formal_begin.json", self.formal_begin)
        self.persist()

    def snapshot(self, parameters):
        """Explicit profile-only RPC, invoked after the Engine request timer closes."""
        phase = parameters["phase"]
        if phase not in {"prefill", "extend"}:
            raise ValueError("Expected prefill or extend snapshot")
        if phase in self.snapshots:
            raise ValueError(f"Duplicate snapshot: {phase}")
        batch, metadata = self.last_batch, self.last_metadata
        expected_end = self.history_tokens + (self.candidate_tokens if phase == "extend" else 0)
        if batch is None or metadata["phase"] != phase or metadata["end"] != expected_end:
            raise ValueError("Snapshot does not follow the expected completed phase")
        if metadata["layers"] != [0, 1, 2] or not metadata["completed"]:
            raise ValueError("Profile forward did not execute exactly layers 0, 1, 2")
        request_indices = metadata["request_pool_indices"]
        if request_indices is None or len(request_indices) != 1:
            raise ValueError("Snapshot needs an existing single CPU request index")
        with nvtx_range(
            self.torch.cuda.nvtx,
            self.label("diagnostic", phase=phase, purpose="post_request_residency"),
        ):
            self.torch.cuda.synchronize()
            logical_ids_tensor = (
                batch.req_to_token_pool.req_to_token[request_indices[0], :expected_end]
                .detach()
                .cpu()
            )
            logical_ids = [int(value) for value in logical_ids_tensor.tolist()]
            if len(set(logical_ids)) != expected_end:
                raise ValueError("Snapshot request mapping contains duplicate token slots")
            pool = batch.token_to_kv_pool
            result = {
                "schema": "echo-engine-post-request-residency-v1",
                "phase": phase,
                "forward_id": metadata["id"],
                "sequence_tokens": expected_end,
                "history_tokens": self.history_tokens,
                "pool_class": type(pool).__name__,
                "logical_ids_sha256": hashlib.sha256(
                    logical_ids_tensor.numpy().tobytes()
                ).hexdigest(),
                "boundary": "Explicit synchronized snapshot after Engine.generate returned and outside request timing/NVTX and all forward NVTX scopes. It observes residency at this point only; it does not measure per-layer transfer bytes or prove residency during preceding work.",
                "layers": [],
            }
            if hasattr(pool, "host_token_to_device"):
                # The retained request slot has not been reused between generate and this RPC.
                # Select only the observed request's logical IDs, never copy the full host arena.
                lookup_ids = self.torch.tensor(
                    logical_ids, device=pool.host_token_to_device.device, dtype=self.torch.long
                )
                mapped = pool.host_token_to_device[:, lookup_ids].detach().cpu().tolist()
                reverse = pool.device_token_to_host.detach().cpu().tolist()
                device_tokens = int(pool.device_pool.size)
                for layer, slots in enumerate(mapped):
                    valid = [
                        0 < int(slot) <= device_tokens and reverse[layer][int(slot)] == host_id
                        for host_id, slot in zip(logical_ids, slots)
                    ]
                    result["layers"].append(
                        {
                            "layer": layer,
                            "history_resident_tokens": sum(valid[: self.history_tokens]),
                            "candidate_resident_tokens": sum(valid[self.history_tokens :]),
                            "all_history_resident": all(valid[: self.history_tokens]),
                            "all_sequence_resident": all(valid),
                            "bidirectional_mapping_checked": True,
                        }
                    )
                result["device_pool_tokens"] = device_tokens
                result["host_pool_tokens"] = int(pool.size)
                result["last_residual_recall_counter"] = int(pool.recall_counter.detach().cpu()[0])
                result["recall_counter_boundary"] = (
                    "Only the last layer's final residual recall; reset within each recall and excludes prior fused prefetch. Not total H2D traffic."
                )
            else:
                valid = all(
                    0 <= slot < int(pool.size) + int(pool.page_size) for slot in logical_ids
                )
                if not valid:
                    raise ValueError("Resident request mapping exceeds the device pool")
                result["device_pool_tokens"] = int(pool.size)
                result["layers"] = [
                    {
                        "layer": layer,
                        "history_resident_tokens": self.history_tokens,
                        "candidate_resident_tokens": expected_end - self.history_tokens,
                        "all_history_resident": True,
                        "all_sequence_resident": True,
                        "bidirectional_mapping_checked": False,
                        "basis": "Resident-only pool and in-range retained request token slots",
                    }
                    for layer in metadata["layers"]
                ]
        _write_json(self.output / f"snapshot_{phase}.json", result)
        self.snapshots.append(phase)
        self.persist()


def install_hooks():
    import deep_gemm
    import torch
    from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
    from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend, NSAIndexerMetadata
    from sglang.srt.layers.communicator import LayerCommunicator
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache import memory_pool, memory_pool_host
    from sglang.srt.mem_cache.allocator import CudaGraphTokenToKVPoolAllocator
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.models.deepseek_v2 import (
        DeepseekV2AttentionMLA,
        DeepseekV2DecoderLayer,
        DeepseekV2MLP,
    )

    output = Path(os.environ["ECHO_ENGINE_PROFILE_DIR"])
    case = os.environ["ECHO_ENGINE_PROFILE_CASE"]
    if not output.is_absolute() or case not in {"hbm", "echo"}:
        raise ValueError("Profile directory must be absolute and case hbm or echo")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Profile hooks directory must be empty: {output}")
    profiler = _Profiler(torch, output, case)
    profiler.patch(ModelRunner, "forward", profiler.forward_wrapper)
    profiler.patch(DeepseekV2DecoderLayer, "forward", profiler.layer_wrapper)
    for owner, name, stage in [
        (DeepseekV2AttentionMLA, "forward_prepare", "projection"),
        (DeepseekV2AttentionMLA, "forward_core", "output_mlp"),
        (DeepseekV2MLP, "forward", "output_mlp"),
        (LayerCommunicator, "prepare_attn", "projection"),
        (LayerCommunicator, "prepare_mlp", "output_mlp"),
        (LayerCommunicator, "postprocess_layer", "output_mlp"),
        (Indexer, "forward_cuda", "indexer"),
        (NSAIndexerMetadata, "topk_transform", "topk"),
        (NativeSparseAttnBackend, "init_forward_metadata", "cache_control"),
        (NativeSparseAttnBackend, "forward_extend", "cache_control"),
        (NativeSparseAttnBackend, "_forward_flashmla_prefill", "attention"),
        (memory_pool.MLATokenToKVPool, "set_mla_kv_buffer", "cache_write"),
        (memory_pool.NSATokenToKVPool, "set_index_k_and_scale_buffer", "cache_write"),
        (memory_pool_host.NSATokenToKVPoolHost, "set_mla_kv_buffer", "cache_write"),
        (memory_pool_host.NSATokenToKVPoolHost, "set_index_k_and_scale_buffer", "cache_write"),
        (
            memory_pool_host.NSATokenToKVPoolHost,
            "recall_miss_tokens_extend_cuda_graph",
            "cache_recall",
        ),
        (memory_pool_host.NSATokenToKVPoolHost, "free_device_pool_cuda_graph", "cache_control"),
        (deep_gemm, "fp8_mqa_logits", "indexer"),
        (deep_gemm, "fp8_mqa_logits_fuse_prefetch", "indexer_prefetch"),
        (memory_pool_host, "recall_update_extend", "io_h2d"),
    ]:
        profiler.patch(owner, name, profiler.stage_wrapper(stage))
    profiler.patch(Indexer, "_get_topk_ragged", profiler.ragged_wrapper)
    profiler.patch(memory_pool_host, "set_mla_kv_buffer_triton", profiler.write_wrapper)
    for owner, name in [
        (CudaGraphTokenToKVPoolAllocator, "post_alloc"),
        (memory_pool_host.NSATokenToKVPoolHost, "update_priority_when_use"),
    ]:
        profiler.patch(
            owner,
            name,
            profiler.stage_wrapper(
                lambda: (
                    "echo_finalize"
                    if getattr(profiler.local, "in_ragged", False)
                    else "cache_control"
                )
            ),
        )

    if hasattr(Scheduler, "echo_engine_profile_snapshot") or hasattr(
        Scheduler, "echo_engine_profile_begin"
    ):
        raise RuntimeError("Profile observation RPC already exists")

    def snapshot_rpc(scheduler, parameters):
        return profiler.snapshot(parameters)

    def begin_rpc(scheduler, parameters):
        return profiler.begin(parameters)

    Scheduler.echo_engine_profile_snapshot = snapshot_rpc
    Scheduler.echo_engine_profile_begin = begin_rpc
    profiler.persist()
    return profiler


def scheduler_with_profile(*args, **kwargs):
    """Picklable spawn target; patch only the newly owned scheduler process."""
    # Match the normal spawn target's import order before importing hook targets.
    # Importing NSA Indexer first can re-enter a partially initialized linear module.
    from sglang.srt.managers.scheduler import run_scheduler_process

    profiler = install_hooks()
    primary = None
    try:
        return run_scheduler_process(*args, **kwargs)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            profiler.persist()
        except BaseException as error:
            if primary is not None:
                raise BaseExceptionGroup(
                    "Scheduler and profile persistence failed", [primary, error]
                )
            raise
