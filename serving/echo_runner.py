"""Single-process ECHO prefix branches for reduced-model correctness checks.

This is not a scheduler or a heat-based cache manager. Prefixes are explicitly
owned handles; exhausted pools fail rather than evicting or changing the input.
Each backend configuration must run in a fresh process because ECHO reads its
offload switches at import time.
"""

from __future__ import annotations

import os
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(eq=False)
class EchoPrefix:
    token_ids: tuple[int, ...]
    locations: torch.Tensor
    predictor_state: tuple[torch.Tensor, ...]
    owner: object = field(repr=False)
    released: bool = False


class EchoPrefixRunner:
    """Serial, page-aligned prefix reuse with private candidate allocations.

    Construction is managed by ``open_echo_runner``. Synchronization is
    deliberately conservative, including host transfer streams. Do not use
    these correctness-path timings as baseline serving performance.
    """

    def __init__(
        self, runner, *, model_instance_id: str, prefill_chunk: int, dense_controller=None
    ):
        self.runner = runner
        self.model_instance_id = model_instance_id
        self.prefill_chunk = prefill_chunk
        self.dense_controller = dense_controller
        self._prefixes: set[EchoPrefix] = set()
        self._failed = False
        self._closed = False

    def _ensure_alive(self):
        if self._closed or self._failed:
            raise RuntimeError("ECHO runner is closed or failed; use a new process")

    def _validate_ids(self, ids):
        values = tuple(ids)
        vocab_size = self.runner.model_config.hf_config.vocab_size
        if not values or any(type(x) is not int or not 0 <= x < vocab_size for x in values):
            raise ValueError("token IDs must be a nonempty sequence of integers in vocabulary")
        return values

    def _check_prefix(self, prefix: EchoPrefix):
        self._ensure_alive()
        if prefix.owner is not self or prefix.released or prefix not in self._prefixes:
            raise ValueError("prefix is released or belongs to another model instance")

    def _predictors(self):
        return tuple(
            layer.self_attn.indexer.extend_logits_offsets
            for layer in self.runner.model.model.layers
        )

    def _check_capacity(self, total: int, prefix_length: int) -> int:
        runner = self.runner
        allocator = runner.token_to_kv_pool_allocator
        additional = total - prefix_length
        if additional <= 0:
            raise ValueError("extend must contain new tokens")
        # SGLang's paged allocator reserves one extra page conservatively.
        required = additional + runner.page_size
        if allocator.available_size() < required:
            raise MemoryError(f"ECHO pool needs {required} free tokens for this extend")
        if total > runner.model_config.context_len:
            raise ValueError("request exceeds the configured model context")
        device_pool = getattr(runner.token_to_kv_pool, "device_pool", None)
        if device_pool is not None and total > device_pool.size:
            raise MemoryError(
                "the initial ECHO correctness runner requires the entire active context "
                "to fit the per-layer device pool; exact selected-union admission is not implemented"
            )
        return additional

    def _forward(self, all_ids: tuple[int, ...], prefix_locations):
        additional = self._check_capacity(len(all_ids), len(prefix_locations))
        import torch
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather

        runner = self.runner
        allocator = runner.token_to_kv_pool_allocator
        req = Req(
            rid="gr-prefix-check",
            origin_input_text="",
            origin_input_ids=list(all_ids),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=0),
        )
        req.fill_ids = list(all_ids)
        req.prefix_indices = prefix_locations
        req.extend_input_len = additional
        req.logprob_start_len = len(all_ids)
        tree = SimpleNamespace(
            page_size=runner.page_size,
            device=runner.device,
            token_to_kv_pool_allocator=allocator,
        )
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree,
            model_config=runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        try:
            batch.prepare_for_extend()
            if require_mlp_sync(runner.server_args):
                Scheduler.prepare_mlp_sync_batch_raw(
                    batch,
                    dp_size=1,
                    attn_tp_size=1,
                    tp_group=runner.tp_group,
                    get_idle_batch=None,
                    disable_cuda_graph=True,
                    speculative_num_draft_tokens=None,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    require_mlp_tp_gather=require_mlp_tp_gather(runner.server_args),
                    disable_overlap_schedule=True,
                    offload_tags=set(),
                )
            forward_batch = ForwardBatch.init_new(batch.get_model_worker_batch(), runner)
            with torch.inference_mode():
                dense_scope = (
                    self.dense_controller.forward_batch(forward_batch)
                    if self.dense_controller is not None
                    else nullcontext()
                )
                with dense_scope:
                    output, _ = runner.forward(forward_batch)
                if output.next_token_logits.numel() != 0 or output.hidden_states is None:
                    raise RuntimeError("expected hidden-only ECHO adapter output")
                hidden = output.hidden_states.clone()
                new_locations = batch.out_cache_loc.clone()
            torch.cuda.synchronize(runner.gpu_id)
        except BaseException:
            # An interrupted model may own in-flight writes; never reuse its pools.
            self._failed = True
            raise
        finally:
            if req.req_pool_idx is not None and not self._failed:
                runner.req_to_token_pool.free(req.req_pool_idx)
        return hidden, new_locations

    def prefill(self, token_ids) -> EchoPrefix:
        import torch

        self._ensure_alive()
        ids = self._validate_ids(token_ids)
        if len(ids) % self.runner.page_size:
            raise ValueError("this initial adapter requires page-aligned stable prefixes")
        locations = torch.empty(0, dtype=torch.int64, device=self.runner.device)
        try:
            for end in range(self.prefill_chunk, len(ids) + self.prefill_chunk, self.prefill_chunk):
                hidden, new = self._forward(ids[: min(end, len(ids))], locations)
                locations = torch.cat((locations, new))
                del hidden
            state = tuple(value.clone() for value in self._predictors())
            prefix = EchoPrefix(ids, locations, state, self)
            self._prefixes.add(prefix)
            return prefix
        except BaseException:
            self._failed = True
            raise

    def extend(self, prefix: EchoPrefix, candidate_ids):
        self._check_prefix(prefix)
        candidate = self._validate_ids(candidate_ids)
        try:
            for destination, saved in zip(self._predictors(), prefix.predictor_state, strict=True):
                destination.copy_(saved)
            hidden, suffix_locations = self._forward(prefix.token_ids + candidate, prefix.locations)
            # The stable prefix ends on a page boundary, so no freed page is shared.
            self.runner.token_to_kv_pool_allocator.free(suffix_locations)
        except BaseException:
            self._failed = True
            raise
        return hidden

    def release(self, prefix: EchoPrefix):
        self._check_prefix(prefix)
        try:
            self.runner.token_to_kv_pool_allocator.free(prefix.locations)
        except BaseException:
            self._failed = True
            raise
        self._prefixes.remove(prefix)
        prefix.released = True
        prefix.locations = None
        prefix.predictor_state = ()

    def evict_hbm(self, prefix: EchoPrefix):
        """Drop only the offloaded prefix's HBM replicas for correctness checks."""
        import torch

        self._check_prefix(prefix)
        pool = self.runner.token_to_kv_pool
        if not hasattr(pool, "free_req_device_pool"):
            raise ValueError("resident mode has no authoritative host copy")
        try:
            torch.cuda.synchronize(self.runner.gpu_id)
            pool.free_req_device_pool(prefix.locations)
            torch.cuda.synchronize(self.runner.gpu_id)
        except BaseException:
            self._failed = True
            raise

    def close(self):
        if self._closed:
            return
        try:
            if not self._failed:
                for prefix in tuple(self._prefixes):
                    self.release(prefix)
        finally:
            for prefix in self._prefixes:
                prefix.released = True
                prefix.locations = None
                prefix.predictor_state = ()
            self._prefixes.clear()
            self._closed = True
            self.runner = None
            self.dense_controller = None


@contextmanager
def open_echo_runner(
    *,
    model_path: Path,
    echo_path: Path,
    num_layers: int = 3,
    mode: str = "resident",
    max_total_tokens: int = 16384,
    device_cache_tokens: int = 8192,
    prefill_chunk: int = 1024,
    kernel_patch: str | None = None,
    mem_fraction_static: float = 0.6,
    dense_context_tokens: int | None = None,
    dense_prefetch_schedule: str | None = None,
    dense_prefetch_transport: str | None = None,
):
    """Build the ECHO backend with explicit model and optional kernel adapters."""
    if mode not in ("resident", "echo_gr_adapted", "sparse_sync", "dense_prefetch"):
        raise ValueError("mode must be resident, echo_gr_adapted, sparse_sync, or dense_prefetch")
    if dense_prefetch_transport is not None:
        from models.deepseek_v32.echo_dense import DENSE_TRANSPORTS

        if mode != "dense_prefetch" or dense_prefetch_transport not in DENSE_TRANSPORTS:
            raise ValueError("dense_prefetch_transport requires dense mode and a valid transport")
    if dense_prefetch_schedule is not None:
        from models.deepseek_v32.echo_dense import DENSE_SCHEDULES

        if mode != "dense_prefetch" or dense_prefetch_schedule not in DENSE_SCHEDULES:
            raise ValueError("dense_prefetch_schedule requires dense mode and a valid schedule")
    if not 0 < mem_fraction_static < 1:
        raise ValueError("mem_fraction_static must be between zero and one")
    for name, value in (
        ("max_total_tokens", max_total_tokens),
        ("device_cache_tokens", device_cache_tokens),
        ("prefill_chunk", prefill_chunk),
    ):
        if type(value) is not int or value <= 0 or value % 64:
            raise ValueError(f"{name} must be a positive multiple of 64")
    if max_total_tokens < prefill_chunk + 64:
        raise ValueError("max_total_tokens must leave an extra page beyond a prefill chunk")
    if mode != "resident":
        if max_total_tokens <= device_cache_tokens:
            raise ValueError("ECHO requires host token capacity to exceed device cache capacity")
        if prefill_chunk > device_cache_tokens:
            raise ValueError("prefill chunks must fit in the device cache")
    if any(name.startswith("sglang.srt") for name in sys.modules):
        raise RuntimeError("start a fresh process before selecting the ECHO backend")
    updates = {
        "NSA_KV_OFFLOAD": None if mode == "resident" else "1",
        "NSA_DEV_CACHE_SIZE": str(device_cache_tokens),
        "SGLANG_NSA_FUSE_LOGITS_RECALL_EXTEND": "1" if mode == "echo_gr_adapted" else "0",
        "SGLANG_NSA_FUSE_LOGITS_RECALL_DECODE": "0",
    }
    previous = {name: os.environ.get(name) for name in updates}
    local = None
    initialized_dist = False
    try:
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        import torch

        from models.deepseek_v32.echo_adapter import scoped_echo_adapter
        from models.deepseek_v32.echo_kernel import scoped_echo_kernel

        if torch.distributed.is_initialized():
            raise RuntimeError("the local ECHO adapter requires its own single-rank process")
        with (
            scoped_echo_kernel(echo_path, patch_id=kernel_patch) as kernel_info,
            scoped_echo_adapter(echo_path, model_path, num_layers) as bindings,
        ):
            import json

            from sglang.srt.configs.model_config import ModelConfig
            from sglang.srt.entrypoints.engine import _set_envs_and_config
            from sglang.srt.layers.moe import initialize_moe_config
            from sglang.srt.model_executor.model_runner import ModelRunner
            from sglang.srt.server_args import PortArgs, ServerArgs

            args = ServerArgs(
                model_path=str(model_path),
                json_model_override_args=json.dumps({"num_hidden_layers": num_layers}),
                dtype="bfloat16",
                kv_cache_dtype="auto",
                tp_size=1,
                pp_size=1,
                ep_size=1,
                dp_size=1,
                max_running_requests=1,
                max_total_tokens=max_total_tokens,
                mem_fraction_static=mem_fraction_static,
                chunked_prefill_size=prefill_chunk,
                disable_cuda_graph=True,
                enable_piecewise_cuda_graph=False,
                disable_overlap_schedule=True,
                disable_shared_experts_fusion=True,
                disable_radix_cache=True,
            )
            args.load_format = bindings.loader_class
            _set_envs_and_config(args)
            initialize_moe_config(args)
            config = ModelConfig.from_server_args(args)
            ports = PortArgs.init_new(args)
            initialized_dist = True
            runner = ModelRunner(
                model_config=config,
                mem_fraction_static=args.mem_fraction_static,
                gpu_id=0,
                tp_rank=0,
                tp_size=1,
                moe_ep_rank=0,
                moe_ep_size=1,
                pp_rank=0,
                pp_size=1,
                dp_rank=0,
                nccl_port=ports.nccl_port,
                server_args=args,
            )
            with ExitStack() as backend_scopes:
                if runner.token_to_kv_pool_allocator.size != max_total_tokens:
                    raise MemoryError("ECHO silently changed the requested logical pool capacity")
                index_fix = {
                    "enabled": False,
                    "reason": "index buffers fit signed int32 byte offsets",
                }
                if any(
                    buf.nbytes > 2**31 - 1
                    for buf in runner.token_to_kv_pool.index_k_with_scale_buffer
                ):
                    from sglang.srt.layers.attention.nsa import index_buf_accessor

                    from models.deepseek_v32.echo_index import scoped_index_address_fix

                    index_fix = backend_scopes.enter_context(
                        scoped_index_address_fix(index_buf_accessor, torch)
                    )
                    index_fix["enabled"] = True
                recall_address_fix = None
                if mode != "resident":
                    from sglang.srt.mem_cache import recall_ops

                    from models.deepseek_v32.echo_recall import scoped_extend_recall_address_fix

                    recall_address_fix = backend_scopes.enter_context(
                        scoped_extend_recall_address_fix(recall_ops)
                    )
                dense = None
                if mode == "dense_prefetch":
                    from models.deepseek_v32.echo_dense import scoped_dense_prefetch

                    dense = backend_scopes.enter_context(
                        scoped_dense_prefetch(
                            runner,
                            max_context_tokens=dense_context_tokens or device_cache_tokens,
                            schedule=dense_prefetch_schedule or "attention_window",
                            transport=dense_prefetch_transport or "gpu_direct",
                        )
                    )
                local = EchoPrefixRunner(
                    runner,
                    model_instance_id=bindings.model_instance_id,
                    prefill_chunk=prefill_chunk,
                    dense_controller=dense,
                )
                local.provenance = {
                    "mode": mode,
                    "num_layers": num_layers,
                    "layer_mlp_classes": [
                        type(layer.mlp).__name__ for layer in runner.model.model.layers
                    ],
                    "moe_runner_backend": str(args.moe_runner_backend),
                    "model_instance_id": bindings.model_instance_id,
                    "checkpoint_metadata_sha256": bindings.selection.metadata_sha256,
                    "weight_payload_fingerprint": "not_hashed; reuse limited to this model instance",
                    "echo_revision": bindings.echo_revision,
                    "echo_patch_sha256": bindings.echo_patch_sha256,
                    "cache_adaptations": bindings.cache_adaptations,
                    "cache_adapter_sha256": bindings.cache_adapter_sha256,
                    "kernel_overlay": kernel_info,
                    "index_address_fix": index_fix,
                    "recall_address_fix": recall_address_fix,
                    "timing_scope": "synchronous correctness path; not a performance benchmark",
                }
                if dense is not None:
                    local.provenance["dense_prefetch"] = dense.provenance
                    local.provenance["dense_memory_accounting"] = dense.memory_accounting
                try:
                    yield local
                finally:
                    local.close()
    finally:
        try:
            try:
                if local is not None:
                    local.close()
            finally:
                if initialized_dist:
                    from sglang.srt.distributed.parallel_state import (
                        destroy_distributed_environment,
                    )

                    destroy_distributed_environment()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
