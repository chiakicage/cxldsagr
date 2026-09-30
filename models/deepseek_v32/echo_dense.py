"""Experimental all-prefix-transfer adapter for pinned ECHO sparse attention.

This preserves ECHO's indexer, model math, host writes, and FlashMLA sparse kernel.
It replaces only the main-attention KV access with two layer-pipelined buffers.
The original device write pool remains allocated and MUST also count against HBM.
Construction does not validate numerical accuracy or establish measured overlap;
the separate integration tests exercise real-weight cache correctness.
"""

from __future__ import annotations

import hashlib
import importlib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

DENSE_ADAPTATION_ID = "gr_dense_all_prefix_transfer_v1"
DENSE_MODE = "dense_all_prefix_staging_v1"
MLA_RECORD_BYTES = 576 * 2


@dataclass(frozen=True)
class DenseBatchPlan:
    host_locations: tuple[int, ...]
    prefix_tokens: int
    num_layers: int

    @property
    def total_tokens(self):
        return len(self.host_locations)

    @property
    def suffix_tokens(self):
        return self.total_tokens - self.prefix_tokens

    @property
    def h2d_bytes(self):
        return self.prefix_tokens * MLA_RECORD_BYTES * self.num_layers


def plan_dense_batch(
    host_locations,
    *,
    prefix_tokens: int,
    max_context_tokens: int,
    host_capacity_tokens: int,
    num_layers: int,
) -> DenseBatchPlan:
    """Validate the ENTIRE active context, never a truncated selected union."""
    for name, value in (
        ("max_context_tokens", max_context_tokens),
        ("host_capacity_tokens", host_capacity_tokens),
        ("num_layers", num_layers),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_layers not in (1, 2, 3):
        raise ValueError("dense adaptation supports only the complete first 1-3 layers")
    locations = tuple(host_locations)
    if not locations or len(locations) > max_context_tokens:
        raise ValueError("complete active context must fit both dense scratch buffers")
    if type(prefix_tokens) is not int or not 0 <= prefix_tokens < len(locations):
        raise ValueError("prefix length must leave a nonempty newly computed suffix")
    # ECHO reserves 0 for padding and its last mapping element for the -1 sentinel.
    if any(type(x) is not int or not 0 < x < host_capacity_tokens for x in locations):
        raise ValueError("host locations must exclude padding and sentinel slots")
    if len(set(locations)) != len(locations):
        raise ValueError("active host locations must be unique")
    return DenseBatchPlan(locations, prefix_tokens, num_layers)


def dense_allocation_plan(max_context_tokens: int, host_capacity_tokens: int) -> dict:
    for value in (max_context_tokens, host_capacity_tokens):
        if type(value) is not int or value <= 0:
            raise ValueError("capacities must be positive integers")
    scratch = 2 * (max_context_tokens + 64) * MLA_RECORD_BYTES
    mapping = (host_capacity_tokens + 1) * 4
    positions = (max_context_tokens + 1) * 4
    return {
        "scratch_buffers": 2,
        "scratch_hbm_bytes": scratch,
        "mapping_hbm_bytes": mapping,
        "positions_hbm_bytes": positions,
        "additional_persistent_hbm_bytes": scratch + mapping + positions,
        "pinned_staging_bytes": scratch,
        "full_hbm_budget_verified": False,
        "excluded_from_additional_bytes": (
            "original_write_pool, resident_index, pool_metadata, model, activations, "
            "attention_workspace, temporary_query_and_index_tensors"
        ),
    }


@contextmanager
def _override_extend(backend, replacement):
    """Restore even an existing instance override, without changing its class."""
    present = "forward_extend" in vars(backend)
    previous = vars(backend).get("forward_extend")
    backend.forward_extend = replacement
    try:
        yield
    finally:
        if present:
            backend.forward_extend = previous
        else:
            del backend.forward_extend


class DensePrefetchController:
    """Serial single-request EXTEND controller, installed by the scoped factory.

    Existing history is gathered into pinned CPU staging and copied in full,
    regardless of original-pool hits. New suffix KV is generated on this layer,
    written by the original host-pool method, and appended directly to scratch.
    A post-layer hook queues the next layer's transfer while current-layer GPU
    work may still be running. Only a GPU timeline can establish actual overlap.
    """

    def __init__(self, runner, *, max_context_tokens: int):
        from models.deepseek_v32.echo_adapter import verify_echo_checkout

        self.torch = importlib.import_module("torch")
        self.source = importlib.import_module("sglang.srt.layers.attention.nsa_backend")
        indexer = importlib.import_module("sglang.srt.layers.attention.nsa.nsa_indexer")
        source_path = Path(self.source.__file__).resolve()
        root, revision, patch_sha256 = verify_echo_checkout(source_path.parents[6])
        if source_path != root / "sglang/python/sglang/srt/layers/attention/nsa_backend.py":
            raise ValueError("dense adaptation requires the pinned ECHO source layout")
        if not Path(indexer.__file__).resolve().is_relative_to(root / "sglang/python"):
            raise ValueError("dense adaptation imported an indexer outside pinned ECHO")
        if indexer.NSA_FUSE_LOGITS_RECALL_EXTEND or indexer.NSA_FUSE_LOGITS_RECALL:
            raise ValueError("disable both ECHO fused-recall switches before importing SGLang")
        self.runner = runner
        self.pool = runner.token_to_kv_pool
        self.backend = runner.attn_backend
        self.layers = tuple(runner.model.model.layers)
        self.num_layers = len(self.layers)
        self.max_context_tokens = max_context_tokens
        plan_dense_batch(
            [1],
            prefix_tokens=0,
            max_context_tokens=max_context_tokens,
            host_capacity_tokens=self.pool.size,
            num_layers=self.num_layers,
        )
        if max_context_tokens > runner.model_config.context_len:
            raise ValueError("dense scratch capacity exceeds the model context limit")
        if type(self.backend) is not self.source.NativeSparseAttnBackend:
            raise ValueError("dense adaptation requires the unwrapped ECHO NSA backend")
        args = runner.server_args
        if (
            args.tp_size != 1
            or args.pp_size != 1
            or args.dp_size != 1
            or args.enable_two_batch_overlap
            or args.enable_torch_compile
            or not args.disable_cuda_graph
            or not args.disable_overlap_schedule
        ):
            raise ValueError("dense adaptation requires serial TP=PP=DP=1 without CUDA graphs")
        if not getattr(args.load_format, "checkpoint_metadata_fingerprint", None):
            raise ValueError("dense adaptation requires the real prefix-only model loader")
        if (
            self.pool.device != "cpu"
            or self.pool.start_layer != 0
            or self.pool.layer_num != self.num_layers
            or self.pool.page_size != 64
            or self.pool.device_pool.dtype != self.torch.bfloat16
            or self.pool.device_pool.kv_lora_rank != 512
            or self.pool.device_pool.qk_rope_head_dim != 64
            or self.pool.index_head_dim != 128
        ):
            raise ValueError("dense adaptation requires first-layer BF16 MLA 512+64 host storage")
        self._host_layers = tuple(value.view(self.torch.bfloat16) for value in self.pool.kv_buffer)
        if any(
            value.device.type != "cpu" or not value.is_pinned() or value.shape[1:] != (1, 576)
            for value in self._host_layers
        ):
            raise ValueError("dense prefix reads require pinned layer-first BF16 host KV")
        self.device = self.pool.device_pool.device
        shape = (max_context_tokens + 64, 1, 576)
        self._device_buffers = tuple(
            self.torch.zeros(shape, dtype=self.torch.bfloat16, device=self.device) for _ in range(2)
        )
        self._host_staging = tuple(
            self.torch.empty(shape, dtype=self.torch.bfloat16, pin_memory=True) for _ in range(2)
        )
        self._mapping = self.torch.full(
            (self.pool.size + 1,), -1, dtype=self.torch.int32, device=self.device
        )
        self._positions = self.torch.arange(
            max_context_tokens + 1, dtype=self.torch.int32, device=self.device
        )
        self._copy_stream = self.torch.cuda.Stream(device=self.device)
        self._copy_stream.wait_stream(self.torch.cuda.current_stream(self.device))
        self._staging_ready = [None, None]
        self._buffer_released = [None, None]
        self._last_compute_event = None
        self._active_batch = None
        self._failed = False
        self._closed = False
        self.last_batch_stats = None
        self.provenance = {
            "mode": DENSE_MODE,
            "adaptation": DENSE_ADAPTATION_ID,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "echo_revision": revision,
            "echo_patch_sha256": patch_sha256,
            "policy": "all-prefix-transfer ablation",
            "storage": "staging+original_write_pool",
            "gpu_correctness": "not_checked_by_controller",
            "measured_overlap": "unverified",
            "transfer_scope": "existing prefix only; new suffix generated and appended on device",
        }
        self.memory_accounting = dense_allocation_plan(max_context_tokens, self.pool.size)
        self.memory_accounting.update(
            retained_original_write_pool_hbm_bytes=sum(
                value.numel() * value.element_size() for value in self.pool.device_pool.kv_buffer
            ),
            resident_index_hbm_bytes=sum(
                value.numel() * value.element_size()
                for value in self.pool.index_k_with_scale_buffer
            ),
            original_host_kv_bytes=sum(
                value.numel() * value.element_size() for value in self._host_layers
            ),
        )

    def _ensure_alive(self):
        if self._failed or self._closed:
            raise RuntimeError("dense controller is failed or closed; use a fresh process")

    def _prepare_batch(self, batch):
        if (
            batch.batch_size != 1
            or not batch.forward_mode.is_extend()
            or batch.forward_mode.is_target_verify()
            or batch.forward_mode.is_draft_extend()
            or batch.token_to_kv_pool is not self.pool
            or batch.attn_backend is not self.backend
        ):
            raise ValueError("dense adaptation accepts one ordinary EXTEND request only")
        if self._last_compute_event is not None:
            self._last_compute_event.synchronize()
        # CPU gather must not observe an unfinished prior GPU -> host write.
        for streams in self.pool.host_transfer_streams:
            for stream in streams:
                stream.synchronize()
        total = int(batch.seq_lens_cpu[0])
        prefix = int(batch.extend_prefix_lens_cpu[0])
        req_ids = batch.req_pool_indices_cpu
        if len(req_ids) != 1:
            raise ValueError("dense adaptation requires one request-pool slot")
        req_id = int(req_ids[0])
        host_ids_gpu = batch.req_to_token_pool.req_to_token[req_id, :total]
        host_ids_cpu = host_ids_gpu.to(device="cpu", dtype=self.torch.int64)
        self._plan = plan_dense_batch(
            host_ids_cpu.tolist(),
            prefix_tokens=prefix,
            max_context_tokens=self.max_context_tokens,
            host_capacity_tokens=self.pool.size,
            num_layers=self.num_layers,
        )
        if self._plan.suffix_tokens > self.pool.device_pool.size:
            raise ValueError("new suffix does not fit the retained original write pool")
        if (
            batch.input_ids.numel() != self._plan.suffix_tokens
            or batch.out_cache_loc.cpu().tolist() != list(self._plan.host_locations[prefix:])
        ):
            raise ValueError("dense adaptation rejects padded or inconsistent EXTEND allocation")
        self._prefix_ids_cpu = host_ids_cpu[:prefix]
        self._mapping.fill_(-1)
        self._mapping[0] = 0
        self._mapping[host_ids_gpu] = self._positions[1 : total + 1]
        self._next_layer = 0
        self._current_layer = None
        self._attention_seen = False
        self._copy_events = [None] * self.num_layers

    def _enqueue_prefix(self, layer_id):
        slot = layer_id % 2
        previous_copy = self._staging_ready[slot]
        if previous_copy is not None:
            # Pinned CPU memory cannot be overwritten until its prior DMA is done.
            previous_copy.synchronize()
        count = self._plan.prefix_tokens
        staging = self._host_staging[slot][:count]
        if count:
            self.torch.index_select(
                self._host_layers[layer_id], 0, self._prefix_ids_cpu, out=staging
            )
        with self.torch.cuda.stream(self._copy_stream):
            released = self._buffer_released[slot]
            if released is not None:
                self._copy_stream.wait_event(released)
            if count:
                self._device_buffers[slot][1 : count + 1].copy_(staging, non_blocking=True)
            ready = self.torch.cuda.Event()
            ready.record(self._copy_stream)
        self._staging_ready[slot] = ready
        self._copy_events[layer_id] = ready
        self._staged_prefix_tokens[layer_id] = count

    def _before_layer(self, layer_id):
        self._ensure_alive()
        if self._active_batch is None or self._current_layer is not None:
            raise RuntimeError("layer execution must occur inside controller.forward_batch")
        if layer_id != self._next_layer or self._copy_events[layer_id] is None:
            raise RuntimeError("dense layer execution order changed")
        self.torch.cuda.current_stream(self.device).wait_event(self._copy_events[layer_id])
        self._current_layer = layer_id
        self._attention_seen = False

    def _after_layer(self, layer_id):
        if self._current_layer != layer_id or not self._attention_seen:
            raise RuntimeError("expected exactly one complete sparse attention per layer")
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream(self.device))
        self._buffer_released[layer_id % 2] = event
        self._last_compute_event = event
        self._next_layer += 1
        self._current_layer = None
        if self._next_layer < self.num_layers:
            self._enqueue_prefix(self._next_layer)

    def _forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        q_rope=None,
        k_rope=None,
        topk_indices=None,
    ):
        self._ensure_alive()
        if (
            forward_batch is not self._active_batch
            or layer.layer_id != self._current_layer
            or self._attention_seen
            or not save_kv_cache
            or layer.is_cross_attention
            or layer.v_head_dim != 512
            or layer.head_dim != 576
            or any(value is None for value in (q, k, v, q_rope, k_rope, topk_indices))
        ):
            raise RuntimeError("unsupported or repeated dense-adapted NSA attention call")
        metadata = self.backend.forward_metadata
        self.pool.set_mla_kv_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            k_rope,
            metadata=metadata,
            host_transfer_req_pool_indices=forward_batch.req_pool_indices_cpu,
        )
        prefix, total = self._plan.prefix_tokens, self._plan.total_tokens
        scratch = self._device_buffers[layer.layer_id % 2]
        scratch[prefix + 1 : total + 1, :, :512].copy_(k.reshape(-1, 1, 512))
        scratch[prefix + 1 : total + 1, :, 512:].copy_(k_rope.reshape(-1, 1, 64))
        if self.source.NSA_FUSE_TOPK:
            host_selection = topk_indices
        else:
            host_selection = self.source.transform_index_page_table_prefill(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                extend_lens_cpu=metadata.nsa_extend_seq_lens_list,
                page_size=1,
            )
        self.torch._assert_async(
            ((host_selection >= -1) & (host_selection < self.pool.size)).all(),
            "top-k contains an invalid host location",
        )
        dense_selection = self._mapping[host_selection]
        self.torch._assert_async(
            ((host_selection == -1) | (dense_selection >= 0)).all(),
            "top-k refers outside the complete active request",
        )
        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope = q_rope.view(-1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim)
        self._attention_seen = True
        return self.backend._forward_flashmla_prefill(
            q_all=self.torch.cat([q_nope, q_rope], dim=-1),
            kv_cache=scratch,
            page_table_1=dense_selection,
            sm_scale=layer.scaling,
            v_head_dim=layer.v_head_dim,
        )

    @contextmanager
    def forward_batch(self, batch):
        """Wrap exactly one ModelRunner.forward, after ForwardBatch.init_new."""
        self._ensure_alive()
        if self._active_batch is not None:
            raise RuntimeError("dense forward contexts cannot overlap or nest")
        try:
            self._staged_prefix_tokens = [0] * self.num_layers
            self._prepare_batch(batch)
            self._active_batch = batch
            self._enqueue_prefix(0)
            yield
            if self._next_layer != self.num_layers or self._current_layer is not None:
                raise RuntimeError("forward did not execute every complete model layer")
            self.last_batch_stats = {
                "prefix_tokens": self._plan.prefix_tokens,
                "new_suffix_tokens": self._plan.suffix_tokens,
                "num_layers": self.num_layers,
                "staged_prefix_tokens": sum(self._staged_prefix_tokens),
                "staged_prefix_tokens_per_layer": list(self._staged_prefix_tokens),
                "h2d_payload_bytes": sum(self._staged_prefix_tokens) * MLA_RECORD_BYTES,
                "h2d_payload_bytes_per_layer": [
                    count * MLA_RECORD_BYTES for count in self._staged_prefix_tokens
                ],
                "byte_counter_scope": "submitted MLA KV H2D payload; caller waits for completion",
                "device_generated_suffix_bytes": (
                    self._plan.suffix_tokens * self.num_layers * MLA_RECORD_BYTES
                ),
                "copy_submission": "next layer after current-layer kernel enqueue",
            }
        except BaseException:
            self._failed = True
            raise
        finally:
            self._active_batch = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            # Includes failed forwards without a post-layer completion event.
            self.torch.cuda.synchronize(self.device)
        except BaseException:
            self._failed = True
            raise
        finally:
            self.runner = None
            self.pool = None
            self.backend = None
            self.layers = ()
            self._host_layers = ()
            self._device_buffers = ()
            self._host_staging = ()
            self._mapping = None
            self._positions = None
            self._prefix_ids_cpu = None
            self._active_batch = None
            self._plan = None
            self._copy_stream = None
            self._copy_events = ()
            self._staging_ready = ()
            self._buffer_released = ()
            self._last_compute_event = None


@contextmanager
def scoped_dense_prefetch(runner, *, max_context_tokens: int):
    """Install on an existing ModelRunner while scoped_echo_adapter remains open.

    Set NSA_KV_OFFLOAD=1 and both SGLANG_NSA_FUSE_LOGITS_RECALL_* flags to 0
    BEFORE SGLang imports. Build the prefix-only ModelRunner normally, then keep
    this scope open for its lifetime. For every already-initialized ForwardBatch:

        with torch.inference_mode(), controller.forward_batch(forward_batch):
            output, _ = runner.forward(forward_batch)

    The caller must retain its existing prefix/suffix ownership and synchronization.
    The reported additional persistent tensors do not replace a full memory audit:
    original pool metadata and transient workspaces require global measurement.
    """
    controller = DensePrefetchController(runner, max_context_tokens=max_context_tokens)
    with ExitStack() as stack:
        stack.callback(controller.close)
        stack.enter_context(_override_extend(runner.attn_backend, controller._forward_extend))
        for index, layer in enumerate(controller.layers):
            before = layer.register_forward_pre_hook(
                lambda module, args, index=index: controller._before_layer(index)
            )
            stack.callback(before.remove)
            after = layer.register_forward_hook(
                lambda module, args, output, index=index: controller._after_layer(index)
            )
            stack.callback(after.remove)
        yield controller
