"""Single-GPU, checkpoint-backed dense-layer replay model for GR serving.

This is a workload surrogate, not a trained smaller DeepSeek model. Every
physical layer owns an independent copy of one of checkpoint layers 0, 1, 2.
Its input is copied from that source layer's input in the first three layers,
including the separate residual stream. Copies therefore do not invent a
deeper residual trajectory. Their KV/indexer states are independent.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

from cache.sparse_token_cache import MISSING, CacheStats, SparseTokenCache
from models.deepseek_v32.echo_attention import EchoAttentionRunner
from models.deepseek_v32.echo_model import CheckpointReader, Config, rms_norm

SCHEMES = ("hbm", "echo", "serial_sparse", "dense_prefetch")


def replay_parameter_count(reader, num_layers):
    """Count model parameters, excluding FP8 scaling metadata."""
    if type(num_layers) is not int or num_layers < 3:
        raise ValueError("the replay model requires at least three physical layers")
    counts = []
    for layer in range(3):
        prefix = f"model.layers.{layer}."
        names = [name for name in reader.tensor_metadata if name.startswith(prefix)]
        if not names:
            raise ValueError(f"checkpoint is missing source layer {layer}")
        counts.append(
            sum(
                math.prod(reader.tensor_metadata[name]["shape"])
                for name in names
                if not name.endswith(".weight_scale_inv")
            )
        )
    endpoints = sum(
        math.prod(reader.tensor_metadata[name]["shape"])
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
    )
    return {
        "source_layer_parameters": counts,
        "backbone_parameters": sum(counts[layer % 3] for layer in range(num_layers)),
        "endpoint_parameters": endpoints,
        "total_parameters": endpoints + sum(counts[layer % 3] for layer in range(num_layers)),
    }


def _storage_bytes(tensors):
    seen = set()
    result = {"hbm": 0, "dram": 0}
    for tensor in tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr())
        if identity not in seen:
            seen.add(identity)
            result["hbm" if tensor.is_cuda else "dram"] += storage.nbytes()
    return result


class _SerialCache(SparseTokenCache):
    def prepare_prefetch(self, new_start, new_count, offset, *, limit=8192):
        # The same exact selection and recall path executes after index scoring.
        return None


class _DenseCache(_SerialCache):
    """Pinned full-layer backing and a borrowed double-buffered HBM stage."""

    def __init__(self, capacity, width, *, device, records):
        self.capacity, self.width = capacity, width
        self.offload = True
        self.device = torch.device(device)
        self.slots = capacity
        self.records = records
        self.host = torch.empty((capacity, width), dtype=records.dtype, pin_memory=True)
        self.host_to_device = torch.full((capacity,), MISSING, dtype=torch.int32, device=device)
        self.device_to_host = torch.full((capacity,), MISSING, dtype=torch.int64, device=device)
        self.age = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.length = self.written = 0
        self._step_end = None
        self._clock = 0
        self.stats = CacheStats()
        self.prefetch_counts = []
        self.dense_fetched_records = 0

    def append(self, records):
        start = super().append(records)
        # The new suffix is already on the compute GPU; no host round-trip.
        self.records[start : self.written].copy_(records)
        return start

    def reset_stats(self):
        super().reset_stats()
        self.dense_fetched_records = 0

    def metrics(self):
        result = super().metrics()
        result["dense_fetched_records"] = self.dense_fetched_records
        result["host_to_device_bytes"] += self.dense_fetched_records * self.record_bytes
        return result


class _ServingAttention(EchoAttentionRunner):
    def __init__(self, attention, capacity, *, scheme, slots, chunk_size, dense_records=None):
        self.attention = attention
        self.cfg = attention.cfg
        self.chunk_size = chunk_size
        self.scheme = scheme
        width = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim
        if scheme == "dense_prefetch":
            self.cache = _DenseCache(
                capacity, width, device=attention.device, records=dense_records
            )
        else:
            cache_type = _SerialCache if scheme == "serial_sparse" else SparseTokenCache
            self.cache = cache_type(
                capacity,
                width,
                device=attention.device,
                slots=min(slots, capacity) if scheme != "hbm" else None,
            )
        self.offset = torch.zeros(16, device=attention.device, dtype=torch.float32)
        self.index_keys = torch.empty(
            (capacity, self.cfg.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device=attention.device,
        )
        self.index_scales = torch.empty(capacity, dtype=torch.float32, device=attention.device)
        self.last_indices = None
        self.capture_hook = None

    def _consume(self, q, indices, scope):
        if self.scheme == "dense_prefetch":
            from operators.deepseek_v32.attention.device_only.mla import sparse_mla

            with scope("sparse_mla"):
                return sparse_mla(q, self.cache.records, indices, self.cfg.attention_scale)
        return super()._consume(q, indices, scope)


@dataclass
class DeepSeekServingSession:
    capacity: int
    scheme: str
    runners: list
    owner: object = field(repr=False)
    length: int = 0
    stages: list = field(default_factory=list)
    copy_stream: object = None
    stage_consumed: list = field(default_factory=list)
    released: bool = False


class DeepSeekServingBackend:
    """Serve retained user sessions using one physical single-GPU replay model.

    Cache allocations are fixed at session creation. The serving cache manager
    can therefore reserve their upper bound before admitting a user. Temporary
    activations, selections and GEMM workspace are model execution memory and
    are reported separately from retained cache memory.
    """

    def __init__(
        self,
        model_path,
        *,
        scheme="hbm",
        device="cuda:0",
        num_layers=10,
        chunk_size=256,
        slots=4096,
        linear_backend="fp8",
    ):
        from models.deepseek_v32.echo_block import CheckpointBlock

        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        if chunk_size < 1 or slots < chunk_size:
            raise ValueError("chunk_size must be positive and fit in the sparse pool")
        self.path = Path(model_path)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DeepSeek serving requires one SM90 CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (9, 0):
            raise ValueError("DeepSeek serving requires one SM90 CUDA device")
        self.scheme = scheme
        self.cfg = Config.from_checkpoint(model_path)
        if self.cfg.first_k_dense_replace < 3:
            raise ValueError("checkpoint layers 0, 1, 2 must all have dense MLPs")
        self.max_seq_len = self.cfg.max_seq_len
        self.reader = CheckpointReader(model_path)
        self.parameter_counts = replay_parameter_count(self.reader, num_layers)
        self.num_layers, self.chunk_size, self.slots = num_layers, chunk_size, slots
        if scheme in ("echo", "serial_sparse") and slots < self.cfg.index_topk:
            raise ValueError("the sparse pool must fit one query's exact top-k selection")
        self.linear_backend = linear_backend
        self.embedding_weight = self.reader.get_tensor("model.embed_tokens.weight").to(self.device)
        self.final_norm = self.reader.get_tensor("model.norm.weight").to(self.device)
        self.head_weight = self.reader.get_tensor("lm_head.weight").to(self.device)
        self.blocks, self.attentions = [], []
        # Load every copy independently. No weight aliases or shared KV rows are
        # introduced between copies, even when the source layer is the same.
        for layer in range(num_layers):
            block = CheckpointBlock(
                model_path,
                layer % 3,
                self.device,
                self.reader,
                capacity=1,
                chunk_size=chunk_size,
                linear_backend=linear_backend,
            )
            self.attentions.append(block.attention.attention)
            block.attention = block.cache = None
            self.blocks.append(block)
        self.last_logits = None
        self.capture_hook = None
        self._busy = False
        self.synchronize()

    def estimate_session_bytes(self, capacity, prefix_tokens=0):
        if type(capacity) is not int or not 1 <= capacity <= self.max_seq_len:
            raise ValueError("session capacity exceeds the model context limit")
        if not 0 <= prefix_tokens <= capacity:
            raise ValueError("prefix length must fit session capacity")
        width = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim
        record = width * 2
        index = capacity * (self.cfg.index_head_dim + 4) + 64
        slots = capacity if self.scheme in ("hbm", "dense_prefetch") else min(self.slots, capacity)
        maps = capacity * 4 + slots * 16
        records = slots * record if self.scheme != "dense_prefetch" else 0
        # ECHO retains one four-byte prefetch counter per forward chunk until
        # truncate or the next request. This is an upper bound including prefix.
        counters = math.ceil(capacity / self.chunk_size) * 4 if self.scheme == "echo" else 0
        return {
            "hbm": self.num_layers * (index + maps + records + counters)
            + (2 * capacity * record if self.scheme == "dense_prefetch" else 0),
            "dram": self.num_layers * capacity * record if self.scheme != "hbm" else 0,
        }

    def create_session(self, capacity):
        self.estimate_session_bytes(capacity)
        width = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim
        stages = (
            [
                torch.empty((capacity, width), device=self.device, dtype=torch.bfloat16)
                for _ in range(2)
            ]
            if self.scheme == "dense_prefetch"
            else []
        )
        runners = [
            _ServingAttention(
                attention,
                capacity,
                scheme=self.scheme,
                slots=self.slots,
                chunk_size=self.chunk_size,
                dense_records=stages[layer % 2] if stages else None,
            )
            for layer, attention in enumerate(self.attentions)
        ]
        return DeepSeekServingSession(
            capacity,
            self.scheme,
            runners,
            self,
            stages=stages,
            copy_stream=torch.cuda.Stream(device=self.device) if stages else None,
            stage_consumed=[None, None] if stages else [],
        )

    def _check_session(self, session):
        if session.owner is not self or session.released:
            raise ValueError("session belongs to another backend or has been released")
        if session.scheme != self.scheme:
            raise ValueError("session cache policy differs from the backend")

    def session_bytes(self, session):
        self._check_session(session)
        tensors = list(session.stages)
        for runner in session.runners:
            cache = runner.cache
            tensors.extend(
                [
                    cache.records,
                    cache.host,
                    cache.host_to_device,
                    cache.device_to_host,
                    cache.age,
                    runner.index_keys,
                    runner.index_scales,
                    runner.offset,
                    *(counter for counter, _ in cache.prefetch_counts),
                ]
            )
        return _storage_bytes(tensors)

    def _prefetch_dense_layer(self, session, layer):
        cache = session.runners[layer].cache
        compute = torch.cuda.current_stream(self.device)
        stream = session.copy_stream
        with torch.cuda.stream(stream):
            stream.wait_stream(compute)
            previous = session.stage_consumed[layer % 2]
            if previous is not None:
                stream.wait_event(previous)
            cache.records[: cache.length].copy_(cache.host[: cache.length], non_blocking=True)
            cache.dense_fetched_records += cache.length
            ready = torch.cuda.Event()
            ready.record(stream)
        return ready

    @torch.inference_mode()
    def _forward(self, session, token_ids, *, scope=None):
        self._check_session(session)
        if self._busy:
            raise RuntimeError("the single-GPU backend executes one request at a time")
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or not len(ids) or session.length + len(ids) > session.capacity:
            raise ValueError("nonempty token IDs must fit the session cache")
        if int(ids.min()) < 0 or int(ids.max()) >= self.cfg.vocab_size:
            raise ValueError("token ID is outside the checkpoint vocabulary")
        scope = scope or (lambda _: nullcontext())
        started = []
        self._busy = True
        try:
            for runner in session.runners:
                runner.cache.reset_stats()
                runner.cache.begin_step(len(ids))
                started.append(runner.cache)
            with scope("embedding"):
                hidden = F.embedding(ids, self.embedding_weight)
                residual = None
            sources = []
            ready = self._prefetch_dense_layer(session, 0) if session.stages else None
            for layer, (block, runner) in enumerate(zip(self.blocks, session.runners, strict=True)):
                if ready is not None:
                    torch.cuda.current_stream(self.device).wait_event(ready)
                following = (
                    self._prefetch_dense_layer(session, layer + 1)
                    if session.stages and layer + 1 < self.num_layers
                    else None
                )
                if layer < 3:
                    sources.append((hidden, residual))
                else:
                    with scope("replay_input_copy"):
                        source_hidden, source_residual = sources[layer % 3]
                        hidden = source_hidden.clone()
                        residual = source_residual.clone() if source_residual is not None else None
                block.attention, block.cache = runner, runner.cache
                with scope(f"layer_{layer}_source_{layer % 3}"):
                    hidden, residual = block.forward(hidden, residual, scope=scope)
                if self.capture_hook is not None:
                    self.capture_hook(layer, hidden, residual)
                if session.stages:
                    consumed = torch.cuda.Event()
                    consumed.record(torch.cuda.current_stream(self.device))
                    session.stage_consumed[layer % 2] = consumed
                ready = following
            with scope("final_norm_lm_head"):
                output = rms_norm(
                    hidden.float() + residual.float(), self.final_norm, self.cfg.norm_eps
                )
                output = output.bfloat16()
                self.last_logits = F.linear(output[-1:], self.head_weight).float()
            self.synchronize()
            if any(cache.written != cache._step_end for cache in started):
                raise RuntimeError("one or more layers did not complete their cache step")
            for cache in started:
                cache.commit()
            session.length += len(ids)
            return output
        except BaseException as error:
            # Drain valid in-flight work before making append slots reusable.
            try:
                self.synchronize()
            except RuntimeError as synchronization_error:
                error.add_note(f"draining GPU work also failed: {synchronization_error}")
            for cache in started:
                if cache._step_end is not None:
                    cache.rollback()
            raise
        finally:
            for block in self.blocks:
                block.attention = block.cache = None
            self._busy = False

    def prefill(self, session, token_ids):
        self._check_session(session)
        if session.length:
            raise ValueError("prefill requires an empty user session")
        return self._forward(session, token_ids)

    def extend(self, session, token_ids):
        return self._forward(session, token_ids)

    def truncate(self, session, prefix_tokens):
        self._check_session(session)
        if not 0 <= prefix_tokens <= session.length:
            raise ValueError("truncation cannot extend a prefix")
        self.synchronize()
        for runner in session.runners:
            runner.cache.truncate(prefix_tokens)
            runner.cache.reset_stats()
        session.length = prefix_tokens

    def session_metrics(self, session):
        self._check_session(session)
        metrics = [runner.cache.metrics() for runner in session.runners]
        return {
            "host_to_device_bytes": sum(item["host_to_device_bytes"] for item in metrics),
            "device_to_host_bytes": sum(item["device_to_host_bytes"] for item in metrics),
            "prefetched_records": sum(item["prefetched_records"] for item in metrics),
            "recalled_records": sum(item["recalled_records"] for item in metrics),
            "layers": metrics,
        }

    def release_session(self, session):
        self._check_session(session)
        self.synchronize()
        session.runners.clear()
        session.stages.clear()
        session.stage_consumed.clear()
        session.copy_stream = None
        session.released = True

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def describe(self):
        tensors = [self.embedding_weight, self.head_weight, self.final_norm]
        for attention, block in zip(self.attentions, self.blocks, strict=True):
            for item in (attention, block, block.mlp):
                for value in vars(item).values():
                    if isinstance(value, torch.Tensor):
                        tensors.append(value)
                    elif hasattr(value, "weight") and isinstance(value.weight, torch.Tensor):
                        tensors.extend((value.weight, getattr(value, "scales", None)))
        return {
            "model": "DeepSeek-V3.2-dense-layer-input-replay",
            "checkpoint": str(self.path),
            "scheme": self.scheme,
            "device": str(self.device),
            "physical_layers": self.num_layers,
            "source_layers": [layer % 3 for layer in range(self.num_layers)],
            "input_semantics": "copy_source_layer_hidden_and_residual_for_each_physical_copy",
            "model_scope": "checkpoint-backed workload surrogate; not a trained 8B model",
            "linear_backend": self.linear_backend,
            "chunk_size": self.chunk_size,
            "sparse_slots": self.slots,
            "weights": _storage_bytes(tensors),
            "output": "all_candidate_normalized_hidden_and_last_token_lm_head",
            "cache_budget_scope": "all retained user KV, indexer, mappings, counters and staging",
            "cache_budget_excludes": "weights, transient activations, selections and GEMM workspace",
            **self.parameter_counts,
        }
