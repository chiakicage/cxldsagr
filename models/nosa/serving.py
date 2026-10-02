"""Persistent NOSA sessions for the serial multi-user GR serving runner.

The shared serving cache manager owns admission and eviction. This adapter
owns model-specific cache layout and execution; it never reads GR requests.
All schemes execute complete NOSA sparse attention, including CIS. ``dense``
describes the volume transferred, not a different attention mask.
"""

from dataclasses import replace

import torch

from models.nosa.cache import NosaKVCache
from models.nosa.model import NosaForCausalLM
from models.nosa.offload_cache import NosaOffloadCache


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


class NosaDensePrefetchCache(NosaOffloadCache):
    """Pinned historical K/V with two alternating, full-layer HBM buffers.

    Layer zero is queued at the start of each append. Once the current layer has
    written its suffix, its stream waits for that layer's history and queues the
    next layer's complete history on a copy stream. The copy stream waits for all
    previous consumers before reusing a slot. Thus next-layer copies can overlap
    the current layer's indexer, attention, and MLP. No sparse fetch is performed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (2, self.max_seq_len, self.config.num_key_value_heads, self.config.head_dim)
        self._stage_keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        self._stage_values = torch.empty_like(self._stage_keys)
        self._prefetch_stream = None
        self._ready = [None, None]
        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
            self._ready = [torch.cuda.Event(), torch.cuda.Event()]
        self._staged_layers = [None, None]
        self.last_prefetch_bytes = 0

    @property
    def attention_workspace(self):
        raise RuntimeError("Dense prefetch uses double staging, not sparse fetch workspace")

    def begin_step(self, token_count):
        result = super().begin_step(token_count)
        self._staged_layers = [None, None]
        self.last_prefetch_bytes = 0
        try:
            self._prefetch(0)
        except BaseException:
            self.abort_step()
            raise
        return result

    @torch.no_grad()
    def _prefetch(self, layer_idx):
        if layer_idx >= self.spec.num_layers:
            return
        slot = layer_idx % 2
        host = self.host_layer_view(layer_idx)

        def copy_history():
            self._stage_keys[slot, : self.length].copy_(
                host["keys"][: self.length], non_blocking=self.device.type == "cuda"
            )
            self._stage_values[slot, : self.length].copy_(
                host["values"][: self.length], non_blocking=self.device.type == "cuda"
            )

        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            # The current stream includes every earlier attention consumer.
            # This fences slot reuse, while later current-layer compute can
            # overlap the next-layer history copy queued below.
            self._prefetch_stream.wait_stream(current)
            with torch.cuda.stream(self._prefetch_stream):
                copy_history()
                self._ready[slot].record(self._prefetch_stream)
        else:
            copy_history()
        self.last_prefetch_bytes += (
            2
            * self.length
            * self.config.num_key_value_heads
            * self.config.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )
        self._staged_layers[slot] = layer_idx

    @torch.no_grad()
    def write_layer(self, layer_idx, **records):
        if self._staged_layers[layer_idx % 2] != layer_idx:
            raise RuntimeError("Dense prefetch requires layers in increasing order")
        super().write_layer(layer_idx, **records)
        slot = layer_idx % 2
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_event(self._ready[slot])
        self._stage_keys[slot, self.length : self._pending_end].copy_(records["keys"])
        self._stage_values[slot, self.length : self._pending_end].copy_(records["values"])
        self._prefetch(layer_idx + 1)

    def dense_layer_view(self, layer_idx):
        end = self.visible_length(layer_idx)
        slot = layer_idx % 2
        if self._staged_layers[slot] != layer_idx:
            raise RuntimeError("Requested layer no longer owns its dense staging slot")
        return {
            "keys": self._stage_keys[slot, :end],
            "values": self._stage_values[slot, :end],
            "cis_scores": self.cis_scores[layer_idx, :end],
        }

    def stats(self):
        result = super().stats()
        staging = sum(_bytes(t) for t in (self._stage_keys, self._stage_values) if t is not None)
        result["resident_bytes"] += staging
        result["capacity_bytes"] += staging
        return result

    def release(self):
        if self.released:
            return
        super().release()
        self._stage_keys = self._stage_values = None
        self._prefetch_stream = None
        self._ready = [None, None]
        self._staged_layers = [None, None]


class _DensePrefetchAttention:
    def __call__(self, q, selection, cache_access, context):
        if not isinstance(cache_access, NosaDensePrefetchCache):
            raise TypeError("Dense prefetch attention requires its matching cache")
        records = cache_access.dense_layer_view(context.layer_idx)
        if q.is_cuda:
            from operators.nosa.attention.device_only.api import nosa_block_sparse_attention

            attention = nosa_block_sparse_attention
        else:
            from operators.nosa.attention.reference.torch import (
                reference_nosa_block_sparse_attention,
            )

            attention = reference_nosa_block_sparse_attention
        return attention(
            q,
            records["keys"],
            records["values"],
            selection,
            query_start=context.query_start,
            cis_bias=records["cis_scores"],
        )


class NosaServingBackend:
    """Full NOSA model with persistent, independently evictable user sessions.

    ``estimate_session_bytes`` reserves cache tensor capacities, including the
    largest owned pending append and scratch. It excludes model weights, hidden
    activations, and temporary operator outputs; these are not KV cache budgets.
    There is no per-request weight loading, CPU snapshotting, or prefix replay on
    a hit. Caller-owned LRU eviction discards both tiers of one user's session.
    """

    schemes = ("hbm", "serial_sparse", "dense_prefetch", "overlap")

    def __init__(self, model, scheme, *, chunk_size=1024):
        if scheme not in self.schemes:
            raise ValueError(f"NOSA serving scheme must be one of {self.schemes}")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if model.attention_mode != "sparse":
            raise ValueError("All NOSA serving schemes require the complete sparse NOSA policy")
        self.model = model
        self.scheme = scheme
        self.chunk_size = chunk_size
        self.config = model.config
        self.max_seq_len = self.config.max_position_embeddings
        self.device = model.model.embed_tokens.weight.device
        self.dtype = model.model.embed_tokens.weight.dtype
        self._dense_attention = _DensePrefetchAttention()

    @classmethod
    def from_pretrained(
        cls, checkpoint, *, scheme, device="cuda:0", chunk_size=1024, max_seq_len=None
    ):
        model = NosaForCausalLM.from_pretrained(
            checkpoint,
            device=device,
            dtype=torch.bfloat16,
            attention_mode="sparse",
            sparse_backend="auto",
        )
        if (model.config.num_hidden_layers, len(model.model.layers)) != (32, 32):
            raise ValueError("NOSA-8B serving requires all 32 checkpoint layers")
        if max_seq_len is not None:
            if type(max_seq_len) is not int or not 0 < max_seq_len <= 262144:
                raise ValueError("max_seq_len must be an integer in [1,262144]")
            model.config = replace(model.config, max_position_embeddings=max_seq_len)
        backend = cls(model, scheme, chunk_size=chunk_size)
        backend.checkpoint = str(checkpoint)
        return backend

    def _validate_capacity(self, capacity):
        if type(capacity) is not int or not 0 < capacity <= self.max_seq_len:
            raise ValueError("Session capacity must fit the model context limit")

    def estimate_session_bytes(self, capacity, prefix_tokens):
        self._validate_capacity(capacity)
        if type(prefix_tokens) is not int or not 0 < prefix_tokens < capacity:
            raise ValueError("Prefix must leave a nonempty candidate suffix")
        cfg = self.config
        width = cfg.num_key_value_heads * cfg.head_dim
        heads, layers = cfg.num_key_value_heads, cfg.num_hidden_layers
        size = torch.empty((), dtype=self.dtype).element_size()
        queries = max(min(self.chunk_size, prefix_tokens), capacity - prefix_tokens)
        compressed, pooled = NosaKVCache._indexer_lengths(capacity)
        kv = 2 * layers * capacity * width * size
        cis = layers * capacity * heads * size
        derived = layers * (compressed * (width + heads) + pooled * heads) * size
        pages = (capacity + 63) // 64
        score = queries * heads * pages * size
        # Both native checked preparation and generic preparation fit here.
        indexer_scratch = ((score + 255) // 256 + 13) * 256 + heads * 64 * 4
        indexer_scratch = (indexer_scratch + 255) // 256 * 256
        # Growing a lazy buffer briefly retains the previous allocation until
        # the new tensor is assigned. Reserve both to cover that cache peak.
        indexer_scratch *= 2
        if self.scheme == "hbm":
            return {"hbm": kv + cis + derived + indexer_scratch, "dram": 0}
        pending = 2 * layers * queries * width * size
        hbm = cis + derived + indexer_scratch + pending
        if self.scheme == "dense_prefetch":
            hbm += 4 * capacity * width * size
        else:
            hbm += 2 * capacity * width * size
            batches = ((queries + 7) // 8) * heads
            # first_use/ready/fetch_queue, tile counters, and native FA3 scratch.
            hbm += (3 * pages * heads + 2) * 4
            hbm += ((capacity + 127) // 128 + 1) * 8
            hbm += (((queries + 3) // 4) * heads + batches * (2 * 8 * 64 + 2)) * 8
        return {"hbm": hbm, "dram": kv}

    def create_session(self, capacity):
        self._validate_capacity(capacity)
        options = {"device": self.device, "dtype": self.dtype, "with_cis": True}
        if self.scheme == "hbm":
            return NosaKVCache(self.config, capacity, **options)
        cache_type = NosaDensePrefetchCache if self.scheme == "dense_prefetch" else NosaOffloadCache
        return cache_type(
            self.config, capacity, overlap=self.scheme == "overlap", query_tile_size=128, **options
        )

    def _forward(self, session, ids):
        original = self.model.main_attention
        if self.scheme == "dense_prefetch":
            self.model.main_attention = self._dense_attention
        try:
            return self.model(ids, session, return_hidden=True)
        finally:
            self.model.main_attention = original

    @torch.inference_mode()
    def prefill(self, session, ids):
        if session.length != 0:
            raise ValueError("Prefix construction requires an empty session")
        if ids.ndim != 1 or not len(ids):
            raise ValueError("Prefix must be a nonempty one-dimensional token tensor")
        for start in range(0, len(ids), self.chunk_size):
            self._forward(session, ids[start : start + self.chunk_size])

    @torch.inference_mode()
    def extend(self, session, ids):
        return self._forward(session, ids)

    def truncate(self, session, prefix):
        # Resident truncate publishes metadata only, so synchronize before the
        # next user or a possible eviction may reuse/free its tensors.
        self.synchronize()
        session.truncate(prefix)

    def session_bytes(self, session):
        stats = session.stats()
        return {"hbm": stats["resident_bytes"], "dram": stats["host_bytes"]}

    def release_session(self, session):
        self.synchronize()
        session.release()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def describe(self):
        return {
            "model": "nosa",
            "checkpoint": getattr(self, "checkpoint", None),
            "scheme": self.scheme,
            "layers": len(self.model.model.layers),
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "dtype": str(self.dtype),
            "device": str(self.device),
            "max_seq_len": self.max_seq_len,
            "prefix_chunk_size": self.chunk_size,
            "attention_policy": "full NOSA: 33 QA incl sink/local, then CIS to 64 blocks",
            "output": "all candidate normalized hidden states; no LM head or decode",
            "cache_eviction": "whole-user session; both HBM and DRAM released",
            "cache_budget_scope": "tensor capacities, derived records, staging, owned append, scratch",
            "dense_prefetch": "full next-layer historical K/V on copy stream, two fenced HBM slots",
            "sparse_offload": "native full-batch unique sparse union; overlap uses cooperative FA3",
        }
