"""NOSA's post-RoPE K / unmodified V layout on the shared resident backend."""

from copy import deepcopy

from cache.indexer_cache import IndexerBufferSpec, IndexerCache
from cache.manager import CacheSpec, ResidentCache
from models.nosa.config import NosaConfig


class NosaKVCache(ResidentCache):
    """One unpadded request, with [layer, capacity, kv_head, head_dim] tensors.

    The original constructor and ``config``, ``keys``, ``values`` attributes are
    kept for callers. Model layers write through the session operations rather
    than managing the physical buffers themselves.
    """

    def __init__(self, config: NosaConfig, max_seq_len: int, *, device, dtype, with_cis=False):
        if with_cis and isinstance(max_seq_len, int) and max_seq_len > 262144:
            raise ValueError("Full NOSA indexer cache supports at most 262144 tokens")
        # LongRoPE factors live in a nested mutable dict even though NosaConfig
        # is frozen. A snapshot makes the model reject changed rotation settings
        # instead of appending differently rotated keys to an existing cache.
        self.config = deepcopy(config)
        self.with_cis = with_cis
        shape = (config.num_key_value_heads, config.head_dim)
        records = {"keys": shape, "values": shape}
        if with_cis:
            records["cis_scores"] = (config.num_key_value_heads,)
        spec = CacheSpec(
            num_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            record_shapes=records,
            compatibility_key=(self.config, with_cis),
        )
        super().__init__(spec, max_seq_len, device=device, dtype=dtype)
        self.indexer_cache = None
        if with_cis:
            compressed, pooled = self._indexer_lengths(max_seq_len)
            self.indexer_cache = IndexerCache(
                config.num_hidden_layers,
                max_seq_len,
                {
                    "compressed_keys": IndexerBufferSpec(compressed, shape, dtype),
                    "compressed_cis": IndexerBufferSpec(compressed, (shape[0],), dtype),
                    "pooled_cis": IndexerBufferSpec(pooled, (shape[0],), dtype),
                },
                device=self.device,
            )

    @staticmethod
    def _indexer_lengths(length):
        return max(0, length // 16 - 1), max(0, (length - 16) // 64)

    def begin_step(self, token_count):
        start, end = super().begin_step(token_count)
        try:
            if self.indexer_cache is not None:
                self.indexer_cache.begin_step(end)
        except BaseException:
            super().abort_step()
            raise
        return start, end

    def validate_commit(self):
        self._ensure_alive()
        if self._pending_end is None:
            raise RuntimeError("No cache step is pending")
        if len(self._written_layers) != self.spec.num_layers:
            raise RuntimeError("Every layer must write before committing a cache step")
        if self.indexer_cache is not None:
            self.indexer_cache.validate_commit()

    def commit_step(self):
        # Validate both sides before either committed cursor advances. The
        # following commits only publish metadata and cannot launch work.
        self.validate_commit()
        super().commit_step()
        if self.indexer_cache is not None:
            self.indexer_cache.commit_step()

    def abort_step(self):
        super().abort_step()
        if self.indexer_cache is not None:
            self.indexer_cache.abort_step()

    def truncate(self, length):
        self._ensure_alive()
        if self._pending_end is not None:
            raise RuntimeError("Cannot truncate while a cache step is pending")
        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError("Invalid cache length: truncate only accepts a committed prefix")
        if 0 < length < self.length and self._layer_states:
            raise ValueError("Cannot truncate opaque layer state")
        if self.indexer_cache is not None:
            compressed, pooled = self._indexer_lengths(length)
            self.indexer_cache.truncate(
                length,
                {
                    "compressed_keys": compressed,
                    "compressed_cis": compressed,
                    "pooled_cis": pooled,
                },
            )
        self._length = length
        if length == 0:
            self._layer_states.clear()

    def reset(self):
        super().reset()
        if self.indexer_cache is not None:
            self.indexer_cache.reset()

    def release(self):
        if self.released:
            return
        super().release()
        if self.indexer_cache is not None:
            self.indexer_cache.release()

    def stats(self):
        result = super().stats()
        if self.indexer_cache is not None:
            derived = self.indexer_cache.stats()
            result["capacity_bytes"] += derived["capacity_bytes"]
            result["resident_bytes"] += derived["resident_bytes"]
        return result

    @property
    def keys(self):
        return self.buffers["keys"]

    @property
    def values(self):
        return self.buffers["values"]

    @property
    def cis_scores(self):
        return self.buffers["cis_scores"]

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        """Keep legacy rewinds without exposing uncommitted or stale records."""
        self.truncate(value)
