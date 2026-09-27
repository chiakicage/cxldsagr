"""NOSA's post-RoPE K / unmodified V layout on the shared resident backend."""

from copy import deepcopy

from cache.manager import CacheSpec, ResidentCache
from models.nosa.config import NosaConfig


class NosaKVCache(ResidentCache):
    """One unpadded request, with [layer, capacity, kv_head, head_dim] tensors.

    The original constructor and ``config``, ``keys``, ``values`` attributes are
    kept for callers. Model layers write through the session operations rather
    than managing the physical buffers themselves.
    """

    def __init__(self, config: NosaConfig, max_seq_len: int, *, device, dtype, with_cis=False):
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
        """Keep the legacy experiment cursor override outside active steps.

        New callers use session transactions/reset. The model still validates
        manually supplied lengths before a forward, as the original API did.
        """
        self._ensure_alive()
        if self._pending_end is not None:
            raise RuntimeError("Cannot override length while a cache step is pending")
        self._length = value
