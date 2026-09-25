"""NOSA's post-RoPE K / unmodified V layout on the shared resident backend."""

from cache.manager import CacheSpec, ResidentCache
from models.nosa.config import NosaConfig


class NosaKVCache(ResidentCache):
    """One unpadded request, with [layer, capacity, kv_head, head_dim] tensors.

    The original constructor and ``config``, ``keys``, ``values`` attributes are
    kept for callers. Model layers write through the session operations rather
    than managing the physical buffers themselves.
    """

    def __init__(self, config: NosaConfig, max_seq_len: int, *, device, dtype):
        self.config = config
        shape = (config.num_key_value_heads, config.head_dim)
        spec = CacheSpec(
            num_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            record_shapes={"keys": shape, "values": shape},
            compatibility_key=config,
        )
        super().__init__(spec, max_seq_len, device=device, dtype=dtype)

    @property
    def keys(self):
        return self.buffers["keys"]

    @property
    def values(self):
        return self.buffers["values"]

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
