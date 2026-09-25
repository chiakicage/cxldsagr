"""Metadata shared by resident and future offloaded cache access.

Main attention receives cache access before fetching selected blocks. This
protocol deliberately does not require a resident tensor view: an offloading
operator must be free to overlap its fetches with attention computation.
"""

from typing import Protocol


class CacheAccess(Protocol):
    """Committed token count and allocated token capacity for one request."""

    @property
    def length(self) -> int: ...

    @property
    def max_seq_len(self) -> int: ...

    def get_layer_state(self, layer_idx: int) -> object | None: ...

    def set_layer_state(self, layer_idx: int, state: object) -> None: ...


class CacheSession(Protocol):
    """Manager ownership independent of record storage and transfer backend."""

    @property
    def released(self) -> bool: ...

    def release(self) -> None: ...
