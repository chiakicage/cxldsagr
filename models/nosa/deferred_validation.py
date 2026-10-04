"""One pre-commit finite decision for a serial, transactional model forward."""

import torch


class DeferredValidation:
    def __init__(self, layers, device):
        self.layers, self.device = layers, device
        self.flags = self.result = None
        self.owner = None
        self.seen = set()

    def allocate(self):
        self.flags = torch.empty(self.layers, device=self.device, dtype=torch.bool)
        self.result = torch.empty((), device=self.device, dtype=torch.bool)
        self.views = tuple(self.flags[index] for index in range(self.layers))

    @property
    def tensors(self):
        return () if self.flags is None else (self.flags, self.result)

    def begin(self, owner):
        if self.owner is not None or getattr(owner, "_deferred_validation", None) is not None:
            raise RuntimeError("Deferred validation requires an exclusive cache transaction")
        if self.flags is None or owner._pending_end is None or owner.device != self.device:
            raise RuntimeError("Deferred validation requires allocated flags and a pending step")
        self.owner = owner
        self.seen.clear()
        owner._deferred_validation = self

    def flag(self, owner, layer):
        if owner is not self.owner or type(layer) is not int or not 0 <= layer < self.layers:
            raise RuntimeError("Deferred finite flag must belong to its active cache/layer")
        if layer in self.seen:
            raise RuntimeError("Every deferred layer must be validated exactly once")
        self.seen.add(layer)
        return self.views[layer]

    def check(self, owner):
        if owner is not self.owner or len(self.seen) != self.layers:
            raise RuntimeError("Every model layer must submit a finite check before commit")
        torch.all(self.flags, out=self.result)
        if not self.result.item():
            raise ValueError("NOSA indexer requires finite Q, K and CIS scores")

    def clear(self):
        if self.owner is not None:
            self.owner._deferred_validation = None
        self.owner = None
        self.seen.clear()

    def close(self):
        self.clear()
        self.views = ()
        self.flags = self.result = None


def finite_flag(owner, layer):
    batch = getattr(owner, "_deferred_validation", None)
    if batch is None:
        return None
    if not isinstance(batch, DeferredValidation):
        raise TypeError("Invalid deferred NOSA validation owner")
    return batch.flag(owner, layer)
