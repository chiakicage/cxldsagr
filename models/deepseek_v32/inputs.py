"""Validate a sequence on its source device before checkpoint execution."""

import torch


def prepare_token_ids(token_ids, *, device, vocab_size, max_tokens):
    """Keep host input validation on CPU; copy only the accepted sequence.

    CUDA callers retain value validation, with a single bounds transfer. No
    caller can bypass vocabulary or shape checks by changing input storage.
    """
    ids = (
        token_ids.to(dtype=torch.long)
        if isinstance(token_ids, torch.Tensor)
        else torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
    )
    if ids.ndim != 1 or not 0 < ids.numel() <= max_tokens:
        raise ValueError("expected a nonempty single sequence fitting cache capacity")
    low, high = torch.aminmax(ids)
    if ids.device.type != "cpu":
        low, high = torch.stack((low, high)).cpu().unbind()
    if int(low) < 0 or int(high) >= vocab_size:
        raise ValueError("token ID is outside the checkpoint vocabulary")
    # The forward's execution stream consumes this copy and synchronizes before
    # committing. A separate input-copy synchronization only delays enqueue.
    return ids.to(device=device, non_blocking=True)
