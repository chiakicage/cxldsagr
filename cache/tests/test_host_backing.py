"""Host-backed appends own their storage and publish only successful steps."""

import pytest
import torch

from cache.host_backing import HostBackingCache
from cache.manager import CacheSpec


def make_cache():
    return HostBackingCache(
        CacheSpec(2, 16, {"record": (3,)}), 16, device="cpu", dtype=torch.float32
    )


def write(cache, layer, values):
    cache.write_layer(layer, record=values)


def test_owned_append_is_not_changed_by_caller_and_prefix_survives_abort():
    cache = make_cache()
    assert not hasattr(cache, "layer_view")
    assert cache.begin_step(3) == (0, 3)
    values = torch.arange(9, dtype=cache.dtype).reshape(3, 3)
    expected = values.clone()
    write(cache, 0, values)
    write(cache, 1, values)
    values.fill_(-100)
    torch.testing.assert_close(cache.suffix_layer_view(0)["record"], expected)
    assert cache.length == 0
    cache.commit_step()
    torch.testing.assert_close(cache.host_layer_view(0)["record"][:3], expected)
    assert cache.suffix_layer_view(0)["record"].shape == (0, 3)
    cache.begin_step(2)
    write(cache, 0, torch.zeros((2, 3)))
    cache.set_layer_state(0, "pending")
    with pytest.raises(RuntimeError, match="Every layer"):
        cache.commit_step()
    cache.abort_step()
    assert cache.length == 3 and cache.get_layer_state(0) is None
    torch.testing.assert_close(cache.host_layer_view(0)["record"][:3], expected)
    cache.begin_step(2)
    for layer in range(2):
        write(cache, layer, torch.full((2, 3), 4.0))
    cache.commit_step()
    assert cache.length == 5
    assert cache.stats()["host_bytes"] == 2 * 16 * 3 * 4
    assert cache.stats()["resident_bytes"] == 0


def test_failed_synchronization_does_not_publish_or_discard_pending_storage(monkeypatch):
    cache = make_cache()
    cache.begin_step(2)
    for layer in range(2):
        write(cache, layer, torch.ones((2, 3)))

    def fail():
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(cache, "synchronize", fail)
    with pytest.raises(RuntimeError, match="transfer failed"):
        cache.commit_step()
    assert cache.length == 0
    assert cache.visible_length(0) == 2
    assert cache.stats()["resident_bytes"] == 2 * 2 * 3 * 4
    monkeypatch.setattr(cache, "synchronize", lambda: None)
    cache.abort_step()
    assert cache.length == 0 and cache.stats()["resident_bytes"] == 0


def test_invalid_writes_and_lifecycle_do_not_expose_uninitialized_records():
    cache = make_cache()
    with pytest.raises(RuntimeError, match="begin_step"):
        write(cache, 0, torch.ones((2, 3)))
    cache.begin_step(2)
    with pytest.raises(RuntimeError, match="already pending"):
        cache.begin_step(1)
    with pytest.raises(RuntimeError, match="has not written"):
        cache.suffix_layer_view(0)
    with pytest.raises(ValueError, match="shape"):
        write(cache, 0, torch.ones((3, 3)))
    write(cache, 0, torch.ones((2, 3)))
    with pytest.raises(RuntimeError, match="already written"):
        write(cache, 0, torch.ones((2, 3)))
    cache.reset()
    assert cache.length == 0
    cache.release()
    cache.release()
    assert cache.stats()["resident_bytes"] == cache.stats()["host_bytes"] == 0
    with pytest.raises(RuntimeError, match="released"):
        cache.begin_step(1)
