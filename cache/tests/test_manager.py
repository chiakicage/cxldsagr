"""Request cache isolation, failure recovery, and allocation lifetime."""

import gc
import weakref

import pytest
import torch

from cache.manager import CacheManager, CacheSpec, ResidentCache


def cache_spec():
    return CacheSpec(2, 16, {"keys": (2, 4), "values": (2, 4)})


def new_cache(capacity=8):
    return ResidentCache(cache_spec(), capacity, device="cpu", dtype=torch.float32)


def write_step(session, token_count, offset=0):
    session.begin_step(token_count)
    for layer in range(session.spec.num_layers):
        values = torch.arange(token_count * 8).reshape(token_count, 2, 4).float()
        session.write_layer(layer, keys=values + offset + layer, values=-values - offset - layer)
    session.commit_step()


def test_layout_capacity_and_cpu_backend_statistics():
    # Different record layouts are model-declared, with no fixed MLA assumption.
    spec = CacheSpec(3, 20, {"keys": (2, 4), "values": (2, 6), "score": ()})
    session = ResidentCache(spec, 10, device="cpu", dtype=torch.bfloat16)
    assert session.buffers["keys"].shape == (3, 10, 2, 4)
    assert session.buffers["values"].shape == (3, 10, 2, 6)
    assert session.buffers["score"].shape == (3, 10)
    assert session.stats() == {
        "capacity_bytes": 3 * 10 * (8 + 12 + 1) * 2,
        "resident_bytes": 3 * 10 * (8 + 12 + 1) * 2,
        "host_bytes": 0,
        "length": 0,
        "max_seq_len": 10,
    }
    with pytest.raises(ValueError, match="capacity"):
        ResidentCache(spec, 21, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="positive integer"):
        new_cache(0)


def test_layers_see_pending_records_before_length_is_committed():
    session = new_cache()
    keys = torch.arange(24).reshape(3, 2, 4).float()
    assert session.begin_step(3) == (0, 3)
    session.write_layer(0, keys=keys, values=-keys)
    assert session.length == 0
    torch.testing.assert_close(session.layer_view(0)["keys"], keys)
    with pytest.raises(RuntimeError, match="has not written"):
        session.layer_view(1)
    with pytest.raises(RuntimeError, match="Every layer"):
        session.commit_step()
    assert session.length == 0
    session.write_layer(1, keys=keys + 1, values=-keys - 1)
    session.commit_step()
    assert session.length == 3
    torch.testing.assert_close(session.layer_view(1)["values"], -keys - 1)
    # A subsequent append keeps the already committed prefix intact.
    write_step(session, 2, offset=100)
    assert session.length == 5
    torch.testing.assert_close(session.layer_view(0)["keys"][:3], keys)
    torch.testing.assert_close(
        session.layer_view(0)["keys"][3:], torch.arange(16).reshape(2, 2, 4).float() + 100
    )


def test_failed_append_can_abort_and_overwrite_without_exposing_suffix():
    session = new_cache(capacity=4)
    write_step(session, 2)
    prefix = {name: tensor.clone() for name, tensor in session.layer_view(0).items()}
    session.begin_step(2)
    session.write_layer(0, keys=torch.full((2, 2, 4), 99.0), values=torch.zeros(2, 2, 4))
    with pytest.raises(ValueError, match="shape"):
        session.write_layer(1, keys=torch.zeros(2, 2, 4), values=torch.zeros(1, 2, 4))
    session.abort_step()
    assert session.length == 2
    for name, expected in prefix.items():
        torch.testing.assert_close(session.layer_view(0)[name], expected)
    with pytest.raises(ValueError, match="capacity"):
        session.begin_step(3)
    write_step(session, 2, offset=10)
    assert session.length == 4
    torch.testing.assert_close(
        session.layer_view(0)["keys"][2:], torch.arange(16).reshape(2, 2, 4).float() + 10
    )


def test_invalid_step_and_record_usage_is_rejected():
    session = new_cache()
    records = {"keys": torch.zeros(1, 2, 4), "values": torch.zeros(1, 2, 4)}
    with pytest.raises(RuntimeError, match="begin_step"):
        session.write_layer(0, **records)
    with pytest.raises(RuntimeError, match="No cache step"):
        session.commit_step()
    with pytest.raises(ValueError, match="positive integer"):
        session.begin_step(0)
    session.begin_step(1)
    with pytest.raises(RuntimeError, match="already pending"):
        session.begin_step(1)
    with pytest.raises(ValueError, match="names"):
        session.write_layer(0, keys=records["keys"])
    with pytest.raises(ValueError, match="device and dtype"):
        session.write_layer(0, keys=records["keys"].half(), values=records["values"])
    with pytest.raises(ValueError, match="Layer index"):
        session.write_layer(-1, **records)
    session.write_layer(0, **records)
    with pytest.raises(RuntimeError, match="already written"):
        session.write_layer(0, **records)
    session.abort_step()
    session.abort_step()


def test_model_state_replacements_commit_abort_and_reset_with_request():
    session = new_cache()
    original, replacement = object(), object()
    session.set_layer_state(0, original)
    session.begin_step(1)
    session.set_layer_state(0, replacement)
    assert session.get_layer_state(0) is replacement
    session.abort_step()
    assert session.get_layer_state(0) is original
    session.begin_step(1)
    session.set_layer_state(0, replacement)
    for layer in range(2):
        session.write_layer(layer, keys=torch.zeros(1, 2, 4), values=torch.zeros(1, 2, 4))
    session.commit_step()
    assert session.get_layer_state(0) is replacement
    pointers = {name: tensor.data_ptr() for name, tensor in session.buffers.items()}
    session.begin_step(1)
    session.reset()
    assert session.length == 0
    assert session.get_layer_state(0) is None
    assert {name: tensor.data_ptr() for name, tensor in session.buffers.items()} == pointers
    write_step(session, 1)


def test_manager_isolates_requests_and_releases_owned_buffers():
    manager = CacheManager(new_cache)
    first, second = manager.allocate(4), manager.allocate(4)
    write_step(first, 2)
    assert second.length == 0
    assert first.buffers["keys"].data_ptr() != second.buffers["keys"].data_ptr()
    tensor_ref = weakref.ref(first.buffers["keys"])
    first.set_layer_state(1, torch.ones(1))
    manager.release(first)
    manager.release(first)
    assert tensor_ref() is None
    assert first.stats()["resident_bytes"] == 0
    assert first.released
    assert second.stats()["resident_bytes"] > 0
    for operation in (
        lambda: first.begin_step(1),
        lambda: first.layer_view(0),
        lambda: first.get_layer_state(1),
        first.reset,
    ):
        with pytest.raises(RuntimeError, match="released"):
            operation()
    with pytest.raises(ValueError, match="does not belong"):
        CacheManager(new_cache).release(second)
    manager.release(second)


def test_manager_does_not_keep_legacy_sessions_alive():
    manager = CacheManager(new_cache)
    session = manager.allocate(4)
    session_ref, tensor_ref = weakref.ref(session), weakref.ref(session.buffers["keys"])
    del session
    gc.collect()
    assert session_ref() is None
    assert tensor_ref() is None


def test_manager_accepts_a_session_without_resident_tensor_views():
    class Session:
        def __init__(self, capacity):
            self.max_seq_len = capacity
            self.released = False

        def release(self):
            self.released = True

    manager = CacheManager(Session)
    session = manager.allocate(10)
    assert session.max_seq_len == 10
    manager.release(session)
    assert session.released
