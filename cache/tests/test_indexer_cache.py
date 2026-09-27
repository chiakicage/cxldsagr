"""Derived-record transactions and request-wide workspace lifetime."""

import weakref

import pytest
import torch

from cache.indexer_cache import IndexerBufferSpec, IndexerCache


def new_cache():
    return IndexerCache(
        2,
        32,
        {
            "compressed": IndexerBufferSpec(8, (2,), torch.float32),
            "pooled": IndexerBufferSpec(4, (), torch.float16),
        },
        device="cpu",
    )


def materialize(cache, layer, tokens, compressed, pooled, value):
    reservation = cache.reserve_layer(layer, tokens, {"compressed": compressed, "pooled": pooled})
    for name, buffer in reservation.buffers.items():
        buffer[reservation.previous.lengths[name] : reservation.target.lengths[name]].fill_(value)
    cache.finish_layer(layer)


def test_unused_layers_do_not_allocate_or_block_owner_commit():
    cache = new_cache()
    assert cache.stats()["resident_bytes"] == 0
    assert cache.stats()["capacity_bytes"] == 2 * (8 * 2 * 4 + 4 * 2)
    cache.begin_step(8)
    cache.commit_step()
    assert cache.length == 8
    assert cache.stats()["resident_bytes"] == 0
    for layer in range(2):
        assert cache.layer_state(layer).validated_tokens == 0
        assert all(len(view) == 0 for view in cache.layer_view(layer).values())
    materialize(cache, 1, 8, 3, 1, 7)
    assert cache.layer_state(1).validated_tokens == 8
    assert cache.layer_state(0).validated_tokens == 0
    assert cache.stats()["resident_bytes"] == 8 * 2 * 4 + 4 * 2


def test_abort_hides_dirty_suffix_and_retry_preserves_committed_prefix():
    cache = new_cache()
    cache.begin_step(8)
    materialize(cache, 0, 8, 3, 1, 5)
    cache.commit_step()
    original = {name: view.clone() for name, view in cache.layer_view(0).items()}
    cache.begin_step(16)
    materialize(cache, 0, 16, 6, 3, 99)
    assert cache.length == 8
    assert cache.layer_state(0).validated_tokens == 16
    cache.abort_step()
    assert cache.layer_state(0).validated_tokens == 8
    for name, view in cache.layer_view(0).items():
        torch.testing.assert_close(view, original[name])
    cache.begin_step(16)
    materialize(cache, 0, 16, 6, 3, 11)
    cache.commit_step()
    for name, view in cache.layer_view(0).items():
        torch.testing.assert_close(view[: len(original[name])], original[name])
        assert torch.all(view[len(original[name]) :] == 11)


def test_shorter_views_and_reservations_keep_longer_materialized_state():
    cache = new_cache()
    cache.begin_step(16)
    materialize(cache, 0, 16, 6, 3, 9)
    cache.commit_step()
    state = cache.layer_state(0)
    visible = {"compressed": 2, "pooled": 0}
    assert cache.layer_view(0, lengths=visible)["compressed"].shape == (2, 2)
    reservation = cache.reserve_layer(0, 5, visible)
    assert reservation.previous == reservation.target == state
    assert reservation.views["pooled"].numel() == 0
    cache.finish_layer(0)
    assert cache.layer_state(0) == state
    with pytest.raises(TypeError):
        state.lengths["compressed"] = 0
    cache.truncate(5, visible)
    assert cache.layer_state(0).lengths == visible
    assert cache.layer_state(0).validated_tokens == 5
    assert cache.layer_state(1).validated_tokens == 0


def test_validated_tokens_advance_before_any_compression_window_exists():
    cache = new_cache()
    cache.begin_step(3)
    materialize(cache, 0, 3, 0, 0, 0)
    cache.commit_step()
    assert cache.layer_state(0).validated_tokens == 3
    assert cache.layer_state(0).lengths == {"compressed": 0, "pooled": 0}


def test_reservations_require_completion_and_cannot_expose_unwritten_records():
    cache = new_cache()
    cache.begin_step(8)
    cache.reserve_layer(0, 8, {"compressed": 3, "pooled": 1})
    with pytest.raises(RuntimeError, match="finish"):
        cache.validate_commit()
    with pytest.raises(RuntimeError, match="pending"):
        cache.reserve_layer(0, 8, {"compressed": 3, "pooled": 1})
    with pytest.raises(ValueError, match="materialized"):
        cache.layer_view(0, lengths={"compressed": 3, "pooled": 1})
    with pytest.raises(RuntimeError, match="pending"):
        cache.truncate(0, {"compressed": 0, "pooled": 0})
    cache.abort_layer(0)
    cache.commit_step()
    assert cache.layer_state(0).validated_tokens == 0
    with pytest.raises(RuntimeError, match="No.*reservation"):
        cache.finish_layer(0)


@pytest.mark.parametrize(
    ("tokens", "lengths"),
    [
        (9, {"compressed": 3, "pooled": 1}),
        (-1, {"compressed": 3, "pooled": 1}),
        (True, {"compressed": 3, "pooled": 1}),
        (8, {"compressed": 9, "pooled": 1}),
        (8, {"compressed": 3}),
        (8, {"compressed": 3.0, "pooled": 1}),
    ],
)
def test_invalid_reservations_preserve_state(tokens, lengths):
    cache = new_cache()
    cache.begin_step(8)
    with pytest.raises(ValueError):
        cache.reserve_layer(0, tokens, lengths)
    cache.commit_step()
    assert cache.layer_state(0).validated_tokens == 0
    assert cache.stats()["resident_bytes"] == 0


def test_workspace_is_one_flat_allocation_shared_between_layers_and_dtypes():
    cache = new_cache()
    floating = cache.workspace(33, torch.float32)
    integer = cache.workspace(64, torch.int32)
    assert floating.data_ptr() == integer.data_ptr()
    integer.fill_(0)
    assert torch.all(floating == 0)
    cache.begin_step(8)
    materialize(cache, 0, 8, 3, 1, 2)
    materialize(cache, 1, 8, 3, 1, 3)
    cache.commit_step()
    assert cache.workspace(16, torch.float16).data_ptr() == floating.data_ptr()
    assert cache.stats()["workspace_bytes"] == 256
    enlarged = cache.workspace(100, torch.float32)
    assert enlarged.numel() == 100
    assert cache.stats()["workspace_bytes"] == 512
    pointer = enlarged.data_ptr()
    cache.reset()
    assert cache.length == 0
    assert cache.workspace(16, torch.float32).data_ptr() == pointer
    assert cache.layer_state(0).validated_tokens == 0


def test_release_drops_owned_buffers_and_rejects_further_use():
    cache = new_cache()
    cache.begin_step(8)
    reservation = cache.reserve_layer(0, 8, {"compressed": 3, "pooled": 1})
    ref = weakref.ref(reservation.buffers["compressed"])
    del reservation
    cache.finish_layer(0)
    cache.commit_step()
    cache.workspace(16, torch.float32)
    cache.release()
    cache.release()
    assert ref() is None
    assert cache.stats()["resident_bytes"] == cache.stats()["workspace_bytes"] == 0
    for action in (
        lambda: cache.layer_state(0),
        lambda: cache.workspace(1, torch.float32),
        lambda: cache.begin_step(1),
        cache.reset,
    ):
        with pytest.raises(RuntimeError, match="released"):
            action()


def test_zero_capacity_records_are_valid_for_short_contexts():
    cache = IndexerCache(1, 1, {"empty": IndexerBufferSpec(0, (2,), torch.bfloat16)}, device="cpu")
    cache.begin_step(1)
    reservation = cache.reserve_layer(0, 1, {"empty": 0})
    assert reservation.buffers["empty"].shape == (0, 2)
    cache.finish_layer(0)
    cache.commit_step()
    assert cache.stats()["resident_bytes"] == 0
