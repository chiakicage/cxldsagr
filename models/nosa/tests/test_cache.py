"""NOSA KV and derived indexer records share one request transaction."""

import pytest
import torch

from models.nosa.cache.resident import NosaKVCache
from models.nosa.config import NosaConfig


def new_cache(*, with_cis=True):
    config = NosaConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=160,
    )
    return NosaKVCache(config, 160, device="cpu", dtype=torch.float32, with_cis=with_cis)


def write_layers(cache, count, *, layers=(0, 1), value=3):
    for layer in layers:
        cache.write_layer(
            layer,
            **{
                name: torch.full((count, *shape), value, dtype=cache.dtype)
                for name, shape in cache.spec.record_shapes.items()
            },
        )


def materialize(cache, layer, length, value=7):
    compressed, pooled = max(0, length // 16 - 1), max(0, (length - 16) // 64)
    reservation = cache.indexer_cache.reserve_layer(
        layer,
        length,
        {"compressed_keys": compressed, "compressed_cis": compressed, "pooled_cis": pooled},
    )
    for name, buffer in reservation.buffers.items():
        buffer[reservation.previous.lengths[name] : reservation.target.lengths[name]].fill_(value)
    cache.indexer_cache.finish_layer(layer)


def test_dense_trajectory_can_commit_before_lazy_sidecar_materialization():
    cache = new_cache()
    cache.begin_step(80)
    write_layers(cache, 80)
    cache.commit_step()
    assert cache.length == cache.indexer_cache.length == 80
    assert cache.indexer_cache.stats()["resident_bytes"] == 0
    materialize(cache, 0, 80)
    assert cache.indexer_cache.layer_state(0).validated_tokens == 80
    assert cache.indexer_cache.layer_state(1).validated_tokens == 0
    assert all(cache.get_layer_state(layer) is None for layer in range(2))
    assert new_cache(with_cis=False).indexer_cache is None


def test_both_sides_validate_before_either_commit_and_abort_preserves_prefix():
    cache = new_cache()
    cache.begin_step(80)
    write_layers(cache, 80)
    for layer in range(2):
        materialize(cache, layer, 80)
    cache.commit_step()
    state = cache.indexer_cache.layer_state(0)
    prefix = {name: view.clone() for name, view in cache.indexer_cache.layer_view(0).items()}
    cache.begin_step(16)
    write_layers(cache, 16, layers=(0,))
    materialize(cache, 0, 96, value=99)
    with pytest.raises(RuntimeError, match="Every layer"):
        cache.commit_step()
    assert cache.length == cache.indexer_cache.length == 80
    write_layers(cache, 16, layers=(1,))
    cache.indexer_cache.reserve_layer(
        1, 96, {"compressed_keys": 5, "compressed_cis": 5, "pooled_cis": 1}
    )
    with pytest.raises(RuntimeError, match="reservation"):
        cache.commit_step()
    assert cache.length == cache.indexer_cache.length == 80
    cache.abort_step()
    assert cache.indexer_cache.layer_state(0) == state
    for name, view in cache.indexer_cache.layer_view(0).items():
        torch.testing.assert_close(view, prefix[name])
    cache.begin_step(16)
    write_layers(cache, 16)
    for layer in range(2):
        materialize(cache, layer, 96, value=11)
    cache.commit_step()
    assert cache.length == cache.indexer_cache.length == 96


def test_truncate_discards_stable_pool_that_now_depends_on_a_removed_token():
    cache = new_cache()
    cache.begin_step(80)
    write_layers(cache, 80)
    materialize(cache, 0, 80)
    cache.commit_step()
    cache.length = 79
    state = cache.indexer_cache.layer_state(0)
    assert cache.length == cache.indexer_cache.length == state.validated_tokens == 79
    assert state.lengths == {"compressed_keys": 3, "compressed_cis": 3, "pooled_cis": 0}
    cache.begin_step(1)
    write_layers(cache, 1)
    materialize(cache, 0, 80, value=19)
    cache.commit_step()
    assert torch.all(cache.indexer_cache.layer_view(0)["pooled_cis"] == 19)


@pytest.mark.parametrize("length", [-1, 81, 161, True, 79.0, None])
def test_invalid_or_forward_legacy_assignment_does_not_change_either_cache(length):
    cache = new_cache()
    cache.begin_step(80)
    write_layers(cache, 80)
    materialize(cache, 0, 80)
    cache.commit_step()
    state = cache.indexer_cache.layer_state(0)
    with pytest.raises(ValueError, match="cache length"):
        cache.length = length
    assert cache.length == cache.indexer_cache.length == 80
    assert cache.indexer_cache.layer_state(0) == state


def test_pending_and_opaque_state_prevent_unsafe_rewind_but_reset_clears_both():
    cache = new_cache()
    cache.begin_step(80)
    write_layers(cache, 80)
    materialize(cache, 0, 80)
    cache.commit_step()
    cache.set_layer_state(1, object())
    with pytest.raises(ValueError, match="opaque"):
        cache.truncate(79)
    assert cache.length == cache.indexer_cache.length == 80
    cache.begin_step(1)
    with pytest.raises(RuntimeError, match="pending"):
        cache.truncate(0)
    cache.reset()
    assert cache.length == cache.indexer_cache.length == 0
    assert cache.get_layer_state(1) is None
    assert cache.indexer_cache.layer_state(0).validated_tokens == 0
    cache.release()
    cache.release()
    assert cache.indexer_cache.released
    assert cache.stats()["resident_bytes"] == 0
    with pytest.raises(RuntimeError, match="released"):
        cache.length = 0


def test_stats_include_derived_allocations_and_shared_workspace():
    cache = new_cache()
    kv_bytes = sum(t.numel() * t.element_size() for t in cache.buffers.values())
    cache.begin_step(80)
    write_layers(cache, 80)
    materialize(cache, 0, 80)
    cache.commit_step()
    cache.indexer_cache.workspace(100, torch.float32)
    assert (
        cache.stats()["resident_bytes"] == kv_bytes + cache.indexer_cache.stats()["resident_bytes"]
    )
