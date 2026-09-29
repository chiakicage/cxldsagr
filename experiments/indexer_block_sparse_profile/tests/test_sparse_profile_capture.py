"""Workload boundaries required for equivalent full and prefix-ready runs."""

import pytest
import torch

from experiments.indexer_block_sparse_profile.src.capture import (
    operator_workload,
    rewind_cache,
    validate_workload,
)
from experiments.indexer_block_sparse_profile.src.mfu import validate_kernel_backend
from models.nosa.cache import NosaKVCache
from models.nosa.config import NosaConfig

FIELDS = (
    "prefix_tokens",
    "new_tokens",
    "chunk_size",
    "warmup",
    "repeats",
    "profile_repeats",
)
DEFAULTS = dict(zip(FIELDS, (65536, 1024, 1024, 2, 5, 1), strict=True))


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_capture_declares_the_current_score_selection_and_preparation_dispatch(backend):
    workload = operator_workload(backend)
    assert workload["attention_execution"] == (
        "native_fa3_v3" if backend == "native" else "triton_v1"
    )
    assert validate_kernel_backend(workload) == (
        "cuda_tvm_ffi" if backend == "native" else "triton"
    )
    assert workload["indexer_execution"] == (
        "cached_native_v5" if backend == "native" else "cached_flashinfer_v1"
    )
    assert workload["indexer_preparation"] == (
        "native_guarded_ranked_checked_v1" if backend == "native" else "triton_v1"
    )


def test_capture_rejects_unknown_operator_backend():
    with pytest.raises(ValueError):
        operator_workload("unknown")


@pytest.mark.parametrize(
    ("prefix_tokens", "new_tokens", "chunk_size"),
    [(65536, 1024, 1024), (8192, 1024, 2048), (1024, 1, 512)],
)
def test_workload_keeps_candidate_boundary_between_forward_calls(
    prefix_tokens, new_tokens, chunk_size
):
    validate_workload(
        **(
            DEFAULTS
            | {
                "prefix_tokens": prefix_tokens,
                "new_tokens": new_tokens,
                "chunk_size": chunk_size,
            }
        )
    )


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_workload_lengths_and_repetition_counts_are_positive_integers(field, value):
    with pytest.raises(ValueError):
        validate_workload(**(DEFAULTS | {field: value}))


def test_rejects_chunk_crossing_the_gr_candidate_boundary():
    # Full execution would combine the last prefix token and candidates in a
    # shared call, while prefix-ready execution would begin at a fresh call.
    with pytest.raises(ValueError):
        validate_workload(**(DEFAULTS | {"prefix_tokens": 65535}))


def test_rejects_splitting_candidate_extend_across_multiple_calls():
    with pytest.raises(ValueError):
        validate_workload(**(DEFAULTS | {"chunk_size": 512}))


@pytest.fixture
def sparse_cache():
    config = NosaConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=16,
    )
    cache = NosaKVCache(config, 16, device="cpu", dtype=torch.float32, with_cis=True)
    yield cache
    cache.release()


def append_records(cache, count, value):
    cache.begin_step(count)
    for layer in range(cache.config.num_hidden_layers):
        cache.write_layer(
            layer,
            **{
                name: torch.full((count, *shape), value + layer, dtype=cache.dtype)
                for name, shape in cache.spec.record_shapes.items()
            },
        )
    cache.commit_step()


def test_extend_rewind_preserves_prefix_kv_and_cis_and_hides_old_suffix(sparse_cache):
    append_records(sparse_cache, 4, 3.0)
    prefix = {name: tensor[:, :4].clone() for name, tensor in sparse_cache.buffers.items()}
    append_records(sparse_cache, 3, 7.0)
    rewind_cache(sparse_cache, 4)
    assert sparse_cache.length == 4
    for layer in range(sparse_cache.config.num_hidden_layers):
        for name, view in sparse_cache.layer_view(layer).items():
            torch.testing.assert_close(view, prefix[name][layer], atol=0, rtol=0)
    append_records(sparse_cache, 2, 11.0)
    assert sparse_cache.length == 6
    for layer in range(sparse_cache.config.num_hidden_layers):
        for name, view in sparse_cache.layer_view(layer).items():
            assert len(view) == 6
            torch.testing.assert_close(view[:4], prefix[name][layer], atol=0, rtol=0)
            torch.testing.assert_close(view[4:], torch.full_like(view[4:], 11.0 + layer))


def test_extend_rewind_rejects_unbuilt_prefix_without_mutating_length(sparse_cache):
    append_records(sparse_cache, 3, 1.0)
    with pytest.raises(ValueError, match="not been built"):
        rewind_cache(sparse_cache, 4)
    assert sparse_cache.length == 3


def test_extend_rewind_rejects_opaque_state_in_any_layer(sparse_cache):
    append_records(sparse_cache, 7, 1.0)
    state = object()
    sparse_cache.set_layer_state(1, state)
    with pytest.raises(ValueError, match="opaque"):
        rewind_cache(sparse_cache, 4)
    assert sparse_cache.length == 7
    assert sparse_cache.get_layer_state(1) is state


def test_full_prefill_rewind_clears_committed_cursor_and_opaque_state(sparse_cache):
    append_records(sparse_cache, 7, 1.0)
    sparse_cache.set_layer_state(1, object())
    rewind_cache(sparse_cache, 0)
    assert sparse_cache.length == 0
    for layer in range(sparse_cache.config.num_hidden_layers):
        assert sparse_cache.get_layer_state(layer) is None
        assert all(len(view) == 0 for view in sparse_cache.layer_view(layer).values())
    append_records(sparse_cache, 2, 9.0)
    assert sparse_cache.length == 2


def test_extend_rewind_does_not_interrupt_pending_transaction(sparse_cache):
    append_records(sparse_cache, 4, 1.0)
    sparse_cache.begin_step(2)
    with pytest.raises(RuntimeError, match="pending"):
        rewind_cache(sparse_cache, 4)
    assert sparse_cache.length == 4
    sparse_cache.abort_step()
    append_records(sparse_cache, 1, 2.0)
    assert sparse_cache.length == 5
