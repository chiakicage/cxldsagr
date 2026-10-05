"""Three-arm pattern recording preserves the dense/sparse inference trajectories."""

from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import pytest
import torch

from experiments.nosa_indexer_pattern_65536_1024.src.sparse_capture import (
    PatternRecorder,
    capture_extend,
    run_trajectory,
)
from models.attention_contracts import AttentionContext
from models.nosa.attention import DenseMainAttention, ResidentLayerView
from models.nosa.indexer import NosaIndexer
from models.nosa.tests.test_model import dense_attention, tiny_config
from models.nosa.tests.test_sparse_model import initialized_sparse_model

PREFIX = 65
QUERIES = 64
DENSE_ARMS = {"dense_qa64", "dense_nosa64"}


@pytest.fixture
def model():
    return initialized_sparse_model(
        tiny_config(hidden_size=16, intermediate_size=24, head_dim=4, max_position_embeddings=160)
    )


def tokens_for(model):
    return (torch.arange(PREFIX + QUERIES) * 7 + 3) % model.config.vocab_size


@contextmanager
def dense_trajectory(model, attention=dense_attention):
    with (
        patch.object(model, "indexer", None),
        patch.object(model, "main_attention", DenseMainAttention(attention)),
    ):
        yield


def prepare_prefix(model, tokens):
    cache = model.new_cache(len(tokens))
    model(tokens[:PREFIX], cache, return_hidden=True)
    assert cache.length == PREFIX
    return cache


def assert_recorded_padding(recorder, arms, model):
    assert set(recorder.block_ids) == set(recorder.valid_masks) == arms
    expected_counts = ((torch.arange(PREFIX, PREFIX + QUERIES) // 64) + 1)[:, None]
    expected_counts = expected_counts.expand(-1, model.config.num_key_value_heads)
    for arm in arms:
        assert len(recorder.block_ids[arm]) == len(recorder.valid_masks[arm]) == 2
        for ids, valid in zip(recorder.block_ids[arm], recorder.valid_masks[arm], strict=True):
            assert ids.device.type == valid.device.type == "cpu"
            assert ids.shape == valid.shape == (QUERIES, model.config.num_key_value_heads, 64)
            assert valid.dtype == torch.bool
            torch.testing.assert_close(valid.sum(-1), expected_counts)
            assert not valid.all()
            assert (ids[valid] >= 0).all()
    assert len(recorder.attention_shapes) == model.config.num_hidden_layers


def direct_inputs(model):
    generator = torch.Generator().manual_seed(655)
    config = model.config
    q = torch.randn(QUERIES, config.num_attention_heads, config.head_dim, generator=generator)
    keys = torch.randn(
        PREFIX + QUERIES, config.num_key_value_heads, config.head_dim, generator=generator
    )
    values = torch.randn(keys.shape, generator=generator)
    cis = torch.randn(PREFIX + QUERIES, config.num_key_value_heads, generator=generator)
    cache = ResidentLayerView(0, keys=keys, values=values, cis_scores=cis)
    context = AttentionContext(0, PREFIX, QUERIES)
    return q, cache, context


def test_actual_recorder_forwards_selection_identity_and_saves_independent_cpu_copy(
    model, monkeypatch
):
    import experiments.nosa_indexer_pattern_65536_1024.src.sparse_capture as capture_module

    q, cache, context = direct_inputs(model)
    selection = NosaIndexer(mode="nosa")(q, cache, context)
    expected_ids, expected_valid = selection.block_ids.clone(), selection.valid_mask.clone()
    output = torch.randn_like(q)
    forwarded = []

    def attention(query, chosen, access, execution):
        assert query is q and chosen is selection and access is cache and execution is context
        forwarded.append(chosen)
        return output

    def forbidden_indexer(*args, **kwargs):
        raise AssertionError("Actual selection recording must never create/recompute an indexer")

    monkeypatch.setattr(capture_module, "NosaIndexer", forbidden_indexer)
    recorder = PatternRecorder(attention, model.config, PREFIX, QUERIES, selection_mode="actual")
    assert recorder(q, selection, cache, context) is output
    assert forwarded == [selection]
    assert set(recorder.block_ids) == {"sparse_nosa64"}
    saved_ids, saved_valid = (
        recorder.block_ids["sparse_nosa64"][0],
        recorder.valid_masks["sparse_nosa64"][0],
    )
    torch.testing.assert_close(saved_ids.long(), expected_ids)
    torch.testing.assert_close(saved_valid, expected_valid)
    selection.block_ids.fill_(99)
    selection.valid_mask.zero_()
    torch.testing.assert_close(saved_ids.long(), expected_ids)
    torch.testing.assert_close(saved_valid, expected_valid)


def test_observer_arms_receive_identical_inputs_and_forward_dense_selection_none(model):
    q, cache, context = direct_inputs(model)
    observations = []
    output = torch.randn_like(q)

    def indexer(arm, backend):
        def observe(query, access, execution):
            observations.append((arm, query, access, execution))
            return backend(query, access, execution)

        return observe

    def attention(query, selected, access, execution):
        assert query is q and selected is None and access is cache and execution is context
        return output

    recorder = PatternRecorder(
        attention,
        model.config,
        PREFIX,
        QUERIES,
        selection_mode="observe",
        indexers={
            "dense_qa64": indexer("dense_qa64", NosaIndexer()),
            "dense_nosa64": indexer("dense_nosa64", NosaIndexer(mode="nosa")),
        },
    )
    assert recorder(q, None, cache, context) is output
    assert {observation[0] for observation in observations} == DENSE_ARMS
    assert all(
        query is q and access is cache and execution is context
        for _, query, access, execution in observations
    )
    assert set(recorder.block_ids) == DENSE_ARMS


@pytest.mark.parametrize("selection_mode", ["observe", "actual"])
def test_recorder_rejects_selection_from_the_wrong_execution_mode(model, selection_mode):
    q, cache, context = direct_inputs(model)
    selection = NosaIndexer()(q, cache, context) if selection_mode == "observe" else None

    def forbidden_attention(*args):
        raise AssertionError("Invalid mode must fail before reaching attention")

    recorder = PatternRecorder(
        forbidden_attention,
        model.config,
        PREFIX,
        QUERIES,
        selection_mode=selection_mode,
        indexers={"dense_qa64": NosaIndexer()} if selection_mode == "observe" else None,
    )
    with pytest.raises(ValueError):
        recorder(q, selection, cache, context)


@torch.inference_mode()
def test_dense_observation_does_not_change_hidden_and_prefix_trajectories_are_independent(model):
    tokens = tokens_for(model)
    sparse_cache = prepare_prefix(model, tokens)
    original_indexer, original_attention = model.indexer, model.main_attention
    with dense_trajectory(model):
        dense_cache = prepare_prefix(model, tokens)
        expected_cache = prepare_prefix(model, tokens)
        # Dense and sparse use the same weights/cache layout, but their later-layer
        # prefix KV differs. Sharing either prefix would invalidate this comparison.
        assert dense_cache is not sparse_cache
        torch.testing.assert_close(dense_cache.keys[0, :PREFIX], sparse_cache.keys[0, :PREFIX])
        assert not torch.equal(dense_cache.keys[1, :PREFIX], sparse_cache.keys[1, :PREFIX])
        expected = model(tokens[PREFIX:], expected_cache, return_hidden=True)
        attention = model.main_attention
        recorder, actual = capture_extend(
            model, tokens[PREFIX:], dense_cache, PREFIX, mode="dense", backend="reference"
        )
        assert model.indexer is None and model.main_attention is attention
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert_recorded_padding(recorder, DENSE_ARMS, model)
        for name in dense_cache.buffers:
            torch.testing.assert_close(
                dense_cache.buffers[name], expected_cache.buffers[name], rtol=0, atol=0
            )
    assert model.indexer is original_indexer and model.main_attention is original_attention
    assert dense_cache.length == len(tokens) and sparse_cache.length == PREFIX


@torch.inference_mode()
def test_sparse_capture_records_each_actual_indexer_result_once_and_restores_callables(model):
    tokens = tokens_for(model)
    cache, expected_cache = prepare_prefix(model, tokens), prepare_prefix(model, tokens)
    expected = model(tokens[PREFIX:], expected_cache, return_hidden=True)
    indexer, attention = model.indexer, model.main_attention
    selections, consumed = [], []

    def count_indexer(q, access, context):
        selected = indexer(q, access, context)
        selections.append(selected)
        return selected

    def record_attention(q, selected, access, context):
        consumed.append(selected)
        return attention(q, selected, access, context)

    with (
        patch.object(model, "indexer", count_indexer),
        patch.object(model, "main_attention", record_attention),
    ):
        recorder, actual = capture_extend(
            model, tokens[PREFIX:], cache, PREFIX, mode="sparse", backend="reference"
        )
        assert model.indexer is count_indexer and model.main_attention is record_attention
    assert len(selections) == len(consumed) == model.config.num_hidden_layers
    assert all(
        selected is forwarded for selected, forwarded in zip(selections, consumed, strict=True)
    )
    for selected, saved in zip(selections, recorder.block_ids["sparse_nosa64"], strict=True):
        torch.testing.assert_close(selected.block_ids.cpu().long(), saved.long())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_recorded_padding(recorder, {"sparse_nosa64"}, model)
    assert model.indexer is indexer and model.main_attention is attention
    assert cache.length == len(tokens)


@torch.inference_mode()
@pytest.mark.parametrize("validate_observer", [False, True])
def test_run_trajectory_builds_and_releases_a_fresh_prefix_for_each_mode(model, validate_observer):
    tokens = tokens_for(model)
    original_indexer, original_attention = model.indexer, model.main_attention
    original_allocate = model.new_cache
    caches = []

    def track_cache(capacity):
        cache = original_allocate(capacity)
        assert cache.length == 0
        caches.append(cache)
        return cache

    with (
        patch.object(model, "new_cache", track_cache),
        patch.object(model, "attention", dense_attention),
    ):
        dense_recorder, dense_hidden = run_trajectory(
            model,
            tokens[:PREFIX],
            tokens[PREFIX:],
            mode="dense",
            backend="reference",
            validate_observer=validate_observer,
        )
        assert caches[0].released
        assert model.indexer is original_indexer and model.main_attention is original_attention
        sparse_recorder, sparse_hidden = run_trajectory(
            model,
            tokens[:PREFIX],
            tokens[PREFIX:],
            mode="sparse",
            backend="reference",
            validate_observer=validate_observer,
        )
        assert model.indexer is original_indexer and model.main_attention is original_attention
    assert len(caches) == 2 and caches[0] is not caches[1]
    assert all(cache.released for cache in caches)
    assert not torch.equal(torch.from_numpy(dense_hidden), torch.from_numpy(sparse_hidden))
    assert_recorded_padding(dense_recorder, DENSE_ARMS, model)
    assert_recorded_padding(sparse_recorder, {"sparse_nosa64"}, model)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@torch.inference_mode()
def test_capture_failure_restores_attention_and_preserves_committed_prefix(model, mode):
    tokens = tokens_for(model)
    trajectory = dense_trajectory(model) if mode == "dense" else nullcontext()
    with trajectory:
        cache = prepare_prefix(model, tokens)
        previous = {name: buffer[:, :PREFIX].clone() for name, buffer in cache.buffers.items()}
        states = tuple(
            {"committed": PREFIX, "layer": index} for index in range(model.config.num_hidden_layers)
        )
        for index, state in enumerate(states):
            cache.set_layer_state(index, state)
        original_attention, original_indexer = model.main_attention, model.indexer
        calls = 0

        def fail_late(q, selection, access, context):
            nonlocal calls
            calls += 1
            access.set_layer_state(context.layer_idx, {"pending": len(tokens)})
            if calls == model.config.num_hidden_layers:
                raise RuntimeError("injected candidate failure")
            return original_attention(q, selection, access, context)

        with patch.object(model, "main_attention", fail_late):
            with pytest.raises(RuntimeError, match="injected candidate failure"):
                capture_extend(
                    model, tokens[PREFIX:], cache, PREFIX, mode=mode, backend="reference"
                )
            assert model.main_attention is fail_late and model.indexer is original_indexer
        assert model.main_attention is original_attention
        assert cache.length == PREFIX
        assert all(cache.get_layer_state(index) is state for index, state in enumerate(states))
        for name, snapshot in previous.items():
            torch.testing.assert_close(cache.buffers[name][:, :PREFIX], snapshot, rtol=0, atol=0)
        # A retry proves there is no dangling pending step after the capture error.
        capture_extend(model, tokens[PREFIX:], cache, PREFIX, mode=mode, backend="reference")
        assert cache.length == len(tokens)
