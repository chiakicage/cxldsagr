"""Candidate-only indexer capture preserves dense NOSA execution and cache state."""

import pytest
import torch

from executor.model_executor import run_chunks
from experiments.nosa_indexer_pattern_65536_1024.src.capture import (
    CaptureAttention,
    capture_extend,
)
from models.nosa.indexer import NosaIndexer
from models.nosa.tests.test_model import initialized_model, tiny_config
from operators.flashinfer import FlashInferFullAttention


def assert_capture_matches_dense(model, *, prefix_tokens, new_tokens, block_budget):
    """Compare identical chunk boundaries using two independent request caches."""
    config = model.config
    total = prefix_tokens + new_tokens
    device = model.model.embed_tokens.weight.device
    ids = torch.arange(total, device=device, dtype=torch.long) % config.vocab_size
    dense_cache = model.new_cache(total)
    capture_cache = model.new_cache(total)
    original_attention = model.main_attention
    try:
        for cache in (dense_cache, capture_cache):
            run_chunks(model, ids[:prefix_tokens], cache, 64)
            assert cache.length == prefix_tokens
            assert model.main_attention is original_attention
        prefix_keys = capture_cache.keys[:, :prefix_tokens].clone()
        prefix_values = capture_cache.values[:, :prefix_tokens].clone()

        expected = run_chunks(model, ids[prefix_tokens:], dense_cache, new_tokens)
        record, actual = capture_extend(
            model,
            ids[prefix_tokens:],
            capture_cache,
            prefix_tokens,
            block_budget=block_budget,
        )

        assert isinstance(record, CaptureAttention)
        assert isinstance(record.indexer, NosaIndexer)
        assert record.indexer.policy.block_budget == block_budget
        assert model.main_attention is original_attention
        assert model.indexer is None
        assert capture_cache.length == dense_cache.length == total
        assert actual.shape == (new_tokens, config.hidden_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(capture_cache.keys, dense_cache.keys, rtol=0, atol=0)
        torch.testing.assert_close(capture_cache.values, dense_cache.values, rtol=0, atol=0)
        torch.testing.assert_close(
            capture_cache.keys[:, :prefix_tokens], prefix_keys, rtol=0, atol=0
        )
        torch.testing.assert_close(
            capture_cache.values[:, :prefix_tokens], prefix_values, rtol=0, atol=0
        )
        assert all(
            capture_cache.get_layer_state(layer) is None
            for layer in range(config.num_hidden_layers)
        )

        # One candidate call per layer, even when the prefix required several chunks.
        assert len(record.attention_shapes) == config.num_hidden_layers
        assert len(record.block_ids) == len(record.valid_masks) == config.num_hidden_layers
        positions = torch.arange(prefix_tokens, total)
        visible_counts = (positions // 64 + 1)[:, None].expand(-1, config.num_key_value_heads)
        for layer, shapes in enumerate(record.attention_shapes):
            assert shapes["layer"] == layer
            assert shapes["query_start"] == prefix_tokens
            assert shapes["q"] == [new_tokens, config.num_attention_heads, config.head_dim]
            assert (
                shapes["k"]
                == shapes["v"]
                == [
                    total,
                    config.num_key_value_heads,
                    config.head_dim,
                ]
            )
            blocks, valid = record.block_ids[layer], record.valid_masks[layer]
            assert blocks.device.type == valid.device.type == "cpu"
            assert (
                blocks.shape
                == valid.shape
                == (
                    new_tokens,
                    config.num_key_value_heads,
                    block_budget,
                )
            )
            assert valid.dtype == torch.bool
            torch.testing.assert_close(valid.sum(-1), visible_counts, rtol=0, atol=0)
            # These short requests fit entirely inside sink/local and expose
            # absolute positioning at the 64-token boundary without top-k ties.
            for query, position in enumerate(positions.tolist()):
                for head in range(config.num_key_value_heads):
                    assert blocks[query, head][valid[query, head]].tolist() == list(
                        range(position // 64 + 1)
                    )
    finally:
        model.cache_manager.release(dense_cache)
        model.cache_manager.release(capture_cache)


@pytest.mark.parametrize("block_budget", [32, 64])
@pytest.mark.parametrize("prefix_tokens,new_tokens", [(11, 7), (127, 65)])
@torch.inference_mode()
def test_candidate_capture_keeps_dense_hidden_and_cache(prefix_tokens, new_tokens, block_budget):
    model = initialized_model(tiny_config(max_position_embeddings=192))
    assert_capture_matches_dense(
        model, prefix_tokens=prefix_tokens, new_tokens=new_tokens, block_budget=block_budget
    )


@pytest.mark.parametrize("failure_stage", ["indexer", "attention"])
@torch.inference_mode()
def test_capture_failure_restores_attention_and_preserves_prefix(monkeypatch, failure_stage):
    model = initialized_model(tiny_config(max_position_embeddings=192))
    prefix_tokens, total = 63, 96
    ids = torch.arange(total, dtype=torch.long) % model.config.vocab_size
    cache = model.new_cache(total)
    reference_cache = model.new_cache(total)
    original_attention = model.main_attention
    indexer = NosaIndexer()
    try:
        for session in (cache, reference_cache):
            run_chunks(model, ids[:prefix_tokens], session, 32)
        expected = run_chunks(model, ids[prefix_tokens:], reference_cache, total - prefix_tokens)
        prefix_keys = cache.keys[:, :prefix_tokens].clone()
        prefix_values = cache.values[:, :prefix_tokens].clone()
        indexed_layers = []

        def failing_indexer(q, cache_access, context):
            selection = indexer(q, cache_access, context)
            indexed_layers.append(context.layer_idx)
            if failure_stage == "indexer" and context.layer_idx == 1:
                raise RuntimeError("injected indexer failure")
            return selection

        def failing_attention(q, selection, cache_access, context):
            if context.layer_idx == 1:
                raise RuntimeError("injected attention failure")
            return original_attention(q, selection, cache_access, context)

        with monkeypatch.context() as patch:
            if failure_stage == "attention":
                patch.setattr(model, "main_attention", failing_attention)
            before_capture = model.main_attention
            with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
                capture_extend(
                    model, ids[prefix_tokens:], cache, prefix_tokens, indexer=failing_indexer
                )
            assert model.main_attention is before_capture
        assert indexed_layers == [0, 1]
        assert model.main_attention is original_attention
        assert model.indexer is None
        assert cache.length == prefix_tokens
        torch.testing.assert_close(cache.keys[:, :prefix_tokens], prefix_keys, rtol=0, atol=0)
        torch.testing.assert_close(cache.values[:, :prefix_tokens], prefix_values, rtol=0, atol=0)
        # Aborted suffix bytes are unspecified; a successful retry must overwrite
        # them and commit every layer, without retaining a pending transaction.
        record, actual = capture_extend(model, ids[prefix_tokens:], cache, prefix_tokens)
        assert len(record.block_ids) == model.config.num_hidden_layers
        assert cache.length == total
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(cache.keys, reference_cache.keys, rtol=0, atol=0)
        torch.testing.assert_close(cache.values, reference_cache.values, rtol=0, atol=0)
    finally:
        model.cache_manager.release(cache)
        model.cache_manager.release(reference_cache)


@torch.inference_mode()
def test_capture_requires_the_committed_prefix_boundary():
    model = initialized_model()
    ids = torch.tensor([1, 3, 7, 5, 2, 8])
    cache = model.new_cache(ids.numel())
    original_attention = model.main_attention
    try:
        run_chunks(model, ids[:3], cache, 3)
        prefix_keys = cache.keys[:, :3].clone()
        prefix_values = cache.values[:, :3].clone()
        with pytest.raises(ValueError, match="prefix|length"):
            capture_extend(model, ids[3:], cache, 2)
        assert cache.length == 3
        assert model.main_attention is original_attention
        assert model.indexer is None
        torch.testing.assert_close(cache.keys[:, :3], prefix_keys, rtol=0, atol=0)
        torch.testing.assert_close(cache.values[:, :3], prefix_values, rtol=0, atol=0)
    finally:
        model.cache_manager.release(cache)


@pytest.mark.parametrize("block_budget", [32, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FlashInfer")
@torch.inference_mode()
def test_flashinfer_candidate_capture_keeps_dense_hidden_and_cache(block_budget):
    pytest.importorskip("flashinfer")
    config = tiny_config(
        hidden_size=256, head_dim=64, intermediate_size=384, max_position_embeddings=192
    )
    model = initialized_model(
        config, device="cuda", dtype=torch.bfloat16, attention=FlashInferFullAttention()
    )
    assert_capture_matches_dense(model, prefix_tokens=127, new_tokens=65, block_budget=block_budget)
