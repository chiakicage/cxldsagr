from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v32.serving_backend import (
    DeepSeekServingBackend,
    DeepSeekServingSession,
    _storage_bytes,
    replay_parameter_count,
)


class _Cache:
    def __init__(self):
        self.length = self.written = 0
        self._step_end = None
        self.records = []
        self.rollbacks = 0

    def reset_stats(self):
        pass

    def begin_step(self, count):
        if self._step_end is not None:
            raise RuntimeError("step already active")
        self._step_end = self.length + count

    def append(self, records):
        self.records.extend(row.clone() for row in records)
        self.written += len(records)

    def commit(self):
        assert self.written == self._step_end
        self.length = self.written
        self._step_end = None

    def rollback(self):
        self._step_end = None
        self.truncate(self.length)
        self.rollbacks += 1

    def truncate(self, length):
        self.records = self.records[:length]
        self.length = self.written = length


class _Block:
    def __init__(self, source):
        self.source = source
        self.attention = self.cache = None
        self.inputs = []
        self.fail = False

    def forward(self, hidden, residual, scope=None):
        if self.fail:
            raise RuntimeError("injected layer failure")
        self.inputs.append((hidden, residual))
        combined = hidden.float() if residual is None else hidden.float() + residual.float()
        cache = self.cache
        past = torch.stack(cache.records) if cache.records else combined[:0]
        cumulative = torch.cat((past, combined)).cumsum(0)[cache.written :]
        denominator = torch.arange(cache.written + 1, cache.written + len(hidden) + 1)[:, None]
        cache.append(combined)
        hidden_out = (combined * 0.25 + cumulative / denominator * 0.5 + self.source).bfloat16()
        return hidden_out, (combined * 0.125).bfloat16()


def _model(layers=10, scheme="hbm"):
    model = object.__new__(DeepSeekServingBackend)
    model.device = torch.device("cpu")
    model.scheme = scheme
    model.num_layers = layers
    model.chunk_size = 4
    model.slots = 16
    model.max_seq_len = 128
    model.cfg = SimpleNamespace(
        vocab_size=23,
        norm_eps=1e-6,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
    )
    generator = torch.Generator().manual_seed(82)
    model.embedding_weight = torch.randn(23, 8, generator=generator).bfloat16()
    model.head_weight = torch.randn(23, 8, generator=generator).bfloat16()
    model.final_norm = torch.ones(8)
    model.blocks = [_Block(layer % 3) for layer in range(layers)]
    model.capture_hook = None
    model._busy = False
    model.synchronize = lambda: None
    session = DeepSeekServingSession(
        128, scheme, [SimpleNamespace(cache=_Cache()) for _ in range(layers)], model
    )
    return model, session


def test_source_inputs_are_replayed_with_independent_cache_and_storage():
    model, session = _model()
    outputs = []
    model.capture_hook = lambda layer, hidden, residual: outputs.append((hidden, residual))
    for ids in ([1, 4, 2, 8], [6, 3, 5]):
        outputs.clear()
        result = model.extend(session, ids)
        assert result.shape == (len(ids), 8)
        for layer in range(3, 10):
            torch.testing.assert_close(outputs[layer], outputs[layer % 3], rtol=0, atol=0)
            copied_input = model.blocks[layer].inputs[-1]
            source_input = model.blocks[layer % 3].inputs[-1]
            for copied, source in zip(copied_input, source_input, strict=True):
                if source is not None:
                    assert source.data_ptr() != copied.data_ptr()
                    torch.testing.assert_close(copied, source, rtol=0, atol=0)
            assert session.runners[layer].cache is not session.runners[layer % 3].cache
        assert all(block.attention is None and block.cache is None for block in model.blocks)
    assert session.length == 7


def test_truncate_candidate_retains_prefix_and_revisit_matches_fresh_session():
    model, session = _model()
    oracle, independent = _model()
    prefix, candidate = [1, 3, 8], [2, 6]
    model.prefill(session, prefix)
    oracle.prefill(independent, prefix)
    model.extend(session, [7, 9, 10])
    model.truncate(session, len(prefix))
    actual = model.extend(session, candidate)
    expected = oracle.extend(independent, candidate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.last_logits, oracle.last_logits, rtol=0, atol=0)


def test_failed_copy_rolls_back_all_owned_transactions_and_allows_retry():
    model, session = _model()
    model.prefill(session, [1, 4, 3])
    model.blocks[7].fail = True
    with pytest.raises(RuntimeError, match="injected"):
        model.extend(session, [5, 2])
    assert not model._busy and session.length == 3
    assert all(runner.cache.length == runner.cache.written == 3 for runner in session.runners)
    assert all(runner.cache.rollbacks == 1 for runner in session.runners)
    model.blocks[7].fail = False
    assert model.extend(session, [5, 2]).shape == (2, 8)


def test_alternating_users_keep_independent_prefixes_on_shared_model_weights():
    model, alice = _model()
    bob = DeepSeekServingSession(
        128, "hbm", [SimpleNamespace(cache=_Cache()) for _ in range(10)], model
    )
    oracle, independent = _model()
    model.prefill(alice, [1, 4, 3])
    oracle.prefill(independent, [1, 4, 3])
    model.prefill(bob, [9, 11, 8, 7])
    model.extend(bob, [12, 13])
    model.truncate(bob, 4)
    actual = model.extend(alice, [5, 2])
    expected = oracle.extend(independent, [5, 2])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert alice.length == 5 and bob.length == 4
    assert all(runner.cache.length == 4 for runner in bob.runners)


def test_failed_begin_does_not_rollback_an_existing_transaction():
    model, session = _model()
    session.runners[2].cache.begin_step(1)
    with pytest.raises(RuntimeError, match="already active"):
        model.prefill(session, [1, 2])
    assert [runner.cache.rollbacks for runner in session.runners] == [1, 1] + [0] * 8
    assert session.runners[2].cache._step_end == 1


def test_estimates_include_indexer_metadata_pool_counters_and_double_stage():
    capacity = 32
    model, _ = _model(scheme="hbm")
    assert model.estimate_session_bytes(capacity) == {"hbm": 10 * (1304 * 32 + 64), "dram": 0}
    model.scheme = "serial_sparse"
    assert model.estimate_session_bytes(capacity) == {
        "hbm": 10 * (136 * 32 + 1168 * 16 + 64),
        "dram": 10 * 32 * 1152,
    }
    model.scheme = "echo"
    assert model.estimate_session_bytes(capacity)["hbm"] == 10 * (136 * 32 + 1168 * 16 + 64 + 8 * 4)
    model.scheme = "dense_prefetch"
    assert model.estimate_session_bytes(capacity) == {
        "hbm": 10 * (152 * 32 + 64) + 2 * 32 * 1152,
        "dram": 10 * 32 * 1152,
    }


def test_storage_accounting_counts_shared_buffers_once_and_entire_storage():
    storage = torch.empty(128, dtype=torch.bfloat16)
    other = torch.empty(16, dtype=torch.float32)
    assert _storage_bytes([storage[:2], storage[32:], other, None]) == {"hbm": 0, "dram": 320}


def test_parameter_count_uses_physical_copies_and_excludes_scale_metadata():
    metadata = {
        "model.embed_tokens.weight": {"shape": [11, 7]},
        "model.norm.weight": {"shape": [7]},
        "lm_head.weight": {"shape": [11, 7]},
    }
    for layer in range(3):
        metadata[f"model.layers.{layer}.mlp.up.weight"] = {"shape": [layer + 1, 8]}
        metadata[f"model.layers.{layer}.mlp.up.weight_scale_inv"] = {"shape": [1, 1]}
    result = replay_parameter_count(SimpleNamespace(tensor_metadata=metadata), 10)
    assert result == {
        "source_layer_parameters": [8, 16, 24],
        "backbone_parameters": 152,
        "endpoint_parameters": 161,
        "total_parameters": 313,
    }


def test_released_or_foreign_sessions_fail_before_execution():
    model, session = _model()
    other, _ = _model()
    with pytest.raises(ValueError, match="another backend"):
        other.extend(session, [1])
    model.release_session(session)
    with pytest.raises(ValueError, match="released"):
        model.extend(session, [1])
