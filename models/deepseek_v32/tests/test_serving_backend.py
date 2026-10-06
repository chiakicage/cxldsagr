from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from cache.lifecycle import ResourceLifecycle
from models.deepseek_v32.cache.session import DeepSeekServingSession
from models.deepseek_v32.execution.adapter import (
    DeepSeekServingBackend,
    _storage_bytes,
)
from models.deepseek_v32.execution.pipeline import LocalPipeline
from models.deepseek_v32.replay import ReplayLayout, replay_parameter_count


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
        self.execution_chunk_sizes = []
        self.fail = False

    def forward(self, hidden, residual, scope=None, *, attention, chunk_size):
        assert self.attention is None and self.cache is None
        if self.fail:
            raise RuntimeError("injected layer failure")
        self.inputs.append((hidden, residual))
        self.execution_chunk_sizes.append(chunk_size)
        combined = hidden.float() if residual is None else hidden.float() + residual.float()
        cache = attention.cache
        past = torch.stack(cache.records) if cache.records else combined[:0]
        cumulative = torch.cat((past, combined)).cumsum(0)[cache.written :]
        denominator = torch.arange(cache.written + 1, cache.written + len(hidden) + 1)[:, None]
        cache.append(combined)
        hidden_out = (combined * 0.25 + cumulative / denominator * 0.5 + self.source).bfloat16()
        return hidden_out, (combined * 0.125).bfloat16()


def _model(layers=10, scheme="hbm", *, owner=None):
    model = object.__new__(DeepSeekServingBackend)
    model.device = torch.device("cpu")
    model.lifecycle = ResourceLifecycle("DeepSeek test")
    if owner is not None:
        model.lifecycle.bind_owner(owner)
    model.pipeline = LocalPipeline()
    model.replay_layout = ReplayLayout.repeated_sources(layers)
    model.scheme = scheme
    model.num_layers = layers
    model.chunk_size = 4
    model.host_arena_tokens = None
    model.workspace_query_tokens = 4
    model.extend_chunk_size = None
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
    model.synchronize = lambda: None
    session = DeepSeekServingSession(
        128, scheme, [SimpleNamespace(cache=_Cache()) for _ in range(layers)], model
    )
    session.registration = model.lifecycle.register(session, owner=owner, allow_unplanned=True)
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


def test_graph_replay_copies_source_once_and_eager_fallback_remains_independent():
    model, session = _model()
    reference, reference_session = _model()

    class CopyingGraphs:
        failed = False
        eager_fallbacks = 0

        @contextmanager
        def execution(self):
            yield

        def supports(self, layer, hidden, residual):
            # Exercise mixed graph/eager source copies, including source zero's
            # absent residual after a layer that did return a residual tensor.
            return layer != 7 and (residual is None) == (layer % 3 == 0)

        def forward_block(self, layer, runner, hidden, residual, *, scope):
            if layer >= 3:
                source = model.blocks[layer % 3].inputs[-1]
                # Base graph input copies are owned independently. Its incoming
                # tensor is recorded separately, just as the real graph bank.
                original = incoming[layer % 3]
                assert hidden is original[0] and residual is original[1]
                assert hidden.data_ptr() != source[0].data_ptr()
            incoming[layer] = (hidden, residual)
            return model.blocks[layer].forward(
                hidden.clone(),
                residual.clone() if residual is not None else None,
                scope=scope,
                attention=runner,
                chunk_size=len(hidden),
            )

    incoming = {}
    model._compute_graphs = CopyingGraphs()
    output = model.extend(session, [1, 4, 2])
    expected = reference.extend(reference_session, [1, 4, 2])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert model._compute_graphs.eager_fallbacks == 1
    for layer in range(3, 10):
        copied = model.blocks[layer].inputs[-1]
        source = model.blocks[layer % 3].inputs[-1]
        for value, original in zip(copied, source, strict=True):
            if value is not None:
                assert value.data_ptr() != original.data_ptr()
                torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert session.length == 3


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
    assert not model._execution_active() and session.length == 3
    assert all(runner.cache.length == runner.cache.written == 3 for runner in session.runners)
    assert all(runner.cache.rollbacks == 1 for runner in session.runners)
    model.blocks[7].fail = False
    assert model.extend(session, [5, 2]).shape == (2, 8)


def test_alternating_users_keep_independent_prefixes_on_shared_model_weights():
    model, alice = _model()
    bob = DeepSeekServingSession(
        128, "hbm", [SimpleNamespace(cache=_Cache()) for _ in range(10)], model
    )
    bob.registration = model.lifecycle.register(bob, allow_unplanned=True)
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


def test_offset_restore_failures_preserve_execution_error_and_poison_owner():
    owner = object()
    model, session = _model(layers=3, owner=owner)
    model.blocks[0].fail = True
    restores = []

    class Offset:
        def __init__(self, layer):
            self.layer = layer

        def clone(self):
            return self

        def copy_(self, saved):
            assert saved is self
            restores.append(self.layer)
            raise OSError(f"offset restore {self.layer}")

    for layer, runner in enumerate(session.runners):
        runner.offset = Offset(layer)
    with pytest.raises(ExceptionGroup) as failed:
        model.prefill(session, [1, 2], owner=owner)
    assert [str(error) for error in failed.value.exceptions] == [
        "injected layer failure",
        "offset restore 0",
        "offset restore 1",
        "offset restore 2",
    ]
    assert restores == [0, 1, 2]
    assert all(runner.cache.rollbacks == 1 for runner in session.runners)
    assert model.lifecycle.poisoned
    assert model.lifecycle.admission_owner is owner
    assert model.lifecycle.active_session is session
    with pytest.raises(RuntimeError, match="poisoned"):
        model.release_session(session, owner=owner)


def test_session_estimate_excludes_shared_pools_and_includes_hint_backups():
    capacity = 32
    model, _ = _model(scheme="hbm")
    model.device = torch.device("cuda:0")  # Allocation-free estimate only.
    assert model.estimate_session_bytes(capacity) == {"hbm": 10 * (1304 * 32 + 192), "dram": 0}
    model.scheme = "serial_sparse"
    expected = {"hbm": 10 * (132 * 32 + 192 + 64) + 4, "dram": 4}
    assert model.estimate_session_bytes(capacity) == expected
    model.scheme = "echo"
    model.slots = 32768
    assert model.estimate_session_bytes(capacity) == expected
    model.scheme = "dense_prefetch"
    assert model.estimate_session_bytes(capacity) == {
        "hbm": 10 * (152 * 32 + 192),
        "dram": 10 * 65536,
    }


@pytest.mark.parametrize("scheme", ["echo", "serial_sparse"])
def test_small_pool_resource_plan_rejects_without_mutating_existing_sessions(scheme):
    from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint

    model, session = _model(scheme=scheme)
    oracle, reference = _model(scheme=scheme)
    prefix, candidate = [1, 2, 3], [4, 5]
    model.prefill(session, prefix)
    oracle.prefill(reference, prefix)
    model.cfg.index_topk, model.slots = 16, 8
    old_plan = model.lifecycle.plan = SimpleNamespace(metadata={"workspace_query_tokens": 128})
    old_pool = model._shared_pool = object()
    old_sessions = model.lifecycle.sessions
    caches = [runner.cache for runner in session.runners]
    contents = [torch.stack(cache.records).clone() for cache in caches]
    with pytest.raises(CacheBudgetExceeded, match="full exact selection"):
        model.plan_resources(CacheFootprint(2**30, 2**30), {"max_session_capacity": 32})
    assert model.lifecycle.plan is old_plan and model._shared_pool is old_pool
    assert model.lifecycle.sessions is old_sessions and model.lifecycle.sessions == {session}
    assert session.length == len(prefix) and not session.released
    for runner, cache, expected in zip(session.runners, caches, contents, strict=True):
        assert runner.cache is cache and cache.length == cache.written == len(prefix)
        torch.testing.assert_close(torch.stack(cache.records), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        model.extend(session, candidate), oracle.extend(reference, candidate), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_pinned_bin_does_not_inflate_record_transfer_bytes():
    from models.deepseek_v32.cache.session import DenseCache

    records = torch.empty((65, 576), dtype=torch.bfloat16, device="cuda")
    backend = SimpleNamespace(_dense_sources=[], _check_dense_execution=lambda session: None)
    cache = DenseCache(65, 576, device="cuda", backend=backend)
    cache.records = records
    assert cache.host.shape == (65, 576) and cache.host.is_pinned()
    assert _storage_bytes([cache.host, cache.host[:1]]) == {"hbm": 0, "dram": 131072}
    cache.begin_step(1)
    cache.append(torch.ones((1, 576), dtype=torch.bfloat16, device="cuda"))
    torch.cuda.synchronize()
    cache.commit()
    metrics = cache.metrics()
    assert metrics["device_to_host_bytes"] == 1152
    assert metrics["host_record_bytes"] == 65 * 1152
    assert metrics["host_allocation_bytes"] == 131072
    torch.testing.assert_close(cache.host[:1], torch.ones((1, 576), dtype=torch.bfloat16))


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
    with pytest.raises(RuntimeError, match="released"):
        model.extend(session, [1])


def test_prefill_chunks_all_layers_and_executes_last_token_head_once(monkeypatch):
    import models.deepseek_v32.execution.adapter as module

    model, session = _model()
    oracle, reference = _model()
    model.chunk_size, oracle.chunk_size = 2, 64
    calls = []
    original = module.F.linear

    def head(input, weight):
        calls.append(tuple(input.shape))
        return original(input, weight)

    ids = [1, 4, 3, 8, 2, 6, 5]
    expected = oracle.prefill(reference, ids)
    with monkeypatch.context() as patch:
        patch.setattr(module.F, "linear", head)
        actual = model.prefill(session, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.last_logits, oracle.last_logits, rtol=0, atol=0)
    assert calls == [(1, 8)]
    for layer, block in enumerate(model.blocks):
        assert [len(hidden) for hidden, _ in block.inputs] == [2, 2, 2, 1]
        if layer >= 3:
            for actual_input, source_input in zip(block.inputs, model.blocks[layer % 3].inputs):
                for actual_tensor, source_tensor in zip(actual_input, source_input):
                    if source_tensor is not None:
                        torch.testing.assert_close(actual_tensor, source_tensor, rtol=0, atol=0)
                        assert actual_tensor.data_ptr() != source_tensor.data_ptr()
    assert session.length == len(ids)


def test_prefill_chunk_size_does_not_split_candidate_batch():
    model, session = _model()
    model.chunk_size = 2
    model.prefill(session, [1, 2, 3])
    model.extend(session, [4, 5, 6, 7, 8])
    assert all(len(block.inputs[-1][0]) == 5 for block in model.blocks)
    assert all(block.execution_chunk_sizes[-1] == 5 for block in model.blocks)


def test_serving_unrecoverable_drain_poison_prevents_retry():
    model, session = _model()
    model.prefill(session, [1, 2, 3])

    def failure():
        raise RuntimeError("permanent CUDA failure")

    model.synchronize = failure
    with pytest.raises(ExceptionGroup) as failed:
        model.extend(session, [4, 5])
    assert all("permanent CUDA" in str(error) for error in failed.value.exceptions)
    assert model.lifecycle.poisoned
    assert session.length == 3
    with pytest.raises(RuntimeError, match="poisoned"):
        model.extend(session, [4, 5])


def test_graph_completion_failure_retains_backend_and_admission_owner():
    owner = object()
    model, session = _model(owner=owner)

    class FailedCompletion:
        failed = False

        @contextmanager
        def execution(self):
            yield
            self.failed = True
            raise RuntimeError("injected graph completion event failure")

    graph = model._compute_graphs = FailedCompletion()
    with (
        pytest.raises(RuntimeError, match="graph completion event"),
        model._execution(session, owner=owner),
    ):
        assert model.lifecycle.active_session is session
    assert model.lifecycle.poisoned and model.lifecycle.active_session is session
    assert model.lifecycle.admission_owner is owner and model._compute_graphs is graph
    assert model._execution_active()
    with pytest.raises(RuntimeError, match="poisoned"):
        model.extend(session, [4, 5])
    with pytest.raises(RuntimeError, match="release execution"):
        model.unbind_owner(owner)
    with pytest.raises(RuntimeError, match="poisoned"):
        model._release_shared()


def test_completed_transient_cleanup_keeps_history_without_truncating_again():
    model, session = _model()
    model.prefill(session, [1, 2, 3])
    session.last_candidate_transient = True
    for runner in session.runners:
        runner.cache.indexer_visible_end = session.length
        runner.cache.transient_start = None

        def unexpected_truncate(length):
            raise AssertionError("completed transient history was truncated again")

        runner.cache.truncate = unexpected_truncate
    synchronized = []
    model.synchronize = lambda: synchronized.append(True)
    saved = [[row.clone() for row in runner.cache.records] for runner in session.runners]
    model.truncate(session, 3)
    assert not synchronized and session.length == 3
    for runner, records in zip(session.runners, saved, strict=True):
        torch.testing.assert_close(runner.cache.records, records, rtol=0, atol=0)


@pytest.mark.parametrize("target", [2, 3])
def test_ordinary_and_shorter_transient_cleanup_still_truncate(target):
    model, session = _model()
    model.prefill(session, [1, 2, 3])
    session.last_candidate_transient = target < 3
    calls = []
    for runner in session.runners:
        original = runner.cache.truncate

        def truncate(length, original=original):
            calls.append(length)
            original(length)

        runner.cache.truncate = truncate
    model.truncate(session, target)
    assert calls == [target] * len(session.runners)
    assert session.length == target


def test_transient_cleanup_does_not_hide_an_unfinished_layer():
    model, session = _model()
    model.prefill(session, [1, 2, 3])
    session.last_candidate_transient = True
    for runner in session.runners:
        runner.cache.indexer_visible_end = session.length
        runner.cache.transient_start = None
    cache = session.runners[0].cache
    cache._step_end = 4

    def reject_unfinished(length):
        raise RuntimeError("rollback active step before truncation")

    cache.truncate = reject_unfinished
    with pytest.raises(RuntimeError, match="rollback active step"):
        model.truncate(session, 3)


class _DenseBlock(_Block):
    def forward(self, hidden, residual, scope=None, *, attention, chunk_size):
        assert self.attention is None and self.cache is None
        if self.fail:
            raise RuntimeError("injected layer failure")
        self.inputs.append((hidden, residual))
        self.execution_chunk_sizes.append(chunk_size)
        combined = hidden.float() if residual is None else hidden.float() + residual.float()
        cache = attention.cache
        cache.reserve_append_source()
        records = combined.bfloat16()
        past = cache.records[: cache.written].float()
        cumulative = torch.cat((past, records.float())).cumsum(0)[cache.written :]
        denominator = torch.arange(
            cache.written + 1, cache.written + len(hidden) + 1, device=hidden.device
        )[:, None]
        cache.append(records)
        return (
            (combined * 0.25 + cumulative / denominator * 0.5 + self.source).bfloat16(),
            (combined * 0.125).bfloat16(),
        )


def _dense_model(monkeypatch, *, device="cpu"):
    import models.deepseek_v32.execution.pipeline as module
    from cache.prefix_pool import CacheFootprint

    model, _ = _model(scheme="dense_prefetch")
    model.device = torch.device(device)
    model.embedding_weight = model.embedding_weight.to(model.device)
    model.head_weight = model.head_weight.to(model.device)
    model.final_norm = model.final_norm.to(model.device)
    if model.device.type == "cuda":
        model.synchronize = lambda: torch.cuda.synchronize(model.device)
    model.cfg.kv_lora_rank, model.cfg.qk_rope_head_dim = 6, 2
    model.cfg.index_topk, model.cfg.index_head_dim = 4, 4
    model.lifecycle.sessions.clear()
    model._shared_pool = model.lifecycle.plan = model._dense_staging = None
    model._dense_lease = None
    model._dense_sources = []
    model.workspace_query_tokens = 8
    model.extend_chunk_size = None
    model.host_arena_tokens = None
    model.attentions = [object() for _ in range(model.num_layers)]
    model.blocks = [_DenseBlock(layer % 3) for layer in range(model.num_layers)]

    def attention_stub(attention, capacity, *, dense_backend, **kwargs):
        return SimpleNamespace(
            cache=module.DenseCache(capacity, 8, device=model.device, backend=dense_backend),
            index_keys=torch.empty((capacity, 4), dtype=torch.float8_e4m3fn, device=model.device),
            index_scales=torch.empty(capacity, dtype=torch.float32, device=model.device),
            offset=torch.zeros(16, dtype=torch.float32, device=model.device),
        )

    monkeypatch.setattr(module, "ServingAttention", attention_stub)
    plan = model.plan_resources(CacheFootprint(2**20, 2**20), {"max_session_capacity": 64})
    model.allocate_shared(plan)
    model.synchronize()
    return model


def test_dense_shared_staging_alternating_users_chunks_aliases_and_full_copy_bytes(monkeypatch):
    model = _dense_model(monkeypatch)
    alice, bob = model.create_session(32), model.create_session(48)
    storage = model._dense_staging.storage_tensors()[0]
    fixed = model.shared_bytes()
    assert fixed == {"hbm": 2 * 64 * 8 * 2, "dram": 0}
    assert not alice.stages and not bob.stages
    assert all(runner.cache.records is None for runner in alice.runners + bob.runners)
    assert all(
        left.cache.host.data_ptr() != right.cache.host.data_ptr()
        for left, right in zip(alice.runners, bob.runners, strict=True)
    )
    prefix, candidate = [1, 3, 8, 4, 2], [5, 7]
    model.prefill(alice, prefix)
    expected = model.extend(alice, candidate).clone()
    logits = model.last_logits.clone()
    model.truncate(alice, len(prefix))
    model.prefill(bob, [9, 11, 8, 7, 3, 2, 5])
    model.extend(bob, [12, 13, 4])
    model.truncate(bob, 7)
    actual = model.extend(alice, candidate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.last_logits, logits, rtol=0, atol=0)
    assert model.shared_bytes() == fixed
    assert model._dense_staging.storage_tensors()[0] is storage
    assert model._dense_lease is model.lifecycle.active_session is None
    assert not model._dense_sources
    assert all(runner.cache.records is None for runner in alice.runners + bob.runners)
    assert sum(runner.cache.metrics()["host_to_device_bytes"] for runner in alice.runners) == (
        model.num_layers * len(prefix) * 8 * 2
    )
    assert sum(runner.cache.metrics()["device_to_host_bytes"] for runner in alice.runners) == (
        model.num_layers * len(candidate) * 8 * 2
    )
    model.release_session(alice)
    model.release_session(bob)
    assert model._dense_staging.storage_tensors()[0] is storage
    model.close()
    assert model._dense_staging is None


def test_dense_session_accounting_excludes_borrowed_storage_even_during_execution(monkeypatch):
    model = _dense_model(monkeypatch)
    session = model.create_session(32)
    before = model.session_bytes(session)
    observations = []
    model.capture_hook = lambda *args: observations.append(model.session_bytes(session))
    model.extend(session, [1, 2])
    assert observations == [before] * model.num_layers
    assert all(runner.cache.records is None for runner in session.runners)
    with pytest.raises(RuntimeError, match="execution lease"):
        session.runners[0].cache.begin_step(1)
    model.release_session(session)
    model.close()


def test_dense_failure_drains_speculative_copy_then_other_user_and_retry_match(monkeypatch):
    model = _dense_model(monkeypatch)
    alice, bob = model.create_session(32), model.create_session(32)
    model.prefill(alice, [1, 2, 3])
    expected = model.extend(alice, [4, 5]).clone()
    logits = model.last_logits.clone()
    model.truncate(alice, 3)
    model.blocks[4].fail = True
    with pytest.raises(RuntimeError, match="injected layer"):
        model.extend(alice, [4, 5])
    assert alice.length == 3
    assert all(runner.cache.written == runner.cache.length == 3 for runner in alice.runners)
    assert model._dense_staging.active_lease is None
    assert all(runner.cache.records is None for runner in alice.runners)
    model.blocks[4].fail = False
    model.prefill(bob, [8, 9, 6])
    actual = model.extend(alice, [4, 5])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.last_logits, logits, rtol=0, atol=0)
    model.release_session(alice)
    model.release_session(bob)
    model.close()


def test_dense_failed_lease_join_retains_borrowed_aliases_sources_and_ownership(monkeypatch):
    model = _dense_model(monkeypatch)
    session = model.create_session(32)
    model.blocks[1].fail = True

    def fail(lease):
        raise RuntimeError("injected copy stream join failure")

    monkeypatch.setattr(model._dense_staging, "_join", fail)
    with pytest.raises(ExceptionGroup) as failed:
        model.extend(session, [1, 2])
    assert "injected layer failure" in str(failed.value.exceptions[0])
    assert "copy stream join failure" in str(failed.value.exceptions[1])
    assert model.lifecycle.poisoned and model.lifecycle.active_session is session
    assert model._dense_lease is not None and model._dense_sources
    assert session.runners[1].cache.records is not None
    with pytest.raises(RuntimeError, match="poisoned"):
        model.extend(session, [3])
    with pytest.raises(RuntimeError, match="execution"):
        model.close()


def test_dense_failed_reset_join_stays_poisoned_after_later_cleanup_join(monkeypatch):
    model = _dense_model(monkeypatch)
    session = model.create_session(32)
    staging = model._dense_staging
    storage = staging.storage_tensors()[0]
    original = staging._join
    attempted = []

    def fail_once(lease):
        attempted.append(True)
        if len(attempted) == 1:
            error = RuntimeError("injected chunk join failure")
            staging._poison(error)
            raise error
        return original(lease)

    monkeypatch.setattr(staging, "_join", fail_once)
    with pytest.raises(ExceptionGroup) as failed:
        model.prefill(session, [1, 2, 3, 4, 5])
    assert "injected chunk join failure" in str(failed.value.exceptions[0])
    assert len(attempted) == 2 and model.lifecycle.poisoned
    assert model.lifecycle.active_session is session and model._dense_lease is not None
    assert staging.storage_tensors()[0] is storage
    with pytest.raises(RuntimeError, match="poisoned"):
        model.extend(session, [6])


def test_dense_source_reservation_waits_before_a_third_projection(monkeypatch):
    from models.deepseek_v32.cache.session import DenseWrite

    model = _dense_model(monkeypatch)
    session = model.create_session(32)
    waited = []
    with model._execution(session):
        model.device = torch.device("cuda:0")  # Fake events; no CUDA call.
        model._dense_sources = [
            DenseWrite(
                torch.empty((8, 8), dtype=torch.bfloat16),
                SimpleNamespace(query=lambda: False, synchronize=lambda: waited.append(True)),
            )
            for _ in range(2)
        ]
        model._reserve_dense_source()
        assert waited == [True] and len(model._dense_sources) == 1
        model.device = torch.device("cpu")
    model.release_session(session)
    model.close()


def test_planned_candidate_rejects_before_forward_or_any_cache_mutation(monkeypatch):
    from cache.prefix_pool import CacheBudgetExceeded

    model = _dense_model(monkeypatch)
    session = model.create_session(32)
    model.prefill(session, [1, 2, 3])
    storage = model._dense_staging.storage_tensors()[0]
    before = [runner.cache.host[:3].clone() for runner in session.runners]

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized candidate reached forward or allocation")

    monkeypatch.setattr(model, "_forward", forbidden)
    monkeypatch.setattr(torch, "as_tensor", forbidden)
    with pytest.raises(CacheBudgetExceeded, match="planned candidate limit"):
        model.extend(session, list(range(1, 10)))
    assert session.length == 3 and model.lifecycle.active_session is None
    assert model._dense_staging.storage_tensors()[0] is storage
    for runner, expected in zip(session.runners, before, strict=True):
        assert runner.cache.written == runner.cache.length == 3
        torch.testing.assert_close(runner.cache.host[:3], expected, rtol=0, atol=0)
    model.release_session(session)
    model.close()


def test_partial_dense_session_failure_drains_and_retains_shared_staging(monkeypatch):

    model = _dense_model(monkeypatch)
    staging = model._dense_staging
    original = model.pipeline.create_runner
    calls, drained = [], []

    def failure(*args, **kwargs):
        if calls:
            raise RuntimeError("injected partial dense session failure")
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.pipeline, "create_runner", failure)
    monkeypatch.setattr(model, "synchronize", lambda: drained.append(True))
    with pytest.raises(RuntimeError, match="partial dense session failure"):
        model.create_session(32)
    assert drained == [True] and not model.lifecycle.sessions
    assert model._dense_staging is staging and not staging.closed
    model.close()


def test_borrowed_sparse_cache_requires_its_model_operation_for_all_transactions():
    from cache.sparse_token_pool import SharedSparseTokenPool
    from models.deepseek_v32.cache.session import ServingSparseTokenCache

    lifecycle = ResourceLifecycle("DeepSeek cache test")
    owner = object()
    lifecycle.bind_owner(owner)
    lifecycle.allocated(object(), owner=owner)
    pool = SharedSparseTokenPool(64, 8, 1, 64, device="cpu")
    backing = pool.allocate_session(64)
    cache = ServingSparseTokenCache.for_layer(backing, 0)
    session = DeepSeekServingSession(64, "echo", [], SimpleNamespace(lifecycle=lifecycle))
    session.registration = lifecycle.register(session, owner=owner)
    backing.bind_release_guard(session.check_backing_release)
    cache.bind_serving(session, lifecycle)
    for release in (backing.release, lambda: pool.release_session(backing)):
        with pytest.raises(RuntimeError, match="release lease"):
            release()
    assert pool.free_host_pages == 0 and not backing.released
    with pytest.raises(RuntimeError, match="execution lease"):
        cache.begin_step(1)
    with lifecycle.execution(session, session.registration, owner=owner):
        with pytest.raises(RuntimeError, match="release lease"):
            backing.release()
        cache.begin_step(1)
        cache.append(torch.ones((1, 8), dtype=torch.bfloat16))
        cache.commit()
    assert cache.length == 1
    with pytest.raises(RuntimeError, match="active session operation"):
        cache.truncate(0)
    with pytest.raises(RuntimeError, match="execution lease"):
        cache.append(torch.ones((1, 8), dtype=torch.bfloat16))
    with lifecycle.mutation(session, session.registration, owner=owner):
        cache.truncate(0)
    assert cache.length == 0
    with lifecycle.mutation(session, session.registration, owner=owner, releasing=True):
        backing.release()
        lifecycle.detach(session)
    assert pool.free_host_pages == 1 and backing._release_guard is None
    lifecycle.unbind_owner(owner)
    lifecycle.close(pool.close)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_cuda_session_ready_stream_handoff_and_failed_speculative_copy(monkeypatch):
    model = _dense_model(monkeypatch, device="cuda:0")
    allocation_stream = torch.cuda.Stream()
    caller = torch.cuda.Stream()
    with torch.cuda.stream(allocation_stream):
        torch.cuda._sleep(200_000_000)
        alice = model.create_session(32)
    assert alice.dense_allocation_ready is not None
    with torch.cuda.stream(caller):
        model.prefill(alice, [1, 2, 3, 4, 5])
        expected = model.extend(alice, [6, 7]).clone()
        logits = model.last_logits.clone()
        model.truncate(alice, 5)
    bob = model.create_session(48)
    model.prefill(bob, [8, 9, 10, 11, 12, 13])
    actual = model.extend(alice, [6, 7])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.last_logits, logits, rtol=0, atol=0)
    model.truncate(alice, 5)
    assert all(runner.cache.records is None for runner in alice.runners + bob.runners)
    original = model._prefetch_dense_layer
    speculative_done = torch.cuda.Event()

    def delayed(session, layer):
        if layer == 2:
            with torch.cuda.stream(model._dense_staging._copy_stream):
                torch.cuda._sleep(20_000_000)
        original(session, layer)
        if layer == 2:
            speculative_done.record(model._dense_staging._copy_stream)

    monkeypatch.setattr(model, "_prefetch_dense_layer", delayed)
    model.blocks[1].fail = True
    with torch.cuda.stream(caller), pytest.raises(RuntimeError, match="injected layer"):
        model.extend(alice, [6, 7])
    assert speculative_done.query()
    assert alice.length == 5 and model._dense_staging.active_lease is None
    assert all(runner.cache.records is None for runner in alice.runners)
    model.blocks[1].fail = False
    with torch.cuda.stream(caller):
        actual = model.extend(alice, [6, 7])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.last_logits, logits, rtol=0, atol=0)
    model.release_session(alice)
    model.release_session(bob)
    model.close()
