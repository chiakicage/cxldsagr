"""Pure-compute graph equivalence and invalidation with real fixed caches."""

from types import SimpleNamespace

import pytest
import torch

from models.nosa.execution.fixed import NosaFixedServingBackend
from models.nosa.tests.test_model import tiny_config
from models.nosa.tests.test_sparse_model import initialized_sparse_model
from serving.persistent import PersistentGRRunner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def model():
    from operators.nosa._native import native_enabled

    assert torch.cuda.get_device_capability() == (9, 0)
    assert native_enabled()
    return initialized_sparse_model(
        tiny_config(
            hidden_size=256,
            intermediate_size=384,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=2,
            head_dim=128,
            vocab_size=127,
            max_position_embeddings=65664,
        ),
        device="cuda:0",
        dtype=torch.bfloat16,
        backend="triton",
    )


def backend_for(model, scheme, *, graph=False):
    backend = NosaFixedServingBackend(
        model, scheme, chunk_size=1024, sparse_pool_tokens=2048, host_arena_tokens=4096
    )
    if graph:
        backend.enable_compute_graphs((128, 1024))
    return backend


LIMITS = {
    "max_session_capacity": 2176,
    "max_history_tokens": 2048,
    "max_candidate_tokens": 128,
}


def request(model, user=0, *, candidate_salt=0):
    history = 2048 if user == 0 else 1024
    tokens = (torch.arange(history + 128) * 19 + user * 23) % model.config.vocab_size
    tokens[history:] = (tokens[history:] + candidate_salt) % model.config.vocab_size
    return {"user_id": user, "input_ids": tokens.tolist(), "stable_prefix_tokens": history}


@pytest.mark.parametrize("scheme", NosaFixedServingBackend.schemes)
def test_graphs_match_eager_across_positions_users_data_and_preserve_outputs(model, scheme):
    requests = [request(model), request(model, 1), request(model, candidate_salt=17)]
    eager = backend_for(model, scheme)
    with PersistentGRRunner(eager, resource_limits=LIMITS) as runner:
        expected = [runner.execute(value).hidden.clone() for value in requests]
    eager.close()
    backend = backend_for(model, scheme, graph=True)
    graphs = backend.compute_graphs
    retained = []
    with PersistentGRRunner(backend, resource_limits=LIMITS) as runner:
        assert runner.resource_plan.metadata["compute_graphs"]["enabled"]
        for value, control in zip(requests, expected, strict=True):
            result = runner.execute(value)
            torch.testing.assert_close(result.hidden, control, atol=0, rtol=0)
            retained.append((result.hidden, result.hidden.clone()))
            session = runner.pool._entries[value["user_id"]].session
            assert session.length == session.indexer_cache.length == value["stable_prefix_tokens"]
        for result, snapshot in retained:
            torch.testing.assert_close(result, snapshot, atol=0, rtol=0)
        assert graphs.describe()["project_replays"] > 0
        assert graphs.describe()["project_replays"] == graphs.describe()["finish_replays"]
        assert graphs.private_reserved_bytes > 0
        assert runner.resource_plan.shared.hbm >= graphs.shared_bytes()["hbm"]
    backend.close()
    assert graphs.closed


def test_parameter_mutation_is_rejected_before_replay_and_releases_session(model):
    backend = backend_for(model, "hbm", graph=True)
    with PersistentGRRunner(backend, resource_limits=LIMITS) as runner:
        runner.execute(request(model))
        session = runner.pool._entries[0].session
        replays = backend.compute_graphs.describe()["project_replays"]
        with torch.no_grad():
            model.model.layers[1].mlp.gate_up_proj.weight.add_(0.01)
        with pytest.raises(RuntimeError, match="parameters changed"):
            runner.execute(request(model, candidate_salt=1))
        assert session.released
        assert backend.compute_graphs.describe()["project_replays"] == replays
    backend.close()


def test_precision_policy_mutation_rejects_and_restoration_allows_new_session(model):
    backend = backend_for(model, "hbm", graph=True)
    with PersistentGRRunner(backend, resource_limits=LIMITS) as runner:
        expected = runner.execute(request(model)).hidden.clone()
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not previous
            with pytest.raises(RuntimeError, match="precision policy changed"):
                runner.execute(request(model))
            assert len(runner.pool) == 0
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
        torch.testing.assert_close(runner.execute(request(model)).hidden, expected, rtol=0, atol=0)
    backend.close()


def test_captured_norm_policy_mutation_rejects_before_replay(model):
    backend = backend_for(model, "hbm", graph=True)
    with PersistentGRRunner(backend, resource_limits=LIMITS) as runner:
        runner.execute(request(model))
        model.model.layers[1].input_layernorm.eps *= 10
        with pytest.raises(RuntimeError, match="configuration|policy"):
            runner.execute(request(model))
        assert len(runner.pool) == 0
    backend.close()


def test_failed_graph_capacity_audit_releases_its_private_pools(model):
    from cache.allocator.budget import validate_allocator

    backend = backend_for(model, "hbm")
    with pytest.raises(RuntimeError, match="capacity|limit"):
        backend.enable_compute_graphs((128, 1024), private_limit_bytes=1)
    assert backend.compute_graphs is None
    assert backend.resources.allowed_graph_pool_ids == set()
    assert validate_allocator("cuda:0")["policy"]
    with PersistentGRRunner(backend, resource_limits=LIMITS) as runner:
        assert torch.isfinite(runner.execute(request(model)).hidden).all()
    backend.close()


def test_unsupported_query_shape_aborts_without_hidden_fallback(model):
    backend = backend_for(model, "serial_sparse", graph=True)
    with PersistentGRRunner(backend, resource_limits=LIMITS) as runner:
        expected = runner.execute(request(model)).hidden.clone()
        short = request(model)
        short["input_ids"] = short["input_ids"][:-64]
        with pytest.raises(ValueError, match="No NOSA compute graph"):
            runner.execute(short)
        assert len(runner.pool) == 0
        torch.testing.assert_close(runner.execute(request(model)).hidden, expected, rtol=0, atol=0)
        assert backend.compute_graphs.describe()["eager_fallbacks"] == 0
    backend.close()


def test_attention_failure_rolls_back_then_recreates_session(model, monkeypatch):
    backend = backend_for(model, "dense_prefetch", graph=True)
    with PersistentGRRunner(backend, resource_limits=LIMITS) as runner:
        expected = runner.execute(request(model)).hidden.clone()
        with monkeypatch.context() as patch:

            def fail(*args, **kwargs):
                raise RuntimeError("injected eager attention failure")

            patch.setattr(model.model.layers[1].self_attn, "attend", fail)
            with pytest.raises(RuntimeError, match="injected eager attention"):
                runner.execute(request(model))
        assert len(runner.pool) == 0
        assert not backend.compute_graphs._active
        assert backend.resources._active_session is None
        torch.testing.assert_close(runner.execute(request(model)).hidden, expected, rtol=0, atol=0)
    backend.close()


def test_rejected_model_call_preserves_existing_validation_owner(model):
    backend = backend_for(model, "hbm", graph=True)
    batch = backend.compute_graphs.validation
    owner = SimpleNamespace(_pending_end=128, device=model.model.embed_tokens.weight.device)
    batch.begin(owner)
    try:
        with pytest.raises(ValueError, match="transactional cache"):
            model(
                torch.zeros(128, device=owner.device, dtype=torch.long),
                return_hidden=True,
                compute_graphs=backend.compute_graphs,
            )
        assert batch.owner is owner
        assert owner._deferred_validation is batch
    finally:
        batch.clear()
        backend.close()


@pytest.mark.parametrize("scheme", NosaFixedServingBackend.schemes)
@pytest.mark.parametrize("layer_idx", [0, 1])
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("query", float("nan")), ("keys", float("inf")), ("cis_scores", -float("inf"))],
)
def test_nonfinite_graph_prefix_and_candidate_abort_without_publishing(
    model, monkeypatch, scheme, layer_idx, field, bad_value
):
    backend = NosaFixedServingBackend(
        model, scheme, chunk_size=1024, sparse_pool_tokens=65536, host_arena_tokens=131072
    )
    backend.enable_compute_graphs((128, 1024))
    limits = {
        "max_session_capacity": 65664,
        "max_history_tokens": 65536,
        "max_candidate_tokens": 128,
    }
    value = {
        "user_id": 0,
        "input_ids": ((torch.arange(65664) * 19) % model.config.vocab_size).tolist(),
        "stable_prefix_tokens": 65536,
    }
    graphs = backend.compute_graphs
    captured = {}
    original_release = backend.release_session

    def release(session, *, owner=None):
        if captured.get("session") is session:
            # Inspect the failed transaction before release resets its cursors.
            assert session.length == session.indexer_cache.length == captured["length"]
            assert session._pending_end is session.indexer_cache._pending_end is None
            assert not session.indexer_cache._pending_states
            assert not session.indexer_cache._reservations
            states = tuple(
                (dict(state.lengths), state.validated_tokens)
                for state in session.indexer_cache._committed
            )
            assert states == captured["states"]
            for view, expected in captured["history"]:
                torch.testing.assert_close(view, expected, rtol=0, atol=0)
            assert getattr(session, "_deferred_validation", None) is None
            assert graphs.validation.owner is None
            captured["release_checked"] = True
        original_release(session, owner=owner)

    monkeypatch.setattr(backend, "release_session", release)
    attention = model.model.layers[layer_idx].self_attn
    original_attend = attention.attend
    with PersistentGRRunner(backend, resource_limits=limits) as runner:
        expected = runner.execute(value).hidden.clone()
        for phase in ("candidate", "prefix"):
            captured.clear()

            def corrupt(q, records, main_attention, cache, index, *, indexer=None, phase=phase):
                selected = (
                    cache.transient_candidate if phase == "candidate" else cache.length == 32768
                )
                if selected and not captured:
                    length = cache.length
                    history = [tensor[:, :length] for tensor in cache.buffers.values()]
                    if hasattr(cache, "cis_scores"):
                        history.append(cache.cis_scores[:, :length])
                    states = cache.indexer_cache._committed
                    history.extend(
                        tensor[: states[layer].lengths[name]]
                        for layer, buffers in cache.indexer_cache._buffers.items()
                        for name, tensor in buffers.items()
                    )
                    captured.update(
                        session=cache,
                        length=length,
                        states=tuple(
                            (dict(state.lengths), state.validated_tokens) for state in states
                        ),
                        history=[(view, view.clone()) for view in history],
                    )
                    value = q if field == "query" else records[field]
                    value[(0,) * value.ndim] = bad_value
                return original_attend(q, records, main_attention, cache, index, indexer=indexer)

            with monkeypatch.context() as patch:
                patch.setattr(attention, "attend", corrupt)
                with pytest.raises(ValueError, match="finite"):
                    runner.execute(value)
            assert captured["release_checked"]
            assert captured["length"] == (65536 if phase == "candidate" else 32768)
            assert captured["session"].released
            assert len(runner.pool) == 0
            assert not graphs._active
            assert graphs.validation.owner is None
            assert not graphs.validation.seen
            assert backend.resources._active_session is None
            assert not backend.resources.poisoned
        captured.clear()
        torch.testing.assert_close(runner.execute(value).hidden, expected, rtol=0, atol=0)
    backend.close()
