"""Fixed P/NH history admission, page ownership, and candidate transactions."""

import pytest
import torch

from cache.prefix_pool import CacheBudgetExceeded
from models.nosa.fixed_serving import NosaFixedServingBackend
from models.nosa.tests.test_model import tiny_config
from models.nosa.tests.test_sparse_model import initialized_sparse_model
from serving.persistent import PersistentGRRunner


def backend_for(model, scheme, *, history=128, candidate=16, pool=None, host=None):
    backend = NosaFixedServingBackend(
        model,
        scheme,
        sparse_pool_tokens=history if pool is None else pool,
        host_arena_tokens=2 * history if host is None else host,
        chunk_size=64,
    )
    limits = {
        "max_session_capacity": history + candidate,
        "max_history_tokens": history,
        "max_candidate_tokens": candidate,
    }
    return backend, limits


@pytest.mark.parametrize("scheme", NosaFixedServingBackend.schemes)
def test_fixed_schemes_match_resident_and_discard_candidate(scheme):
    config = tiny_config(max_position_embeddings=160, num_hidden_layers=2)
    model = initialized_sparse_model(config)
    tokens = (torch.arange(144) * 3 + 1) % config.vocab_size
    reference, limits = backend_for(model, "hbm")
    candidate, _ = backend_for(model, scheme)
    with PersistentGRRunner(reference, resource_limits=limits) as control:
        expected = control.execute(
            {"user_id": 0, "input_ids": tokens.tolist(), "stable_prefix_tokens": 128}
        ).hidden.clone()
    reference.close()
    with PersistentGRRunner(candidate, resource_limits=limits) as runner:
        request = {"user_id": 0, "input_ids": tokens.tolist(), "stable_prefix_tokens": 128}
        result = runner.execute(request)
        torch.testing.assert_close(result.hidden, expected, atol=1e-6, rtol=1e-5)
        session = runner.pool._entries[0].session
        assert session.length == session.indexer_cache.length == 128
        assert session._pending_end is None
        assert result.metrics["candidate_suffix_tokens"] == 16
        assert result.metrics["session_host_pages"] == (0 if scheme == "hbm" else 2)
        host = None if scheme == "hbm" else session.buffers["keys"].clone()
        if host is not None:
            assert host.shape[1] == 128
        request["input_ids"] = tokens[:136].tolist()
        revisit = runner.execute(request)
        assert revisit.metrics["prefix_cache_hit"]
        assert runner.pool._entries[0].session is session
        assert session.length == session.indexer_cache.length == 128
        if host is not None:
            torch.testing.assert_close(session.buffers["keys"], host, rtol=0, atol=0)
    candidate.close()


@pytest.mark.parametrize("scheme", NosaFixedServingBackend.schemes)
def test_fixed_quota_evicts_whole_sessions_and_revisit_is_not_first(scheme):
    config = tiny_config(max_position_embeddings=80, num_hidden_layers=1)
    backend, limits = backend_for(initialized_sparse_model(config), scheme, history=64, host=128)
    with PersistentGRRunner(backend, resource_limits=limits) as runner:
        results = []
        sessions = []
        for user in (0, 1, 2, 0):
            ids = ((torch.arange(72) + 7 * user) % config.vocab_size).tolist()
            results.append(
                runner.execute({"user_id": user, "input_ids": ids, "stable_prefix_tokens": 64})
            )
            sessions.append(runner.pool._entries[user].session)
        assert results[-1].metrics["is_revisit"]
        assert not results[-1].metrics["prefix_cache_hit"]
        assert sessions[0].released
        quota = 1 if scheme == "hbm" else 2
        assert len(runner.pool) == quota
        assert results[-1].metrics["cached_users"] == quota
    backend.close()


def test_fixed_page_conflict_replaces_only_requested_logical_pages():
    model = initialized_sparse_model(tiny_config(max_position_embeddings=208, num_hidden_layers=1))
    backend, limits = backend_for(model, "serial_sparse", history=192)
    backend.allocate_shared(backend.plan_resources(None, limits))
    first, second = backend.create_session(192), backend.create_session(192)
    try:
        for session, value in ((first, 11), (second, 23)):
            session._length = 192
            session.buffers["keys"].fill_(value)
            session.buffers["values"].fill_(value + 1)
        resources = backend.resources
        with resources.lease(first):
            resources._copy_reference(first, 0, range(3))
        with resources.lease(second):
            resources._copy_reference(second, 0, [[1]] * resources.kv_heads)
        assert torch.all(resources.tags[0, 0] == first.cache_owner)
        assert torch.all(resources.tags[0, 1] == second.cache_owner)
        assert torch.all(resources.tags[0, 2] == first.cache_owner)
        assert torch.all(resources.keys[0, :64] == 11)
        assert torch.all(resources.keys[0, 64:128] == 23)
        assert torch.all(resources.keys[0, 128:192] == 11)
        with pytest.raises(CacheBudgetExceeded):
            backend.create_session(192)
    finally:
        backend.release_session(first)
        backend.release_session(second)
        backend.close()


def test_failed_candidate_releases_session_and_shared_storage_survives(monkeypatch):
    model = initialized_sparse_model(tiny_config(max_position_embeddings=80, num_hidden_layers=2))
    backend, limits = backend_for(model, "dense_prefetch", history=64)
    with PersistentGRRunner(backend, resource_limits=limits) as runner:
        request = {
            "user_id": 0,
            "input_ids": (torch.arange(72) % model.config.vocab_size).tolist(),
            "stable_prefix_tokens": 64,
        }
        expected = runner.execute(request).hidden.clone()
        session = runner.pool._entries[0].session
        keys = backend.resources.keys
        with monkeypatch.context() as patch:

            def fail(*args, **kwargs):
                raise RuntimeError("injected candidate failure")

            patch.setattr(model.model.layers[1], "forward", fail)
            with pytest.raises(RuntimeError, match="injected candidate failure"):
                runner.execute(request)
        assert session.released
        assert len(runner.pool) == 0
        assert backend.resources.keys is keys
        torch.testing.assert_close(runner.execute(request).hidden, expected)
    backend.close()


@pytest.mark.parametrize("scheme", ["dense_prefetch", "serial_sparse", "overlap"])
def test_shorter_history_candidate_invalidates_longer_session_pages(scheme):
    model = initialized_sparse_model(tiny_config(max_position_embeddings=144, num_hidden_layers=2))
    backend, limits = backend_for(model, scheme)
    long_request = {
        "user_id": 0,
        "input_ids": (torch.arange(144) % model.config.vocab_size).tolist(),
        "stable_prefix_tokens": 128,
    }
    short_request = {
        "user_id": 1,
        "input_ids": ((torch.arange(72) + 13) % model.config.vocab_size).tolist(),
        "stable_prefix_tokens": 64,
    }
    with PersistentGRRunner(backend, resource_limits=limits) as runner:
        expected = runner.execute(long_request).hidden.clone()
        runner.execute(short_request)
        assert torch.all(backend.resources.tags[:, 1] == 0)
        actual = runner.execute(long_request)
        assert actual.metrics["prefix_cache_hit"]
        torch.testing.assert_close(actual.hidden, expected, rtol=0, atol=0)
    backend.close()


@pytest.mark.parametrize("history,pool,candidate", [(65, 128, 16), (128, 64, 16)])
def test_fixed_plan_rejects_unsupported_history(history, pool, candidate):
    backend, limits = backend_for(
        initialized_sparse_model(tiny_config(max_position_embeddings=256)),
        "serial_sparse",
        history=history,
        pool=pool,
        host=256,
        candidate=candidate,
    )
    with pytest.raises(ValueError, match="page-aligned"):
        backend.plan_resources(None, limits)
