"""Native fixed-cache integration; random layers are not checkpoint acceptance."""

import os

import pytest
import torch

from models.nosa.fixed_serving import NosaFixedServingBackend
from models.nosa.tests.test_model import tiny_config
from models.nosa.tests.test_sparse_model import initialized_sparse_model
from serving.persistent import PersistentGRRunner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
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
            max_position_embeddings=8320,
        ),
        device="cuda:0",
        dtype=torch.bfloat16,
        backend="triton",
    )


def setup(model, scheme):
    backend = NosaFixedServingBackend(
        model, scheme, chunk_size=1024, sparse_pool_tokens=8192, host_arena_tokens=16384
    )
    limits = {
        "max_session_capacity": 8320,
        "max_history_tokens": 8192,
        "max_candidate_tokens": 128,
    }
    return backend, limits


def request(model, user):
    return {
        "user_id": user,
        "input_ids": ((torch.arange(8320) * 19 + user * 23) % model.config.vocab_size).tolist(),
        "stable_prefix_tokens": 8192,
    }


@pytest.mark.parametrize("scheme", ["dense_prefetch", "serial_sparse", "overlap"])
@torch.inference_mode()
def test_cuda_fixed_independent_prefixes_conflicts_hits_and_discard(model, scheme):
    control, limits = setup(model, "hbm")
    expected = {}
    with PersistentGRRunner(control, resource_limits=limits) as runner:
        for user in (0, 1):
            expected[user] = runner.execute(request(model, user)).hidden.clone()
    control.close()
    backend, limits = setup(model, scheme)
    with PersistentGRRunner(backend, resource_limits=limits) as runner:
        results = []
        history = None
        for user in (0, 1, 0, 0):
            result = runner.execute(request(model, user))
            results.append(result)
            torch.testing.assert_close(result.hidden, expected[user], rtol=0.05, atol=0.05)
            session = runner.pool._entries[user].session
            assert session.length == session.indexer_cache.length == 8192
            assert session._pending_end is None
            assert session.buffers["keys"].shape[1] == 8192
            diagnostics = result.metrics["cache_diagnostics"]
            assert diagnostics["candidate_main_kv_device_to_host_bytes"] == 0
            if user == 0:
                if history is None:
                    history = session.buffers["keys"].clone()
                else:
                    torch.testing.assert_close(session.buffers["keys"], history, rtol=0, atol=0)
        assert results[2].metrics["prefix_cache_hit"]
        assert results[2].metrics["cache_diagnostics"]["candidate_main_kv_host_to_device_bytes"] > 0
        assert (
            results[3].metrics["cache_diagnostics"]["candidate_main_kv_host_to_device_bytes"] == 0
        )
        for result in results:
            assert result.metrics["session_host_pages"] == 128
    backend.close()


@torch.inference_mode()
def test_cuda_fixed_dense_failure_drains_unconsumed_prefetch(model, monkeypatch):
    backend, limits = setup(model, "dense_prefetch")
    with PersistentGRRunner(backend, resource_limits=limits) as runner:
        runner.execute(request(model, 0))
        runner.execute(request(model, 1))
        session = runner.pool._entries[0].session
        resources = backend.resources
        copy_finished = torch.cuda.Event()
        prefetch = resources.prefetch

        def delayed_prefetch(session, layer):
            prefetch(session, layer)
            if layer == 1:
                with torch.cuda.stream(resources._prefetch_stream):
                    torch.cuda._sleep(20_000_000)
                    copy_finished.record()

        def failure(*args, **kwargs):
            raise RuntimeError("injected after queued next-layer prefetch")

        with monkeypatch.context() as patch:
            patch.setattr(resources, "prefetch", delayed_prefetch)
            patch.setattr(model.model.layers[1], "forward", failure)
            with pytest.raises(RuntimeError, match="injected after queued"):
                runner.execute(request(model, 0))
        assert copy_finished.query()
        assert session.released
        assert resources._active_session is None
        assert not resources.poisoned
        result = runner.execute(request(model, 0))
        assert torch.isfinite(result.hidden).all()
    backend.close()


@pytest.mark.parametrize("scheme", ["dense_prefetch", "serial_sparse", "overlap"])
@torch.inference_mode()
def test_cuda_short_history_suffix_evicts_conflicting_long_history_pages(model, scheme):
    backend, limits = setup(model, scheme)
    long_request = request(model, 0)
    short_request = request(model, 1)
    short_request["stable_prefix_tokens"] = 4096
    short_request["input_ids"] = short_request["input_ids"][: 4096 + 81]
    with PersistentGRRunner(backend, resource_limits=limits) as runner:
        expected = runner.execute(long_request).hidden.clone()
        runner.execute(short_request)
        assert torch.all(backend.resources.tags[:, 64:66] == 0)
        actual = runner.execute(long_request)
        assert actual.metrics["prefix_cache_hit"]
        torch.testing.assert_close(actual.hidden, expected, atol=0, rtol=0)
    backend.close()


def test_cuda_fixed_full_checkpoint_independent_histories_and_all_candidate_hidden():
    checkpoint = os.environ.get("NOSA_FIXED_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set NOSA_FIXED_CHECKPOINT for full 32-layer H64K/A128 acceptance")
    loaded = NosaFixedServingBackend.from_pretrained(
        checkpoint,
        scheme="hbm",
        sparse_pool_tokens=65536,
        host_arena_tokens=16777216,
        device="cuda:0",
        chunk_size=1024,
        max_seq_len=65664,
    )
    model = loaded.model
    assert model.config.num_hidden_layers == 32
    loaded.close()
    limits = {
        "max_session_capacity": 65664,
        "max_history_tokens": 65536,
        "max_candidate_tokens": 128,
    }
    requests = {
        user: {
            "user_id": user,
            "input_ids": (
                (torch.arange(65664) * 19 + 11 + user * 37) % model.config.vocab_size
            ).tolist(),
            "stable_prefix_tokens": 65536,
        }
        for user in (0, 1)
    }
    compute_graphs = os.environ.get("NOSA_FIXED_COMPUTE_GRAPHS") == "1"

    def run_scheme(scheme, *, graphs=False, expected=None):
        backend = NosaFixedServingBackend(
            model,
            scheme,
            chunk_size=1024,
            sparse_pool_tokens=65536,
            host_arena_tokens=16777216,
        )
        if graphs:
            backend.enable_compute_graphs((128, 1024))
        observed = {}
        with PersistentGRRunner(backend, resource_limits=limits) as runner:
            if graphs:
                assert runner.resource_plan.metadata["compute_graphs"]["enabled"]
            for visit, user in enumerate((0, 1, 0, 0)):
                result = runner.execute(requests[user])
                output = result.hidden.cpu()
                assert output.shape == (128, model.config.hidden_size)
                assert torch.isfinite(output).all()
                observed[(visit, user)] = output.clone()
                if expected is not None:
                    torch.testing.assert_close(output, expected[(visit, user)], rtol=0, atol=0)
                    torch.testing.assert_close(
                        output.view(torch.uint8),
                        expected[(visit, user)].view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                if scheme != "hbm":
                    counters = result.metrics["cache_diagnostics"]
                    assert counters["candidate_main_kv_device_to_host_bytes"] == 0
                    if visit == 2:
                        assert counters["candidate_main_kv_host_to_device_bytes"] > 0
                    if visit == 3:
                        assert counters["candidate_main_kv_host_to_device_bytes"] == 0
                session = runner.pool._entries[user].session
                assert session.length == session.indexer_cache.length == 65536
                assert session._pending_end is None
                print(
                    f"full_checkpoint {scheme} graphs={graphs} visit={visit} user={user} passed",
                    flush=True,
                )
            if graphs:
                evidence = backend.compute_graphs.describe()
                assert evidence["project_replays"] > 0
                assert evidence["project_replays"] == evidence["finish_replays"]
                assert evidence["eager_fallbacks"] == 0
        backend.close()
        return observed

    # Graph execution must be compared with independent eager resident histories.
    expected = run_scheme("hbm")
    for scheme in NosaFixedServingBackend.schemes:
        if scheme != "hbm" or compute_graphs:
            run_scheme(scheme, graphs=compute_graphs, expected=expected)
