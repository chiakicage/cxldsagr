"""Persistent NOSA session correctness and cache-budget accounting."""

import os

import pytest
import torch

from models.nosa.serving import NosaDensePrefetchCache, NosaServingBackend
from models.nosa.tests.test_model import tiny_config
from models.nosa.tests.test_sparse_model import initialized_sparse_model


@pytest.mark.parametrize("scheme", NosaServingBackend.schemes)
def test_independent_prefixes_revisits_and_cache_reservations(scheme):
    config = tiny_config(max_position_embeddings=256, num_hidden_layers=4)
    model = initialized_sparse_model(config)
    control = NosaServingBackend(model, "hbm", chunk_size=47)
    backend = NosaServingBackend(model, scheme, chunk_size=47)
    tokens = (torch.arange(160) * 7 + 3) % config.vocab_size
    expected_cache = control.create_session(160)
    session = backend.create_session(160)
    reservation = backend.estimate_session_bytes(160, 79)
    peaks = {"hbm": 0, "dram": 0}

    def observe(*_):
        for name, value in backend.session_bytes(session).items():
            peaks[name] = max(peaks[name], value)
            assert value <= reservation[name]

    hooks = [layer.register_forward_hook(observe) for layer in model.model.layers]
    try:
        control.prefill(expected_cache, tokens[:79])
        backend.prefill(session, tokens[:79])
        for suffix in (tokens[79:112], tokens[100:160], tokens[79:160]):
            expected = control.extend(expected_cache, suffix)
            actual = backend.extend(session, suffix)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            assert len(actual) == len(suffix)
            control.truncate(expected_cache, 79)
            backend.truncate(session, 79)
            assert session.length == session.indexer_cache.length == 79
            if scheme == "dense_prefetch":
                assert session.last_prefetch_bytes == (
                    2
                    * 79
                    * config.num_hidden_layers
                    * config.num_key_value_heads
                    * config.head_dim
                    * torch.empty((), dtype=model.model.embed_tokens.weight.dtype).element_size()
                )
        assert peaks["hbm"] > 0
        if scheme != "hbm":
            assert peaks["dram"] == reservation["dram"]
    finally:
        for hook in hooks:
            hook.remove()
        backend.release_session(session)
        control.release_session(expected_cache)
    assert backend.session_bytes(session) == {"hbm": 0, "dram": 0}


def test_dense_prefetch_failure_restores_attention_and_prefix(monkeypatch):
    config = tiny_config(max_position_embeddings=192, num_hidden_layers=4)
    model = initialized_sparse_model(config)
    backend = NosaServingBackend(model, "dense_prefetch", chunk_size=40)
    tokens = torch.arange(128) % config.vocab_size
    session = backend.create_session(192)
    original = model.main_attention
    try:
        backend.prefill(session, tokens[:79])
        expected = backend.extend(session, tokens[79:111]).clone()
        backend.truncate(session, 79)
        original_forward = model.model.layers[2].forward

        def fail(*args, **kwargs):
            raise RuntimeError("injected layer failure")

        with monkeypatch.context() as patch:
            patch.setattr(model.model.layers[2], "forward", fail)
            with pytest.raises(RuntimeError, match="injected"):
                backend.extend(session, tokens[79:111])
        assert model.main_attention is original
        assert model.model.layers[2].forward == original_forward
        assert session.length == session.indexer_cache.length == 79
        actual = backend.extend(session, tokens[79:111])
        torch.testing.assert_close(actual, expected)
    finally:
        backend.release_session(session)


def test_dense_prefetch_rejects_unordered_layers():
    config = tiny_config(max_position_embeddings=256, num_hidden_layers=4)
    cache = NosaDensePrefetchCache(config, 128, device="cpu", dtype=torch.float32)
    try:
        cache.begin_step(8)
        with pytest.raises(RuntimeError, match="increasing order"):
            cache.write_layer(1)
        cache.abort_step()
    finally:
        cache.release()


@pytest.mark.parametrize("capacity,prefix", [(0, 1), (257, 1), (128, 0), (128, 128)])
def test_reservation_rejects_invalid_geometry(capacity, prefix):
    backend = NosaServingBackend(initialized_sparse_model(), "hbm")
    with pytest.raises(ValueError):
        backend.estimate_session_bytes(capacity, prefix)


@torch.inference_mode()
def test_cuda_serving_checkpoint_independent_prefixes_and_revisits():
    """Explicit opt-in checks every hidden element after all 32 checkpoint layers.

    CPU tests do not opt in. Set NOSA_SERVING_CHECKPOINT for an explicit GPU
    check; unavailable hardware then fails instead of yielding a skip.
    """
    checkpoint = os.environ.get("NOSA_SERVING_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set NOSA_SERVING_CHECKPOINT for full NOSA serving correctness")
    assert torch.cuda.is_available(), "Explicit NOSA serving check requires CUDA"
    assert torch.cuda.get_device_capability() == (9, 0), "NOSA serving requires Hopper"
    prefix = int(os.environ.get("NOSA_SERVING_PREFIX_TOKENS", "8192"))
    suffix = int(os.environ.get("NOSA_SERVING_SUFFIX_TOKENS", "128"))
    capacity = prefix + suffix
    control = NosaServingBackend.from_pretrained(
        checkpoint, scheme="hbm", device="cuda:0", max_seq_len=capacity
    )
    tokens = (torch.arange(capacity, device="cuda:0") * 19 + 137) % control.config.vocab_size
    expected = []
    for scheme in NosaServingBackend.schemes:
        backend = NosaServingBackend(control.model, scheme, chunk_size=1024)
        session = backend.create_session(capacity)
        reservation = backend.estimate_session_bytes(capacity, prefix)
        try:
            backend.prefill(session, tokens[:prefix])
            for visit in range(2):
                candidate = (tokens[prefix:] + visit * 11) % control.config.vocab_size
                actual = backend.extend(session, candidate)
                backend.synchronize()
                assert actual.shape == (suffix, control.config.hidden_size)
                assert torch.isfinite(actual).all()
                for name, value in backend.session_bytes(session).items():
                    assert value <= reservation[name], (scheme, name, value, reservation)
                if scheme == "hbm":
                    expected.append(actual.cpu())
                else:
                    torch.testing.assert_close(
                        actual.cpu(), expected[visit], atol=0.016, rtol=0.016
                    )
                print(
                    f"NOSA serving correctness: scheme={scheme}, visit={visit}, "
                    f"equal={torch.equal(actual.cpu(), expected[visit])}, "
                    f"max_abs={(actual.cpu().float() - expected[visit].float()).abs().max().item()}",
                    flush=True,
                )
                backend.truncate(session, prefix)
                assert session.length == session.indexer_cache.length == prefix
        finally:
            backend.release_session(session)
