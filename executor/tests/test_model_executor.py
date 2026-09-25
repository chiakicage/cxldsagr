"""Chunk boundaries, output selection and cache ownership without GPU kernels."""

from types import SimpleNamespace

import pytest
import torch

from executor.model_executor import ModelExecutor, run_chunks


class RecordingModel:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=32)
        self.calls = []

    def new_cache(self, max_seq_len):
        return SimpleNamespace(length=0, max_seq_len=max_seq_len)

    def __call__(self, ids, cache, *, return_hidden=False, logits_to_keep=0):
        assert not torch.is_grad_enabled()
        self.calls.append((ids.tolist(), return_hidden, logits_to_keep))
        cache.length += ids.numel()
        result = torch.stack((ids, ids + 100), dim=-1).float()
        return result[-logits_to_keep:] if logits_to_keep else result


def test_run_chunks_consumes_every_token_and_returns_last_chunk_features():
    model = RecordingModel()
    cache = model.new_cache(6)
    hidden = run_chunks(model, torch.arange(6), cache, 4)
    assert model.calls == [([0, 1, 2, 3], True, 0), ([4, 5], True, 0)]
    assert cache.length == 6
    torch.testing.assert_close(hidden, torch.tensor([[4, 104], [5, 105]]).float())


def test_prefill_extend_and_single_token_decode_share_output_selection():
    model = RecordingModel()
    executor = ModelExecutor(model, chunk_size=2)
    cache = executor.allocate(8)
    with pytest.raises(ValueError, match="nonempty"):
        executor.extend(torch.tensor([1]), cache)
    logits = executor.prefill(torch.tensor([1, 2, 3]), cache, output="logits", logits_to_keep=1)
    assert logits.tolist() == [[3, 103]]
    with pytest.raises(ValueError, match="empty"):
        executor.prefill(torch.tensor([4]), cache)
    hidden = executor.extend(torch.tensor([4, 5]), cache)
    assert hidden.shape == (2, 2)
    logits = executor.extend(torch.tensor([6]), cache, output="logits", logits_to_keep=1)
    assert logits.tolist() == [[6, 106]]
    assert model.calls == [
        ([1, 2], False, 1),
        ([3], False, 1),
        ([4, 5], True, 0),
        ([6], False, 1),
    ]
    assert cache.length == 6
    executor.release(cache)  # Legacy fake models own no manager resources.


def test_executor_uses_injected_manager_for_lifecycle():
    model = RecordingModel()
    events = []
    cache = SimpleNamespace(length=0)
    manager = SimpleNamespace(
        allocate=lambda capacity: events.append(("allocate", capacity)) or cache,
        release=lambda session: events.append(("release", session)),
    )
    executor = ModelExecutor(model, cache_manager=manager)
    assert executor.allocate(12) is cache
    executor.release(cache)
    assert events == [("allocate", 12), ("release", cache)]
    model.cache_manager = manager
    assert ModelExecutor(model).cache_manager is manager


@pytest.mark.parametrize(("capacity", "context"), [(3, 32), (8, 3)])
def test_full_input_overflow_leaves_cache_empty_for_valid_prefill_retry(capacity, context):
    model = RecordingModel()
    model.config.max_position_embeddings = context
    executor = ModelExecutor(model, chunk_size=2)
    cache = executor.allocate(capacity)
    with pytest.raises(ValueError, match="capacity|max_position_embeddings"):
        executor.prefill(torch.tensor([1, 2, 3, 4]), cache)
    assert cache.length == 0
    assert model.calls == []
    hidden = executor.prefill(torch.tensor([1, 2, 3]), cache)
    assert cache.length == 3
    assert hidden.tolist() == [[3, 103]]


@pytest.mark.parametrize(
    ("ids", "chunk_size", "kwargs", "message"),
    [
        (torch.tensor([], dtype=torch.long), 2, {}, "nonempty"),
        (torch.tensor([[1, 2]]), 2, {}, "1-D"),
        (torch.tensor([1.5]), 2, {}, "torch.long"),
        (torch.tensor([1]), 0, {}, "chunk_size"),
        (torch.tensor([1]), True, {}, "chunk_size"),
        (torch.tensor([1]), 2, {"output": "other"}, "output"),
        (torch.tensor([1]), 2, {"logits_to_keep": -1}, "nonnegative"),
        (torch.tensor([1]), 2, {"logits_to_keep": 1}, "hidden output"),
    ],
)
def test_invalid_execution_inputs_fail_before_model_call(ids, chunk_size, kwargs, message):
    model = RecordingModel()
    with pytest.raises(ValueError, match=message):
        run_chunks(model, ids, model.new_cache(10), chunk_size, **kwargs)
    assert not model.calls


@pytest.mark.parametrize("fail", [False, True])
def test_generation_releases_real_cache_after_completion_or_model_error(monkeypatch, fail):
    from models.nosa.infer import generate
    from models.nosa.tests.test_model import initialized_model

    model = initialized_model()
    allocate = model.cache_manager.allocate
    sessions = []

    def record_allocate(capacity):
        session = allocate(capacity)
        sessions.append(session)
        return session

    monkeypatch.setattr(model.cache_manager, "allocate", record_allocate)
    if fail:

        def fail_attention(*args):
            raise RuntimeError("attention failed")

        model.attention = fail_attention
        with pytest.raises(RuntimeError, match="attention failed"):
            generate(model, torch.tensor([1, 2, 3]), max_new_tokens=2, prefill_chunk_size=2)
    else:
        generated, _ = generate(
            model, torch.tensor([1, 2, 3]), max_new_tokens=2, prefill_chunk_size=2
        )
        assert generated
    assert len(sessions) == 1
    assert sessions[0].released and sessions[0].stats()["resident_bytes"] == 0
