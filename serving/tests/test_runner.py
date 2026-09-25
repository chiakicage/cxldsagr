"""Generated-request boundaries and independent session lifetime on CPU."""

import json
import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from executor.model_executor import ModelExecutor
from serving.run_gr import _build_parser, main
from serving.runner import GRRunner


class RecordingExecutor:
    max_seq_len = 16

    def __init__(self, *, fail=None):
        self.events = []
        self.sessions = []
        self.fail = fail
        self.hidden_ref = None

    def allocate(self, capacity):
        cache = SimpleNamespace(length=0, capacity=capacity, ordinal=len(self.sessions))
        self.sessions.append(cache)
        self.events.append(("allocate", cache.ordinal, capacity))
        return cache

    def prefill(self, ids, cache):
        self.events.append(("prefill", cache.ordinal, ids.tolist()))
        assert cache.length == 0
        if self.fail == "prefill":
            raise RuntimeError("prefill failed")
        cache.length += ids.numel()
        return ids[:, None].float()

    def extend(self, ids, cache):
        self.events.append(("extend", cache.ordinal, ids.tolist()))
        assert cache.length > 0
        if self.fail == "extend":
            raise RuntimeError("extend failed")
        cache.length += ids.numel()
        hidden = torch.stack((ids, ids + 100), dim=-1).float()
        self.hidden_ref = weakref.ref(hidden)
        return hidden

    def release(self, cache):
        self.events.append(("release", cache.ordinal))


def request(**changes):
    row = {
        "model": "nosa",
        "task_id": 0,
        "user_id": 3,
        "timestamp": 1000000.0,
        "prompt": "already encoded; do not tokenize",
        "input_ids": [5, 6, 7, 8, 9, 10],
        "attention_mask": [1] * 6,
        "stable_prefix_tokens": 4,
        "instruction_tokens": 1,
        "user_tokens": 3,
        "item_tokens": 3,
        "candidate_suffix_tokens": 2,
        "total_input_tokens": 6,
        "history_token_span": [1, 4],
        "candidate_token_span": [4, 6],
    }
    row.update(changes)
    return row


def test_runner_uses_original_ids_and_releases_each_request_before_yield():
    executor = RecordingExecutor()
    requests = iter((request(), request(task_id=1, timestamp=2000000.0)))
    results = GRRunner(executor, device="cpu").run(requests)
    first = next(results)
    assert executor.events == [
        ("allocate", 0, 6),
        ("prefill", 0, [5, 6, 7, 8]),
        ("extend", 0, [9, 10]),
        ("release", 0),
    ]
    assert first.last_hidden.tolist() == [10, 110]
    assert first.last_hidden.untyped_storage().nbytes() == first.last_hidden.nbytes
    assert executor.hidden_ref() is None
    assert first.metadata["timestamp"] == 1000000.0
    assert not {"prompt", "input_ids", "attention_mask"}.intersection(first.metadata)
    second = next(results)
    assert second.metadata["task_id"] == 1
    assert executor.sessions[0] is not executor.sessions[1]
    assert executor.events[4:] == [
        ("allocate", 1, 6),
        ("prefill", 1, [5, 6, 7, 8]),
        ("extend", 1, [9, 10]),
        ("release", 1),
    ]
    with pytest.raises(StopIteration):
        next(results)


@pytest.mark.parametrize("phase", ("prefill", "extend"))
def test_model_failure_releases_session_and_propagates_without_retry(phase):
    executor = RecordingExecutor(fail=phase)
    iterator = GRRunner(executor, device="cpu").run([request(), request(task_id=1)])
    with pytest.raises(RuntimeError, match=f"{phase} failed"):
        next(iterator)
    assert executor.events[-1] == ("release", 0)
    assert len(executor.sessions) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"model": "deepseek_v32"},
        {"input_ids": []},
        {"input_ids": [[1, 2], [3, 4]]},
        {"input_ids": [1.0, 2.0]},
        {"input_ids": [True, False]},
        {"input_ids": [1, -2]},
        {"input_ids": list(range(17))},
        {"stable_prefix_tokens": 0},
        {"stable_prefix_tokens": 6},
        {"stable_prefix_tokens": True},
        {"stable_prefix_tokens": 3.5},
        {"total_input_tokens": 7},
        {"candidate_suffix_tokens": 3},
        {"instruction_tokens": 2},
        {"user_tokens": 2},
        {"item_tokens": 2},
        {"history_token_span": [1, 3]},
        {"history_token_span": [0, 4]},
        {"candidate_token_span": [3, 6]},
        {"candidate_token_span": [4, 7]},
        {"history_token_span": "14"},
        {"attention_mask": [1, 1, 1, 1, 1, 0]},
        {"attention_mask": [1]},
        {"common_prefix_tokens": 7},
    ],
)
def test_invalid_request_is_rejected_before_allocation(change):
    executor = RecordingExecutor()
    with pytest.raises(ValueError):
        next(GRRunner(executor, device="cpu").run([request(**change)]))
    assert executor.events == []


def test_minimal_request_defaults_model_and_preserves_input():
    executor = RecordingExecutor()
    row = {"input_ids": [1, 2], "stable_prefix_tokens": 1}
    result = next(GRRunner(executor, device="cpu").run([row]))
    assert result.metadata == {
        "model": "nosa",
        "total_input_tokens": 2,
        "stable_prefix_tokens": 1,
        "candidate_suffix_tokens": 1,
    }
    assert row == {"input_ids": [1, 2], "stable_prefix_tokens": 1}


def test_partial_length_metadata_is_checked_without_spans():
    executor = RecordingExecutor()
    row = {"input_ids": [1, 2, 3, 4], "stable_prefix_tokens": 3, "user_tokens": 2, "item_tokens": 3}
    with pytest.raises(ValueError, match=r"user_tokens \+"):
        next(GRRunner(executor, device="cpu").run([row]))
    assert executor.events == []


def test_runner_with_generic_executor_only_requests_hidden():
    events = []

    class Model:
        config = SimpleNamespace(max_position_embeddings=16)

        def new_cache(self, capacity):
            return SimpleNamespace(length=0, max_seq_len=capacity)

        def __call__(self, ids, cache, *, return_hidden):
            assert return_hidden
            events.append(ids.tolist())
            cache.length += ids.numel()
            return ids[:, None].float()

    runner = GRRunner(ModelExecutor(Model(), chunk_size=3), device="cpu")
    result = next(runner.run([request()]))
    assert events == [[5, 6, 7], [8], [9, 10]]
    assert result.last_hidden.tolist() == [10]


def test_real_nosa_runner_matches_full_dense_hidden_and_releases_cache(monkeypatch):
    from models.nosa.tests.test_model import initialized_model

    model = initialized_model()
    rows = [request(), request(task_id=1, input_ids=[5, 6, 7, 8, 15, 16])]
    expected = [model(torch.tensor(row["input_ids"]), return_hidden=True)[-1] for row in rows]
    sessions = []
    allocate = model.cache_manager.allocate

    def record_allocate(capacity):
        session = allocate(capacity)
        sessions.append(session)
        return session

    def reject_lm_head(*args, **kwargs):
        raise AssertionError("GR execution must skip the LM head")

    monkeypatch.setattr(model.cache_manager, "allocate", record_allocate)
    monkeypatch.setattr(model.lm_head, "forward", reject_lm_head)
    results = list(GRRunner(ModelExecutor(model, chunk_size=3), device="cpu").run(rows))
    for result, reference in zip(results, expected, strict=True):
        torch.testing.assert_close(result.last_hidden, reference, atol=2e-6, rtol=2e-5)
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert all(session.released and session.stats()["resident_bytes"] == 0 for session in sessions)


def test_cli_defaults_and_help_without_runtime_imports():
    args = _build_parser().parse_args([])
    assert (args.device, args.dtype, args.count, args.num_users, args.seed) == (
        "cuda:0",
        "bfloat16",
        1,
        1000,
        42,
    )
    assert args.prefill_chunk_size == 1024
    assert args.user_lengths is None and args.item_lengths is None
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; from serving.run_gr import main; "
        "assert 'torch' not in sys.modules; "
        "assert 'GR.input_generator' not in sys.modules; "
        "main(['--help'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert "--user-lengths" in result.stdout


def test_cli_streams_summary_without_feature_vectors(monkeypatch, capsys):
    import GR.input_generator as generator_module
    import models.nosa.model as model_module

    class Model:
        config = SimpleNamespace(max_position_embeddings=32768)

    class FakeRunner:
        def __init__(self, executor, *, device):
            assert executor.chunk_size == 5
            assert str(device) == "cuda:0"

        def run(self, requests):
            assert list(requests) == [request()]
            yield SimpleNamespace(metadata={"task_id": 0}, last_hidden=torch.zeros(3))

    import serving.runner as runner_module

    monkeypatch.setattr(model_module.NosaConfig, "from_pretrained", lambda path: Model.config)
    monkeypatch.setattr(
        model_module.NosaForCausalLM, "from_pretrained", lambda *args, **kwargs: Model()
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    def create_generator(**kwargs):
        assert kwargs["tokenizer"] == Path("/some/model/tokenizer.json")
        assert kwargs["schedule_config"].seed == 42
        assert kwargs["text_config"].user_lengths == (128,)
        assert kwargs["text_config"].item_lengths == (256,)
        return SimpleNamespace(iter_generate=lambda count: [request()])

    monkeypatch.setattr(generator_module, "create_input_generator", create_generator)
    monkeypatch.setattr(runner_module, "GRRunner", FakeRunner)
    main(
        [
            "--model-path",
            "/some/model",
            "--device",
            "cuda",
            "--prefill-chunk-size",
            "5",
            "--user-lengths",
            "128",
            "--item-lengths",
            "256",
        ]
    )
    row = json.loads(capsys.readouterr().out)
    assert row == {
        "status": "completed",
        "metadata": {"task_id": 0},
        "feature_shape": [3],
        "feature_dtype": "torch.float32",
        "feature_device": "cpu",
    }
