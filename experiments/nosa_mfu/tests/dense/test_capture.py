"""Regression checks for the actual 65536 KV + 1024 query execution boundary."""

from types import SimpleNamespace

import pytest
import torch

from experiments.nosa_mfu.src.dense.capture import audit_extend, execution_split


def test_execution_split_matches_gr_semantic_boundary():
    ids = list(range(66560))
    request = {"input_ids": ids, "stable_prefix_tokens": 65536, "candidate_suffix_tokens": 1024}
    prefix, new = execution_split(request, 65536, 1024)
    assert len(prefix) == 65536
    assert len(new) == 1024
    assert new[0] == 65536
    assert prefix + new == ids


def test_execution_split_rejects_wrong_total():
    with pytest.raises(ValueError, match="input length"):
        execution_split({"input_ids": list(range(66559))}, 65536, 1024)


class ShapeModel:
    config = SimpleNamespace(num_hidden_layers=2, max_position_embeddings=66560)

    def attention(self, q, k, v):
        return q

    def __call__(self, ids, cache, *, return_hidden):
        assert return_hidden
        for _ in range(self.config.num_hidden_layers):
            self.attention(
                torch.empty((len(ids), 32, 128), device="meta"),
                torch.empty((cache.length + len(ids), 2, 128), device="meta"),
                torch.empty((cache.length + len(ids), 2, 128), device="meta"),
            )
        cache.length += len(ids)
        return torch.empty((len(ids), 4096), device="meta")


def test_audit_checks_attention_and_restores_callable():
    model = ShapeModel()
    cache = SimpleNamespace(length=65564, max_seq_len=66560)
    calls, _ = audit_extend(model, torch.empty(1024, dtype=torch.long), cache, 65536, 1024)
    assert cache.length == 66560
    assert len(calls) == 2
    assert all(row["q"] == [1024, 32, 128] for row in calls)
    assert all(row["k"] == row["v"] == [66560, 2, 128] for row in calls)
    assert model.attention.__func__ is ShapeModel.attention


def test_audit_rejects_splitting_extend_into_smaller_chunks():
    with pytest.raises(ValueError, match="one chunk"):
        audit_extend(ShapeModel(), torch.empty(1024), SimpleNamespace(length=0), 65536, 512)


def test_audit_rejects_actual_wrong_kv_length():
    class WrongKVModel(ShapeModel):
        def __call__(self, ids, cache, *, return_hidden):
            cache.length += 28
            return super().__call__(ids, cache, return_hidden=return_hidden)

    model = WrongKVModel()
    with pytest.raises(ValueError, match="actual attention shapes"):
        audit_extend(
            model,
            torch.empty(1024, dtype=torch.long),
            SimpleNamespace(length=0, max_seq_len=66560),
            65536,
            1024,
        )
    assert model.attention.__func__ is ShapeModel.attention


def test_execution_split_rejects_moving_history_into_new_tokens():
    request = {
        "input_ids": list(range(66560)),
        "stable_prefix_tokens": 65564,
        "candidate_suffix_tokens": 996,
    }
    with pytest.raises(ValueError, match="semantic boundary"):
        execution_split(request, 65536, 1024)


def test_audit_runs_through_framework_executor_and_cache():
    from executor.model_executor import run_chunks
    from models.nosa.tests.test_model import initialized_model, tiny_config

    config = tiny_config()
    model = initialized_model(config)
    tokens = torch.tensor([1, 7, 6, 3, 2, 8])
    cache = model.new_cache(len(tokens))
    try:
        run_chunks(model, tokens[:4], cache, 2)
        calls, hidden = audit_extend(model, tokens[4:], cache, 4, 2)
        assert hidden.shape == (2, config.hidden_size)
        assert cache.length == len(tokens)
        assert len(calls) == config.num_hidden_layers
        assert all(row["q"] == [2, config.num_attention_heads, config.head_dim] for row in calls)
        assert all(
            row["k"] == row["v"] == [6, config.num_key_value_heads, config.head_dim]
            for row in calls
        )
    finally:
        model.cache_manager.release(cache)


@pytest.mark.parametrize("problem", ["earlier_row", "nonfinite", "missing_row"])
def test_candidate_hidden_acceptance_checks_more_than_the_last_token(problem):
    from experiments.nosa_mfu.src.dense.capture import compare_candidate_hidden

    full = torch.ones((4, 8), dtype=torch.bfloat16)
    split = full.clone()
    if problem == "earlier_row":
        split[0, 0] = 2
    elif problem == "nonfinite":
        split[0, 0] = float("nan")
    else:
        split = split[-1:]
    assert torch.equal(full[-1], split[-1])
    with pytest.raises(ValueError):
        compare_candidate_hidden(full, split, query_tokens=4, hidden_size=8)


def test_candidate_hidden_acceptance_reports_the_full_compared_shape():
    from experiments.nosa_mfu.src.dense.capture import compare_candidate_hidden

    output = torch.ones((4, 8), dtype=torch.bfloat16)
    checks = compare_candidate_hidden(output, output.clone(), query_tokens=4, hidden_size=8)
    assert checks == {
        "finite": True,
        "candidate_hidden_max_abs": 0.0,
        "compared_shape": [4, 8],
        "scope": "all_candidate_hidden",
    }


def dense_profile_metadata():
    from dataclasses import asdict

    from models.nosa.config import NosaConfig

    config = NosaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=128,
        eos_token_id=(2, 127),
    )
    return {
        "args": {
            "mode": "profile",
            "prefix_tokens": 64,
            "new_tokens": 64,
            "chunk_size": 64,
            "warmup": 2,
            "repeats": 5,
        },
        "model_config": asdict(config),
        "checkpoint_path": "/model",
        "checkpoint_files": {"weights.safetensors": {"size": 12, "mtime_ns": 1}},
        "checkpoint_config_sha256": "model-config",
        "request_sha256": "request",
        "source_sha256": {"model.py": "source"},
        "torch": "torch",
        "cuda": "cuda",
        "flashinfer": "flashinfer",
        "gpu": {"uuid": "device-a", "capability": [9, 0]},
        "runtime_settings": {"matmul_allow_tf32": False},
        "native_artifacts": {"library.so": {"sha256": "binary"}},
    }


def test_dense_profile_accepts_model_config_json_roundtrip():
    import json

    from experiments.nosa_mfu.src.dense.capture import verify_benchmark_identity

    current = dense_profile_metadata()
    benchmark = json.loads(json.dumps(current))
    benchmark["args"].update(mode="bench", repeats=3)
    assert isinstance(current["model_config"]["eos_token_id"], tuple)
    assert isinstance(benchmark["model_config"]["eos_token_id"], list)
    verify_benchmark_identity(benchmark, current)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("args", "mode"), "profile"),
        (("args", "prefix_tokens"), 128),
        (("model_config", "eos_token_id"), [2, 126]),
        (("checkpoint_files", "weights.safetensors", "mtime_ns"), 2),
        (("request_sha256",), "other-request"),
        (("source_sha256", "model.py"), "other-source"),
        (("gpu", "uuid"), "device-b"),
        (("runtime_settings", "matmul_allow_tf32"), True),
        (("native_artifacts", "library.so", "sha256"), "other-binary"),
    ],
)
def test_dense_profile_rejects_changed_benchmark_execution(path, value):
    import json

    from experiments.nosa_mfu.src.dense.capture import verify_benchmark_identity

    current = dense_profile_metadata()
    benchmark = json.loads(json.dumps(current))
    benchmark["args"]["mode"] = "bench"
    target = benchmark
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="same independently benchmarked dense execution"):
        verify_benchmark_identity(benchmark, current)
