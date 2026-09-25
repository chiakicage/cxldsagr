"""Small-checkpoint checks across GR, serving, execution, cache, and CLI boundaries."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from executor.model_executor import ModelExecutor
from GR.input_generator import TextConfig, create_input_generator
from models.nosa.model import NosaForCausalLM
from models.nosa.tests.test_model import (
    dense_attention,
    initialized_model,
    tiny_config,
    write_checkpoint,
)
from serving.runner import GRRunner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def small_checkpoint(tmp_path):
    vocabulary = {"<unk>": 0, "hello": 1, "<|im_start|>": 2, "<|im_end|>": 3}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    config = tiny_config(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=192,
        max_position_embeddings=256,
        vocab_size=len(vocabulary),
        eos_token_id=(),
    )
    write_checkpoint(tmp_path, config, dict(initialized_model(config).state_dict()))
    return tmp_path, config


@torch.inference_mode()
def test_generated_requests_use_independent_loaded_model_sessions(small_checkpoint, monkeypatch):
    checkpoint, config = small_checkpoint
    generator = create_input_generator(
        tokenizer=checkpoint / "tokenizer.json",
        num_users=1,
        text_config=TextConfig(
            user_lengths=(64,),
            user_probabilities=(1,),
            item_lengths=(128,),
            item_probabilities=(1,),
            max_input_tokens=config.max_position_embeddings,
        ),
    )
    requests = list(generator.iter_generate(2))
    model = NosaForCausalLM.from_pretrained(
        checkpoint, device="cpu", dtype=torch.float32, attention=dense_attention
    )
    sessions = []
    allocate = model.cache_manager.allocate

    def record_allocate(capacity):
        session = allocate(capacity)
        sessions.append(session)
        return session

    def reject_lm_head(*args, **kwargs):
        raise AssertionError("GR features must not execute the LM head")

    monkeypatch.setattr(model.cache_manager, "allocate", record_allocate)
    monkeypatch.setattr(model.lm_head, "forward", reject_lm_head)
    results = GRRunner(ModelExecutor(model, chunk_size=64), device="cpu").run(requests)
    for index, result in enumerate(results):
        assert result.metadata["task_id"] == index
        assert result.metadata["total_input_tokens"] == 192
        assert result.last_hidden.shape == (config.hidden_size,)
        assert torch.isfinite(result.last_hidden).all()
        assert len(sessions) == index + 1
        assert sessions[index].released
        assert sessions[index].stats()["resident_bytes"] == 0
    assert len(sessions) == 2 and sessions[0] is not sessions[1]


def run_cli(*arguments):
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CLI execution")
def test_cuda_serving_cli_completes_two_requests(small_checkpoint):
    checkpoint, config = small_checkpoint
    completed = run_cli(
        "-m",
        "serving.run_gr",
        "--model-path",
        str(checkpoint),
        "--device",
        "cuda",
        "--count",
        "2",
        "--num-users",
        "1",
        "--user-lengths",
        "64",
        "--item-lengths",
        "128",
        "--prefill-chunk-size",
        "64",
    )
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(row["status"] == "completed" for row in rows)
    assert all(row["feature_shape"] == [config.hidden_size] for row in rows)
    assert all(row["feature_device"].startswith("cuda:") for row in rows)
    assert all(row["metadata"]["total_input_tokens"] == 192 for row in rows)
    assert [row["metadata"]["task_id"] for row in rows] == [0, 1]
    assert rows[0]["metadata"]["user_id"] == rows[1]["metadata"]["user_id"]
    assert rows[1]["metadata"]["previous_request_id"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CLI execution")
def test_cuda_direct_generation_cli_obeys_token_budget(small_checkpoint):
    checkpoint, _ = small_checkpoint
    completed = run_cli(
        "models/nosa/infer.py",
        "--model-path",
        str(checkpoint),
        "--device",
        "cuda",
        "--prompt",
        "hello hello hello",
        "--raw-prompt",
        "--max-new-tokens",
        "4",
        "--prefill-chunk-size",
        "2",
    )
    stats = next(
        json.loads(line) for line in reversed(completed.stderr.splitlines()) if line.startswith("{")
    )
    assert stats["prompt_tokens"] == 3
    assert stats["generated_tokens"] == 4
    assert stats["decode_steps"] == 3
    assert not stats["stopped_on_eos"]
