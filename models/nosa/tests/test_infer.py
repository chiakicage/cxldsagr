"""Generation boundaries and local tokenizer behavior without model downloads."""

import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from models.nosa.infer import encode_prompt, generate, load_tokenizer


class ScriptedModel:
    """Predict a fixed continuation, recording every token consumed by the cache."""

    def __init__(self, prompt_length, continuation):
        self.config = SimpleNamespace(max_position_embeddings=32, eos_token_id=(2, 73440))
        self.prompt_length = prompt_length
        self.continuation = continuation
        self.calls = []
        self.cache = None

    def eval(self):
        return self

    def new_cache(self, max_seq_len):
        self.cache = SimpleNamespace(length=0, max_seq_len=max_seq_len, tokens=[])
        return self.cache

    def __call__(self, input_ids, cache, *, logits_to_keep):
        assert logits_to_keep == 1
        self.calls.append(input_ids.tolist())
        cache.tokens.extend(input_ids.tolist())
        cache.length += input_ids.numel()
        assert cache.length <= cache.max_seq_len
        step = max(0, cache.length - self.prompt_length)
        token = self.continuation[min(step, len(self.continuation) - 1)]
        logits = torch.full((1, 73448), -torch.inf)
        logits[0, token] = 0
        return logits


@pytest.mark.parametrize("eos_id", [2, 73440])
def test_generate_chunked_prefill_stops_on_each_nosa_eos(eos_id):
    prompt = torch.tensor([1, 6, 7, 8, 9])
    model = ScriptedModel(len(prompt), [13, eos_id, 19])
    generated, stats = generate(model, prompt, max_new_tokens=4, prefill_chunk_size=2)
    assert generated == [13, eos_id]
    assert model.calls == [[1, 6], [7, 8], [9], [13]]
    assert model.cache.tokens == [1, 6, 7, 8, 9, 13]
    assert stats["prompt_tokens"] == 5
    assert stats["generated_tokens"] == 2
    assert stats["decode_steps"] == 1
    assert stats["stopped_on_eos"] is True


@pytest.mark.parametrize("budget", [1, 3])
def test_generate_respects_token_budget(budget):
    prompt = torch.tensor([1, 6])
    model = ScriptedModel(len(prompt), [13, 14, 15, 16])
    generated, stats = generate(model, prompt, max_new_tokens=budget)
    assert generated == [13, 14, 15][:budget]
    assert len(model.calls) == budget
    assert model.cache.length == len(prompt) + budget - 1
    assert stats["decode_steps"] == budget - 1
    assert stats["generated_tokens"] == budget
    assert stats["stopped_on_eos"] is False


def test_generate_rejects_context_overflow_before_allocating_cache():
    model = ScriptedModel(2, [13])
    model.config.max_position_embeddings = 4
    with pytest.raises(ValueError, match="max_position_embeddings"):
        generate(model, torch.tensor([1, 6]), max_new_tokens=3)
    assert model.cache is None


@pytest.fixture
def tokenizer_files(tmp_path):
    vocabulary = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "user": 3,
        "assistant": 4,
        ":": 5,
        "hello": 6,
        "system": 7,
        "brief": 8,
        "<closed>": 9,
    }
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<s>", "</s>", "<closed>"])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)]
    )
    # Simulate tokenizer.json saved with constraints that inference must disable.
    tokenizer.enable_truncation(max_length=2)
    tokenizer.enable_padding(length=10, pad_id=2, pad_token="</s>")
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    config = {
        "bos_token": "<s>",
        "chat_template": (
            "{{ bos_token }} "
            "{% for message in messages %}{{ message.role }} : {{ message.content }} {% endfor %}"
            "{% if add_generation_prompt %}assistant : {% endif %}"
            "{% if enable_thinking is defined and not enable_thinking %}<closed>{% endif %}"
        ),
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    return tmp_path


def test_local_tokenizer_disables_padding_and_truncation(tokenizer_files):
    tokenizer, config = load_tokenizer(tokenizer_files)
    assert encode_prompt(tokenizer, config, "hello hello hello", raw_prompt=True) == [1, 6, 6, 6]


def test_chat_template_special_tokens_and_options(tokenizer_files):
    tokenizer, config = load_tokenizer(tokenizer_files)
    assert encode_prompt(tokenizer, config, "hello") == [1, 3, 5, 6, 4, 5]
    assert encode_prompt(
        tokenizer, config, "hello", system_prompt="brief", disable_thinking=True
    ) == [1, 7, 5, 8, 3, 5, 6, 4, 5, 9]
    with pytest.raises(ValueError, match="require chat mode"):
        encode_prompt(tokenizer, config, "hello", raw_prompt=True, disable_thinking=True)
