"""A single supplied step must not be passed off as a complete GR candidate."""

from pathlib import Path
from types import SimpleNamespace

from experiments.deepseek_v32_mfu.src import measure


def test_single_token_reuses_history_and_records_its_source(monkeypatch):
    generated = []
    source = {
        "model": "deepseek_v32",
        "input_ids": list(range(512 + 128)),
        "stable_prefix_tokens": 512,
        "candidate_suffix_tokens": 128,
        "history_sha256": "history-text-hash",
        "history_token_span": [7, 512],
        "prompt": "complete synthetic GR prompt",
        "candidate_item_ids": [123],
    }
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: SimpleNamespace(ids=[0] * 7))
    monkeypatch.setattr(measure, "Tokenizer", SimpleNamespace(from_file=lambda path: tokenizer))

    def generator(**kwargs):
        generated.append(kwargs)
        return SimpleNamespace(iter_generate=lambda count: iter([source]))

    monkeypatch.setattr(measure, "create_input_generator", generator)
    request = measure.make_request(Path("/fixture"), 512, 1, 42)
    assert len(generated) == 1
    assert generated[0]["text_config"].item_lengths == (128 + 7,)
    assert request["input_ids"] == source["input_ids"][:513]
    assert request["candidate_token_span"] == [512, 513]
    assert request["candidate_suffix_tokens"] == 1
    assert request["total_input_tokens"] == 513
    assert request["history_sha256"] == source["history_sha256"]
    assert request["history_token_span"] == source["history_token_span"]
    assert request["token_source"]["token_id"] == 512
    assert request["token_source"]["sampling_included"] is False
    assert "prompt" not in request and "candidate_item_ids" not in request
    assert len(source["input_ids"]) == 640
