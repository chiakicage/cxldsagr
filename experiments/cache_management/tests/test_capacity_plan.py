"""CPU-only CLI verification for history-only capacity planning metadata."""

import json

import torch

from experiments.cache_management.src import capacity_plan


def test_supplied_memory_plan_records_transient_candidate_semantics_without_cuda(
    monkeypatch, tmp_path
):
    from models.deepseek_v32.config import Config

    cfg = {
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "index_topk": 16,
        "max_seq_len": 4096,
    }
    monkeypatch.setattr(Config, "from_checkpoint", lambda _: cfg)
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    def forbidden(*_, **__):
        raise AssertionError("supplied-memory static planning must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    output = tmp_path / "plan"
    capacity_plan.main(
        [
            "--run-id",
            "history_only",
            "--output-dir",
            str(output),
            "--model-path",
            str(tmp_path),
            "--dram-budget-gib",
            "1",
            "--total-hbm-gib",
            "2",
            "--model-hbm-gib",
            "1",
            "--fixed-p",
            "32",
            "--history-tokens",
            "64",
            "--candidate-tokens",
            "8",
            "--chunk-size",
            "8",
        ]
    )
    saved = json.loads((output / "plan.json").read_text())
    assert saved["status"] == "static_plan_not_execution"
    assert (
        saved["candidate_persistence"]
        == saved["parameters"]["candidate_persistence"]
        == "gpu_transient"
    )
    point = saved["plan"]["selected"]
    assert point["session_capacity_tokens"] == point["session_page_tokens"] == 64
    assert point["execution_context_tokens"] == 72
    assert point["full_sessions"] == point["host_arena_tokens"] // 64
    assert point["candidate_host_tokens"] == point["candidate_device_to_host_bytes"] == 0
    assert "models/deepseek_v32/execution/adapter.py" in saved["sources"]
    assert "models/deepseek_v32/execution/planning.py" in saved["sources"]
    assert (output / "checkpoint_config.json").read_bytes() == (
        tmp_path / "config.json"
    ).read_bytes()
