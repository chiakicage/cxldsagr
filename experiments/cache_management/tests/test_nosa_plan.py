"""Offline NOSA declarations distinguish quota, storage, charge and graph limits."""

import json
from dataclasses import asdict

import pytest
import torch

from experiments.cache_management.src import nosa_plan
from models.nosa.config import NosaConfig


def config():
    return NosaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=32000,
        max_position_embeddings=262144,
    )


def forbid_runtime_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline planning tried to allocate tensors or query CUDA")

    for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "full", "tensor"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in (
        "_lazy_init",
        "current_device",
        "get_device_capability",
        "mem_get_info",
        "memory_snapshot",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)


def test_offline_plan_preserves_pinned_bin_and_quota_boundaries_without_cuda(monkeypatch):
    forbid_runtime_allocation(monkeypatch)
    result = nosa_plan.build_plan(config())
    assert result["status"] == "static_plan_not_execution"
    assert not result["assumptions"]["live_allocator_validated"]
    assert result["graph"]["chosen_private_limit_bytes"] == 8 * 2**30
    assert result["graph"]["static_allocation_limit_bytes"] is None
    assert result["graph"]["private_reserved_bytes"] is None
    hbm = result["cases"]["hbm"]
    assert hbm["quota"]["full_history_sessions"] == 1
    assert not hbm["populations"]["requested_users_if_all_retained"]["admissible_under_token_quota"]
    assert hbm["populations"]["admissible_requested_users"]["sessions"] == 1
    for scheme in ("serial_sparse", "dense_prefetch", "overlap"):
        case = result["cases"][scheme]
        host = case["one_session"]["categories"]["host_history"]
        # Two actual 1-GiB H histories; the preserved H+A bound enters two
        # independent 2-GiB pinned bins. NH remains a lazy quota.
        assert host["storage_payload_bound"]["dram"] == 2 * 2**30
        assert host["allocator_reservation_bound"]["dram"] == 4 * 2**30
        assert case["quota"]["full_history_sessions"] == 256
        assert case["populations"]["admissible_requested_users"]["sessions"] == 16
        assert case["populations"]["full_token_quota"]["sessions"] == 256
        assert not case["populations"]["full_token_quota"]["physical_fit_validated"]
        assert case["shared"]["storage_payload_bound"]["dram"] == 0
        declared = case["shared"]["allocations"]
        if scheme != "dense_prefetch":
            assert {item["alias_of"] for item in declared if item["alias_of"]} == {
                "pool.keys",
                "pool.values",
            }


def test_cli_reads_only_config_and_publishes_source_bound_plan(monkeypatch, tmp_path):
    model = tmp_path / "checkpoint"
    model.mkdir()
    values = asdict(config())
    values["max_position_embeddings"] = 32768
    (model / "config.json").write_text(json.dumps(values))
    output = tmp_path / "offline_plan"
    forbid_runtime_allocation(monkeypatch)
    result = nosa_plan.main(
        [
            "--run-id",
            "nosa_static",
            "--model-path",
            str(model),
            "--output-dir",
            str(output),
        ]
    )
    saved = json.loads((output / "plan.json").read_text())
    assert saved["config_source"] == result["config_source"]
    assert set(saved["sources"]) == set(nosa_plan.SOURCE_PATHS)
    assert saved["parameters"]["requested_users"] == 16
    assert saved["parameters"]["context_limit"] == 65664
    assert saved["assumptions"]["checkpoint_context_limit"] == 32768
    assert saved["model_config"]["max_position_embeddings"] == 65664
    assert saved["assumptions"]["long_context_quality_validated"] is False
    with pytest.raises(FileExistsError):
        nosa_plan.main(["--run-id", "nosa_static", "--output-dir", str(output)])


@pytest.mark.parametrize(
    "changes", [{"history": 65}, {"candidate": 0}, {"pool": 32768}, {"host": 0}, {"users": True}]
)
def test_invalid_fixed_geometry_is_rejected(changes):
    with pytest.raises(ValueError):
        nosa_plan.build_plan(config(), **changes)
