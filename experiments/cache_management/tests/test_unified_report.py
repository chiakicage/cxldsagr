"""Publication rejects altered plans and distinguishes memory observations."""

import json

import pytest

from experiments.cache_management.src import nosa_plan, report
from experiments.cache_management.tests.test_nosa_plan import config


@pytest.fixture
def plan(tmp_path):
    model = tmp_path / "checkpoint"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(nosa_plan.asdict(config())))
    output = tmp_path / "plan"
    nosa_plan.main(["--run-id", "nosa", "--model-path", str(model), "--output-dir", str(output)])
    return output


def test_static_plan_archive_remains_verifiable_after_source_move(plan, monkeypatch):
    monkeypatch.setattr(nosa_plan, "ROOT", plan / "no_live_source")
    assert report.read_nosa_plan(plan)["status"] == "static_plan_not_execution"


@pytest.mark.parametrize("change", ["source", "charge", "physical_fit", "population"])
def test_static_plan_tampering_cannot_publish(plan, change):
    path = plan / "plan.json"
    data = json.loads(path.read_text())
    if change == "source":
        (plan / "source" / next(iter(data["sources"]))).write_text("altered source")
    elif change == "charge":
        data["cases"]["overlap"]["one_session"]["allocator_reservation_bound"]["hbm"] += 1
    elif change == "physical_fit":
        data["cases"]["overlap"]["populations"]["full_token_quota"]["physical_fit_validated"] = True
    else:
        data["cases"]["overlap"]["populations"]["full_token_quota"]["sessions"] -= 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report.read_nosa_plan(plan)


def test_memory_table_does_not_add_allocated_and_reserved_or_claim_device_peak():
    row = {
        "scheme": "overlap",
        "cached_users": 16,
        "is_revisit": True,
        "prefix_cache_hit": True,
        "memory_after": {
            "torch_peak_allocated_bytes": 21,
            "torch_peak_reserved_bytes": 30,
            "torch_allocated_bytes": 20,
            "torch_reserved_bytes": 25,
            "device_used_bytes": 29,
        },
    }
    for field in (
        "cache_hbm_bytes",
        "cache_dram_bytes",
        "reserved_hbm_bytes",
        "reserved_dram_bytes",
        "shared_cache_hbm_bytes",
        "shared_cache_dram_bytes",
        "session_reserved_hbm_bytes",
        "session_reserved_dram_bytes",
        "cache_host_pages",
        "cache_hbm_tokens",
    ):
        row[field] = 1
    result = report.memory_rows("nosa", {"run_id": "measured"}, [row])[0]
    assert result["torch_peak_allocated_bytes"] == 21
    assert result["torch_peak_reserved_bytes"] == 30
    assert result["maximum_after_request_device_used_bytes"] == 29
    assert result["maximum_after_request_reserved_minus_allocated_bytes"] == 5
    assert result["maximum_after_request_device_used_minus_reserved_bytes"] == 4
    assert result["revisit_history_hits"] == result["revisit_requests"] == 1


@pytest.mark.parametrize("change", [None, "history_tokens", "model", "checkpoint", "source"])
def test_only_matched_nosa_layout_and_source_can_supply_graph_observations(plan, change):
    static = report.read_nosa_plan(plan)
    trace = {
        "config": {**static["parameters"], "num_users": static["parameters"]["requested_users"]},
        "model_config": static["model_config"].copy(),
        "checkpoint": {"metadata_sha256": {"config.json": static["config_source"]["sha256"]}},
        "source_manifest": static["sources"].copy(),
        "cases": [{"scheme": scheme} for scheme in static["cases"]],
    }
    if change == "history_tokens":
        trace["config"]["history_tokens"] += 64
    elif change == "model":
        trace["model_config"]["num_hidden_layers"] += 1
    elif change == "checkpoint":
        trace["checkpoint"]["metadata_sha256"]["config.json"] = "different"
    elif change == "source":
        trace["source_manifest"]["models/nosa/execution/resources.py"] = "different"
    if change is None:
        report.validate_nosa_observation(static, trace)
    else:
        with pytest.raises(ValueError):
            report.validate_nosa_observation(static, trace)
