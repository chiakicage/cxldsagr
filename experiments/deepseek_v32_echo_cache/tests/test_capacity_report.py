"""CPU publication gates for exact output evidence and independent capacity plans."""

import csv
import json

import pytest

from experiments.deepseek_v32_echo_cache.src import capacity_report as report
from experiments.deepseek_v32_echo_cache.src.capacity_probe import reference_binding, write_json
from experiments.deepseek_v32_echo_cache.tests.test_capacity_reference import make_run
from models.deepseek_v32.capacity import EchoCapacityPlanner


def accepted_case(tmp_path):
    run = make_run(tmp_path)
    state = report.read_json(run / "status.json")
    source = run / "source/fixture.py"
    source.parent.mkdir()
    source.write_text("fixture source\n")
    manifest = {"fixture.py": report.digest(source)}
    write_json(run / "source_manifest.json", manifest)
    state.update(
        schema="echo-capacity-probe-v1",
        numerical_status="passed",
        completed_requests=4,
        max_retained_users=2,
        cleanup={"passed": True},
        source_sha256=report.aggregate(manifest),
    )
    state["config"].update(
        resource_mode="fixed_pools", hbm_budget_bytes=None, dram_budget_bytes=None
    )
    state["hardware"].update(name="fixture GPU", total_memory=1 << 30)
    state["backend"]["weights"] = {"hbm": 100 << 20}
    write_json(run / "status.json", state)
    numerical = report.read_json(run / "numerical/manifest.json")
    numerical["binding"] = reference_binding(
        state["config"], state["source_sha256"], state["checkpoint"], "fixture"
    )
    write_json(run / "numerical/manifest.json", numerical)
    reference = tmp_path / "reference"
    reference.mkdir()
    write_json(reference / "manifest.json", {**numerical, "scheme": "hbm"})
    for path in (run / "numerical").glob("*.pt"):
        (reference / path.name).write_bytes(path.read_bytes())
    audit = {
        "schema": "echo-capacity-numerical-audit-v1",
        "run_id": state["run_id"],
        "status": "passed",
        "binding": numerical["binding"],
        "reference_directory": str(reference),
        "reference_manifest_sha256": report.digest(reference / "manifest.json"),
        "requests_file_sha256": report.digest(run / "requests.jsonl"),
        "requests": [
            {
                "request_id": i,
                "status": "passed",
                "atol": 0,
                "rtol": 0,
                "hidden": {"exact": True},
                "logits": {"exact": True},
            }
            for i in range(4)
        ],
    }
    write_json(run / "numerical_audit.json", audit)
    stages = ["before_model_loading", "after_model_loading", "after_shared_pool_allocation"]
    stages += [f"{side}_request_{i}" for i in range(4) for side in ("before", "after")]
    stages.append("after_cleanup")
    write_json(
        run / "memory.json",
        [
            {
                "stage": stage,
                "torch_peak_allocated_bytes": 200 + i,
                "torch_peak_reserved_bytes": 300 + i,
                "cuda_total_bytes": 1000,
                "cuda_free_bytes": 600 - i,
            }
            for i, stage in enumerate(stages)
        ],
    )
    return run


def accepted_plan(tmp_path, variable="NH"):
    planner = EchoCapacityPlanner(
        {
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "index_head_dim": 8,
            "index_topk": 2,
            "max_seq_len": 64,
        },
        num_layers=10,
        history_tokens=3,
        candidate_tokens=2,
        chunk_size=1024,
        total_hbm_bytes=1 << 30,
        model_hbm_bytes=100 << 20,
        dram_budget_bytes=64 << 20,
    )
    plan = (
        planner.max_host_capacity(sparse_pool_tokens=32768)
        if variable == "NH"
        else planner.max_device_capacity(host_arena_tokens=128)
    )
    sources = {
        name: report.digest(report.ROOT / name)
        for name in (
            "models/deepseek_v32/capacity.py",
            "models/deepseek_v32/cache_resources.py",
            "cache/sparse_token_pool.py",
            "cache/host_allocation.py",
            "experiments/deepseek_v32_echo_cache/src/capacity_plan.py",
        )
    }
    data = {
        "schema": "echo-capacity-plan-v1",
        "run_id": f"plan_{variable}",
        "status": "static_plan_not_execution",
        "hardware": {"total_hbm_bytes": 1 << 30, "model_hbm_bytes": 100 << 20},
        "parameters": {
            "fixed_p": 32768 if variable == "NH" else None,
            "fixed_nh": 128 if variable == "P" else None,
            "model_path": str(tmp_path / "checkpoint"),
        },
        "sources": sources,
        "plan": plan,
    }
    path = tmp_path / f"plan_{variable}.json"
    write_json(path, data)
    return path


def test_publish_validated_artifacts_keeps_measured_and_static_claims_separate(tmp_path):
    run, plan = accepted_case(tmp_path), accepted_plan(tmp_path)
    output = tmp_path / "report"
    assert (
        report.main(
            [
                "--runs",
                str(run),
                "--plans",
                str(plan),
                "--output-dir",
                str(output),
                "--run-id",
                "publication",
            ]
        )
        == 0
    )
    summary = report.read_json(output / "summary.json")
    assert summary["cases"][0]["revisit_hits"] == 2
    assert summary["cases"][0]["torch_peak_allocated_bytes"] == 211
    assert summary["plans"][0]["measured_selected_run_ids"] == ""
    assert summary["plans"][0]["status"] == "static_plan_not_execution"
    assert summary["plans"][0]["budget_observation_status"] == "unmeasured"
    with (output / "plans.csv").open(newline="") as stream:
        saved_plan = next(csv.DictReader(stream))
    assert json.loads(saved_plan["budget_observations"]) == {"selected": [], "useful": []}
    provenance = report.read_json(output / "report_provenance.json")
    for name, expected in provenance["output_files_sha256"].items():
        assert report.digest(output / name) == expected
    prose = (output / "results.md").read_text()
    assert "不是 NVML 进程峰值" in prose
    assert "不代表实测 OOM" in prose
    assert "未实测" in prose
    assert {path.name for path in output.iterdir()} == {
        "summary.json",
        "cases.csv",
        "plans.csv",
        "results.md",
        "report_provenance.json",
    }


@pytest.mark.parametrize("damage", ["source", "numerical", "audit_count", "status", "memory"])
def test_publication_rejects_incomplete_or_corrupt_run_evidence(tmp_path, damage):
    run = accepted_case(tmp_path)
    if damage == "source":
        (run / "source/fixture.py").write_text("modified\n")
    elif damage == "numerical":
        (run / "numerical/000000.pt").write_bytes(b"corrupted")
    elif damage == "audit_count":
        value = report.read_json(run / "numerical_audit.json")
        value["requests"].pop()
        write_json(run / "numerical_audit.json", value)
    elif damage == "status":
        value = report.read_json(run / "status.json")
        value["numerical_status"] = "unverified"
        write_json(run / "status.json", value)
    else:
        value = report.read_json(run / "memory.json")
        value.pop()
        write_json(run / "memory.json", value)
    with pytest.raises(ValueError):
        report.read_case(run)


@pytest.mark.parametrize("damage", ["formula", "source", "selected", "next"])
def test_plan_claims_require_verified_sources_formula_and_boundary(tmp_path, damage):
    path = accepted_plan(tmp_path)
    data = report.read_json(path)
    if damage == "formula":
        data["plan"]["budgets"]["raw_hbm_envelope_bytes"] += 1
    elif damage == "source":
        data["sources"]["models/deepseek_v32/capacity.py"] = "unavailable"
    elif damage == "selected":
        data["plan"]["selected"]["feasible"] = False
    else:
        data["plan"]["next_candidate"]["violated_constraints"] = []
    write_json(path, data)
    with pytest.raises(ValueError):
        report.read_plan(path, [])


def test_raw_p_above_nh_is_retained_as_allocation_limit(tmp_path):
    path = accepted_plan(tmp_path, variable="P")
    plan, provenance = report.read_plan(path, [])
    assert plan["P"] > plan["NH"] == plan["useful_P"]
    report.match_runs(plan, provenance, [], [])
    prose = report.markdown([], [plan], "fixture")
    assert "多出的 slots 不增加" in prose
    assert plan["measured_selected_run_ids"] == plan["measured_useful_run_ids"] == ""


def test_eviction_cannot_be_hidden_by_regenerating_the_audit_ledger_hash(tmp_path):
    run = accepted_case(tmp_path)
    path = run / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[2]["metrics"]["evicted_users"] = [0]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit = report.read_json(run / "numerical_audit.json")
    audit["requests_file_sha256"] = report.digest(path)
    write_json(run / "numerical_audit.json", audit)
    with pytest.raises(AssertionError, match="evicted_users"):
        report.read_case(run)


def budget_fixture():
    shared = {
        "H": 64,
        "A": 4,
        "C": 1024,
        "layers": 10,
        "checkpoint_path": "/checkpoint",
        "total_hbm_bytes": 1001,
    }
    plan = {
        **shared,
        "run_id": "original_plan",
        "P": 500,
        "NH": 1000,
        "useful_P": 500,
        "hbm_fraction": 0.9,
        "fraction_hbm_limit_bytes": 900,
        "model_hbm_bytes": 500,
        "hbm_envelope_bytes": 400,
    }
    case = {
        **shared,
        "run_id": "completed_run",
        "P": 500,
        "NH": 1000,
        "torch_peak_allocated_bytes": 800,
        "torch_peak_reserved_bytes": 850,
        "maximum_sampled_device_used_bytes": 880,
    }
    provenance = {"source_hashes": {"model.py": "same_source"}}
    runs = [{"run_id": case["run_id"], "source_manifest": {"model.py": "same_source"}}]
    return plan, case, provenance, runs


@pytest.mark.parametrize("metric", report.BUDGET_METRICS)
def test_any_exceeded_observation_marks_the_budget_exceeded(metric):
    plan, case, _, _ = budget_fixture()
    case[metric] = 901
    observation = report.observe_budget(case, plan)
    assert observation["fraction_hbm_limit_bytes"] == 900
    assert observation["status"] == "observed_exceeded"
    assert observation["exceeded_metrics"] == [metric]
    assert observation["metrics"][metric]["excess_bytes"] == 1
    assert observation["metrics"][metric]["remaining_bytes"] == -1
    assert observation["continuous_process_peak_verified"] is False


def test_fraction_threshold_includes_model_memory_and_equality_is_not_an_excess():
    plan, case, _, _ = budget_fixture()
    case["torch_peak_reserved_bytes"] = 900
    observation = report.observe_budget(case, plan)
    assert observation["status"] == "no_observed_excess"
    assert observation["metrics"]["torch_peak_reserved_bytes"]["excess_bytes"] == 0
    assert observation["continuous_process_peak_verified"] is False
    # The 400-byte cache envelope must not be reused as the whole-run limit.
    assert observation["metrics"]["torch_peak_allocated_bytes"]["status"] == "no_observed_excess"


def test_new_headroom_capacity_remains_unmeasured_despite_shared_fraction_policy():
    plan, case, provenance, runs = budget_fixture()
    case["torch_peak_reserved_bytes"] = 950
    changed_capacity = {**plan, "run_id": "headroom_plan", "P": 400, "useful_P": 400}
    report.match_runs(plan, provenance, [case], runs)
    report.match_runs(changed_capacity, provenance, [case], runs)
    assert plan["budget_observation_status"] == "observed_exceeded"
    assert (
        plan["budget_observations"]["selected"][0]["metrics"]["torch_peak_allocated_bytes"][
            "status"
        ]
        == "no_observed_excess"
    )
    assert changed_capacity["budget_observation_status"] == "unmeasured"
    assert changed_capacity["budget_observations"] == {"selected": [], "useful": []}
    report.observe_cases([case], [plan, changed_capacity], [provenance, provenance], runs)
    assert case["budget_observation_status"] == "observed_exceeded"
    assert len(case["budget_observations"]) == 1
    observed = case["budget_observations"][0]
    assert observed["compatible_policy_plan_ids"] == ["original_plan", "headroom_plan"]
    assert observed["selected_capacity_plan_ids"] == ["original_plan"]


def test_useful_point_observation_does_not_validate_unmeasured_raw_allocation():
    plan, case, provenance, runs = budget_fixture()
    plan.update(P=1100, useful_P=1000)
    case["P"] = 1000
    report.match_runs(plan, provenance, [case], runs)
    assert plan["budget_observation_status"] == "unmeasured"
    assert plan["useful_budget_observation_status"] == "no_observed_excess"
    assert plan["budget_observations"]["selected"] == []


def test_markdown_names_the_overbudget_run_and_does_not_claim_continuous_peak_guarantee(tmp_path):
    path = accepted_plan(tmp_path)
    plan, provenance = report.read_plan(path, [])
    case = {
        **{
            name: plan[name]
            for name in (
                "P",
                "NH",
                "U",
                "H",
                "A",
                "C",
                "layers",
                "checkpoint_path",
                "total_hbm_bytes",
                "candidate_persistence",
            )
        },
        "run_id": "overbudget_run",
        "requests": 4,
        "revisit_hits": 2,
        "revisits": 2,
        "torch_peak_allocated_bytes": plan["fraction_hbm_limit_bytes"] - 1000,
        "torch_peak_reserved_bytes": plan["fraction_hbm_limit_bytes"] + (1 << 30),
        "maximum_sampled_device_used_bytes": plan["fraction_hbm_limit_bytes"] + (2 << 30),
    }
    runs = [{"run_id": case["run_id"], "source_manifest": provenance["source_hashes"]}]
    report.match_runs(plan, provenance, [case], runs)
    report.observe_cases([case], [plan], [provenance], runs)
    prose = report.markdown([case], [plan], "publication")
    assert "`overbudget_run`" in prose
    assert "已观察到超额" in prose
    assert "（超 1.000）" in prose and "（超 2.000）" in prose
    assert "不保证连续的进程峰值始终满足预算" in prose


def test_static_only_report_does_not_require_or_claim_a_completed_capacity_run(tmp_path):
    plan = accepted_plan(tmp_path)
    output = tmp_path / "static_report"
    assert (
        report.main(["--plans", str(plan), "--run-id", "static_only", "--output-dir", str(output)])
        == 0
    )
    summary = report.read_json(output / "summary.json")
    assert summary["cases"] == []
    assert summary["plans"][0]["candidate_persistence"] == "gpu_transient"
    assert summary["plans"][0]["budget_observation_status"] == "unmeasured"
    prose = (output / "results.md").read_text()
    assert "只包含静态规划" in prose
    assert "没有新实现的 allocated/reserved 峰值观测" in prose
    assert "## 固定 P/NH 的实测" not in prose
    assert "## HBM fraction 预算的实测观察" not in prose
    assert "全部候选 hidden 和末 token logits 与独立 HBM 参考精确一致" not in prose


def test_legacy_capacity_run_cannot_validate_gpu_transient_plan_even_with_same_sources():
    plan, case, provenance, runs = budget_fixture()
    plan["candidate_persistence"] = "gpu_transient"
    report.match_runs(plan, provenance, [case], runs)
    assert plan["measured_selected_run_ids"] == ""
    assert plan["budget_observation_status"] == "unmeasured"
    report.observe_cases([case], [plan], [provenance], runs)
    assert case["budget_observations"] == []
