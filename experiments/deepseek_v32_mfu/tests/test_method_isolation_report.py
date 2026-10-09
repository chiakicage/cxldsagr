"""Cohort joins reject aliased children, mixed receipts and changed evidence."""

import json
import shlex
from copy import deepcopy

import pytest

from experiments.deepseek_v32_mfu.src import method_isolation_report as report


def write_json(path, value):
    path.write_text(json.dumps(value))
    return report.audit.Evidence().digest(path)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    checked = []
    monkeypatch.setattr(
        report, "validate_completed_result", lambda path, **kwargs: checked.append((path, kwargs))
    )
    manifest = {
        "schema_version": 1,
        "kind": report.KIND,
        "methods": {},
        "execution_order": [],
        "completed_at_utc": "2026-10-08T10:00:00+00:00",
    }
    for method in report.METHODS:
        manifest["methods"][method] = {}
        for phase in report.CHILDREN:
            run_id = f"cohort_{method}_{phase}"
            directory = tmp_path / run_id
            directory.mkdir()
            sha = write_json(
                directory / "result.json",
                {"schema_version": 4, "methods": [method], "run_id": run_id},
            )
            runner_sha = write_json(
                directory / "runner.json", {"argv": ["script", "--method", method]}
            )
            manifest["methods"][method][phase] = {
                "directory": str(directory),
                "run_id": run_id,
                "result_sha256": sha,
                "runner_record_sha256": runner_sha,
                "invocation": ["script", "--method", method],
                "order": len(manifest["execution_order"]),
            }
            manifest["execution_order"].append({"method": method, "phase": phase, "run_id": run_id})
    manifest["execution_order"] = []
    for method, phase in [(method, "check") for method in report.METHODS] + [
        (method, phase) for method in report.METHODS for phase in report.CHILDREN[1:]
    ]:
        child = manifest["methods"][method][phase]
        child["order"] = len(manifest["execution_order"])
        manifest["execution_order"].append(
            {"method": method, "phase": phase, "run_id": child["run_id"]}
        )
    return manifest, checked


def test_cohort_validates_every_original_child_and_profile_role(cohort):
    manifest, checked = cohort
    directories, results = report.manifest_children(manifest, report.audit.Evidence())
    assert len(checked) == 16
    assert set(directories) == set(results) == set(report.METHODS)
    assert [row[1]["expected_mode"] for row in checked[:4]] == [
        "check",
        "bench",
        "profile",
        "profile",
    ]
    assert checked[2][1]["expected_profile_detail"] == "minimal_node_model_scopes"
    assert checked[3][1]["expected_profile_detail"] == "matrix_api_node_ownership"


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "foreign", "mixed", "result", "runner", "order"]
)
def test_cohort_rejects_incomplete_or_misbound_children(cohort, defect):
    manifest, _ = cohort
    child = manifest["methods"]["echo"]["profile"]
    path = report.Path(child["directory"])
    if defect == "missing":
        del manifest["methods"]["serial_sparse"]
    elif defect == "duplicate":
        manifest["methods"]["echo"]["profile"] = deepcopy(manifest["methods"]["hbm"]["profile"])
    elif defect in ("foreign", "mixed"):
        data = json.loads((path / "result.json").read_text())
        data["methods"] = ["hbm"] if defect == "foreign" else list(report.METHODS)
        child["result_sha256"] = write_json(path / "result.json", data)
    elif defect == "result":
        (path / "result.json").write_text("{}")
    elif defect == "runner":
        (path / "runner.json").write_text("{}")
    else:
        child["order"] = 0
    with pytest.raises(ValueError):
        report.manifest_children(manifest, report.audit.Evidence())


def test_useful_work_comparison_ignores_graph_ids_but_preserves_dtype_and_layer():
    base = [
        {
            "phase": "extend_annotated",
            "layer": "layer_0",
            "stage": "indexer_qk",
            "precision": "FP8",
            "useful_flops": 2048,
            "graph_id": 1,
        }
    ]
    echo = deepcopy(base)
    echo[0].update(stage="indexer_fused", graph_id=1928)
    assert report.matrix_work(base) == report.matrix_work(echo)
    echo[0]["precision"] = "BF16"
    assert report.matrix_work(base) != report.matrix_work(echo)


@pytest.mark.parametrize(
    "field",
    [
        "warmups",
        "prefill_repeats",
        "repeats",
        "seed",
        "workspace_query_tokens",
        "hbm_cache_budget_bytes",
        "trace_warmups",
    ],
)
def test_declared_configuration_is_verified_against_every_child(field):
    configuration = {
        **report.CONFIGURATION,
        "trace_warmups": 1,
        "physical_device": "0",
        "cpu_affinity": list(range(8)),
        "model": "/checkpoint",
    }
    results = {
        method: {
            phase: {
                **report.CONFIGURATION,
                "model": "/checkpoint",
                "measurement_identity": {"trace_warmups_per_capture": 1},
            }
            for phase in report.CHILDREN
        }
        for method in report.METHODS
    }
    report.validate_configuration({"configuration": configuration}, results)
    if field == "trace_warmups":
        results["echo"]["profile"]["measurement_identity"]["trace_warmups_per_capture"] = 2
    else:
        results["echo"]["bench"][field] += 1
    with pytest.raises(ValueError, match="differs"):
        report.validate_configuration({"configuration": configuration}, results)


def test_incomplete_cohort_and_bench_before_checks_are_rejected(cohort):
    manifest, _ = cohort
    incomplete = deepcopy(manifest)
    del incomplete["completed_at_utc"]
    with pytest.raises(ValueError, match="completion"):
        report.manifest_children(incomplete, report.audit.Evidence())
    order = manifest["execution_order"]
    order[1], order[4] = order[4], order[1]
    for index, child in enumerate(order):
        manifest["methods"][child["method"]][child["phase"]]["order"] = index
    with pytest.raises(ValueError, match="checks must precede"):
        report.manifest_children(manifest, report.audit.Evidence())


def runner_fixture(root, phase):
    wrapper, module, temporary, trace = {
        "check": ("run.sh", "measure", "deepseek-echo", None),
        "bench": ("run.sh", "measure", "deepseek-echo", None),
        "profile": ("gap_profile.sh", "gap_profile", "deepseek-minimal-node", "minimal"),
        "operators": ("profile_layers.sh", "profile_layers", "deepseek-layers3", "layers3"),
    }[phase]
    source = "experiments/deepseek_v32_mfu/scripts/" + wrapper
    run_id = "cohort_hbm_" + phase
    output = report.Path("/tmp") / (temporary + "-" + run_id + ".ABCDEF") / "data"
    forwarded = ["--request", "/tmp/request.json", "--method", "hbm"]
    invocation = [str(root / source), *forwarded]
    if phase in ("check", "bench"):
        invocation.extend(["--mode", phase])
        command = [
            "python",
            "-m",
            "experiments.deepseek_v32_mfu.src." + module,
            "--mode",
            phase,
            "--run-id",
            run_id,
            "--output",
            str(output),
            *forwarded,
        ]
    else:
        command = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=repeat",
            "--output",
            str(output.parent / "profile" / trace),
            "python",
            "-m",
            "experiments.deepseek_v32_mfu.src." + module,
            "--run-id",
            run_id,
            "--output",
            str(output),
            "--nsys",
            *forwarded,
        ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source": source,
        "helper_source": "experiments/deepseek_v32_mfu/scripts/runner_common.sh",
        "invocation": invocation,
        "command": command,
        "command_text": shlex.join(command),
    }, {"run_id": run_id}


@pytest.mark.parametrize("phase", report.CHILDREN)
def test_literal_wrapper_transform_is_bound_for_each_child_role(tmp_path, phase):
    runner, result = runner_fixture(tmp_path, phase)
    report.validate_runner_command(runner, result, method="hbm", phase=phase, root=tmp_path)


@pytest.mark.parametrize("phase", report.CHILDREN)
@pytest.mark.parametrize(
    "defect",
    [
        "wrapper",
        "helper",
        "invocation_path",
        "method",
        "duplicate_method",
        "module",
        "run_id",
        "request",
        "output",
        "command_text",
        "extra_command_argument",
    ],
)
def test_phase_provenance_rejects_mutated_wrapper_and_child_arguments(tmp_path, phase, defect):
    runner, result = runner_fixture(tmp_path, phase)
    command = runner["command"]
    if defect == "wrapper":
        runner["source"] = "experiments/deepseek_v32_mfu/scripts/method_isolation.sh"
        runner["invocation"][0] = str(tmp_path / runner["source"])
    elif defect == "helper":
        runner["helper_source"] = "experiments/deepseek_v32_mfu/scripts/run.sh"
    elif defect == "invocation_path":
        runner["invocation"][0] = str(tmp_path / "untracked-wrapper.sh")
    elif defect == "method":
        runner["invocation"][runner["invocation"].index("--method") + 1] = "echo"
        command[command.index("--method") + 1] = "echo"
    elif defect == "duplicate_method":
        runner["invocation"].extend(["--method", "echo"])
        command.extend(["--method", "echo"])
    elif defect == "module":
        command[command.index("-m") + 1] = "experiments.deepseek_v32_mfu.src.untracked_driver"
    elif defect == "run_id":
        command[command.index("--run-id") + 1] = "another_run"
    elif defect == "request":
        command[command.index("--request") + 1] = "/tmp/another_request.json"
    elif defect == "output":
        command[8 if phase in ("check", "bench") else 16] = "/tmp/foreign-run/data"
    elif defect == "command_text":
        runner["command_text"] += " --method echo"
    else:
        command.extend(["--warmups", "99"])
    if defect != "command_text":
        runner["command_text"] = shlex.join(command)
    with pytest.raises(ValueError):
        report.validate_runner_command(runner, result, method="hbm", phase=phase, root=tmp_path)


@pytest.mark.parametrize("phase", ["check", "bench"])
def test_runner_cannot_change_or_duplicate_mode(tmp_path, phase):
    runner, result = runner_fixture(tmp_path, phase)
    runner["invocation"].extend(["--mode", "check" if phase == "bench" else "bench"])
    with pytest.raises(ValueError, match="duplicate recorded --mode"):
        report.validate_runner_command(runner, result, method="hbm", phase=phase, root=tmp_path)


@pytest.mark.parametrize("phase", ["profile", "operators"])
def test_profile_trace_flags_are_part_of_command_binding(tmp_path, phase):
    runner, result = runner_fixture(tmp_path, phase)
    runner["command"][5] = "--cuda-graph-trace=graph"
    runner["command_text"] = shlex.join(runner["command"])
    with pytest.raises(ValueError, match="differs from its wrapper invocation"):
        report.validate_runner_command(runner, result, method="hbm", phase=phase, root=tmp_path)
