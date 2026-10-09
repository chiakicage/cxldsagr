import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RUN_ID = "fixture-run"


@pytest.fixture
def script_repo(tmp_path):
    root = tmp_path / "repo"
    experiment = root / "experiments" / "deepseek_v32_mfu"
    scripts = experiment / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "run.sh",
        "ncu.sh",
        "profile_layers.sh",
        "gap_profile.sh",
        "runner_common.sh",
        "method_isolation.sh",
    ):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    source = experiment / "src"
    source.mkdir()
    (source / "run_contract.py").write_text(
        "import json, os\nfrom pathlib import Path\n"
        "def validate_completed_result(path, *, expected_mode, expected_profile_detail=None):\n"
        "    result = json.loads(Path(path).read_text())\n"
        "    with open(os.environ['MOCK_CALLS'], 'a') as stream:\n"
        "        stream.write(json.dumps(['validate_completed_result', expected_mode, expected_profile_detail]) + '\\n')\n"
        "    assert os.environ.get('MOCK_FAILURE') != 'validation'\n"
        "    assert result['accepted'] and result['mode'] == expected_mode\n"
        "    assert result['schema_version'] in (3, 4)\n"
        "    if expected_profile_detail: assert result['profile_detail'] == expected_profile_detail\n"
        "    return result\n"
        "def validate_operator_analysis(result_path, analysis_path):\n"
        "    result = json.loads(Path(result_path).read_text())\n"
        "    analysis = json.loads(Path(analysis_path).read_text())\n"
        "    assert len(analysis['captures']) == 2 * len(result['methods'])\n"
        "    assert analysis['calls_outside_selected_captures'] == 0\n"
        "    return analysis\n"
    )
    binaries = root / ".venv" / "bin"
    binaries.mkdir(parents=True)
    mock = (
        "#!"
        + sys.executable
        + "\n"
        + r"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["MOCK_CALLS"], "a") as stream:
    stream.write(json.dumps([name, *args]) + "\n")

def value(option):
    return args[args.index(option) + 1]

phase = os.environ.get("MOCK_FAILURE", "")
if name == "python":
    if args[0] == "-":
        sys.argv = args
        sys.path.insert(0, str(Path.cwd()))
        exec(compile(sys.stdin.read(), "<script-result-validation>", "exec"))
        sys.exit(0)
    if "--help" in args:
        print("Mock experiment options")
        sys.exit(0)
    print("mock stdout")
    print("mock stderr", file=sys.stderr)
    if any(args[1].endswith("." + module) for module in ("measure", "profile_layers", "gap_profile")):
        output = Path(value("--output"))
        output.mkdir()
        request = json.loads(Path(value("--request")).read_text()) if "--request" in args else {}
        (output / "request.json").write_text(json.dumps(request, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        if phase == "measure":
            sys.exit(37)
        mode = value("--mode") if args[1].endswith(".measure") else "profile"
        methods = [value("--method")] if "--method" in args else ["hbm", "echo", "serial_sparse", "dense_prefetch"]
        (output / "result.json").write_text(json.dumps({
            "run_id": value("--run-id"), "accepted": True,
            "schema_version": 4 if "--method" in args else 3, "mode": mode, "num_layers": 3,
            "methods": methods,
            "selected_method": value("--method") if "--method" in args else None,
            "request_sha256": hashlib.sha256((output / "request.json").read_bytes()).hexdigest(),
            "process_provenance": {"pid": os.getpid()},
            "profile_detail": "minimal_node_model_scopes" if args[1].endswith(".gap_profile") else "operator",
            "measurement_identity": {"operator_wrappers_during_forward": not args[1].endswith(".gap_profile")},
            "correctness": {str(i): {} for i in range({"check": 13, "profile": 25, "bench": 0}[mode])},
            "validation_receipt": {"fixture": True},
            "nsys_capture_order": ["graph_setup"] + [f"{m}/{p}" for m in methods for p in (("prefill", "extend_graph_setup", "extend") if "--extend-graph" in args else ("prefill", "extend"))],
        }))
        if mode == "check":
            (output / "receipt.json").write_text("{}")
    elif args[1].endswith(".launch_gap"):
        Path(value("--output")).write_text(json.dumps({"all_methods_pass": None}))
    elif args[1].endswith(".operator_report"):
        output = Path(value("--output-dir"))
        output.mkdir()
        if phase == "analysis":
            sys.exit(39)
        (output / "analysis.json").write_text(json.dumps({
            "captures": [{"audit": {"kernel_count_and_time_conserved": True,
                                     "metadata_call_counts_match": True}}] * args.count("--sqlite"),
            "calls_outside_selected_captures": 0,
        }))
    else:
        Path(value("--metadata")).write_text(json.dumps({
            "run_id": value("--run-id"), "accepted": True,
            "input": str(Path(value("--input")).resolve()),
        }))
elif name == "nsys":
    if args == ["--version"]:
        print("NSYS fixture")
    elif args[0] == "export":
        Path(value("--output")).write_text("sqlite fixture")
    else:
        status = subprocess.call(args[args.index("python"):])
        if status:
            sys.exit(status)
        count = (4 if "--extend-graph" in args else 3) if "--method" in args else (13 if "--extend-graph" in args else 9)
        for capture in range(1, count + 1):
            Path(value("--output") + f".{capture}.nsys-rep").write_text("trace\n")
elif name == "taskset":
    sys.exit(subprocess.call(args[2:]))
elif name == "ncu":
    if args == ["--version"]:
        if phase == "ncu_version":
            sys.exit(31)
        print("NCU fixture")
    elif args == ["--list-sets"]:
        print("source fixture")
    else:
        capture = value("--capture-label")
        print(f"profiling {capture}")
        print(f"diagnostics {capture}", file=sys.stderr)
        if phase == "ncu_" + capture:
            sys.exit(31)
        status = subprocess.call(args[args.index("python"):])
        if status:
            sys.exit(status)
        if phase != "ncu_missing_report":
            Path(value("--export") + ".ncu-rep").write_text("report\n")
        if phase == "ncu_no_kernels":
            print("==WARNING== No kernels were profiled")
elif name == "cp":
    if "/log/." in args[-2]:
        sys.exit(29)
    sys.exit(subprocess.call([os.environ["REAL_CP"], *args]))
"""
    )
    for name in ("python", "nsys", "ncu", "taskset"):
        target = binaries / name
        target.write_text(mock)
        target.chmod(0o755)
    temp = tmp_path / "staging"
    temp.mkdir()
    input_path = tmp_path / "kernel_inputs.pt"
    input_path.write_text("fixture input")
    env = {
        **os.environ,
        "ECHO_RUN_ID": RUN_ID,
        "ECHO_NCU_RUN_ID": RUN_ID,
        "ECHO_NSYS": "0",
        "NCU": str(binaries / "ncu"),
        "TMPDIR": str(temp),
        "MOCK_CALLS": str(tmp_path / "calls.jsonl"),
        "MOCK_FAILURE": "",
    }
    return root, experiment, scripts, temp, input_path, env


def invoke(fixture, script="run.sh", *extra, **env_overrides):
    _, _, scripts, _, input_path, env = fixture
    args = ["bash", str(scripts / script)]
    if script == "ncu.sh":
        args.extend(["--input", str(input_path), "--kernel", "recall"])
    return subprocess.run(
        [*args, *extra],
        cwd=input_path.parent,
        env={**env, **env_overrides},
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def assert_unpublished(fixture):
    _, experiment, _, _, _, _ = fixture
    for category in ("data", "log", "profile"):
        assert not (experiment / "output" / category / RUN_ID).exists()


def test_measure_success_publishes_relocatable_run(script_repo):
    _, experiment, _, temp, _, env = script_repo
    completed = invoke(script_repo, "run.sh", "--physical-device", "2", ECHO_NSYS="0")
    assert completed.returncode == 0, completed.stderr
    assert not list(temp.iterdir())
    output = experiment / "output"
    result_path = output / "data" / RUN_ID / "result.json"
    result = json.loads(result_path.read_text())
    assert result["accepted"] and result["run_id"] == result_path.parent.name
    assert (output / "log" / RUN_ID / "stdout.log").read_text() == "mock stdout\n"
    assert (output / "log" / RUN_ID / "stderr.log").read_text() == "mock stderr\n"
    assert not list((output / "profile" / RUN_ID).iterdir())
    calls = [json.loads(line) for line in Path(env["MOCK_CALLS"]).read_text().splitlines()]
    measure = next(call for call in calls if call[0] == "python")
    assert measure[-2:] == ["--physical-device", "2"]
    assert "nsys" not in {call[0] for call in calls}


def test_measure_failure_keeps_diagnostics_only_in_temporary_directory(script_repo):
    _, _, _, temp, _, _ = script_repo
    failed = invoke(script_repo, ECHO_NSYS="0", MOCK_FAILURE="measure")
    assert failed.returncode == 37
    assert_unpublished(script_repo)
    (staging,) = temp.iterdir()
    assert str(staging) in failed.stderr
    assert (staging / "data" / "request.json").is_file()
    assert (staging / "log" / "stderr.log").read_text() == "mock stderr\n"
    assert invoke(script_repo).returncode == 0


@pytest.mark.parametrize(
    "option", ["--run-id", "--run", "--output", "--out", "--profile-dir", "--profile", "--nsys"]
)
@pytest.mark.parametrize("equals", [False, True])
def test_measure_rejects_script_owned_option_overrides(script_repo, option, equals):
    _, _, _, temp, _, env = script_repo
    args = [f"{option}=elsewhere"] if equals else [option, "elsewhere"]
    failed = invoke(script_repo, "run.sh", *args)
    assert failed.returncode == 2
    assert "script owns" in failed.stderr
    assert_unpublished(script_repo)
    assert not list(temp.iterdir())
    assert not Path(env["MOCK_CALLS"]).exists()


@pytest.mark.parametrize("script", ["run.sh", "ncu.sh"])
@pytest.mark.parametrize("category", ["data", "log", "profile"])
def test_existing_run_is_not_modified(script_repo, script, category):
    _, experiment, _, temp, _, _ = script_repo
    existing = experiment / "output" / category / RUN_ID
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("existing valid data")
    failed = invoke(script_repo, script)
    assert failed.returncode == 2
    assert marker.read_text() == "existing valid data"
    assert not list(temp.iterdir())


def test_ncu_success_publishes_profiles_and_metadata(script_repo):
    _, experiment, _, temp, input_path, _ = script_repo
    completed = invoke(script_repo, "ncu.sh")
    assert completed.returncode == 0, completed.stderr
    assert not list(temp.iterdir())
    for capture in ("full", "source"):
        data = experiment / "output" / "data" / RUN_ID
        metadata = json.loads((data / f"{capture}.json").read_text())
        assert metadata["run_id"] == data.name and metadata["accepted"]
        assert Path(metadata["input"]) == input_path and input_path.is_file()
        assert (data / f"{capture}.command.txt").is_file()
        assert (experiment / "output" / "profile" / RUN_ID / f"{capture}_recall.ncu-rep").is_file()
        assert (experiment / "output" / "log" / RUN_ID / f"{capture}.stderr.log").is_file()


@pytest.mark.parametrize(
    ("phase", "status"),
    [
        ("ncu_version", 31),
        ("ncu_full", 31),
        ("ncu_source", 31),
        ("ncu_no_kernels", 1),
        ("ncu_missing_report", 1),
    ],
)
def test_ncu_failure_does_not_publish_partial_profiles(script_repo, phase, status):
    _, _, _, temp, _, _ = script_repo
    failed = invoke(script_repo, "ncu.sh", MOCK_FAILURE=phase)
    assert failed.returncode == status
    assert_unpublished(script_repo)
    (staging,) = temp.iterdir()
    assert str(staging) in failed.stderr
    assert (staging / "log").is_dir()
    if phase == "ncu_source":
        assert (staging / "profile" / "full_recall.ncu-rep").is_file()
    assert invoke(script_repo, "ncu.sh").returncode == 0


@pytest.mark.parametrize("script", ["run.sh", "ncu.sh"])
def test_publish_failure_rolls_back_owned_directories(script_repo, script):
    root, _, _, temp, _, _ = script_repo
    binaries = root / ".venv" / "bin"
    shutil.copyfile(binaries / "python", binaries / "cp")
    (binaries / "cp").chmod(0o755)
    failed = invoke(script_repo, script, REAL_CP=shutil.which("cp"))
    assert failed.returncode == 29
    assert_unpublished(script_repo)
    (staging,) = temp.iterdir()
    assert str(staging) in failed.stderr
    assert any((staging / "data").iterdir())


@pytest.mark.parametrize("script", ["run.sh", "ncu.sh"])
def test_help_leaves_no_outputs(script_repo, script):
    _, _, _, temp, _, _ = script_repo
    completed = invoke(script_repo, script, "--help")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout
    assert_unpublished(script_repo)
    assert not list(temp.iterdir())


def test_check_outputs_stay_outside_experiments(script_repo):
    _, _, _, temp, _, _ = script_repo
    completed = invoke(script_repo, "run.sh", "--mode", "check")
    assert completed.returncode == 0, completed.stderr
    assert_unpublished(script_repo)
    check = temp / "cxldsagr-checks/deepseek_v32_mfu/data" / RUN_ID
    assert (check / "receipt.json").is_file()


def test_combined_profile_is_rejected(script_repo):
    failed = invoke(script_repo, ECHO_NSYS="1")
    assert failed.returncode == 2
    assert "independent profiling" in failed.stderr
    assert_unpublished(script_repo)


def test_profile_success_publishes_nine_captures(script_repo):
    _, experiment, _, _, _, env = script_repo
    completed = invoke(
        script_repo, "profile_layers.sh", "--validation-receipt", "/tmp/check/receipt.json"
    )
    assert completed.returncode == 0, completed.stderr
    assert len(list((experiment / "output/profile" / RUN_ID).glob("*.nsys-rep"))) == 9
    assert len(list((experiment / "output/data" / RUN_ID).glob("capture_*.sqlite"))) == 9
    calls = [json.loads(line) for line in Path(env["MOCK_CALLS"]).read_text().splitlines()]
    assert not any("experiments.deepseek_v32_mfu.src.measure" in call for call in calls)


def test_full_graph_profile_sends_all_five_setup_lineages_to_analysis(script_repo):
    _, experiment, _, _, _, env = script_repo
    completed = invoke(
        script_repo,
        "profile_layers.sh",
        "--extend-graph",
        "--validation-receipt",
        "/tmp/check/receipt.json",
    )
    assert completed.returncode == 0, completed.stderr
    assert len(list((experiment / "output/profile" / RUN_ID).glob("*.nsys-rep"))) == 13
    assert len(list((experiment / "output/data" / RUN_ID).glob("capture_*.sqlite"))) == 13
    calls = [json.loads(line) for line in Path(env["MOCK_CALLS"]).read_text().splitlines()]
    analysis = next(
        call for call in calls if "experiments.deepseek_v32_mfu.src.operator_report" in call
    )
    assert analysis.count("--graph-setup") == 5 and analysis.count("--sqlite") == 8


@pytest.mark.parametrize("phase,status", [("measure", 37), ("analysis", 39)])
def test_profile_failure_stays_in_temporary_directory(script_repo, phase, status):
    failed = invoke(script_repo, "profile_layers.sh", MOCK_FAILURE=phase)
    assert failed.returncode == status
    assert_unpublished(script_repo)


@pytest.mark.parametrize("script", ["run.sh", "profile_layers.sh", "gap_profile.sh"])
def test_single_method_child_records_literal_runner_and_process(script_repo, script):
    _, experiment, scripts, _, _, env = script_repo
    arguments = ["--method", "echo", "--extend-graph"]
    if script == "run.sh":
        arguments += ["--mode", "bench"]
    completed = invoke(script_repo, script, *arguments, MFU_RUN_ID=RUN_ID)
    assert completed.returncode == 0, completed.stderr
    data = experiment / "output/data" / RUN_ID
    runner = json.loads((data / "runner.json").read_text())
    result = json.loads((data / "result.json").read_text())
    assert result["schema_version"] == 4 and result["methods"] == ["echo"]
    assert runner["invocation"] == [str(scripts / script), *arguments]
    assert runner["target_process"] == result["process_provenance"]
    assert runner["shell_process"]["pid"] != runner["target_process"]["pid"]
    assert runner["shell_process"]["proc_start_ticks"] > 0
    assert runner["command"].count("--method") == 1
    calls = [json.loads(line) for line in Path(env["MOCK_CALLS"]).read_text().splitlines()]
    assert any(call[0] == "validate_completed_result" for call in calls)
    if script != "run.sh":
        assert len(list(data.glob("capture_*.sqlite"))) == 4
        assert len(list((experiment / "output/profile" / RUN_ID).glob("*.nsys-rep"))) == 4
    if script == "profile_layers.sh":
        analysis = next(call for call in calls if ".operator_report" in " ".join(map(str, call)))
        assert analysis.count("--graph-setup") == 2 and analysis.count("--sqlite") == 2


@pytest.mark.parametrize("script", ["run.sh", "profile_layers.sh", "gap_profile.sh"])
def test_shared_completion_validator_failure_prevents_publication(script_repo, script):
    failed = invoke(script_repo, script, MOCK_FAILURE="validation", MFU_RUN_ID=RUN_ID)
    assert failed.returncode != 0
    assert_unpublished(script_repo)


@pytest.mark.parametrize("script", ["profile_layers.sh", "gap_profile.sh"])
@pytest.mark.parametrize("argument", ["--run", "--out=elsewhere", "--nsy"])
def test_profiles_reject_argparse_owned_prefixes_before_launch(script_repo, script, argument):
    _, _, _, temp, _, env = script_repo
    failed = invoke(script_repo, script, argument)
    assert failed.returncode == 2
    assert not list(temp.iterdir())
    assert not Path(env["MOCK_CALLS"]).exists()


def _isolation_fixture(script_repo):
    _, _, _, _, input_path, env = script_repo
    request = input_path.with_name("request.json")
    request.write_text(
        json.dumps(
            {
                "stable_prefix_tokens": 65536,
                "candidate_suffix_tokens": 1,
                "input_ids": [0] * 65536 + [111090],
                "token_source": {"seed": 42},
            }
        )
    )
    # The entrypoint rejects extra execution overrides. Tests use its declared
    # environment without inheriting the machine's optional BLAS/allocator flags.
    for key in list(env):
        if key.startswith(
            (
                "CXLDSAGR_",
                "DG_",
                "DJ_",
                "FLASHINFER_",
                "CUTE_DSL_",
                "TRITON_",
                "PYTORCH_",
                "OMP_",
                "MKL_",
                "OPENBLAS_",
            )
        ) or key in ("CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING"):
            del env[key]
    return request


def test_isolation_runs_four_checks_then_twelve_distinct_measurement_children(script_repo):
    _, experiment, _, temp, _, env = script_repo
    request = _isolation_fixture(script_repo)
    completed = invoke(
        script_repo, "method_isolation.sh", "--request", str(request), MFU_RUN_ID=RUN_ID
    )
    assert completed.returncode == 0, completed.stderr
    manifest_path = experiment / "output/data" / RUN_ID / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    methods = ("hbm", "echo", "serial_sparse", "dense_prefetch")
    expected = [(method, "check") for method in methods]
    expected += [
        (method, phase) for method in methods for phase in ("bench", "profile", "operators")
    ]
    assert [(row["method"], row["phase"]) for row in manifest["execution_order"]] == expected
    assert list(manifest["methods"]) == list(methods)
    calls = [json.loads(line) for line in Path(env["MOCK_CALLS"]).read_text().splitlines()]
    launches = [call for call in calls if call[0] == "taskset"]
    assert len(launches) == 16
    assert all(call[1:3] == ["-c", "0-7"] for call in launches)
    children = [manifest["methods"][method][phase] for method, phase in expected]
    assert [row["order"] for row in children] == list(range(16))
    assert (
        len(
            {
                json.loads((Path(row["directory"]) / "runner.json").read_text())["target_process"][
                    "pid"
                ]
                for row in children
            }
        )
        == 16
    )
    requests = []
    for method, phase in expected:
        row = manifest["methods"][method][phase]
        invocation = row["invocation"]
        requests.append(invocation[invocation.index("--request") + 1])
        assert invocation[invocation.index("--method") + 1] == method
        assert row["result_sha256"] == row["files"]["data"]["result.json"]
        assert row["runner_record_sha256"] == row["files"]["data"]["runner.json"]
        if phase == "check":
            assert Path(row["directory"]).is_relative_to(temp)
            assert row["receipt_sha256"] == row["files"]["data"]["receipt.json"]
            if method == "hbm":
                assert "--hbm-check-receipt" not in invocation
            else:
                assert (
                    invocation[invocation.index("--hbm-check-receipt") + 1]
                    == (manifest["methods"]["hbm"]["check"]["receipt_path"])
                )
        else:
            assert (
                invocation[invocation.index("--validation-receipt") + 1]
                == (manifest["methods"][method]["check"]["receipt_path"])
            )
            if phase != "bench":
                assert (
                    invocation[invocation.index("--benchmark-run") + 1]
                    == (manifest["methods"][method]["bench"]["directory"])
                )
    assert len(set(requests)) == 1
    assert Path(manifest["request"]["path"]).is_file()


def test_isolation_child_failure_stops_before_next_child_and_leaves_no_manifest(script_repo):
    _, experiment, _, _, _, env = script_repo
    request = _isolation_fixture(script_repo)
    failed = invoke(
        script_repo,
        "method_isolation.sh",
        "--request",
        str(request),
        MFU_RUN_ID=RUN_ID,
        MOCK_FAILURE="measure",
    )
    assert failed.returncode == 37
    assert not (experiment / "output/data" / RUN_ID).exists()
    calls = [json.loads(line) for line in Path(env["MOCK_CALLS"]).read_text().splitlines()]
    assert len([call for call in calls if call[0] == "taskset"]) == 1


@pytest.mark.parametrize("script", ["gap_profile.sh", "profile_layers.sh", "method_isolation.sh"])
def test_additional_help_paths_do_not_create_outputs(script_repo, script):
    _, _, _, temp, _, _ = script_repo
    completed = invoke(script_repo, script, "--help")
    assert completed.returncode == 0, completed.stderr
    assert_unpublished(script_repo)
    assert not list(temp.iterdir())


def test_isolation_rejects_execution_overrides_before_any_child(script_repo):
    _, _, _, temp, _, env = script_repo
    request = _isolation_fixture(script_repo)
    failed = invoke(
        script_repo,
        "method_isolation.sh",
        "--request",
        str(request),
        PYTORCH_ALLOC_CONF="backend:native",
    )
    assert failed.returncode != 0 and "Unexpected execution environment overrides" in failed.stderr
    assert not list(temp.iterdir())
    calls = [json.loads(line) for line in Path(env["MOCK_CALLS"]).read_text().splitlines()]
    assert not any(call[0] == "taskset" for call in calls)
