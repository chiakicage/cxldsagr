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
    for name in ("run.sh", "ncu.sh", "profile_layers.sh"):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    binaries = root / ".venv" / "bin"
    binaries.mkdir(parents=True)
    mock = (
        "#!"
        + sys.executable
        + "\n"
        + r"""
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
        exec(compile(sys.stdin.read(), "<script-result-validation>", "exec"))
        sys.exit(0)
    if "--help" in args:
        print("Mock experiment options")
        sys.exit(0)
    print("mock stdout")
    print("mock stderr", file=sys.stderr)
    if args[1].endswith(".measure") or args[1].endswith(".profile_layers"):
        output = Path(value("--output"))
        output.mkdir()
        (output / "request.json").write_text("{}\n")
        if phase == "measure":
            sys.exit(37)
        mode = "profile" if args[1].endswith(".profile_layers") else value("--mode")
        (output / "result.json").write_text(json.dumps({
            "run_id": value("--run-id"), "accepted": True,
            "schema_version": 3, "mode": mode, "num_layers": 3,
            "correctness": {str(i): {} for i in range({"check": 13, "profile": 25, "bench": 0}[mode])},
            "validation_receipt": {"fixture": True},
            "nsys_capture_order": ["graph_setup"] + [f"{m}/{p}" for m in ("hbm", "echo", "serial_sparse", "dense_prefetch") for p in ("prefill", "extend")],
        }))
        if mode == "check":
            (output / "receipt.json").write_text("{}")
    elif args[1].endswith(".operator_report"):
        output = Path(value("--output-dir"))
        output.mkdir()
        if phase == "analysis":
            sys.exit(39)
        (output / "analysis.json").write_text(json.dumps({
            "captures": [{"audit": {"kernel_count_and_time_conserved": True,
                                     "metadata_call_counts_match": True}}] * 8,
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
        for capture in range(1, 10):
            Path(value("--output") + f".{capture}.nsys-rep").write_text("trace\n")
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
    for name in ("python", "nsys", "ncu"):
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


@pytest.mark.parametrize("phase,status", [("measure", 37), ("analysis", 39)])
def test_profile_failure_stays_in_temporary_directory(script_repo, phase, status):
    failed = invoke(script_repo, "profile_layers.sh", MOCK_FAILURE=phase)
    assert failed.returncode == status
    assert_unpublished(script_repo)
