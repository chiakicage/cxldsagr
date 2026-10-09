"""Check matrix sequencing, isolation and failure propagation without GPU execution."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "shape_matrix.sh"


@pytest.fixture
def matrix_repo(tmp_path):
    root = tmp_path / "repo"
    scripts = root / "experiments/deepseek_v32_mfu/scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
    child = tmp_path / "child.py"
    child.write_text(
        """
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args[0] == "-":
    sys.argv = args
    exec(compile(sys.stdin.read(), "<matrix-metadata>", "exec"))
    raise SystemExit(0)

def value(key):
    return args[args.index(key) + 1]

operation = (
    value("--mode") if args[0] == "run"
    else "profile" if args[0] == "profile"
    else value("--window")
)
run_id = os.environ["MFU_RUN_ID"]
with Path(os.environ["MOCK_CALLS"]).open("a") as stream:
    stream.write(json.dumps({"operation": operation, "run_id": run_id, "args": args}) + "\\n")
if operation == os.environ.get("MOCK_FAILURE"):
    raise SystemExit(37)
if args[0] in ("run", "profile"):
    base = (
        Path(os.environ["TMPDIR"]) / "cxldsagr-checks/deepseek_v32_mfu"
        if operation == "check"
        else Path("experiments/deepseek_v32_mfu/output")
    )
    output = base / "data" / run_id
    output.mkdir(parents=True)
    (output / "receipt.json").write_text("{}\\n")
else:
    assert args[:2] == ["-m", "experiments.deepseek_v32_mfu.src.compact_timeline"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    output = Path(value("--output-dir"))
    output.mkdir()
    (output / "compact_receipt.json").write_text("{}\\n")
print("mock child stdout")
print("mock child stderr", file=sys.stderr)
"""
    )
    binaries = root / ".venv/bin"
    binaries.mkdir(parents=True)
    python = binaries / "python"
    python.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{child}" "$@"\n')
    python.chmod(0o755)
    for name, mode in (("run.sh", "run"), ("gap_profile.sh", "profile")):
        (scripts / name).write_text(
            f'#!/usr/bin/env bash\nexec "{sys.executable}" "{child}" {mode} "$@"\n'
        )
    temp = tmp_path / "temp"
    temp.mkdir()
    calls = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "MFU_RUN_ID": "fixture_matrix",
        "TMPDIR": str(temp),
        "MOCK_CALLS": str(calls),
        "MOCK_FAILURE": "",
    }
    return root, temp, calls, env


def invoke(fixture, *args, failure=""):
    root, temp, _, env = fixture
    return subprocess.run(
        ["bash", str(root / "experiments/deepseek_v32_mfu/scripts/shape_matrix.sh"), *args],
        cwd=temp,
        env={**env, "MOCK_FAILURE": failure},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def load_calls(fixture):
    return [json.loads(line) for line in fixture[2].read_text().splitlines()]


def test_full_matrix_links_each_check_to_matching_bench_and_profile(matrix_repo):
    root, temp, _, _ = matrix_repo
    completed = invoke(matrix_repo, "--", "--physical-device", "3", "--model", "/fixture")
    assert completed.returncode == 0, completed.stderr
    base = root / "experiments/deepseek_v32_mfu/output"
    manifest = json.loads((base / "data/fixture_matrix/manifest.json").read_text())
    expected = [(h, a) for h in (4096, 16384, 65536) for a in (128, 256, 512, 1024)]
    assert [(r["prefix_tokens"], r["extend_tokens"]) for r in manifest["shapes"]] == expected
    calls = load_calls(matrix_repo)
    assert len(calls) == 60
    for index, (history, append) in enumerate(expected):
        row = manifest["shapes"][index]
        operations = calls[index * 5 : (index + 1) * 5]
        assert [r["operation"] for r in operations] == [
            "check",
            "bench",
            "profile",
            "three-layers",
            "extend-startup",
        ]
        for call in operations[:3]:
            args = call["args"]
            for key, value in (
                ("--prefix", history),
                ("--extend", append),
                ("--chunk-size", 1024),
                ("--extend-chunk-size", append),
                ("--sparse-pool-tokens", history + append),
                ("--host-arena-tokens", history + append),
                ("--physical-device", 3),
                ("--extend-residency", "cold"),
            ):
                assert args[args.index(key) + 1] == str(value)
            assert "--extend-graph" in args and "--compute-graphs" in args
        for call in operations[1:3]:
            args = call["args"]
            assert (
                args[args.index("--validation-receipt") + 1] == row["check_run"] + "/receipt.json"
            )
        profile_args = operations[2]["args"]
        assert profile_args[profile_args.index("--benchmark-run") + 1] == row["bench_run"]
        assert Path(row["check_run"]).is_relative_to(temp)
        assert (root / row["timeline_run"] / "compact_receipt.json").is_file()
        assert (root / row["startup_run"] / "compact_receipt.json").is_file()
        assert (base / "log" / row["timeline_run_id"] / "stderr.log").read_text() == (
            "mock child stderr\n"
        )
    assert not list(temp.glob("deepseek-matrix-*"))


def test_subset_without_startup_and_profile_warmups(matrix_repo):
    completed = invoke(
        matrix_repo,
        "--prefix-tokens=16384",
        "--extend-tokens",
        "256,1024",
        "--trace-warmups",
        "2",
        "--no-startup",
    )
    assert completed.returncode == 0, completed.stderr
    calls = load_calls(matrix_repo)
    assert len(calls) == 8
    assert not any(call["operation"] == "extend-startup" for call in calls)
    for call in calls:
        if call["operation"] == "profile":
            assert call["args"][call["args"].index("--trace-warmups") + 1] == "2"
    manifest_path = (
        matrix_repo[0] / "experiments/deepseek_v32_mfu/output/data/fixture_matrix/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert all(row["startup_run"] is None for row in manifest["shapes"])


def test_single_decode_point_keeps_cold_history_and_page_aligned_capacity(matrix_repo):
    completed = invoke(matrix_repo, "--prefix-tokens", "65536", "--extend-tokens", "1")
    assert completed.returncode == 0, completed.stderr
    calls = load_calls(matrix_repo)
    assert len(calls) == 5
    for call in calls[:3]:
        args = call["args"]
        for key, value in (
            ("--prefix", "65536"),
            ("--extend", "1"),
            ("--extend-chunk-size", "1"),
            ("--sparse-pool-tokens", "65600"),
            ("--host-arena-tokens", "65600"),
            ("--extend-residency", "cold"),
        ):
            assert args[args.index(key) + 1] == value
    path = matrix_repo[0] / "experiments/deepseek_v32_mfu/output/data/fixture_matrix/manifest.json"
    (row,) = json.loads(path.read_text())["shapes"]
    assert row["pool_tokens"] == row["host_tokens"] == 65600


@pytest.mark.parametrize(
    "operation,count", [("check", 1), ("bench", 2), ("profile", 3), ("three-layers", 4)]
)
def test_child_failure_stops_without_retry_or_completed_manifest(matrix_repo, operation, count):
    failed = invoke(matrix_repo, failure=operation)
    assert failed.returncode == 37
    assert len(load_calls(matrix_repo)) == count
    root, temp, _, _ = matrix_repo
    base = root / "experiments/deepseek_v32_mfu/output"
    assert not (base / "data/fixture_matrix").exists()
    assert not (base / "data/fixture_matrix_h4096_a128_timeline").exists()
    (staging,) = temp.glob("deepseek-matrix-*")
    assert str(staging) in failed.stderr


@pytest.mark.parametrize(
    "args",
    [
        ["--prefix-tokens", "4096,4096"],
        ["--extend-tokens", "64"],
        ["--extend-tokens", "1"],
        ["--prefix-tokens", "4096,"],
        ["--trace-warmups", "-1"],
        ["--", "--extend-residency", "warm"],
        ["--", "--prefix=128"],
        ["--", "--request", "/some/request.json"],
        ["--model"],
    ],
)
def test_invalid_options_do_not_start_children(matrix_repo, args):
    failed = invoke(matrix_repo, *args)
    assert failed.returncode == 2
    assert not matrix_repo[2].exists()
    assert not list(matrix_repo[1].iterdir())


def test_existing_later_shape_is_rejected_before_first_check(matrix_repo):
    root, temp, calls, _ = matrix_repo
    old = root / "experiments/deepseek_v32_mfu/output/data/fixture_matrix_h65536_a1024_bench"
    old.mkdir(parents=True)
    marker = old / "keep.json"
    marker.write_text("valid result")
    failed = invoke(matrix_repo)
    assert failed.returncode == 2
    assert marker.read_text() == "valid result"
    assert not calls.exists()
    assert not list(temp.iterdir())


def test_help_does_not_start_children_or_create_outputs(matrix_repo):
    completed = invoke(matrix_repo, "--help")
    assert completed.returncode == 0
    assert "P=NH=H+A" in completed.stdout
    assert not matrix_repo[2].exists()
    assert not list(matrix_repo[1].iterdir())
