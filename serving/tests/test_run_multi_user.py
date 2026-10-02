import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from serving import run_multi_user
from serving.tests.test_persistent import Backend, request


def test_cli_help_works_without_cuda():
    process = subprocess.run(
        [sys.executable, "-m", "serving.run_multi_user", "--help"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    for option in (
        "--model",
        "--scheme",
        "--history-tokens",
        "--candidate-tokens",
        "--hbm-budget-gib",
        "--dram-budget-gib",
        "--deepseek-slots",
        "--max-revisits",
    ):
        assert option in process.stdout
    assert "no separate warmup" in process.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model", "nosa", "--scheme", "echo"],
        ["--scheme", "overlap"],
        ["--count", "0"],
        ["--num-users", "0"],
        ["--history-tokens", "0"],
        ["--candidate-tokens", "0"],
        ["--seed", "-1"],
        ["--max-revisits", "-1"],
        ["--hbm-budget-gib", "nan"],
        ["--dram-budget-gib", "inf"],
        ["--dram-budget-gib", "-1"],
        ["--deepseek-slots", "1"],
    ],
)
def test_invalid_options_fail_before_loading_resources(monkeypatch, arguments):
    def forbidden(*_):
        pytest.fail("invalid options must not load a tokenizer or model")

    monkeypatch.setattr(run_multi_user, "_build_generator", forbidden)
    monkeypatch.setattr(run_multi_user, "_build_backend", forbidden)
    with pytest.raises(SystemExit) as error:
        run_multi_user.main(arguments)
    assert error.value.code == 2


def fake_resources(monkeypatch, backend):
    generator = SimpleNamespace(
        population=SimpleNamespace(metadata={"dataset": "beauty"}),
        iter_generate=lambda count: iter(
            [request(candidate=(3 + visit, 4 + visit)) for visit in range(count)]
        ),
    )
    backend.describe = lambda: {"scope": "CPU fixture"}
    monkeypatch.setattr(run_multi_user, "_build_generator", lambda _: generator)
    monkeypatch.setattr(run_multi_user, "_build_backend", lambda _: backend)


def test_cli_runs_persistent_requests_emits_shapes_and_releases(monkeypatch, capsys):
    backend = Backend()
    fake_resources(monkeypatch, backend)
    run_multi_user.main(["--count", "2", "--history-tokens", "2", "--candidate-tokens", "2"])
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]["model"] == "deepseek_v32"
    assert output[0]["model_path"] == "/preset-models"
    assert "no separate warmup" in output[0]["timing"]
    assert [row["status"] for row in output] == ["started", "completed", "completed", "finished"]
    assert output[1]["hidden_shape"] == [2, 1]
    assert not output[1]["metrics"]["is_revisit"]
    assert output[2]["metrics"]["is_revisit"]
    assert output[2]["metrics"]["prefix_cache_hit"]
    assert output[-1]["revisits"] == output[-1]["prefix_cache_hits"] == 1
    assert output[-1]["separate_warmup"] is False
    assert "input_ids" not in json.dumps(output)
    assert "prompt" not in json.dumps(output)
    assert backend.built == backend.released == 1


def test_model_failure_closes_session_and_exits_nonzero(monkeypatch, capsys):
    backend = Backend()
    backend.fail = True
    fake_resources(monkeypatch, backend)
    with pytest.raises(SystemExit) as error:
        run_multi_user.main(["--count", "2"])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "model failure" in output.err
    assert '"status": "finished"' not in output.out
    assert backend.released == 1


def test_runtime_is_uncapped_by_default_and_reports_actual_returning_users(monkeypatch, capsys):
    backend = Backend()
    fake_resources(monkeypatch, backend)
    run_multi_user.main(["--num-users", "8", "--count", "13"])
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]["max_revisits"] is None
    assert output[-1]["requests"] == 13
    assert output[-1]["first_visits"] == 1
    assert output[-1]["revisits"] == 12
    assert output[-1]["returning_users"] == 1


def test_explicit_legacy_cap_remains_available():
    parser = run_multi_user._build_parser()
    args = parser.parse_args(["--max-revisits", "8"])
    run_multi_user._validate_args(args, parser)
    assert args.max_revisits == 8
