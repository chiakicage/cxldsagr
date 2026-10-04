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
        "--resource-mode",
        "--hbm-budget-gib",
        "--dram-budget-gib",
        "--sparse-pool-tokens",
        "--host-arena-tokens",
        "--workspace-query-tokens",
        "--extend-chunk-size",
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
        ["--sparse-pool-tokens", "0"],
        ["--host-arena-tokens", "65"],
        ["--workspace-query-tokens", "1"],
        ["--extend-chunk-size", "0"],
        ["--resource-mode", "fixed-pools"],
        ["--resource-mode", "fixed-pools", "--scheme", "echo"],
        [
            "--resource-mode",
            "fixed-pools",
            "--scheme",
            "echo",
            "--host-arena-tokens",
            "128",
            "--hbm-budget-gib",
            "4",
        ],
        [
            "--resource-mode",
            "fixed-pools",
            "--model",
            "nosa",
            "--scheme",
            "serial_sparse",
            "--host-arena-tokens",
            "128",
        ],
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
    backend.close_calls = 0

    def close():
        assert backend.built == backend.released
        backend.close_calls += 1

    backend.close = close
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
    assert backend.close_calls == 1


def test_cli_supports_legacy_backend_without_shared_resource_close(monkeypatch, capsys):
    backend = Backend()
    fake_resources(monkeypatch, backend)
    del backend.close
    run_multi_user.main(["--count", "2", "--history-tokens", "2", "--candidate-tokens", "2"])
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[-1]["status"] == "finished"
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
    assert backend.close_calls == 1


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


def test_fixed_pools_has_no_byte_sub_budget():
    parser = run_multi_user._build_parser()
    args = parser.parse_args(
        ["--resource-mode", "fixed-pools", "--scheme", "echo", "--host-arena-tokens", "1050624"]
    )
    run_multi_user._validate_args(args, parser)
    assert args.hbm_budget_bytes is None and args.dram_budget_bytes is None
    assert args.sparse_pool_tokens == 32768 and args.host_arena_tokens == 1050624


def test_budget_mode_retains_default_limits():
    parser = run_multi_user._build_parser()
    args = parser.parse_args([])
    run_multi_user._validate_args(args, parser)
    assert args.hbm_budget_bytes == 2**30
    assert args.dram_budget_bytes == 16 * 2**30


def test_cli_passes_explicit_capacity_and_candidate_bounds(monkeypatch):
    from serving import persistent

    captured = []
    original = persistent.PersistentGRRunner

    def runner(*args, **kwargs):
        captured.append(kwargs["resource_limits"])
        return original(*args, **kwargs)

    backend = Backend()
    fake_resources(monkeypatch, backend)
    monkeypatch.setattr(persistent, "PersistentGRRunner", runner)
    run_multi_user.main(["--count", "1", "--history-tokens", "2", "--candidate-tokens", "2"])
    assert captured == [
        {"max_session_capacity": 4, "max_history_tokens": 2, "max_candidate_tokens": 2}
    ]


@pytest.mark.parametrize("fail_at", ["started", "completed", "finished"])
def test_output_failure_closes_backend(monkeypatch, fail_at):
    backend = Backend()
    fake_resources(monkeypatch, backend)

    def emit(row):
        if row["status"] == fail_at:
            raise OSError("broken output")

    monkeypatch.setattr(run_multi_user, "_emit", emit)
    with pytest.raises(SystemExit) as error:
        run_multi_user.main(["--count", "1"])
    assert error.value.code == 1
    assert backend.close_calls == 1


def test_runtime_builds_deepseek_with_shared_pool_options(monkeypatch):
    import sys

    import torch

    captured = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)

    def backend(path, **kwargs):
        captured.update(kwargs)
        return path

    monkeypatch.setitem(
        sys.modules,
        "models.deepseek_v32.serving_backend",
        SimpleNamespace(DeepSeekServingBackend=backend),
    )
    parser = run_multi_user._build_parser()
    args = parser.parse_args(
        [
            "--sparse-pool-tokens",
            "65536",
            "--host-arena-tokens",
            "131072",
            "--workspace-query-tokens",
            "2048",
            "--extend-chunk-size",
            "128",
        ]
    )
    run_multi_user._validate_args(args, parser)
    assert run_multi_user._build_backend(args) == args.model_path
    assert "slots" not in captured
    assert captured["sparse_pool_tokens"] == 65536
    assert captured["host_arena_tokens"] == 131072
    assert captured["workspace_query_tokens"] == 2048
    assert captured["extend_chunk_size"] == 128
