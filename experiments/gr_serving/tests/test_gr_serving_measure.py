import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.gr_serving.src import measure


def test_numerical_gate_checks_every_token_and_rejects_nonfinite():
    reference = torch.ones((4, 8), dtype=torch.bfloat16)
    actual = reference.clone()
    actual[1, 2] = 2
    with pytest.raises(AssertionError):
        measure.numerical_comparison(actual, reference, atol=0, rtol=0)
    actual[1, 2] = torch.nan
    with pytest.raises(AssertionError, match="nonfinite"):
        measure.numerical_comparison(actual, reference, atol=0, rtol=0)


def test_source_gate_rejects_drift_after_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "source_root"
    source.mkdir()
    code = source / "operator.py"
    code.write_text("old source\n")
    monkeypatch.setattr(measure, "ROOT", source)
    manifest = {"operator.py": hashlib.sha256(code.read_bytes()).hexdigest()}
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest))
    measure.verify_source_snapshot(tmp_path)
    code.write_text("changed while measuring\n")
    with pytest.raises(RuntimeError, match="implementation changed"):
        measure.verify_source_snapshot(tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        "layers/attention.py",
        "experiments/gr_serving/src/measure.py",
        "experiments/deepseek_v32_echo_cache/src/artifact_locations.py",
        "experiments/deepseek_v32_echo_cache/scripts/run.sh",
    ],
)
def test_snapshot_includes_shared_model_layers_and_experiment_entrypoints(
    tmp_path, monkeypatch, relative
):
    root = tmp_path / "repo"
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_text("shared = 1\n")
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(measure, "ROOT", root)
    measure.source_snapshot(output)
    manifest = json.loads((output / "source_manifest.json").read_text())
    assert relative in manifest
    assert (output / "source" / relative).read_text() == "shared = 1\n"


def test_measure_has_no_default_revisit_cap():
    args = measure.parser().parse_args(["--run-id", "cpu", "--output-dir", "/tmp/unused"])
    assert args.requests == 32
    assert args.max_revisits is None


def test_access_trace_gate_rejects_changed_draw_or_truncated_workload(tmp_path):
    trace = tmp_path / "requests.csv"
    trace.write_text(
        "request_id,user_id,visit_index,is_revisit,previous_request_id,synthetic_timestamp\n"
        "0,17,0,0,,0.0\n"
        "1,17,1,1,0,1.0\n"
    )
    rows = [
        {
            "request_id": 0,
            "user_id": 17,
            "visit_index": 0,
            "is_revisit": False,
            "previous_request_id": None,
            "timestamp": 0.0,
        },
        {
            "request_id": 1,
            "user_id": 17,
            "visit_index": 1,
            "is_revisit": True,
            "previous_request_id": 0,
            "timestamp": 1.0,
        },
    ]
    workload = SimpleNamespace(requests=rows)
    assert (
        measure.verify_access_trace(workload, trace)
        == hashlib.sha256(trace.read_bytes()).hexdigest()
    )
    rows[1]["user_id"] = 18
    with pytest.raises(ValueError, match="differs from saved access trace"):
        measure.verify_access_trace(workload, trace)
    rows.pop()
    with pytest.raises(ValueError, match="different lengths"):
        measure.verify_access_trace(workload, trace)


@pytest.mark.parametrize("cap", [None, 8])
def test_optional_legacy_cap_validation_reaches_device_gate_without_cuda(monkeypatch, cap):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (0, 0))
    options = ["--run-id", "cpu", "--output-dir", "/tmp/unused"]
    if cap is not None:
        options += ["--max-revisits", str(cap)]
    with pytest.raises(RuntimeError, match="requires one SM90"):
        measure.main(options)


def test_shared_pool_cli_does_not_reinterpret_legacy_session_slots(capsys):
    options = ["--run-id", "cpu", "--output-dir", "/tmp/unused"]
    with pytest.raises(SystemExit):
        measure.parser().parse_args([*options, "--deepseek-slots", "32768"])
    assert "was per-session" in capsys.readouterr().err
    args = measure.parser().parse_args(
        [
            *options,
            "--sparse-pool-tokens",
            "32768",
            "--host-arena-tokens",
            "65536",
            "--workspace-query-tokens",
            "2048",
            "--extend-chunk-size",
            "128",
        ]
    )
    assert (
        args.sparse_pool_tokens,
        args.host_arena_tokens,
        args.workspace_query_tokens,
        args.extend_chunk_size,
    ) == (32768, 65536, 2048, 128)


def test_scheme_switch_calls_backend_owner_cleanup():
    configured = []
    backend = SimpleNamespace(configure_scheme=configured.append)
    assert measure._select_scheme(backend, "deepseek_v32", "echo", None) is backend
    assert configured == ["echo"]


def test_warmup_uses_fixed_capacity_and_candidate_limits(monkeypatch):
    events = []

    class Runner:
        def __init__(self, backend, **kwargs):
            events.append(kwargs["resource_limits"])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append("session_closed")

        def execute(self, request):
            events.append(request)

    monkeypatch.setattr(measure, "PersistentGRRunner", Runner)
    backend = SimpleNamespace(synchronize=lambda: events.append("synchronized"))
    args = SimpleNamespace(
        history_tokens=65536,
        candidate_tokens=128,
        warmup=2,
        hbm_budget_bytes=10,
        dram_budget_bytes=20,
    )
    measure._warmup(backend, "request", args)
    assert events == [
        {"max_session_capacity": 65664, "max_history_tokens": 65536, "max_candidate_tokens": 128},
        "request",
        "request",
        "session_closed",
        "synchronized",
    ]


@pytest.mark.parametrize("users,requests", [([16], 31), ([16], 16), ([8, 16], 32)])
def test_sequential_measure_rejects_incomplete_or_unequal_rounds_before_cuda(users, requests):
    options = [
        "--run-id",
        "cpu",
        "--output-dir",
        "/tmp/unused",
        "--sampling",
        "sequential",
        "--requests",
        str(requests),
        "--users",
        *map(str, users),
    ]
    with pytest.raises(ValueError, match="rounds|population"):
        measure.main(options)


def test_sequential_round_count():
    assert (
        measure._sequential_rounds(SimpleNamespace(sampling="sequential", users=[16], requests=32))
        == 2
    )


@pytest.mark.parametrize("failure", ["description", "warmup", "numerical", None])
def test_measure_closes_backend_on_success_and_failures(tmp_path, monkeypatch, failure):
    from executor.serving_backend import SharedCachePlan

    events, limits = [], []
    requests = [{"request_id": rid, "is_revisit": bool(rid)} for rid in range(2)]
    workload = SimpleNamespace(
        requests=requests,
        manifest={
            "workload_sha256": "fixture",
            "observed": {
                "unique_users": 1,
                "revisits": 1,
                "max_revisits": 1,
            },
        },
        write=lambda *_: None,
    )

    def fail(stage):
        if failure == stage:
            raise RuntimeError(stage)

    def describe():
        fail("description")
        return {}

    backend = SimpleNamespace(
        describe=describe,
        configure_scheme=lambda scheme: events.append(scheme),
        close=lambda: events.append("backend_closed"),
        estimate_session_bytes=lambda *_: {"hbm": 10, "dram": 0},
        estimate_session_host_pages=lambda *_: 0,
        last_logits=torch.ones((1, 2)),
    )

    class Runner:
        resource_plan = SharedCachePlan()

        def __init__(self, *_args, **kwargs):
            limits.append(kwargs["resource_limits"])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append("runner_closed")

        def execute(self, request):
            return SimpleNamespace(
                hidden=torch.ones((2, 1)),
                metrics={
                    "request_id": request["request_id"],
                    "latency_ms": 1.0,
                    "prefix_cache_hit": request["is_revisit"],
                    "cached_users": 1,
                },
            )

    monkeypatch.setattr(measure, "_backend", lambda *_: backend)
    monkeypatch.setattr(measure, "build_workload", lambda *_args, **_kwargs: workload)
    monkeypatch.setattr(measure, "_warmup", lambda *_: fail("warmup"))
    original_comparison = measure.numerical_comparison

    def compare(*args, **kwargs):
        fail("numerical")
        return original_comparison(*args, **kwargs)

    monkeypatch.setattr(measure, "numerical_comparison", compare)
    monkeypatch.setattr(measure, "PersistentGRRunner", Runner)
    monkeypatch.setattr(measure, "source_snapshot", lambda *_args, **_kwargs: "source")
    monkeypatch.setattr(measure, "verify_source_snapshot", lambda *_: None)
    monkeypatch.setattr(measure, "backend_provenance", lambda: None)
    monkeypatch.setattr(measure, "_git", lambda *_: "fixture")
    monkeypatch.setattr(measure, "write_report", lambda *_: None)
    monkeypatch.setattr(measure.importlib.metadata, "version", lambda *_: "fixture")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: 1)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: SimpleNamespace(
            name="fixture",
            uuid="fixture",
            total_memory=100,
            multi_processor_count=1,
        ),
    )
    output = tmp_path / "run"
    options = [
        "--run-id",
        "fixture",
        "--output-dir",
        str(output),
        "--models",
        "deepseek_v32",
        "--users",
        "1",
        "--requests",
        "2",
        "--history-tokens",
        "2",
        "--candidate-tokens",
        "2",
    ]
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            measure.main(options)
    else:
        measure.main(options)
        assert json.loads((output / "metadata.json").read_text())["status"] == "accepted"
    assert events[-1] == "backend_closed"
    assert events.count("backend_closed") == 1
    assert all(
        value == {"max_session_capacity": 4, "max_history_tokens": 2, "max_candidate_tokens": 2}
        for value in limits
    )
