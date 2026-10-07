import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_mfu.src import backend_provenance, measure, profile_layers
from experiments.deepseek_v32_mfu.src.run_contract import benchmark_view
from models.deepseek_v32.nonmatrix import rms_norm


@pytest.mark.parametrize("mode", ["check", "bench", "profile"])
def test_default_a128_keeps_history_chunk_and_complete_extend(tmp_path, mode):
    argv = ["--run-id", "a128", "--output", str(tmp_path / "run")]
    if mode != "check":
        argv += ["--validation-receipt", str(tmp_path / "receipt.json")]
    if mode == "profile":
        argv += ["--nsys"]
    actual_mode, args = profile_layers.parse_run_args(
        "profile" if mode == "profile" else None,
        argv if mode == "profile" else ["--mode", mode, *argv],
    )
    assert actual_mode == mode
    assert (args.prefix, args.extend, args.chunk_size, args.extend_chunk_size) == (
        65536,
        128,
        1024,
        128,
    )
    assert args.slots == 65664 and args.workspace_query_tokens == 1024
    prepared = []
    model = SimpleNamespace(prepare_compute_graphs=prepared.append)
    profile_layers.prepare_graphs(model, args, "check")
    assert prepared == [[128, 1024]]


def test_graph_profile_rejects_split_a128_output(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        profile_layers.parse_run_args(
            "profile",
            [
                "--run-id",
                "split",
                "--output",
                str(tmp_path / "run"),
                "--validation-receipt",
                str(tmp_path / "receipt.json"),
                "--extend-chunk-size",
                "64",
            ],
        )
    assert error.value.code == 2
    assert "one complete extend chunk" in capsys.readouterr().err


def test_full_extend_graph_is_explicit_and_requires_complete_batch(tmp_path, capsys):
    argv = ["--mode", "check", "--run-id", "full", "--output", str(tmp_path / "run")]
    _, args = profile_layers.parse_run_args(argv=argv)
    assert not args.extend_graph
    _, args = profile_layers.parse_run_args(argv=[*argv, "--extend-graph"])
    assert args.extend_graph
    with pytest.raises(SystemExit):
        profile_layers.parse_run_args(argv=[*argv, "--extend-graph", "--extend-chunk-size", "64"])
    assert "full extend graph requires one complete extend chunk" in capsys.readouterr().err


class FakeBlock:
    def __init__(self, layer):
        self.layer = layer
        self.is_moe = False
        self.attention = SimpleNamespace(
            capture_hook=None,
            attention=SimpleNamespace(precision={"fixture": "CPU FP32"}),
        )
        self.cache = SimpleNamespace(metrics=lambda: {"layer": layer})

    def forward(self, hidden, residual=None, *, scope=None):
        combined = hidden.float() if residual is None else hidden.float() + residual.float()
        return (combined * 0.75 + self.layer * 0.125).bfloat16(), (combined * 0.25).bfloat16()


class FakeModel:
    """Deterministic CPU stand-in recording the driver's cache lifecycle."""

    def __init__(self, *_args, **kwargs):
        self.num_layers = kwargs.get("num_layers", 3)
        self.cfg = SimpleNamespace(num_hidden_layers=61, norm_eps=1e-6, attention_scale=1.0)
        self.final_norm = torch.tensor([0.75, 1.25])
        self.head_weight = torch.tensor([[1.0, 0.5], [0.25, -1.0]]).bfloat16()
        self.chunk_size = kwargs.get("chunk_size", 2)
        self.blocks = [FakeBlock(layer) for layer in range(self.num_layers)]
        self.instrumented = False
        self.length = 0
        self.history = []
        self.offload = False
        self.events = []

    def synchronize(self):
        self.events.append(("sync",))

    def set_cache_mode(self, offload):
        self.offload, self.length, self.history = offload, 0, []
        self.events.append(("empty", offload))

    def set_cache_method(self, method):
        self.method = method
        self.set_cache_mode(method != "hbm")
        self._shared_pools = (
            {"cpu": SimpleNamespace(dense_contiguous=method == "dense_prefetch")}
            if self.offload
            else {}
        )

    def evict_prefix_residency(self):
        self.events.append(("evict", self.method))

    def snapshot_prefix(self):
        self.events.append(("snapshot", self.offload, self.length, self.instrumented))
        return self.offload, tuple(self.history)

    def restore_prefix(self, snapshot):
        offload, history = snapshot
        assert offload == self.offload
        self.history, self.length = list(history), len(history)
        self.events.append(("restore", self.offload, self.length))

    def forward(self, ids, *, scope=None, return_hidden=False):
        self.events.append(
            ("forward", self.offload, self.length, len(ids), return_hidden, self.instrumented)
        )
        outputs = []
        for start in range(0, len(ids), self.chunk_size):
            values = torch.tensor(ids[start : start + self.chunk_size]).float()
            # Include the committed prefix to catch failed reset/restore behavior.
            hidden = torch.stack((values + sum(self.history), values + 1), -1).bfloat16()
            residual = None
            for block in self.blocks:
                hidden, residual = block.forward(hidden, residual, scope=scope)
                if block.attention.capture_hook is not None:
                    block.attention.capture_hook(
                        SimpleNamespace(q=hidden, index_q=hidden, index_weights=hidden),
                        hidden,
                        hidden,
                        torch.tensor([[0]] * len(hidden)),
                        SimpleNamespace(host_records=lambda hidden=hidden: hidden),
                        self.length + start,
                    )
            outputs.append(
                rms_norm(
                    hidden.float() + residual.float(), self.final_norm, self.cfg.norm_eps
                ).bfloat16()
            )
        hidden = torch.cat(outputs)
        logits = torch.nn.functional.linear(hidden[-1:], self.head_weight).float()
        self.history.extend(ids)
        self.length = len(self.history)
        self.synchronize()
        return {"hidden": hidden, "logits": logits} if return_hidden else logits


@pytest.mark.parametrize("method", ["hbm", "echo", "serial_sparse", "dense_prefetch"])
def test_driver_transport_declaration_requires_matching_allocated_layout(method):
    pools = {"cpu": SimpleNamespace(dense_contiguous=method != "dense_prefetch")}
    model = SimpleNamespace(set_cache_method=lambda _: None, _shared_pools=pools)
    with pytest.raises(RuntimeError, match="allocated cache layout"):
        profile_layers.select_cache_method(model, method)


@pytest.fixture
def cpu_annotation(monkeypatch):
    events = []

    class Scopes:
        def __init__(self, mode, phase):
            self.calls = []

        @contextmanager
        def __call__(self, stage):
            self.calls.append({"stage": stage})
            yield

    @contextmanager
    def instrument(model, _scopes):
        assert not model.instrumented
        model.instrumented = True
        try:
            yield
        finally:
            model.instrumented = False

    profiler = SimpleNamespace(
        cudaProfilerStart=lambda: events.append("start"),
        cudaProfilerStop=lambda: events.append("stop"),
    )
    monkeypatch.setattr(profile_layers, "OperatorScopes", Scopes)
    monkeypatch.setattr(profile_layers, "InstrumentOperators", instrument)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: profiler)
    return events


def test_annotated_hidden_normalization_occurs_after_capture_and_wrappers(
    cpu_annotation, monkeypatch
):
    model = FakeModel(chunk_size=2)
    original_forward = model.blocks[-1].forward
    norm_calls = []

    def verify_norm(*args):
        assert cpu_annotation == ["start", "stop"]
        assert not model.instrumented
        norm_calls.append(args[0].shape[0])
        return rms_norm(*args)

    monkeypatch.setattr(profile_layers, "rms_norm", verify_norm)
    expected = FakeModel(chunk_size=2).forward([2, 4, 6], return_hidden=True)
    logits, hidden, _, _ = profile_layers.annotate(
        model, [2, 4, 6], "resident", "extend_annotated", True
    )
    assert norm_calls == [2, 1]
    assert model.blocks[-1].forward == original_forward
    assert ("forward", False, 0, 3, False, True) in model.events
    torch.testing.assert_close(logits, expected["logits"], rtol=0, atol=0)
    torch.testing.assert_close(hidden, expected["hidden"], rtol=0, atol=0)


def test_timed_path_keeps_default_output_and_sync_boundary(monkeypatch):
    events = []

    def forward(ids, **kwargs):
        assert ids == [1, 2] and kwargs == {}
        events.append("forward_with_internal_sync")
        return "last_token_logits"

    clock = iter([1.0, 1.25])
    monkeypatch.setattr(profile_layers.time, "perf_counter", lambda: next(clock))
    model = SimpleNamespace(synchronize=lambda: events.append("sync"), forward=forward)
    output, elapsed = profile_layers.timed(model, [1, 2])
    assert output == "last_token_logits" and elapsed == 250.0
    assert events == ["sync", "forward_with_internal_sync"]


def test_benchmark_records_each_samples_counters_after_timer_stops(monkeypatch):
    current = {"timed": False, "sample": 0}

    def metrics():
        assert not current["timed"]
        return {"prefetch_capacity_failures": current["sample"]}

    def timed(model, ids):
        current["timed"] = True
        current["sample"] += 1
        current["timed"] = False
        return None, float(current["sample"])

    model = SimpleNamespace(
        blocks=[SimpleNamespace(cache=SimpleNamespace(metrics=metrics)) for _ in range(3)],
        snapshot_prefix=lambda: "snapshot",
    )
    args = SimpleNamespace(prefix=4, prefill_repeats=2, repeats=3)
    monkeypatch.setattr(profile_layers, "timed", timed)
    monkeypatch.setattr(profile_layers, "select_cache_method", lambda *_: None)
    monkeypatch.setattr(profile_layers, "restore_extend_prefix", lambda *_: None)
    monkeypatch.setattr(profile_layers, "prepare_extend_graph", lambda *_: None)
    monkeypatch.setattr(profile_layers, "cache_metrics", lambda *_: {})
    result = {"measurements": {}}
    profile_layers.run_benchmark(model, [0] * 6, args, result)
    for measured in result["measurements"].values():
        for phase, count in (("prefill", 2), ("extend", 3)):
            samples = measured[f"{phase}_cache_samples"]
            assert len(samples) == count and all(len(sample) == 3 for sample in samples)
            assert [sample[0]["prefetch_capacity_failures"] for sample in samples] == measured[
                f"{phase}_samples_ms"
            ]


def test_annotated_hidden_norm_keeps_inference_mode_after_capture(cpu_annotation, monkeypatch):
    model = FakeModel(chunk_size=2)
    norm_modes = []

    def verify_norm(*args):
        assert cpu_annotation == ["start", "stop"]
        assert not model.instrumented
        norm_modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        return rms_norm(*args)

    monkeypatch.setattr(profile_layers, "rms_norm", verify_norm)
    with torch.enable_grad():
        assert not torch.is_inference_mode_enabled()
        profile_layers.annotate(model, [2, 4, 6], "resident", "extend_annotated", True)
        assert torch.is_grad_enabled()
    assert norm_modes == [(True, False), (True, False)]


def test_full_graph_annotation_reads_retained_hidden_after_capture(cpu_annotation, monkeypatch):
    model = FakeModel(chunk_size=2)
    graph = SimpleNamespace(
        last_hidden=torch.tensor([[1.0, 2.0], [3.0, 4.0]]).bfloat16(),
        last_residual=torch.tensor([[0.25, 0.5], [0.75, 1.0]]).bfloat16(),
    )
    model._extend_graph = graph
    # Full replay must not use the prefill bank's forward-block or query hooks.
    model._compute_graphs = object()
    logits = torch.tensor([[2.0, 1.0]])
    model.forward = lambda ids, scope: logits

    def verify_norm(*args):
        assert cpu_annotation == ["start", "stop"] and not model.instrumented
        return rms_norm(*args)

    monkeypatch.setattr(profile_layers, "rms_norm", verify_norm)
    actual_logits, actual_hidden, _, _ = profile_layers.annotate(
        model, [2, 4], "hbm", "extend_annotated", True
    )
    expected = rms_norm(
        graph.last_hidden.float() + graph.last_residual.float(),
        model.final_norm,
        model.cfg.norm_eps,
    ).bfloat16()
    torch.testing.assert_close(actual_logits, logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_hidden, expected, rtol=0, atol=0)


def test_full_graph_operator_input_diagnostics_explicitly_execute_python_hooks(tmp_path):
    model = FakeModel(chunk_size=2)
    model.set_cache_method("echo")
    model.forward([1, 2])
    snapshot = model.snapshot_prefix()
    original = model.forward
    graph_choices = []

    def forward(ids, *, use_extend_graph):
        graph_choices.append(use_extend_graph)
        return original(ids)

    model.forward = forward
    args = SimpleNamespace(
        output=tmp_path, run_id="fixture", extend_graph=True, extend_residency="cold"
    )
    profile_layers.capture_kernel_inputs(model, [3, 4], snapshot, args)
    assert graph_choices == [False]
    assert all(block.attention.capture_hook is None for block in model.blocks)
    assert len(list(tmp_path.glob("kernel_inputs_layer_*.pt"))) == 3


@pytest.fixture
def driver(tmp_path, monkeypatch, cpu_annotation):
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {"stable_prefix_tokens": 3, "candidate_suffix_tokens": 2, "input_ids": [1, 2, 3, 4, 5]}
        )
    )
    for name in ("config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "fixture.safetensors"}})
    )
    (tmp_path / "fixture.safetensors").write_bytes(b"weight fixture")
    models = []

    def create(*args, **kwargs):
        model = FakeModel(num_layers=3, chunk_size=2)
        models.append(model)
        return model

    monkeypatch.setattr(profile_layers, "DeepSeekEchoModel", create)
    monkeypatch.setattr(profile_layers, "sources", dict)
    monkeypatch.setattr(
        profile_layers,
        "gather_hardware",
        lambda _: {
            "gpu": {"uuid": "GPU-1"},
            "is_sm90": True,
            "pci_identity": {"device_id": "2335"},
        },
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="GPU-1")
    )
    monkeypatch.setattr(profile_layers, "build_info", dict)
    monkeypatch.setattr(backend_provenance, "collect_backend_provenance", lambda: {"fixture": True})
    monkeypatch.setattr(
        backend_provenance,
        "collect_flashinfer_runtime_artifacts",
        lambda **kwargs: {"fixture": True, "local_native_jit": []},
    )
    monkeypatch.setattr(profile_layers.importlib.metadata, "version", lambda _: "test")
    arguments = [
        "--model",
        str(tmp_path),
        "--request",
        str(request),
        "--prefix",
        "3",
        "--extend",
        "2",
        "--chunk-size",
        "2",
        "--slots",
        "5",
        "--no-compute-graphs",
        "--warmups",
        "1",
        "--prefill-repeats",
        "2",
        "--repeats",
        "2",
    ]

    def execute(mode, *extra):
        output = tmp_path / mode
        argv = [*arguments, "--run-id", mode, "--output", str(output), *extra]
        if mode == "profile":
            profile_layers.main(argv)
        else:
            measure.main(["--mode", mode, *argv])
        return output, json.loads((output / "result.json").read_text()), models[-1]

    return execute, models, cpu_annotation


def test_separate_check_bench_profile_preserve_prefix_and_extend_semantics(driver, monkeypatch):
    execute, _models, profiler = driver
    check_dir, check, checked_model = execute("check")
    assert check["accepted"] and len(check["correctness"]) == 13
    assert check["measurements"] == {} and profiler == []
    assert check["dense_history_transport"] == {
        "hbm": None,
        "echo": None,
        "serial_sparse": None,
        "dense_prefetch": "cuda_memcpy_async_contiguous",
    }
    assert (
        check["execution_identity"]["dense_history_transport"] == check["dense_history_transport"]
    )
    receipt = check_dir / "receipt.json"
    assert receipt.is_file()
    for offload in (False, True):
        forwards = [
            event for event in checked_model.events if event[0] == "forward" and event[1] == offload
        ]
        assert all(event[2] == 0 for event in forwards if event[3] == 3)
        assert all(event[2] == 3 for event in forwards if event[3] == 2)
        assert sum(event[4] for event in forwards) == (3 if offload else 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Clean bench invoked a diagnostic/output-copy operation")

    with monkeypatch.context() as clean:
        for name in ("annotate", "comparison", "capture_kernel_inputs"):
            clean.setattr(profile_layers, name, forbidden)
        clean.setattr(torch, "save", forbidden)
        clean.setattr(torch.Tensor, "cpu", forbidden)
        bench_dir, bench, bench_model = execute("bench", "--validation-receipt", str(receipt))
    assert bench["correctness"] == {} and profiler == []
    assert not list(bench_dir.glob("*.pt"))
    for offload in (False, True):
        forwards = [
            event for event in bench_model.events if event[0] == "forward" and event[1] == offload
        ]
        assert len(forwards) == (
            18 if offload else 6
        )  # independent warmup prefix/extend, then two of each sample
        assert all(not event[4] and not event[5] for event in forwards)
        assert all(event[2] == 0 for event in forwards if event[3] == 3)
        assert all(event[2] == 3 for event in forwards if event[3] == 2)
    profile_dir, profile, profiled_model = execute(
        "profile", "--nsys", "--validation-receipt", str(receipt), "--benchmark-run", str(bench_dir)
    )
    assert profiler == ["start", "stop"] * 8
    assert len(profile["correctness"]) == 25
    assert profile["mode"] == "profile" and profile["benchmark"]["run_id"] == "bench"
    assert all("prefix_samples_ms" not in row for row in profile["measurements"].values())
    view = benchmark_view(profile_dir, profile)
    assert view["measurements"] == bench["measurements"]
    assert view["wall_time_denominator"]["run_id"] == "bench"
    assert len(list(profile_dir.glob("kernel_inputs_layer_*.pt"))) == 3
    assert set(profile["measurements"]) == {"hbm", "echo", "serial_sparse", "dense_prefetch"}
    assert len([event for event in profiled_model.events if event[0] == "evict"]) >= 6
    for offload in (False, True):
        calls = [
            event
            for event in profiled_model.events
            if event[0] == "forward" and event[1] == offload
        ]
        assert all(not event[4] for event in calls)
        assert sum(event[5] for event in calls) == (6 if offload else 2)
        assert all(event[2] == 0 for event in calls if event[3] == 3)
        assert all(event[2] == 3 for event in calls if event[3] == 2)


def test_profile_without_bench_stays_diagnostic(driver):
    execute, _, _ = driver
    check_dir, _, _ = execute("check")
    directory, result, _ = execute(
        "profile", "--validation-receipt", str(check_dir / "receipt.json")
    )
    assert result["accepted"] and "benchmark" not in result
    assert benchmark_view(directory, result, required=False) == result
    with pytest.raises(ValueError, match="no matching independent bench"):
        benchmark_view(directory, result)


def test_changed_check_artifact_rejected_before_benchmark(driver, monkeypatch):
    execute, _, _ = driver
    check_dir, _, _ = execute("check")
    (check_dir / "hbm_control.pt").write_bytes(b"changed")
    monkeypatch.setattr(profile_layers, "run_benchmark", lambda *_: pytest.fail("samples began"))
    with pytest.raises(ValueError, match="evidence changed"):
        execute("bench", "--validation-receipt", str(check_dir / "receipt.json"))


def test_changed_checkpoint_rejected_before_benchmark(driver, monkeypatch):
    execute, _, _ = driver
    check_dir, _, _ = execute("check")
    (check_dir.parent / "fixture.safetensors").write_bytes(b"changed checkpoint")
    monkeypatch.setattr(profile_layers, "run_benchmark", lambda *_: pytest.fail("samples began"))
    with pytest.raises(ValueError, match="does not cover"):
        execute("bench", "--validation-receipt", str(check_dir / "receipt.json"))


def test_snapshot_bytes_include_pool_dataclass_tensors_once():
    from dataclasses import make_dataclass

    record = torch.ones(4, 8, dtype=torch.bfloat16)
    pool_type = make_dataclass("PoolSnapshotFixture", [("records", object), ("alias", object)])
    snapshot = {"pool": pool_type(record, record[:1]), "offset": torch.zeros(16)}
    assert profile_layers.tensor_storage_bytes(snapshot) == record.nbytes + 64


def test_bound_benchmark_cannot_be_replaced(driver):
    execute, _, _ = driver
    check_dir, _, _ = execute("check")
    receipt = str(check_dir / "receipt.json")
    bench_dir, bench, _ = execute("bench", "--validation-receipt", receipt)
    profile_dir, profile, _ = execute(
        "profile", "--validation-receipt", receipt, "--benchmark-run", str(bench_dir)
    )
    bench["run_id"] = "substituted"
    (bench_dir / "result.json").write_text(json.dumps(bench))
    with pytest.raises(ValueError, match="bench changed"):
        benchmark_view(profile_dir, profile)


def test_driver_requires_mapped_local_libraries_before_and_after_execution(driver, monkeypatch):
    execute, _, _ = driver
    requirements = []

    def collect(*, require_local_native):
        requirements.append(require_local_native)
        return {"local_native_jit": [{"library": {"sha256": "unchanged"}}]}

    monkeypatch.setattr(backend_provenance, "collect_flashinfer_runtime_artifacts", collect)
    _, result, _ = execute("check")
    assert requirements == [True, True]
    assert result["execution_runtime_artifacts"] == result["flashinfer_runtime_artifacts"]


def test_driver_rejects_local_library_change_before_acceptance(driver, monkeypatch, tmp_path):
    execute, _, _ = driver
    hashes = iter(("before", "after"))

    def collect(*, require_local_native):
        assert require_local_native
        return {"local_native_jit": [{"library": {"sha256": next(hashes)}}]}

    monkeypatch.setattr(backend_provenance, "collect_flashinfer_runtime_artifacts", collect)
    with pytest.raises(RuntimeError, match="Loaded local native libraries changed"):
        execute("check")
    assert not (tmp_path / "check/result.json").exists()
    assert not (tmp_path / "check/receipt.json").exists()


def test_backend_identity_failure_preserves_both_observations_and_field_paths(
    driver, monkeypatch, tmp_path
):
    execute, _, _ = driver
    identities = iter(
        (
            {"installed": {"provider": {"sha256": "original"}}, "jit_environment": {}},
            {"installed": {"provider": {"sha256": "changed"}}, "jit_environment": {"NEW": "value"}},
        )
    )
    monkeypatch.setattr(backend_provenance, "collect_backend_provenance", lambda: next(identities))
    with pytest.raises(RuntimeError, match="installed.provider.sha256"):
        execute("check")
    before = json.loads((tmp_path / "check/backend_provenance_before.json").read_text())
    after = json.loads((tmp_path / "check/backend_provenance_after.json").read_text())
    assert before["installed"]["provider"]["sha256"] == "original"
    assert after["installed"]["provider"]["sha256"] == "changed"
    assert json.loads((tmp_path / "check/backend_provenance_differences.json").read_text()) == [
        "backend_provenance.installed.provider.sha256",
        "backend_provenance.jit_environment.NEW",
    ]
    assert not (tmp_path / "check/result.json").exists()


def test_full_graph_check_covers_default_repeated_changed_inputs_and_cache(tmp_path, monkeypatch):
    from experiments.deepseek_v32_mfu.src import extend_graph_validation

    class GraphModel(FakeModel):
        def __init__(self):
            super().__init__(chunk_size=2)
            self.cfg.vocab_size = 32
            self.prepared = {}
            self.graph_replays = []
            self.baselines = []

        def prepare_extend_graph(self, ids, *, return_hidden, capture_scope):
            assert self.length == 3
            self.prepared[return_hidden] = tuple(ids)
            return SimpleNamespace(describe=lambda: {"fixture": True})

        def forward(self, ids, *, return_hidden=False, use_extend_graph=True):
            if self.length:
                if use_extend_graph:
                    assert return_hidden in self.prepared
                    self.graph_replays.append((tuple(ids), return_hidden))
                else:
                    self.baselines.append(tuple(ids))
            return super().forward(ids, return_hidden=return_hidden)

    model = GraphModel()
    model.set_cache_method("hbm")
    model.forward([1, 2, 3])
    snapshot = model.snapshot_prefix()
    monkeypatch.setattr(
        extend_graph_validation,
        "cache_state",
        lambda model: {"length": model.length, "layers": [], "history": tuple(model.history)},
    )
    args = SimpleNamespace(extend_graph=True, extend_residency="cold", output=tmp_path)
    result = {}
    profile_layers._check_extend_graph(model, [4, 5], snapshot, args, result, "hbm")
    checks = result["extend_graph_checks"]["hbm"]["checks"]
    assert len(checks) == 11
    assert model.graph_replays == [((4, 5), False), ((4, 5), True), ((4, 5), True), ((4, 6), True)]
    assert model.baselines == [(4, 5), (4, 6)]
    assert all(row["bitwise_equal"] for key, row in checks.items() if not key.endswith("cache"))
    assert (tmp_path / "hbm_extend_graph_check.pt").is_file()
