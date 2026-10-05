import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_echo_prefill.src import backend_provenance, measure, profile_layers
from experiments.deepseek_v32_echo_prefill.src.compare_backends import compare
from experiments.deepseek_v32_echo_prefill.src.postrun_audit import audit_saved_outputs
from experiments.deepseek_v32_echo_prefill.src.run_contract import benchmark_view
from models.deepseek_v32.nonmatrix import rms_norm


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
        backend_provenance, "collect_flashinfer_runtime_artifacts", lambda: {"fixture": True}
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
        "4",
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
    assert check["accepted"] and len(check["correctness"]) == 5
    assert check["measurements"] == {} and profiler == []
    receipt = check_dir / "receipt.json"
    assert receipt.is_file()
    for offload in (False, True):
        forwards = [
            event for event in checked_model.events if event[0] == "forward" and event[1] == offload
        ]
        assert all(event[2] == 0 for event in forwards if event[3] == 3)
        assert all(event[2] == 3 for event in forwards if event[3] == 2)
        assert sum(event[4] for event in forwards) == 1

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
        assert len(forwards) == 6  # independent warmup prefix/extend, then two of each sample
        assert all(not event[4] and not event[5] for event in forwards)
        assert all(event[2] == 0 for event in forwards if event[3] == 3)
        assert all(event[2] == 3 for event in forwards if event[3] == 2)
    profile_dir, profile, profiled_model = execute(
        "profile", "--nsys", "--validation-receipt", str(receipt), "--benchmark-run", str(bench_dir)
    )
    assert profiler == ["start", "stop"] * 4
    assert len(profile["correctness"]) == 8
    assert profile["mode"] == "profile" and profile["benchmark"]["run_id"] == "bench"
    assert all("prefix_samples_ms" not in row for row in profile["measurements"].values())
    view = benchmark_view(profile_dir, profile)
    assert view["measurements"] == bench["measurements"]
    assert view["wall_time_denominator"]["run_id"] == "bench"
    assert len(list(profile_dir.glob("kernel_inputs_layer_*.pt"))) == 3
    assert len(audit_saved_outputs(profile_dir, profile)["tensor_checks"]) == 8
    cross = compare(profile_dir, profile_dir, profile_dir.parent / "comparison")
    assert all(row["speedup"] == 1 for row in cross["latencies"])
    assert cross["runs"]["official"]["wall_time_denominator"]["run_id"] == "bench"
    for offload in (False, True):
        calls = [
            event
            for event in profiled_model.events
            if event[0] == "forward" and event[1] == offload
        ]
        assert all(not event[4] for event in calls)
        assert sum(event[5] for event in calls) == 2
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
    (check_dir / "resident_control.pt").write_bytes(b"changed")
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
