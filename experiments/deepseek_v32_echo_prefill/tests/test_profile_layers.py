import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_echo_prefill.src import profile_layers
from models.deepseek_v32.echo_model import rms_norm


class FakeBlock:
    def __init__(self, layer):
        self.layer = layer
        self.is_moe = False
        self.attention = SimpleNamespace(capture_hook=None)
        self.cache = SimpleNamespace(metrics=lambda: {"layer": layer})

    def forward(self, hidden, residual=None, *, scope=None):
        combined = hidden.float() if residual is None else hidden.float() + residual.float()
        return (combined * 0.75 + self.layer * 0.125).bfloat16(), (combined * 0.25).bfloat16()


class FakeModel:
    """Deterministic CPU stand-in recording the driver's cache lifecycle."""

    def __init__(self, *_args, **kwargs):
        self.num_layers = kwargs.get("num_layers", 3)
        self.cfg = SimpleNamespace(num_hidden_layers=61, norm_eps=1e-6)
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


def test_driver_uses_independent_prefixes_and_restores_each_extend(
    tmp_path, monkeypatch, cpu_annotation
):
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "stable_prefix_tokens": 3,
                "candidate_suffix_tokens": 2,
                "input_ids": [1, 2, 3, 4, 5],
            }
        )
    )
    output = tmp_path / "output"
    model = FakeModel(num_layers=3, chunk_size=2)
    monkeypatch.setattr(profile_layers, "DeepSeekEchoModel", lambda *_args, **_kwargs: model)
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
    monkeypatch.setattr(profile_layers.importlib.metadata, "version", lambda _: "test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_layers",
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
            "--run-id",
            "cpu-lifecycle",
            "--output",
            str(output),
            "--nsys",
        ],
    )
    profile_layers.main()
    result = json.loads((output / "result.json").read_text())
    assert result["accepted"] and len(result["correctness"]) == 8
    assert cpu_annotation == ["start", "stop"] * 4
    for offload in (False, True):
        calls = [event for event in model.events if event[0] == "forward" and event[1] == offload]
        prefix_calls = [event for event in calls if event[3] == 3]
        extend_calls = [event for event in calls if event[3] == 2]
        assert len(prefix_calls) == 4
        assert all(event[2] == 0 and not event[4] for event in prefix_calls)
        assert sum(event[5] for event in prefix_calls) == 1
        assert all(event[2] == 3 for event in extend_calls)
        assert sum(event[4] for event in extend_calls) == 1
        assert sum(event[5] for event in extend_calls) == 1
        assert ("snapshot", offload, 3, False) in model.events
        mode = "offload" if offload else "resident"
        assert len(result["measurements"][mode]["prefix_samples_ms"]) == 2
        assert len(result["measurements"][mode]["extend_samples_ms"]) == 2
