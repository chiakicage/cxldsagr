"""CPU checks for stage separation; these fixtures are not model acceptance."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.nosa_motivation.src.validation import record_runtime


@pytest.mark.parametrize("model", ["nosa", "deepseek"])
def test_default_bench_requires_receipt_before_runtime_setup(model, tmp_path):
    if model == "nosa":
        from experiments.nosa_motivation.src import measure
    else:
        from experiments.deepseek_v32_motivation.src import measure
    with pytest.raises(ValueError, match="bench requires --validation-receipt"):
        measure.main(["--run-id", "unvalidated", "--output-dir", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


def test_runtime_receipt_tracks_graph_policy_and_native_bytes_not_replay_counts():
    state = {
        "scheme": "hbm",
        "sparse_pool_tokens": 64,
        "compute_graphs": {
            "enabled": True,
            "policy_revision": "policy-a",
            "query_sizes": [2, 64],
            "finish_replays": 10,
            "setup_seconds": 1.0,
        },
    }
    backend = SimpleNamespace(describe=lambda: deepcopy(state))
    native = {"kernel.so": {"sha256": "original", "size": 3}}
    checked = {"mode": "check", "validation_identity": {"methods": {}}}
    record_runtime(checked, "hbm", backend, native, {"backend": "native"})
    bench = {
        "mode": "bench",
        "validation_identity": {"methods": {}},
        "correctness_receipt": {"identity": deepcopy(checked["validation_identity"])},
    }
    state["compute_graphs"].update(finish_replays=0, setup_seconds=9.0)
    record_runtime(bench, "hbm", backend, native, {"backend": "native"})
    state["compute_graphs"]["policy_revision"] = "policy-b"
    with pytest.raises(ValueError, match="identity differs"):
        record_runtime(bench, "hbm", backend, native, {"backend": "native"})
    state["compute_graphs"]["policy_revision"] = "policy-a"
    native["kernel.so"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="identity differs"):
        record_runtime(bench, "hbm", backend, native, {"backend": "native"})


def test_nosa_bench_request_does_not_detach_or_save_full_output(tmp_path, monkeypatch):
    import torch

    from experiments.nosa_motivation.src import measure

    class Hidden:
        shape = (2, 4)

        def detach(self):
            pytest.fail("bench tried to copy/compare full hidden output")

    config = {
        "history_tokens": 64,
        "candidate_tokens": 2,
        "sparse_pool_tokens": 64,
        "host_arena_tokens": 128,
        "requests_per_method": 1,
        "enable_compute_graphs": False,
    }
    plan = SimpleNamespace(
        metadata={"sparse_pool_tokens": 64, "candidate_persistence": "gpu_transient"},
        hbm_tokens=64,
        host_pages=0,
        shared=SimpleNamespace(hbm=8, dram=0),
    )
    events = []

    class Runner:
        pool = ()
        resource_plan = plan

        def __init__(self, *args, **kwargs):
            self.token_validation_identity = {"backend": "fixture"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("sessions_released")

        def execute(self, request):
            return SimpleNamespace(
                hidden=Hidden(),
                metrics={"is_revisit": False, "prefix_cache_hit": False, "latency_ms": 1.0},
            )

    backend = SimpleNamespace(
        scheme="hbm",
        device="cpu",
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=4)),
        describe=lambda: {"compute_graphs": {"enabled": False}},
        close=lambda: events.append("backend_closed"),
    )
    for name in ("loaded_native_artifacts", "allocator_snapshot_runtime_info"):
        monkeypatch.setattr(measure, name, dict)
    monkeypatch.setattr(measure, "verify_native_artifacts", lambda value: value)
    monkeypatch.setattr(measure.pool_scan, "snapshot", dict)
    monkeypatch.setattr(measure.pool_scan, "finish_case", lambda value: {})
    monkeypatch.setattr(measure, "check_request", lambda *args: None)
    monkeypatch.setattr(measure, "effective_work", lambda *args: {})
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: pytest.fail("bench saved tensors"))
    metadata = {
        "mode": "bench",
        "run_id": "fixture",
        "workload_sha256": "fixture",
        "warmup_traces": {"hbm": [1, 2, 3]},
    }
    measure.run_case(
        backend,
        "hbm",
        SimpleNamespace(requests=[{"request_id": 0}]),
        config,
        tmp_path,
        metadata,
        lambda stage: {},
        Runner,
    )
    assert events == ["sessions_released", "backend_closed"]
    assert not (tmp_path / "numerical").exists()
