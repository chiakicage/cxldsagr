import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cache.prefix_pool import CacheFootprint
from experiments.gr_serving.src.cache_memory_capture import (
    CHUNKS,
    PROFILE_REQUESTS,
    ServingMemorySampler,
    cache_tensors,
    inventory_and_limits,
    parser,
    screen_cases,
    selected_schemes,
    validate_completed_matrix,
)


def config():
    return SimpleNamespace(
        max_seq_len=163840,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_topk=2048,
    )


def test_fixed_geometry_screen_exposes_chunk_dependent_sixteen_user_admission():
    plans = screen_cases(config())
    assert len(plans) == 16
    assert all(row["feasible"] for row in plans)
    capacities = {(row["chunk"], row["scheme"]): row["max_cached_users"] for row in plans}
    assert [capacities[c, "echo"] for c in CHUNKS] == [16, 16, 16, 15]
    assert [capacities[c, "serial_sparse"] for c in CHUNKS] == [16, 16, 16, 15]
    assert [capacities[c, "hbm"] for c in CHUNKS] == [4, 4, 3, 2]
    # The two dense slots now belong to the backend and are charged once.
    assert [capacities[c, "dense_prefetch"] for c in CHUNKS] == [16, 16, 16, 16]
    for row in (row for row in plans if row["scheme"] == "dense_prefetch"):
        assert row["resource_plan"]["dense_staging_scope"] == "backend"
        assert row["resource_plan"]["dense_staging_bytes"] == 2 * 65664 * 576 * 2
        assert row["shared"]["hbm"] + 16 * row["session"]["hbm"] <= 4 * 2**30
    assert PROFILE_REQUESTS == (0, 15, 16, 31)


def test_capture_cli_defaults_use_requested_checkpoint_and_all_chunks():
    args = parser().parse_args(["--run-id", "audit", "--output-dir", "/tmp/audit", "--plan-only"])
    assert args.model == Path("/preset-models")
    assert args.chunks == list(CHUNKS)
    assert args.schemes == ["hbm", "echo", "serial_sparse", "dense_prefetch"]
    assert args.plan_only


def test_subset_schemes_keep_same_chunk_resident_reference_and_exact_case_matrix():
    plans = screen_cases(config(), chunks=(512,), schemes=("hbm",))
    assert [(row["chunk"], row["scheme"]) for row in plans] == [(512, "hbm")]
    assert selected_schemes(("dense_prefetch", "hbm")) == ("hbm", "dense_prefetch")
    plans = screen_cases(config(), chunks=(256,), schemes=("dense_prefetch", "hbm"))
    assert [(row["chunk"], row["scheme"]) for row in plans] == [
        (256, "hbm"),
        (256, "dense_prefetch"),
    ]


@pytest.mark.parametrize("schemes", [(), ("echo",), ("hbm", "hbm"), ("hbm", "unknown")])
def test_invalid_or_unreferenced_subsets_fail_before_gpu_execution(schemes):
    with pytest.raises(ValueError, match="containing hbm reference"):
        selected_schemes(schemes)


def test_completion_rejects_missing_extra_or_short_scheme_cases():
    expected = [{"chunk": 256, "scheme": scheme} for scheme in ("hbm", "dense_prefetch")]
    complete = [{**row, "requests": 32} for row in expected]
    validate_completed_matrix(complete, expected, requests=32)
    for incomplete in (
        complete[:1],
        complete + [{"chunk": 256, "scheme": "echo", "requests": 32}],
        [complete[0], {**complete[1], "requests": 31}],
    ):
        with pytest.raises(RuntimeError, match="full case/request matrix"):
            validate_completed_matrix(incomplete, expected, requests=32)


def test_plan_only_persists_requested_parameters_and_canonical_expected_matrix(
    tmp_path, monkeypatch
):
    from experiments.gr_serving.src import cache_memory_capture as capture

    monkeypatch.setattr(capture.Config, "from_checkpoint", lambda _: config())
    output = tmp_path / "selected"
    capture.main(
        [
            "--plan-only",
            "--run-id",
            "subset",
            "--output-dir",
            str(output),
            "--chunks",
            "256",
            "--schemes",
            "dense_prefetch",
            "hbm",
        ]
    )
    saved = json.loads((output / "configuration.json").read_text())
    assert saved["parameters"]["schemes"] == ["dense_prefetch", "hbm"]
    assert saved["schemes"] == ["hbm", "dense_prefetch"]
    assert saved["expected_case_matrix"] == [
        {"chunk": 256, "scheme": "hbm"},
        {"chunk": 256, "scheme": "dense_prefetch"},
    ]
    assert len(json.loads((output / "feasibility.json").read_text())) == 2


def test_fixed_workspace_protocol_preserves_identical_admission_across_chunks():
    plans = screen_cases(config(), workspace_query_tokens=2048)
    echoes = [row for row in plans if row["scheme"] == "echo"]
    assert {row["max_cached_users"] for row in echoes} == {15}
    assert len({row["shared"]["hbm"] for row in echoes}) == 1
    assert all(row["resource_plan"]["workspace_query_tokens"] == 2048 for row in plans)
    with pytest.raises(ValueError, match="must cover"):
        screen_cases(config(), workspace_query_tokens=1024)


def test_independent_inventory_rejects_backend_byte_omission(monkeypatch):
    from experiments.gr_serving.src import cache_memory_capture as capture

    backend = SimpleNamespace(device=torch.device("cuda:0"))
    runner = SimpleNamespace(
        pool=SimpleNamespace(
            audit=lambda: CacheFootprint(400, 20), reserved=CacheFootprint(1000, 20)
        )
    )
    monkeypatch.setattr(capture, "cache_tensors", lambda _: [])
    inventory = {
        "devices": {
            "cuda:0": {"storage_bytes": 400, "allocator_bytes": 512},
            "cpu": {"storage_bytes": 20, "allocator_bytes": 20},
        },
        "storages": [],
    }
    monkeypatch.setattr(capture, "storage_inventory", lambda _: inventory)
    actual, limits = inventory_and_limits(backend, runner)
    assert actual is inventory
    assert limits == {"cuda:0": 488, "cpu": 0}
    inventory["devices"]["cpu"]["allocator_bytes"] = 32
    with pytest.raises(RuntimeError, match="differs from backend ledger"):
        inventory_and_limits(backend, runner)
    actual, limits = inventory_and_limits(backend, runner, observe_cpu_accounting=True)
    assert limits["cpu"] == 0
    assert not actual["backend_ledger_matches"]
    assert actual["fixed_reservation_deficit_bytes"]["cpu"] == 12


def test_independent_inventory_includes_shared_dense_stage_and_pending_sources():
    from experiments.gr_serving.src.cache_memory_audit import storage_inventory

    stage = torch.empty((2, 64, 8), dtype=torch.bfloat16)
    source = torch.empty((7, 8), dtype=torch.bfloat16)
    histories = [torch.empty((64, 8), dtype=torch.bfloat16) for _ in range(2)]
    sessions = []
    for history in histories:
        cache = SimpleNamespace(
            records=None, host=history, host_to_device=None, device_to_host=None, age=None
        )
        sessions.append(
            SimpleNamespace(
                stages=[],
                prefix_offsets=[],
                sparse_session=None,
                runners=[
                    SimpleNamespace(index_keys=None, index_scales=None, offset=None, cache=cache)
                ],
            )
        )
    backend = SimpleNamespace(
        _dense_staging=SimpleNamespace(_buffers={"records": stage}),
        _dense_sources=[SimpleNamespace(source=source)],
        _shared_pool=None,
        _sessions=sessions,
    )
    tensors = cache_tensors(backend)
    assert len(tensors) == 4
    assert {tensor.data_ptr() for tensor in tensors} == {
        tensor.data_ptr() for tensor in (stage, source, *histories)
    }
    inventory = storage_inventory(tensors)
    assert inventory["devices"]["cpu"]["storage_bytes"] == sum(
        tensor.numel() * tensor.element_size() for tensor in (stage, source, *histories)
    )


def test_sampler_restores_all_bound_methods_and_only_instruments_selected_requests(
    tmp_path, monkeypatch
):
    class Backend:
        def _forward(self, session, tokens, *, scope=None):
            return (session, tokens, scope)

        def prefill(self, session, tokens):
            return self._forward(session, tokens)

        extend = prefill

        def truncate(self, session, tokens):
            return (session, tokens)

        def session_metrics(self, session):
            return {"session": session}

    backend = Backend()
    original = {
        name: getattr(backend, name)
        for name in ("_forward", "prefill", "extend", "truncate", "session_metrics")
    }
    sampler = ServingMemorySampler(
        backend, None, chunk=256, data_dir=tmp_path, profile_dir=tmp_path, history_max_entries=20
    )
    with sampler.installed():
        sampler.request_id = 4
        assert backend.prefill("session", "tokens") == ("session", "tokens", None)
        sampler.request_id = 0
        calls = []
        monkeypatch.setattr(
            sampler, "capture", lambda phase, fn, *args, **kwargs: calls.append(phase)
        )
        backend.prefill("session", "tokens")
        backend.extend("session", "tokens")
        backend.truncate("session", "tokens")
        backend.session_metrics("session")
        assert calls == ["prefill", "extend", "truncate", "session_metrics"]
    assert not vars(backend)
    assert all(getattr(backend, name) == value for name, value in original.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA in-place truncate requires GPU")
def test_sampler_accepts_zero_allocation_truncate_with_real_copy_evidence(tmp_path, monkeypatch):
    from experiments.gr_serving.src import cache_memory_capture as capture

    source = torch.ones(16, dtype=torch.int32, device="cuda:0")
    target = torch.zeros_like(source)
    monkeypatch.setattr(capture, "cache_tensors", lambda _: [source, target])
    backend = SimpleNamespace(device=torch.device("cuda:0"), scheme="dense_prefetch", attentions=[])
    runner = SimpleNamespace(
        pool=SimpleNamespace(
            audit=lambda: CacheFootprint(128, 0), reserved=CacheFootprint(2048, 40)
        )
    )
    sampler = ServingMemorySampler(
        backend,
        runner,
        chunk=1024,
        data_dir=tmp_path / "data",
        profile_dir=tmp_path / "profile",
        history_max_entries=1000,
    )
    sampler.request_id = 0
    sampler.capture("truncate", lambda: target.copy_(source))
    report = json.loads((tmp_path / "data/request_00_truncate.json").read_text())
    assert report["passed_observed_temporary_bound"], report["evidence_errors"]
    assert report["memory_event_count"] == 0
    assert report["devices"]["cuda:0"]["observed_allocator_active_peak_bytes"] == 0
    assert report["devices"]["cpu"]["observed_allocator_active_peak_bytes"] == 0
