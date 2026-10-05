"""CPU checks of capacity, lifecycle, output preservation and failure reporting."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.cache_management.src import capacity_probe as probe


def arguments(*extra):
    return probe.parser().parse_args(
        [
            "--run-id",
            "cpu_check",
            "--sparse-pool-tokens",
            "32768",
            "--host-arena-tokens",
            "1050624",
            *extra,
        ]
    )


def test_capacity_uses_padded_history_and_never_changes_pools():
    config = probe.capacity_config(arguments())
    assert config["users"] == 16
    assert config["requests"] == 32
    assert config["padded_session_tokens"] == config["retained_capacity_tokens"] == 65536
    assert config["capacity_tokens"] == 65664
    assert config["candidate_persistence"] == "gpu_transient"
    assert config["unused_host_tokens_at_full_admission"] == 2048
    args = arguments("--history-tokens", "65537", "--host-arena-tokens", "1050688")
    config = probe.capacity_config(args)
    assert config["padded_session_tokens"] == 65600
    assert config["users"] == 16
    assert config["sparse_pool_tokens"] == 32768
    assert config["host_arena_tokens"] == 1050688
    assert "workspace_query_tokens" not in config


@pytest.mark.parametrize(
    "options,match",
    [
        (["--rounds", "1"], "two complete"),
        (["--host-arena-tokens", "65472"], "one complete"),
        (["--host-arena-tokens", "1000"], "multiple of 64"),
        (["--sparse-pool-tokens", "64"], "query batch"),
    ],
)
def test_invalid_capacity_is_rejected_before_execution(options, match):
    with pytest.raises(ValueError, match=match):
        probe.capacity_config(arguments(*options))


def small_case():
    config = probe.capacity_config(
        arguments("--history-tokens", "60", "--candidate-tokens", "4", "--host-arena-tokens", "128")
    )
    requests = tuple(
        {"request_id": i, "user_id": i % 2, "input_sha256": f"tokens-{i}"} for i in range(4)
    )
    return config, SimpleNamespace(requests=requests)


def metrics(config, index):
    return {
        "user_id": index % 2,
        "visit_index": index // 2,
        "is_revisit": index >= 2,
        "prefix_cache_hit": index >= 2,
        "evicted_users": [],
        "cached_users": min(index + 1, 2),
        "cache_host_pages": min(index + 1, 2),
        "host_page_capacity": 2,
        "stable_prefix_tokens": config["history_tokens"],
        "candidate_suffix_tokens": config["candidate_tokens"],
        "hbm_budget_bytes": None,
        "dram_budget_bytes": None,
        "resource_mode": "fixed_pools",
        "cache_diagnostics": {
            "candidate_persistence": "gpu_transient",
            "retained_length": config["history_tokens"],
            "candidate_device_to_host_bytes": 0,
        },
    }


@pytest.mark.parametrize(
    "change",
    [
        {"prefix_cache_hit": False},
        {"evicted_users": [0]},
        {"cached_users": 1},
        {"cache_host_pages": 1},
    ],
)
def test_revisit_must_hit_and_retain_the_full_host_population(change):
    config, workload = small_case()
    with pytest.raises(AssertionError):
        probe.check_lifecycle(workload.requests[2], {**metrics(config, 2), **change}, config, 2)


class FakeRunner:
    def __init__(self, config):
        self.config = config
        self.pool = [0, 1]
        self.backend = SimpleNamespace(
            cfg=SimpleNamespace(dim=2, vocab_size=3), last_logits=torch.ones((1, 3))
        )

    def execute(self, request):
        return SimpleNamespace(
            metrics=metrics(self.config, request["request_id"]),
            hidden=torch.ones((self.config["candidate_tokens"], 2), dtype=torch.bfloat16),
        )


def test_complete_loop_saves_every_full_output_without_claiming_numerical_acceptance(tmp_path):
    config, workload = small_case()
    metadata = {"completed_requests": 0}
    status = probe.run_requests(
        FakeRunner(config), workload, config, tmp_path, metadata, lambda _: None
    )
    assert status == "unverified"
    assert metadata["completed_requests"] == 4
    assert metadata["max_retained_users"] == 2
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    for index, row in enumerate(rows):
        saved = torch.load(tmp_path / row["output_file"], weights_only=True)
        assert saved["hidden"].shape == (4, 2)
        assert saved["hidden"].dtype == torch.bfloat16
        assert saved["logits"].shape == (1, 3)
        assert saved["input_sha256"] == f"tokens-{index}"
        assert row["numerical"]["status"] == "unverified"


def test_nonfinite_output_is_not_treated_as_success():
    with pytest.raises(AssertionError, match="nonfinite"):
        probe.tensor_summary(torch.tensor([float("nan")]))


def test_plan_must_preserve_requested_capacities():
    config = probe.capacity_config(arguments())
    plan = SimpleNamespace(
        metadata={name: config[name] for name in ("host_arena_tokens", "sparse_pool_tokens")}
        | {
            "max_session_capacity": config["capacity_tokens"],
            "max_history_tokens": config["retained_capacity_tokens"],
            "max_candidate_tokens": config["candidate_tokens"],
            "candidate_slots": config["candidate_tokens"],
            "candidate_persistence": "gpu_transient",
        },
        host_pages=config["host_arena_tokens"] // 64,
    )
    probe.check_plan(plan, config)
    plan.metadata["host_arena_tokens"] -= 64
    with pytest.raises(AssertionError, match="changed the requested"):
        probe.check_plan(plan, config)


def test_reference_requires_independent_hbm_and_exact_input_identity(tmp_path):
    binding = {"source_sha256": "source"}
    manifest = {
        "schema": probe.REFERENCE_SCHEMA,
        "scheme": "echo",
        "status": "complete",
        "binding": binding,
    }
    probe.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="independent HBM"):
        probe.load_reference(tmp_path, binding)
    manifest["scheme"] = "hbm"
    probe.write_json(tmp_path / "manifest.json", manifest)
    accepted = probe.load_reference(tmp_path, binding)
    payload = {"input_sha256": "ids", "hidden": torch.ones(2, 3), "logits": torch.ones(1, 4)}
    assert probe.compare_reference(tmp_path, accepted, 0, payload)["status"] == "unverified"
    torch.save(payload, tmp_path / "000000.pt")
    assert probe.compare_reference(tmp_path, accepted, 0, payload)["status"] == "passed"
    with pytest.raises(AssertionError, match="token identity"):
        probe.compare_reference(tmp_path, accepted, 0, {**payload, "input_sha256": "other"})
    with pytest.raises(AssertionError):
        probe.compare_reference(tmp_path, accepted, 0, {**payload, "hidden": torch.zeros(2, 3)})


def saved_case(tmp_path):
    output, reference = tmp_path / "run", tmp_path / "reference"
    output.mkdir()
    reference.mkdir()
    config, workload = small_case()
    metadata = {"completed_requests": 0}
    probe.run_requests(FakeRunner(config), workload, config, output, metadata, lambda _: None)
    source = output / "source"
    source.mkdir()
    (source / "fixture.py").write_text("source bytes\n")
    source_manifest = {"fixture.py": hashlib.sha256(b"source bytes\n").hexdigest()}
    probe.write_json(output / "source_manifest.json", source_manifest)
    source_digest = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest()
    metadata.update(
        status="complete",
        capacity_passed=True,
        config=config,
        source_sha256=source_digest,
        checkpoint={"path": "fixture checkpoint"},
        run_id="fixture",
        numerical_status="unverified",
    )
    probe.write_json(output / "status.json", metadata)
    binding = probe.reference_binding(config, source_digest, metadata["checkpoint"], "tokenizer")
    manifest = {
        "schema": probe.REFERENCE_SCHEMA,
        "scheme": "echo",
        "status": "complete",
        "binding": binding,
    }
    probe.write_json(output / "numerical/manifest.json", manifest)
    probe.write_json(reference / "manifest.json", {**manifest, "scheme": "hbm"})
    for path in (output / "numerical").glob("*.pt"):
        (reference / path.name).write_bytes(path.read_bytes())
    return output, reference


def test_existing_run_is_audited_from_saved_full_outputs_without_cuda(monkeypatch, tmp_path):
    output, reference = saved_case(tmp_path)
    requests_before = (output / "requests.jsonl").read_bytes()

    def unexpected_cuda(*_, **__):
        pytest.fail("post-run numerical audit must not initialize a model or CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_cuda)
    assert probe.main(["--audit-existing", str(output), "--reference-dir", str(reference)]) == 0
    audit = json.loads((output / "numerical_audit.json").read_text())
    assert audit["status"] == "passed"
    assert len(audit["requests"]) == 4
    state = json.loads((output / "status.json").read_text())
    assert state["numerical_status"] == "passed"
    assert state["capacity_passed"] is True
    assert (output / "requests.jsonl").read_bytes() == requests_before


@pytest.mark.parametrize(
    "damage", ["source", "saved_output", "reference_output", "missing_reference"]
)
def test_existing_run_audit_rejects_modified_or_missing_evidence(tmp_path, damage):
    output, reference = saved_case(tmp_path)
    if damage == "source":
        (output / "source/fixture.py").write_text("different source\n")
    elif damage == "saved_output":
        path = output / "numerical/000000.pt"
        payload = torch.load(path, weights_only=True)
        payload["hidden"].zero_()
        torch.save(payload, path)
    elif damage == "reference_output":
        path = reference / "000000.pt"
        payload = torch.load(path, weights_only=True)
        payload["hidden"].zero_()
        torch.save(payload, path)
    else:
        (reference / "000000.pt").unlink()
    assert probe.audit_existing(output, reference) == 1
    audit = json.loads((output / "numerical_audit.json").read_text())
    assert audit["status"] == "failed"
    assert audit["failure"]["type"] == "AssertionError"
    state = json.loads((output / "status.json").read_text())
    assert state["numerical_status"] == "failed"


def test_model_oom_remains_a_failed_temporary_artifact(monkeypatch, tmp_path):
    from evaluation import provenance as measure
    from GR import workload

    temporary = tmp_path / "temporary"
    temporary.mkdir()
    target = tmp_path / "published"
    monkeypatch.setattr(probe.tempfile, "mkdtemp", lambda **_: str(temporary))
    monkeypatch.setattr(measure, "source_snapshot", lambda *_, **__: "snapshot")
    monkeypatch.setattr(measure, "backend_provenance", dict)
    monkeypatch.setattr(measure, "_git", lambda *_: "revision")
    monkeypatch.setattr(probe, "memory_sample", lambda _, stage: {"stage": stage})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="mock", uuid="mock", total_memory=1, multi_processor_count=1
        ),
    )
    monkeypatch.setattr(
        workload,
        "build_workload",
        lambda *_, **__: SimpleNamespace(
            manifest={"tokenizer_sha256": "tokenizer", "workload_sha256": "workload"},
            write=lambda _: None,
        ),
    )

    def fail(*_, **__):
        raise torch.OutOfMemoryError("deliberate model allocation failure")

    monkeypatch.setitem(
        sys.modules,
        "models.deepseek_v32.execution.adapter",
        SimpleNamespace(DeepSeekServingBackend=fail),
    )
    result = probe.main(
        [
            "--run-id",
            "oom",
            "--output-dir",
            str(target),
            "--model-path",
            str(tmp_path),
            "--sparse-pool-tokens",
            "32768",
            "--host-arena-tokens",
            "1050624",
        ]
    )
    assert result == 1
    assert not target.exists()
    state = json.loads((temporary / "status.json").read_text())
    assert state["status"] == "failed"
    assert state["capacity_passed"] is False
    assert state["failure"]["stage"] == "model_loading"
    assert state["failure"]["type"] == "OutOfMemoryError"
    assert state["config"]["host_arena_tokens"] == 1050624
    assert state["config"]["sparse_pool_tokens"] == 32768


def test_candidate_capacity_does_not_reserve_host_pages():
    small = probe.capacity_config(
        arguments("--history-tokens", "64", "--candidate-tokens", "8", "--host-arena-tokens", "128")
    )
    large = probe.capacity_config(
        arguments(
            "--history-tokens", "64", "--candidate-tokens", "128", "--host-arena-tokens", "128"
        )
    )
    assert small["users"] == large["users"] == 2
    assert small["padded_session_tokens"] == large["padded_session_tokens"] == 64
    assert large["capacity_tokens"] == 192 and large["retained_capacity_tokens"] == 64


@pytest.mark.parametrize(
    "change",
    [
        {"candidate_persistence": "committed"},
        {"retained_length": 64},
        {"candidate_device_to_host_bytes": 1152},
    ],
)
def test_transient_candidate_requires_discarded_length_and_zero_actual_host_writes(change):
    config, workload = small_case()
    value = metrics(config, 2)
    value["cache_diagnostics"].update(change)
    with pytest.raises(AssertionError, match="candidate diagnostic"):
        probe.check_lifecycle(workload.requests[2], value, config, 2)


def test_reference_binding_distinguishes_transient_candidates_from_legacy_outputs():
    config, _ = small_case()
    current = probe.reference_binding(config, "source", {"path": "checkpoint"}, "tokenizer")
    legacy = {
        key: value
        for key, value in config.items()
        if key not in ("candidate_persistence", "retained_capacity_tokens")
    }
    old = probe.reference_binding(legacy, "source", {"path": "checkpoint"}, "tokenizer")
    assert current["candidate_persistence"] == "gpu_transient"
    assert current["retained_capacity_tokens"] == 60
    assert "candidate_persistence" not in old
    assert old != current
