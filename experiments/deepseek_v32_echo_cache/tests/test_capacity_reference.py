"""CPU checks of independent per-user HBM replay and evidence binding."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_echo_cache.src import capacity_reference as reference
from experiments.deepseek_v32_echo_cache.src.capacity_probe import (
    REFERENCE_SCHEMA,
    reference_binding,
    tensor_summary,
    write_json,
)
from GR.workload import token_sha256


def make_run(tmp_path, *, changed_prefix=False):
    run = tmp_path / "echo"
    (run / "numerical").mkdir(parents=True)
    (run / "workload").mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {
        "scheme": "echo",
        "num_layers": 10,
        "chunk_size": 1024,
        "history_tokens": 3,
        "candidate_tokens": 2,
        "candidate_persistence": "gpu_transient",
        "retained_capacity_tokens": 3,
        "capacity_tokens": 5,
        "padded_session_tokens": 64,
        "host_arena_tokens": 128,
        "sparse_pool_tokens": 32768,
        "users": 2,
        "rounds": 2,
        "requests": 4,
    }
    requests, identities, outputs = [], [], []
    for index in range(4):
        user, visit = index % 2, index // 2
        prefix, candidate = [1 + user, 2 + user, 3 + user], [10 + index, 11 + index]
        if changed_prefix and index == 2:
            prefix[0] += 1
        ids = prefix + candidate
        identity = {
            "request_id": index,
            "user_id": user,
            "visit_index": visit,
            "stable_prefix_tokens": 3,
            "candidate_suffix_tokens": 2,
            "prefix_sha256": token_sha256(prefix),
            "candidate_sha256": token_sha256(candidate),
            "input_sha256": token_sha256(ids),
        }
        identities.append(identity)
        requests.append({**identity, "input_ids": ids})
        hidden, hidden_summary = tensor_summary(torch.full((2, 3), sum(ids), dtype=torch.bfloat16))
        logits, logits_summary = tensor_summary(torch.full((1, 5), float(sum(ids))))
        payload = {
            "request_id": index,
            "input_sha256": token_sha256(ids),
            "hidden": hidden,
            "logits": logits,
        }
        path = run / "numerical" / f"{index:06d}.pt"
        torch.save(payload, path)
        outputs.append(
            {
                "request_id": index,
                "input_sha256": token_sha256(ids),
                "hidden": hidden_summary,
                "logits": logits_summary,
                "output_file": str(path.relative_to(run)),
                "output_file_sha256": reference.digest(path),
                "metrics": {
                    "user_id": user,
                    "visit_index": visit,
                    "is_revisit": visit > 0,
                    "prefix_cache_hit": visit > 0,
                    "evicted_users": [],
                    "cached_users": min(index + 1, 2),
                    "cache_host_pages": min(index + 1, 2),
                    "host_page_capacity": 2,
                    "stable_prefix_tokens": 3,
                    "candidate_suffix_tokens": 2,
                    "hbm_budget_bytes": None,
                    "dram_budget_bytes": None,
                    "resource_mode": "fixed_pools",
                    "cache_diagnostics": {
                        "candidate_persistence": "gpu_transient",
                        "retained_length": 3,
                        "candidate_device_to_host_bytes": 0,
                    },
                },
            }
        )
    identity = {
        "config": config,
        "heat_sha256": None,
        "tokenizer_sha256": "fixture",
        "requests": identities,
    }
    workload_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    write_json(run / "workload/workload.json", {**identity, "workload_sha256": workload_digest})
    for path, rows in (
        (run / "workload/requests.jsonl", requests),
        (run / "requests.jsonl", outputs),
    ):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(run / "source_manifest.json", {})
    source_digest = hashlib.sha256(b"{}").hexdigest()
    state = {
        "status": "complete",
        "capacity_passed": True,
        "run_id": "fixture",
        "config": config,
        "source_sha256": source_digest,
        "workload_sha256": workload_digest,
        "checkpoint": {"path": str(checkpoint.resolve()), "files": {}},
        "backend_provenance": {},
        "backend": {"linear_backend": "fp8"},
        "hardware": {"device": "cuda:0"},
    }
    write_json(run / "status.json", state)
    binding = reference_binding(config, source_digest, state["checkpoint"], "fixture")
    write_json(
        run / "numerical/manifest.json",
        {"schema": REFERENCE_SCHEMA, "scheme": "echo", "status": "complete", "binding": binding},
    )
    return run


class FakeBackend:
    def __init__(self, *, fail_extend=False):
        self.cfg = SimpleNamespace(dim=3, vocab_size=5)
        self.last_logits = None
        self.active = None
        self.events = []
        self.fail_extend = fail_extend
        self.closed = False

    def create_session(self, capacity):
        assert self.active is None
        assert capacity == 5
        self.active = SimpleNamespace(length=0, prefix=None)
        self.events.append("create")
        return self.active

    def prefill(self, session, token_ids):
        assert session is self.active and session.length == 0
        assert len(token_ids) == 3
        session.prefix = list(token_ids)
        session.length = 3
        self.events.append(("prefill", tuple(token_ids)))

    def extend(self, session, token_ids):
        assert session is self.active and session.length == 3
        if self.fail_extend:
            raise torch.OutOfMemoryError("deliberate reference extension failure")
        self.events.append(("extend", tuple(token_ids)))
        value = sum(session.prefix) + sum(token_ids)
        session.length = 5
        self.last_logits = torch.full((1, 5), float(value))
        return torch.full((2, 3), value, dtype=torch.bfloat16)

    def truncate(self, session, length):
        assert session is self.active and session.length == 5 and length == 3
        session.length = length
        self.events.append("truncate")

    def release_session(self, session):
        assert session is self.active
        self.active = None
        self.events.append("release")

    def describe(self):
        return {"scheme": "hbm"}

    def close(self):
        assert self.active is None
        self.closed = True


def test_reference_replays_one_empty_user_session_at_a_time(tmp_path):
    run = make_run(tmp_path)
    state, _, groups, results = reference.load_run(run)
    output = tmp_path / "reference"
    output.mkdir()
    backend = FakeBackend()
    metadata = {"completed_users": 0, "completed_requests": 0}
    reference.run_users(backend, run, output, state, groups, results, metadata)
    assert backend.active is None
    assert metadata["completed_users"] == 2
    assert metadata["completed_requests"] == 4
    assert backend.events == [
        "create",
        ("prefill", (1, 2, 3)),
        ("extend", (10, 11)),
        "truncate",
        ("extend", (12, 13)),
        "truncate",
        "release",
        "create",
        ("prefill", (2, 3, 4)),
        ("extend", (11, 12)),
        "truncate",
        ("extend", (13, 14)),
        "truncate",
        "release",
    ]
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert [row["request_id"] for row in rows] == [0, 2, 1, 3]
    assert all(row["comparison"]["hidden"]["bitwise_equal"] for row in rows)
    for index in range(4):
        payload = torch.load(output / f"{index:06d}.pt", weights_only=True)
        assert payload["request_id"] == index
        assert payload["hidden"].shape == (2, 3)
        assert payload["logits"].shape == (1, 5)


def test_even_consistently_resigned_workload_cannot_change_a_user_prefix(tmp_path):
    run = make_run(tmp_path, changed_prefix=True)
    with pytest.raises(AssertionError, match="prefix changed across visits"):
        reference.load_run(run)


def test_modified_token_ids_fail_identity_verification(tmp_path):
    run = make_run(tmp_path)
    path = run / "workload/requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["input_ids"][0] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(AssertionError, match="prefix_sha256 differs"):
        reference.load_run(run)


def test_saved_echo_corruption_is_detected_and_session_is_released(tmp_path):
    run = make_run(tmp_path)
    state, _, groups, results = reference.load_run(run)
    output = tmp_path / "reference"
    output.mkdir()
    (run / "numerical/000000.pt").write_bytes(b"corrupted")
    backend = FakeBackend()
    with pytest.raises(AssertionError, match="output file digest differs"):
        reference.run_users(
            backend,
            run,
            output,
            state,
            groups,
            results,
            {"completed_users": 0, "completed_requests": 0},
        )
    assert backend.active is None
    assert backend.events[-1] == "release"
    assert (output / "000000.pt").is_file()


@pytest.mark.parametrize("fail_extend", [False, True])
def test_main_publishes_only_complete_independent_reference(monkeypatch, tmp_path, fail_extend):
    from evaluation import provenance as measure

    run = make_run(tmp_path)
    temporary, target = tmp_path / "temporary", tmp_path / "published"
    temporary.mkdir()
    backend = FakeBackend(fail_extend=fail_extend)
    monkeypatch.setattr(reference.tempfile, "mkdtemp", lambda **_: str(temporary))
    monkeypatch.setattr(measure, "backend_provenance", dict)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="fake", uuid="fake", total_memory=1, multi_processor_count=1
        ),
    )

    def create_backend(*_, **kwargs):
        assert kwargs["scheme"] == "hbm"
        assert kwargs["num_layers"] == 10
        assert kwargs["chunk_size"] == 1024
        return backend

    monkeypatch.setitem(
        sys.modules,
        "models.deepseek_v32.serving_backend",
        SimpleNamespace(DeepSeekServingBackend=create_backend),
    )
    result = reference.main(["--run-dir", str(run), "--output-dir", str(target)])
    assert backend.closed
    if fail_extend:
        assert result == 1
        assert not target.exists()
        status = json.loads((temporary / "status.json").read_text())
        assert status["status"] == "failed"
        assert status["failure"]["type"] == "OutOfMemoryError"
        assert not (temporary / "manifest.json").exists()
    else:
        assert result == 0
        assert not temporary.exists()
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["scheme"] == "hbm"
        assert manifest["status"] == "complete"
        assert manifest["requests"] == 4
        assert (
            manifest["binding"]
            == json.loads((run / "numerical/manifest.json").read_text())["binding"]
        )
        assert "not a latency" in manifest["measurement_boundary"]
