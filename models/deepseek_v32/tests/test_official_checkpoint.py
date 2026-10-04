"""Opt-in full 64K checkpoint cache check; all diagnostics stay in the terminal.

This intrusive correctness test retains independent GPU mirrors of the freshly
projected KV. It checks every selected record, and compares the actual FlashMLA
consumer with the same queries and the same ordered logical indices. It never
re-runs top-k or derives its oracle from host backing or recalled records.

Run from the repository root, for example:
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 DEEPSEEK_OFFICIAL_CHECKPOINT=/preset-models \
    .venv/bin/python -m pytest -s -q models/deepseek_v32/tests/test_official_checkpoint.py
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import MethodType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
HISTORY = 65536
CANDIDATE = 128
CHUNK = 1024
LAYERS = 10
POOL = 65536
HOST = 16777216
TOPK = 2048


def _source_identity():
    paths = {Path(__file__).resolve()}
    for directory in (
        "models/deepseek_v32",
        "cache",
        "operators/deepseek_v32",
        "operators/common",
        "layers",
        "executor",
    ):
        paths.update(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file()
            and "tests" not in path.parts
            and path.suffix in (".py", ".cu", ".cuh", ".cpp", ".h", ".hpp")
        )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _assert_same_bytes(actual, expected, *, stage, location):
    if not torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ):
        difference = (actual.float() - expected.float()).abs()
        pytest.fail(
            f"{stage} mismatch at {location}: max_abs={difference.max().item()}, "
            f"mean_abs={difference.mean().item()}; exact byte equality is required"
        )


class _ProjectionMirror:
    def __init__(self, runner, *, user, layer, totals):
        self.runner, self.user, self.layer, self.totals = runner, user, layer, totals
        self.records = torch.empty(
            (HISTORY + CANDIDATE, 576), device=runner.cache.device, dtype=torch.bfloat16
        )
        self.history_end = 0
        self.visible_end = 0
        self.calls = 0

    def capture(self, projection, index_keys, index_scales, indices, cache, position):
        del index_keys, index_scales, indices
        assert cache is self.runner.cache
        count = projection.kv.shape[0]
        if position < HISTORY:
            assert position == self.history_end
            assert count == CHUNK and position + count <= HISTORY
            self.history_end = position + count
        else:
            assert position == self.history_end == HISTORY and count == CANDIDATE
        # This source exists before any comparison with cache/host state. Each
        # layer/user owns separate storage, including a separate candidate tail.
        self.records[position : position + count].copy_(projection.kv)
        self.visible_end = position + count

    def consume(self, q, indices, scope):
        from operators.deepseek_v32.attention.device_only.mla import sparse_mla
        from operators.deepseek_v32.attention.offload.mla import sparse_mla_from_pool

        cache = self.runner.cache
        location = f"user={self.user}, layer={self.layer}, end={self.visible_end}"
        with scope("offload_exact_recall"):
            physical = cache.ensure(indices)
        valid = indices >= 0
        assert torch.equal(physical >= 0, valid), location
        logical_ids = indices[valid].long()
        physical_ids = physical[valid].long()
        assert logical_ids.numel() and logical_ids.max().item() < self.visible_end, location
        assert physical_ids.max().item() < cache.records.shape[0], location
        selected = cache.records[physical_ids]
        expected = self.records[logical_ids]
        _assert_same_bytes(selected, expected, stage="selected KV", location=location)
        self.totals["selected_record_occurrences"] += logical_ids.numel()
        del selected, expected, logical_ids, physical_ids, valid
        with scope("sparse_mla"):
            actual = sparse_mla_from_pool(
                q, cache.records, physical, self.runner.cfg.attention_scale
            )
        # Preserve top-k order exactly. Any difference now concerns record
        # placement or the consumer, not independent atomic top-k executions.
        reference = sparse_mla(
            q, self.records[: self.visible_end], indices, self.runner.cfg.attention_scale
        )
        _assert_same_bytes(actual, reference, stage="same-order FlashMLA", location=location)
        self.totals["attention_value_occurrences"] += actual.numel()
        self.totals["query_layer_occurrences"] += q.shape[0]
        self.totals["attention_calls"] += 1
        self.calls += 1
        if (
            self.layer == LAYERS - 1
            and self.visible_end <= HISTORY
            and self.visible_end % 8192 == 0
        ):
            print(f"checked user={self.user} history={self.visible_end}/{HISTORY}", flush=True)
        return actual


def _attach(model, session, *, user, totals):
    mirrors = []
    for layer, runner in enumerate(session.runners):
        mirror = _ProjectionMirror(runner, user=user, layer=layer, totals=totals)
        runner.capture_hook = mirror.capture

        def consume_bound(self, q, indices, scope, mirror=mirror):
            assert self is mirror.runner
            return mirror.consume(q, indices, scope)

        runner._consume = MethodType(consume_bound, runner)
        mirrors.append(mirror)
    assert len(mirrors) == LAYERS
    assert len({mirror.records.data_ptr() for mirror in mirrors}) == LAYERS
    return mirrors


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_OFFICIAL_CHECKPOINT"),
    reason="set DEEPSEEK_OFFICIAL_CHECKPOINT for the full 64K official ECHO check",
)
@pytest.mark.parametrize("compute_graphs", [False, True], ids=["eager", "compute_graphs"])
def test_official_full_checkpoint_selected_records_and_same_order_attention(compute_graphs):
    from cache.prefix_pool import CacheFootprint
    from experiments.gr_serving.src.workload import WorkloadConfig, build_workload
    from models.deepseek_v32.official_serving import OfficialDeepSeekServingBackend

    start = time.perf_counter()
    source_before = _source_identity()
    # Explicit opt-in fails if hardware, dependencies, or checkpoint are absent.
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    assert shutil.which("ninja"), "activate the repository .venv so FlashInfer can invoke ninja"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = os.environ["DEEPSEEK_OFFICIAL_CHECKPOINT"]
    print(
        f"official full-checkpoint validation: P={POOL}, NH={HOST}, H={HISTORY}, "
        f"A={CANDIDATE}, compute_graphs={compute_graphs}",
        flush=True,
    )
    model = OfficialDeepSeekServingBackend(
        checkpoint,
        scheme="echo",
        device="cuda:0",
        num_layers=LAYERS,
        chunk_size=CHUNK,
        sparse_pool_tokens=POOL,
        host_arena_tokens=HOST,
        workspace_query_tokens=CHUNK,
        linear_backend="fp8",
        enable_compute_graphs=compute_graphs,
    )
    assert model.parameter_counts["total_parameters"] == 7827793408
    assert len(model.attentions) == len(model.blocks) == LAYERS
    for layer in range(3, LAYERS):
        assert (
            model.attentions[layer].wq_a.weight.data_ptr()
            != model.attentions[layer % 3].wq_a.weight.data_ptr()
        )
        assert (
            model.blocks[layer].mlp.up.weight.data_ptr()
            != model.blocks[layer % 3].mlp.up.weight.data_ptr()
        )
    plan = model.plan_resources(
        None,
        {
            "max_session_capacity": HISTORY + CANDIDATE,
            "max_history_tokens": HISTORY,
            "max_candidate_tokens": CANDIDATE,
        },
    )
    sessions = []
    mirrors = []
    totals = {
        name: 0
        for name in (
            "attention_calls",
            "selected_record_occurrences",
            "attention_value_occurrences",
            "query_layer_occurrences",
        )
    }
    phases = []
    try:
        model.allocate_shared(plan)
        assert CacheFootprint.from_mapping(model.shared_bytes()).fits(plan.shared)
        native_before = model.official_provenance()["source_files"]
        # Use requests 0, 1, and 16 from the same sixteen-user cyclic workload.
        workload = build_workload(
            WorkloadConfig(
                "deepseek_v32",
                16,
                17,
                history_tokens=HISTORY,
                candidate_tokens=CANDIDATE,
                seed=42,
                sampling="sequential",
            ),
            tokenizer=checkpoint,
        )
        for request_index in (0, 1, 16):
            request = workload.requests[request_index]
            user = request["user_id"]
            phase_start = time.perf_counter()
            if request_index < 2:
                session = model.create_session(HISTORY)
                sessions.append(session)
                mirrors.extend(_attach(model, session, user=user, totals=totals))
                assert session.length == 0
                model.prefill(session, request["input_ids"][:HISTORY])
            else:
                session = sessions[0]
                assert request["input_ids"][:HISTORY] == workload.requests[0]["input_ids"][:HISTORY]
            assert session.length == HISTORY
            for runner in session.runners:
                runner.cache.reset_stats()
            hidden = model.extend_candidate(session, request["input_ids"][HISTORY:])
            assert hidden.shape == (CANDIDATE, model.cfg.hidden_size)
            assert torch.isfinite(hidden).all().item()
            assert torch.isfinite(model.last_logits).all().item()
            metrics = model.session_metrics(session)
            assert metrics["candidate_persistence"] == "gpu_transient"
            assert metrics["candidate_device_to_host_bytes"] == 0
            assert session.length == HISTORY and not model._shared_pool._transient_owners
            if request_index == 16:
                assert metrics["host_to_device_bytes"] > 0, "revisit did not exercise host recall"
            phases.append(
                {
                    "request_id": request_index,
                    "user_id": user,
                    "candidate_host_to_device_bytes": metrics["host_to_device_bytes"],
                    "elapsed_s": time.perf_counter() - phase_start,
                }
            )
            print(
                json.dumps(
                    {"phase_complete": phases[-1], "attention_calls": totals["attention_calls"]}
                ),
                flush=True,
            )
        assert len({mirror.records.data_ptr() for mirror in mirrors}) == 2 * LAYERS
        assert all(mirror.history_end == HISTORY for mirror in mirrors)
        assert totals["attention_calls"] == (2 * (HISTORY // CHUNK) + 3) * LAYERS
        assert totals["query_layer_occurrences"] == (2 * HISTORY + 3 * CANDIDATE) * LAYERS
        expected_history = TOPK * (TOPK + 1) // 2 + (HISTORY - TOPK) * TOPK
        assert (
            totals["selected_record_occurrences"]
            == (2 * expected_history + 3 * CANDIDATE * TOPK) * LAYERS
        )
        graph_state = model.describe()["compute_graphs"]
        if compute_graphs:
            assert graph_state["allocated"] and graph_state["eager_fallbacks"] == 0
            assert len(graph_state["graphs"]) == 2 * LAYERS
            assert {(pair["layer"], pair["queries"]) for pair in graph_state["graphs"]} == {
                (layer, queries) for layer in range(LAYERS) for queries in (CHUNK, CANDIDATE)
            }
            for pair in graph_state["graphs"]:
                expected = 2 * (HISTORY // CHUNK) if pair["queries"] == CHUNK else 3
                assert pair["projection_replays"] == pair["finish_replays"] == expected
            for field in ("projection_replays", "finish_replays"):
                assert (
                    sum(pair[field] for pair in graph_state["graphs"]) == totals["attention_calls"]
                )
        else:
            assert not graph_state["enabled"] and not graph_state["allocated"]
        assert _source_identity() == source_before, "production sources changed during validation"
        assert model.official_provenance()["source_files"] == native_before
        print(
            json.dumps(
                {
                    "validation": "passed",
                    "comparison": "byte equality; no nonzero tolerance",
                    "max_selected_record_error": 0,
                    "max_same_order_attention_error": 0,
                    "history_users": 2,
                    "candidate_requests": 3,
                    "physical_layers": LAYERS,
                    "P": POOL,
                    "NH": HOST,
                    "H": HISTORY,
                    "A": CANDIDATE,
                    "chunk": CHUNK,
                    "compute_graphs": graph_state,
                    **totals,
                    "phases": phases,
                    "elapsed_s": time.perf_counter() - start,
                    "source_manifest_sha256": hashlib.sha256(
                        json.dumps(source_before, sort_keys=True).encode()
                    ).hexdigest(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        for session in reversed(sessions):
            if not session.released:
                model.release_session(session)
        model.close()
