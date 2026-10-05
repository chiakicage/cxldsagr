"""Bounded real-checkpoint official callback gate, using the ordinary consumer.

Opt in with DEEPSEEK_OFFICIAL_CHECKPOINT. This H2304 test complements the
H64K selected-record mirror and the common graph ownership/failure suite.
It inherits the official numerical thresholds without recalibrating them.
It creates no reports or persistent test outputs.
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest
import torch

HISTORY = 2304
CHUNK = 256
CANDIDATE = 128
FALLBACK = 121
LAYERS = 10


def _bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().clone()


def _history_snapshot(session):
    return {
        "capacity": session.capacity,
        "layers": [
            (
                id(runner),
                runner.index_keys.data_ptr(),
                runner.index_scales.data_ptr(),
                _bytes(runner.index_keys[:HISTORY]),
                _bytes(runner.index_scales[:HISTORY]),
                _bytes(runner.cache.host_records()[:HISTORY]),
            )
            for runner in session.runners
        ],
    }


def _assert_history(session, snapshot):
    assert session.length == HISTORY and session.capacity == snapshot["capacity"]
    for runner, saved in zip(session.runners, snapshot["layers"], strict=True):
        assert (id(runner), runner.index_keys.data_ptr(), runner.index_scales.data_ptr()) == saved[
            :3
        ]
        for actual, expected in zip(
            (
                runner.index_keys[:HISTORY],
                runner.index_scales[:HISTORY],
                runner.cache.host_records()[:HISTORY],
            ),
            saved[3:],
            strict=True,
        ):
            assert torch.equal(_bytes(actual), expected)


def _graph_snapshot(bank):
    bank._audit_capacity()
    return {
        "identities": {
            key: (
                id(pair.projection_graph),
                id(pair.finish_graph),
                pair.projection_graph.pool(),
                pair.finish_graph.pool(),
            )
            for key, pair in bank.pairs.items()
        },
        "replays": {
            key: (pair.projection_replays, pair.finish_replays) for key, pair in bank.pairs.items()
        },
        "fallbacks": bank.eager_fallbacks,
        "bytes": bank.shared_bytes(),
    }


class _OfficialCalls:
    """Observe real official entry points without replacing their work."""

    def __init__(self, monkeypatch):
        from models.deepseek_v32.cache.official import OfficialSparseTokenCache
        from operators.deepseek_v32.indexer import official

        self.active = None
        self.events = []
        original_module, original_topk = official.module(), official.topk_module()
        probe = self

        class Proxy:
            def __init__(self, target, names):
                self.target, self.names = target, names

            def __getattr__(self, name):
                function = getattr(self.target, name)
                if name not in self.names:
                    return function

                def observed(*args, **kwargs):
                    assert probe.active is not None, "official compute outside a declared phase"
                    probe.events.append((name, len(args[0])))
                    return function(*args, **kwargs)

                return observed

        module_proxy = Proxy(original_module, {"fp8_mqa_logits", "fp8_mqa_logits_fuse_prefetch"})
        topk_proxy = Proxy(original_topk, {"fast_topk_transform"})
        monkeypatch.setattr(official, "module", lambda: module_proxy)
        monkeypatch.setattr(official, "topk_module", lambda: topk_proxy)
        original_ensure = OfficialSparseTokenCache.ensure

        def ensure(cache, indices):
            assert probe.active is not None
            session, candidate = probe.active
            assert any(runner.cache is cache for runner in session.runners)
            assert (cache.transient_start is not None) is candidate
            probe.events.append(("official_ensure", len(indices)))
            return original_ensure(cache, indices)

        monkeypatch.setattr(OfficialSparseTokenCache, "ensure", ensure)

    def begin(self, session, candidate):
        assert self.active is None
        self.active = session, candidate
        return len(self.events)

    def finish(self, start, *, offload, queries):
        self.active = None
        pipeline = (
            ("fp8_mqa_logits_fuse_prefetch", "fast_topk_transform", "official_ensure")
            if offload
            else ("fp8_mqa_logits", "fast_topk_transform")
        )
        expected = [(name, count) for count in queries for _ in range(LAYERS) for name in pipeline]
        observed = self.events[start:]
        assert observed == expected
        return dict(Counter(name for name, _ in observed))


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_OFFICIAL_CHECKPOINT"),
    reason="set DEEPSEEK_OFFICIAL_CHECKPOINT for the bounded official callback GPU gate",
)
@torch.inference_mode()
def test_official_callbacks_keep_consumer_dispatch_streams_and_owned_outputs(monkeypatch):
    from cache.prefix_pool import CacheFootprint
    from experiments.deepseek_v32_echo_official.src.measure import (
        NUMERICAL_POLICY,
        configure_precision,
        numerical_comparison,
        precision_settings,
    )
    from experiments.deepseek_v32_echo_prefill.src.backend_provenance import (
        collect_backend_provenance,
        collect_flashinfer_runtime_artifacts,
    )
    from models.deepseek_v32.attention import EchoAttentionRunner
    from models.deepseek_v32.execution.official import build_official_backend
    from models.deepseek_v32.tests.test_official_checkpoint import _source_identity

    assert torch.cuda.is_available(), "explicit official callback gate requires CUDA"
    assert torch.cuda.get_device_capability() == (9, 0), "SM90/Hopper is required"
    checkpoint = os.environ["DEEPSEEK_OFFICIAL_CHECKPOINT"]
    source_before = _source_identity()
    source_before[str(Path(__file__).resolve())] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    models, sessions, retained, phases = [], [], [], []
    try:
        precision = configure_precision(torch)
        library_before = collect_backend_provenance()
        official_calls = _OfficialCalls(monkeypatch)
        for enabled in (False, True):
            models.append(
                build_official_backend(
                    checkpoint,
                    scheme="hbm",
                    device="cuda:0",
                    num_layers=LAYERS,
                    chunk_size=CHUNK,
                    sparse_pool_tokens=HISTORY,
                    host_arena_tokens=2 * HISTORY,
                    workspace_query_tokens=CHUNK,
                    linear_backend="fp8",
                    enable_compute_graphs=enabled,
                )
            )
        assert all(model.parameter_counts["total_parameters"] == 7827793408 for model in models)
        for left, right in zip(models[0].attentions, models[1].attentions, strict=True):
            assert left.wq_a.weight.data_ptr() != right.wq_a.weight.data_ptr()
        native_before = models[0].pipeline.provenance()["source_files"]
        default = torch.cuda.current_stream()
        alternate = torch.cuda.Stream()
        alternate.wait_stream(default)
        prefixes = (
            [1 + token * 137 % 100000 for token in range(HISTORY)],
            [1 + (token * 131 + 31) % 100000 for token in range(HISTORY)],
        )
        candidates = (
            [2 + token * 53 for token in range(CANDIDATE)],
            [3 + token * 79 for token in range(FALLBACK)],
            [5 + token * 83 for token in range(CANDIDATE)],
        )
        limits = {
            "max_session_capacity": HISTORY + CANDIDATE,
            "max_history_tokens": HISTORY,
            "max_candidate_tokens": CANDIDATE,
        }

        def check_scheme(scheme):
            plans = []
            users = [[], []]
            saved_history = {}
            retained.clear()
            for model in models:
                model.configure_scheme(scheme)
                plan = model.plan_resources(None, limits)
                model.allocate_shared(plan)
                plans.append(plan)
            bank = models[1]._compute_graphs
            original_bank = _graph_snapshot(bank)
            assert set(bank.pairs) == {
                (layer, queries) for layer in range(LAYERS) for queries in (CHUNK, CANDIDATE)
            }

            def check_retained():
                for actual, expected in retained:
                    assert all(
                        torch.equal(_bytes(a), e) for a, e in zip(actual, expected, strict=True)
                    )

            def execute(label, user, ids, *, prefill, stream):
                outputs, call_counts, metrics = [], [], []
                before = _graph_snapshot(bank)
                queries = [CHUNK] * (HISTORY // CHUNK) if prefill else [len(ids)]
                for index, model in enumerate(models):
                    if prefill:
                        capacity = model.retained_session_capacity(HISTORY + CANDIDATE, HISTORY)
                        session = model.create_session(capacity)
                        sessions.append((model, session))
                        users[index].append(session)
                    else:
                        session = users[index][user]
                        for runner in session.runners:
                            runner.cache.reset_stats()
                    assert not session.released
                    for runner in session.runners:
                        assert runner._consume.__func__ is EchoAttentionRunner._consume
                    start = official_calls.begin(session, not prefill)
                    try:
                        with torch.cuda.stream(stream):
                            hidden = (
                                model.prefill(session, ids)
                                if prefill
                                else model.extend_candidate(session, ids)
                            )
                    finally:
                        official_calls.active = None
                    default.wait_stream(stream)
                    call_counts.append(
                        official_calls.finish(start, offload=scheme == "echo", queries=queries)
                    )
                    values = hidden, model.last_logits
                    expected = tuple(_bytes(tensor) for tensor in values)
                    retained.append((values, expected))
                    outputs.append(tuple(tensor.cpu().clone() for tensor in values))
                    if prefill:
                        saved_history[index, user] = _history_snapshot(session)
                    else:
                        observed = model.session_metrics(session)
                        if observed["candidate_persistence"] == "gpu_transient":
                            assert session.length == HISTORY
                            assert observed["candidate_device_to_host_bytes"] == 0
                        else:
                            assert scheme == "hbm" and session.length == HISTORY + len(ids)
                            assert observed["candidate_device_to_host_bytes"] is None
                            assert observed["device_to_host_bytes"] == 0
                        model.truncate(session, HISTORY)
                        _assert_history(session, saved_history[index, user])
                        if scheme == "echo" and len(ids) == FALLBACK:
                            assert observed["host_to_device_bytes"] > 0, "no interleaved recall"
                        metrics.append(observed["host_to_device_bytes"])
                    assert CacheFootprint.from_mapping(model.shared_bytes()).fits(
                        plans[index].shared
                    )
                for left, right in zip(users[0][user].runners, users[1][user].runners, strict=True):
                    assert left.cache.records.data_ptr() != right.cache.records.data_ptr()
                    assert left.index_keys.data_ptr() != right.index_keys.data_ptr()
                numerical = {
                    name: numerical_comparison(actual, reference, name)
                    for name, actual, reference in zip(
                        ("hidden", "logits"), outputs[1], outputs[0], strict=True
                    )
                }
                assert all(value["numerical_pass"] for value in numerical.values()), numerical
                after = _graph_snapshot(bank)
                assert after["identities"] == original_bank["identities"]
                assert after["bytes"] == original_bank["bytes"]
                for key in bank.pairs:
                    delta = queries.count(key[1])
                    assert after["replays"][key] == tuple(x + delta for x in before["replays"][key])
                fallback = LAYERS if len(ids) == FALLBACK else 0
                assert after["fallbacks"] - before["fallbacks"] == fallback
                assert bank._last_stream == stream and bank._last_event is not None
                assert not bank._active and bank._stream is None
                check_retained()
                phases.append(
                    {
                        "scheme": scheme,
                        "phase": label,
                        "tokens": len(ids),
                        "nondefault_stream": stream != default,
                        "official_calls": call_counts,
                        "candidate_host_to_device_bytes": metrics,
                        "numerical": numerical,
                    }
                )

            execute("alice_history", 0, prefixes[0], prefill=True, stream=default)
            execute("alice_a128", 0, candidates[0], prefill=False, stream=default)
            execute("bob_history", 1, prefixes[1], prefill=True, stream=alternate)
            execute("alice_a121_revisit", 0, candidates[1], prefill=False, stream=default)
            execute("alice_a128_changed", 0, candidates[2], prefill=False, stream=alternate)
            for index, model_users in enumerate(users):
                for user, session in enumerate(model_users):
                    _assert_history(session, saved_history[index, user])
                    model = models[index]
                    model.release_session(session)
                    sessions.remove((model, session))
            check_retained()
            print(
                json.dumps(
                    {
                        "official_callback_scheme_passed": scheme,
                        "H": HISTORY,
                        "P": HISTORY,
                        "NH": 2 * HISTORY,
                        "chunk": CHUNK,
                        "candidate_lengths": [CANDIDATE, FALLBACK, CANDIDATE],
                        "ordinary_consume_retained": True,
                        "graph_state": bank.describe(),
                    }
                ),
                flush=True,
            )

        for scheme in ("hbm", "echo"):
            check_scheme(scheme)
        source_after = _source_identity()
        source_after[str(Path(__file__).resolve())] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        assert source_after == source_before
        assert collect_backend_provenance() == library_before
        assert precision_settings(torch) == precision
        assert models[0].pipeline.provenance()["source_files"] == native_before
        print(
            json.dumps(
                {
                    "official_callback_gate": "passed",
                    "boundary": "H2304 callback/dispatch/stream/lifetime check; no H64K or serving performance claim",
                    "numerical_policy": NUMERICAL_POLICY,
                    "numerical_reference": "independent eager same-scheme empty histories; inherited thresholds, no recalibration",
                    "phases": phases,
                    "source_manifest_sha256": hashlib.sha256(
                        json.dumps(source_before, sort_keys=True).encode()
                    ).hexdigest(),
                    "backend_provenance_sha256": hashlib.sha256(
                        json.dumps(library_before, sort_keys=True).encode()
                    ).hexdigest(),
                    "flashinfer_runtime_artifacts": collect_flashinfer_runtime_artifacts(),
                }
            ),
            flush=True,
        )
    finally:
        for model, session in reversed(sessions):
            if not session.released:
                model.release_session(session)
        for model in reversed(models):
            model.close()
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
