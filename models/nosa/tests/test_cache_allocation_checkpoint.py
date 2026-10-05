"""Intrusive NOSA cache-allocation acceptance, never a latency experiment.

Opt in with NOSA_CACHE_AUDIT_CHECKPOINT=/path/to/NOSA-8B. The defaults exercise
all 32 layers, four shared schemes, two independently built 64K prefixes and
interleaved 1K candidates/truncation. Set NOSA_CACHE_AUDIT_SUFFIX_TOKENS=128 for
the second required geometry. Traces and reports use pytest's temporary directory.
Compilation is warmed with a separate session; audited sessions start empty.

This observes PyTorch allocation blocks and CUDA free_completed lifetimes.
Direct library allocations, CPU malloc overhead and CUDA fragmentation are not
proved. Cache tensor reservations, rounded allocator bounds, ordinary activation
peaks, model weights and process allocated/reserved peaks are reported separately.
The direct backend fixture does not claim multi-user LRU/admission acceptance.
"""

import json
import os
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from cache.prefix_pool import CacheFootprint
from evaluation.cache_memory_audit import (
    CACHE_CATEGORIES,
    ORDINARY_PREFIX,
    SCOPE_PREFIX,
    SOURCE_PREFIX,
    CacheMemoryAudit,
    audit_trace,
    storage_inventory,
)
from models.nosa.attention import NosaDensePrefetchAttention, NosaSparseAttention
from models.nosa.execution.adapter import NosaServingBackend
from models.nosa.indexer import NosaIndexer
from models.nosa.tests.cache_allocation_join import CudaGenerationJoin


def _nonempty(tensors):
    return [tensor for tensor in tensors if tensor is not None and tensor.numel()]


def _shared_tensors(resources):
    """Enumerate owned fields independently of production byte/tensor helpers."""
    tensors = []
    fetch = resources.fetch_workspace
    if fetch is not None:
        tensors.extend(
            getattr(fetch, name)
            for name in (
                "keys",
                "values",
                "_first_use",
                "_tile_bytes",
                "last_transfer_bytes",
                "_empty_mask",
                "_ready_blocks",
                "_fetch_queue",
                "_trace_storage",
            )
        )
        tensors.extend(fetch._scratch.values())
        if fetch._attention_scratch is not None:
            tensors.extend(fetch._attention_scratch._storage.values())
            tensors.append(fetch._attention_scratch.empty_mask)
    attention = resources.attention_workspace
    if attention is not None:
        tensors.extend(attention._storage.values())
        tensors.append(attention.empty_mask)
    if resources.staging is not None:
        tensors.extend(resources.staging._buffers.values())
    return _nonempty(tensors)


def _session_tensors(session):
    tensors = list(session.buffers.values())
    tensors.append(getattr(session, "cis_scores", None))
    tensors.append(getattr(session, "_native_indexer_host_flag", None))
    indexer = session.indexer_cache
    tensors.extend(tensor for layer in indexer._buffers.values() for tensor in layer.values())
    tensors.append(indexer._workspace)
    for suffix in getattr(session, "_suffixes", {}).values():
        tensors.extend(suffix.values())
    if session._transfer_metrics is not None:
        tensors.append(session._transfer_metrics._sparse_fetch_bytes)
    # A borrowed session must never fall back to another owned workspace/stage.
    for name in ("_attention_workspace", "_stage_keys", "_stage_values"):
        assert getattr(session, name, None) is None, name
    return _nonempty(tensors)


def _storage_keys(inventory):
    return {(row["device"], row["address"]) for row in inventory["storages"]}


def _logical_bytes(inventory, device):
    totals = inventory["devices"]
    if device.type == "cpu":
        # CPU fixtures check total storage only; they never claim physical HBM.
        return sum(row["storage_bytes"] for row in totals.values())
    return {
        "hbm": totals.get(str(device), {}).get("storage_bytes", 0),
        "dram": totals.get("cpu", {}).get("storage_bytes", 0),
    }


def _inventory(backend, sessions):
    shared = storage_inventory(_shared_tensors(backend.resources))
    expected = backend.shared_bytes()
    assert _logical_bytes(shared, backend.device) == (
        sum(expected.values()) if backend.device.type == "cpu" else expected
    )
    occupied = _storage_keys(shared)
    tensors = _shared_tensors(backend.resources)
    for session in sessions:
        owned = _session_tensors(session)
        current = storage_inventory(owned)
        assert not occupied.intersection(_storage_keys(current))
        occupied.update(_storage_keys(current))
        expected = backend.session_storage_bytes(session)
        assert _logical_bytes(current, backend.device) == (
            sum(expected.values()) if backend.device.type == "cpu" else expected
        )
        if backend.device.type == "cuda":
            assert backend.session_bytes(session)["dram"] == current["devices"].get("cpu", {}).get(
                "allocator_bytes", 0
            )
        tensors.extend(owned)
    return storage_inventory(tensors), shared


class _AllocationScopes:
    """Temporary wrappers preserve every original computation and return object."""

    def __init__(self, backend, sessions, audit):
        self.backend, self.sessions, self.audit = backend, sessions, audit
        self.calls = Counter()
        self.schedule = defaultdict(list)

    def _wrap(self, original, category, name, *, layer=None, position=None, ordinary_result=False):
        def call(*args, **kwargs):
            index = layer(*args, **kwargs) if layer is not None else None
            self.calls[name, index] += 1
            if position is not None:
                self.schedule[name].append(position(*args, **kwargs))
            with self.audit.scopes(category):
                result = original(*args, **kwargs)
                if ordinary_result:
                    # Native attention's output is an activation. All other
                    # allocations within the adapter remain cache/helper work.
                    self.audit.mark_ordinary_result(result)
                return result

        return call

    @contextmanager
    def installed(self):
        from models.nosa import scoring

        with ExitStack() as stack:
            # execution_lease replaces the adapter instance. Class wrappers
            # survive that replacement and Python's special-method lookup.
            for cls, category, name, output in (
                (NosaIndexer, "indexer", "indexer", False),
                (NosaSparseAttention, "offload_prepare", "attention", True),
                (NosaDensePrefetchAttention, "offload_prepare", "attention", True),
            ):
                stack.enter_context(
                    patch.object(
                        cls,
                        "__call__",
                        self._wrap(
                            cls.__call__,
                            category,
                            name,
                            layer=lambda self, q, *rest: rest[-1].layer_idx,
                            position=lambda self, q, *rest: (
                                rest[-1].query_start,
                                rest[-1].query_length,
                                rest[-1].layer_idx,
                            ),
                            ordinary_result=output,
                        ),
                    )
                )
            stack.enter_context(
                patch.object(
                    scoring,
                    "cis_scores",
                    self._wrap(scoring.cis_scores, "indexer", "cis"),
                )
            )
            stack.enter_context(
                patch.object(
                    self.backend.model,
                    "forward",
                    self._wrap(self.backend.model.forward, "ordinary_model", "forward"),
                )
            )
            for session in self.sessions:
                for name in (
                    "begin_step",
                    "write_layer",
                    "prepare_indexer_inputs",
                    "commit_step",
                    "abort_step",
                    "truncate",
                    "reset",
                    "release",
                ):
                    if not hasattr(session, name):
                        continue
                    category = (
                        "cache_write"
                        if name == "write_layer"
                        else "indexer"
                        if name == "prepare_indexer_inputs"
                        else "cache_maintenance"
                    )
                    layer = (
                        (lambda layer_idx, *args, **kwargs: layer_idx)
                        if name in ("write_layer", "prepare_indexer_inputs")
                        else None
                    )
                    stack.enter_context(
                        patch.object(
                            session,
                            name,
                            self._wrap(
                                getattr(session, name),
                                category,
                                name,
                                layer=layer,
                                position=(
                                    lambda layer_idx, *args, session=session, **kwargs: (
                                        session.length,
                                        session._pending_end - session.length,
                                        layer_idx,
                                    )
                                )
                                if name == "write_layer"
                                else None,
                            ),
                        )
                    )
            yield self

    def check_coverage(self, phase, batches):
        forwards = len(batches)
        assert self.calls["forward", None] == forwards
        expected = [
            (start, length, layer)
            for start, length in batches
            for layer in range(self.backend.config.num_hidden_layers)
        ]
        for name in ("write_layer", "indexer", "attention"):
            assert self.schedule[name] == expected, (name, self.schedule[name], expected)
        for name in ("begin_step", "commit_step", "cis"):
            assert self.calls[name, None] == (
                forwards * self.backend.config.num_hidden_layers if name == "cis" else forwards
            ), (name, self.calls)
        for layer in range(self.backend.config.num_hidden_layers):
            for name in ("write_layer", "indexer", "attention"):
                assert self.calls[name, layer] == forwards, (name, layer, self.calls)
            if self.backend.scheme != "hbm" and forwards:
                # Offload commit revisits preparation after attention as well.
                assert self.calls["prepare_indexer_inputs", layer] >= forwards
        assert self.calls["abort_step", None] == 0
        if phase == "truncate":
            assert self.calls["truncate", None] == 1


def _ordinary_allocations(trace, history, devices, *, join=None):
    """Apply the same lifetime parser to the complementary activation scopes.

    The parser's generic diagnostics category is used only as a tracking bit in
    this second pass. No allocation or copy is made on the observed device.
    NOSA supplies no projected-source marker: the original combined QKV remains
    ordinary while write_layer's independent K/V clones are charged to cache.
    """
    ordinary = {**trace, "traceEvents": []}
    for event in trace["traceEvents"]:
        name = event.get("name", "")
        assert not name.startswith(SOURCE_PREFIX), "NOSA must not promote combined QKV storage"
        if name.startswith(SCOPE_PREFIX):
            category, label = name.removeprefix(SCOPE_PREFIX).split("|", 1)
            category = "ordinary" if category in CACHE_CATEGORIES else "diagnostics"
            event = {**event, "name": f"{SCOPE_PREFIX}{category}|{label}"}
        elif name.startswith(ORDINARY_PREFIX):
            event = {**event, "name": SOURCE_PREFIX + name.removeprefix(ORDINARY_PREFIX)}
        ordinary["traceEvents"].append(event)
    limits = dict.fromkeys(devices, 1 << 62)
    if join is not None:
        return join.audit(ordinary, limits=limits)
    return audit_trace(ordinary, limits=limits, cuda_history=history)


def _capture(
    backend,
    sessions,
    prefix,
    phase,
    operation,
    directory,
    *,
    active_session,
    batches,
    reports,
    expected=None,
):
    forwards = len(batches)
    before, shared = _inventory(backend, sessions)
    active_before = storage_inventory(
        _shared_tensors(backend.resources) + _session_tensors(active_session)
    )
    session_before = storage_inventory(_session_tensors(active_session))
    active_reservation = backend.estimate_session_bytes(active_session.max_seq_len, prefix)
    shared_reservation = asdict(backend.resources.plan.shared)
    assert backend.shared_bytes() == {
        "hbm": backend.resources.plan.metadata["logical_shared_hbm_bytes"],
        "dram": 0,
    }
    assert CacheFootprint.from_mapping(backend.shared_bytes()).fits(backend.resources.plan.shared)
    for location, tier in ((str(backend.device), "hbm"), ("cpu", "dram")):
        assert (
            shared["devices"].get(location, {}).get("allocator_bytes", 0)
            <= shared_reservation[tier]
        ), (shared, shared_reservation)
    reserved = dict(shared_reservation)
    for session in sessions:
        cost = backend.estimate_session_bytes(session.max_seq_len, prefix)
        for tier in reserved:
            reserved[tier] += cost[tier]
    device = str(backend.device)
    limits = {
        location: reserved[tier] - before["devices"].get(location, {}).get("allocator_bytes", 0)
        for location, tier in ((device, "hbm"), ("cpu", "dram"))
    }
    assert all(value >= 0 for value in limits.values()), (reserved, before)
    active_limits = {
        location: active_reservation[tier]
        - session_before["devices"].get(location, {}).get("allocator_bytes", 0)
        for location, tier in ((device, "hbm"), ("cpu", "dram"))
    }
    audit = CacheMemoryAudit(
        directory / f"{phase}.json",
        devices=[backend.device],
        history_max_entries=4_000_000,
        record_shapes=False,
        with_stack=False,
    )
    scopes = _AllocationScopes(backend, sessions, audit)
    torch.cuda.synchronize(backend.device)
    baseline_allocated = torch.cuda.memory_allocated(backend.device)
    baseline_reserved = torch.cuda.memory_reserved(backend.device)
    torch.cuda.reset_peak_memory_stats(backend.device)
    with audit, scopes.installed(), audit.scopes("ordinary_model"):
        result = operation()
        # Includes commit-time derivation and queued D2H/H2D completion.
        backend.synchronize()
        # Public request accounting reads sparse counters via Tensor.tolist(),
        # which owns a temporary CPU copy. Include that normal boundary too.
        with audit.scopes("cache_maintenance"):
            backend.session_metrics(active_session)
    allocated_peak = torch.cuda.max_memory_allocated(backend.device)
    reserved_peak = torch.cuda.max_memory_reserved(backend.device)
    numerical_ok = None if expected is None else torch.equal(result.cpu(), expected)
    scopes.check_coverage(phase.split("_")[-1], batches)
    assert active_session.length == active_session.indexer_cache.length == prefix
    after, shared_after = _inventory(backend, sessions)
    assert shared_after == shared, "shared storage changed during execution"
    trace = json.loads(audit.trace_path.read_text())
    history = json.loads(audit.history_path.read_text())
    join = CudaGenerationJoin(trace, history, baseline=before)
    cache = join.audit(trace, limits=limits)
    ordinary = _ordinary_allocations(trace, history, limits, join=join)
    assert cache["ordinary_result_markers"] == forwards * backend.config.num_hidden_layers
    assert cache["projected_source_markers"] == 0
    peak_bound = {
        location: before["devices"].get(location, {}).get("allocator_bytes", 0)
        + (values["observed_allocator_active_peak_bytes"] or 0)
        for location, values in cache["devices"].items()
    }
    active_bounds = {
        location: {
            "temporary_limit_bytes": active_limits[location],
            "observed_allocator_active_peak_bytes": values["observed_allocator_active_peak_bytes"],
            "fits": values["observed_allocator_active_peak_bytes"] is not None
            and values["observed_allocator_active_peak_bytes"] <= active_limits[location],
        }
        for location, values in cache["devices"].items()
    }
    report = {
        "schema": "nosa-checkpoint-cache-allocation-validation-v2",
        "scheme": backend.scheme,
        "phase": phase,
        "layers": backend.config.num_hidden_layers,
        "prefix_tokens": prefix,
        "session_capacities": [session.max_seq_len for session in sessions],
        "resource_plan": asdict(backend.resources.plan.shared),
        "plan_limits": dict(backend.resources.plan.metadata),
        "cache_reserved_bytes": reserved,
        "active_session_reserved_bytes": active_reservation,
        "active_session_and_shared_fixed_before": active_before,
        "active_session_temporary_bounds": active_bounds,
        "active_session_bound_semantics": (
            "active-session reservation minus its rounded fixed baseline; "
            "no unused reservation from shared storage or inactive sessions is lent"
        ),
        "shared_allocator_padding_bytes": {
            location: values["padding_bytes"] for location, values in shared["devices"].items()
        },
        "fixed_cache_before": before,
        "fixed_cache_after": after,
        "cache_allocator_peak_upper_bound_bytes": peak_bound,
        "cache_allocator_peak_bound_semantics": (
            "rounded fixed baseline plus observed allocations through free_completed; "
            "conservatively retains baseline blocks even if freed during capture"
        ),
        "cache": cache,
        "ordinary_activation_allocator_peak_bytes": {
            location: values["observed_allocator_active_peak_bytes"]
            for location, values in ordinary["devices"].items()
        },
        "ordinary_evidence_errors": ordinary["evidence_errors"],
        "ordinary_scope": "model execution and attention outputs; excludes cache/CIS/indexer work",
        "model_weight_storage_bytes": sum(
            row["storage_bytes"]
            for row in storage_inventory(backend.model.parameters())["storages"]
        ),
        "process_cuda_baseline_allocated_bytes": baseline_allocated,
        "process_cuda_baseline_reserved_bytes": baseline_reserved,
        "process_cuda_peak_allocated_bytes": allocated_peak,
        "process_cuda_peak_reserved_bytes": reserved_peak,
        "wrapper_calls": {
            f"{name}:{layer}": value for (name, layer), value in scopes.calls.items()
        },
        "observed_layer_schedule": dict(scopes.schedule),
        "latency_measurement": False,
        "candidate_hidden_bitwise_equal": numerical_ok,
        "admission_scope": "direct backend; no PrefixPool/LRU capacity claim",
    }
    report["passed"] = (
        cache["passed_observed_temporary_bound"]
        and all(values["fits"] for values in active_bounds.values())
        and not ordinary["evidence_errors"]
        and numerical_ok is not False
    )
    reports.append(report)
    audit.trace_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "nosa_memory_audit": phase,
                "scheme": backend.scheme,
                "cache_allocator_peak_upper_bound_bytes": peak_bound,
                "cache_reserved_bytes": reserved,
                "active_session_temporary_bounds": active_bounds,
                "ordinary_activation_allocator_peak_bytes": report[
                    "ordinary_activation_allocator_peak_bytes"
                ],
                "process_cuda_peak_allocated_bytes": allocated_peak,
                "process_cuda_peak_reserved_bytes": reserved_peak,
                "passed": report["passed"],
                "candidate_hidden_bitwise_equal": numerical_ok,
                "temporary_report": str(audit.trace_path.with_suffix(".report.json")),
            }
        ),
        flush=True,
    )
    # Preserve a failed result and continue the other phases. The final test
    # assertion reports every observed bound/evidence failure without adding an
    # allowance for profiler-observed CPU scalars or CUDA allocator padding.
    return result


@pytest.fixture(scope="module")
def checkpoint_backend():
    checkpoint = os.environ.get("NOSA_CACHE_AUDIT_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set NOSA_CACHE_AUDIT_CHECKPOINT for full checkpoint allocation acceptance")
    assert Path(checkpoint).is_dir(), checkpoint
    assert torch.cuda.is_available(), "Explicit memory audit requires CUDA"
    assert torch.cuda.get_device_capability() == (9, 0), "Memory audit requires SM90/Hopper"
    from operators.nosa._native import native_enabled

    assert native_enabled(), "Memory audit requires the native NOSA backend"
    suffix = int(os.environ.get("NOSA_CACHE_AUDIT_SUFFIX_TOKENS", "1024"))
    assert suffix in (128, 1024), "Use either required 64K checkpoint geometry"
    backend = NosaServingBackend.from_pretrained(
        checkpoint, scheme="hbm", device="cuda:0", max_seq_len=65536 + suffix
    )
    assert backend.config.num_hidden_layers == 32
    prefix, capacity = 65536, 65536 + suffix
    plan = backend.plan_resources(
        CacheFootprint(1 << 40, 1 << 40),
        {"max_session_capacity": capacity, "max_candidate_tokens": suffix},
    )
    backend.allocate_shared(plan)
    expected = {}
    try:
        with torch.inference_mode():
            ids = (
                torch.arange(capacity, device=backend.device) * 19 + 137
            ) % backend.config.vocab_size
            references = [backend.create_session(capacity) for _ in range(2)]
            for user, session in enumerate(references):
                history = (ids[:prefix] + user * 37) % backend.config.vocab_size
                backend.prefill(session, history)
            for visit in range(2):
                for user, session in enumerate(references):
                    candidate = (ids[prefix:] + user * 29 + visit * 11) % backend.config.vocab_size
                    expected[user, visit] = backend.extend(session, candidate).cpu()
                    backend.truncate(session, prefix)
    finally:
        for session in tuple(backend.resources._sessions):
            backend.release_session(session)
        backend.close()
    # All reference histories/resources are gone before audited sessions exist.
    yield backend, suffix, expected
    backend.close()


@pytest.mark.parametrize("scheme", NosaServingBackend.schemes)
@torch.inference_mode()
def test_checkpoint_cache_allocation_lifetimes(checkpoint_backend, scheme, tmp_path):
    control, candidate, expected = checkpoint_backend
    prefix, capacity = 65536, 65536 + candidate
    backend = NosaServingBackend(control.model, scheme, chunk_size=1024)
    plan = backend.plan_resources(
        CacheFootprint(1 << 40, 1 << 40),
        {"max_session_capacity": capacity, "max_candidate_tokens": candidate},
    )
    backend.allocate_shared(plan)
    shared_before_warmup = storage_inventory(_shared_tensors(backend.resources))
    ids = (torch.arange(capacity, device=backend.device) * 19 + 137) % backend.config.vocab_size
    reports = []
    try:
        # Cover the real long-context/q branches before profiling, then discard
        # all warm session-derived records and scratch. Shared storage is fixed.
        warm = backend.create_session(capacity)
        backend.prefill(warm, ids[:prefix])
        backend.extend(warm, ids[prefix:])
        backend.release_session(warm)
        assert storage_inventory(_shared_tensors(backend.resources)) == shared_before_warmup
        sessions, host_allocations = [], []
        for _ in range(2):
            before_host = torch.cuda.memory.host_memory_stats()
            session = backend.create_session(capacity)
            after_host = torch.cuda.memory.host_memory_stats()
            sessions.append(session)
            inventory = storage_inventory(_session_tensors(session))
            host_bytes = inventory["devices"].get("cpu", {}).get("allocator_bytes", 0)
            measured = {
                name: after_host[name] - before_host[name]
                for name in ("active_bytes.allocated", "active_requests.allocated")
            }
            # Cumulative allocated counters count the actual HostBlock.size
            # both for new bins and reuse. No other pinned allocation or stats
            # reset occurs in this isolated constructor window; current/peak
            # counters have separate asynchronous/cached-block semantics.
            assert measured["active_bytes.allocated"] == host_bytes
            assert measured["active_requests.allocated"] == (0 if scheme == "hbm" else 2)
            host_allocations.append({"inventory_bytes": host_bytes, **measured})
        (tmp_path / "host-creation.json").write_text(json.dumps(host_allocations, indent=2))
        assert all(not session.indexer_cache._buffers for session in sessions)
        assert all(session.indexer_cache._workspace is None for session in sessions)
        for user, session in enumerate(sessions):
            history = (ids[:prefix] + user * 37) % backend.config.vocab_size
            output = _capture(
                backend,
                sessions,
                prefix,
                f"user{user}_prefill",
                lambda session=session, history=history: backend.prefill(session, history),
                tmp_path,
                active_session=session,
                batches=[(start, 1024) for start in range(0, prefix, 1024)],
                reports=reports,
            )
            del output
        for visit in range(2):
            for user, session in enumerate(sessions):
                suffix = (ids[prefix:] + user * 29 + visit * 11) % backend.config.vocab_size

                def extend_and_truncate(session=session, suffix=suffix):
                    result = backend.extend(session, suffix)
                    backend.truncate(session, prefix)
                    return result

                output = _capture(
                    backend,
                    sessions,
                    prefix,
                    f"visit{visit}_user{user}_extend_truncate",
                    extend_and_truncate,
                    tmp_path,
                    active_session=session,
                    batches=[(prefix, candidate)],
                    reports=reports,
                    expected=expected[user, visit],
                )
                assert output.shape == (candidate, backend.config.hidden_size)
                del output
    finally:
        for session in tuple(backend.resources._sessions):
            backend.release_session(session)
        backend.close()
    failures = [
        {
            "phase": report["phase"],
            "cache_devices": report["cache"]["devices"],
            "active_session_temporary_bounds": report["active_session_temporary_bounds"],
            "cache_evidence_error_count": len(report["cache"]["evidence_errors"]),
            "cache_evidence_examples": report["cache"]["evidence_errors"][:3],
            "ordinary_evidence_error_count": len(report["ordinary_evidence_errors"]),
            "ordinary_evidence_examples": report["ordinary_evidence_errors"][:3],
            "candidate_hidden_bitwise_equal": report["candidate_hidden_bitwise_equal"],
        }
        for report in reports
        if not report["passed"]
    ]
    assert len(reports) == 6 and not failures, failures


class _RecordingScopes:
    def __init__(self):
        self.outputs = 0

    @contextmanager
    def scopes(self, name):
        yield

    def mark_ordinary_result(self, result):
        assert isinstance(result, torch.Tensor)
        self.outputs += 1


@pytest.mark.parametrize("scheme", NosaServingBackend.schemes)
@torch.inference_mode()
def test_allocation_wrappers_cover_layers_phases_and_restore(scheme):
    from models.nosa.tests.test_serving_resources import allocated_backend, tokens

    originals = (
        NosaIndexer.__call__,
        NosaSparseAttention.__call__,
        NosaDensePrefetchAttention.__call__,
    )
    with allocated_backend(scheme) as (backend, _):
        expected = {}
        for user in range(2):
            reference = backend.create_session(160)
            backend.prefill(reference, tokens(79, user))
            expected[user] = backend.extend(reference, tokens(17)).clone()
            backend.release_session(reference)
        sessions = [backend.create_session(160) for _ in range(2)]
        _inventory(backend, sessions)
        for user, session in enumerate(sessions):
            recorder = _RecordingScopes()
            wrappers = _AllocationScopes(backend, sessions, recorder)
            with wrappers.installed():
                backend.prefill(session, tokens(79, user))
            wrappers.check_coverage("prefill", [(0, 32), (32, 32), (64, 15)])
            assert recorder.outputs == 3 * backend.config.num_hidden_layers
        for user in (0, 1, 0):
            session = sessions[user]
            wrappers = _AllocationScopes(backend, sessions, _RecordingScopes())
            with wrappers.installed():
                actual = backend.extend(session, tokens(17))
            torch.testing.assert_close(actual, expected[user], atol=0, rtol=0)
            wrappers.check_coverage("extend", [(79, 17)])
            wrappers = _AllocationScopes(backend, sessions, _RecordingScopes())
            with wrappers.installed():
                backend.truncate(session, 79)
            wrappers.check_coverage("truncate", [])
        _inventory(backend, sessions)
        with (
            pytest.raises(RuntimeError, match="injected"),
            _AllocationScopes(backend, sessions, _RecordingScopes()).installed(),
        ):
            raise RuntimeError("injected")
    assert originals == (
        NosaIndexer.__call__,
        NosaSparseAttention.__call__,
        NosaDensePrefetchAttention.__call__,
    )


def test_complementary_parser_separates_cache_clones_and_attention_output(tmp_path):
    audit = CacheMemoryAudit(tmp_path / "cpu.json", record_shapes=False, with_stack=False)
    with audit, audit.scopes("ordinary_model"):
        qkv = torch.empty(128, dtype=torch.uint8)
        with audit.scopes("cache_write"):
            suffix = qkv[:64].clone()
        with audit.scopes("offload_prepare"):
            output = torch.empty(96, dtype=torch.uint8)
            audit.mark_ordinary_result(output)
        del output, suffix, qkv
    trace = json.loads(audit.trace_path.read_text())
    cache = audit_trace(trace, limits={"cpu": 64})
    ordinary = _ordinary_allocations(trace, None, ("cpu",))
    assert cache["passed_observed_temporary_bound"]
    assert cache["devices"]["cpu"]["observed_cache_peak_bytes"] == 64
    assert not ordinary["evidence_errors"]
    assert ordinary["devices"]["cpu"]["observed_cache_peak_bytes"] == 224
