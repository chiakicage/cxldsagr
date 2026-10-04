"""Unconfirmed execution completion must retain shared ownership and storage."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from models.nosa.tests.test_serving_resources import (
    BUDGET,
    SCHEMES,
    limits,
    make_backend,
    storages,
    tokens,
)


@pytest.mark.parametrize("staging_fails,device_fails", [(True, False), (False, True), (True, True)])
def test_failed_drain_attempts_both_paths_and_preserves_poisoned_owner(
    monkeypatch, staging_fails, device_fails
):
    backend = make_backend("dense_prefetch")
    backend.allocate_shared(backend.plan_resources(BUDGET, limits()))
    session = backend.create_session(160)
    resources = backend.resources
    original_device = resources.device
    original_attention = backend.model.main_attention
    shared = storages(resources, stop=(backend, backend.model, session))
    owned = storages(session, stop=(backend, backend.model, resources))
    calls = []
    waited = []
    staging_lease = None
    try:
        # All allocated tensors remain CPU tensors. Only the completion branch
        # is simulated; no CUDA context, kernel, or actual stream is required.
        with monkeypatch.context() as patch:
            resources.device = torch.device("cuda:0")
            patch.setattr(resources, "validate_native", lambda: None)
            patch.setattr(resources, "_validate_allocator", lambda: None)
            patch.setattr(torch.cuda, "device", lambda *args: nullcontext())
            patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
            allocation_event, session_event = object(), object()
            patch.setattr(resources, "_allocation_ready", allocation_event)
            patch.setattr(session, "_execution_ready", session_event)
            patch.setattr(
                torch.cuda,
                "current_stream",
                lambda *args: SimpleNamespace(wait_event=waited.append),
            )

            def device_drain(device):
                calls.append("device")
                assert device == resources.device
                if device_fails:
                    raise RuntimeError("injected device drain failure")

            patch.setattr(torch.cuda, "synchronize", device_drain)
            with pytest.raises(RuntimeError, match="drain NOSA"), backend.execution_lease(session):
                staging_lease = resources.staging_lease
                original_close = staging_lease.close

                def staging_drain():
                    calls.append("staging")
                    if staging_fails:
                        raise RuntimeError("injected staging drain failure")
                    original_close()

                patch.setattr(staging_lease, "close", staging_drain)

            assert calls == ["staging", "device"]
            assert waited == [allocation_event, session_event]
            assert resources.poisoned
            assert resources._active_session is session
            assert resources.staging_lease is staging_lease
            assert backend.model.main_attention is original_attention
            assert storages(resources, stop=(backend, backend.model, session)) == shared
            assert storages(session, stop=(backend, backend.model, resources)) == owned
            for mutation in (session.release, session.reset, lambda: session.truncate(0)):
                with pytest.raises(RuntimeError, match="poisoned"):
                    mutation()
            assert not session.released
            with pytest.raises(RuntimeError, match="poisoned"), backend.execution_lease(session):
                pytest.fail("unconfirmed storage must not be lent again")
            with pytest.raises(RuntimeError):
                backend.close()
    finally:
        # This test injected errors without enqueuing work. Remove only those
        # simulated failures to dispose the actual CPU allocations safely.
        resources.device = original_device
        if staging_lease is not None:
            staging_lease.close()
        resources.staging_lease = None
        resources._active_session = None
        resources.poisoned = False
        backend.release_session(session)
        backend.close()


def test_successful_drain_after_body_error_returns_lease_and_restores_attention(monkeypatch):
    backend = make_backend("dense_prefetch")
    backend.allocate_shared(backend.plan_resources(BUDGET, limits()))
    session = backend.create_session(160)
    original = backend.model.main_attention
    try:
        with pytest.raises(ValueError, match="body failed"), backend.execution_lease(session):
            raise ValueError("body failed")
        assert not backend.resources.poisoned
        assert backend.resources._active_session is backend.resources.staging_lease is None
        assert backend.model.main_attention is original
        with backend.execution_lease(session):
            assert backend.resources._active_session is session
    finally:
        backend.release_session(session)
        backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("failure_phase", ["event", "stream", "record"])
@pytest.mark.parametrize("drain_fails", [False, True])
def test_failed_session_ready_event_cleans_up_or_retains_poisoned_ownership(
    monkeypatch, scheme, failure_phase, drain_fails
):
    backend = make_backend(scheme)
    plan = backend.plan_resources(BUDGET, limits())
    backend.allocate_shared(plan)
    existing = backend.create_session(160)
    backend.prefill(existing, tokens(17))
    resources = backend.resources
    original_device = resources.device
    original_attach = resources.attach
    shared = storages(resources)
    existing_storage = storages(existing, stop=(resources,))
    created, before, drained = [], [], []

    def attach(session):
        created.append(session)
        before.append(storages(session, stop=(resources,)))
        # The constructor and tensors remain CPU-only. Simulate only CUDA
        # readiness setup so every allocation failure is deterministic.
        resources.device = torch.device("cuda:0")
        try:
            return original_attach(session)
        finally:
            resources.device = original_device

    def fail_at(phase):
        if phase == failure_phase:
            raise RuntimeError("injected session readiness failure")

    def event():
        fail_at("event")
        return SimpleNamespace(record=lambda stream: fail_at("record"))

    def current_stream(device):
        fail_at("stream")
        return object()

    def synchronize(device):
        drained.append(device)
        if drain_fails:
            raise RuntimeError("injected attachment drain failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(resources, "attach", attach)
            patch.setattr(torch.cuda, "Event", event)
            patch.setattr(torch.cuda, "current_stream", current_stream)
            patch.setattr(torch.cuda, "synchronize", synchronize)
            message = "drain failed NOSA" if drain_fails else "session readiness failure"
            with pytest.raises(RuntimeError, match=message):
                backend.create_session(128)
        failed = created[0]
        assert drained == [torch.device("cuda:0")]
        assert resources.plan is plan
        assert storages(resources) == shared
        assert storages(existing, stop=(resources,)) == existing_storage
        assert resources.poisoned is drain_fails
        if drain_fails:
            assert failed in resources._sessions
            assert failed._execution_resources is resources
            assert not failed.released
            assert storages(failed, stop=(resources,)) == before[0]
            resources.check_session(failed)
            for session in (existing, failed):
                with pytest.raises(RuntimeError, match="poisoned"):
                    backend.release_session(session)
                with (
                    pytest.raises(RuntimeError, match="poisoned"),
                    backend.execution_lease(session),
                ):
                    pytest.fail("unconfirmed initialization must not be reused")
            with pytest.raises(RuntimeError):
                backend.close()
        else:
            assert resources._sessions == {existing}
            assert failed.released
            assert failed._execution_resources is None
            assert storages(failed, stop=(resources,)) == {}
            assert backend.session_bytes(failed) == {"hbm": 0, "dram": 0}
            backend.allocate_shared(plan)
            backend.extend(existing, tokens(1, 2))
            replacement = backend.create_session(128)
            backend.prefill(replacement, tokens(17, 3))
            backend.release_session(replacement)
    finally:
        # Faults above never submitted real CUDA work. Dispose CPU allocations
        # only after removing the simulated poison, as in the lease-drain test.
        resources.device = original_device
        resources.poisoned = False
        for session in tuple(resources._sessions):
            backend.release_session(session)
        backend.close()
