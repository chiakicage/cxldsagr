"""Admission ownership and rollback around bounded NOSA shared allocation."""

import gc
import weakref

import pytest
import torch

from models.nosa.tests.test_serving_resources import BUDGET, SCHEMES, limits, make_backend, storages


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("preallocated", (False, True))
@pytest.mark.parametrize("rollback", (False, True))
def test_owner_release_preserves_prior_plan_and_rolls_back_only_new_storage(
    scheme, preallocated, rollback
):
    backend = make_backend(scheme)
    plan = backend.plan_resources(BUDGET, limits())
    owner = object()
    if preallocated:
        backend.allocate_shared(plan)
    backend.bind_owner(owner)
    try:
        backend.allocate_shared(plan, owner=owner)
        before = storages(backend.resources, stop=(backend, backend.model))
        assert before
        backend.unbind_owner(owner, rollback=rollback)
        assert backend.resources._admission_owner is None
        if rollback and not preallocated:
            assert backend.resources.plan is None
            assert backend.shared_bytes() == {"hbm": 0, "dram": 0}
            assert not storages(backend.resources, stop=(backend, backend.model))
        else:
            assert backend.resources.plan is plan
            assert storages(backend.resources, stop=(backend, backend.model)) == before
        # Admission can bind again without any stale rollback identity.
        backend.bind_owner(owner)
        backend.unbind_owner(owner)
    finally:
        if backend.resources._admission_owner is owner:
            backend.unbind_owner(owner)
        backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_owner_requires_identity_and_rejects_poison_before_changing_state(scheme):
    backend = make_backend(scheme)
    try:
        with pytest.raises(ValueError, match="non-None"):
            backend.bind_owner(None)
        with pytest.raises(ValueError, match="Foreign"):
            backend.unbind_owner(None, rollback=True)
        backend.resources.poisoned = True
        with pytest.raises(RuntimeError, match="poisoned"):
            backend.bind_owner(object())
        assert backend.resources._admission_owner is None
        assert backend.resources.plan is None
    finally:
        # The test injects the poison without submitting any asynchronous work.
        backend.resources.poisoned = False
        backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_failed_initial_allocation_drains_before_owner_rollback(monkeypatch, scheme):
    backend = make_backend(scheme)
    plan = backend.plan_resources(BUDGET, limits())
    owner = object()
    backend.bind_owner(owner)
    original_empty = torch.empty
    calls = 0

    def fail_second_allocation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected allocation failure")
        return original_empty(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", fail_second_allocation)
            with pytest.raises(RuntimeError, match="injected allocation failure"):
                backend.allocate_shared(plan, owner=owner)
        assert backend.resources._admission_owner is owner
        assert not backend.resources.poisoned
        assert backend.resources._allocation_failure is None
        assert backend.resources.plan is None
        assert not storages(backend.resources, stop=(backend, backend.model))
        backend.unbind_owner(owner, rollback=True)
        backend.bind_owner(owner)
        backend.allocate_shared(plan, owner=owner)
        backend.unbind_owner(owner)
    finally:
        if backend.resources._admission_owner is owner:
            backend.unbind_owner(owner)
        backend.close()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_rollback_failed_device_drain_retains_owner_plan_and_storage(monkeypatch, scheme):
    backend = make_backend(scheme)
    plan = backend.plan_resources(BUDGET, limits())
    owner = object()
    backend.bind_owner(owner)
    backend.allocate_shared(plan, owner=owner)
    before = storages(backend.resources, stop=(backend, backend.model))
    observed = []

    def fail_drain(device):
        observed.append(device)
        assert backend.resources._admission_owner is owner
        assert storages(backend.resources, stop=(backend, backend.model)) == before
        raise RuntimeError("injected device drain failure")

    try:
        with monkeypatch.context() as patch:
            # CPU tensors model retained ownership; no CUDA work is submitted.
            patch.setattr(backend.resources, "device", torch.device("cuda:0"))
            patch.setattr(torch.cuda, "synchronize", fail_drain)
            with pytest.raises(RuntimeError, match="Unable to drain"):
                backend.unbind_owner(owner, rollback=True)
        assert observed == [torch.device("cuda:0")]
        assert backend.resources.poisoned
        assert backend.resources._admission_owner is owner
        assert backend.resources.plan is plan
        assert storages(backend.resources, stop=(backend, backend.model)) == before
        with pytest.raises(RuntimeError, match="poisoned"):
            backend.unbind_owner(owner)
        with pytest.raises(RuntimeError, match="poisoned"):
            backend.allocate_shared(plan, owner=owner)
    finally:
        backend.resources.poisoned = False
        backend.unbind_owner(owner, rollback=True)
        backend.close()


def test_rollback_failed_staging_release_retains_owner_and_all_storage(monkeypatch):
    backend = make_backend("dense_prefetch")
    plan = backend.plan_resources(BUDGET, limits())
    owner = object()
    backend.bind_owner(owner)
    backend.allocate_shared(plan, owner=owner)
    before = storages(backend.resources, stop=(backend, backend.model))

    def fail_close():
        raise RuntimeError("injected staging release failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(backend.resources.staging, "close", fail_close)
            with pytest.raises(RuntimeError, match="Unable to drain"):
                backend.unbind_owner(owner, rollback=True)
        assert backend.resources.poisoned
        assert backend.resources._admission_owner is owner
        assert backend.resources.plan is plan
        assert storages(backend.resources, stop=(backend, backend.model)) == before
    finally:
        backend.resources.poisoned = False
        backend.unbind_owner(owner, rollback=True)
        backend.close()


def test_failed_constructor_drain_retains_partial_object_until_safe_disposal(monkeypatch):
    from operators.nosa.attention.offload.api import NosaFetchWorkspace

    backend = make_backend("overlap")
    plan = backend.plan_resources(BUDGET, limits())
    owner = object()
    backend.bind_owner(owner)
    partial, drains = [], []

    def fail_constructor(workspace, *args, **kwargs):
        workspace.keys = torch.empty(16)
        partial.append(weakref.ref(workspace))
        raise RuntimeError("injected partial constructor failure")

    def fail_drain(device):
        drains.append(device)
        assert partial[0]() is not None
        raise RuntimeError("injected device drain failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(backend.resources, "device", torch.device("cuda:0"))
            patch.setattr(backend.resources, "plan_resources", lambda *args: plan)
            patch.setattr(NosaFetchWorkspace, "__init__", fail_constructor)
            patch.setattr(torch.cuda, "synchronize", fail_drain)
            with pytest.raises(BaseExceptionGroup, match="operation and cleanup") as caught:
                backend.allocate_shared(plan, owner=owner)
            assert str(caught.value.exceptions[0]) == "injected partial constructor failure"
            assert "Unable to drain" in str(caught.value.exceptions[1])
        assert drains == [torch.device("cuda:0")]
        assert partial[0]() is not None
        assert backend.resources.fetch_workspace is None
        assert str(backend.resources._allocation_failure) == "injected partial constructor failure"
        assert backend.resources._admission_owner is owner
        assert backend.resources.poisoned
        with pytest.raises(RuntimeError, match="poisoned"):
            backend.unbind_owner(owner, rollback=True)
    finally:
        backend.resources.poisoned = False
        backend.unbind_owner(owner, rollback=True)
        backend.close()
    # The error group intentionally retains both original tracebacks. Release
    # the test's own observation before checking provider disposal.
    del caught
    gc.collect()
    assert partial[0]() is None
