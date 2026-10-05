"""Ownership, completion failures and generation transitions without device work."""

import pytest

from cache.lifecycle import ResourceLifecycle


def allocated(*, bound=False):
    lifecycle = ResourceLifecycle("test")
    owner = object() if bound else None
    if bound:
        lifecycle.bind_owner(owner)
    plan = object()
    lifecycle.allocated(plan, owner=owner)
    session = object()
    registration = lifecycle.register(session, owner=owner)
    return lifecycle, owner, plan, session, registration


@pytest.mark.parametrize("bound", [False, True])
def test_one_authorized_operation_and_read_only_session_validation(bound):
    lifecycle, owner, _, session, registration = allocated(bound=bound)
    lifecycle.check_session(session, registration)
    with lifecycle.execution(session, registration, owner=owner):
        lifecycle.check_execution(session)
        lifecycle.check_mutation(session)
        with (
            pytest.raises(RuntimeError, match="Concurrent"),
            lifecycle.execution(session, registration, owner=owner),
        ):
            pytest.fail("a second execution acquired the same resources")
        with pytest.raises(RuntimeError, match="conflicts"):
            lifecycle.check_mutation(session, releasing=True)
    assert lifecycle.active_session is None
    assert not lifecycle.poisoned


def test_bound_owner_rejects_direct_and_foreign_mutation():
    lifecycle, owner, _, session, registration = allocated(bound=True)
    # Session inspection is safe without supplying an admission owner.
    lifecycle.check_session(session, registration)
    for caller in (None, object()):
        with pytest.raises(RuntimeError, match="admission owner"):
            lifecycle.register(object(), owner=caller)
        with (
            pytest.raises(RuntimeError, match="admission owner"),
            lifecycle.execution(session, registration, owner=caller),
        ):
            pytest.fail("unauthorized execution")
        with (
            pytest.raises(RuntimeError, match="admission owner"),
            lifecycle.mutation(session, registration, owner=caller),
        ):
            pytest.fail("unauthorized mutation")
    with pytest.raises(RuntimeError, match="admission owner"):
        lifecycle.check_mutation(session)
    with lifecycle.mutation(session, registration, owner=owner, releasing=True):
        lifecycle.check_mutation(session, releasing=True)
        lifecycle.check_mutation(session)  # Internal release -> reset remains authorized.
        with pytest.raises(RuntimeError, match="execution lease"):
            lifecycle.check_execution(session)
        lifecycle.detach(session)
    assert not lifecycle.sessions


@pytest.mark.parametrize("prepare_fails", [False, True])
def test_execution_failure_still_drains_and_preserves_original_error(prepare_fails):
    lifecycle, owner, _, session, registration = allocated(bound=True)
    failure = ValueError("execution failed")
    events = []

    def prepare():
        events.append("prepare")
        if prepare_fails:
            raise failure

    with (
        pytest.raises(ValueError) as caught,
        lifecycle.execution(
            session,
            registration,
            owner=owner,
            prepare=prepare,
            drain=lambda: events.append("drain"),
        ),
    ):
        events.append("body")
        raise failure
    assert caught.value is failure
    assert events == (["prepare", "drain"] if prepare_fails else ["prepare", "body", "drain"])
    assert lifecycle.active_session is None
    assert not lifecycle.poisoned
    with lifecycle.execution(session, registration, owner=owner):
        lifecycle.check_execution(session)


def test_both_execution_and_drain_errors_survive_with_poisoned_ownership():
    lifecycle, owner, plan, session, registration = allocated(bound=True)
    body_error = ValueError("body")
    drain_error = RuntimeError("completion unknown")

    def drain():
        raise drain_error

    with (
        pytest.raises(BaseExceptionGroup) as caught,
        lifecycle.execution(session, registration, owner=owner, drain=drain),
    ):
        raise body_error
    assert caught.value.exceptions == (body_error, drain_error)
    assert lifecycle.poisoned
    assert lifecycle.active_session is session
    assert lifecycle.admission_owner is owner
    assert lifecycle.plan is plan
    assert lifecycle.sessions == {session}
    for operation in (
        lambda: lifecycle.check_access(owner),
        lambda: lifecycle.unbind_owner(owner),
        lambda: lifecycle.close(lambda: pytest.fail("unsafe release")),
        lifecycle.drop_plan,
    ):
        with pytest.raises(RuntimeError):
            operation()


def test_provider_poison_is_latched_after_successful_cleanup():
    lifecycle, owner, _, session, registration = allocated(bound=True)
    failure = RuntimeError("earlier completion could not be confirmed")
    drained = []

    def prepare():
        lifecycle.poison(failure)
        raise failure

    with (
        pytest.raises(RuntimeError) as caught,
        lifecycle.execution(
            session, registration, owner=owner, prepare=prepare, drain=lambda: drained.append(True)
        ),
    ):
        pytest.fail("prepare failed")
    assert caught.value is failure
    assert drained == [True]
    assert lifecycle.active_session is session
    assert lifecycle.failure is failure


@pytest.mark.parametrize("preallocated", [False, True])
def test_construction_rollback_only_releases_new_storage(preallocated):
    lifecycle = ResourceLifecycle("test")
    plan, owner = object(), object()
    if preallocated:
        lifecycle.allocated(plan)
    lifecycle.bind_owner(owner)
    if not preallocated:
        lifecycle.allocated(plan, owner=owner)
    released = []

    def release():
        released.append(plan)
        lifecycle.drop_plan()

    lifecycle.unbind_owner(owner, rollback=True, release_new=release)
    assert released == ([] if preallocated else [plan])
    assert lifecycle.plan is (plan if preallocated else None)
    assert lifecycle.admission_owner is None


def test_failed_rollback_keeps_owner_and_generation():
    lifecycle = ResourceLifecycle("test")
    owner, plan = object(), object()
    lifecycle.bind_owner(owner)
    lifecycle.allocated(plan, owner=owner)
    failure = RuntimeError("cannot release provider storage")

    def release():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        lifecycle.unbind_owner(owner, rollback=True, release_new=release)
    assert caught.value is failure
    assert lifecycle.admission_owner is owner
    assert lifecycle.plan is plan
    assert lifecycle.poisoned


def test_failed_allocation_cleanup_preserves_both_errors():
    lifecycle = ResourceLifecycle("test")
    original, completion = ValueError("allocation"), RuntimeError("drain")

    def cleanup():
        raise completion

    with pytest.raises(BaseExceptionGroup) as caught:
        lifecycle.cleanup_after_failure(original, cleanup)
    assert caught.value.exceptions == (original, completion)
    assert lifecycle.poisoned


def test_close_is_idempotent_and_explicit_allocation_opens_a_new_generation():
    lifecycle, _, plan, session, registration = allocated()
    lifecycle.detach(session)
    released = []
    lifecycle.close(lambda: released.append(plan))
    lifecycle.close(lambda: pytest.fail("closed generation released twice"))
    assert lifecycle.closed
    assert lifecycle.plan is None
    assert released == [plan]
    lifecycle.allocated(object())
    assert not lifecycle.closed
    assert lifecycle.generation > registration.generation
    with pytest.raises(RuntimeError, match="Stale"):
        lifecycle.check_session(session, registration)
    lifecycle.check_session(session, registration, released=True, allow_released=True)


def test_explicit_owned_session_can_exist_without_a_shared_plan():
    lifecycle, session = ResourceLifecycle("owned"), object()
    with pytest.raises(RuntimeError, match="allocate"):
        lifecycle.register(session)
    registration = lifecycle.register(session, allow_unplanned=True)
    with lifecycle.execution(session, registration):
        lifecycle.check_execution(session)
    with pytest.raises(RuntimeError, match="sessions"):
        lifecycle.allocated(object())
    with pytest.raises(RuntimeError, match="direct sessions"):
        lifecycle.bind_owner(object())


def test_foreign_registration_and_released_generation_remain_distinct():
    lifecycle, _, _, session, registration = allocated()
    foreign, _, _, _, foreign_registration = allocated()
    with pytest.raises(ValueError, match="Foreign"):
        foreign.check_session(session, registration)
    with pytest.raises(ValueError, match="Foreign"):
        lifecycle.check_session(session, foreign_registration, released=True, allow_released=True)


def test_owned_registration_explicitly_reopens_an_empty_closed_generation():
    lifecycle, session = ResourceLifecycle("owned"), object()
    old = lifecycle.register(session, allow_unplanned=True)
    lifecycle.detach(session)
    lifecycle.close(lambda: None)
    with pytest.raises(RuntimeError, match="closed"):
        lifecycle.register(session)
    current = lifecycle.register(session, allow_unplanned=True)
    assert current.generation > old.generation
    assert not lifecycle.closed
    with pytest.raises(RuntimeError, match="Stale"):
        lifecycle.check_session(session, old)
    lifecycle.check_session(session, current)
