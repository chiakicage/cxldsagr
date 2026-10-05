"""Resource ownership and completion state, independent of model or device APIs."""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class SessionRegistration:
    """Immutable generation and binding identities, never a runner reference."""

    identity: object
    generation: int
    owner: object | None


class ResourceLifecycle:
    """One admission owner and one active operation for each resource generation.

    Providers implement allocation, stream readiness and completion callbacks.
    A failed completion keeps the active session, owner and provider storage
    reachable. Closing ends a resource generation; explicit allocation or owner
    binding may reopen it, while a public runtime can enforce terminal close.
    """

    def __init__(self, label):
        self.label = label
        self.identity = object()
        self.plan = None
        self.generation = 0
        self._sessions = set()
        self._admission_owner = None
        self._owner_identity = None
        self.admission_initial_plan = None
        self._active_session = None
        self.active_kind = None
        self.poisoned = False
        self.closed = False
        self.failure = None
        self._lock = Lock()

    @property
    def sessions(self):
        return self._sessions

    @property
    def admission_owner(self):
        return self._admission_owner

    @admission_owner.setter
    def admission_owner(self, value):
        self._admission_owner = value

    @property
    def active_session(self):
        return self._active_session

    @active_session.setter
    def active_session(self, value):
        self._active_session = value

    def poison(self, error=None):
        self.poisoned = True
        if error is not None:
            self.failure = error

    def check_access(self, owner=None, *, allow_closed=False):
        if self.poisoned:
            raise RuntimeError(f"{self.label} resources are poisoned")
        if self.closed and not allow_closed:
            raise RuntimeError(f"{self.label} resource generation is closed")
        if owner is not self.admission_owner:
            if self.admission_owner is not None:
                raise RuntimeError(f"{self.label} resources require their admission owner")
            raise ValueError("Foreign admission owner")

    def bind_owner(self, owner):
        if owner is None:
            raise ValueError("Admission requires a non-None owner identity")
        if self.poisoned:
            raise RuntimeError(f"Cannot bind poisoned {self.label} resources")
        if self.admission_owner is not None:
            raise RuntimeError("Backend already has an admission owner")
        if self.sessions or self.active_session is not None:
            raise RuntimeError("Cannot bind admission with existing direct sessions or lease")
        self.admission_owner = owner
        self._owner_identity = object()
        self.admission_initial_plan = self.plan
        self.closed = False

    def unbind_owner(self, owner, *, rollback=False, release_new=None):
        if owner is None or self.admission_owner is not owner:
            raise ValueError("Foreign admission owner")
        if self.sessions or self.active_session is not None:
            raise RuntimeError("Release sessions and leases before unbinding admission")
        self.check_access(owner)
        if rollback and self.admission_initial_plan is None:
            if release_new is None:
                raise ValueError("Rollback requires a provider release callback")
            try:
                release_new()
            except BaseException as error:
                self.poison(error)
                raise
        self.admission_owner = self.admission_initial_plan = None
        self._owner_identity = None

    def allocated(self, plan, *, owner=None):
        self.check_access(owner, allow_closed=True)
        if self.sessions or self.active_session is not None:
            raise RuntimeError("Cannot allocate a new generation with sessions or a lease")
        if plan is None or self.plan is not None:
            raise RuntimeError("Allocation requires a new nonempty resource plan")
        self.plan = plan
        self.generation += 1
        self.closed = False

    def drop_plan(self):
        if self.sessions or self.active_session is not None:
            raise RuntimeError("Cannot drop a plan with sessions or a lease")
        if self.poisoned:
            raise RuntimeError(f"Cannot drop poisoned {self.label} resources")
        self.plan = None
        self.failure = None

    def cleanup_after_failure(self, error, cleanup):
        """Keep the initiating failure when provider cleanup also fails."""
        try:
            cleanup()
        except BaseException as cleanup_error:  # noqa: BLE001 -- preserve both failures.
            self.poison(cleanup_error)
            raise BaseExceptionGroup(
                f"{self.label} operation and cleanup both failed", [error, cleanup_error]
            ) from None

    def register(self, session, *, owner=None, allow_unplanned=False):
        self.check_access(owner, allow_closed=allow_unplanned)
        if self.plan is None and not allow_unplanned:
            raise RuntimeError("Plan and allocate resources before registering sessions")
        if session in self.sessions:
            raise RuntimeError("Expected a new live session")
        if self.closed:
            if self.plan is not None or self.sessions or self.active_session is not None:
                raise RuntimeError("Owned registration requires an empty closed generation")
            self.generation += 1
            self.closed = False
        registration = SessionRegistration(self.identity, self.generation, self._owner_identity)
        self.sessions.add(session)
        return registration

    def check_session(self, session, registration, *, released=False, allow_released=False):
        if (
            not isinstance(registration, SessionRegistration)
            or registration.identity is not self.identity
        ):
            raise ValueError("Foreign session belongs to another backend")
        if allow_released and released:
            return
        if released or session not in self.sessions or registration.generation != self.generation:
            raise RuntimeError(f"Stale or released {self.label} session")

    def detach(self, session):
        self.sessions.remove(session)

    def check_execution(self, session):
        if self.poisoned:
            raise RuntimeError(f"{self.label} resources are poisoned")
        if self.active_session is not session or self.active_kind != "execution":
            raise RuntimeError(f"Borrowed {self.label} cache requires an active execution lease")

    def check_mutation(self, session, *, releasing=False):
        if self.poisoned:
            raise RuntimeError(f"Cannot mutate a session with poisoned {self.label} resources")
        if self.active_session is None:
            self.check_access()
            return
        if self.active_session is not session or releasing and self.active_kind != "release":
            raise RuntimeError("Session mutation conflicts with an execution lease")

    @contextmanager
    def operation(self, session, registration, *, owner=None, kind, prepare=None, drain=None):
        self.check_session(session, registration)
        self.check_access(owner)
        if kind not in ("execution", "mutation", "release"):
            raise ValueError("Unknown resource operation kind")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent execution lease or mutation is not supported")
        if self.active_session is not None:
            self._lock.release()
            raise RuntimeError("An earlier resource operation has not completed")
        self.active_session, self.active_kind = session, kind
        body_error = cleanup_error = None
        try:
            try:
                if prepare is not None:
                    prepare()
                yield self
            except BaseException as error:  # noqa: BLE001 -- drain before propagating.
                body_error = error
            try:
                if drain is not None:
                    drain()
            except BaseException as error:  # noqa: BLE001 -- retain poisoned ownership.
                cleanup_error = error
                self.poison(error)
            if not self.poisoned:
                self.active_session = self.active_kind = None
        finally:
            self._lock.release()
        if body_error is not None and cleanup_error is not None:
            raise BaseExceptionGroup(
                f"{self.label} execution and completion both failed", [body_error, cleanup_error]
            ) from None
        if cleanup_error is not None:
            raise cleanup_error
        if body_error is not None:
            raise body_error

    def execution(self, session, registration, *, owner=None, prepare=None, drain=None):
        return self.operation(
            session, registration, owner=owner, kind="execution", prepare=prepare, drain=drain
        )

    def mutation(self, session, registration, *, owner=None, releasing=False):
        return self.operation(
            session, registration, owner=owner, kind="release" if releasing else "mutation"
        )

    def close(self, release):
        if self.sessions or self.active_session is not None or self.admission_owner is not None:
            raise RuntimeError(
                "Cannot close resources with live sessions, lease or admission owner"
            )
        self.check_access(allow_closed=True)
        if self.closed:
            return
        try:
            release()
        except BaseException as error:
            self.poison(error)
            raise
        self.drop_plan()
        self.closed = True
