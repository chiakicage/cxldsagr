"""Verify and release NSYS descendants across process-group boundaries.

Ownership is anchored to the launched NSYS PID incarnation and observed /proc
parent chains. Process names are never evidence of ownership. This helper is
used only for intrusive profiles; check and benchmark lifecycles are unchanged.
"""

from __future__ import annotations

import ctypes
import os
import signal
import time
from pathlib import Path

from scripts.lifecycle import managed_process
from scripts.run import GPUProcessMonitor, gpu_processes, write_json

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.pidfd_open.argtypes = (ctypes.c_int, ctypes.c_uint)
_LIBC.pidfd_open.restype = ctypes.c_int
_LIBC.pidfd_send_signal.argtypes = (
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_uint,
)
_LIBC.pidfd_send_signal.restype = ctypes.c_int


def _pidfd_open(pid):
    """Use libc even when the isolated Python omits its optional pidfd API."""
    fd = _LIBC.pidfd_open(pid, 0)
    if fd == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _pidfd_send_signal(fd, signum):
    if _LIBC.pidfd_send_signal(fd, signum, None, 0) == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = stat[stat.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "start_time_ticks": int(fields[19]),
        "state": fields[0],
    }


class Descendants:
    def __init__(self, root_pid, path):
        root = process_identity(root_pid)
        if root is None:
            raise ProcessLookupError(
                f"Launched NSYS root disappeared before ownership capture: {root_pid}"
            )
        self.root = root
        self.path = Path(path)
        self.known = {(root_pid, root["start_time_ticks"]): {"identity": root, "chain": [root]}}
        self.actions = []
        self.persist()

    def persist(self):
        write_json(
            self.path,
            {
                "schema": "echo-engine-nsys-descendants-v1",
                "root": self.root,
                "descendants": list(self.known.values()),
                "cleanup_actions": self.actions,
                "scope": "Observed PID-incarnation parent chains, retained across reparenting; PGID and names do not establish ownership.",
            },
        )

    def discover(self):
        states = {}
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                identity = process_identity(int(entry.name))
                if identity is not None:
                    states[identity["pid"]] = identity
        changed = False
        while True:
            additions = []
            for identity in states.values():
                key = (identity["pid"], identity["start_time_ticks"])
                if key in self.known:
                    continue
                parent = states.get(identity["ppid"])
                if parent is None:
                    continue
                parent_key = (parent["pid"], parent["start_time_ticks"])
                if parent_key not in self.known:
                    continue
                if identity["start_time_ticks"] < parent["start_time_ticks"]:
                    raise RuntimeError("Child PID incarnation predates its observed parent")
                additions.append(
                    (
                        key,
                        {
                            "identity": identity,
                            "chain": [*self.known[parent_key]["chain"], identity],
                            "first_observed_ns": time.time_ns(),
                        },
                    )
                )
            if not additions:
                break
            self.known.update(additions)
            changed = True
        if changed:
            self.persist()
        return states

    def proof(self, pid, identity=None):
        if identity is None:
            identity = process_identity(pid)
        if identity is None:
            return None
        return self.known.get((pid, identity["start_time_ticks"]))

    def live(self):
        result = []
        for (pid, ticks), proof in self.known.items():
            if pid == self.root["pid"]:
                continue
            identity = process_identity(pid)
            if (
                identity is not None
                and identity["start_time_ticks"] == ticks
                and identity["state"] not in ("Z", "X")
            ):
                result.append((identity, proof))
        return result

    def signal(self, identity, signum):
        pid = identity["pid"]
        try:
            fd = _pidfd_open(pid)
        except ProcessLookupError:
            return
        primary = None
        try:
            current = process_identity(pid)
            if current is None or current["start_time_ticks"] != identity["start_time_ticks"]:
                self.actions.append({"pid": pid, "action": "not_signalled_incarnation_changed"})
                return
            try:
                _pidfd_send_signal(fd, signum)
            except ProcessLookupError:
                return
            self.actions.append(
                {"identity": current, "signal": int(signum), "time_ns": time.time_ns()}
            )
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                os.close(fd)
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup(
                        "PID-fd signalling and descriptor cleanup failed", [primary, cleanup]
                    )
                raise

    def close(self, timeout):
        """Signal only proven descendant incarnations, even after reparenting."""
        self.discover()
        signalled = set()
        deadline = time.monotonic() + timeout
        errors = []
        for signum in (signal.SIGTERM, signal.SIGKILL):
            while True:
                self.discover()
                live = self.live()
                if not live:
                    self.persist()
                    if errors:
                        raise BaseExceptionGroup("Descendant cleanup had errors", errors)
                    return
                for identity, _ in reversed(live):
                    key = (identity["pid"], identity["start_time_ticks"], signum)
                    if key in signalled:
                        continue
                    signalled.add(key)
                    try:
                        self.signal(identity, signum)
                    except BaseException as error:  # noqa: BLE001
                        errors.append(error)
                self.persist()
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            deadline = time.monotonic() + min(timeout, 5)
        remaining = [identity for identity, _ in self.live()]
        if remaining:
            errors.append(RuntimeError(f"Cannot confirm owned NSYS descendants ended: {remaining}"))
        if errors:
            raise BaseExceptionGroup("NSYS descendant cleanup failed", errors)


class ProfileGPUProcessMonitor(GPUProcessMonitor):
    def __init__(self, gpu, root_pid, path, registry):
        super().__init__(gpu, root_pid, path)
        self.registry = registry

    def _run(self):
        try:
            next_query = 0
            while not self.stop_event.is_set():
                self.registry.discover()
                if time.monotonic() >= next_query:
                    self._observe()
                    next_query = time.monotonic() + self.interval
                self.stop_event.wait(0.1)
        except BaseException as error:  # noqa: BLE001
            self.failure = error

    def _observe(self):
        started = time.time_ns()
        clock = time.perf_counter_ns()
        record = {"gpu": self.gpu, "owned_pgid": self.pgid, "query_started_ns": started}
        try:
            processes = gpu_processes(self.gpu)
            self.registry.discover()
            foreign = []
            for process in processes:
                identity = process_identity(process["pid"])
                if identity is None:
                    if process["pid"] not in self.owned_processes:
                        foreign.append(process)
                    else:
                        process["exited_previously_verified_owned_pid"] = True
                    continue
                proof = self.registry.proof(process["pid"], identity)
                prior = self.owned_processes.get(process["pid"])
                process.update(identity)
                if proof is None or (prior is not None and prior != identity["start_time_ticks"]):
                    foreign.append(process)
                else:
                    self.owned_processes[process["pid"]] = identity["start_time_ticks"]
                    process["ownership_chain"] = proof["chain"]
            record.update(processes=processes, foreign=foreign, query_status="complete")
        except BaseException as primary:
            record.update(
                query_status="failed", query_error=repr(primary), processes=None, foreign=[]
            )
            try:
                self._write_record(record)
            except BaseException as error:  # noqa: BLE001
                raise BaseExceptionGroup(
                    "GPU ownership query and recording failed", [primary, error]
                )
            raise
        record.update(
            query_finished_ns=time.time_ns(),
            time_ns=time.time_ns(),
            query_duration_ms=(time.perf_counter_ns() - clock) / 1e6,
        )
        self._write_record(record)
        if foreign:
            raise RuntimeError(
                f"Unproven or unrelated GPU processes during NSYS profile: {foreign}"
            )


def drain_profile_gpu(gpu, registry, monitor, path, timeout=60):
    """Wait for proven exited owners' NVML records; reject any unrelated PID."""
    record = {
        "schema": "echo-engine-nsys-gpu-release-v1",
        "gpu": gpu,
        "status": "draining",
        "root": registry.root,
        "observations": [],
        "owned_process_start_time_ticks": monitor.owned_processes,
        "scope": "Only observed descendant PID incarnations are accepted after cleanup.",
    }
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("NSYS-owned GPU release deadline expired")
            processes = gpu_processes(gpu, timeout=min(60, remaining))
            observation = {"query_finished_ns": time.time_ns(), "processes": processes}
            record["observations"].append(observation)
            for process in processes:
                pid = process["pid"]
                identity = process_identity(pid)
                prior = monitor.owned_processes.get(pid)
                if identity is None:
                    if prior is None:
                        raise RuntimeError(f"Unobserved GPU PID during NSYS cleanup: {pid}")
                else:
                    if registry.proof(pid, identity) is None:
                        raise RuntimeError(
                            f"Unrelated/reused GPU PID during NSYS cleanup: {identity}"
                        )
                    if prior is not None and prior != identity["start_time_ticks"]:
                        raise RuntimeError(
                            f"GPU PID incarnation changed during release: {identity}"
                        )
                    if identity["state"] not in ("Z", "X"):
                        raise RuntimeError(
                            f"Owned GPU process remains live after cleanup: {identity}"
                        )
                process["verified_exited_owner"] = True
            if not processes:
                record.update(
                    status="complete", result="gpu_observed_empty", finished_ns=time.time_ns()
                )
                write_json(path, record)
                return
            write_json(path, record)
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    except BaseException as primary:
        record.update(status="failed", error=repr(primary), finished_ns=time.time_ns())
        try:
            write_json(path, record)
        except BaseException as cleanup:  # noqa: BLE001
            raise BaseExceptionGroup("NSYS release and recording failed", [primary, cleanup])
        raise


def run_profile_owned(args, command, *, cwd, data, logs, environment, timeout):
    registry = monitor = None
    primary = None
    try:
        with managed_process(
            command,
            cwd=cwd,
            env=environment,
            stdout_path=logs / "worker.stdout.log",
            stderr_path=logs / "worker.stderr.log",
            event_path=data / "process_events.jsonl",
            shutdown_timeout=args.shutdown_timeout,
        ) as process:
            registry = Descendants(process.process.pid, data / "process_ownership.json")
            nested = None
            try:
                with ProfileGPUProcessMonitor(
                    args.gpu, process.process.pid, data / "gpu_processes.jsonl", registry
                ) as monitor:
                    process.wait(timeout, dependencies=(monitor,))
            except BaseException as error:
                nested = error
                raise
            finally:
                try:
                    registry.close(args.shutdown_timeout)
                except BaseException as cleanup:
                    if nested is not None:
                        raise BaseExceptionGroup(
                            "Profile execution and descendant cleanup failed", [nested, cleanup]
                        )
                    raise
    except BaseException as error:
        primary = error
        raise
    finally:
        if registry is not None and monitor is not None:
            try:
                drain_profile_gpu(args.gpu, registry, monitor, data / "gpu_release_drain.json")
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup(
                        "Profile execution and GPU release failed", [primary, cleanup]
                    )
                raise
