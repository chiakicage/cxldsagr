"""CPU loader and literal-helper checks for the optional pool-referrer adapter."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.nosa import _pool_referrers as adapter
from models.nosa import allocation_budget as budget


@pytest.fixture(autouse=True)
def reset_provider(monkeypatch):
    monkeypatch.setattr(adapter, "_selection", None)
    monkeypatch.setattr(adapter, "_setup_claim", {})
    monkeypatch.setattr(budget, "_resolve_pool_referrers", adapter.identity_resolver)


def fake_module():
    configurations = []
    counters = {"filtered_calls": 0, "unfiltered_calls": 0, "fallback_calls": 0, "audit_errors": 0}
    module = SimpleNamespace(
        configure_helper=lambda *args: configurations.append(args),
        resolve_referrers=lambda target: target,
        counters=lambda: dict(counters),
    )
    runtime = {
        "backend": "cpython_native",
        "fallback_reason": None,
        "fingerprint": "fixture",
        "abi_manifest_sha256": adapter._ABI_MANIFEST_SHA256,
        "loaded_binary_path": "fixture.so",
        "loaded_binary_sha256": "fixture",
    }
    return module, runtime, configurations, counters


def mock_cuda_validation(monkeypatch):
    settings = {
        "max_split_size": -1,
        "roundup_power2_divisions": {str(1 << i): 0 for i in range(16)},
        "expandable_segments": False,
        "graph_capture_record_stream_reuse": False,
        "PYTORCH_CUDA_ALLOC_CONF": "",
    }
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch._C, "_cuda_cudaCachingAllocator_is_enabled", lambda: True)
    monkeypatch.setattr(torch._C, "_accelerator_getAllocatorSettings", lambda: "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        budget,
        "_allocator_snapshot",
        lambda: ({"allocator_settings": settings, "segments": []}, {}),
    )
    for name in budget._NO_CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_import_and_cpu_policy_do_not_prepare_or_initialize_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU/import must not prepare native code or inspect CUDA")

    monkeypatch.setattr(adapter, "_load_native", forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    assert adapter.runtime_info() is None
    assert budget.validate_allocator("cpu")["configuration_key"] == budget.CPU_POLICY
    assert budget.allocation_bytes(17, "cpu") == 17
    assert budget.pinned_allocation_bytes(17, "cpu") == 17
    assert adapter.runtime_info() is None


def test_prepare_once_runtime_info_is_fresh_and_has_no_filesystem_work(monkeypatch):
    module, runtime, configurations, counters = fake_module()
    loads = []

    def load():
        loads.append(True)
        return module, runtime

    monkeypatch.setattr(adapter, "_load_native", load)
    resolver = adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    assert resolver is module.resolve_referrers
    assert adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types) is resolver
    assert len(loads) == len(configurations) == 1
    monkeypatch.setattr(adapter, "_digest", lambda *args: pytest.fail("runtime_info did file work"))
    evidence = adapter.runtime_info()
    evidence["backend"] = "modified"
    evidence["counters"]["filtered_calls"] = 99
    counters["filtered_calls"] = 3
    assert adapter.runtime_info()["backend"] == "cpython_native"
    assert adapter.runtime_info()["counters"]["filtered_calls"] == 3


def test_setup_failure_falls_back_once_with_reason(monkeypatch):
    calls = []

    def unavailable():
        calls.append(True)
        raise adapter._SetupUnavailable("compiler absent")

    monkeypatch.setattr(adapter, "_load_native", unavailable)
    target = lambda: "original"
    for _ in range(2):
        resolver = adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
        assert resolver(target) is target and resolver(target)() == "original"
    assert len(calls) == 1
    assert adapter.runtime_info()["backend"] == "python_original"
    assert "compiler absent" in adapter.runtime_info()["fallback_reason"]
    assert adapter.runtime_info()["fingerprint"] is None


@pytest.mark.parametrize(
    "error_type",
    [OSError, ImportError, subprocess.SubprocessError, ValueError, KeyError, TypeError],
)
@pytest.mark.parametrize("entrypoint", ["prepare", "build_info"])
def test_unclassified_setup_error_propagates_exactly_without_publication(
    monkeypatch, entrypoint, error_type
):
    expected = error_type("unexpected setup callback failure")

    def fail():
        raise expected

    target = "_load_native" if entrypoint == "prepare" else "_build_identity"
    monkeypatch.setattr(adapter, target, fail)
    with pytest.raises(error_type) as observed:
        if entrypoint == "prepare":
            adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
        else:
            adapter.build_info()
    assert observed.value is expected
    assert adapter.runtime_info() is None and adapter._setup_claim == {}


def test_configure_valueerror_is_not_reclassified_by_matching_its_message(monkeypatch):
    module, runtime, _, _ = fake_module()
    expected = ValueError("configure once with exact Python functions")

    def fail(*args):
        raise expected

    module.configure_helper = fail
    monkeypatch.setattr(adapter, "_load_native", lambda: (module, runtime))
    with pytest.raises(ValueError) as observed:
        adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    assert observed.value is expected
    assert adapter.runtime_info() is None and adapter._setup_claim == {}


@pytest.mark.parametrize("error_type", [AttributeError, ValueError])
def test_symbol_resolution_error_is_not_reclassified(monkeypatch, error_type):
    expected = error_type("callback during symbol lookup")

    class Symbols:
        def __getattr__(self, name):
            raise expected

    monkeypatch.setattr(adapter.ctypes, "pythonapi", Symbols())
    with pytest.raises(error_type) as observed:
        adapter._symbol_hosts(["PyInterpreterState_Get"], Path(sys.executable).resolve())
    assert observed.value is expected


def test_selected_runtime_exception_propagates_without_retry(monkeypatch):
    module, runtime, _, _ = fake_module()
    calls = []

    def fail():
        calls.append(True)
        raise RuntimeError("selected scan failed")

    module.resolve_referrers = lambda target: fail
    monkeypatch.setattr(adapter, "_load_native", lambda: (module, runtime))
    resolver = adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    with pytest.raises(RuntimeError, match="selected scan failed"):
        resolver(lambda: pytest.fail("runtime fallback is forbidden"))()
    assert calls == [True] and adapter.runtime_info()["backend"] == "cpython_native"


@pytest.mark.parametrize("worker_thread", [False, True])
def test_initialization_callback_validation_uses_original_without_deadlock_or_publication(
    monkeypatch, worker_thread
):
    mock_cuda_validation(monkeypatch)
    module, runtime, configurations, _ = fake_module()
    inspected = []

    class Pool:
        pass

    def inspect():
        inspected.append(adapter.runtime_info())
        assert not budget._has_python_pool_referrers(Pool)
        if len(inspected) == 1:
            assert adapter.runtime_info() is None
            assert budget._resolve_pool_referrers is adapter.identity_resolver

    monkeypatch.setattr(budget, "_reject_python_pools", inspect)

    def load():
        assert adapter.runtime_info() is None
        if worker_thread:
            with ThreadPoolExecutor(max_workers=1) as threads:
                nested = threads.submit(budget.validate_allocator, "cuda:0").result(timeout=5)
        else:
            nested = budget.validate_allocator("cuda:0")
        assert nested["policy"] == budget.CUDA_POLICY
        assert adapter.runtime_info() is None and not configurations
        return module, runtime

    monkeypatch.setattr(adapter, "_load_native", load)
    result = budget.validate_allocator("cuda:0")
    assert result["policy"] == budget.CUDA_POLICY
    assert len(inspected) == 2 and inspected[0] is None
    assert inspected[1]["backend"] == "cpython_native"
    assert len(configurations) == 1 and budget._resolve_pool_referrers is module.resolve_referrers


def test_callback_runtime_error_does_not_become_cached_setup_fallback(monkeypatch):
    mock_cuda_validation(monkeypatch)

    def reject():
        raise RuntimeError("nested original inspection failed")

    monkeypatch.setattr(budget, "_reject_python_pools", reject)
    monkeypatch.setattr(adapter, "_load_native", lambda: budget.validate_allocator("cuda:0"))
    with pytest.raises(RuntimeError, match="nested original inspection failed"):
        budget.validate_allocator("cuda:0")
    assert adapter.runtime_info() is None and adapter._setup_claim == {}


def test_trace_exception_after_setup_claim_releases_owner_and_allows_later_setup(monkeypatch):
    module, runtime, configurations, _ = fake_module()
    monkeypatch.setattr(adapter, "_load_native", lambda: (module, runtime))
    raised = []

    def trace(frame, event, arg):
        if (
            frame.f_code is adapter.prepare.__code__
            and event == "line"
            and adapter._setup_claim
            and not raised
        ):
            raised.append(True)
            raise RuntimeError("trace interrupted claimed setup")
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        with pytest.raises(RuntimeError, match="trace interrupted claimed setup"):
            adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    finally:
        sys.settrace(previous)
    assert raised == [True] and adapter._setup_claim == {} and adapter.runtime_info() is None
    resolver = adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    assert resolver is module.resolve_referrers and len(configurations) == 1


def test_trace_during_setup_can_join_another_validating_thread(monkeypatch):
    mock_cuda_validation(monkeypatch)
    module, runtime, configurations, _ = fake_module()
    monkeypatch.setattr(adapter, "_load_native", lambda: (module, runtime))
    monkeypatch.setattr(budget, "_reject_python_pools", lambda: None)
    entered, nested_results, nested_errors = [], [], []

    def worker():
        try:
            nested_results.append(budget.validate_allocator("cuda:0"))
        except BaseException as error:  # noqa: BLE001 - transport worker failures to the assertion.
            nested_errors.append(error)

    def trace(frame, event, arg):
        if (
            frame.f_code is adapter.prepare.__code__
            and event == "line"
            and adapter._setup_claim
            and not entered
        ):
            entered.append(True)
            assert adapter.runtime_info() is None
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive(), "setup trace callback deadlocked joined validation"
            assert not configurations and adapter.runtime_info() is None
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        result = budget.validate_allocator("cuda:0")
    finally:
        sys.settrace(previous)
    assert entered == [True] and not nested_errors
    assert nested_results[0]["policy"] == result["policy"] == budget.CUDA_POLICY
    assert len(configurations) == 1 and adapter.runtime_info()["backend"] == "cpython_native"


@pytest.mark.parametrize("which", ["helper", "inventory"])
def test_callable_object_substitution_uses_original_guard_and_explicit_fallback(monkeypatch, which):
    mock_cuda_validation(monkeypatch)
    name = "_has_python_pool_referrers" if which == "helper" else "_python_pool_types"
    original = getattr(budget, name)
    calls = []

    class Callable:
        def __call__(self, *args):
            calls.append(True)
            return original(*args)

    class Pool:
        pass

    monkeypatch.setattr(budget, name, Callable())
    monkeypatch.setattr(torch.cuda, "MemPool", Pool)
    monkeypatch.setattr(budget, "_pool_scan_state", None)
    monkeypatch.setattr(
        adapter, "_load_native", lambda: pytest.fail("unsupported helper was loaded")
    )
    assert budget.validate_allocator("cuda:0")["policy"] == budget.CUDA_POLICY
    assert calls and adapter.runtime_info()["backend"] == "python_original"
    assert "must be Python functions" in adapter.runtime_info()["fallback_reason"]


def test_transient_identity_result_cannot_overwrite_selected_budget_resolver(monkeypatch):
    mock_cuda_validation(monkeypatch)
    selected = lambda target: target
    monkeypatch.setattr(budget, "_resolve_pool_referrers", selected)
    monkeypatch.setattr(adapter, "prepare", lambda *args: adapter.identity_resolver)
    monkeypatch.setattr(budget, "_reject_python_pools", lambda: None)
    budget.validate_allocator("cuda:0")
    assert budget._resolve_pool_referrers is selected


@pytest.fixture
def mock_build(tmp_path, monkeypatch):
    header = tmp_path / "included.h"
    header.write_text("original")
    identity = {
        "extension_suffix": ".so",
        "source_and_dependency_sha256": {str(header): adapter._digest(header)},
    }
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def fresh():
        value = deepcopy(identity)
        value["source_and_dependency_sha256"][str(header)] = adapter._digest(header)
        return value, ["fake-compiler"], {}

    monkeypatch.setattr(adapter, "_build_identity", fresh)
    builds, imports = [], []

    def compile_(command, environment):
        builds.append(command)
        Path(command[-1]).write_bytes(b"verified extension")
        return ""

    class Loader:
        def exec_module(self, module):
            imports.append(module.__file__)

    monkeypatch.setattr(adapter, "_run", compile_)
    monkeypatch.setattr(adapter, "_loaded_binary_digest", adapter._digest)
    monkeypatch.setattr(
        adapter.importlib.util,
        "spec_from_file_location",
        lambda name, path: SimpleNamespace(loader=Loader(), path=path),
    )
    monkeypatch.setattr(
        adapter.importlib.util,
        "module_from_spec",
        lambda spec: SimpleNamespace(__file__=str(spec.path)),
    )
    return identity, header, builds, imports, tmp_path


def test_cache_publishes_complete_verified_binary_once(mock_build):
    identity, _, builds, imports, _ = mock_build
    results = [adapter._load_native() for _ in range(3)]
    assert len(builds) == 1 and len(imports) == 3
    for _, runtime in results:
        binary = Path(runtime["loaded_binary_path"])
        assert binary.name == adapter._NAME + identity["extension_suffix"]
        assert binary.parent.name == runtime["fingerprint"] == adapter._fingerprint(identity)
        assert runtime["loaded_binary_sha256"] == adapter._digest(binary)


def test_contended_cache_falls_back_without_waiting_for_callback_owner(mock_build, monkeypatch):
    _, _, builds, imports, _ = mock_build
    original = adapter._run
    nested_results = []

    def compile_with_callback(command, environment):
        with ThreadPoolExecutor(max_workers=1) as threads:
            nested_results.append(
                threads.submit(
                    adapter.prepare, budget._has_python_pool_referrers, budget._python_pool_types
                ).result(timeout=5)
            )
        assert adapter.runtime_info()["backend"] == "python_original"
        assert "already in progress" in adapter.runtime_info()["fallback_reason"]
        assert not imports
        return original(command, environment)

    monkeypatch.setattr(adapter, "_run", compile_with_callback)
    adapter._load_native()
    assert nested_results == [adapter.identity_resolver]
    assert len(builds) == len(imports) == 1


@pytest.mark.parametrize(
    "error_type", [OSError, ImportError, subprocess.SubprocessError, BlockingIOError]
)
def test_unexpected_lock_ffi_error_propagates_and_releases_claim(
    mock_build, monkeypatch, error_type
):
    import fcntl

    expected = error_type("callback during lock FFI invocation")
    operations = []

    class Flock:
        def __call__(self, descriptor, operation):
            operations.append(operation)
            if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
                raise expected
            return 0

    function = Flock()

    def library(name, *, use_errno):
        assert name is None and use_errno is True
        return SimpleNamespace(flock=function)

    monkeypatch.setattr(adapter.ctypes, "CDLL", library)
    with pytest.raises(error_type) as observed:
        adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    assert observed.value is expected
    assert function.argtypes == [adapter.ctypes.c_int, adapter.ctypes.c_int]
    assert function.restype is adapter.ctypes.c_int
    assert operations == [fcntl.LOCK_EX | fcntl.LOCK_NB, fcntl.LOCK_UN]
    assert adapter.runtime_info() is None and adapter._setup_claim == {}


def test_malformed_manifest_error_propagates_without_replacement(mock_build):
    identity, _, builds, imports, root = mock_build
    directory = root / "cache/cxldsagr/pool-referrers" / adapter._fingerprint(identity)
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text("not valid JSON")
    binary = directory / (adapter._NAME + identity["extension_suffix"])
    binary.write_bytes(b"untrusted cache preserved")
    with pytest.raises(json.JSONDecodeError):
        adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    assert (
        manifest.read_text() == "not valid JSON"
        and binary.read_bytes() == b"untrusted cache preserved"
    )
    assert (
        not builds and not imports and adapter.runtime_info() is None and adapter._setup_claim == {}
    )


@pytest.mark.parametrize(
    "module_name,binding",
    [
        ("gc", "get_objects"),
        ("gc", "get_referrers"),
        ("gc", "get_stats"),
        ("gc", "get_freeze_count"),
        ("_signal", "getsignal"),
        ("_signal", "default_int_handler"),
        ("_signal", "valid_signals"),
    ],
)
@pytest.mark.parametrize("replacement", ["wrapper", "other_builtin", "missing"])
def test_modified_initial_builtin_binding_falls_back_before_native_import(
    mock_build, monkeypatch, module_name, binding, replacement
):
    _, _, builds, imports, _ = mock_build
    module = sys.modules[module_name]
    original = getattr(module, binding)
    with monkeypatch.context() as changed:
        if replacement == "missing":
            changed.delattr(module, binding)
        elif replacement == "other_builtin":
            changed.setattr(module, binding, len)
        else:
            changed.setattr(module, binding, lambda *args, **kwargs: original(*args, **kwargs))
        resolver = adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    assert resolver is adapter.identity_resolver
    assert f"{module_name}.{binding} differs" in adapter.runtime_info()["fallback_reason"]
    assert not builds and not imports


@pytest.mark.parametrize("module_name", ["gc", "_signal"])
def test_modified_initial_module_is_rejected_without_attribute_dispatch(
    mock_build, monkeypatch, module_name
):
    _, _, builds, imports, _ = mock_build

    class Untrusted:
        def __getattribute__(self, name):
            pytest.fail("precheck must not dispatch untrusted module attributes")

    with monkeypatch.context() as changed:
        changed.setitem(sys.modules, module_name, Untrusted())
        resolver = adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    assert resolver is adapter.identity_resolver
    assert (
        f"{module_name} must be an exact builtin module"
        in adapter.runtime_info()["fallback_reason"]
    )
    assert not builds and not imports


def test_initial_binding_precheck_does_not_dispatch_module_getattr(mock_build, monkeypatch):
    _, _, builds, imports, _ = mock_build
    module = sys.modules["gc"]

    def fail(name):
        pytest.fail("precheck must inspect the exact module dictionary")

    with monkeypatch.context() as changed:
        changed.delattr(module, "get_referrers")
        changed.setattr(module, "__getattr__", fail, raising=False)
        resolver = adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    assert resolver is adapter.identity_resolver and not builds and not imports


@pytest.mark.parametrize("module_name,binding", [("gc", "get_referrers"), ("_signal", "getsignal")])
def test_rebound_builtin_receiver_uses_original_path(mock_build, monkeypatch, module_name, binding):
    _, _, builds, imports, _ = mock_build
    module = sys.modules[module_name]
    original = getattr(module, binding)

    class MethodDef(ctypes.Structure):
        _fields_ = [
            ("name", ctypes.c_char_p),
            ("method", ctypes.c_void_p),
            ("flags", ctypes.c_int),
            ("doc", ctypes.c_char_p),
        ]

    get_function = ctypes.pythonapi.PyCFunction_GetFunction
    get_function.argtypes, get_function.restype = [ctypes.py_object], ctypes.c_void_p
    get_flags = ctypes.pythonapi.PyCFunction_GetFlags
    get_flags.argtypes, get_flags.restype = [ctypes.py_object], ctypes.c_int
    definition = MethodDef(binding.encode(), get_function(original), get_flags(original), None)
    create = ctypes.pythonapi.PyCFunction_NewEx
    create.argtypes = [ctypes.POINTER(MethodDef), ctypes.py_object, ctypes.py_object]
    create.restype = ctypes.py_object
    receiver = object()
    rebound = create(ctypes.byref(definition), receiver, module_name)
    assert rebound.__self__ is receiver
    # This forged binding is never invoked; only its public identity is read.
    with monkeypatch.context() as changed:
        changed.setattr(module, binding, rebound)
        resolver = adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    assert resolver is adapter.identity_resolver and not builds and not imports
    assert f"{module_name}.{binding} differs" in adapter.runtime_info()["fallback_reason"]


def test_noncontention_lock_errno_propagates(mock_build, monkeypatch):
    import errno
    import fcntl

    class Flock:
        def __call__(self, descriptor, operation):
            if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
                adapter.ctypes.set_errno(errno.EBADF)
                return -1
            return 0

    monkeypatch.setattr(
        adapter.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(flock=Flock())
    )
    with pytest.raises(OSError) as observed:
        adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
    assert observed.value.errno == errno.EBADF
    assert adapter.runtime_info() is None and adapter._setup_claim == {}


def test_compiler_failure_does_not_publish_partial_cache(mock_build, monkeypatch):
    identity, _, _, imports, root = mock_build

    def fail(command, environment):
        Path(command[-1]).write_bytes(b"partial")
        raise adapter._SetupUnavailable("compiler failure")

    monkeypatch.setattr(adapter, "_run", fail)
    with pytest.raises(adapter._SetupUnavailable, match="compiler failure"):
        adapter._load_native()
    cache = root / "cache/cxldsagr/pool-referrers"
    assert not imports and not (cache / adapter._fingerprint(identity)).exists()
    assert list(cache.glob(".*")) == []


@pytest.mark.parametrize("change", ["header", "closure"])
def test_build_input_change_prevents_import_and_publication(mock_build, monkeypatch, change):
    identity, header, _, imports, root = mock_build
    original = adapter._build_identity

    def compile_(command, environment):
        Path(command[-1]).write_bytes(b"unpublished")
        if change == "header":
            header.write_text("changed")
        else:

            def with_new_dependency():
                value, command, environment = original()
                value["source_and_dependency_sha256"]["new.h"] = "0" * 64
                return value, command, environment

            monkeypatch.setattr(adapter, "_build_identity", with_new_dependency)
        return ""

    monkeypatch.setattr(adapter, "_run", compile_)
    with pytest.raises(adapter._SetupUnavailable, match="dependency changed"):
        adapter._load_native()
    assert not imports
    assert not (root / "cache/cxldsagr/pool-referrers" / adapter._fingerprint(identity)).exists()


def test_cache_rechecks_identity_after_lock_and_preserves_corruption(mock_build, monkeypatch):
    _, _, builds, imports, _ = mock_build
    _, runtime = adapter._load_native()
    binary = Path(runtime["loaded_binary_path"])
    calls = []
    original = adapter._build_identity

    def changed():
        value, command, environment = original()
        if calls:
            value["compiler_tools"] = {"ld": "changed"}
        calls.append(True)
        return value, command, environment

    monkeypatch.setattr(adapter, "_build_identity", changed)
    with pytest.raises(adapter._SetupUnavailable, match="dependency changed"):
        adapter._load_native()
    assert len(imports) == len(builds) == 1
    monkeypatch.setattr(adapter, "_build_identity", original)
    binary.write_bytes(b"corrupt cache remains in place")
    assert (
        adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
        is adapter.identity_resolver
    )
    assert binary.read_bytes() == b"corrupt cache remains in place"
    assert "cache integrity failed" in adapter.runtime_info()["fallback_reason"]


def test_actual_identity_is_fresh_without_native_load(monkeypatch):
    monkeypatch.setattr(adapter, "_load_native", lambda: pytest.fail("build_info loaded native"))
    first, second = adapter.build_info(), adapter.build_info()
    assert first["available"], first
    assert first == second
    identity = first["identity"]
    assert identity["header_count"] == len(identity["headers_sha256"]) == 335
    assert identity["module_name"] == adapter._NAME
    assert first["fingerprint"] == adapter._fingerprint(identity)
    assert adapter.runtime_info() is None and not torch.cuda.is_initialized()


@pytest.mark.parametrize("failure", ["missing", "inode", "device", "deleted"])
def test_mapped_binary_identity_rejects_wrong_mapping(tmp_path, failure):
    binary = tmp_path / "adapter.so"
    binary.write_bytes(b"not executable; mapping fixture only")
    stat = binary.stat()
    device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
    inode = stat.st_ino
    suffix = ""
    if failure == "inode":
        inode += 1
    elif failure == "device":
        device = "ff:ff"
    elif failure == "deleted":
        suffix = " (deleted)"
    maps = tmp_path / "maps"
    maps.write_text(
        "" if failure == "missing" else f"1000-2000 r-xp 0 {device} {inode} {binary}{suffix}\n"
    )
    with pytest.raises(adapter._SetupUnavailable, match="mapped|not mapped"):
        adapter._loaded_binary_digest(binary, maps)


def test_whitelist_hash_and_parse_use_same_read(tmp_path, monkeypatch):
    original = adapter._sources()
    copied = tmp_path / "whitelist.json"
    copied.write_bytes(original[2].read_bytes())
    monkeypatch.setattr(adapter, "_sources", lambda: (*original[:2], copied))
    expected = adapter._read_abi()
    copied.write_text(json.dumps({**expected, "executable_sha256": "0" * 64}))
    with pytest.raises(adapter._SetupUnavailable, match="whitelist bytes changed"):
        adapter._read_abi()


@pytest.mark.parametrize("failure", ["executable", "symbol_host", "header_growth", "header_loss"])
def test_actual_authentication_negatives_choose_unavailable(monkeypatch, failure):
    if failure == "executable":
        abi = adapter._read_abi()
        abi["executable_sha256"] = "0" * 64
        monkeypatch.setattr(adapter, "_read_abi", lambda: abi)
    elif failure == "symbol_host":

        def wrong(*args):
            raise adapter._SetupUnavailable("Python symbol host differs")

        monkeypatch.setattr(adapter, "_symbol_hosts", wrong)
    else:
        run = adapter._run

        def changed(command, environment):
            value = run(command, environment)
            if "-M" in command:
                if failure == "header_growth":
                    value += " /additional/header.h"
                else:
                    value = value.replace(next(iter(adapter._read_abi()["headers"])), "")
            return value

        monkeypatch.setattr(adapter, "_run", changed)
    value = adapter.build_info()
    assert not value["available"] and value["reason"]
    assert adapter.runtime_info() is None and not torch.cuda.is_initialized()


def test_actual_abi_and_manifest_open_audit_errors_keep_original_helper_exception(tmp_path):
    script = r"""
import json, os, subprocess, sys, types
from pathlib import Path
import torch
from models.nosa import _pool_referrers as adapter
from models.nosa import allocation_budget as budget
assert adapter.runtime_info() is None and not torch.cuda.is_initialized()
root = Path(sys.argv[1])
identity = {'extension_suffix': '.so'}
(root / 'manifest.json').write_text(json.dumps({'build_identity': identity, 'binary_sha256': 'invalid'}))
(root / (adapter._NAME + '.so')).write_bytes(b'audit fixture only')
records = []
compiler_calls = []
def count_compiler(event, args):
    if event == 'subprocess.Popen': compiler_calls.append(args)
sys.addaudithook(count_compiler)
for operation in ('prepare', 'build_info', 'manifest'):
    for error_type in (OSError, ImportError, subprocess.SubprocessError, ValueError, KeyError, TypeError, BlockingIOError):
        expected = error_type('original helper failed during actual loader file-open audit')
        state = {'active': True, 'callbacks': 0, 'target_calls': 0}
        class Pool: pass
        def fail(*args):
            state['target_calls'] += 1
            raise expected
        budget.gc = types.SimpleNamespace(get_referrers=fail)
        ending = 'manifest.json' if operation == 'manifest' else 'pool_referrers_abi.json'
        def audit(event, args, state=state, ending=ending):
            if state['active'] and event == 'open' and str(args[0]).endswith(ending):
                state['active'] = False
                state['callbacks'] += 1
                budget._has_python_pool_referrers(Pool)
        sys.addaudithook(audit)
        try:
            try:
                if operation == 'prepare':
                    adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
                elif operation == 'build_info': adapter.build_info()
                else: adapter._verified_manifest(root, identity)
            except error_type as observed:
                assert observed is expected
            else: raise AssertionError('original helper exception was swallowed')
        finally:
            state['active'] = False
        assert state['callbacks'] == state['target_calls'] == 1, state
        assert adapter.runtime_info() is None and adapter._setup_claim == {}
        records.append({'operation':operation,'exception':error_type.__name__,'same_exception':True,'callbacks':1,'target_calls':1})
assert not compiler_calls and not torch.cuda.is_initialized()
receipt = {'records':records,'cuda_initialized':False,'compiler_calls':0,
           'cpu_affinity':sorted(os.sched_getaffinity(0)),
           'boundary':'Actual ABI/manifest file-open audit callbacks reenter the literal original helper; 21 exact exception checks, no native load or compiler.'}
(root / 'callback_exception_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'passed','records':len(records)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    (tmp_path / "child.stdout").write_text(result.stdout)
    (tmp_path / "child.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "callback_exception_receipt.json").read_text())
    assert len(receipt["records"]) == 21 and not receipt["cuda_initialized"]


def test_literal_integrated_native_helper_in_fresh_process(tmp_path):
    script = r"""
import gc, json, os, sys, types
from pathlib import Path
import torch
from models.nosa import _pool_referrers as adapter
from models.nosa import allocation_budget as budget
assert Path(adapter.__file__).parent == Path(budget.__file__).parent
assert budget._has_python_pool_referrers.__globals__ is vars(budget)
assert not torch.cuda.is_initialized()
build = adapter.build_info()
assert build['available'], build
resolver = adapter.prepare(budget._has_python_pool_referrers, budget._python_pool_types)
assert adapter.runtime_info()['backend'] == 'cpython_native', adapter.runtime_info()
budget._resolve_pool_referrers = resolver
native = adapter._selection[2]
records = []
class Pool: pass
class Child(Pool): __slots__ = ()
class Grandchild(Child): __slots__ = ()
assert not budget._has_python_pool_referrers(Pool)
assert native.last_scan()['route'] == 'filtered_ordinary', native.last_scan()
old_pool = torch.cuda.MemPool
torch.cuda.MemPool = Pool
def full_guard():
    budget._pool_scan_state = None
    return budget._reject_python_pools()
try:
    for depth, kind in enumerate((Pool, Child, Grandchild)):
        for generation in range(3):
            value = kind()
            if generation: gc.collect(generation - 1)
            try: full_guard()
            except NotImplementedError: pass
            else: raise AssertionError('live pool admitted')
            del value
            full_guard()
            records.append({'case':'depth_generation','depth':depth,'generation':generation})
    value = Pool()
    gc.freeze()
    full_guard()  # Same documented frozen-object exclusion as original inspection.
    gc.unfreeze()
    try: full_guard()
    except NotImplementedError: pass
    else: raise AssertionError('unfrozen pool admitted')
    del value
    full_guard()
    records.append({'case':'freeze_unfreeze_boundary'})
finally:
    gc.unfreeze()
    torch.cuda.MemPool = old_pool
    budget._pool_scan_state = None
state = {'active':False,'action':None,'audits':0}
def audit(event,args):
    if event == 'gc.get_referrers' and state['active']:
        state['audits'] += 1
        if state['action'] is not None:
            action, state['action'] = state['action'], None
            action()
sys.addaudithook(audit)
enabled = gc.isenabled()
gc.disable()
try:
    old_inventory, old_gc, original_getter = budget._python_pool_types, budget.gc, gc.get_referrers
    wrong, held = [], []
    def replacement(*args): wrong.append(True); return []
    class Kinds(dict):
        def values(self):
            gc.get_referrers = replacement
            held.append(Pool())
            return super().values()
    budget._python_pool_types = lambda kind: Kinds(old_inventory(kind))
    state.update(active=True,action=None,audits=0)
    try:
        assert budget._has_python_pool_referrers(Pool)
        assert not wrong and state['audits'] == 1
        assert native.last_scan()['route'] == 'fallback_identity_before_audit'
        records.append({'case':'getter_before_star_builtin','audits':state['audits']})
    finally:
        gc.get_referrers = original_getter
        budget._python_pool_types = old_inventory
        state['active'] = False
        held.clear()
    log = []
    def first(*args): log.append('call_first'); return []
    def second(*args): log.append('call_second'); return []
    class Proxy:
        def __getattr__(self,name):
            assert name == 'get_referrers'
            log.append('lookup_first')
            return first
    class CustomKinds(dict):
        def values(self):
            log.append('expand_starargs')
            budget.gc = types.SimpleNamespace(get_referrers=second)
            return super().values()
    budget.gc = Proxy()
    budget._python_pool_types = lambda kind: CustomKinds(old_inventory(kind))
    try:
        assert not budget._has_python_pool_referrers(Pool)
        assert log == ['lookup_first','expand_starargs','call_first'], log
        records.append({'case':'custom_getter_before_star','order':log})
    finally:
        budget.gc, budget._python_pool_types = old_gc, old_inventory
    def change_alias(): budget.gc = types.SimpleNamespace(get_referrers=replacement)
    state.update(active=True,action=change_alias,audits=0)
    try:
        assert not budget._has_python_pool_referrers(Pool)
        assert state['audits'] == 1 and not wrong
        assert native.last_scan()['route'] == 'unfiltered_after_callback'
        records.append({'case':'after_audit_alias','route':native.last_scan()['route']})
    finally:
        budget.gc = old_gc
        state['active'] = False
    before = native.counters()['fallback_calls']
    budget.issubclass = lambda *args: True
    try:
        assert budget._has_python_pool_referrers(Pool)
        assert native.counters()['fallback_calls'] == before + 1
        records.append({'case':'effective_global_issubclass'})
    finally:
        del budget.issubclass
    class AuditFailure(RuntimeError): pass
    def fail_audit(): raise AuditFailure('intentional audit failure')
    before = native.counters()
    state.update(active=True,action=fail_audit,audits=0)
    try:
        try: budget._has_python_pool_referrers(Pool)
        except AuditFailure: pass
        else: raise AssertionError('audit exception swallowed')
        after = native.counters()
        assert state['audits'] == 1 and after['audit_errors'] == before['audit_errors'] + 1
        assert after['fallback_calls'] == before['fallback_calls']
        records.append({'case':'runtime_audit_exception_no_retry'})
    finally:
        state['active'] = False
finally:
    if enabled: gc.enable()
assert not torch.cuda.is_initialized()
assert adapter.build_info() == build
result = {'status':'passed','records':records,'build':build,'runtime':adapter.runtime_info(),
          'cuda_initialized':False,'cpu_affinity':sorted(os.sched_getaffinity(0)),
          'boundary':'Literal staged helper and integrated loader; synthetic Python pool classes, no actual CUDA pools or GPU work. Bound tests temporarily disable automatic GC; depth/generation checks retain normal GC.'}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'passed','cases':len(records),'fingerprint':build['fingerprint']}))
"""
    output = tmp_path / "literal_helper_receipt.json"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", XDG_CACHE_HOME=str(tmp_path / "cache"))
    result = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    (tmp_path / "child.stdout").write_text(result.stdout)
    (tmp_path / "child.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "passed" and len(receipt["records"]) == 15
    assert not receipt["cuda_initialized"]
