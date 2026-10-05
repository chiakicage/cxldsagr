"""CPU predicate semantics, native setup failures, and verified cache publication."""

import gc
import json
import os
import shutil
import subprocess
import sys
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from serving import token_validation
from serving.persistent import PersistentGRRunner
from serving.tests.test_persistent import Backend, SharedBackend, request


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if token_validation._unsupported_reason() is not None:
        pytest.skip("native predicate requires CPython with GIL on Linux")
    compiler = os.environ.get("CXX") or "c++"
    if shutil.which(compiler.split()[0]) is None:
        pytest.skip("native predicate requires a host C++ compiler")
    cache = tmp_path_factory.mktemp("token-validation")
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(cache)}):
        return token_validation._load_native()


class IntSubclass(int):
    pass


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], False),
        ([0], True),
        ([0, 17, 2**63 - 1, 2**63, 2**4096], True),
        ([-1, 0], False),
        ([-(2**63), 0], False),
        ([-(2**63) - 1, 0], False),
        ([-(2**4096), 0], False),
        ([True, 0], False),
        ([0.0, 0], False),
        ([IntSubclass(1), 0], False),
    ],
)
def test_native_and_reference_strict_integers(native, values, expected):
    predicate, _ = native
    assert predicate(values) is token_validation.reference(values) is expected


def test_predicates_reject_foreign_tokens_without_coercion_or_comparison(native):
    class ForeignType(type):
        def __hash__(cls):
            raise AssertionError("must not hash token types")

        def __eq__(cls, other):
            raise AssertionError("must compare types by identity")

    class Foreign(metaclass=ForeignType):
        def __index__(self):
            raise AssertionError("must not coerce tokens")

        def __lt__(self, other):
            raise AssertionError("must check types before ordering")

    for predicate in (native[0], token_validation.reference):
        assert not predicate([-(2**4096), Foreign(), 1])
        assert not predicate([Foreign(), -1, 1])
        for value in ((1, 2), type("ListSubclass", (list,), {})([1, 2])):
            with pytest.raises(TypeError, match="expected an exact list snapshot"):
                predicate(value)


@pytest.mark.parametrize("use_native", [False, True])
def test_invalid_snapshot_lives_in_original_python_traceback(native, monkeypatch, use_native):
    predicate = native[0] if use_native else token_validation.reference
    monkeypatch.setattr(token_validation, "prepare", lambda: predicate)
    monkeypatch.setattr(
        token_validation,
        "runtime_info",
        lambda: {"backend": "cpython_native" if use_native else "python_reference"},
    )

    class Token:
        pass

    token = Token()
    retained = weakref.ref(token)
    source = [token, 1, 2]
    del token
    with PersistentGRRunner(
        Backend(), hbm_budget_bytes=64, dram_budget_bytes=64, native_token_validation=use_native
    ) as runner:
        try:
            runner._validate({"user_id": 0, "input_ids": source, "stable_prefix_tokens": 1})
        except ValueError as error:
            saved = error
        else:
            pytest.fail("invalid token accepted")
        source.clear()
        gc.collect()
        assert retained() is not None
        frame = saved.__traceback__
        while frame.tb_next is not None:
            frame = frame.tb_next
        assert frame.tb_frame.f_code is runner._validate.__func__.__code__
        assert type(frame.tb_frame.f_locals["ids"]) is list
        assert frame.tb_frame.f_locals["ids"][0] is retained()
        traceback.clear_frames(saved.__traceback__)
        del saved, frame
        gc.collect()
        assert retained() is None


def test_runner_prepares_before_ownership_and_never_loads_during_request(monkeypatch, native):
    backend = SharedBackend()
    calls = []

    def prepare():
        assert backend.owner is None and backend.allocations == 0
        calls.append("prepare")
        return native[0]

    monkeypatch.setattr(token_validation, "prepare", prepare)
    identity = {"backend": "cpython_native", "fallback_reason": None}
    monkeypatch.setattr(token_validation, "runtime_info", lambda: identity.copy())
    with PersistentGRRunner(
        backend, hbm_budget_bytes=80, dram_budget_bytes=64, native_token_validation=True
    ) as runner:
        assert runner.token_validation_identity == {**identity, "requested_native": True}
        assert runner._token_ids_valid is native[0]

        def forbidden():
            raise AssertionError("loader must stay outside measured requests")

        monkeypatch.setattr(token_validation, "prepare", forbidden)
        monkeypatch.setattr(token_validation, "runtime_info", forbidden)
        assert runner.execute(request()).hidden.tolist() == [[6], [10]]
        assert runner.execute(request(candidate=(5, 6))).hidden.tolist() == [[8], [14]]
    assert calls == ["prepare"]


def test_default_runner_keeps_original_method_and_never_prepares(monkeypatch):
    def forbidden():
        raise AssertionError("default runner must not prepare or inspect native runtime")

    monkeypatch.setattr(token_validation, "prepare", forbidden)
    monkeypatch.setattr(token_validation, "runtime_info", forbidden)
    with PersistentGRRunner(Backend(), hbm_budget_bytes=64, dram_budget_bytes=64) as runner:
        assert runner._validate.__func__ is PersistentGRRunner._validate
        assert "_validate" not in vars(runner)
        assert runner.execute(request()).hidden.tolist() == [[6], [10]]
        assert runner.token_validation_identity == {
            "backend": "python_reference",
            "fallback_reason": None,
            "requested_native": False,
        }


@pytest.mark.parametrize(
    "invalid,exception,message",
    [
        ({"user_id": True}, TypeError, "user_id"),
        ({"input_ids": []}, ValueError, "nonempty"),
        ({"input_ids": [True, 2, 3]}, ValueError, "nonnegative integers"),
        ({"input_ids": [IntSubclass(1), 2, 3]}, ValueError, "nonnegative integers"),
        ({"input_ids": [1, -2, 3]}, ValueError, "nonnegative integers"),
        ({"stable_prefix_tokens": 0}, ValueError, "stable_prefix_tokens"),
        ({"input_ids": [1] * 101}, ValueError, "context limit"),
    ],
)
def test_shared_validation_wrapper_preserves_provider_errors(
    native, monkeypatch, invalid, exception, message
):
    monkeypatch.setattr(token_validation, "prepare", lambda: native[0])
    monkeypatch.setattr(token_validation, "runtime_info", lambda: native[1])
    for use_native in (False, True):
        backend = Backend()
        with PersistentGRRunner(
            backend,
            hbm_budget_bytes=64,
            dram_budget_bytes=64,
            native_token_validation=use_native,
        ) as runner:
            with pytest.raises(exception, match=message):
                runner.execute({**request(), **invalid})
            assert backend.built == 0 and len(runner.pool) == 0


@pytest.mark.parametrize(
    "failure",
    [
        ImportError("cannot import native predicate"),
        FileNotFoundError("compiler not found"),
        RuntimeError("compiler failed"),
        ValueError("unsupported compiler flags"),
        subprocess.TimeoutExpired("compiler", 60),
    ],
)
def test_native_setup_failure_propagates_once_before_runner_ownership(monkeypatch, failure):
    monkeypatch.setattr(token_validation, "_predicate", None)
    monkeypatch.setattr(token_validation, "_runtime", None)
    monkeypatch.setattr(token_validation, "_unsupported_reason", lambda: None)
    backend = SharedBackend()
    attempts = []

    def fail():
        assert backend.owner is None and backend.allocations == 0
        attempts.append(True)
        raise failure

    monkeypatch.setattr(token_validation, "_load_native", fail)
    with pytest.raises(type(failure)) as caught:
        PersistentGRRunner(
            backend, hbm_budget_bytes=80, dram_budget_bytes=64, native_token_validation=True
        )
    assert caught.value is failure
    assert attempts == [True]
    assert backend.owner is None and backend.allocations == 0
    assert token_validation._predicate is None and token_validation.runtime_info() is None


def test_missing_compiler_is_an_error_for_explicit_native(monkeypatch):
    monkeypatch.setattr(token_validation, "_predicate", None)
    monkeypatch.setattr(token_validation, "_runtime", None)
    monkeypatch.setattr(token_validation, "_unsupported_reason", lambda: None)
    monkeypatch.setenv("CXX", "/missing/token-validation-compiler")
    with pytest.raises(FileNotFoundError, match="compiler not found"):
        token_validation.prepare()
    assert token_validation._predicate is None and token_validation.runtime_info() is None


def test_successful_native_setup_is_reused_and_identity_is_a_defensive_copy(monkeypatch, native):
    monkeypatch.setattr(token_validation, "_predicate", None)
    monkeypatch.setattr(token_validation, "_runtime", None)
    monkeypatch.setattr(token_validation, "_unsupported_reason", lambda: None)
    calls = []

    def load():
        calls.append(True)
        return native

    monkeypatch.setattr(token_validation, "_load_native", load)
    assert token_validation.prepare() is native[0]
    identity = token_validation.runtime_info()
    identity["build_identity"]["schema_version"] = -1
    assert token_validation.runtime_info() == native[1]
    assert token_validation.prepare() is native[0]
    assert calls == [True]


@pytest.mark.parametrize("compiler", ["ccache g++", "g++ -wrapper,untracked-wrapper"])
def test_untracked_compiler_launcher_or_flags_raise(monkeypatch, compiler):
    monkeypatch.setattr(token_validation, "_predicate", None)
    monkeypatch.setattr(token_validation, "_runtime", None)
    monkeypatch.setenv("CXX", compiler)
    monkeypatch.setattr(token_validation, "_unsupported_reason", lambda: None)
    monkeypatch.setattr(shutil, "which", lambda name, **kwargs: f"/usr/bin/{name}")
    with pytest.raises(ValueError, match="direct GNU/Clang"):
        token_validation.prepare()
    assert token_validation._predicate is None and token_validation.runtime_info() is None


@pytest.mark.parametrize("mode", ["non_cpython", "free_threaded"])
def test_unsupported_runtime_raises_without_loading(monkeypatch, mode):
    monkeypatch.setattr(token_validation, "_predicate", None)
    monkeypatch.setattr(token_validation, "_runtime", None)
    if mode == "non_cpython":
        monkeypatch.setattr(sys, "implementation", SimpleNamespace(name="other"))
    else:
        monkeypatch.setattr(token_validation.sysconfig, "get_config_var", lambda name: 1)

    def forbidden():
        raise AssertionError("unsupported runtime must not compile")

    monkeypatch.setattr(token_validation, "_load_native", forbidden)
    with pytest.raises(NotImplementedError, match="CPython"):
        token_validation.prepare()
    assert token_validation._predicate is None and token_validation.runtime_info() is None


def test_import_and_reference_need_neither_compiler_nor_torch():
    code = """
import pickle, sys
from serving import token_validation as t
assert 'torch' not in sys.modules
assert t.runtime_info() is None
assert pickle.loads(pickle.dumps(t.reference))([1, 2**512])
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "CXX": "/missing/compiler"},
        check=True,
        timeout=30,
    )


def test_actual_local_header_content_changes_scoped_fingerprint(tmp_path, monkeypatch, native):
    root = Path(token_validation.__file__).resolve().parent
    (tmp_path / "csrc").mkdir()
    loader = tmp_path / "token_validation.py"
    loader.write_bytes((root / "token_validation.py").read_bytes())
    source = tmp_path / "csrc" / "token_validation.cpp"
    source.write_text('#include "extra.h"\n' + (root / "csrc/token_validation.cpp").read_text())
    header = tmp_path / "csrc" / "extra.h"
    header.write_text("#define TEST_LOCAL_HEADER 1\n")
    monkeypatch.setattr(token_validation, "__file__", str(loader))
    before, _, _ = token_validation._build_identity()
    header.write_text("#define TEST_LOCAL_HEADER 2\n")
    after, _, _ = token_validation._build_identity()
    assert str(header) in before["source_and_dependency_sha256"]
    assert token_validation._fingerprint(before) != token_validation._fingerprint(after)
    assert not any("operators/" in path for path in before["source_and_dependency_sha256"])


def test_failed_compile_preserves_error_and_unpublished_artifacts(tmp_path, monkeypatch, native):
    monkeypatch.setattr(token_validation, "_predicate", None)
    monkeypatch.setattr(token_validation, "_runtime", None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    run = token_validation._run
    failure = RuntimeError("test compiler failure")

    def fail_compile(command, environment):
        if "-o" in command:
            Path(command[-1]).write_bytes(b"partial binary")
            raise failure
        return run(command, environment)

    monkeypatch.setattr(token_validation, "_run", fail_compile)
    with pytest.raises(RuntimeError) as caught:
        token_validation.prepare()
    assert caught.value is failure
    assert token_validation._predicate is None and token_validation.runtime_info() is None
    cache = tmp_path / "cxldsagr" / "token-validation"
    staged = [path for path in cache.iterdir() if path.is_dir()]
    assert len(staged) == 1 and staged[0].name.startswith(".")
    assert next(staged[0].iterdir()).read_bytes() == b"partial binary"
    assert str(staged[0]) in failure.__notes__[0]
    assert not (staged[0] / "manifest.json").exists()


def test_concurrent_process_cache_reuse_rejects_and_preserves_corruption(tmp_path, native):
    code = """
import json
from serving import token_validation as t
assert t.prepare()([0, 2**512])
print(json.dumps(t.runtime_info()))
"""
    environment = {**os.environ, "XDG_CACHE_HOME": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""}

    def initialize():
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        info = json.loads(result.stdout)
        assert info["backend"] == "cpython_native", info
        return info

    with ThreadPoolExecutor(max_workers=3) as executor:
        identities = list(executor.map(lambda _: initialize(), range(3)))
    assert identities[0] == identities[1] == identities[2]
    binary = Path(identities[0]["loaded_binary_path"])
    stamp = binary.stat().st_mtime_ns
    assert initialize() == identities[0]
    assert binary.stat().st_mtime_ns == stamp
    assert token_validation._digest(binary) == identities[0]["loaded_binary_sha256"]
    binary.write_bytes(b"corrupted cached binary")  # No child process still maps it.
    corrupted_stamp = binary.stat().st_mtime_ns
    with pytest.raises(subprocess.CalledProcessError) as caught:
        initialize()
    assert "cached token validation binary differs" in caught.value.stderr
    assert binary.read_bytes() == b"corrupted cached binary"
    assert binary.stat().st_mtime_ns == corrupted_stamp
    cache = tmp_path / "cxldsagr" / "token-validation"
    assert len([path for path in cache.iterdir() if path.is_dir()]) == 1


def test_only_missing_cache_directory_is_unpublished(tmp_path):
    identity = {"extension_suffix": ".so"}
    assert token_validation._verified_manifest(tmp_path / "new", identity) is None
    with pytest.raises(FileNotFoundError):
        token_validation._verified_manifest(tmp_path, identity)


@pytest.mark.parametrize("corruption", ["json", "identity", "digest", "binary", "manifest_shape"])
def test_existing_invalid_cache_is_never_treated_as_unpublished(tmp_path, corruption):
    identity = {"extension_suffix": ".so"}
    binary = tmp_path / (token_validation._NAME + ".so")
    binary.write_bytes(b"cached binary")
    manifest = {"build_identity": identity, "binary_sha256": token_validation._digest(binary)}
    path = tmp_path / "manifest.json"
    expected_error = RuntimeError
    if corruption == "identity":
        manifest["build_identity"] = {}
    elif corruption == "digest":
        manifest["binary_sha256"] = "wrong"
    elif corruption == "binary":
        binary.unlink()
        expected_error = FileNotFoundError
    elif corruption == "manifest_shape":
        manifest = []
        expected_error = TypeError
    path.write_text("invalid json" if corruption == "json" else json.dumps(manifest))
    if corruption == "json":
        expected_error = json.JSONDecodeError
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    with pytest.raises(expected_error):
        token_validation._verified_manifest(tmp_path, identity)
    assert {item.name: item.read_bytes() for item in tmp_path.iterdir()} == before
