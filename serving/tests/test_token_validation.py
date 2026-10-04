"""CPU predicate semantics, setup fallback, and native cache publication."""

import ast
import gc
import inspect
import json
import os
import pickle
import shutil
import subprocess
import sys
import textwrap
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
        assert runner._validate.__func__ is PersistentGRRunner._validate_native

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


def test_native_wrapper_differs_only_in_token_predicate():
    original = ast.parse(textwrap.dedent(inspect.getsource(PersistentGRRunner._validate)))
    native = ast.parse(textwrap.dedent(inspect.getsource(PersistentGRRunner._validate_native)))
    native.body[0].name = original.body[0].name
    original_condition = ast.parse(
        "not ids or not all(map(is_, map(type, ids), repeat(int))) or min(ids) < 0", mode="eval"
    ).body
    native_condition = ast.parse("not self._token_ids_valid(ids)", mode="eval").body
    changed = [
        node
        for node in ast.walk(native)
        if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(native_condition)
    ]
    assert len(changed) == 1
    changed[0].test = original_condition
    assert ast.dump(native) == ast.dump(original)


def test_setup_failure_selects_pickleable_reference_once(monkeypatch):
    monkeypatch.setattr(token_validation, "_predicate", None)
    monkeypatch.setattr(token_validation, "_runtime", None)
    monkeypatch.setenv("CXX", "/missing/token-validation-compiler")
    predicate = token_validation.prepare()
    info = token_validation.runtime_info()
    assert predicate is token_validation.reference
    assert info["backend"] == "python_reference"
    assert "compiler not found" in info["fallback_reason"]
    assert pickle.loads(pickle.dumps(predicate)) is predicate
    info["fallback_reason"] = "changed copy"
    assert token_validation.runtime_info()["fallback_reason"] != "changed copy"

    def forbidden():
        raise AssertionError("setup selection must not be retried")

    monkeypatch.setattr(token_validation, "_load_native", forbidden)
    assert token_validation.prepare() is predicate
    assert predicate([0, 2**512])
    with PersistentGRRunner(
        Backend(), hbm_budget_bytes=64, dram_budget_bytes=64, native_token_validation=True
    ) as runner:
        assert runner._validate.__func__ is PersistentGRRunner._validate
        assert runner.token_validation_identity["requested_native"] is True
        assert "compiler not found" in runner.token_validation_identity["fallback_reason"]
        assert runner.execute(request()).hidden.tolist() == [[6], [10]]


@pytest.mark.parametrize("compiler", ["ccache g++", "g++ -wrapper,untracked-wrapper"])
def test_untracked_compiler_launcher_or_flags_choose_setup_fallback(monkeypatch, compiler):
    monkeypatch.setattr(token_validation, "_predicate", None)
    monkeypatch.setattr(token_validation, "_runtime", None)
    monkeypatch.setenv("CXX", compiler)
    monkeypatch.setattr(shutil, "which", lambda name, **kwargs: f"/usr/bin/{name}")
    assert token_validation.prepare() is token_validation.reference
    assert "direct GNU/Clang" in token_validation.runtime_info()["fallback_reason"]


@pytest.mark.parametrize("mode", ["non_cpython", "free_threaded"])
def test_unsupported_runtime_selects_reference_without_loading(monkeypatch, mode):
    monkeypatch.setattr(token_validation, "_predicate", None)
    monkeypatch.setattr(token_validation, "_runtime", None)
    if mode == "non_cpython":
        monkeypatch.setattr(sys, "implementation", SimpleNamespace(name="other"))
    else:
        monkeypatch.setattr(token_validation.sysconfig, "get_config_var", lambda name: 1)

    def forbidden():
        raise AssertionError("unsupported runtime must not compile")

    monkeypatch.setattr(token_validation, "_load_native", forbidden)
    assert token_validation.prepare() is token_validation.reference
    assert token_validation.runtime_info()["fallback_reason"]


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


def test_failed_compile_leaves_no_published_or_temporary_binary(tmp_path, monkeypatch, native):
    monkeypatch.setattr(token_validation, "_predicate", None)
    monkeypatch.setattr(token_validation, "_runtime", None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    run = token_validation._run

    def fail_compile(command, environment):
        if "-o" in command:
            Path(command[-1]).write_bytes(b"partial binary")
            raise RuntimeError("test compiler failure")
        return run(command, environment)

    monkeypatch.setattr(token_validation, "_run", fail_compile)
    assert token_validation.prepare() is token_validation.reference
    assert "test compiler failure" in token_validation.runtime_info()["fallback_reason"]
    cache = tmp_path / "cxldsagr" / "token-validation"
    assert all(path.suffix == ".lock" for path in cache.iterdir())


def test_concurrent_process_cache_reuse_and_corruption_repair(tmp_path, native):
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
    repaired = initialize()
    assert repaired["fingerprint"] == identities[0]["fingerprint"]
    assert token_validation._digest(binary) == repaired["loaded_binary_sha256"]
    cache = tmp_path / "cxldsagr" / "token-validation"
    assert len([path for path in cache.iterdir() if path.is_dir()]) == 1
