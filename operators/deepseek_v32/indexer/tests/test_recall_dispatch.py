"""Host bridge ABI, eager preparation, and original native failure boundaries."""

import copy
import hashlib
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from operators.deepseek_v32.indexer import recall_dispatch

DIRECTORY = Path(__file__).resolve().parent
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_FAILED_POOLS = []


@pytest.fixture(scope="module")
def build_identity():
    pytest.importorskip("tvm_ffi")
    return recall_dispatch.build_info()


def _test_module(source, *, includes=()):
    import tvm_ffi.cpp

    files = [source, *includes]
    identity = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()
    return tvm_ffi.cpp.load(
        name=f"cxldsagr_{source.stem}_{identity[:16]}",
        sources=[str(source)],
        extra_cflags=["-O2", "-std=c++17"],
        extra_include_paths=[str(recall_dispatch._cuda_home() / "include")],
        extra_ldflags=[f"-L{recall_dispatch._cuda_home() / 'lib64'}", "-lcudart"],
    )


@pytest.fixture(scope="module")
def boundary_module(build_identity):
    return _test_module(DIRECTORY / "recall_boundary_probe.cpp", includes=[recall_dispatch._SOURCE])


def test_reference_import_does_not_load_tvm_or_initialize_cuda():
    script = """
import sys
from operators.deepseek_v32.indexer import recall_dispatch
assert not any(name == 'tvm_ffi' or name.startswith('tvm_ffi.') for name in sys.modules)
assert 'torch' not in sys.modules
assert recall_dispatch.runtime_info() is None
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_build_identity_covers_actual_tvm_cuda_and_local_source(build_identity):
    sources = build_identity["identity"]["source_and_dependency_sha256"]
    assert str(recall_dispatch._SOURCE.resolve()) in sources
    assert str(Path(recall_dispatch.__file__).resolve()) in sources
    assert any(path.endswith("/tvm/ffi/function.h") for path in sources)
    assert any(path.endswith("/dlpack/dlpack.h") for path in sources)
    assert any(path.endswith("/cuda_runtime_api.h") for path in sources)
    assert not any("operators/nosa" in path for path in sources)
    assert build_identity["identity"]["extra_cflags"] == recall_dispatch._CFLAGS


def test_changed_local_include_changes_identity(tmp_path, monkeypatch, build_identity):
    header = tmp_path / "local.h"
    source = tmp_path / "local.cpp"
    header.write_text("constexpr int local_value = 1;\n")
    source.write_text('#include "local.h"\n')
    monkeypatch.setattr(recall_dispatch, "_SOURCE", source)
    first = recall_dispatch.build_info()
    header.write_text("constexpr int local_value = 2;\n")
    second = recall_dispatch.build_info()
    assert first["fingerprint"] != second["fingerprint"]
    assert str(header.resolve()) in first["identity"]["source_and_dependency_sha256"]


def test_runtime_identity_retains_loaded_identity_and_returns_copy(monkeypatch):
    import tvm_ffi.cpp

    module = object()
    info = {
        "fingerprint": "a" * 64,
        "identity": {"extra_cflags": [], "extra_include_paths": [], "extra_ldflags": []},
    }
    monkeypatch.setattr(recall_dispatch, "_RUNTIME_INFO", None)
    monkeypatch.setattr(recall_dispatch, "build_info", lambda: copy.deepcopy(info))
    monkeypatch.setattr(tvm_ffi.cpp, "load", lambda **kwargs: module)
    assert recall_dispatch._module.__wrapped__() is module
    loaded = recall_dispatch.runtime_info()
    loaded["identity"]["extra_cflags"].append("mutated")
    assert recall_dispatch.runtime_info() == info
    monkeypatch.setattr(recall_dispatch, "build_info", lambda: {"fingerprint": "b" * 64})
    assert recall_dispatch.runtime_info() == info


def test_changed_build_inputs_do_not_publish_runtime_identity(monkeypatch):
    import tvm_ffi.cpp

    original = {"fingerprint": "previous"}
    sequence = iter(
        [
            {
                "fingerprint": "a" * 64,
                "identity": {"extra_cflags": [], "extra_include_paths": [], "extra_ldflags": []},
            },
            {"fingerprint": "b" * 64},
        ]
    )
    monkeypatch.setattr(recall_dispatch, "_RUNTIME_INFO", original)
    monkeypatch.setattr(recall_dispatch, "build_info", lambda: next(sequence))
    monkeypatch.setattr(tvm_ffi.cpp, "load", lambda **kwargs: object())
    with pytest.raises(RuntimeError, match="changed during native preparation"):
        recall_dispatch._module.__wrapped__()
    assert recall_dispatch.runtime_info() == original


@pytest.mark.parametrize("failure_at", [None, "module", "echo", "common", False, True])
def test_initializer_publishes_both_bindings_atomically(monkeypatch, failure_at):
    from operators.common import kv_transfer
    from operators.deepseek_v32.indexer import echo

    calls = []
    failure = KeyboardInterrupt("native preparation failure")
    sentinel = {False: object(), True: object()}

    def record(stage, result):
        calls.append(stage)
        if type(stage) is type(failure_at) and stage == failure_at:
            raise failure
        return result

    module = SimpleNamespace(make_bridge=lambda e, c, free: record(free, sentinel[free]))
    monkeypatch.setattr(recall_dispatch, "_BINDINGS", {})
    monkeypatch.setattr(recall_dispatch, "_module", lambda: record("module", module))
    monkeypatch.setattr(echo, "_module", lambda: record("echo", object()))
    monkeypatch.setattr(kv_transfer, "_module", lambda: record("common", object()))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    device = torch.device("cuda:0")
    if failure_at is None:
        recall_dispatch.initialize(device)
        assert recall_dispatch._BINDINGS == {device: sentinel}
        recall_dispatch.initialize(device)
        assert calls == ["module", "echo", "common", False, True]
    else:
        with pytest.raises(KeyboardInterrupt) as caught:
            recall_dispatch.initialize(device)
        assert caught.value is failure
        assert recall_dispatch._BINDINGS == {}
        assert calls[-1] == failure_at


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initializer_requires_explicit_cuda_device(device):
    with pytest.raises(ValueError, match="explicit CUDA device"):
        recall_dispatch.initialize(device)


def test_request_without_preparation_never_initializes(monkeypatch):
    monkeypatch.setattr(recall_dispatch, "_BINDINGS", {})

    def unexpected(*args):
        raise AssertionError("request attempted initialization")

    monkeypatch.setattr(recall_dispatch, "initialize", unexpected)
    monkeypatch.setattr(recall_dispatch, "_module", unexpected)
    with pytest.raises(RuntimeError, match="not initialized before pool use"):
        recall_dispatch.prepared_call(torch.device("cuda:0"), False)


@pytest.mark.parametrize("free_only", [False, True])
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (0, [123, 1, 0, 0]),
        (1, [1, 0, 1, 1]),
        (2, [1, 1, 1, 0]),
        (3, [12, 1, 1, 1]),
        (4, [12, 1, 1, 0]),
        (5, [123, 1, 1, 1]),
        (6, [14, 1, 1, 0]),
    ],
)
def test_native_metadata_order_and_original_error_identity(
    boundary_module, free_only, mode, expected
):
    assert list(boundary_module.probe(boundary_module, mode, free_only)) == expected


def test_factory_rejects_python_callbacks_and_missing_native_symbols(
    boundary_module, build_identity
):
    with pytest.raises(TypeError):
        boundary_module.make_bridge(lambda: None, boundary_module, False)
    # The production bridge DSO exports its factory but none of the stage names.
    with pytest.raises(ValueError, match="lacks a required symbol"):
        boundary_module.make_bridge(recall_dispatch._module(), boundary_module, False)


@pytest.mark.parametrize("free_only", [False, True])
def test_native_forwarder_retains_original_error_before_cuda(boundary_module, free_only):
    shim = _test_module(DIRECTORY / "recall_faults.cpp")
    shim.configure(boundary_module, boundary_module, 1)
    bridge = recall_dispatch._module().make_bridge(shim, shim, free_only)
    assert shim.cpu_error_identity(bridge)
    assert shim.trace() == 1


@contextmanager
def _closing_pool(pool):
    error = None
    try:
        yield pool
    except BaseException as caught:  # noqa: BLE001 - preserve primary and cleanup failures
        error = caught
    try:
        pool.close()
    except BaseException as cleanup:
        _FAILED_POOLS.append(pool)
        if error is not None:
            raise BaseExceptionGroup("recall test and cleanup failed", [error, cleanup]) from None
        raise
    if error is not None:
        raise error


@pytest.mark.parametrize("session_count", [1, 2])
@pytest.mark.parametrize("stage", ["wait", "allocate", "gather", "publish", "map"])
@requires_cuda
def test_production_native_stage_failure_preserves_full_state(monkeypatch, session_count, stage):
    from cache.sparse_token_pool import SharedSparseTokenPool
    from cache.tests.test_sparse_recall_metadata import _append, _capture, _validate_recall
    from operators.common import kv_transfer
    from operators.deepseek_v32.indexer import cache_ops

    shim = _test_module(DIRECTORY / "recall_faults.cpp")
    observed, transfer, allocator_calls = [], [], []

    class ObservedProvider:
        def __getattr__(self, name):
            return getattr(cache_ops, name)

        def sparse_selection_complete(self, *args, **kwargs):
            allocator_calls.append(kwargs["free_only"])
            try:
                return cache_ops.sparse_selection_complete(*args, **kwargs)
            except BaseException as error:
                observed.append(error)
                transfer.extend(args[8:10])
                raise

    pool = SharedSparseTokenPool(
        128, 7, 2, 8, device="cuda:0", dtype=torch.float32, metadata_ops=ObservedProvider()
    )
    with _closing_pool(pool):
        for layer in pool.layers:
            layer.records.zero_()
        caches = [pool.allocate_session(8).layer(0) for _ in range(session_count)]
        for owner, cache in enumerate(caches):
            _append(cache, torch.arange(56, device="cuda").reshape(8, 7).float() + owner * 100)
        cache = caches[0]
        if session_count == 1:
            with cache.operation():
                pool.release_ids(0, cache.session.global_ids())
        pool.drain()
        shim.configure(
            cache_ops._module(),
            kv_transfer._module(),
            {"allocate": 1, "gather": 2, "publish": 3}.get(stage, 0),
        )
        bridges = {
            free: recall_dispatch._module().make_bridge(shim, shim, free) for free in (False, True)
        }
        monkeypatch.setattr(recall_dispatch, "prepared_call", lambda device, free: bridges[free])
        failure = RuntimeError(f"injected Python {stage}")

        def fail_python(*args, **kwargs):
            observed.append(failure)
            raise failure

        ids = torch.tensor([[-1, 0, 0, 3, 7]], device="cuda", dtype=torch.int32)
        before = _capture(cache)
        initial_clock = pool.layers[0].clock
        with monkeypatch.context() as patch:
            if stage == "wait":
                patch.setattr(pool, "wait_host", fail_python)
            elif stage == "map":
                patch.setattr(cache_ops, "sparse_selection_map", fail_python)
            with pytest.raises(RuntimeError) as caught:
                cache._ensure_from_topk(ids)
        assert len(observed) == 1 and caught.value is observed[0]
        assert allocator_calls == ([] if stage == "wait" else [session_count == 1])
        assert (
            shim.trace()
            == {"wait": 0, "allocate": 1, "gather": 12, "publish": 123, "map": 123}[stage]
        )
        assert pool.layers[0].clock == initial_clock + (2 if stage == "map" else 1)
        assert pool.layers[0].append_owner is None
        assert pool._active is None and not pool.poisoned
        _validate_recall(
            cache,
            ids,
            None,
            before,
            stage={"allocate": "sort", "gather": "copy"}.get(stage, stage),
            transfer=transfer,
        )


@pytest.mark.parametrize("session_count", [1, 2])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@requires_cuda
def test_production_recall_accepts_generic_offset_records(session_count, dtype):
    from cache.sparse_token_pool import SharedSparseTokenPool
    from cache.tests.test_sparse_recall_metadata import _append, _capture, _validate_recall
    from operators.deepseek_v32.indexer import cache_ops

    pool = SharedSparseTokenPool(128, 7, 2, 8, device="cuda:0", dtype=dtype, metadata_ops=cache_ops)
    with _closing_pool(pool):
        for layer in pool.layers:
            layer.host = torch.empty(129, 7, dtype=dtype, pin_memory=True)[1:]
            layer.records = torch.zeros(10, 7, dtype=dtype, device="cuda")[1:]
            assert layer.host.storage_offset() and layer.records.storage_offset()
        caches = [pool.allocate_session(8).layer(0) for _ in range(session_count)]
        for owner, cache in enumerate(caches):
            data = (torch.arange(56, device="cuda").reshape(8, 7) + owner * 100).to(dtype)
            _append(cache, data)
        cache = caches[0]
        if session_count == 1:
            with cache.operation():
                pool.release_ids(0, cache.session.global_ids())
        pool.drain()
        ids = torch.tensor([[-1, 0, 0, 3, 7]], device="cuda", dtype=torch.int32)
        for _ in range(2):
            before = _capture(cache)
            output = cache._ensure_from_topk(ids)
            _validate_recall(cache, ids, output, before)
