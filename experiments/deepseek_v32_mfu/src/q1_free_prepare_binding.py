"""Reversible process-local binding of reviewed preparation mirrors; never edits production."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch

import cache.sparse_token_cache as sparse_cache
from evaluation.validation import identity_digest
from experiments.deepseek_v32_mfu.src.q1_prepare_baseline import digest, require, source_identity
from operators.deepseek_v32.indexer import _native_cache, cache_ops, echo, official_prefetch

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / "experiments/deepseek_v32_mfu/output/data/q1_free_prepare_integration_20261008_02"


def load_mirror(path, suffix):
    name = "_private_free_prepare_" + suffix
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone(function, namespace):
    result = types.FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    return result


class Binding:
    def __init__(self, directory=DEFAULT):
        self.directory = Path(directory).resolve()
        self.source = self.directory / "source"
        self.before = self.source_identity()
        for name, expected in self.before["baseline"].items():
            require(digest(ROOT / name) == expected, "Production base changed: " + name)
        self.echo = load_mirror(self.source / "operators/deepseek_v32/indexer/echo.py", "echo")
        self.cache_ops = load_mirror(
            self.source / "operators/deepseek_v32/indexer/cache_ops.py", "cache_ops"
        )
        self.official = load_mirror(
            self.source / "operators/deepseek_v32/indexer/official_prefetch.py", "official"
        )
        self.cache = load_mirror(self.source / "cache/sparse_token_cache.py", "cache")
        self.module = self.native = None
        self.calls = 0
        self.active = False

    def source_identity(self):
        baseline = json.loads((self.directory / "baseline_manifest.json").read_text())
        expected = json.loads((self.directory / "candidate_manifest.json").read_text())
        actual = {
            str(path.relative_to(self.source)): digest(path)
            for path in sorted(self.source.rglob("*"))
            if path.is_file()
        }
        require(actual == expected, "Integration mirror differs from its manifest")
        return {"baseline": baseline, "candidate": actual, "binder_sha256": digest(__file__)}

    def load_cuda(self):
        require(self.module is None, "Integration native already loaded")
        identity = source_identity()
        files = dict(identity["files"])
        files.update(
            {str(self.source / name): sha for name, sha in self.before["candidate"].items()}
        )
        native_identity = {"source_sha256": files}
        name = "cxldsagr_q1_free_prepare_integrated_" + identity_digest(native_identity)[:16]
        previous = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
        os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
        try:
            self.module = _native_cache.load(
                name=name,
                sources=[str(self.source / "operators/deepseek_v32/indexer/csrc/echo_indexer.cu")],
                extra_include_paths=[
                    str(ROOT / "3rdparty/cutlass/include"),
                    str(ROOT / "operators/deepseek_v32/indexer/csrc"),
                ],
                extra_cuda_cflags=echo._FLAGS,
                extra_ldflags=["-lcuda"],
                source_identity=native_identity,
            )
            self.native = _native_cache.native_info(name)
        finally:
            if previous is None:
                os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
            else:
                os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous
        require(self.source_identity() == self.before, "Mirror changed during native build")

    @contextmanager
    def installed(self):
        require(not self.active, "Candidate dispatch context cannot nest")
        self.active = True

        def call(name, device, *args):
            require(name == "echo_prepare_prefetch_free", "Unexpected private native entry")
            require(self.module is not None, "Private CUDA module was not explicitly loaded")
            self.calls += 1
            import tvm_ffi

            with torch.cuda.device(device), tvm_ffi.use_torch_stream():
                return self.module.echo_prepare_prefetch_free(*args)

        free_namespace = dict(vars(cache_ops), _call=call)
        replacements = (
            (
                echo,
                "_bounded_prepared_storage_identity",
                self.echo._bounded_prepared_storage_identity,
            ),
            (echo, "_BoundedPreparedPrefetch", self.echo._BoundedPreparedPrefetch),
            (
                cache_ops,
                "supports_free_q1_prepare",
                clone(self.cache_ops.supports_free_q1_prepare, vars(cache_ops)),
            ),
            (
                cache_ops,
                "prepare_prefetch_free",
                clone(self.cache_ops.prepare_prefetch_free, free_namespace),
            ),
            (official_prefetch, "_prepare", clone(self.official._prepare, vars(official_prefetch))),
            (
                sparse_cache.SparseTokenCache,
                "prepare_prefetch",
                clone(self.cache.SparseTokenCache.prepare_prefetch, vars(sparse_cache)),
            ),
        )
        absent = object()
        saved = []
        try:
            for owner, name, value in replacements:
                saved.append((owner, name, getattr(owner, name, absent), value))
                setattr(owner, name, value)
            yield self
        finally:
            primary = sys.exception()
            errors = []
            for owner, name, old, replacement in reversed(saved):
                try:
                    require(
                        getattr(owner, name) is replacement,
                        "Concurrent process-local dispatch mutation",
                    )
                    if old is absent:
                        delattr(owner, name)
                    else:
                        setattr(owner, name, old)
                except BaseException as error:  # noqa: BLE001 -- preserve all restoration failures
                    errors.append(error)
            self.active = False
            if errors:
                raise BaseExceptionGroup(
                    "Private dispatch restoration failed", ([primary] if primary else []) + errors
                )

    def verify(self):
        require(not self.active, "Candidate dispatch context is still installed")
        require(self.source_identity() == self.before, "Mirror changed during trial")
        for name, expected in self.before["baseline"].items():
            require(digest(ROOT / name) == expected, "Production source changed during trial")
        if self.native is not None:
            require(
                digest(self.native["artifact_path"]) == self.native["artifact_sha256"],
                "Private native changed",
            )
