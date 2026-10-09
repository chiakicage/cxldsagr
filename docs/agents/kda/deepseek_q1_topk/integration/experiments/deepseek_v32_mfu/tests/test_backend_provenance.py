import functools
import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from experiments.deepseek_v32_mfu.src import backend_provenance as provenance


def sha256(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def empty_runtime(monkeypatch):
    for name in (
        "flashinfer.jit.core",
        "flashinfer.norm.kernels.rmsnorm",
        "flashinfer.norm.kernels.fused_add_rmsnorm",
        "operators.deepseek_v32.norm._compile",
        "operators.deepseek_v32.linear.quantization",
        "operators.deepseek_v32.indexer.q1_topk_cub",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(provenance, "_mapped_libraries", set)
    monkeypatch.setattr(provenance, "collect_local_native_artifacts", lambda **kwargs: [])


@pytest.fixture
def installed_package(tmp_path, monkeypatch):
    package = tmp_path / "flashinfer"
    files = ["__init__.py", "norm.py"]
    files += [
        f"data/csrc/{name}"
        for name in (
            "norm.cu",
            "flashinfer_norm_binding.cu",
            "rope.cu",
            "flashinfer_rope_binding.cu",
            "topk.cu",
            "flashinfer_topk_binding.cu",
            "flashinfer_fast_topk_clusters_binding.cu",
            "tvm_ffi_utils.h",
        )
    ]
    files += [
        "data/include/flashinfer/norm.cuh",
        "data/cccl/cub/cub.cuh",
        "data/cccl/libcudacxx/include/cuda/atomic",
        "data/cccl/thrust/transform.h",
        "data/cutlass/include/cute/tensor.hpp",
    ]
    for name in files:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    distribution = SimpleNamespace(
        version="0.6.18",
        locate_file=lambda name: tmp_path / name,
        read_text=lambda _: None,
    )
    monkeypatch.setattr(provenance.importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(provenance.importlib.metadata, "version", lambda _: "fixture")
    return package, distribution


def test_local_flashinfer_adapters_are_in_source_identity():
    sources = provenance.source_files()
    for relative in (
        "models/deepseek_v32/nonmatrix.py",
        "evaluation/local_native.py",
        "operators/flashinfer.py",
        "operators/deepseek_v32/norm/api.py",
        "operators/deepseek_v32/norm/_compile.py",
        "operators/deepseek_v32/norm/_fingerprint.py",
        "operators/deepseek_v32/norm/_layout.py",
        "operators/deepseek_v32/norm/_plain.py",
        "operators/deepseek_v32/norm/_fused.py",
        "operators/deepseek_v32/linear/fp8.py",
        "operators/deepseek_v32/linear/quantization.py",
        "operators/deepseek_v32/linear/_quantization_kernel.py",
        "operators/deepseek_v32/linear/_quantization_identity.py",
        "operators/deepseek_v32/indexer/q1_topk_cub.py",
        "operators/deepseek_v32/indexer/csrc/q1_topk_sort.cu",
    ):
        path = provenance.ROOT / relative
        assert path in sources and path.is_file()


def test_installed_identity_detects_python_native_and_header_changes(installed_package):
    package, _ = installed_package
    before = provenance._installed_flashinfer()
    changed = {
        "python_sha256": "norm.py",
        "source_sha256": "data/csrc/topk.cu",
        "include_sha256": "data/cccl/cub/cub.cuh",
    }
    for relative in changed.values():
        (package / relative).write_bytes(b"changed implementation")
    after = provenance._installed_flashinfer()
    for field, relative in changed.items():
        assert before[field][relative] != after[field][relative]
        assert after[field][relative] == sha256(b"changed implementation")
    assert (
        before["source_sha256"]["data/csrc/rope.cu"] == after["source_sha256"]["data/csrc/rope.cu"]
    )
    assert after["installed_native_files"] == []


def test_installed_identity_rejects_changed_version_and_missing_headers(installed_package):
    package, distribution = installed_package
    distribution.version = "0.6.19"
    with pytest.raises(RuntimeError, match="revalidate the pinned 0.6.18"):
        provenance._installed_flashinfer()
    distribution.version = "0.6.18"
    (package / "data/cccl/cub/cub.cuh").unlink()
    with pytest.raises(RuntimeError, match="Missing installed FlashInfer include tree"):
        provenance._installed_flashinfer()


def test_empty_process_does_not_claim_cached_binaries(empty_runtime, tmp_path, monkeypatch):
    unrelated = tmp_path / "unrelated-cache.so"
    unrelated.write_bytes(b"unrelated artifact")
    monkeypatch.setattr(provenance, "_mapped_libraries", lambda: {str(unrelated)})
    result = provenance.collect_flashinfer_runtime_artifacts()
    assert result["native_jit"] == []
    assert result["cute_jit"] == []
    assert result["linear_quantization_triton"] is None
    assert result["q1_topk_cub_native"] is None


def test_runtime_observes_already_imported_linear_quantizer(empty_runtime, monkeypatch):
    record = {"build_info": {"kernel_policy": "fixture"}, "specializations": []}
    module = ModuleType("operators.deepseek_v32.linear.quantization")
    module.runtime_info = lambda: record
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = provenance.collect_flashinfer_runtime_artifacts()
    assert result["linear_quantization_triton"] is record


def test_runtime_binds_actual_local_libraries_and_required_mode(empty_runtime, monkeypatch):
    records = [{"name": "mapped_native_fixture", "library": {"sha256": "fixture"}}]
    requirements = []

    def collect(*, required):
        requirements.append(required)
        return records

    monkeypatch.setattr(provenance, "collect_local_native_artifacts", collect)
    for required in (False, True):
        result = provenance.collect_flashinfer_runtime_artifacts(require_local_native=required)
        assert result["local_native_jit"] is records
    assert requirements == [False, True]


def test_runtime_native_identity_requires_registered_and_mapped_library(
    empty_runtime, tmp_path, monkeypatch
):
    library = tmp_path / "rope.so"
    source = tmp_path / "rope.cu"
    ninja = tmp_path / "build.ninja"
    library.write_bytes(b"loaded binary")
    source.write_bytes(b"source code")
    ninja.write_bytes(b"compiler commands")
    specs = {
        "rope": SimpleNamespace(
            get_library_path=lambda: library,
            sources=[source],
            ninja_path=ninja,
            extra_cuda_cflags=["-arch=sm_90a"],
            extra_cflags=["-O3"],
        ),
        # The registered but never loaded module need not have a binary yet.
        "topk": SimpleNamespace(get_library_path=lambda: tmp_path / "missing.so", sources=[]),
        "unrelated": SimpleNamespace(),
    }
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.jit.core",
        SimpleNamespace(jit_spec_registry=SimpleNamespace(get_all_specs=lambda: specs)),
    )
    monkeypatch.setattr(provenance, "_mapped_libraries", lambda: {str(library)})
    result = provenance.collect_flashinfer_runtime_artifacts()
    loaded, registered = result["native_jit"]
    assert loaded["name"] == "rope" and loaded["loaded_in_this_process"]
    assert loaded["library"] == {
        "path": str(library),
        "sha256": sha256(b"loaded binary"),
        "bytes": 13,
    }
    assert loaded["sources"][0]["sha256"] == sha256(b"source code")
    assert loaded["build_metadata"]["sha256"] == sha256(b"compiler commands")
    assert loaded["cuda_flags"] == ["-arch=sm_90a"]
    assert loaded["cxx_flags"] == ["-O3"]
    assert registered["name"] == "topk" and not registered["loaded_in_this_process"]
    assert registered["library"] is None

    monkeypatch.setattr(provenance, "_mapped_libraries", lambda: {str(library) + " (deleted)"})
    with pytest.raises(RuntimeError, match="removed during execution"):
        provenance.collect_flashinfer_runtime_artifacts()


@pytest.mark.parametrize(
    "module_name",
    ("flashinfer.norm.kernels.rmsnorm", "operators.deepseek_v32.norm._compile"),
)
def test_runtime_cute_identity_reads_live_cache_without_compiling(
    empty_runtime, tmp_path, monkeypatch, module_name
):
    cubin = tmp_path / "kernel.cubin"
    cubin.write_bytes(b"device binary")
    calls = []
    bytecode = b"compiled MLIR with embedded device object"

    @functools.cache
    def compiled_kernel(dtype, width, *, contiguous):
        calls.append((dtype, width, contiguous))
        return SimpleNamespace(
            function_name="rmsnorm",
            kernel_info={"rmsnorm_kernel": None},
            ir_module=SimpleNamespace(
                operation=SimpleNamespace(write_bytecode=lambda stream: stream.write(bytecode))
            ),
            artifacts=SimpleNamespace(CUBIN=cubin),
        )

    @functools.cache
    def opaque_kernel(dtype, width):
        calls.append((dtype, width))
        return SimpleNamespace(function_name="opaque_kernel")

    compiled_kernel("float32", 512, contiguous=True)
    opaque_kernel("float32", 7168)
    module = ModuleType(module_name)
    module._get_compiled_rmsnorm_kernel = compiled_kernel
    module._get_compiled_opaque_kernel = opaque_kernel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = provenance.collect_flashinfer_runtime_artifacts()
    assert calls == [("float32", 512, True), ("float32", 7168)]
    assert result["native_jit"] == []
    opaque, compiled = result["cute_jit"]
    assert compiled["compile_key"] == ["float32", 512, {"type": "object"}, "contiguous", True]
    assert compiled["kernel_names"] == ["rmsnorm_kernel"]
    assert compiled["mlir_bytecode"] == {"sha256": sha256(bytecode), "bytes": len(bytecode)}
    assert compiled["files"] == [
        {"kind": "CUBIN", "path": str(cubin), "sha256": sha256(b"device binary"), "bytes": 13}
    ]
    assert opaque["mlir_bytecode"] is None and opaque["files"] == []
    assert "no compiled IR/file" in opaque["unavailable_reason"]


def test_typed_norm_identity_rechecks_loaded_dependency_bytes(tmp_path, monkeypatch):
    source = tmp_path / "dependency.py"
    source.write_bytes(b"loaded implementation")
    identity = {
        "fingerprint": "fixture-fingerprint",
        "sources": {"dependency": {"path": str(source), "sha256": sha256(source.read_bytes())}},
    }
    module = ModuleType("operators.deepseek_v32.norm._fingerprint")
    module.build_info = lambda: identity
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert provenance._typed_norm_identity() is identity
    source.write_bytes(b"changed implementation")
    with pytest.raises(RuntimeError, match="Typed norm source changed during execution"):
        provenance._typed_norm_identity()


def test_source_listing_does_not_import_flashinfer(monkeypatch):
    def forbidden_import(*_args, **_kwargs):
        pytest.fail("Listing build sources must not import or compile a backend")

    monkeypatch.setattr(provenance.importlib, "import_module", forbidden_import)
    assert Path(provenance.__file__).resolve() in provenance.source_files()


@pytest.mark.parametrize("configured", (None, "/configured/ptxas"))
def test_backend_identity_initializes_compiler_environment_before_observation(
    monkeypatch, configured
):
    variable = "TRITON_PTXAS_BLACKWELL_PATH"
    if configured is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, configured)
    expected = configured or "/system/ptxas"
    imports = []

    def initialize(name):
        assert name == "flashinfer.triton"
        if not imports and not provenance.os.environ.get(variable):
            monkeypatch.setenv(variable, expected)
        imports.append(name)

    def git(directory, *arguments):
        assert imports
        assert variable in provenance.os.environ
        return provenance.PINS[directory.name] if arguments == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(provenance.importlib, "import_module", initialize)
    monkeypatch.setattr(provenance, "_git", git)
    monkeypatch.setattr(
        provenance,
        "_installed",
        lambda name, _: {
            "distribution_version": {
                "deep-gemm": "2.8.1+057ca59",
                "flash-mla": "1.0.0+ba89a34",
            }[name]
        },
    )
    for name in (
        "_installed_flashinfer",
        "_recall_dispatch_identity",
        "_typed_norm_identity",
        "_q1_topk_identity",
    ):
        monkeypatch.setattr(provenance, name, dict)
    monkeypatch.setattr(
        provenance,
        "_linear_quantization_identity",
        lambda: {"compiler_environment": {variable: provenance.os.environ[variable]}},
    )

    before = provenance.collect_backend_provenance()
    assert before["linear_activation_quantization"]["compiler_environment"] == {variable: expected}
    assert provenance.collect_backend_provenance() == before
    monkeypatch.setenv(variable, "/changed/ptxas")
    after = provenance.collect_backend_provenance()
    assert after["linear_activation_quantization"]["compiler_environment"] == {
        variable: "/changed/ptxas"
    }
    assert after != before


def test_linear_identity_rechecks_sources_and_compiled_build(tmp_path, monkeypatch):
    source = tmp_path / "quantization.py"
    source.write_bytes(b"validated quantizer")
    identity = {
        "source_and_dependency_sha256": {str(source): sha256(source.read_bytes())},
        "kernel_policy": "fixture",
    }
    module = ModuleType("operators.deepseek_v32.linear.quantization")
    module.build_info = lambda: identity
    module.runtime_info = lambda: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert provenance._linear_quantization_identity() is identity

    runtime = {"build_info": identity, "specializations": [{"hash": "first"}]}
    module.runtime_info = lambda: runtime
    assert provenance._linear_quantization_identity() is identity
    runtime["specializations"].append({"hash": "second"})
    assert provenance._linear_quantization_identity() is identity
    runtime["build_info"] = {**identity, "kernel_policy": "old"}
    with pytest.raises(RuntimeError, match="differs from current build identity"):
        provenance._linear_quantization_identity()

    module.runtime_info = lambda: None
    source.write_bytes(b"unvalidated edit")
    with pytest.raises(RuntimeError, match="source changed during execution"):
        provenance._linear_quantization_identity()


def test_linear_identity_rejects_empty_source_manifest(monkeypatch):
    module = ModuleType("operators.deepseek_v32.linear.quantization")
    module.build_info = lambda: {"source_and_dependency_sha256": {}}
    module.runtime_info = lambda: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="no declared source identity"):
        provenance._linear_quantization_identity()


def test_recall_dispatch_identity_matches_prepared_build(monkeypatch):
    identity = {"identity": {"source_and_dependency_sha256": {"fixture.cpp": "sha256"}}}
    module = ModuleType("operators.deepseek_v32.indexer.recall_dispatch")
    module.build_info = lambda: identity
    module.runtime_info = lambda: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert provenance._recall_dispatch_identity() is identity
    module.runtime_info = lambda: identity
    assert provenance._recall_dispatch_identity() is identity
    module.runtime_info = lambda: {"identity": {"source_and_dependency_sha256": {"old": "old"}}}
    with pytest.raises(RuntimeError, match="differs from current build identity"):
        provenance._recall_dispatch_identity()
    module.build_info = lambda: {"identity": {"source_and_dependency_sha256": {}}}
    with pytest.raises(RuntimeError, match="no declared source/dependency identity"):
        provenance._recall_dispatch_identity()


def test_q1_topk_runtime_does_not_create_a_native_module(empty_runtime, monkeypatch):
    observed = {"artifact_sha256": "already-loaded", "build_identity": {}}
    module = ModuleType("operators.deepseek_v32.indexer.q1_topk_cub")
    module.runtime_info = lambda: observed
    module._module = lambda: pytest.fail("Runtime inspection must not invoke a module factory")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert provenance.collect_flashinfer_runtime_artifacts()["q1_topk_cub_native"] is observed


def test_q1_topk_identity_rejects_source_and_native_build_changes(tmp_path, monkeypatch):
    source = tmp_path / "sort.cu"
    source.write_bytes(b"accepted CUB adapter")
    identity = {"source_sha256": {str(source): sha256(source.read_bytes())}}
    module = ModuleType("operators.deepseek_v32.indexer.q1_topk_cub")
    module.build_info = lambda: identity
    module.runtime_info = lambda: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert provenance._q1_topk_identity() is identity
    module.runtime_info = lambda: {"build_identity": {"source_identity": identity}}
    assert provenance._q1_topk_identity() is identity
    module.runtime_info = lambda: {"build_identity": {"source_identity": {"stale": True}}}
    with pytest.raises(RuntimeError, match="differs from current build identity"):
        provenance._q1_topk_identity()
    module.runtime_info = lambda: None
    source.write_bytes(b"unaccepted source change")
    with pytest.raises(RuntimeError, match="source changed during execution"):
        provenance._q1_topk_identity()
    module.build_info = lambda: {"source_sha256": {}}
    with pytest.raises(RuntimeError, match="no declared source identity"):
        provenance._q1_topk_identity()
