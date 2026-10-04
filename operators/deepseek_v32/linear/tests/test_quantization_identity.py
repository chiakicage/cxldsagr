"""CPU-only import, compiler identity and observed artifact provenance checks."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

HELPER = Path(__file__).resolve().parents[1] / "_quantization_identity.py"


def test_import_and_build_info_do_not_import_gpu_packages(tmp_path):
    wrapper, kernel = tmp_path / "wrapper.py", tmp_path / "kernels.py"
    wrapper.write_text("# wrapper\n")
    kernel.write_text("# kernel\n")
    source = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('isolated_identity', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
identity = module.Identity(sys.argv[2], sys.argv[3], {'target': 'sm_90'})
info = identity.build_info()
assert not any(name in sys.modules for name in ('torch', 'triton', 'tvm_ffi', 'deep_gemm'))
assert identity.runtime_info() is None
assert any(path.endswith('/ptxas') for path in info['source_and_dependency_sha256'])
assert any(path.endswith('.so') for path in info['source_and_dependency_sha256'])
print(json.dumps({'status': 'passed'}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", source, str(HELPER), str(wrapper), str(kernel)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"status": "passed"}


def test_captured_identity_and_live_asm_drift(tmp_path):
    spec = importlib.util.spec_from_file_location("local_identity", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wrapper, kernel = tmp_path / "wrapper.py", tmp_path / "kernels.py"
    wrapper.write_text("# wrapper\n")
    kernel.write_text("# kernel\n")
    identity = module.Identity(wrapper, kernel, {"target": "sm_90"})
    captured = identity.capture()
    Metadata = namedtuple("Metadata", "target num_warps num_stages shared enable_fp_fusion")
    compiled = SimpleNamespace(
        hash="first",
        name="kernel",
        asm={"ptx": "old PTX", "cubin": b"old CUBIN"},
        kernel=b"retained CUBIN",
        module=123,
        function=456,
        metadata=Metadata({"backend": "cuda", "arch": 90, "warp_size": 32}, 2, 1, 128, False),
        metadata_group={},
    )
    identity.observe(compiled)
    before = identity.runtime_info()
    specialization = before["specializations"][0]
    assert specialization["retained_kernel_sha256"] == hashlib.sha256(b"retained CUBIN").hexdigest()
    assert specialization["module_handle_set"] and specialization["function_handle_set"]
    compiled.asm["ptx"] = "new PTX"
    assert identity.runtime_info()["specializations"] != before["specializations"]
    kernel.write_text("# changed kernel\n")
    assert identity.build_info() != captured
    assert identity.runtime_info()["build_info"] == captured
    other = SimpleNamespace(**{**vars(compiled), "hash": "second"})
    with pytest.raises(RuntimeError, match="changed before observing"):
        identity.observe(other)


def test_compiler_flags_change_identity_but_cache_locations_do_not(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("compiler_environment_identity", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wrapper, kernel = tmp_path / "wrapper.py", tmp_path / "kernel.py"
    wrapper.write_text("# wrapper\n")
    kernel.write_text("# kernel\n")
    monkeypatch.delenv("PTXAS_OPTIONS", raising=False)
    monkeypatch.delenv("DISABLE_PTXAS_OPT", raising=False)
    identity = module.Identity(wrapper, kernel, {"target": "sm_90"})
    original = identity.capture()
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRITON_DUMP_DIR", str(tmp_path / "dump"))
    assert identity.build_info() == original
    monkeypatch.setenv("PTXAS_OPTIONS", "--warn-on-spills")
    current = identity.build_info()
    assert current != original
    assert current["compiler_environment"]["PTXAS_OPTIONS"] == "--warn-on-spills"
    monkeypatch.setenv("DISABLE_PTXAS_OPT", "1")
    assert identity.build_info() != current
    assert identity.build_info()["compiler_environment"]["DISABLE_PTXAS_OPT"] == "1"
    assert identity.captured == original
