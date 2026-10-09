"""Resolve the CUDA runtime used by Torch when identical runtime copies are loaded."""

from __future__ import annotations

import ctypes as ct
import hashlib
from pathlib import Path
from unittest.mock import patch

from experiments.deepseek_v32_motivation.src import graph_instrumentation as shared


class DlInfo(ct.Structure):
    _fields_ = [
        ("filename", ct.c_char_p),
        ("base", ct.c_void_p),
        ("symbol", ct.c_char_p),
        ("address", ct.c_void_p),
    ]


class GraphInspector(shared.GraphInspector):
    """Use Torch's actual loaded symbol provider, without loading another runtime."""

    def __init__(self):
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if line.split()[-1].startswith("/")
        }
        torch_paths = [path for path in paths if path.endswith("/libtorch_cuda.so")]
        runtime_paths = sorted(path for path in paths if "/libcudart.so" in path)
        if len(torch_paths) != 1 or not runtime_paths:
            raise RuntimeError("Cannot resolve the already-loaded Torch CUDA/runtime libraries")
        digests = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in runtime_paths
        }
        if len(set(digests.values())) != 1:
            raise RuntimeError("Loaded CUDA runtime copies have different bytes")
        torch_library = ct.CDLL(torch_paths[0])
        symbol = ct.cast(torch_library.cudaRuntimeGetVersion, ct.c_void_p)
        dladdr = ct.CDLL(None).dladdr
        dladdr.argtypes = [ct.c_void_p, ct.POINTER(DlInfo)]
        dladdr.restype = ct.c_int
        info = DlInfo()
        if not dladdr(symbol, ct.byref(info)) or not info.filename:
            raise RuntimeError("dladdr could not identify Torch's CUDA runtime symbol")
        runtime_path = str(Path(info.filename.decode()).resolve(strict=True))
        normalized = {str(Path(path).resolve(strict=True)): path for path in runtime_paths}
        if runtime_path not in normalized:
            raise RuntimeError("Torch runtime symbol points outside the observed loaded libraries")
        runtime_path = normalized[runtime_path]
        original = shared._loaded_library

        def loaded(stem, *, nsight=False):
            if stem == "libcudart.so" and not nsight:
                return ct.CDLL(runtime_path), runtime_path
            return original(stem, nsight=nsight)

        # Only the observer's provider resolver changes during construction.
        with patch.object(shared, "_loaded_library", loaded):
            super().__init__()
        self.provenance["runtime_resolution"] = {
            "method": "dladdr(cudaRuntimeGetVersion resolved through the loaded libtorch_cuda.so)",
            "torch_library": torch_paths[0],
            "selected_runtime": runtime_path,
            "loaded_identical_runtime_sha256": digests,
        }
