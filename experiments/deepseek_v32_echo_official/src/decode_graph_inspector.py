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


def torch_runtime_provider():
    """Resolve Torch's already-loaded runtime without depending on a profiler."""
    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if line.split()[-1].startswith("/")
    }
    torch_paths = [path for path in paths if path.endswith("/libtorch_cuda.so")]
    runtime_paths = sorted(path for path in paths if "/libcudart.so" in path)
    if len(torch_paths) != 1 or not runtime_paths:
        raise RuntimeError("Cannot resolve the already-loaded Torch CUDA/runtime libraries")
    digests = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in runtime_paths}
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
    return runtime_path, {
        "method": "dladdr(cudaRuntimeGetVersion resolved through the loaded libtorch_cuda.so)",
        "torch_library": torch_paths[0],
        "selected_runtime": runtime_path,
        "loaded_identical_runtime_sha256": digests,
    }


class GraphInspector(shared.GraphInspector):
    """Use Torch's actual loaded symbol provider, without loading another runtime."""

    def __init__(self):
        runtime_path, resolution = torch_runtime_provider()
        original = shared._loaded_library

        def loaded(stem, *, nsight=False):
            if stem == "libcudart.so" and not nsight:
                return ct.CDLL(runtime_path), runtime_path
            return original(stem, nsight=nsight)

        with patch.object(shared, "_loaded_library", loaded):
            super().__init__()
        self.provenance["runtime_resolution"] = resolution


class NativeCaptureInspector:
    """Enumerate capture handles using the CUDA runtime without CUPTI/NSYS."""

    def __init__(self):
        runtime_path, resolution = torch_runtime_provider()
        self.runtime = ct.CDLL(runtime_path)
        pointer = ct.c_void_p
        self.runtime.cudaRuntimeGetVersion.argtypes = [ct.POINTER(ct.c_int)]
        version = ct.c_int()
        shared.GraphInspector._check(
            self.runtime.cudaRuntimeGetVersion(ct.byref(version)), "runtime version"
        )
        self.capture_v3 = version.value >= 13000
        name = "cudaStreamGetCaptureInfo" if self.capture_v3 else "cudaStreamGetCaptureInfo_v2"
        self.capture_info = getattr(self.runtime, name)
        args = [
            pointer,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_ulonglong),
            ct.POINTER(pointer),
            ct.POINTER(pointer),
        ]
        if self.capture_v3:
            args.append(ct.POINTER(pointer))
        self.capture_info.argtypes = [*args, ct.POINTER(ct.c_size_t)]
        self.runtime.cudaGraphGetNodes.argtypes = [
            pointer,
            ct.POINTER(pointer),
            ct.POINTER(ct.c_size_t),
        ]
        self.runtime.cudaGraphNodeGetType.argtypes = [pointer, ct.POINTER(ct.c_int)]
        self.provenance = {
            "cuda_runtime_version": version.value,
            "capture_info_api": name,
            "runtime_resolution": resolution,
            "node_identity": "Native capture node handles valid only inside this owned process; not CUPTI node IDs and not compatible with NSYS timeline attribution.",
        }

    def snapshot(self, stream):
        status, capture_id, graph = ct.c_int(), ct.c_ulonglong(), ct.c_void_p()
        dependencies, size = ct.c_void_p(), ct.c_size_t()
        args = [
            stream,
            ct.byref(status),
            ct.byref(capture_id),
            ct.byref(graph),
            ct.byref(dependencies),
        ]
        if self.capture_v3:
            args.append(None)
        check = shared.GraphInspector._check
        check(self.capture_info(*args, ct.byref(size)), "capture info")
        if status.value != 1 or not graph.value:
            raise RuntimeError("Recall observation requires live native CUDA capture")
        check(self.runtime.cudaGraphGetNodes(graph, None, ct.byref(size)), "graph node count")
        nodes = (ct.c_void_p * size.value)()
        if size.value:
            check(self.runtime.cudaGraphGetNodes(graph, nodes, ct.byref(size)), "graph nodes")
        result = {}
        for node in nodes:
            kind = ct.c_int()
            check(self.runtime.cudaGraphNodeGetType(node, ct.byref(kind)), "graph node type")
            result[int(node)] = kind.value
        return capture_id.value, result
