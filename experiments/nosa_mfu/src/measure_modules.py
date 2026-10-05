"""Measure complete resident NOSA modules on unchanged captured activations.

Each indexer call starts from the same committed derived-cache prefix. Validation,
incremental compression and selection are inside the measured public API. Eager
API intervals and separately profiled kernel sums have distinct metric names.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from experiments.nosa_mfu.src.dense.sources import source_hashes
from experiments.nosa_mfu.src.measure import (
    copy_to_device,
    distribution,
    numerical_check,
    tensor_hash,
    write_json,
)
from experiments.nosa_mfu.src.phases import (
    add_phase_arguments,
    check_case_identity,
    finish_validation,
    open_validation,
    validate_phase_arguments,
)
from experiments.nosa_mfu.src.sparse.mfu import work_counts
from models.attention_contracts import AttentionContext, BlockSelection
from models.nosa.attention import NosaSparseAttention
from models.nosa.cache.resident import NosaKVCache
from models.nosa.config import NosaConfig
from models.nosa.indexer import NosaIndexer, prepare_indexer_inputs
from operators.nosa._native import build_info
from operators.nosa.attention.device_only.api import nosa_block_sparse_attention
from operators.nosa.indexer.api import select_contiguous_blocks

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]


class CapturedModules:
    """Invoke model APIs with real request-cache lifecycle and a fresh append."""

    def __init__(self, tensors, prefix):
        self.q, self.k, self.v, self.cis = (tensors[name] for name in ("q", "k", "v", "cis"))
        self.prefix = prefix
        self.selection = BlockSelection(tensors["ids"], 64, tensors["valid_mask"])
        config = NosaConfig(
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=1,
            num_attention_heads=32,
            num_key_value_heads=2,
            head_dim=128,
            vocab_size=128256,
            max_position_embeddings=len(self.k),
        )
        self.cache = NosaKVCache(
            config, len(self.k), device=self.q.device, dtype=self.q.dtype, with_cis=True
        )
        self.indexer = NosaIndexer(mode="nosa", backend="triton")
        self.attention = NosaSparseAttention(backend="triton")
        self.context = AttentionContext(0, prefix, len(self.q))
        self.cache.begin_step(prefix)
        self.cache.write_layer(
            0, keys=self.k[:prefix], values=self.v[:prefix], cis_scores=self.cis[:prefix]
        )
        self.indexer(self.q[:1], self.cache, AttentionContext(0, prefix - 1, 1))
        self.cache.commit_step()
        self.prefix_state = self.cache.indexer_cache.layer_state(0)

    def prepare(self):
        if self.cache.indexer_cache.layer_state(0) != self.prefix_state:
            raise RuntimeError("Derived cache did not roll back to the measured prefix")
        self.cache.begin_step(len(self.q))
        self.cache.write_layer(
            0,
            keys=self.k[self.prefix :],
            values=self.v[self.prefix :],
            cis_scores=self.cis[self.prefix :],
        )
        torch.cuda.synchronize(self.q.device)

    def call(self, module):
        if module == "indexer_total":
            return self.indexer(self.q, self.cache, self.context)
        return self.attention(self.q, self.selection, self.cache, self.context)

    def cleanup(self):
        self.cache.abort_step()


def output_tensors(output):
    if isinstance(output, BlockSelection):
        return (output.block_ids, output.valid_mask)
    return (output,)


def assert_same_output(actual, expected):
    for tensor, reference in zip(output_tensors(actual), expected, strict=True):
        torch.testing.assert_close(tensor, reference, rtol=0, atol=0)


def kernel_activity(trace):
    """Read isolated profiler device events, excluding runtime/CPU intervals."""
    kernels = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"]
    copies = [event for event in trace["traceEvents"] if event.get("cat") == "gpu_memcpy"]
    return {
        "kernel_ms": sum(event["dur"] for event in kernels) / 1000,
        "memcpy_ms": sum(event["dur"] for event in copies) / 1000,
        "kernels": [{"name": event["name"], "us": event["dur"]} for event in kernels],
    }


def check_module(fixture, module):
    expected = None
    for _ in range(3):
        fixture.prepare()
        try:
            output = fixture.call(module)
            torch.cuda.synchronize(fixture.q.device)
            if expected is None:
                expected = tuple(tensor.clone() for tensor in output_tensors(output))
            else:
                assert_same_output(output, expected)
        finally:
            fixture.cleanup()
    if module == "block_sparse_attention_total":
        reference = nosa_block_sparse_attention(
            fixture.q, fixture.k, fixture.v, fixture.selection, fixture.prefix, fixture.cis
        )
        assert_same_output(reference, expected)
    return {
        "repeated_outputs_match_exactly": True,
        "output_sha256": [tensor_hash(tensor) for tensor in expected],
    }


def measure_module(fixture, module, args, label):
    eager, wall = [], []
    if args.mode == "bench":
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        end.record()
        end.synchronize()
        for iteration in range(args.warmup + args.repeats):
            fixture.prepare()
            try:
                begin.record()
                started = time.perf_counter_ns()
                fixture.call(module)
                end.record()
                end.synchronize()
                elapsed_wall = (time.perf_counter_ns() - started) / 1e6
                if iteration >= args.warmup:
                    eager.append(begin.elapsed_time(end))
                    wall.append(elapsed_wall)
            finally:
                fixture.cleanup()
        return {
            "eager_api": {"ms": distribution(eager), "samples_ms": eager},
            "wall_api": {"ms": distribution(wall), "samples_ms": wall},
            "numerical_acceptance_source": "independent_check",
        }
    profiles = []
    for _ in range(args.warmup):
        fixture.prepare()
        try:
            fixture.call(module)
            torch.cuda.synchronize(fixture.q.device)
        finally:
            fixture.cleanup()
    for iteration in range(args.profile_repeats):
        fixture.prepare()
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                fixture.call(module)
                torch.cuda.synchronize(fixture.q.device)
            path = args.profile_dir / f"{label}_{args.kernel_backend}_{module}_{iteration}.json"
            profile.export_chrome_trace(str(path))
            content = path.read_bytes()
            activity = kernel_activity(json.loads(content))
            if not activity["kernels"]:
                raise RuntimeError("Profiler recorded no CUDA kernels for the complete module")
            profiles.append(
                {"file": path.name, "sha256": hashlib.sha256(content).hexdigest(), **activity}
            )
        finally:
            fixture.cleanup()
    return {
        "profile_kernel_sum": {
            "ms": distribution([sample["kernel_ms"] for sample in profiles]),
            "samples_ms": [sample["kernel_ms"] for sample in profiles],
        },
        "profiles": profiles,
        "numerical_acceptance_source": "independent_check",
    }


def check_indexer_composition(fixture, tensors):
    compressed, cis, pool, _ = prepare_indexer_inputs(fixture.k, fixture.cis, len(fixture.q))
    torch.testing.assert_close(compressed, tensors["compressed_k"], rtol=0, atol=0)
    ids, valid = select_contiguous_blocks(
        fixture.q.view(-1, 2, 16, 128),
        compressed,
        cis,
        fixture.prefix,
        len(fixture.k),
        pooled_cis=pool,
        return_valid_mask=True,
    )
    fixture.prepare()
    try:
        selected = fixture.call("indexer_total")
        assert_same_output(selected, (ids, valid))
        visible = fixture.cache.indexer_cache.layer_view(0)
        for name, reference in (
            ("compressed_keys", compressed),
            ("compressed_cis", cis),
            ("pooled_cis", pool),
        ):
            torch.testing.assert_close(visible[name], reference, rtol=0, atol=0)
    finally:
        fixture.cleanup()
    current = torch.arange(fixture.prefix, len(fixture.k), device=ids.device)[:, None, None] // 64
    if not (
        valid.all()
        and (ids >= 0).all()
        and (ids <= current).all()
        and (ids == current).sum(-1).eq(1).all()
        and ids.diff(dim=-1).gt(0).all()
    ):
        raise RuntimeError("Selection does not satisfy the useful-FLOP policy")
    return {
        "incremental_matches_independent_full_compression": True,
        "selection_matches_standalone_operator_exactly": True,
        "selection_differs_from_capture_elements": int((ids != tensors["ids"]).sum().item()),
    }


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase_arguments(parser)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kernel-backend", choices=("native", "triton"), default="native")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--peak-tflops", type=float)
    args = parser.parse_args(argv)
    validate_phase_arguments(parser, args)
    if args.mode == "profile" and args.profile_dir is None:
        parser.error("--mode profile requires --profile-dir")
    if min(args.warmup, args.repeats, args.profile_repeats) < 1:
        parser.error("warmup, repeats and profile-repeats must be positive")
    if args.mode != "check" and args.peak_tflops is None:
        parser.error("bench/profile requires --peak-tflops")
    if args.peak_tflops is not None and (
        not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0
    ):
        parser.error("peak-tflops must be finite and positive")
    os.environ["CXLDSAGR_SM90_BACKEND"] = args.kernel_backend
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if (props.major, props.minor) != (9, 0):
        raise RuntimeError("Complete NOSA module measurement requires Hopper/SM90")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.mode == "profile":
        args.profile_dir.mkdir(parents=True, exist_ok=True)
    sources = source_hashes(
        *sorted((EXPERIMENT / "src").glob("*.py")),
        *sorted((EXPERIMENT / "scripts").glob("*.sh")),
        ROOT / "experiments/nosa_mfu/src/sparse/mfu.py",
        ROOT / "experiments/nosa_mfu/src/sparse/analyze.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "evaluation/validation.py",
    )
    for name, digest in sources.items():
        content = (ROOT / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError("Source changed while saving snapshot")
        destination = args.output_dir / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    capture_bytes = (args.input_dir / "metadata.json").read_bytes()
    capture = json.loads(capture_bytes)
    if (
        capture.get("kind") != "actual_sparse_model_operator_inputs"
        or capture.get("query_start") != 65536
        or capture.get("queries") != 1024
    ):
        raise ValueError("Expected a completed real-model 64K+1K input capture")
    entries = {entry["layer"]: entry for entry in capture["layers"]}
    if not set(args.layers).issubset(entries):
        raise ValueError("Requested layers are missing from the input capture")
    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_sha256": sources,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_capture_metadata_sha256": hashlib.sha256(capture_bytes).hexdigest(),
        "input_capture": capture,
        "native_build": build_info(),
        "gpu": {
            "name": props.name,
            "uuid": str(props.uuid),
            "sm_count": props.multi_processor_count,
            "capability": [props.major, props.minor],
            "total_memory": props.total_memory,
        },
        "nvidia_smi_before": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version,power.limit,clocks.sm,clocks.mem",
                "--format=csv",
            ],
            text=True,
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("triton", "flashinfer-python", "apache-tvm-ffi")
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("CXLDSAGR_") or key == "CUDA_VISIBLE_DEVICES"
        },
        "measurement": {
            "eager_api": "Public model module CUDA-event interval including validation, incremental derived-cache updates, selection, auxiliary kernels and host submission gaps",
            "wall_api": "Public model module host duration through CUDA completion",
            "profile_kernel_sum": "Separate torch.profiler sample: sum of all CUDA kernels within the complete module; excludes host gaps and copies; diagnostic GPU-only MFU, not API latency",
            "excluded": "Prefix construction, cache allocation, suffix K/V/CIS writes, transaction begin/abort, numerical acceptance, warmup and compilation",
            "attention_selection": "Unchanged captured model selection, identical across backends",
            "flops": "One causal compressed QK for the complete indexer; selected causal QK+AV for attention; no recomputation, padding or masked work",
            "acceptance": "Independent check verifies incremental-cache composition, exact repeated module outputs, module/operator agreement and all-row FP32 operator acceptance; bench/profile reuse its matching receipt",
        },
    }
    identity = open_validation(
        args,
        metadata,
        kind="nosa_resident_modules",
        config={
            "layers": args.layers,
            "kernel_backend": args.kernel_backend,
            "prefix": 65536,
            "queries": 1024,
            "q_heads": 32,
            "kv_heads": 2,
            "head_dim": 128,
            "block_budget": 64,
        },
        cache="owned_resident_cache_prefix_append_abort",
    )
    results = []
    checked_cases = {}
    counts = work_counts(65536, 1024, 1024)
    flops = {
        "indexer_total": 2 * 32 * 128 * counts["compressed_key_pairs_per_q_head_layer"],
        "block_sparse_attention_total": 4
        * 32
        * 128
        * counts["sparse_token_pairs_per_q_head_layer"],
    }
    for layer in args.layers:
        entry = entries[layer]
        path = args.input_dir / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["file_sha256"]:
            raise ValueError(f"Input capture hash mismatch: {path}")
        tensors = {
            name: copy_to_device(tensor, args.device)
            for name, tensor in torch.load(path, map_location="cpu", weights_only=True).items()
        }
        label = f"layer_{layer:02d}"
        tensor_identity = {
            name: {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "sha256": tensor_hash(tensor),
            }
            for name, tensor in tensors.items()
        }
        prior = check_case_identity(args, label, tensor_identity)
        fixture = CapturedModules(tensors, 65536)
        if args.mode == "check":
            checked = check_indexer_composition(fixture, tensors)
            workspace = torch.empty(
                (len(fixture.q) * 2, (len(fixture.k) + 63) // 64),
                device=fixture.q.device,
                dtype=fixture.q.dtype,
            )
            checked["fp32_operators"] = numerical_check(
                fixture.q,
                fixture.k,
                fixture.v,
                tensors["compressed_k"],
                fixture.selection,
                fixture.cis,
                fixture.prefix,
                workspace,
                all_rows=True,
            )
            checked_cases[label] = {
                "tensors": tensor_identity,
                "composition": checked,
                "modules": {},
            }
        else:
            checked = prior["composition"]
        for module, useful_flops in flops.items():
            if args.mode == "check":
                checked_cases[label]["modules"][module] = check_module(fixture, module)
                results.append(
                    {
                        "layer": layer,
                        "module": module,
                        "composition_check": checked,
                        **checked_cases[label]["modules"][module],
                    }
                )
                continue
            timings = measure_module(fixture, module, args, label)
            for method in ("eager_api", "wall_api", "profile_kernel_sum"):
                if method not in timings:
                    continue
                timings[method]["mfu_pct"] = (
                    useful_flops / timings[method]["ms"]["median"] / (args.peak_tflops * 1e9) * 100
                )
            result = {
                "layer": layer,
                "module": module,
                "useful_flops": useful_flops,
                "composition_check": checked,
                **timings,
            }
            results.append(result)
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "module": module,
                        **{
                            method: timings[method]
                            for method in ("eager_api", "profile_kernel_sum")
                            if method in timings
                        },
                    }
                ),
                flush=True,
            )
        fixture.cache.release()
    if source_hashes(*(ROOT / name for name in sources)) != sources:
        raise RuntimeError("Source changed during measurement; results cannot be published")
    finish_validation(
        args, metadata, kind="nosa_resident_modules", identity=identity, cases=checked_cases
    )
    metadata["nvidia_smi_after"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version,power.limit,clocks.sm,clocks.mem",
            "--format=csv",
        ],
        text=True,
    )
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "results.json", {"run_id": args.run_id, "results": results})


if __name__ == "__main__":
    main()
