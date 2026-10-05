"""Compare resident NOSA attention with serialized and overlapped sparse host fetch.

Each iteration starts with a cold prefix and fetches its sparse union once.
The control fetches the whole union before full-query FA3. The candidate fuses
host fetch with persistent FA3 consumers. This is a one-layer replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from experiments.nosa_mfu.src.capture_inputs import sha256_file, tensor_metadata
from experiments.nosa_mfu.src.dense.sources import source_hashes
from experiments.nosa_mfu.src.measure import copy_to_device
from experiments.nosa_mfu.src.phases import (
    add_phase_arguments,
    check_case_identity,
    finish_validation,
    offload_config,
    open_validation,
    validate_phase_arguments,
)
from experiments.nosa_offload_overlap.src.analyze import (
    PAGE_ENVELOPE_DEFINITION,
    STRIPE_COPY_DEFINITION,
    validate_fetch_stripes,
)
from models.attention_contracts import BlockSelection
from operators.nosa._native import build_info
from operators.nosa.attention.device_only.api import nosa_block_sparse_attention
from operators.nosa.attention.reference.torch import reference_nosa_block_sparse_attention

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
MODES = ("resident", "serialized", "overlap")


def write_json(path, value):
    with path.open("x") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def native_build_metadata():
    """Record the actual cooperative build, including its pinned FA3 headers."""
    from operators.nosa.attention.offload._fused import build_info as fused_build_info

    return {**build_info(), "offload_fused": fused_build_info()}


def new_work_profile(run_id, native_build):
    """Bind trace geometry to the compiled operator instead of a harness constant."""
    stripes = native_build.get("offload_fused", {}).get("fetch_stripes")
    validate_fetch_stripes(stripes)
    return {
        "schema_version": 3,
        "run_id": run_id,
        "clock": "device_globaltimer_ns",
        "math_coverage": "softmax_only",
        "fetch_coverage": "per_page_copy_envelopes",
        "stripe_coverage": "nonempty_stripe_copy_windows",
        "fetch_stripes": stripes,
        "byte_accounting": "logical_unique_kv_payload",
        "fetch_definition": PAGE_ENVELOPE_DEFINITION,
        "stripe_definition": STRIPE_COPY_DEFINITION,
        "math_definition": "Instrumented consumer softmax updates only; excludes ready polling and does not cover all QK/PV MMA execution",
        "cases": {},
        "records": [],
    }


def prefix_transfer_bytes(ids, valid, prefix, head_dim, element_size, *, tile_size):
    """Independent CPU first-use accounting, including per-head and partial blocks."""
    if ids.device.type != "cpu" or valid.device.type != "cpu":
        raise ValueError("Transfer accounting must run on CPU outside measured calls")
    if ids.ndim != 3 or valid.shape != ids.shape or valid.dtype != torch.bool:
        raise ValueError("Expected [query, KV head, selected block] IDs and boolean validity")
    if min(head_dim, element_size, tile_size) <= 0 or prefix < 0:
        raise ValueError("Positive geometry and nonnegative prefix are required")
    seen = [set() for _ in range(ids.shape[1])]
    per_tile = []
    # FA3 pads an odd page union with physical block zero, with membership
    # masked out. Fetch selected block zero in tile zero to avoid racing that
    # speculative read against a later first-use write.
    early_zero = [
        bool(((ids[:, head] == 0) & valid[:, head]).any()) for head in range(ids.shape[1])
    ]
    for begin in range(0, ids.shape[0], tile_size):
        amount = 0
        for head in range(ids.shape[1]):
            chosen = ids[begin : begin + tile_size, head]
            mask = valid[begin : begin + tile_size, head]
            blocks = set(chosen[mask & (chosen >= 0) & (chosen * 64 < prefix)].tolist())
            if begin == 0 and prefix and early_zero[head]:
                blocks.add(0)
            for block in blocks - seen[head]:
                amount += min(64, prefix - block * 64) * head_dim * element_size * 2
            seen[head].update(blocks)
        per_tile.append(amount)
    return sum(per_tile), per_tile


def expected_fetch_rows(ids, valid, prefix, head_dim=128, element_size=2):
    """CPU provenance for each unique (logical block, KV head) host copy."""
    result = []
    for head in range(ids.shape[1]):
        chosen = ids[:, head][valid[:, head]]
        for block in sorted(set(chosen[(chosen >= 0) & (chosen < (prefix + 63) // 64)].tolist())):
            result.append(
                {
                    "row": block * ids.shape[1] + head,
                    "bytes": min(64, prefix - block * 64) * head_dim * element_size * 2,
                }
            )
    return sorted(result, key=lambda entry: entry["row"])


def _validate_case(tensors, prefix, queries):
    required = {"q", "k", "v", "cis", "ids", "valid_mask"}
    if missing := required - tensors.keys():
        raise ValueError(f"Missing input tensors: {sorted(missing)}")
    q, k, v, cis, ids, valid = (tensors[name] for name in required_order())
    if q.shape != (queries, 32, 128) or q.dtype != torch.bfloat16:
        raise ValueError("This first offload backend requires BF16 NOSA-8B Q=[queries,32,128]")
    if k.shape != (prefix + queries, 2, 128) or v.shape != k.shape:
        raise ValueError("K/V must span prefix + suffix with two KV heads and D128")
    if k.dtype != q.dtype or v.dtype != q.dtype or cis.shape != k.shape[:2]:
        raise ValueError("K/V dtype or CIS shape differs from the attention contract")
    if ids.shape != (queries, 2, 64) or valid.shape != ids.shape or valid.dtype != torch.bool:
        raise ValueError("Expected per-query, per-KV-head 64-block NOSA selection")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Selection IDs must be integral")
    for name in ("q", "k", "v", "cis"):
        if not torch.isfinite(tensors[name]).all():
            raise ValueError(f"Captured {name} contains nonfinite values")
    for row in range(queries):
        for head in range(2):
            chosen = ids[row, head][valid[row, head]]
            if chosen.unique().numel() != chosen.numel():
                raise ValueError("NOSA selections must not duplicate valid blocks")
            if (chosen < 0).any() or (chosen > (prefix + row) // 64).any():
                raise ValueError("NOSA selections must be causal valid logical block IDs")


def required_order():
    return ("q", "k", "v", "cis", "ids", "valid_mask")


def synthetic_case(args):
    """Generate random activations, then obtain full NOSA selection outside timing."""
    from models.nosa.indexer import prepare_indexer_inputs
    from operators.nosa.indexer.api import select_contiguous_blocks

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    kwargs = {"device": args.device, "dtype": torch.bfloat16, "generator": generator}
    q = torch.randn(args.queries, 4608, **kwargs)[:, :4096].view(args.queries, 32, 128)
    keys = torch.randn(args.prefix + args.queries, 2, 128, **kwargs)
    values = torch.randn(keys.shape, **kwargs)
    bias = torch.randn(args.prefix + args.queries, 2, **kwargs) * 0.1
    compressed, cis, stable_cis, _ = prepare_indexer_inputs(keys, bias, len(q))
    ids, valid = select_contiguous_blocks(
        q.view(-1, 2, 16, 128),
        compressed,
        cis,
        args.prefix,
        len(keys),
        return_valid_mask=True,
        pooled_cis=stable_cis,
    )
    tensors = dict(zip(required_order(), (q, keys, values, bias, ids, valid), strict=True))
    return {name: tensor.cpu() for name, tensor in tensors.items()}


def reference_acceptance(q, keys, values, selection, bias, prefix, outputs, *, all_rows):
    rows = (
        list(range(len(q)))
        if all_rows
        else sorted({0, 1, 7, 8, 63, 127, len(q) // 2, len(q) - 1} & set(range(len(q))))
    )
    errors = {
        mode: {"max_abs": 0.0, "difference_squared": 0.0, "reference_squared": 0.0}
        for mode in MODES
    }
    for row in rows:
        chosen = BlockSelection(
            selection.block_ids[row : row + 1], 64, selection.valid_mask[row : row + 1]
        )
        expected = reference_nosa_block_sparse_attention(
            q[row : row + 1], keys, values, chosen, prefix + row, bias
        )
        for mode, output in outputs.items():
            actual = output[row : row + 1]
            torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.016)
            delta = actual.float() - expected.float()
            errors[mode]["max_abs"] = max(errors[mode]["max_abs"], delta.abs().max().item())
            errors[mode]["difference_squared"] += delta.square().sum().item()
            errors[mode]["reference_squared"] += expected.float().square().sum().item()
    return {
        "reference": "Independent FP32 QK + CIS, causal softmax and AV over original logical selections",
        "rows": rows,
        "rtol": 0.016,
        "atol": 0.016,
        "errors": {
            mode: {
                "max_abs": error["max_abs"],
                "relative_l2": (
                    error["difference_squared"] / max(error["reference_squared"], 1e-30)
                )
                ** 0.5,
            }
            for mode, error in errors.items()
        },
    }


def benchmark_case(cpu, args, *, label, prefix, work_profile=None):
    from operators.nosa.attention.offload.api import NosaFetchWorkspace

    tensor_identity = {name: tensor_metadata(cpu[name]) for name in required_order()}
    prior = check_case_identity(args, label, tensor_identity)
    if args.mode == "check":
        _validate_case(cpu, prefix, args.queries)
    tensors = {name: copy_to_device(cpu[name], args.device) for name in required_order()}
    q, keys, values, bias, ids, valid = (tensors[name] for name in required_order())
    selection = BlockSelection(ids, 64, valid)
    # Allocate/copy the stable pinned backing before timing. The suffix is produced
    # on GPU and is not charged as host traffic; workspace staging stays timed.
    begin_setup = time.perf_counter()
    host_keys = keys[:prefix].cpu().contiguous().pin_memory()
    host_values = values[:prefix].cpu().contiguous().pin_memory()
    host_setup_ms = (time.perf_counter() - begin_setup) * 1000
    suffix_keys, suffix_values = keys[prefix:], values[prefix:]
    workspaces = {
        mode: NosaFetchWorkspace(
            len(keys),
            2,
            128,
            device=args.device,
            dtype=q.dtype,
            query_tile_size=args.tile_size,
            overlap=(mode == "overlap"),
            fetch_ctas=args.fetch_ctas,
        )
        for mode in ("serialized", "overlap")
    }
    for workspace in workspaces.values():
        workspace.profile_work_intervals = args.profiled
    calls = {
        "resident": lambda: nosa_block_sparse_attention(q, keys, values, selection, prefix, bias)
    }
    for mode, workspace in workspaces.items():
        calls[mode] = lambda workspace=workspace: workspace.run(
            q, selection, host_keys, host_values, suffix_keys, suffix_values, bias, prefix
        )
    expected_bytes, tile_bytes = prefix_transfer_bytes(
        cpu["ids"], cpu["valid_mask"], prefix, 128, 2, tile_size=args.tile_size
    )
    if work_profile is not None:
        work_profile["cases"][label] = {
            "prefix": prefix,
            "queries": len(q),
            "kv_heads": 2,
            "expected_prefix_bytes": expected_bytes,
            "expected_fetch_rows": expected_fetch_rows(cpu["ids"], cpu["valid_mask"], prefix),
        }

    def check_traffic():
        for mode, workspace in workspaces.items():
            actual = int(workspace.last_transfer_bytes.item())
            if actual != expected_bytes:
                raise AssertionError(
                    f"{mode} copied {actual} bytes, expected deduplicated {expected_bytes}"
                )
            if (
                hasattr(workspace, "last_tile_transfer_bytes")
                and workspace.last_tile_transfer_bytes.tolist() != tile_bytes
            ):
                raise AssertionError(
                    f"{mode} per-tile first-use traffic differs from CPU accounting"
                )

    if args.mode in ("check", "profile"):
        outputs = {mode: call().clone() for mode, call in calls.items()}
        torch.cuda.synchronize()
        check_traffic()
        for mode, output in outputs.items():
            if not torch.isfinite(output).all():
                raise AssertionError(f"{mode} output contains NaN/Inf")
            torch.testing.assert_close(output, outputs["resident"], rtol=0.016, atol=0.016)
        torch.testing.assert_close(outputs["overlap"], outputs["serialized"], rtol=0, atol=0)
    if args.mode == "check":
        acceptance = reference_acceptance(
            q, keys, values, selection, bias, prefix, outputs, all_rows=True
        )
        # Exercise cold reset and reuse separately from performance samples.
        for _ in range(3):
            for mode, call in calls.items():
                actual = call()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, outputs[mode], rtol=0, atol=0)
            check_traffic()
        for workspace in workspaces.values():
            workspace.synchronize()
        return {
            "case": label,
            "tensors": tensor_identity,
            "acceptance": acceptance,
            "serialized_overlap_exact_equal": True,
            "prefix_transfer_bytes": expected_bytes,
            "first_use_tile_bytes": tile_bytes,
            "traffic_checks": "Independent check: CPU sparse union equals GPU total and per-tile counts across cold resets",
        }
    acceptance = prior["acceptance"]
    for _ in range(args.warmup):
        for call in calls.values():
            call()
        torch.cuda.synchronize()
        if args.mode == "profile":
            check_traffic()
    samples = {mode: {"cuda_ms": [], "wall_ms": [], "submit_ms": []} for mode in MODES}
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    end.record()
    end.synchronize()
    # Rotate mode order to avoid assigning a monotonic clock/temperature drift
    # consistently to the same implementation. Calls remain individually isolated.
    for repeat in range(args.repeats):
        for mode in MODES[repeat % 3 :] + MODES[: repeat % 3]:
            torch.cuda.nvtx.range_push(f"nosa_overlap/{label}/{mode}/sample_{repeat}")
            start = time.perf_counter()
            begin.record()
            output = calls[mode]()
            submitted = time.perf_counter()
            end.record()
            end.synchronize()
            finished = time.perf_counter()
            torch.cuda.nvtx.range_pop()
            samples[mode]["cuda_ms"].append(begin.elapsed_time(end))
            samples[mode]["wall_ms"].append((finished - start) * 1000)
            samples[mode]["submit_ms"].append((submitted - start) * 1000)
            if work_profile is not None and mode != "resident":
                # Copy instrumentation only after the end event and outside the
                # measured NVTX range. It must precede the next workspace reuse.
                workspace = workspaces[mode]
                work_profile["records"].append(
                    {
                        "case": label,
                        "mode": mode,
                        "sample": repeat,
                        "recorded_transfer_bytes": int(workspace.last_transfer_bytes.item()),
                        "intervals": workspace.work_intervals(),
                        "stripe_intervals": workspace.stripe_work_intervals(),
                    }
                )
        if args.mode == "profile":
            check_traffic()
    del output
    for workspace in workspaces.values():
        workspace.synchronize()
    result = {
        "case": label,
        "prefix": prefix,
        "queries": len(q),
        "tile_size": args.tile_size,
        "fetch_ctas": args.fetch_ctas,
        "execution_geometry": "whole_query; tile_size is first-use histogram width only",
        "input_kind": "seeded_synthetic_activations_with_full_nosa_selection"
        if args.synthetic
        else "captured_actual_sparse_model_inputs",
        "host_pinned_setup_ms_excluded": host_setup_ms,
        "prefix_transfer_bytes": expected_bytes,
        "first_use_tile_bytes": tile_bytes,
        "dense_prefix_bytes": prefix * 2 * 128 * 2 * 2,
        "suffix_gpu_bytes": len(q) * 2 * 128 * 2 * 2,
        "tensors": tensor_identity,
        "acceptance": acceptance,
        "numerical_acceptance_source": "independent_check",
        "serialized_overlap_exact_equal": prior["serialized_overlap_exact_equal"],
        "traffic_checks": "diagnostic_profile" if args.mode == "profile" else "independent_check",
        "modes": {
            mode: {
                key: {**distribution(values), "samples": values} for key, values in timing.items()
            }
            for mode, timing in samples.items()
        },
    }
    serial = result["modes"]["serialized"]["cuda_ms"]["median"]
    overlap = result["modes"]["overlap"]["cuda_ms"]["median"]
    result["serialized_over_overlap_speedup"] = serial / overlap
    print(
        json.dumps(
            {
                "case": label,
                "prefix_transfer_bytes": expected_bytes,
                "cuda_ms": {mode: result["modes"][mode]["cuda_ms"]["median"] for mode in MODES},
                "serialized_over_overlap_speedup": serial / overlap,
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    add_phase_arguments(result)
    result.add_argument("--run-id", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    inputs = result.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input-dir", type=Path, help="Completed nosa_mfu capture_inputs directory"
    )
    inputs.add_argument(
        "--synthetic",
        action="store_true",
        help="Seeded random activations; never a measured real sparse pattern",
    )
    result.add_argument(
        "--layers", type=int, nargs="+", help="Captured layer indices; default all captured layers"
    )
    result.add_argument("--prefix", type=int, default=65536, help="Synthetic prefix length")
    result.add_argument("--queries", type=int, default=1024)
    result.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="First-use byte histogram width; both offload modes compute the whole query batch",
    )
    result.add_argument(
        "--fetch-ctas",
        type=int,
        default=96,
        help="Maximum CTAs whose spare producer warps fetch KV; every CTA computes attention",
    )
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--repeats", type=int, default=20)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--reference-all",
        action="store_true",
        help="Check all rows against independent FP32 reference outside timers",
    )
    result.add_argument(
        "--profiled", action="store_true", help="Record that this invocation runs under nsys"
    )
    return result


@torch.inference_mode()
def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    validate_phase_arguments(cli, args)
    if args.profiled and args.mode != "profile":
        cli.error("--profiled requires --mode profile")
    if args.mode == "profile" and not args.profiled:
        cli.error("Use scripts/run.sh --profile for the nsys diagnostic phase")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        cli.error("run-id must contain only letters, digits, underscores and hyphens")
    if min(args.queries, args.tile_size, args.fetch_ctas, args.warmup, args.repeats) <= 0:
        cli.error("queries/tile-size/fetch-ctas/warmup/repeats must be positive")
    if args.prefix < 4096 or args.prefix % 64 or args.queries % 8 or args.tile_size % 8:
        cli.error("prefix must be a multiple of 64 >=4096; queries and tile-size multiples of 8")
    if args.layers is not None and (args.synthetic or len(set(args.layers)) != len(args.layers)):
        cli.error("--layers requires captured inputs and distinct layer indices")
    if args.output_dir.exists():
        cli.error("output-dir already exists; choose a new run ID")
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if (props.major, props.minor) != (9, 0):
        cli.error("This experiment requires SM90/Hopper")
    os.environ["CXLDSAGR_SM90_BACKEND"] = "native"
    sources = source_hashes(
        *sorted((EXPERIMENT / "src").glob("*.py")),
        *sorted((EXPERIMENT / "scripts").glob("*.sh")),
        ROOT / "experiments/nosa_mfu/src/capture_inputs.py",
        ROOT / "experiments/nosa_mfu/src/measure.py",
        ROOT / "experiments/nosa_mfu/src/phases.py",
        ROOT / "evaluation/validation.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )
    args.output_dir.mkdir(parents=True)
    for name, expected in sources.items():
        content = (ROOT / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError("Source changed while capturing snapshot")
        destination = args.output_dir / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "gpu": {
            "name": props.name,
            "uuid": str(props.uuid),
            "sm_count": props.multi_processor_count,
            "capability": [props.major, props.minor],
            "total_memory": props.total_memory,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("triton", "flashinfer-python", "apache-tvm-ffi")
        },
        "native_build": native_build_metadata(),
        "source_sha256": sources,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("CXLDSAGR_") or key in ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING")
        },
        "measurement": {
            "scope": "Isolated one-layer replay, all three paths receive identical Q/K/V/CIS/selection; not a 32-layer forward, indexer offload, or serving benchmark",
            "resident": "Original full-query native FA3 attention with complete K/V in HBM",
            "serialized": "Optimized whole sparse-union host fetch by 128-thread CTAs with unrolled per-page loads, followed by full-query FA3; same .cv host-load cache policy and exact selected K+V payload as the candidate, with independent copy scheduling and no query tiling",
            "overlap": "One cooperative main kernel with spare producer threads claiming stripes of unique selected pages from a compact GPU queue and fetching KV through a vector load/store loop, while all CTAs compute persistent FA3; queue compaction, preparation and numerical repair remain included",
            "tile_size": "First-use histogram width only; does not partition attention execution",
            "cold_prefix": "Workspace resets first-use state every invocation; only layer-local unique selected prefix blocks fetched, never persist a warm prefix across repeats",
            "included": "First-use planning, workspace reset, selected-page queue compaction, uncached pinned-host reads, GPU suffix staging, full-query attention, preparation, repair and completion dependencies",
            "excluded": "Input loading, H2D input initialization, pinning/registration, KV cache writeback, indexer/CIS calculation, compilation, warmup, numerical and traffic checks",
            "cuda_ms": "CUDA events on caller stream; workspace run joins its fetch/compute work before the end event",
            "wall_ms": "Host clock from before begin-event record through completion synchronization",
            "submit_ms": "Host clock through operator return, without completion wait",
            "bandwidth": "Actual mapped-host hardware transfer, no enforced 50 GB/s bandwidth limit",
            "capacity": "Workspace capacity is full logical NHD length; counts unique sparse K+V payload, not a bounded HBM replacement cache",
            "byte_accounting": "Independent check compares GPU counters with the CPU sparse union; profile retains diagnostics, bench reuses the receipt without reading counters between samples. Counts cover logical unique K+V payload bytes, not physical PCIe/CXL host-read bytes or hardware rereads",
            "profiled": args.profiled,
            "work_instrumentation": "Profile runs only: device globaltimer records nonempty stripe copy windows, their exact per-page min/max envelopes and consumer softmax updates. Stripe and page unions are independently intersected with softmax; page envelopes alone cannot establish an overlap threshold. Stripe identity and bytes must partition every selected historical page exactly. Queue compaction is timed preparation work and performs no host KV fetch. Softmax coverage excludes QK/PV MMA. These windows do not establish wire occupancy or physical host-read traffic",
        },
    }
    identity = open_validation(
        args,
        metadata,
        kind="nosa_offload_attention",
        config=offload_config(args),
        cache="cold_history_no_tags_full_logical_staging_gpu_suffix",
    )
    work_profile = (
        new_work_profile(args.run_id, metadata["native_build"]) if args.profiled else None
    )
    results = []
    if args.synthetic:
        results.append(
            benchmark_case(
                synthetic_case(args),
                args,
                label=f"synthetic_{args.seed}",
                prefix=args.prefix,
                work_profile=work_profile,
            )
        )
    else:
        raw = (args.input_dir / "metadata.json").read_bytes()
        captured = json.loads(raw)
        if (
            captured.get("kind") != "actual_sparse_model_operator_inputs"
            or captured.get("queries") != args.queries
            or not captured.get("layers")
        ):
            raise ValueError("Expected a completed capture with matching query count")
        metadata["input_capture"] = {
            "path": str(args.input_dir.resolve()),
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "metadata": captured,
        }
        wanted = (
            {entry["layer"] for entry in captured["layers"]}
            if args.layers is None
            else set(args.layers)
        )
        entries = [entry for entry in captured["layers"] if entry["layer"] in wanted]
        if {entry["layer"] for entry in entries} != wanted:
            raise ValueError("Requested layer is absent from the completed input capture")
        for entry in entries:
            path = args.input_dir / entry["file"]
            if sha256_file(path) != entry["file_sha256"]:
                raise ValueError(f"Captured input file hash mismatch: {path}")
            cpu = torch.load(path, map_location="cpu", weights_only=True)
            results.append(
                benchmark_case(
                    cpu,
                    args,
                    label=f"layer_{entry['layer']:02d}",
                    prefix=captured["query_start"],
                    work_profile=work_profile,
                )
            )
    if sources != source_hashes(*(ROOT / name for name in sources)):
        raise RuntimeError("Source changed during measurement; rerun from a stable tree")
    finish_validation(
        args,
        metadata,
        kind="nosa_offload_attention",
        identity=identity,
        cases={result["case"]: result for result in results},
    )
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "results.json", {"run_id": args.run_id, "results": results})
    if work_profile is not None:
        write_json(args.output_dir / "work_intervals.json", work_profile)
    if args.mode == "check":
        return
    with (args.output_dir / "summary.csv").open("x") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "case",
                "mode",
                "tile_size",
                "prefix_transfer_bytes",
                "cuda_ms",
                "wall_ms",
                "submit_ms",
            ],
        )
        writer.writeheader()
        for result in results:
            for mode in MODES:
                writer.writerow(
                    {
                        "case": result["case"],
                        "mode": mode,
                        "tile_size": args.tile_size,
                        "prefix_transfer_bytes": 0
                        if mode == "resident"
                        else result["prefix_transfer_bytes"],
                        **{
                            metric: result["modes"][mode][metric]["median"]
                            for metric in ("cuda_ms", "wall_ms", "submit_ms")
                        },
                    }
                )


if __name__ == "__main__":
    main()
