"""Offline module MFU from correlated nsys kernel durations and NOSA NVTX scopes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from experiments.indexer_block_sparse_profile.src.analyze import (
    NESTED_STAGES,
    PHASES,
    PRIMARY_STAGES,
    STAGES,
    _integer,
)
from experiments.indexer_block_sparse_profile.src.mfu import build_report, matrix_flops

ROOT_RANGE = re.compile(r"NOSA/profile/(full_prefill|extend)/(\d+)\Z")
STAGE_RANGE = re.compile(r"NOSA/(full_prefill|extend)/layer_(\d+)/([a-z_]+)/q(\d+)\+(\d+)\Z")
LINEARS = ("qkv_proj", "cis_projection", "o_proj", "gate_up_proj", "down_proj")
MODULES = (
    *LINEARS,
    *[stage for stage in STAGES if stage != "cis_projection"],
    "cis_projection_gemm",
    "other_non_matrix",
)
PARENTS = {stage: "indexer_total" for stage in NESTED_STAGES} | {
    "cis_projection_gemm": "cis_projection"
}
# This inference graph was reviewed against the captured source snapshot.
# A different graph requires review before relying on its unscoped GEMM order.
SUPPORTED_GRAPH = {
    "models/nosa/model.py": "5acbd9c5349f75b5b02366849ef7cd76d6f5b0676126bc789b8384939dfc51a5",
    "models/nosa/layers.py": "cf54c7c1e96483175da788491450bd264fae322a6c1452f830b5ce721f308b43",
    "models/nosa/scoring.py": "c4c7e75a15a5ffb95e935b02ec482586a288c64ad079a175b293db6baa3f92d8",
    "models/nosa/indexer.py": "4b1ef076fd759ff6c6f114673fa357488e0ca1d4474a1b20b90430d3db6704c7",
    "models/nosa/attention.py": "1592ff0607cda99b94e75fa62ca4a64f4ec086be4e7aa4cd8714f610c6f07d84",
    "layers/feed_forward.py": "95dcebaa67b4f2069f580ac0831ef8335fc15b022e802708e24f58c02408de4a",
    "operators/sm90/nosa_indexer.py": "640cff9441efe24e579b64d7f91eb6684075f35ae32243f0c46a988b5b7453b5",
    "operators/sm90/_nosa_attention_triton.py": "c9b2060928cff8cbf594a876061546521064da78f52b3297922bcc5dd0f96985",
}


def _matrix_kernel(kernel):
    name = kernel["name"].lower()
    return name.startswith("nvjet_") or "gemm" in name


def _interval(record, label, *, start="start"):
    left = _integer(record.get(start), f"{label}.{start}")
    right = _integer(record.get("end"), f"{label}.end")
    if right <= left:
        raise ValueError(f"{label} interval must have positive duration")
    return left, right


def _workload_calls(phase, num_layers, workload):
    prefix = _integer(workload["prefix_tokens"], "prefix_tokens")
    new = _integer(workload["new_tokens"], "new_tokens", minimum=1)
    total = _integer(workload.get("total_tokens", prefix + new), "total_tokens", minimum=1)
    chunk = _integer(workload["chunk_size"], "chunk_size", minimum=1)
    if total != prefix + new:
        raise ValueError("total_tokens must equal prefix_tokens + new_tokens")
    first = 0 if phase == "full_prefill" else prefix
    return [
        (layer, start, min(chunk, total - start))
        for start in range(first, total, chunk)
        for layer in range(num_layers)
    ]


def _empty_module(name):
    return {
        "kernel_ns": 0,
        "kernel_ms": 0.0,
        "kernel_count": 0,
        "parent_module": PARENTS.get(name),
        "inclusive": name in STAGES,
    }


def _add_kernels(target, kernels):
    target["kernel_ns"] += sum(k["end"] - k["start"] for k in kernels)
    target["kernel_ms"] = target["kernel_ns"] / 1e6
    target["kernel_count"] += len(kernels)


def _scope_kernels(scope, kernels, launches):
    return kernels[bisect_left(launches, scope["start"]) : bisect_left(launches, scope["end"])]


def _require_marker(kernels, launches, begin, end, marker):
    selected = kernels[bisect_right(launches, begin) : bisect_left(launches, end)]
    matches = [k for k in selected if marker.lower() in k["name"].lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {marker} kernel between its GEMM anchors")


def _attribute_run(root, scopes, kernels, num_layers, workload):
    kernels = sorted(kernels, key=lambda k: k["launch_start"])
    launches = [k["launch_start"] for k in kernels]
    calls = _workload_calls(root["phase"], num_layers, workload)
    grouped = defaultdict(lambda: defaultdict(list))
    for scope in scopes:
        key = (scope["layer_idx"], scope["query_start"], scope["query_length"])
        grouped[key][scope["stage"]].append(scope)
    if set(grouped) != set(calls):
        raise ValueError("NVTX layer/query calls do not match the complete workload")
    primary = sorted(
        (scope for scope in scopes if scope["stage"] in PRIMARY_STAGES),
        key=lambda scope: scope["start"],
    )
    if any(left["end"] > right["start"] for left, right in pairwise(primary)):
        raise ValueError("Primary layer scopes must not overlap")
    primary_starts = [scope["start"] for scope in primary]
    matrices = [kernel for kernel in kernels if _matrix_kernel(kernel)]
    if len(matrices) != 5 * len(calls):
        raise ValueError("Expected five GEMM kernels per layer/query call, four outside CIS")
    modules = {name: _empty_module(name) for name in MODULES}
    covered = set()
    layer_calls = []
    query_tile = _integer(
        workload.get("indexer_query_chunk_size", 64), "indexer_query_chunk_size", minimum=1
    )
    for call_index, key in enumerate(calls):
        group = grouped[key]
        if any(
            len(group[stage]) != 1
            for stage in (*PRIMARY_STAGES, "compression_k", "compression_cis")
        ):
            raise ValueError("Each layer call needs exactly one primary and two compression scopes")
        cis, indexer, attention = [group[name][0] for name in PRIMARY_STAGES]
        if not (cis["end"] <= indexer["start"] and indexer["end"] <= attention["start"]):
            raise ValueError("Expected CIS -> indexer -> block attention scope order")
        nested = sorted(
            (scope for stage in NESTED_STAGES for scope in group[stage]), key=lambda s: s["start"]
        )
        for scope in nested:
            if not (indexer["start"] <= scope["start"] < scope["end"] <= indexer["end"]):
                raise ValueError("Indexer child scope is not nested in its matching parent")
        if any(left["end"] > right["start"] for left, right in pairwise(nested)):
            raise ValueError("Indexer child scopes must not overlap")
        layer, start, length = key
        scored = (start + length + 63) // 64 > 64
        tiles = (length + query_tile - 1) // query_tile if scored else 0
        expected = ["compression_k", "compression_cis"] + [
            stage for _ in range(tiles) for stage in ("compressed_scores", "select_from_scores")
        ]
        if [scope["stage"] for scope in nested] != expected:
            raise ValueError(
                "Indexer child count/order disagrees with query tiling or short-context bypass"
            )
        per_call = {name: _empty_module(name) for name in MODULES if name != "other_non_matrix"}
        for stage in STAGES:
            for scope in group[stage]:
                selected = _scope_kernels(scope, kernels, launches)
                if not selected:
                    raise ValueError(f"Scope {stage} contains no correlated CUDA kernels")
                if stage == "block_sparse_attention" and (
                    len(selected) != 1 or "_nosa_block_attention" not in selected[0]["name"]
                ):
                    raise ValueError(
                        "Block attention scope must contain one _nosa_block_attention kernel"
                    )
                if (
                    stage == "compressed_scores"
                    and sum(k["name"] == "_scores" for k in selected) != 1
                ):
                    raise ValueError(
                        "Compressed score scope must contain exactly one _scores kernel"
                    )
                _add_kernels(modules[stage], selected)
                _add_kernels(per_call[stage], selected)
                if stage in PRIMARY_STAGES:
                    covered.update(k["correlationId"] for k in selected)
        gemms = dict(zip(LINEARS, matrices[5 * call_index : 5 * (call_index + 1)], strict=True))
        for name, kernel in gemms.items():
            position = bisect_right(primary_starts, kernel["launch_start"]) - 1
            owner = (
                primary[position]
                if position >= 0 and kernel["launch_start"] < primary[position]["end"]
                else None
            )
            if name == "cis_projection":
                if owner is not cis:
                    raise ValueError(
                        "Five-GEMM sequence has a CIS projection outside its matching CIS scope"
                    )
            elif owner is not None:
                raise ValueError("The four non-CIS GEMMs must be outside all primary scopes")
        if not (gemms["qkv_proj"]["launch_start"] < cis["start"]):
            raise ValueError("QKV GEMM must precede its CIS scope")
        if gemms["o_proj"]["launch_start"] < attention["end"]:
            raise ValueError("Output projection GEMM must follow its block attention scope")
        _require_marker(
            kernels,
            launches,
            gemms["qkv_proj"]["launch_start"],
            gemms["cis_projection"]["launch_start"],
            "BatchQKApplyRotary",
        )
        _require_marker(
            kernels,
            launches,
            gemms["o_proj"]["launch_start"],
            gemms["gate_up_proj"]["launch_start"],
            "fused_add_rmsnorm",
        )
        _require_marker(
            kernels,
            launches,
            gemms["gate_up_proj"]["launch_start"],
            gemms["down_proj"]["launch_start"],
            "act_and_mul",
        )
        for name, kernel in gemms.items():
            module = "cis_projection_gemm" if name == "cis_projection" else name
            _add_kernels(modules[module], [kernel])
            _add_kernels(per_call[module], [kernel])
            covered.add(kernel["correlationId"])
        layer_calls.append(
            {
                "layer_idx": layer,
                "query_start": start,
                "query_length": length,
                "modules": per_call,
                "gemm_correlations": {
                    name: kernel["correlationId"] for name, kernel in gemms.items()
                },
                "cis_scope": {name: cis[name] for name in ("start", "end")},
                "attention_scope": {name: attention[name] for name in ("start", "end")},
            }
        )
    other = [kernel for kernel in kernels if kernel["correlationId"] not in covered]
    if any(_matrix_kernel(kernel) for kernel in other):
        raise ValueError("Unattributed matrix kernel remains after layer sequence reconstruction")
    _add_kernels(modules["other_non_matrix"], other)
    total_ns = sum(kernel["end"] - kernel["start"] for kernel in kernels)
    partition = [row for name, row in modules.items() if name not in PARENTS]
    if sum(row["kernel_ns"] for row in partition) != total_ns or sum(
        row["kernel_count"] for row in partition
    ) != len(kernels):
        raise ValueError(
            "Exclusive module partition must conserve all correlated kernel time/counts"
        )
    return {
        "phase": root["phase"],
        "iteration": root["iteration"],
        "kernel_ms": total_ns / 1e6,
        "kernel_count": len(kernels),
        "modules": modules,
        "layer_calls": layer_calls,
        "root_scope": {name: root[name] for name in ("start", "end", "text", "globalTid")},
    }


def attribute_kernels(scopes, kernels, num_layers, workload):
    """Attribute launch-correlated kernels, independent of SQLite or model FLOPs.

    Times are integer nanoseconds. Scopes need start/end/text/globalTid;
    kernels need start/end/launch_start/globalTid/correlationId/name/deviceId/
    streamId. Optional globalPid/contextId are also checked for uniqueness.
    GPU start/end may lie outside the CPU NVTX interval: launch_start owns the
    attribution, while end-start supplies the measured denominator.

    Returns per-phase/per-iteration module dictionaries plus per-layer calls.
    The fixed source graph is an external precondition, verified by analyze().
    Synthetic workloads are supported here to test attribution independently.
    """
    _integer(num_layers, "num_layers", minimum=1)
    roots, stages = [], []
    for source in scopes:
        text = source.get("text", "")
        if not text.startswith("NOSA/"):
            continue
        scope = dict(source)
        _interval(scope, "NVTX scope")
        _integer(scope.get("globalTid"), "scope.globalTid")
        if match := ROOT_RANGE.fullmatch(text):
            scope.update(phase=match[1], iteration=int(match[2]))
            roots.append(scope)
        elif match := STAGE_RANGE.fullmatch(text):
            if match[3] not in STAGES:
                raise ValueError(f"Unknown NOSA stage: {match[3]}")
            scope.update(
                phase=match[1],
                layer_idx=int(match[2]),
                stage=match[3],
                query_start=int(match[4]),
                query_length=int(match[5]),
            )
            stages.append(scope)
        else:
            raise ValueError(f"Unrecognized NOSA NVTX range: {text}")
    roots.sort(key=lambda root: root["start"])
    if not roots or any(a["end"] > b["start"] for a, b in pairwise(roots)):
        raise ValueError("Expected nonoverlapping profile root ranges")
    counts = Counter(root["phase"] for root in roots)
    if set(counts) != set(PHASES) or len(set(counts.values())) != 1:
        raise ValueError("Both profile phases must contain the same complete run count")
    repeats = workload.get("profile_repeats", counts["extend"])
    _integer(repeats, "profile_repeats", minimum=1)
    for phase in PHASES:
        if sorted(root["iteration"] for root in roots if root["phase"] == phase) != list(
            range(repeats)
        ):
            raise ValueError("Profile root iterations must exactly cover profile_repeats")
    threads = {scope["globalTid"] for scope in (*roots, *stages)}
    if len(threads) != 1:
        raise ValueError("Module reconstruction requires exactly one CPU launch thread")
    kernels = [dict(kernel) for kernel in kernels]
    if not kernels:
        raise ValueError("No correlated CUDA kernels were supplied")
    for kernel in kernels:
        _interval(kernel, "kernel")
        for field in ("launch_start", "globalTid", "correlationId", "deviceId", "streamId"):
            _integer(kernel.get(field), f"kernel.{field}")
        if kernel["launch_start"] > kernel["start"]:
            raise ValueError("CUDA kernel begins before its correlated host launch")
        if not isinstance(kernel.get("name"), str) or not kernel["name"]:
            raise ValueError("CUDA kernel names are required")
    if {kernel["globalTid"] for kernel in kernels} != threads:
        raise ValueError("Kernels must originate from the profile's single launch thread")
    for field in ("deviceId", "streamId", "globalPid", "contextId"):
        values = {kernel.get(field) for kernel in kernels}
        if len(values) != 1:
            raise ValueError(f"Module reconstruction requires exactly one {field}")
    if len({kernel["correlationId"] for kernel in kernels}) != len(kernels):
        raise ValueError("Every kernel requires a unique launch correlationId")
    launch_order = sorted(kernels, key=lambda kernel: kernel["launch_start"])
    if any(
        a["launch_start"] == b["launch_start"] or a["start"] > b["start"]
        for a, b in pairwise(launch_order)
    ):
        raise ValueError("Single-stream GPU order must match the unambiguous host launch order")
    overlaps = []
    for left, right in pairwise(launch_order):
        overlap = max(0, left["end"] - right["start"])
        if overlap:
            if overlap > 1000:
                raise ValueError(
                    "GPU kernel overlap exceeds the small nsys timestamp-boundary tolerance"
                )
            overlaps.append(overlap)
    starts = [root["start"] for root in roots]
    root_scopes, root_kernels = defaultdict(list), defaultdict(list)
    for scope in stages:
        index = bisect_right(starts, scope["start"]) - 1
        if (
            index < 0
            or scope["end"] > roots[index]["end"]
            or scope["phase"] != roots[index]["phase"]
        ):
            raise ValueError("Every stage scope must belong to its matching profile root")
        root_scopes[index].append(scope)
    excluded = []
    for kernel in kernels:
        index = bisect_right(starts, kernel["launch_start"]) - 1
        if index >= 0 and kernel["launch_start"] < roots[index]["end"]:
            root_kernels[index].append(kernel)
        else:
            excluded.append(kernel)
    result = {
        "runs": [
            _attribute_run(root, root_scopes[index], root_kernels[index], num_layers, workload)
            for index, root in enumerate(roots)
        ],
        "validation": {
            "single_launch_thread": next(iter(threads)),
            "device_id": kernels[0]["deviceId"],
            "stream_id": kernels[0]["streamId"],
            "global_pid": kernels[0].get("globalPid"),
            "context_id": kernels[0].get("contextId"),
            "correlation_count": len(kernels),
            "excluded_outside_profile_roots": len(excluded),
            "overlap_count": len(overlaps),
            "overlap_ns": sum(overlaps),
            "max_overlap_ns": max(overlaps, default=0),
            "overlap_tolerance": "<=1000 ns on the verified single CUDA stream",
            "gemm_order": list(LINEARS),
            "time_definition": "Sum of kernel end-start; not union active time or CUDA-event span",
        },
    }
    return result


def read_trace(path):
    """Read an exported SQLite trace without writing indexes or modifying it."""
    path = Path(path).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        scopes = [
            dict(row)
            for row in connection.execute(
                "SELECT n.start,n.end,COALESCE(n.text,s.value) AS text,n.globalTid "
                "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
                "WHERE COALESCE(n.text,s.value) LIKE 'NOSA/%'"
            )
        ]
        kernels = [
            dict(row)
            for row in connection.execute(
                "SELECT k.start,k.end,r.start AS launch_start,r.globalTid,k.correlationId,"
                "s.value AS name,k.deviceId,k.streamId,k.contextId,k.globalPid "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId "
                "LEFT JOIN StringIds s ON k.demangledName=s.id ORDER BY r.start"
            )
        ]
    return scopes, kernels


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_trace_gpu(path, device_id, gpu):
    """Bind the actual kernel device in SQLite to the measured metadata GPU."""
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id,uuid,name,smCount,computeMajor,computeMinor FROM TARGET_INFO_GPU WHERE id=?",
            (device_id,),
        ).fetchall()
    if len(rows) != 1:
        raise ValueError("SQLite must identify exactly one GPU for the kernel device ID")
    device = dict(rows[0])

    def uuid(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("GPU UUID is required in SQLite and metadata")
        return value.lower().removeprefix("gpu-")

    if uuid(device["uuid"]) != uuid(gpu.get("uuid")):
        raise ValueError("SQLite GPU UUID differs from benchmark metadata")
    if device["smCount"] != gpu.get("sm_count") or [
        device["computeMajor"],
        device["computeMinor"],
    ] != gpu.get("capability"):
        raise ValueError("SQLite GPU architecture differs from benchmark metadata")
    return device


def _validate_inputs(data_dir, metadata, summary, profile_metadata):
    if profile_metadata.get("run_id") != metadata.get("run_id"):
        raise ValueError("Profile and benchmark run_id must agree")
    if profile_metadata.get("args", {}).get("mode") != "profile":
        raise ValueError("Expected profile-mode metadata")
    for key in (
        "model_config",
        "source_sha256",
        "request_sha256",
        "checkpoint_config_sha256",
        "gpu",
        "torch",
        "cuda",
        "triton",
        "flashinfer",
    ):
        if profile_metadata.get(key) != metadata.get(key):
            raise ValueError(f"Profile and benchmark metadata disagree on {key}")
    for key in ("prefix_tokens", "new_tokens", "chunk_size", "profile_repeats", "device"):
        if profile_metadata["args"][key] != metadata["args"][key]:
            raise ValueError(f"Profile and benchmark arguments disagree on {key}")
    if profile_metadata.get("validation", {}).get("finite") is not True:
        raise ValueError("The captured profile must have passed its finite-output check")
    hashes = metadata["source_sha256"]
    for name, expected in SUPPORTED_GRAPH.items():
        if hashes.get(name) != expected:
            raise ValueError(f"Unreviewed inference graph for unscoped GEMM attribution: {name}")
    verified = {}
    for name, expected in hashes.items():
        path = (data_dir / "sources" / name).resolve()
        if not path.is_relative_to((data_dir / "sources").resolve()) or _sha256(path) != expected:
            raise ValueError(f"Captured source snapshot hash mismatch: {name}")
        verified[name] = expected
    config = metadata["model_config"]
    if config.get("attention_bias") or config.get("mlp_bias") or config.get("tie_word_embeddings"):
        raise ValueError("Only the reviewed bias-free NOSA inference graph is supported")
    # Shapes come from the captured model configuration and the independent
    # untimed resident-geometry audit, never from a GEMM kernel's name/grid.
    audit = json.loads((data_dir / "attention_audit.json").read_text())
    calls = [record for record in audit if record["stage"] == "block_sparse_attention"]
    if sorted(record["layer_idx"] for record in calls) != list(range(config["num_hidden_layers"])):
        raise ValueError("Geometry audit must cover every layer exactly once")
    workload = summary["workload"]
    heads, kv_heads, dim = (
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
    )
    for record in calls:
        details = record["details"]
        expected = {
            "q_shape": [workload["new_tokens"], heads, dim],
            "k_shape": [workload["total_tokens"], kv_heads, dim],
            "v_shape": [workload["total_tokens"], kv_heads, dim],
            "selection_shape": [workload["new_tokens"], kv_heads, 64],
            "block_size": 64,
            "block_budget": 64,
            "valid_blocks_min": 64,
            "valid_blocks_max": 64,
        }
        if record.get("status") != "ok" or record.get("phase") != "extend":
            raise ValueError("Geometry audit must contain successful extend calls")
        if (
            record["query_start"] != workload["prefix_tokens"]
            or record["query_length"] != workload["new_tokens"]
        ):
            raise ValueError("Geometry audit query range differs from the workload")
        if any(details.get(name) != value for name, value in expected.items()):
            raise ValueError(
                "Resident Q/K/V/selection shape or block budget audit disagrees with config"
            )
    return verified


def _flops_by_module(config, prefix, query, chunk_size):
    flops = matrix_flops(config, prefix, query, chunk_size)
    return {
        **{name: flops.get(name, 0) for name in MODULES},
        "indexer_total": flops["indexer_qk"],
        "compressed_scores": flops["indexer_qk"],
        "cis_projection_gemm": flops["cis_projection"],
    }


def _add_mfu(module, flops, peak):
    module["matrix_flops"] = flops
    if flops:
        if module["kernel_ms"] <= 0:
            raise ValueError("Positive matrix work requires a positive measured kernel duration")
        module["effective_tflops"] = flops / module["kernel_ms"] / 1e9
        module["mfu_pct"] = module["effective_tflops"] / peak * 100
    else:
        module["effective_tflops"] = module["mfu_pct"] = None


def build_module_report(metadata, summary, profile_metadata, attribution, *, peak_tflops=None):
    """Attach useful matrix FLOPs to each attributed run and preserve E2E MFU."""
    baseline = build_report(metadata, summary, peak_tflops=peak_tflops)
    peak = baseline["peak_tflops"]
    config, workload = metadata["model_config"], summary["workload"]
    runs = attribution["runs"]
    repeats = _integer(workload["profile_repeats"], "profile_repeats", minimum=1)
    for phase in PHASES:
        if (
            sum(run["phase"] == phase for run in runs) != repeats
            or summary["profiles"][phase]["sample_count"] != repeats
            or len(profile_metadata["instrumented_timings"][phase]) != repeats
        ):
            raise ValueError("SQLite profile run count differs from the validated summary")
    for run in runs:
        phase = run["phase"]
        prefix, query = (
            (0, workload["total_tokens"])
            if phase == "full_prefill"
            else (workload["prefix_tokens"], workload["new_tokens"])
        )
        flops = _flops_by_module(config, prefix, query, workload["chunk_size"])
        for name, module in run["modules"].items():
            _add_mfu(module, flops[name], peak)
        for call in run["layer_calls"]:
            one_layer = config | {"num_hidden_layers": 1}
            call_flops = _flops_by_module(
                one_layer, call["query_start"], call["query_length"], workload["chunk_size"]
            )
            for name, module in call["modules"].items():
                _add_mfu(module, call_flops[name], peak)
            length, dim = call["query_length"], config["head_dim"]
            q_width, kv_width = (
                config["num_attention_heads"] * dim,
                config["num_key_value_heads"] * dim,
            )
            call["matrix_shapes_m_n_k"] = {
                "qkv_proj": [length, q_width + 2 * kv_width, config["hidden_size"]],
                "cis_projection": [length, config["num_key_value_heads"], kv_width],
                "o_proj": [length, config["hidden_size"], q_width],
                "gate_up_proj": [length, 2 * config["intermediate_size"], config["hidden_size"]],
                "down_proj": [length, config["hidden_size"], config["intermediate_size"]],
            }
        # Sum per-layer/chunk FLOPs to audit both causal work and inferred shapes.
        for name in run["modules"]:
            if (
                name != "other_non_matrix"
                and sum(call["modules"][name]["matrix_flops"] for call in run["layer_calls"])
                != flops[name]
            ):
                raise ValueError("Layer/chunk matrix FLOPs do not conserve the phase work count")
    phases = {}
    for phase in PHASES:
        selected = [run for run in runs if run["phase"] == phase]
        modules = {}
        for name in MODULES:
            values = [run["modules"][name] for run in selected]
            kernel_ms_median = statistics.median(value["kernel_ms"] for value in values)
            modules[name] = {
                "sample_count": len(values),
                "parent_module": PARENTS.get(name),
                "inclusive": name in STAGES,
                "matrix_flops_per_run": values[0]["matrix_flops"],
                "kernel_ms_median": kernel_ms_median,
                "kernel_count_per_run": [value["kernel_count"] for value in values],
                "mfu_pct_median": values[0]["matrix_flops"] / kernel_ms_median / 1e9 / peak * 100
                if values[0]["mfu_pct"] is not None
                else None,
            }
            if name in STAGES:
                interval = summary["profiles"][phase]["stage_totals"][name]["cuda_elapsed_ms"][
                    "median"
                ]
                modules[name]["event_interval_ms_median"] = interval
                modules[name]["event_interval_mfu_pct"] = (
                    values[0]["matrix_flops"] / interval / 1e9 / peak * 100
                    if values[0]["matrix_flops"] and interval > 0
                    else None
                )
        phases[phase] = {"modules": modules, "end_to_end": baseline["phases"][phase]}
    return {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "gpu": baseline["gpu"],
        "peak_tflops": peak,
        "peak_source": baseline["peak_source"],
        "peak_kind": baseline["peak_kind"],
        "workload": workload,
        "dimensions": baseline["dimensions"],
        "runs": runs,
        "phases": phases,
        "validation": attribution["validation"],
        "definitions": baseline["definitions"]
        | {
            "mfu_pct": "100 * useful matrix FLOPs / (sum of attributed GPU kernel durations in seconds * peak FLOP/s)",
            "kernel_attribution": "Host CUDA launch correlation within NVTX; GPU timestamps supply durations only",
            "unscoped_gemms": "Reviewed runtime source order [QKV,CIS,O,gate_up,down] plus CIS/attention/RoPE/norm/activation anchors",
            "kernel_shape_proof": "Captured config and source graph, audited resident Q/K/V shapes and complete layer/chunk counts; kernel names/grids are not shape evidence",
            "no_matrix_modules": "Compression, selection and other non-matrix modules have MFU null, not zero",
            "nested_modules": "Indexer children and cis_projection_gemm are inclusive subsets; never add them to their parents",
            "scope_denominator": "CIS/indexer/attention scopes include all correlated kernels, including non-matrix work; cis_projection_gemm is an additional delta-only view",
            "event_interval_metric": "Secondary recorded CUDA-event interval MFU includes submission/launch gaps and is distinct from kernel-duration MFU",
            "profile_limit": "Kernel metrics come from instrumented nsys runs; end-to-end MFU remains the independent uninstrumented wall-time result",
        },
    }


def _markdown(report):
    lines = [
        "# NOSA module MFU from nsys kernel durations",
        "",
        f"Run: `{report['run_id']}`. Dense BF16 peak: {report['peak_tflops']:g} TFLOP/s.",
        "",
        (
            "Times sum actual correlated GPU kernel durations, not NVTX host ranges or CUDA-event intervals. "
            "Nested modules overlap their parents and must not be added. Compression/selection have no matrix MFU."
        ),
        "",
        "| Phase | Module | Parent | Kernel ms (median) | Kernels/run | Useful matrix FLOPs/run | MFU % |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for phase in PHASES:
        for name, row in report["phases"][phase]["modules"].items():
            mfu = "—" if row["mfu_pct_median"] is None else f"{row['mfu_pct_median']:.4f}"
            lines.append(
                f"| {phase} | {name} | {row['parent_module'] or '—'} | {row['kernel_ms_median']:.6f} | {','.join(map(str, row['kernel_count_per_run']))} | {row['matrix_flops_per_run']} | {mfu} |"
            )
    lines.extend(
        [
            "",
            "Kernel duration sum is not union active time. Timestamp-boundary overlap is audited in JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(data_dir, *, sqlite_path=None, peak_tflops=None, write=True):
    """Derive new output files; leave all captured measurements/source snapshots intact."""
    data_dir = Path(data_dir).resolve()
    sqlite_path = (
        Path(sqlite_path).resolve() if sqlite_path is not None else data_dir / "nsys.sqlite"
    )
    names = ("metadata.json", "summary.json", "profile_metadata.json", "attention_audit.json")
    inputs = {name: (data_dir / name).read_bytes() for name in names}
    metadata, summary, profile_metadata = [json.loads(inputs[name]) for name in names[:3]]
    sources = _validate_inputs(data_dir, metadata, summary, profile_metadata)
    scopes, kernels = read_trace(sqlite_path)
    attribution = attribute_kernels(
        scopes, kernels, metadata["model_config"]["num_hidden_layers"], summary["workload"]
    )
    attribution["validation"]["trace_gpu"] = validate_trace_gpu(
        sqlite_path, attribution["validation"]["device_id"], metadata["gpu"]
    )
    report = build_module_report(
        metadata, summary, profile_metadata, attribution, peak_tflops=peak_tflops
    )
    root = Path(__file__).resolve().parents[3]
    report["provenance"] = {
        "derived_at_utc": datetime.now(UTC).isoformat(),
        "input_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()},
        "sqlite": {
            "path": str(sqlite_path),
            "bytes": sqlite_path.stat().st_size,
            "sha256": _sha256(sqlite_path),
        },
        "captured_source_sha256": sources,
        "analysis_source_sha256": {
            name: _sha256(root / name)
            for name in (
                "experiments/indexer_block_sparse_profile/src/module_mfu.py",
                "experiments/indexer_block_sparse_profile/src/mfu.py",
                "experiments/indexer_block_sparse_profile/src/analyze.py",
                "experiments/nosa_gr_65536_1024/src/mfu.py",
            )
        },
    }
    if write:
        (data_dir / "module_mfu.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        fields = (
            "phase",
            "iteration",
            "module",
            "parent_module",
            "inclusive",
            "kernel_ms",
            "kernel_count",
            "matrix_flops",
            "effective_tflops",
            "mfu_pct",
        )
        with (data_dir / "module_mfu.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for run in report["runs"]:
                for name, row in run["modules"].items():
                    writer.writerow(
                        {
                            field: (
                                {
                                    "phase": run["phase"],
                                    "iteration": run["iteration"],
                                    "module": name,
                                }
                                | row
                            ).get(field)
                            for field in fields
                        }
                    )
        (data_dir / "module_mfu.md").write_text(_markdown(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Completed benchmark/profile run directory")
    parser.add_argument(
        "--sqlite", type=Path, help="nsys SQLite export; default data_dir/nsys.sqlite"
    )
    parser.add_argument(
        "--peak-tflops", type=float, help="Dense BF16 peak; defaults to 989 for H200"
    )
    args = parser.parse_args(argv)
    report = analyze(args.data_dir, sqlite_path=args.sqlite, peak_tflops=args.peak_tflops)
    print(json.dumps({phase: report["phases"][phase]["modules"] for phase in PHASES}, indent=2))


if __name__ == "__main__":
    main()
