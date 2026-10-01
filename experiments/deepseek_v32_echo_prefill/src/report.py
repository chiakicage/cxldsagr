"""Validate and report complete checkpoint ECHO prefill/extend measurements.

No report is written until shape, repetitions, annotated layer coverage,
cache-byte accounting, archived sources, input, and saved logits pass the
publication gates. Annotated CUDA scope durations remain separate from the
uninstrumented end-to-end timing samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LAYERS, PREFIX, EXTEND, RECORD_BYTES = 61, 65536, 1024, 1152
MODES = ("resident", "offload")
PHASES = ("prefill", "extend")
EXPERIMENT = "deepseek_v32_echo_prefill"
BEGIN = "<!-- BEGIN ECHO GENERATED RESULTS -->"
END = "<!-- END ECHO GENERATED RESULTS -->"
LEGACY_REQUIRED_SOURCES = frozenset(
    {
        "models/deepseek_v32/echo_model.py",
        "models/deepseek_v32/echo_block.py",
        "models/deepseek_v32/echo_attention.py",
        "models/deepseek_v32/echo_infer.py",
        "cache/sparse_token_cache.py",
        "operators/sm90/deepseek_linear.py",
        "operators/sm90/deepseek_mla.py",
        "operators/sm90/echo_indexer.py",
        "operators/sm90/kv_transfer.py",
        "operators/sm90/csrc/echo_helpers.cuh",
        "operators/sm90/csrc/echo_indexer.cu",
        "operators/sm90/csrc/echo_logits.cuh",
        "operators/sm90/csrc/kv_transfer.cu",
        f"experiments/{EXPERIMENT}/src/measure.py",
    }
)
REQUIRED_SOURCES = frozenset(
    {
        "models/deepseek_v32/echo_model.py",
        "models/deepseek_v32/echo_block.py",
        "models/deepseek_v32/echo_attention.py",
        "models/deepseek_v32/echo_infer.py",
        "cache/sparse_token_cache.py",
        "operators/deepseek_v32/linear/fp8.py",
        "operators/deepseek_v32/attention/_validation.py",
        "operators/deepseek_v32/attention/device_only/mla.py",
        "operators/deepseek_v32/attention/offload/mla.py",
        "operators/deepseek_v32/indexer/echo.py",
        "operators/common/kv_transfer.py",
        "operators/deepseek_v32/indexer/csrc/echo_helpers.cuh",
        "operators/deepseek_v32/indexer/csrc/echo_indexer.cu",
        "operators/deepseek_v32/indexer/csrc/echo_logits.cuh",
        "operators/common/csrc/kv_transfer.cu",
        f"experiments/{EXPERIMENT}/src/measure.py",
    }
)


class InvalidResult(ValueError):
    """The supplied run does not meet this experiment's publication boundary."""


def require(condition, message):
    if not condition:
        raise InvalidResult(message)


def integer(value, name, *, minimum=0):
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def number(value, name, *, positive=False):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        f"{name} must be finite and {'positive' if positive else 'nonnegative'}",
    )
    return float(value)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidResult(f"Cannot read required JSON {path}: {exc}") from exc


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_sources_and_request(result, directory):
    sources = result.get("source_sha256")
    require(
        isinstance(sources, dict)
        and any(
            required <= sources.keys() for required in (LEGACY_REQUIRED_SOURCES, REQUIRED_SOURCES)
        ),
        "Required implementation snapshots are missing",
    )
    require(
        read_json(directory / "sources.json") == sources,
        "sources.json differs from result source_sha256",
    )
    archive = (directory / "source").resolve()
    for relative, digest in sources.items():
        require(
            isinstance(relative, str)
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "Unsafe source snapshot path",
        )
        require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            f"Invalid source SHA256: {relative}",
        )
        path = (archive / relative).resolve()
        require(
            path.is_relative_to(archive) and path.is_file(), f"Missing source snapshot: {relative}"
        )
        require(sha256(path) == digest, f"Source snapshot SHA256 mismatch: {relative}")
    request_path = directory / "request.json"
    request = read_json(request_path)
    request_bytes = request_path.read_bytes().removesuffix(b"\n")
    require(
        hashlib.sha256(request_bytes).hexdigest() == result.get("request_sha256"),
        "Request SHA256 mismatch",
    )
    require(request.get("model") == "deepseek_v32", "Request must use DeepSeek V3.2")
    require(
        isinstance(request.get("prompt"), str) and bool(request["prompt"].strip()),
        "Readable request prompt is missing",
    )
    ids = request.get("input_ids")
    require(
        isinstance(ids, list) and len(ids) == PREFIX + EXTEND,
        "Request token count must be 65536 + 1024",
    )
    require(
        all(type(token) is int and 0 <= token < 129280 for token in ids),
        "Invalid checkpoint token IDs",
    )
    require(
        request.get("attention_mask") == [1] * len(ids),
        "Request contains padding or a nontrivial attention mask",
    )
    require(
        request.get("stable_prefix_tokens") == PREFIX
        and request.get("candidate_suffix_tokens") == EXTEND,
        "Request history/extend boundaries differ from measurement",
    )
    require(request.get("total_input_tokens") == len(ids), "Request total_input_tokens mismatch")
    instruction = integer(request.get("instruction_tokens"), "instruction_tokens", minimum=1)
    require(instruction < PREFIX, "Instruction consumes the history")
    require(request.get("history_token_span") == [instruction, PREFIX], "History span mismatch")
    require(
        request.get("candidate_token_span") == [PREFIX, PREFIX + EXTEND], "Candidate span mismatch"
    )
    require(
        isinstance(request.get("history_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", request["history_sha256"]),
        "History provenance digest is missing",
    )
    return sources


def validate_annotation(annotation, *, mode, phase, placement, chunks):
    label = f"{mode}/{phase} annotated"
    number(annotation.get("wall_ms"), label + " wall_ms", positive=True)
    calls = annotation.get("stage_calls")
    stages = annotation.get("stages_cuda_exclusive_ms")
    require(
        isinstance(calls, list) and calls and isinstance(stages, dict) and stages,
        label + " stage events are missing",
    )
    totals = defaultdict(float)
    counts = Counter()
    devices = {int(device.split(":")[1]) for device in placement}
    for call in calls:
        stage, layer = call.get("stage"), call.get("layer")
        require(isinstance(stage, str) and bool(stage), label + " empty stage name")
        require(
            type(call.get("device")) is int and call["device"] in devices,
            label + " event has unmapped device",
        )
        if layer != "shared":
            require(
                isinstance(layer, str) and re.fullmatch(r"layer_\d+", layer),
                label + " invalid layer label",
            )
            index = int(layer[6:])
            require(
                index < LAYERS and placement[index] == f"cuda:{call['device']}",
                label + " layer/device mismatch",
            )
        inclusive = number(call.get("cuda_inclusive_ms"), label + " cuda_inclusive_ms")
        exclusive = number(call.get("cuda_exclusive_ms"), label + " cuda_exclusive_ms")
        number(call.get("host_ms"), label + " host_ms")
        require(
            exclusive <= inclusive + 1e-6, label + " exclusive CUDA time exceeds inclusive time"
        )
        totals[stage] += exclusive
        counts[layer, stage] += 1
    require(stages.keys() == totals.keys(), label + " stage total keys mismatch")
    for stage, total in stages.items():
        number(total, label + " stage total")
        require(
            math.isclose(total, totals[stage], rel_tol=1e-9, abs_tol=1e-5),
            label + f" stage total mismatch: {stage}",
        )
    for layer in range(LAYERS):
        name = f"layer_{layer}"
        required = [
            "input_residual_norm",
            "attention_projection",
            "cache_write",
            "offload_prepare",
            "exact_topk",
            "attention_output",
            "post_attention_residual_norm",
            "indexer_prefetch" if mode == "offload" else "indexer",
        ]
        required += (
            ["dense_mlp"]
            if layer < 3
            else ["moe", "moe_routing", "moe_experts", "moe_shared_experts"]
        )
        for stage in required:
            require(
                counts[name, stage] == chunks,
                label + f" incomplete {name}/{stage}: expected {chunks} calls",
            )
        require(counts[name, "sparse_mla"] >= chunks, label + f" missing {name}/sparse_mla")
        if mode == "offload":
            require(
                counts[name, "offload_exact_recall"] == 2 * counts[name, "sparse_mla"] - chunks,
                label + f" {name} exact-recall event count omits or duplicates split attempts",
            )
    require(counts["shared", "embedding"] == chunks, label + " embedding coverage mismatch")
    require(
        counts["shared", "final_norm_lm_head"] == 1,
        label + " final norm / LM head coverage mismatch",
    )
    require(
        counts["shared", "hidden_transfer"] == chunks * LAYERS,
        label + " layer transfer coverage mismatch",
    )
    return dict(totals)


def validate_cache(rows, *, mode, phase, slots):
    require(
        isinstance(rows, list) and len(rows) == LAYERS, f"{mode}/{phase}: expected 61 cache rows"
    )
    written = PREFIX if phase == "prefill" else EXTEND
    capacity = PREFIX + EXTEND
    expected_slots = slots if mode == "offload" else capacity
    totals = defaultdict(int)
    fields = (
        "written_records",
        "recalled_records",
        "evicted_records",
        "max_working_set",
        "prefetched_records",
        "host_to_device_bytes",
        "device_to_host_bytes",
        "device_record_bytes",
        "host_record_bytes",
        "record_bytes",
        "device_slots",
    )
    for layer, row in enumerate(rows):
        label = f"{mode}/{phase}/layer_{layer} cache"
        for field in fields:
            integer(row.get(field), label + " " + field)
        require(row["record_bytes"] == RECORD_BYTES, label + " must use 1152-byte BF16 records")
        require(row["written_records"] == written, label + " history/extend writes are incomplete")
        require(row["device_slots"] == expected_slots, label + " device slot count mismatch")
        require(
            row["device_record_bytes"] == expected_slots * RECORD_BYTES,
            label + " device bytes mismatch",
        )
        require(
            row["host_record_bytes"] == (capacity * RECORD_BYTES if mode == "offload" else 0),
            label + " host bytes mismatch",
        )
        require(
            row["host_to_device_bytes"]
            == (row["prefetched_records"] + row["recalled_records"]) * RECORD_BYTES,
            label + " H2D byte accounting mismatch",
        )
        require(
            row["device_to_host_bytes"] == (written * RECORD_BYTES if mode == "offload" else 0),
            label + " D2H byte accounting mismatch",
        )
        require(row["max_working_set"] <= expected_slots, label + " working set exceeds capacity")
        if mode == "resident":
            require(
                all(
                    row[field] == 0
                    for field in ("prefetched_records", "recalled_records", "evicted_records")
                ),
                label + " resident path reports offload traffic",
            )
        else:
            require(
                row["prefetched_records"] + row["recalled_records"] >= written,
                label + " missing real offload transfers",
            )
        for field in fields:
            if field not in ("record_bytes", "device_slots", "max_working_set"):
                totals[field] += row[field]
    require(totals["written_records"] == written * LAYERS, "Total layer writes mismatch")
    require(
        totals["device_record_bytes"] == expected_slots * RECORD_BYTES * LAYERS,
        "Total HBM cache allocation mismatch",
    )
    return dict(totals)


def validate_logits(result, directory):
    correctness = result.get("correctness", {})
    require(
        correctness.get("bitwise_equal") is True,
        "Publication requires bitwise_equal=true; investigate differing logits",
    )
    require(correctness.get("same_next_token") is True, "Resident/offload next token differs")
    for name in ("max_abs", "nrmse"):
        require(
            number(correctness.get(name), "correctness " + name) == 0,
            "Bitwise equality conflicts with nonzero error",
        )
    import torch

    tensors = []
    for mode in MODES:
        path = directory / f"{mode}_logits.pt"
        require(path.is_file(), f"Saved {mode} logits are missing")
        value = torch.load(path, map_location="cpu", weights_only=True)
        require(
            isinstance(value, torch.Tensor) and tuple(value.shape) == (1, 129280),
            f"{mode} logits must cover the checkpoint vocabulary for the last token",
        )
        require(
            value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
            f"{mode} logits are invalid",
        )
        tensors.append(value)
    require(
        torch.equal(*tensors), "Saved resident/offload logits differ despite correctness metadata"
    )


def validate_result(path):
    """Return validated source data plus compact phase summaries, without writes."""
    path = Path(path)
    result = read_json(path)
    require(result.get("accepted") is True, "Run has not been accepted")
    run_id = result.get("run_id")
    require(
        isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id),
        "Invalid run ID",
    )
    require(path.parent.name == run_id, "Run directory and run ID differ")
    require(
        result.get("scope")
        == "complete_checkpoint_61_transformer_layers_embedding_final_norm_lm_head",
        "The result does not cover the complete checkpoint",
    )
    require(
        result.get("layers") == LAYERS
        and result.get("prefix_tokens") == PREFIX
        and result.get("extend_tokens") == EXTEND,
        "Report requires 61 layers and 65536 + 1024 tokens",
    )
    require(result.get("logits") == "last_token_only", "Unsupported output selection boundary")
    require(
        result.get("cache_record") == "BF16_512_latent_plus_64_RoPE_1152_bytes",
        "KV precision/layout differs from ECHO experiment",
    )
    require(
        result.get("checkpoint_weights")
        == "resident_block_FP8_with_BF16_MLA_absorbed_KV_projections",
        "Checkpoint precision/residency differs from experiment",
    )
    require(
        result.get("indexer") == "64x128_FP8_after_normalized_Hadamard_top2048",
        "Indexer semantics differ from experiment",
    )
    require(
        result.get("offload")
        == "mapped_pinned_local_DRAM_fused_indexer_prefetch_then_exact_recall",
        "Unsupported offload backend",
    )
    placement = result.get("placement")
    require(
        isinstance(placement, list)
        and len(placement) == LAYERS
        and all(
            isinstance(device, str) and re.fullmatch(r"cuda:\d+", device) for device in placement
        ),
        "Every transformer layer must map to a CUDA device",
    )
    devices = list(dict.fromkeys(placement))
    hardware = result.get("hardware")
    require(
        isinstance(hardware, list)
        and len(hardware) == len(devices)
        and all(isinstance(item, str) and item for item in hardware),
        "Hardware inventory and device placement differ",
    )
    dependencies = result.get("dependencies", {})
    require(
        all(
            isinstance(dependencies.get(name), str) and dependencies[name]
            for name in ("torch", "triton", "safetensors", "apache-tvm-ffi")
        ),
        "Dependency versions are missing",
    )
    integer(result.get("warmups"), "warmups", minimum=1)
    repeats = {
        "prefill": integer(result.get("prefill_repeats"), "prefill_repeats", minimum=1),
        "extend": integer(result.get("repeats"), "repeats", minimum=1),
    }
    chunk_size = integer(result.get("chunk_size"), "chunk_size", minimum=1)
    slots = integer(result.get("slots"), "slots", minimum=2048)
    require(
        chunk_size <= slots < PREFIX + EXTEND,
        "Offload must use a bounded pool fitting the query chunk",
    )
    validate_sources_and_request(result, path.parent)
    validate_logits(result, path.parent)
    measurements = result.get("measurements")
    require(
        isinstance(measurements, dict) and set(measurements) == set(MODES),
        "Both resident and offload measurements are required",
    )
    summaries = []
    for mode in MODES:
        measurement = measurements[mode]
        for phase in PHASES:
            label = f"{mode}/{phase}"
            rows = measurement.get(phase)
            require(
                isinstance(rows, list) and len(rows) == repeats[phase],
                label + " repetition count mismatch",
            )
            walls = []
            peaks = []
            for row in rows:
                walls.append(number(row.get("wall_ms"), label + " wall_ms", positive=True))
                peak = row.get("peak_allocated_bytes")
                require(
                    isinstance(peak, list) and len(peak) == len(devices),
                    label + " GPU peak memory coverage mismatch",
                )
                peaks.append([integer(value, label + " peak memory", minimum=1) for value in peak])
                require(
                    "stage_calls" not in row and "stages_cuda_exclusive_ms" not in row,
                    label + " annotated samples cannot be used as uninstrumented latency",
                )
            median = statistics.median(walls)
            number(measurement.get(phase + "_median_ms"), label + " reported median", positive=True)
            require(
                math.isclose(measurement[phase + "_median_ms"], median, rel_tol=1e-12),
                label + " median mismatch",
            )
            tokens = PREFIX if phase == "prefill" else EXTEND
            annotation = measurement.get("annotated_" + phase, {})
            stage_totals = validate_annotation(
                annotation,
                mode=mode,
                phase=phase,
                placement=placement,
                chunks=math.ceil(tokens / chunk_size),
            )
            cache = validate_cache(
                measurement.get(
                    "prefill_cache_per_layer" if phase == "prefill" else "cache_per_layer"
                ),
                mode=mode,
                phase=phase,
                slots=slots,
            )
            summaries.append(
                {
                    "run_id": run_id,
                    "mode": mode,
                    "phase": phase,
                    "tokens": tokens,
                    "repeats": len(walls),
                    "wall_median_ms": median,
                    "wall_min_ms": min(walls),
                    "wall_max_ms": max(walls),
                    "wall_stdev_ms": statistics.stdev(walls) if len(walls) > 1 else None,
                    "tokens_per_second": tokens * 1000 / median,
                    "peak_allocated_bytes_by_device": {
                        device: max(row[i] for row in peaks) for i, device in enumerate(devices)
                    },
                    "annotated_wall_ms": annotation["wall_ms"],
                    "annotated_cuda_exclusive_ms_by_stage": stage_totals,
                    "annotated_event_count": len(annotation["stage_calls"]),
                    "annotated_cache_totals": cache,
                    "wall_samples_ms": walls,
                }
            )
    return result, summaries


def make_summary(result_path):
    result, rows = validate_result(result_path)
    return {
        "run_id": result["run_id"],
        "accepted": True,
        "layers": LAYERS,
        "prefix_tokens": PREFIX,
        "extend_tokens": EXTEND,
        "warmups": result["warmups"],
        "chunk_size": result["chunk_size"],
        "slots": result["slots"],
        "placement": result["placement"],
        "hardware": result["hardware"],
        "dependencies": result["dependencies"],
        "correctness": result["correctness"],
        "measurement_boundaries": {
            "wall": "Uninstrumented complete model forward; checkpoint load, compilation and prefix state restoration excluded.",
            "annotated": "Separate instrumented samples. CUDA-event scopes subtract child scopes only; cross-device overlap, stream dependency waits and host launch gaps can remain. These elapsed intervals are not summed kernel durations, and their sums are not uninstrumented end-to-end latency.",
            "hidden_transfer": "Destination-stream transfer/dependency-wait intervals; these include source-computation dependencies and do not measure isolated NVLink copy time or bandwidth.",
            "indexer_prefetch": "Combined indexer computation and fused prefetch; no independent prefetch duration is inferred.",
            "cache_bytes": "Annotated phase counters. Main-KV HBM record allocation excludes weights, indexer cache, maps, scratch and activations. H2D bytes count prefetched records plus all gathered records; D2H bytes count new records written to backing storage.",
            "recalled_records": "All gathered H2D records, including current-chunk staging during offload_prepare and residual exact recall; this counter is not residual recall alone.",
            "prefix": "Every layer writes and computes the 65536-token prefix from an empty cache.",
            "extend": "Each 1024-token repeat restores the same prepared prefix, including offload HBM residency.",
        },
        "provenance": {
            "result_path": str(Path(result_path)),
            "result_sha256": sha256(result_path),
            "request_sha256": result["request_sha256"],
            "source_sha256": result["source_sha256"],
            "report_source_sha256": sha256(__file__),
            "saved_logits_sha256": {
                mode: sha256(Path(result_path).parent / f"{mode}_logits.pt") for mode in MODES
            },
        },
        "phases": rows,
    }


def write_csv(summary, path):
    fields = [
        "run_id",
        "phase",
        "mode",
        "tokens",
        "repeats",
        "wall_median_ms",
        "wall_min_ms",
        "wall_max_ms",
        "wall_stdev_ms",
        "tokens_per_second",
        "annotated_wall_ms",
        "device_record_bytes",
        "host_record_bytes",
        "host_to_device_bytes",
        "device_to_host_bytes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in summary["phases"]:
            writer.writerow(
                {name: row.get(name, row["annotated_cache_totals"].get(name)) for name in fields}
            )


def plot(summary, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {(row["mode"], row["phase"]): row for row in summary["phases"]}
    colors = ("#3176a8", "#d17b36")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, phase in zip(axes, PHASES):
        for i, mode in enumerate(MODES):
            row = rows[mode, phase]
            median = row["wall_median_ms"] / 1000
            ax.bar(i, median, color=colors[i], width=0.6, alpha=0.85)
            ax.errorbar(
                i,
                median,
                yerr=[[median - row["wall_min_ms"] / 1000], [row["wall_max_ms"] / 1000 - median]],
                color="#222222",
                capsize=6,
            )
            ax.scatter(
                [i] * row["repeats"],
                [value / 1000 for value in row["wall_samples_ms"]],
                color="#222222",
                s=16,
                zorder=4,
            )
            ax.text(
                i,
                row["wall_max_ms"] / 1000,
                f"{median:.3f} s\nn={row['repeats']}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(range(2), ["Resident", "Offload"])
        ax.set_ylabel("Uninstrumented wall time (s)")
        ax.set_title("65,536-token prefix" if phase == "prefill" else "1,024-token extend")
        ax.set_ylim(0, max(rows[mode, phase]["wall_max_ms"] for mode in MODES) / 1000 * 1.3)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"DeepSeek V3.2: all 61 layers\n{summary['run_id']}")
    fig.supxlabel("Bars: median; points: measured repeats; whiskers: minimum–maximum.", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"latency.{suffix}", dpi=180)
    plt.close(fig)

    categories = {
        "Attention projections": {"attention_projection", "attention_output"},
        "Sparse MLA": {"sparse_mla"},
        "Indexer / fused prefetch": {"indexer", "indexer_prefetch"},
        "Exact top-k": {"exact_topk"},
        "Offload preparation": {"offload_prepare"},
        "Exact recall": {"offload_exact_recall"},
        "Hidden transfer / dependency wait": {"hidden_transfer"},
        "Dense MLP": {"dense_mlp"},
        "MoE": {"moe", "moe_routing", "moe_experts", "moe_shared_experts"},
        "Norms, embedding, other": set(),
    }
    assigned = set.union(*categories.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.985, top=0.84, bottom=0.18, wspace=0.20)
    for ax, phase in zip(axes, PHASES):
        for mode_index, mode in enumerate(MODES):
            totals = rows[mode, phase]["annotated_cuda_exclusive_ms_by_stage"]
            values = []
            for stages in categories.values():
                values.append(
                    sum(
                        value
                        for key, value in totals.items()
                        if key in stages or not stages and key not in assigned
                    )
                    / 1000
                )
            ax.barh(
                [i + (mode_index - 0.5) * 0.36 for i in range(len(categories))],
                values,
                height=0.34,
                color=colors[mode_index],
                label=mode.capitalize(),
            )
        ax.set_yticks(range(len(categories)), list(categories))
        ax.set_xlabel("CUDA-event elapsed time per category (s)")
        ax.set_title("65,536-token prefix" if phase == "prefill" else "1,024-token extend")
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=2, fontsize=10
    )
    fig.suptitle(f"Separate annotated scope samples\n{summary['run_id']}")
    fig.text(
        0.5,
        0.01,
        "Scopes subtract children only; cross-device overlap, stream waits and host launch gaps can remain.\n"
        "Scope sums are not uninstrumented end-to-end latency or summed kernel duration; fused prefetch stays in the indexer scope.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"annotated_scopes.{suffix}", dpi=180)
    plt.close(fig)


def report_fragment(summary):
    rows = {(row["mode"], row["phase"]): row for row in summary["phases"]}
    run_id = summary["run_id"]
    devices = list(dict.fromkeys(summary["placement"]))
    text = [
        "## 已验收结果",
        "",
        f"来源 run ID：`{run_id}`。完整 61 层 checkpoint，65,536-token prefix 从空 cache 构建，随后执行 1,024-token extend；只输出最后 token 的完整词表 logits。",
        f"设备：{', '.join(devices)}；chunk={summary['chunk_size']}；每层 offload pool={summary['slots']} tokens；预热 {summary['warmups']} 次。硬件详情、依赖版本、源码 SHA256 与逐阶段数据见 [summary.json](report/summary.json)。",
        "",
        "| 阶段 | Resident 中位延迟 (ms) | Offload 中位延迟 (ms) | Offload / resident | 重复次数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for phase in PHASES:
        a, b = rows["resident", phase], rows["offload", phase]
        text.append(
            f"| {'Prefix prefill' if phase == 'prefill' else 'Extend'} | {a['wall_median_ms']:.3f} | {b['wall_median_ms']:.3f} | {b['wall_median_ms'] / a['wall_median_ms']:.4f}× | {a['repeats']} |"
        )
    a, b = (
        rows["resident", "extend"]["annotated_cache_totals"],
        rows["offload", "extend"]["annotated_cache_totals"],
    )
    text += [
        "",
        f"主 MLA KV 的 HBM record allocation 总计从 {a['device_record_bytes'] / 2**30:.4f} GiB 降至 {b['device_record_bytes'] / 2**30:.4f} GiB；该数字不含权重、indexer cache、映射、scratch 或激活。",
        "",
    ]
    for phase in PHASES:
        counters = rows["offload", phase]["annotated_cache_totals"]
        text.append(
            f"{phase} 的独立 annotated 采样合计 H2D {counters['host_to_device_bytes'] / 2**30:.4f} GiB、D2H {counters['device_to_host_bytes'] / 2**30:.4f} GiB，来自 61 层实际 record 计数，每 record 为 1152 B。"
        )
    text += [
        "",
        "H2D 统计包含融合预取与全部 gather；`recalled_records` 同时包含 `offload_prepare` 对当前 chunk 的 staging 与后续 exact recall，不能全部解释为 residual recall。D2H 统计为新 KV 写入 host backing。",
        "",
        "Resident/offload 保存的末 token logits 已逐元素核验 bitwise 相同（max_abs=0，NRMSE=0），next token 相同。",
        "",
        "![完整模型无插桩延迟](report/latency.png)",
        "",
        "![独立 annotated CUDA scope](report/annotated_scopes.png)",
        "",
        "延迟表来自无插桩完整 forward。Scope 图来自独立带事件采样，各类别分别对照，CUDA-event elapsed time 仅扣除嵌套子 scope；仍可能包含 CPU 提交空隙、stream 依赖等待与跨 GPU 重叠。这些区间不等价于 nsys 的实际 kernel duration，不能相加作端到端分解或加回无插桩 wall time。`hidden_transfer` 包含等待源 GPU 计算完成的时间，不能当成独立 NVLink copy 耗时或据此推算带宽。",
        "",
        "`indexer_prefetch` 同时包含 indexer 计算与融合预取，不能解读为独立搬运耗时；`offload_prepare` 与 `offload_exact_recall` 分别呈现。Prefix 与 extend 分别测量；输入准备、权重加载、编译及每次恢复 prefix residency 均在计时外。",
        "",
        f"图表、[CSV](report/summary.csv) 与 JSON 由 `python -m experiments.{EXPERIMENT}.src.report --result experiments/{EXPERIMENT}/output/data/{run_id}/result.json --publish` 生成。完整原始产物保留在 `experiments/{EXPERIMENT}/output/data/{run_id}/`；原始 trace 位于对应 `output/profile/{run_id}/`。",
        "",
    ]
    return "\n".join(text)


def generate_report(result_path, *, publish=False, report_dir=None):
    """Write derived output only after validation; publish selected artifacts on request."""
    require(publish or report_dir is None, "--report-dir requires --publish")
    summary = make_summary(result_path)
    directory = Path(result_path).parent
    fragment = report_fragment(summary)
    pending_readme = None
    if publish:
        report_dir = (
            Path(report_dir) if report_dir else Path(__file__).resolve().parents[1] / "report"
        )
        readme = report_dir.parent / "README.md"
        if readme.is_file():
            content = readme.read_text()
            if BEGIN in content or END in content:
                require(
                    content.count(BEGIN) == content.count(END) == 1
                    and content.index(BEGIN) < content.index(END),
                    "README generated-results markers are malformed",
                )
                before, remainder = content.split(BEGIN, 1)
                _, after = remainder.split(END, 1)
                pending_readme = (readme, before + BEGIN + "\n" + fragment + END + after)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(summary, directory / "summary.csv")
    plot(summary, directory)
    (directory / "report_fragment.md").write_text(fragment)
    readme_updated = False
    if publish:
        report_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "summary.json",
            "summary.csv",
            "latency.png",
            "latency.svg",
            "annotated_scopes.png",
            "annotated_scopes.svg",
        ):
            temporary = report_dir / ("." + name + ".tmp")
            shutil.copyfile(directory / name, temporary)
            temporary.replace(report_dir / name)
        if pending_readme:
            pending_readme[0].write_text(pending_readme[1])
            readme_updated = True
    return {
        "run_id": summary["run_id"],
        "data_dir": str(directory),
        "published": publish,
        "report_dir": str(report_dir) if publish else None,
        "readme_updated": readme_updated,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument(
        "--publish", action="store_true", help="copy selected artifacts into report/"
    )
    parser.add_argument(
        "--report-dir", type=Path, help="publication destination; requires --publish"
    )
    args = parser.parse_args(argv)
    try:
        result = generate_report(args.result, publish=args.publish, report_dir=args.report_dir)
    except InvalidResult as exc:
        parser.exit(2, f"Result rejected: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
