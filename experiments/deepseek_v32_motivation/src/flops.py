"""CPU-only matrix-work and mixed-precision effective-MFU analysis of saved runs.

Useful FLOPs come from actual projection, indexer and absorbed-MLA shapes.
They exclude scalar operations and copies, whose time remains in wall latency.
This is reference-peak normalization, not measured Tensor Core activity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path

from experiments.deepseek_v32_echo_prefill.src.operator_flops import (
    build_operator_work,
    linear_work,
)
from experiments.deepseek_v32_motivation.src.measure import INDEXER_DISPATCH_POLICY
from experiments.deepseek_v32_motivation.src.report import SCHEMES, read_jsonl, require

ROOT = Path(__file__).resolve().parents[3]
LINEARS = {
    "q_a_proj": "self_attn.q_a_proj",
    "q_b_proj": "self_attn.q_b_proj",
    "kv_a_proj": "self_attn.kv_a_proj_with_mqa",
    "index_q_proj": "self_attn.indexer.wq_b",
    "index_k_proj": "self_attn.indexer.wk",
    "o_proj": "self_attn.o_proj",
    "mlp_gate": "mlp.gate_proj",
    "mlp_up": "mlp.up_proj",
    "mlp_down": "mlp.down_proj",
}
SOURCE_FILES = (
    "models/deepseek_v32/execution/adapter.py",
    "models/deepseek_v32/config.py",
    "models/deepseek_v32/checkpoint.py",
    "models/deepseek_v32/rotary.py",
    "models/deepseek_v32/projections.py",
    "models/deepseek_v32/layers.py",
    "models/deepseek_v32/attention.py",
    "models/attention_contracts.py",
    "operators/deepseek_v32/indexer/echo.py",
    "operators/deepseek_v32/attention/_config.py",
    "experiments/deepseek_v32_echo_prefill/src/operator_flops.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_evidence(path, recorded):
    """Read only configuration and safetensors headers; never load weight arrays."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    headers, header_hashes, tensors, precisions = {}, {}, {}, {}

    def tensor(name):
        shard = weight_map[name]
        if shard not in headers:
            stat = (path / shard).stat()
            expected = recorded["files"][shard]
            require(
                stat.st_size == expected["size"] and stat.st_mtime_ns == expected["mtime_ns"],
                f"checkpoint shard inventory changed: {shard}",
            )
            with (path / shard).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                raw = stream.read(size)
            headers[shard] = json.loads(raw)
            header_hashes[shard] = hashlib.sha256(raw).hexdigest()
        item = headers[shard][name]
        tensors[name] = {"dtype": item["dtype"], "shape": item["shape"], "shard": shard}
        return item

    for source in range(3):
        precisions[source] = {}
        for operator, suffix in LINEARS.items():
            name = f"model.layers.{source}.{suffix}.weight"
            item = tensor(name)
            if item["dtype"] == "F8_E4M3":
                scale = name.removesuffix(".weight") + ".weight_scale_inv"
                tensor(scale)
                precisions[source][operator] = "fp8"
            else:
                require(item["dtype"] == "BF16", f"unsupported linear dtype: {name}")
                precisions[source][operator] = "bf16"
        tensor(f"model.layers.{source}.self_attn.kv_b_proj.weight")
        tensor(f"model.layers.{source}.self_attn.indexer.weights_proj.weight")
        precisions[source]["index_weights_proj"] = "fp32"
    require(tensor("lm_head.weight")["dtype"] == "BF16", "expected BF16 LM head")
    return (
        config,
        precisions,
        {
            "config_sha256": digest(path / "config.json"),
            "weight_index_sha256": digest(path / "model.safetensors.index.json"),
            "header_sha256": header_hashes,
            "tensors": tensors,
            "identity_boundary": (
                "Headers/config read during analysis; selected shard size/mtime match the saved run. "
                "The original run did not hash checkpoint weights/config; no stronger identity is claimed."
            ),
        },
    )


def phase_ledger(model, config, precisions, *, scheme, phase):
    """Expand the replay schedule; source-input reuse does not eliminate any GEMM."""
    require(scheme in SCHEMES and phase in ("history", "candidate"), "unknown scheme or phase")
    dispatch_policy = config.get("indexer_dispatch_policy")
    require(dispatch_policy in (None, INDEXER_DISPATCH_POLICY), "unknown indexer dispatch policy")
    dynamic_indexer = scheme == "echo" and dispatch_policy == INDEXER_DISPATCH_POLICY
    history, candidate = config["history_tokens"], config["candidate_tokens"]
    if phase == "history":
        chunks = [
            (start, min(config["chunk_size"], history - start))
            for start in range(0, history, config["chunk_size"])
        ]
    else:
        chunks = [(history, candidate)]
    copies = Counter(layer % 3 for layer in range(config["layers"]))
    ledger = []
    for chunk, (start, count) in enumerate(chunks):
        for source, multiplicity in sorted(copies.items()):
            work = build_operator_work(
                model,
                query_tokens=count,
                query_start=start,
                linear_dtypes=precisions[source],
                prefetch=scheme == "echo",
            )
            for item in work.values():
                record = item.as_dict()
                if dynamic_indexer and record["name"] == "indexer_prefetch":
                    # Residency is runtime state. A scheme label alone cannot
                    # establish fused vs resident tile padding or call counts.
                    record.update(
                        name="indexer",
                        executed_matmul_flops=None,
                        executed_formula=None,
                        dispatch_policy=dispatch_policy,
                    )
                    for name in ("padded_pairs", "BLOCK_KV"):
                        record["dimensions"].pop(name)
                    record["notes"] += (
                        " Runtime residency selects resident or fused execution; actual "
                        "dispatch and padded work require the profiled call ledger."
                    )
                record.update(
                    scheme=scheme,
                    phase=phase,
                    chunk=chunk,
                    source_layer=source,
                    layer_copies=multiplicity,
                )
                record["useful_flops"] *= multiplicity
                if record["executed_matmul_flops"] is not None:
                    record["executed_matmul_flops"] *= multiplicity
                ledger.append(record)
    # _forward_leased executes the head once at its final chunk. Both prefill()
    # and extend_candidate() call it; the prefix head is computed then discarded.
    head = linear_work(
        "lm_head",
        rows=1,
        in_features=model["hidden_size"],
        out_features=model["vocab_size"],
        precision="bf16",
    ).as_dict()
    head.update(scheme=scheme, phase=phase, chunk=len(chunks) - 1, source_layer=-1, layer_copies=1)
    ledger.append(head)
    return ledger


def aggregate_operators(ledger):
    groups = defaultdict(list)
    for row in ledger:
        groups[row["scheme"], row["phase"], row["name"], row["precision"]].append(row)
    result = []
    for (scheme, phase, name, precision), group in sorted(groups.items()):
        known = all(row["executed_matmul_flops"] is not None for row in group)
        useful = sum(row["useful_flops"] for row in group)
        executed = sum(row["executed_matmul_flops"] for row in group) if known else None
        result.append(
            {
                "scheme": scheme,
                "phase": phase,
                "operator": name,
                "precision": precision,
                "invocations": sum(row["layer_copies"] for row in group),
                "useful_flops": useful,
                "executed_matmul_flops": executed,
                "known_padding_extra_flops": executed - useful if known else None,
            }
        )
    return result


def work_summary(operators, peaks):
    useful = defaultdict(int)
    for row in operators:
        useful[row["precision"]] += row["useful_flops"]
    known = [row for row in operators if row["executed_matmul_flops"] is not None]
    ideal = {precision: flops / peaks[precision] / 1e9 for precision, flops in useful.items()}
    return {
        "useful_flops_by_precision": dict(useful),
        "useful_flops": sum(useful.values()),
        "ideal_compute_ms_by_precision": ideal,
        "ideal_compute_ms": sum(ideal.values()),
        "executed_matmul_flops": None,
        "known_padding_extra_flops": sum(row["known_padding_extra_flops"] for row in known),
        "padding_known_operators": [row["operator"] for row in known],
        "padding_unknown_operators": [row["operator"] for row in operators if row not in known],
    }


def request_mfu(rows, work, peaks):
    result = []
    for row in rows:
        rebuilt = not row["prefix_cache_hit"]
        for stage, time_field, phases in (
            ("prefix", "prefix_ms", ["history"] if rebuilt else []),
            ("extend", "extend_ms", ["candidate"]),
            ("e2e", "latency_ms", ["history", "candidate"] if rebuilt else ["candidate"]),
        ):
            flops = {
                precision: sum(
                    work[row["scheme"]][phase]["useful_flops_by_precision"].get(precision, 0)
                    for phase in phases
                )
                for precision in peaks
            }
            ideal = sum(flops[precision] / peaks[precision] / 1e9 for precision in peaks)
            # Retain an alternative TF32 peak normalization. With a verified
            # disabled policy this is counterfactual, not execution uncertainty.
            tf32_ideal = ideal - flops["fp32"] / peaks["fp32"] / 1e9 + flops["fp32"] / 494.5 / 1e9
            wall = row[time_field]
            require(wall > 0, f"nonpositive stage time: {stage}")
            result.append(
                {
                    "scheme": row["scheme"],
                    "request_id": row["request_id"],
                    "visit_kind": "revisit" if row["is_revisit"] else "first",
                    "stage": stage,
                    "history_rebuilt": rebuilt,
                    "wall_ms": wall,
                    **{f"useful_{precision}_flops": value for precision, value in flops.items()},
                    "ideal_compute_ms": ideal,
                    "tf32_sensitivity_ideal_ms": tf32_ideal,
                    "effective_mfu_pct": 100 * ideal / wall if phases else None,
                    "tf32_sensitivity_mfu_pct": 100 * tf32_ideal / wall if phases else None,
                }
            )
    return result


def summarize_mfu(requests):
    groups = defaultdict(list)
    for row in requests:
        groups[row["scheme"], row["visit_kind"], row["stage"]].append(row)
    result = []
    for (scheme, visit, stage), group in sorted(groups.items()):
        wall = sum(row["wall_ms"] for row in group)
        ideal = sum(row["ideal_compute_ms"] for row in group)
        tf32_ideal = sum(row["tf32_sensitivity_ideal_ms"] for row in group)
        result.append(
            {
                "scheme": scheme,
                "visit_kind": visit,
                "stage": stage,
                "requests": len(group),
                "history_builds": sum(row["history_rebuilt"] for row in group),
                "wall_mean_ms": statistics.mean(row["wall_ms"] for row in group),
                "wall_sum_ms": wall,
                "ideal_compute_mean_ms": ideal / len(group),
                "effective_mfu_pct": 100 * ideal / wall if ideal else None,
                "tf32_sensitivity_mfu_pct": 100 * tf32_ideal / wall if ideal else None,
            }
        )
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def precision_evidence(metadata):
    settings = metadata.get("precision_settings", {})
    verified = (
        settings.get("cuda_matmul_allow_tf32") is False
        and metadata.get("precision_settings_verified_after_execution") is True
    )
    if verified:
        boundary = (
            "The run recorded CUDA matmul allow_tf32=False before execution and verified "
            "unchanged precision settings afterward. FP32 weights projection uses the FP32 "
            "reference peak; TF32 sensitivity is a counterfactual normalization."
        )
    elif "cuda_matmul_allow_tf32" in settings:
        boundary = (
            "Recorded precision settings do not verify TF32 remained disabled throughout "
            "execution. Main values assume true FP32 weights projection; TF32 sensitivity "
            "is separate."
        )
    else:
        boundary = (
            "This run did not record TF32 policy. Main values assume true FP32 weights "
            "projection; TF32 sensitivity is separate."
        )
    return {
        "settings": settings,
        "verified_tf32_disabled": verified,
        "fp32_normalization": "recorded_fp32_policy" if verified else "assumed_fp32_policy",
        "boundary": boundary,
    }


def analyze(run_dir, output_dir, peak_report, model_path=None):
    run_dir, output = Path(run_dir), Path(output_dir)
    require(not output.exists(), "analysis output must be a new directory")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    require(metadata["status"] == "accepted", "input run is not accepted")
    require(metadata["hardware"]["name"] == "NVIDIA H200", "reference peaks require H200")
    config = metadata["config"]
    source_manifest = json.loads((run_dir / "source_manifest.json").read_text())
    source_hashes = {}
    for name in SOURCE_FILES:
        source_hashes[name] = digest(ROOT / name)
        if name.endswith("/operator_flops.py"):
            # Analysis-only dependency, absent from the original serving run.
            continue
        require(
            source_hashes[name] == source_manifest[name],
            f"execution/formula source changed: {name}",
        )
        require(
            digest(run_dir / "source" / name) == source_manifest[name], f"snapshot changed: {name}"
        )
    model, precisions, checkpoint = checkpoint_evidence(
        model_path or metadata["checkpoint"]["path"], metadata["checkpoint"]
    )
    require(
        model["hidden_size"] == metadata["model_dimensions"]["hidden"], "hidden dimension mismatch"
    )
    require(
        model["vocab_size"] == metadata["model_dimensions"]["vocabulary"], "vocabulary mismatch"
    )
    reference = json.loads(Path(peak_report).read_text())
    peaks = {name.lower(): value for name, value in reference["dense_peaks_tflops"].items()}
    require(
        peaks == {"fp8": 1979.0, "bf16": 989.5, "fp32": 67.0}, "unexpected H200 reference peaks"
    )
    ledger, work = [], {}
    for scheme in SCHEMES:
        work[scheme] = {}
        for phase in ("history", "candidate"):
            calls = phase_ledger(model, config, precisions, scheme=scheme, phase=phase)
            ledger.extend(calls)
            work[scheme][phase] = work_summary(aggregate_operators(calls), peaks)
    rows = read_jsonl(run_dir / "measurements.jsonl")
    require(
        len(rows) == config["requests_per_scheme"] * len(SCHEMES), "incomplete measurement matrix"
    )
    requests = request_mfu(rows, work, peaks)
    summary = summarize_mfu(requests)
    precision = precision_evidence(metadata)
    payload = {
        "schema": "deepseek-v32-motivation-effective-mfu-v1",
        "analysis_run_id": output.name,
        "source_run_id": metadata["run_id"],
        "config": config,
        "model_config": model,
        "hardware": metadata["hardware"],
        "precision_evidence": precision,
        "indexer_dispatch_policy": config.get("indexer_dispatch_policy", "legacy_scheme_fixed"),
        "linear_precisions_by_source": precisions,
        "checkpoint_evidence": checkpoint,
        "dense_peaks_tflops": peaks,
        "peak_reference": reference["peak_reference"],
        "peak_report_sha256": digest(peak_report),
        "normalization": "100 * sum_precision(useful_flops / (dense_peak_tflops * 1e9)) / wall_ms",
        "aggregation": "sum ideal compute milliseconds / sum measured wall milliseconds",
        "work": work,
        "stage_mfu": summary,
        "boundaries": [
            "Ten independent block computations, source layers 0/1/2 copied 4/3/3 times; no MoE or 6N formula.",
            f"History uses {(config['history_tokens'] + config['chunk_size'] - 1) // config['chunk_size']} chunks; candidate executes once. LM head runs once per forward, including prefix.",
            "Useful MLA excludes causal/padded slots; selection-pair counts assume finite causal indexer scores.",
            "Known WGMMA/FlashMLA padding is listed separately. DeepGEMM/cuBLAS GEMM padding is unknown.",
            "Quantization, norms, nonlinearities, selection, copies, cache control and launch gaps add no matrix FLOPs.",
            "Their time remains in saved prefix/extend/E2E wall time; effective MFU is not GPU busy time.",
            precision["boundary"],
            "Reference SXM peaks are not achieved-clock normalization. Cache-hit prefixes have no matrix MFU.",
            "Execution files match the saved run snapshot. The reused operator_flops.py is a separately hashed analysis dependency.",
        ],
        "source_sha256": {**source_hashes, str(Path(__file__).relative_to(ROOT)): digest(__file__)},
        "input_sha256": {
            name: digest(run_dir / name)
            for name in ("metadata.json", "measurements.jsonl", "source_manifest.json")
        },
    }
    output.mkdir(parents=True)
    (output / "flops.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(output / "operator_flops.csv", aggregate_operators(ledger))
    write_csv(output / "request_mfu.csv", requests)
    write_csv(output / "stage_mfu.csv", summary)
    with (output / "call_ledger.jsonl").open("w") as stream:
        for row in ledger:
            stream.write(json.dumps(row) + "\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--peak-report",
        type=Path,
        default=ROOT / "experiments/deepseek_v32_echo_prefill/report/layers3/summary.json",
    )
    args = parser.parse_args(argv)
    analyze(args.run_dir, args.output_dir, args.peak_report, args.model_path)
    print(f"CPU FLOPs/MFU analysis: {args.output_dir}")


if __name__ == "__main__":
    main()
